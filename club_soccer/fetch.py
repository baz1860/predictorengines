#!/usr/bin/env python3
"""Fetch/cache Club Soccer fixtures from BSD, with CSV fallback.

Primary source: BSD (Bzzoiro Sports Data) — free, no rate limits.
  https://sports.bzzoiro.com/docs/football/

BSD replaces the former API-Football integration. The output DataFrame
schema is identical so all downstream code (model.py, edge.py, etc.)
is unaffected.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_keys import get_key
from bsd_client import (event_date_utc, fixture_detail_fields, get_all_events,
                        get_event, league_name as bsd_league_name)
from .competitions import COMPETITIONS, comp_from_bsd_league
from . import schema
from .identities import dedupe_fixtures

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
RAW = DATA / "bsd_cache"

# BSD event status values (raw, lowercase, as returned by the API)
_FINISHED_STATUSES = {"finished", "ft", "aet", "pen"}
_UPCOMING_STATUSES = {"notstarted", "upcoming", "scheduled", "ns"}

# BSD's /api/events/ status filter only accepts these exact enum values
# (notstarted|inprogress|finished|postponed|cancelled). Our CLI/callers use a
# looser vocabulary ("upcoming") for readability — translate before querying.
_STATUS_TO_BSD = {
    "upcoming": "notstarted",
    "scheduled": "notstarted",
    "ns": "notstarted",
    "finished": "finished",
    "ft": "finished",
}

# BSD's /api/events/ defaults to a rolling ~7-day window when no date_from/
# date_to/season is given. To get a useful "current" fetch (recent results +
# real upcoming fixtures) we must always pass an explicit window.
_DEFAULT_LOOKBACK_DAYS = 14   # re-pull recent results to catch late score/status corrections
_DEFAULT_HORIZON_DAYS = 1095  # ~3 years: effectively "everything BSD has scheduled"

# Finished rows dated further than this into the future are corrupt (mirrors
# the guard in _bsd_to_fixture_row below).
_FUTURE_FT_GRACE_DAYS = 1


def _present(value) -> bool:
    """Whether a CSV/API value is real rather than an empty placeholder."""
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return not (isinstance(value, str) and value == "")


def _merge_fixture_rows(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge incoming BSD rows without erasing richer existing observations."""
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing.copy()

    columns = list(existing.columns) + [c for c in incoming.columns if c not in existing.columns]
    # Object dtype is intentional: source CSVs often infer a numeric column
    # before a later BSD row supplies a textual stage/venue value.
    old = existing.reindex(columns=columns).astype(object).copy()
    new = incoming.reindex(columns=columns).astype(object).copy()
    if "fixture_id" not in columns:
        return pd.concat([old, new], ignore_index=True)

    from .schema import RESULT_COLUMNS, VOID_STATUSES

    old_by_id = {str(v): i for i, v in old["fixture_id"].items() if _present(v)}
    for _, row in new.iterrows():
        key = str(row.get("fixture_id")) if _present(row.get("fixture_id")) else None
        if key is None or key not in old_by_id:
            old = pd.concat([old, pd.DataFrame([row], columns=columns)], ignore_index=True)
            if key is not None:
                old_by_id[key] = len(old) - 1
            i = len(old) - 1
        else:
            i = old_by_id[key]
            for col in columns:
                value = row.get(col)
                if _present(value):
                    old.at[i, col] = value
        # A void transition (postponed/cancelled/abandoned/...) must clear
        # every result/stat field. Present-values-only merging would otherwise
        # keep an old FT score under the new POS status, and the fixture
        # would keep training the model on a result that never stood.
        status = str(old.at[i, "status"] or "").strip().upper()[:3] \
            if "status" in columns else ""
        if status in VOID_STATUSES:
            for col in RESULT_COLUMNS:
                if col in columns:
                    old.at[i, col] = None
    return old.drop_duplicates(subset=["fixture_id"], keep="last").reset_index(drop=True)


def _enrich_rows_from_details(df: pd.DataFrame, api_key: str | None,
                              max_details: int = 400, pause: float = 0.1) -> pd.DataFrame:
    """Fill BZD detail fields from cache/API without making list rows lossy."""
    if df.empty:
        return df
    RAW.mkdir(parents=True, exist_ok=True)
    needed = []
    for i, row in df.iterrows():
        if pd.isna(row.get("home_goals")) or pd.isna(row.get("away_goals")):
            continue
        if all(_present(row.get(c)) for c in ("home_shots", "away_shots", "home_sot",
                                              "away_sot", "home_corners", "away_corners",
                                              "home_xg", "away_xg")):
            continue
        needed.append(i)
    if not needed:
        return df

    # Spend the detail budget on the newest played fixtures first. Those are
    # the rows that will influence the next season's form most heavily, and
    # they are also the rows most likely to have current BSD enrichments.
    needed.sort(key=lambda i: str(df.at[i, "date"]), reverse=True)

    processed = 0
    for i in needed[:max_details]:
        fid = df.at[i, "fixture_id"]
        cache = RAW / f"event_{fid}.json"
        detail = None
        if cache.exists():
            try:
                detail = json.loads(cache.read_text())
            except Exception:
                detail = None
        if detail is None and api_key:
            try:
                detail = get_event(api_key, fid)
                cache.write_text(json.dumps(detail, indent=2, ensure_ascii=False))
                if pause:
                    import time
                    time.sleep(pause)
            except Exception as exc:
                print(f"  fetch: detail {fid} failed ({exc})")
                continue
        if not detail:
            continue
        fields = fixture_detail_fields(detail)
        for col, value in fields.items():
            if col in df.columns and _present(value):
                df.at[i, col] = value
            elif col not in df.columns and _present(value):
                df[col] = pd.NA
                df.at[i, col] = value
        processed += 1
    if len(needed) > max_details:
        print(f"  fetch: enriched {processed}/{len(needed)} rows (detail cap {max_details})")
    elif processed:
        print(f"  fetch: enriched {processed} finished rows from BSD detail")
    return df


def _bsd_to_fixture_row(event: dict, comp_name: str, comp_api_id: int,
                        country: str, kind: str) -> dict | None:
    """Map a single BSD event dict to our fixtures.csv schema.

    Returns None (caller skips + counts) if the event claims to be finished
    but its kickoff date is implausibly far in the future — seen in the wild
    as mis-dated Champions League league-phase rows (e.g. status "finished",
    date a year ahead). Trusting that would poison Elo recency weighting.
    """
    home = event.get("home_team") or ""
    away = event.get("away_team") or ""
    kickoff = event_date_utc(event)
    date_str = str(kickoff)[:10]          # YYYY-MM-DD
    status_raw = str(event.get("status") or "").lower()

    if status_raw in _FINISHED_STATUSES and date_str:
        cutoff = str((datetime.now(timezone.utc) + timedelta(days=_FUTURE_FT_GRACE_DAYS)).date())
        if date_str > cutoff:
            return None
    # Scores
    score = event.get("score") or event.get("result") or {}
    if isinstance(score, dict):
        home_goals = score.get("home") if score.get("home") is not None else (
            event.get("home_score") if event.get("home_score") is not None else
            event.get("goals_home"))
        away_goals = score.get("away") if score.get("away") is not None else (
            event.get("away_score") if event.get("away_score") is not None else
            event.get("goals_away"))
    else:
        home_goals = event.get("home_score") or event.get("goals_home")
        away_goals = event.get("away_score") or event.get("goals_away")

    # Only record score for finished matches
    if status_raw not in _FINISHED_STATUSES:
        home_goals = None
        away_goals = None

    # Extract season from date
    try:
        year = int(date_str[:4])
        # Season: if match is Aug-Dec it's the start of the season, else it's the end
        month = int(date_str[5:7])
        season = year if month >= 7 else year - 1
    except (ValueError, IndexError):
        season = None

    row = {
        "fixture_id": event.get("id"),
        # Preserve the full kickoff instant — truncating to a date makes a
        # 12:00 UTC kickoff look "future" until midnight, so a settled match
        # could still be priced at 23:00. `date` stays as the display field.
        "kickoff_utc": str(kickoff) if kickoff else "",
        "date": date_str,
        "season": season,
        "competition": comp_name,
        "competition_id": comp_api_id,
        "country": country,
        "type": kind,
        "home_id": event.get("home_team_id") or "",
        "home": home,
        "away_id": event.get("away_team_id") or "",
        "away": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "status": status_raw.upper()[:3] if status_raw else "",
        "neutral": 0,
        "home_shots": "", "away_shots": "", "home_sot": "", "away_sot": "",
        "home_corners": "", "away_corners": "", "xg_source": "",
    }
    # Some BSD list responses already contain detail fields. Use them when
    # present; the detail pass below fills any remaining gaps.
    for col, value in fixture_detail_fields(event).items():
        if _present(value):
            row[col] = value
    return row


def _fetch_bsd_events(api_key: str, status: str | None = None,
                      date_from: str | None = None,
                      date_to: str | None = None) -> list[dict]:
    """Fetch all BSD football events matching the given filters.

    date_from/date_to should normally both be set: BSD's /api/events/
    silently defaults to a rolling ~7-day window when neither is given.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    if status:
        kwargs["status"] = _STATUS_TO_BSD.get(status.lower(), status)
    if date_from:
        kwargs["date_from"] = date_from
    if date_to:
        kwargs["date_to"] = date_to
    return get_all_events(api_key, **kwargs)


def fetch_fixtures(season: int | None = None,
                   current: bool = False,
                   api_key: str | None = None,
                   status: str | None = None,
                   date_from: str | None = None,
                   date_to: str | None = None,
                   days_ahead: int | None = None,
                   enrich_stats: bool = True,
                   max_details: int = 400,
                   pause: float = 0.1) -> pd.DataFrame:
    """Fetch club soccer fixtures from BSD and return as a DataFrame.

    Parameters
    ----------
    season:     If provided, filter rows to this season start year.
    current:    If True, merge fetched rows onto the existing fixtures.csv
                (de-duped by fixture_id, keeping latest).
    api_key:    BSD API key.  Falls back to BSD_API_KEY env / api_keys.json.
    status:     BSD status filter: "upcoming", "finished", or None (all).
    date_from:  ISO date lower bound. Defaults to today - 14d (catches late
                score/status corrections on recent results) when both
                date_from and date_to are omitted.
    date_to:    ISO date upper bound. Defaults to today + days_ahead (or
                ~3 years, i.e. "everything BSD has scheduled") when both
                date_from and date_to are omitted.
    days_ahead: Caps the default upper bound; ignored if date_to is given.
    enrich_stats: Pull/cache BSD event detail for finished rows with missing
        shots/xG/corners. Existing rows are never overwritten by blanks.
    max_details: Maximum detail requests per run.
    """
    key = api_key or get_key("bsd", env="BSD_API_KEY")
    if not key:
        raise ValueError(
            "No BSD key. Register at https://sports.bzzoiro.com/register/ "
            "and add 'bsd' to data/api_keys.json, or set BSD_API_KEY."
        )

    if date_from is None and date_to is None:
        today = datetime.now(timezone.utc).date()
        date_from = str(today - timedelta(days=_DEFAULT_LOOKBACK_DAYS))
        horizon = days_ahead if days_ahead is not None else _DEFAULT_HORIZON_DAYS
        date_to = str(today + timedelta(days=horizon))

    events = _fetch_bsd_events(key, status=status, date_from=date_from, date_to=date_to)

    # Build a name->Competition lookup for fast resolution
    rows: list[dict] = []
    unmatched_leagues: set[str] = set()
    future_finished_skipped = 0

    for ev in events:
        lname = bsd_league_name(ev)
        comp = comp_from_bsd_league(lname)
        if comp is None:
            unmatched_leagues.add(lname)
            continue
        row = _bsd_to_fixture_row(ev, comp.name, comp.api_id,
                                  comp.country, comp.kind)
        if row is None:
            future_finished_skipped += 1
            continue
        rows.append(row)

    if unmatched_leagues:
        # Log at most 10 so it's not noisy
        shown = sorted(unmatched_leagues)[:10]
        suffix = f" (+{len(unmatched_leagues) - 10} more)" if len(unmatched_leagues) > 10 else ""
        print(f"  fetch: unrecognised BSD leagues (ignored): {shown}{suffix}")

    if future_finished_skipped:
        print(f"  fetch: skipped {future_finished_skipped} finished event(s) with "
              f"implausible future kickoff dates")

    df = pd.DataFrame(rows)

    if season is not None and not df.empty:
        df = df[df["season"] == season].copy()

    if not df.empty:
        df = df.drop_duplicates(subset=["fixture_id"], keep="last")

    if enrich_stats and not df.empty:
        df = _enrich_rows_from_details(df, key, max_details=max_details, pause=pause)

    if current and FIXTURES.exists():
        existing = pd.read_csv(FIXTURES, low_memory=False)
        df = _merge_fixture_rows(existing, df)

    # Reconcile the same match arriving from multiple providers under
    # different IDs before writing the source-of-truth CSV.
    before_identity_rows = len(df)
    df = dedupe_fixtures(df)
    if len(df) < before_identity_rows:
        print(f"  fetch: reconciled {before_identity_rows - len(df)} duplicate match identity row(s)")

    # Keep every newly supported column present even when a source has no value
    # yet. This makes downstream numeric coercion and schema inspection stable.
    for col in schema.FIXTURE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    ordered = [c for c in schema.FIXTURE_COLUMNS if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]

    DATA.mkdir(exist_ok=True)
    df.to_csv(FIXTURES, index=False)

    if not df.empty:
        played_n = int(df["home_goals"].notna().sum())
        upcoming_n = int(df["home_goals"].isna().sum())
        try:
            dates = pd.to_datetime(df["date"], errors="coerce")
            recent_cut = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=7)
            finished_last_7d = int(((dates >= recent_cut) & df["home_goals"].notna()).sum())
        except Exception:
            finished_last_7d = 0
        print(f"  fetch: played={played_n} upcoming={upcoming_n} "
              f"finished-in-last-7-days={finished_last_7d}")

    return df


def repair_future_dated(api_key: str | None = None) -> tuple[int, int]:
    """Repair or drop 'finished' fixtures.csv rows dated implausibly in the future.

    For each such row, re-fetch the event by fixture_id and take its real
    kickoff date. If the refetch fails (event not found/expired on BSD) or
    still yields a future date, drop the row rather than trust it.

    Returns (repaired, dropped).
    """
    if not FIXTURES.exists():
        print("No fixtures.csv found; nothing to repair.")
        return (0, 0)

    key = api_key or get_key("bsd", env="BSD_API_KEY")
    df = pd.read_csv(FIXTURES, low_memory=False)
    cutoff = str((datetime.now(timezone.utc) + timedelta(days=_FUTURE_FT_GRACE_DAYS)).date())
    mask = df["status"].astype(str).str.upper().isin({"FT", "FIN", "AET", "PEN"}) & (df["date"] > cutoff)
    bad_idx = list(df[mask].index)

    if not bad_idx:
        print("repaired 0, dropped 0 (no future-dated finished rows found)")
        return (0, 0)

    repaired = 0
    drop_idx: list[int] = []
    for idx in bad_idx:
        fid = df.at[idx, "fixture_id"]
        new_date: str | None = None
        if key:
            try:
                ev = get_event(key, fid)
                d = event_date_utc(ev)[:10]
                if d and d <= cutoff:
                    new_date = d
            except Exception:
                new_date = None
        if new_date:
            df.at[idx, "date"] = new_date
            month, year = int(new_date[5:7]), int(new_date[:4])
            df.at[idx, "season"] = year if month >= 7 else year - 1
            repaired += 1
        else:
            drop_idx.append(idx)
            print(f"  dropped unrepairable row: fixture_id={fid} "
                  f"{df.at[idx, 'home']} vs {df.at[idx, 'away']} ({df.at[idx, 'date']})")

    if drop_idx:
        df = df.drop(index=drop_idx).reset_index(drop=True)

    DATA.mkdir(exist_ok=True)
    df.to_csv(FIXTURES, index=False)
    dropped = len(drop_idx)
    print(f"repaired {repaired}, dropped {dropped}")
    return (repaired, dropped)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch Club Soccer fixtures from BSD (free, no rate limits)."
    )
    ap.add_argument("--season", type=int,
                    help="filter to this season start year (e.g. 2025)")
    ap.add_argument("--current", action="store_true",
                    help="merge fetched rows into existing fixtures.csv")
    ap.add_argument("--status", choices=["upcoming", "finished"],
                    help="BSD status filter (default: fetch all)")
    ap.add_argument("--date-from", dest="date_from", help="ISO date lower bound")
    ap.add_argument("--date-to", dest="date_to", help="ISO date upper bound")
    ap.add_argument("--days-ahead", dest="days_ahead", type=int,
                    help="how far ahead to fetch upcoming fixtures "
                         "(default: ~3 years, i.e. everything BSD has scheduled)")
    ap.add_argument("--no-details", action="store_true",
                    help="skip BSD event-detail enrichment for finished rows")
    ap.add_argument("--max-details", type=int, default=400,
                    help="maximum finished-event detail requests (default 400)")
    ap.add_argument("--pause", type=float, default=0.1,
                    help="seconds between uncached detail requests (default 0.1)")
    ap.add_argument("--api-key", dest="api_key",
                    help="BSD API key (overrides env / api_keys.json)")
    ap.add_argument("--repair", action="store_true",
                    help="repair or drop finished fixtures.csv rows dated "
                         "implausibly in the future, then exit")
    args = ap.parse_args()

    if args.repair:
        try:
            repair_future_dated(api_key=args.api_key)
        except Exception as e:
            sys.exit(str(e))
        return

    try:
        df = fetch_fixtures(
            season=args.season,
            current=args.current,
            api_key=args.api_key,
            status=args.status,
            date_from=args.date_from,
            date_to=args.date_to,
            days_ahead=args.days_ahead,
            enrich_stats=not args.no_details,
            max_details=args.max_details,
            pause=args.pause,
        )
    except Exception as e:
        sys.exit(str(e))
    print(f"Wrote {len(df)} fixture rows -> {FIXTURES}")


if __name__ == "__main__":
    main()
