#!/usr/bin/env python3
"""Fetch and normalize live World Cup provider data.

Canonical CSVs are latest-state tables except `market_snapshots.csv`, which is a
deduped snapshot history.  Raw provider JSON is saved append-only for audit.

Data sources
------------
BSD (Bzzoiro Sports Data) — FREE, no rate limits.
  Replaces API-Football for fixtures, injuries, lineups, and match stats.
  Register at https://sports.bzzoiro.com/register/

The Odds API — market odds snapshots (unchanged).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_keys import get_key  # noqa: E402
from bsd_client import (  # noqa: E402
    get_all_events, get_event,
    league_name as bsd_league_name,
    event_date_utc,
    unavailable_players as bsd_unavailable,
    lineups as bsd_lineups,
    match_statistics as bsd_stats,
)
from contracts import fixture_key  # noqa: E402
from engines.worldcup.names import canonical_team, require_known_team, known_teams  # noqa: E402

DATA_DIR = ROOT / "data" / "worldcup"
RAW_DIR = DATA_DIR / "raw"
FIXTURES_CSV = DATA_DIR / "fixtures_live.csv"
AVAILABILITY_CSV = DATA_DIR / "player_availability.csv"
LINEUPS_CSV = DATA_DIR / "lineups.csv"
MATCH_STATS_CSV = DATA_DIR / "match_stats.csv"
MARKET_SNAPSHOTS_CSV = DATA_DIR / "market_snapshots.csv"

ODDS_BASE = "https://api.the-odds-api.com/v4"
WORLD_CUP_SPORT_KEY = "soccer_fifa_world_cup"
_TZ_PDT = timezone(timedelta(hours=-7))

BSD_KEY = get_key("bsd", env="BSD_API_KEY")
ODDS_API_KEY = get_key("the-odds-api", env="THE_ODDS_API_KEY")

ODDS_MARKETS = ("h2h", "h2h_3_way", "totals", "btts", "draw_no_bet",
                "double_chance")

# BSD league name substrings that identify World Cup matches
_WC_LEAGUE_HINTS = ("world cup", "fifa world cup", "coupe du monde", "copa mundial")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _local_match_date(iso: str) -> str:
    try:
        return (datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
                .astimezone(_TZ_PDT).date().isoformat())
    except ValueError:
        return str(iso)[:10]


def _event_id(match_date: str, home: str, away: str,
              competition: str = "FIFA World Cup") -> str:
    return fixture_key(match_date, home, away, competition)


def _save_raw(provider: str, kind: str, payload: Any,
              fetched_at: str | None = None) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ts = (fetched_at or utc_now()).replace(":", "").replace("+", "Z")
    path = RAW_DIR / f"{provider}_{kind}_{ts}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def _upsert_csv(path: Path, df: pd.DataFrame, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old = pd.read_csv(path)
            df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass
    if keys and not df.empty:
        df = df.drop_duplicates(subset=keys, keep="last")
    df.to_csv(path, index=False)


def _append_snapshot_csv(path: Path, df: pd.DataFrame, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old = pd.read_csv(path)
            df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass
    if keys and not df.empty:
        df = df.drop_duplicates(subset=keys, keep="last")
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# BSD helpers
# ---------------------------------------------------------------------------

def _is_worldcup_event(event: dict) -> bool:
    """True if this BSD event belongs to the World Cup."""
    name = bsd_league_name(event).lower()
    return any(hint in name for hint in _WC_LEAGUE_HINTS)


def _try_team(name: object, known: set[str], context: str) -> str | None:
    """Return canonical team name, or None if unresolvable (logs a warning)."""
    try:
        return require_known_team(name, known, context)
    except Exception:
        try:
            return canonical_team(str(name))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Parse functions (BSD → canonical DataFrames)
# ---------------------------------------------------------------------------

def parse_fixtures_bsd(events: list[dict],
                       fetched_at: str | None = None,
                       teams: set[str] | None = None) -> pd.DataFrame:
    fetched = fetched_at or utc_now()
    known = teams if teams is not None else known_teams()
    rows = []
    for ev in events:
        if not _is_worldcup_event(ev):
            continue
        home_raw = ev.get("home_team") or ""
        away_raw = ev.get("away_team") or ""
        home = _try_team(home_raw, known, "fixture")
        away = _try_team(away_raw, known, "fixture")
        if not home or not away:
            continue
        kickoff = event_date_utc(ev)
        match_date = _local_match_date(kickoff)
        comp = bsd_league_name(ev) or "FIFA World Cup"
        rnd = ev.get("round") or ev.get("stage") or ""
        status_raw = str(ev.get("status") or "").lower()
        # Map BSD status strings to short codes similar to old api-football codes
        status_map = {
            "upcoming": ("Not Started", "NS"),
            "live": ("In Progress", "1H"),
            "finished": ("Match Finished", "FT"),
            "cancelled": ("Cancelled", "CANC"),
            "postponed": ("Postponed", "PST"),
        }
        status_long, status_short = status_map.get(
            status_raw, (str(ev.get("status") or ""), "")
        )
        venue = ev.get("venue") or {}
        if isinstance(venue, str):
            venue_name, venue_city = venue, ""
        else:
            venue_name = venue.get("name") or venue.get("stadium") or ""
            venue_city = venue.get("city") or ""
        rows.append({
            "event_id": _event_id(match_date, home, away, comp),
            "provider_fixture_id": ev.get("id"),
            "match_date": match_date,
            "kickoff_utc": kickoff,
            "home": home,
            "away": away,
            "competition": comp,
            "round": rnd,
            "group": _group_from_round(rnd),
            "status_long": status_long,
            "status_short": status_short,
            "elapsed": ev.get("elapsed") or ev.get("minute"),
            "venue_name": venue_name,
            "venue_city": venue_city,
            "provider_home_id": ev.get("home_team_id"),
            "provider_away_id": ev.get("away_team_id"),
            "source": "bsd",
            "fetched_at": fetched,
        })
    return pd.DataFrame(rows)


def _group_from_round(round_name: str) -> str:
    text = str(round_name or "")
    low = text.lower()
    if "group" not in low:
        return ""
    parts = text.replace("-", " ").split()
    for i, p in enumerate(parts):
        if p.lower() == "group" and i + 1 < len(parts):
            return parts[i + 1].strip().upper()
    return ""


_DOUBTFUL = ("doubt", "questionable", "fitness", "late test", "knock",
             "illness", "race against time", "expected to return")
_LIMITED = ("limited", "light training", "individual training")
_SUSPENDED = ("suspend", "ban", "red card", "yellow card accumulation")
_WITHDRAWN = ("withdraw", "left squad", "replaced in squad")
_CERTAIN_OUT = ("out", "ruled out", "acl", "cruciate", "fracture", "surgery",
                "tear", "rupture", "season-ending")


def classify_availability(reason: object, kind: object = "") -> tuple[str, str, bool]:
    text = f"{kind or ''} {reason or ''}".strip().lower()
    if any(k in text for k in _SUSPENDED):
        return "suspended", "certain", True
    if any(k in text for k in _WITHDRAWN):
        return "withdrawn", "certain", True
    if any(k in text for k in _LIMITED):
        return "limited_training", "uncertain", False
    if any(k in text for k in _DOUBTFUL):
        return "doubtful", "uncertain", False
    if any(k in text for k in _CERTAIN_OUT) or "injur" in text:
        return "out", "certain", True
    return "out", "uncertain", True


def parse_availability_bsd(events: list[dict],
                            fetched_at: str | None = None,
                            teams: set[str] | None = None) -> pd.DataFrame:
    """Extract unavailable players embedded in BSD event responses."""
    fetched = fetched_at or utc_now()
    known = teams if teams is not None else known_teams()
    rows = []
    for ev in events:
        if not _is_worldcup_event(ev):
            continue
        unavail = bsd_unavailable(ev)
        fixture_id = ev.get("id")
        for side in ("home", "away"):
            team_raw = ev.get(f"{side}_team") or ""
            team = _try_team(team_raw, known, "availability")
            if not team:
                continue
            for player in unavail.get(side) or []:
                reason = (player.get("reason") or player.get("description")
                          or player.get("type") or "")
                kind = player.get("type") or player.get("status") or ""
                status, certainty, affects = classify_availability(reason, kind)
                rows.append({
                    "team": team,
                    "player": player.get("name"),
                    "status": status,
                    "reason": reason,
                    "certainty": certainty,
                    "affects_availability": bool(affects),
                    "source": "bsd",
                    "fetched_at": fetched,
                    "provider_fixture_id": fixture_id,
                    "provider_team_id": player.get("team_id"),
                    "provider_player_id": player.get("id") or player.get("player_id"),
                })
    return pd.DataFrame(rows)


def parse_lineups_bsd(event: dict, fixture_meta: dict | None = None,
                      fetched_at: str | None = None,
                      published_at: str | None = None,
                      teams: set[str] | None = None) -> pd.DataFrame:
    """Parse lineups from a single BSD event-detail response."""
    fetched = fetched_at or utc_now()
    published = published_at or fetched
    known = teams if teams is not None else known_teams()
    meta = fixture_meta or {}
    raw_lineups = bsd_lineups(event)
    rows = []
    for side in ("home", "away"):
        team_raw = event.get(f"{side}_team") or ""
        team = _try_team(team_raw, known, "lineup")
        if not team:
            continue
        side_data = raw_lineups.get(side) or {}
        formation = side_data.get("formation") or ""
        starters = side_data.get("starters") or side_data.get("starting_xi") or []
        bench = side_data.get("bench") or side_data.get("substitutes") or []
        for role, players in (("starter", starters), ("bench", bench)):
            for player in players:
                if isinstance(player, str):
                    player = {"name": player}
                rows.append({
                    "event_id": meta.get("event_id", ""),
                    "provider_fixture_id": meta.get("provider_fixture_id"),
                    "match_date": meta.get("match_date", ""),
                    "team": team,
                    "player": player.get("name"),
                    "provider_team_id": player.get("team_id"),
                    "provider_player_id": player.get("id") or player.get("player_id"),
                    "starter": role == "starter",
                    "role": role,
                    "position": player.get("pos") or player.get("position"),
                    "shirt_number": player.get("number") or player.get("shirt_number"),
                    "formation": formation,
                    "lineup_status": "confirmed",
                    "published_at": published,
                    "source": "bsd",
                    "fetched_at": fetched,
                })
    return pd.DataFrame(rows)


_STAT_MAP = {
    "shots on goal": "shots_on_target",
    "shots on target": "shots_on_target",
    "total shots": "shots",
    "shots total": "shots",
    "shots": "shots",
    "corner kicks": "corners",
    "corners": "corners",
    "ball possession": "possession",
    "possession": "possession",
    "yellow cards": "yellow_cards",
    "red cards": "red_cards",
    "fouls": "fouls",
    "expected goals": "xg",
    "xg": "xg",
}

# BSD may also return stats as direct fields (e.g. home_shots, away_xg)
_BSD_DIRECT_STAT_FIELDS = {
    "shots": "shots", "shots_on_target": "shots_on_target",
    "sot": "shots_on_target", "corners": "corners",
    "possession": "possession", "xg": "xg",
    "yellow_cards": "yellow_cards", "red_cards": "red_cards",
    "fouls": "fouls",
}


def _num(v: object) -> float | int | None:
    if v is None:
        return None
    s = str(v).replace("%", "").strip()
    if not s:
        return None
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def parse_match_stats_bsd(event: dict, fixture_meta: dict | None = None,
                          fetched_at: str | None = None,
                          teams: set[str] | None = None) -> pd.DataFrame:
    """Parse match statistics from a BSD event-detail response."""
    fetched = fetched_at or utc_now()
    known = teams if teams is not None else known_teams()
    meta = fixture_meta or {}
    raw = bsd_stats(event)
    rows = []
    for side in ("home", "away"):
        team_raw = event.get(f"{side}_team") or ""
        team = _try_team(team_raw, known, "stats")
        if not team:
            continue
        row: dict[str, Any] = {
            "event_id": meta.get("event_id", ""),
            "provider_fixture_id": meta.get("provider_fixture_id"),
            "match_date": meta.get("match_date", ""),
            "team": team,
            "source": "bsd",
            "fetched_at": fetched,
        }
        side_stats = raw.get(side) or {}
        # BSD may return stats as a list [{"type": ..., "value": ...}]
        # or as a flat dict {"shots": 8, "xg": 1.23}
        if isinstance(side_stats, list):
            for s in side_stats:
                key = _STAT_MAP.get(str(s.get("type", "")).strip().lower())
                if key:
                    row[key] = _num(s.get("value"))
        elif isinstance(side_stats, dict):
            for field, col in _BSD_DIRECT_STAT_FIELDS.items():
                if field in side_stats:
                    row[col] = _num(side_stats[field])
        # Also check top-level event fields like home_shots, away_xg
        for field, col in _BSD_DIRECT_STAT_FIELDS.items():
            full_key = f"{side}_{field}"
            if full_key in event and col not in row:
                row[col] = _num(event[full_key])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main BSD fetch orchestrator
# ---------------------------------------------------------------------------

def fetch_bsd(mode: str, api_key: str) -> None:
    """Fetch World Cup data from BSD for the given *mode*.

    morning    — fixtures + availability (injuries/suspensions)
    prekickoff — availability + lineups (per fixture)
    postmatch  — lineups + match stats (per fixture)
    all        — everything above
    """
    if not api_key:
        print("live-data: BSD key missing; skipping BSD feeds.")
        return
    fetched = utc_now()
    try:
        # Fetch all upcoming + live + recently finished WC events in one sweep
        events = get_all_events(api_key, status="upcoming")
        if mode in ("postmatch", "all"):
            events += get_all_events(api_key, status="finished")
        if mode in ("all",):
            events += get_all_events(api_key, status="live")

        # Deduplicate
        seen: set = set()
        unique_events: list[dict] = []
        for ev in events:
            eid = ev.get("id")
            if eid not in seen:
                seen.add(eid)
                unique_events.append(ev)

        wc_events = [ev for ev in unique_events if _is_worldcup_event(ev)]
        if not wc_events:
            print("live-data: no World Cup events found in BSD response.")
        _save_raw("bsd", "events", wc_events, fetched)

        # ── fixtures ────────────────────────────────────────────────────────
        if mode in ("morning", "postmatch", "all"):
            df = parse_fixtures_bsd(wc_events, fetched)
            if not df.empty:
                _upsert_csv(FIXTURES_CSV, df, ["provider_fixture_id"])
                print(f"live-data: fixtures -> {len(df)} row(s)")

        # ── availability (injuries/suspensions embedded in events) ──────────
        if mode in ("morning", "prekickoff", "all"):
            df = parse_availability_bsd(wc_events, fetched)
            if not df.empty:
                _upsert_csv(AVAILABILITY_CSV, df,
                            ["team", "player", "provider_fixture_id", "source"])
                print(f"live-data: availability -> {len(df)} row(s)")

        # ── per-fixture lineups + stats (require event detail call) ─────────
        if mode in ("prekickoff", "postmatch", "all"):
            fixtures_meta = _load_fixture_rows_for_api()
            lineups_list, stats_list = [], []
            for meta in fixtures_meta:
                eid = meta.get("provider_fixture_id")
                if not eid:
                    continue
                try:
                    detail = get_event(api_key, eid)
                except Exception as exc:
                    print(f"  live-data: BSD event {eid} skipped ({exc})")
                    continue
                _save_raw("bsd", f"event_{eid}", detail, fetched)

                if mode in ("prekickoff", "all"):
                    df_lu = parse_lineups_bsd(detail, meta, fetched)
                    if not df_lu.empty:
                        lineups_list.append(df_lu)

                if mode in ("postmatch", "all"):
                    df_st = parse_match_stats_bsd(detail, meta, fetched)
                    if not df_st.empty:
                        stats_list.append(df_st)

            if lineups_list:
                df = pd.concat(lineups_list, ignore_index=True)
                _upsert_csv(LINEUPS_CSV, df, ["event_id", "team", "player", "role"])
                print(f"live-data: lineups -> {len(df)} row(s)")

            if stats_list:
                df = pd.concat(stats_list, ignore_index=True)
                _upsert_csv(MATCH_STATS_CSV, df, ["event_id", "team"])
                print(f"live-data: stats -> {len(df)} row(s)")

    except Exception as exc:
        print(f"live-data: BSD refresh skipped ({exc})")


def _load_fixture_rows_for_api() -> list[dict]:
    if not FIXTURES_CSV.exists():
        return []
    df = pd.read_csv(FIXTURES_CSV)
    today = datetime.now(_TZ_PDT).date()
    df["match_date"] = df["match_date"].astype(str)
    active = df[df["match_date"] >= today.isoformat()].copy()
    if active.empty:
        active = df.tail(8)
    return active.to_dict("records")


# ---------------------------------------------------------------------------
# Odds (The Odds API — unchanged)
# ---------------------------------------------------------------------------

def _odds_get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def _select_worldcup_sport_key(sports: list[dict]) -> str | None:
    keys = [s.get("key", "") for s in sports]
    if WORLD_CUP_SPORT_KEY in keys:
        return WORLD_CUP_SPORT_KEY
    for s in sports:
        key = s.get("key", "")
        title = s.get("title", "").lower()
        if key.startswith("soccer_") and "world cup" in title and "winner" not in key:
            return key
    return None


def _odds_side(market: str, outcome: dict, home: str, away: str) -> str:
    name = str(outcome.get("name") or "").strip()
    low = name.lower()
    if market in ("h2h", "h2h_3_way"):
        if canonical_team(name) == home:
            return "home"
        if canonical_team(name) == away:
            return "away"
        if low == "draw":
            return "draw"
    if market == "totals":
        return "over" if low == "over" else ("under" if low == "under" else low)
    if market == "btts":
        return "yes" if low == "yes" else ("no" if low == "no" else low)
    if market == "draw_no_bet":
        if canonical_team(name) == home:
            return "home"
        if canonical_team(name) == away:
            return "away"
    if market == "double_chance":
        has_home = home.lower() in low or "home" in low
        has_away = away.lower() in low or "away" in low
        has_draw = "draw" in low or "tie" in low
        parts = []
        if has_home:
            parts.append("home")
        if has_draw:
            parts.append("draw")
        if has_away:
            parts.append("away")
        return "_".join(parts) if parts else low.replace(" ", "_")
    return low.replace(" ", "_")


def normalize_market_snapshots(events: list[dict], fetched_at: str | None = None,
                               teams: set[str] | None = None) -> pd.DataFrame:
    fetched = fetched_at or utc_now()
    known = teams if teams is not None else known_teams()
    rows = []
    for ev in events or []:
        if not ev.get("bookmakers"):
            continue
        home = require_known_team(ev.get("home_team"), known, "odds")
        away = require_known_team(ev.get("away_team"), known, "odds")
        match_date = _local_match_date(ev.get("commence_time", ""))
        event_id = _event_id(match_date, home, away)
        for bk in ev.get("bookmakers") or []:
            book = bk.get("key") or bk.get("title")
            for market in bk.get("markets") or []:
                mkey = market.get("key")
                for oc in market.get("outcomes") or []:
                    price = oc.get("price")
                    try:
                        odds = float(price)
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        "snapshot_time": bk.get("last_update") or fetched,
                        "event_id": event_id,
                        "match_date": match_date,
                        "home": home,
                        "away": away,
                        "provider_event_id": ev.get("id"),
                        "bookmaker": book,
                        "market": mkey,
                        "side": _odds_side(str(mkey), oc, home, away),
                        "line": oc.get("point", ""),
                        "odds": odds,
                        "source": "the-odds-api",
                        "fetched_at": fetched,
                    })
    return pd.DataFrame(rows)


def summarize_wide_market(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Return the existing edge.py wide odds shape from normalized snapshots."""
    cols = ["event_id", "date", "home", "away", "odds_home", "odds_draw", "odds_away",
            "odds_over25", "odds_under25", "odds_btts_yes", "odds_btts_no",
            "bookmaker_count", "market_dispersion_h", "market_dispersion_d",
            "market_dispersion_a"]
    if snapshots is None or snapshots.empty:
        return pd.DataFrame(columns=cols)
    df = snapshots.copy()
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    rows = []
    for (eid, match_date, home, away), ev in df.groupby(
            ["event_id", "match_date", "home", "away"], dropna=False):
        row = {"date": match_date, "home": home, "away": away}
        row["event_id"] = eid
        h2h = ev[(ev["market"].isin(["h2h", "h2h_3_way"]))]
        for side, col in (("home", "odds_home"), ("draw", "odds_draw"),
                          ("away", "odds_away")):
            s = h2h[h2h["side"] == side]["odds"]
            if not s.empty:
                row[col] = float(np.median(s))
        totals = ev[(ev["market"] == "totals") & np.isclose(ev["line"], 2.5)]
        for side, col in (("over", "odds_over25"), ("under", "odds_under25")):
            s = totals[totals["side"] == side]["odds"]
            if not s.empty:
                row[col] = float(np.median(s))
        btts = ev[ev["market"] == "btts"]
        for side, col in (("yes", "odds_btts_yes"), ("no", "odds_btts_no")):
            s = btts[btts["side"] == side]["odds"]
            if not s.empty:
                row[col] = float(np.median(s))
        row["bookmaker_count"] = int(h2h["bookmaker"].nunique())
        disp = _h2h_dispersion(h2h)
        row.update(disp)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def _h2h_dispersion(h2h: pd.DataFrame) -> dict[str, float]:
    out = {"market_dispersion_h": np.nan, "market_dispersion_d": np.nan,
           "market_dispersion_a": np.nan}
    probs = []
    for _, g in h2h.groupby("bookmaker"):
        sides = {r.side: float(r.odds) for r in g.itertuples(index=False)}
        if {"home", "draw", "away"} <= set(sides):
            inv = np.array([1 / sides["home"], 1 / sides["draw"],
                            1 / sides["away"]], float)
            probs.append(inv / inv.sum())
    if probs:
        arr = np.vstack(probs)
        out["market_dispersion_h"] = float(arr[:, 0].std())
        out["market_dispersion_d"] = float(arr[:, 1].std())
        out["market_dispersion_a"] = float(arr[:, 2].std())
    return out


def _merge_bookmakers(existing: dict, new: dict) -> None:
    bks = {b.get("key"): b for b in existing.get("bookmakers", [])}
    for bk in new.get("bookmakers", []):
        key = bk.get("key")
        if key in bks:
            bks[key].setdefault("markets", []).extend(bk.get("markets", []))
        else:
            bks[key] = bk
    existing["bookmakers"] = list(bks.values())


def fetch_odds(api_key: str) -> pd.DataFrame:
    fetched = utc_now()
    if not api_key:
        print("live-data: The Odds API key missing; skipping market snapshots.")
        return pd.DataFrame()
    try:
        sports = _odds_get(f"{ODDS_BASE}/sports/?apiKey={api_key}")
        sport_key = _select_worldcup_sport_key(sports)
        if not sport_key:
            print("live-data: no World Cup match odds sport key found.")
            return pd.DataFrame()
        events_by_id: dict[str, dict] = {}
        for market in ODDS_MARKETS:
            url = (f"{ODDS_BASE}/sports/{sport_key}/odds/?apiKey={api_key}"
                   f"&regions=eu&markets={market}&oddsFormat=decimal")
            try:
                events = _odds_get(url)
            except Exception:
                continue
            for ev in events:
                eid = ev["id"]
                if eid not in events_by_id:
                    events_by_id[eid] = ev
                else:
                    _merge_bookmakers(events_by_id[eid], ev)
        events = list(events_by_id.values())
        _save_raw("the-odds-api", "markets", events, fetched)
        snaps = normalize_market_snapshots(events, fetched)
        _append_snapshot_csv(
            MARKET_SNAPSHOTS_CSV, snaps,
            ["snapshot_time", "event_id", "bookmaker", "market", "side", "line"],
        )
        print(f"live-data: market snapshots -> {len(snaps)} row(s)")
        return snaps
    except Exception as exc:
        print(f"live-data: odds refresh skipped ({exc})")
        return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description="World Cup live data refresh")
    ap.add_argument("--mode", choices=["morning", "prekickoff", "postmatch", "all"],
                    default="all")
    ap.add_argument("--bsd-key", default=BSD_KEY,
                    help="BSD API key (env: BSD_API_KEY, or data/api_keys.json 'bsd')")
    ap.add_argument("--odds-key", default=ODDS_API_KEY)
    ap.add_argument("--no-bsd", action="store_true",
                    help="skip BSD football feeds")
    ap.add_argument("--no-odds", action="store_true",
                    help="skip The Odds API market snapshots")
    args = ap.parse_args()

    if not args.no_bsd:
        fetch_bsd(args.mode, args.bsd_key)
    if not args.no_odds and args.mode in ("morning", "prekickoff", "all"):
        fetch_odds(args.odds_key)


if __name__ == "__main__":
    main()
