#!/usr/bin/env python3
"""P6.1 — daily multi-bookmaker odds snapshots from BSD.

For each upcoming BSD event in our competitions, reads the multi-bookmaker
odds comparison (bsd_client.odds_comparison — only populated close to
kickoff; empty well in advance, which is fine, this is meant to run daily)
and appends to data/odds_history_club.csv:
    snapshot_time (UTC ISO), match_date, competition, home, away,
    market (1x2|total25), side, odds_median, n_books, disp

`disp` = std, across bookmakers, of that side's de-vigged implied
probability (each bookmaker's own odds for the full market are de-vigged
together before taking the cross-bookmaker std — so `disp` reflects
disagreement in fair probability, not just raw price spread).

Dedupe: at most one snapshot per (event, market, side) per 6 hours.
Run from season.py; BSD is free so 2x/day is fine.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_keys import get_key
from bsd_client import get_all_events, league_name as bsd_league_name, odds_comparison

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ODDS_HISTORY_CSV = DATA / "odds_history_club.csv"
POLL_STATE = DATA / "odds_poll_state.json"
COLUMNS = ["snapshot_time", "match_date", "competition", "home", "away",
          "market", "side", "odds_median", "n_books", "disp"]

DEDUPE_WINDOW_HOURS = 6
SNAPSHOT_HORIZON_DAYS = 14   # only near-term events realistically have odds posted
NEAR_HORIZON_DAYS = 3
NEAR_POLL_HOURS = 18
FAR_POLL_HOURS = 72


def _load_poll_state() -> dict[str, str]:
    try:
        raw = json.loads(POLL_STATE.read_text())
    except (OSError, ValueError):
        return {}
    return {
        str(k): str(v) for k, v in raw.items()
        if isinstance(k, str) and isinstance(v, str)
    } if isinstance(raw, dict) else {}


def _save_poll_state(state: dict[str, str]) -> None:
    POLL_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = POLL_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(POLL_STATE)


def _devig_per_book(side_odds: dict[str, dict[str, float]]) -> dict[str, list[float]]:
    """side_odds: {side_key: {book_slug: decimal_odds}}. Returns
    {side_key: [de-vigged implied prob per book that quoted ALL sides]}."""
    books = set()
    for d in side_odds.values():
        books |= set(d.keys())
    out: dict[str, list[float]] = {k: [] for k in side_odds}
    for book in books:
        prices = {}
        ok = True
        for side, d in side_odds.items():
            o = d.get(book)
            if o is None or float(o) <= 1.0:
                ok = False
                break
            prices[side] = float(o)
        if not ok:
            continue
        inv = {s: 1.0 / p for s, p in prices.items()}
        total = sum(inv.values())
        for s, v in inv.items():
            out[s].append(v / total)
    return out


def _market_rows(markets: dict, outcome_map: dict[str, str]) -> dict[str, dict]:
    """outcome_map: {our_side_name: bsd_outcome_key}. Returns
    {our_side_name: {"odds_median": float|None, "n_books": int, "disp": float|None}}."""
    side_odds: dict[str, dict[str, float]] = {}
    for our_side, bsd_key in outcome_map.items():
        entry = markets.get(bsd_key) or {}
        side_odds[our_side] = {
            book: float(v["decimal_odds"])
            for book, v in (entry.get("bookmakers") or {}).items()
            if v.get("decimal_odds") and float(v["decimal_odds"]) > 1.0
        }
    devigged = _devig_per_book(side_odds)
    out = {}
    for our_side, raw in side_odds.items():
        vals = list(raw.values())
        out[our_side] = {
            "odds_median": float(np.median(vals)) if vals else None,
            "n_books": len(vals),
            "disp": float(np.std(devigged[our_side])) if len(devigged[our_side]) >= 3 else None,
        }
    return out


def build_snapshot_rows(api_key: str, days_ahead: int = SNAPSHOT_HORIZON_DAYS,
                        verbose: bool = True,
                        events: list[dict] | None = None,
                        odds_cache: dict[str, dict] | None = None) -> list[dict]:
    from .competitions import comp_from_bsd_league

    today = datetime.now(timezone.utc).date()
    snap_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if events is None:
        try:
            events = get_all_events(
                api_key, status="notstarted", date_from=str(today),
                date_to=str(today + timedelta(days=days_ahead))
            )
        except Exception as exc:
            if verbose:
                print(f"  snapshot_odds: BSD fetch failed ({exc}) — skipped")
            return []
    else:
        from .schema import normalize_status
        end = str(today + timedelta(days=days_ahead))
        events = [
            ev for ev in events
            if normalize_status(ev.get("status")) == "NOT"
            and str(ev.get("event_date") or ev.get("date") or "")[:10] >= str(today)
            and str(ev.get("event_date") or ev.get("date") or "")[:10] <= end
        ]

    rows: list[dict] = []
    n_with_odds = 0
    n_polled = 0
    poll_state = _load_poll_state()
    live_ids = {str(ev.get("id")) for ev in events if ev.get("id") is not None}
    poll_state = {eid: stamp for eid, stamp in poll_state.items() if eid in live_ids}
    for ev in events:
        comp = comp_from_bsd_league(bsd_league_name(ev))
        if comp is None:
            continue
        eid = ev.get("id")
        if eid is None:
            continue
        cache_key = str(eid)
        try:
            kickoff = datetime.fromisoformat(
                str(ev.get("event_date") or ev.get("date") or "")
                .replace("Z", "+00:00")
            )
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
            lead_days = (kickoff - datetime.now(timezone.utc)).total_seconds() / 86400
        except (TypeError, ValueError):
            lead_days = 0
        due_hours = (
            NEAR_POLL_HOURS if lead_days <= NEAR_HORIZON_DAYS
            else FAR_POLL_HOURS
        )
        try:
            last_poll = datetime.fromisoformat(
                poll_state.get(cache_key, "").replace("Z", "+00:00")
            )
            age_hours = (
                datetime.now(timezone.utc) - last_poll
            ).total_seconds() / 3600
        except (TypeError, ValueError):
            age_hours = float("inf")
        if age_hours < due_hours:
            continue
        cmp = odds_cache.get(cache_key) if odds_cache is not None else None
        if cmp is None:
            try:
                cmp = odds_comparison(api_key, eid)
            except Exception:
                continue
            if odds_cache is not None:
                odds_cache[cache_key] = cmp
        poll_state[cache_key] = snap_time
        n_polled += 1
        markets = cmp.get("markets") or {}
        if not markets or cmp.get("bookmakers_count", 0) == 0:
            continue
        n_with_odds += 1

        match_date = str(ev.get("event_date") or ev.get("date") or "")[:10]
        home = str(ev.get("home_team") or "")
        away = str(ev.get("away_team") or "")
        base = {"snapshot_time": snap_time, "match_date": match_date,
               "competition": comp.name, "home": home, "away": away}

        m1x2 = _market_rows(markets.get("1x2") or {}, {"home": "HOME", "draw": "DRAW", "away": "AWAY"})
        for side, vals in m1x2.items():
            if vals["odds_median"] is not None:
                rows.append({**base, "market": "1x2", "side": side, **vals})

        # over_under_25's entries live one level down, inside "over_under_25"
        # (keyed "over@2.5"/"under@2.5"), not directly under "markets".
        ou_market = markets.get("over_under_25") or {}
        mou = _market_rows({"over@2.5": ou_market.get("over@2.5"),
                           "under@2.5": ou_market.get("under@2.5")},
                           {"over": "over@2.5", "under": "under@2.5"})
        for side, vals in mou.items():
            if vals["odds_median"] is not None:
                rows.append({**base, "market": "total25", "side": side, **vals})

    _save_poll_state(poll_state)
    if verbose:
        print(f"  snapshot_odds: {n_with_odds} events with live odds "
              f"({n_polled} polled of {len(events)} upcoming), "
              f"{len(rows)} side-rows")
    return rows


def append_snapshots(rows: list[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(rows, columns=COLUMNS)
    if ODDS_HISTORY_CSV.exists():
        try:
            old = pd.read_csv(ODDS_HISTORY_CSV)
        except Exception:
            old = pd.DataFrame(columns=COLUMNS)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    if combined.empty:
        DATA.mkdir(exist_ok=True)
        combined.to_csv(ODDS_HISTORY_CSV, index=False)
        return combined

    # mixed=True: old rows round-tripped through CSV may render as
    # "YYYY-MM-DD HH:MM:SS+00:00" (pandas' str(Timestamp)) while freshly
    # generated rows use datetime.isoformat()'s "T" separator — without
    # mixed=True, pandas infers one format from the first rows and silently
    # NaT's every row that doesn't match it instead of parsing per-row.
    combined["snapshot_time"] = pd.to_datetime(combined["snapshot_time"], utc=True,
                                               format="mixed", errors="coerce")
    combined = combined.dropna(subset=["snapshot_time"]).sort_values("snapshot_time")

    # Dedupe: at most one snapshot per (event, market, side) per
    # DEDUPE_WINDOW_HOURS — keep the first snapshot in each window, walking
    # chronologically per group.
    kept_rows = []
    last_kept: dict[tuple, pd.Timestamp] = {}
    window = pd.Timedelta(hours=DEDUPE_WINDOW_HOURS)
    for r in combined.itertuples(index=False):
        key = (r.match_date, r.home, r.away, r.market, r.side)
        prev = last_kept.get(key)
        if prev is not None and (r.snapshot_time - prev) < window:
            continue
        last_kept[key] = r.snapshot_time
        kept_rows.append(r._asdict())
    out = pd.DataFrame(kept_rows, columns=COLUMNS)
    out["snapshot_time"] = out["snapshot_time"].apply(lambda ts: ts.isoformat())
    DATA.mkdir(exist_ok=True)
    out.to_csv(ODDS_HISTORY_CSV, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", dest="api_key")
    ap.add_argument("--days-ahead", type=int, default=SNAPSHOT_HORIZON_DAYS)
    args = ap.parse_args()
    key = args.api_key or get_key("bsd", env="BSD_API_KEY")
    if not key:
        sys.exit("No BSD key — set BSD_API_KEY or use --api-key.")
    rows = build_snapshot_rows(key, days_ahead=args.days_ahead)
    df = append_snapshots(rows)
    print(f"Wrote {len(df)} total rows -> {ODDS_HISTORY_CSV} ({len(rows)} new this run)")


if __name__ == "__main__":
    main()
