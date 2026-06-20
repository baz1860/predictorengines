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
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_keys import get_key
from bsd_client import get_all_events, league_name as bsd_league_name, event_date_utc
from .competitions import COMPETITIONS, comp_from_bsd_league

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
RAW = DATA / "bsd_cache"

# BSD event status values
_FINISHED_STATUSES = {"finished", "ft", "aet", "pen"}
_UPCOMING_STATUSES = {"upcoming", "scheduled", "ns"}


def _bsd_to_fixture_row(event: dict, comp_name: str, comp_api_id: int,
                        country: str, kind: str) -> dict:
    """Map a single BSD event dict to our fixtures.csv schema."""
    home = event.get("home_team") or ""
    away = event.get("away_team") or ""
    kickoff = event_date_utc(event)
    date_str = str(kickoff)[:10]          # YYYY-MM-DD
    status_raw = str(event.get("status") or "").lower()
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

    return {
        "fixture_id": event.get("id"),
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
        "home_shots": "",
        "away_shots": "",
        "home_sot": "",
        "away_sot": "",
        "home_corners": "",
        "away_corners": "",
    }


def _fetch_bsd_events(api_key: str, status: str | None = None) -> list[dict]:
    """Fetch all BSD football events (optionally filtered by status)."""
    RAW.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    if status:
        kwargs["status"] = status
    return get_all_events(api_key, **kwargs)


def fetch_fixtures(season: int | None = None,
                   current: bool = False,
                   api_key: str | None = None,
                   status: str | None = None) -> pd.DataFrame:
    """Fetch club soccer fixtures from BSD and return as a DataFrame.

    Parameters
    ----------
    season:   If provided, filter rows to this season start year.
    current:  If True, merge fetched rows onto the existing fixtures.csv
              (de-duped by fixture_id, keeping latest).
    api_key:  BSD API key.  Falls back to BSD_API_KEY env / api_keys.json.
    status:   BSD status filter: "upcoming", "finished", or None (all).
    """
    key = api_key or get_key("bsd", env="BSD_API_KEY")
    if not key:
        raise ValueError(
            "No BSD key. Register at https://sports.bzzoiro.com/register/ "
            "and add 'bsd' to data/api_keys.json, or set BSD_API_KEY."
        )

    events = _fetch_bsd_events(key, status=status)

    # Build a name->Competition lookup for fast resolution
    rows: list[dict] = []
    unmatched_leagues: set[str] = set()

    for ev in events:
        lname = bsd_league_name(ev)
        comp = comp_from_bsd_league(lname)
        if comp is None:
            unmatched_leagues.add(lname)
            continue
        row = _bsd_to_fixture_row(ev, comp.name, comp.api_id,
                                  comp.country, comp.kind)
        rows.append(row)

    if unmatched_leagues:
        # Log at most 10 so it's not noisy
        shown = sorted(unmatched_leagues)[:10]
        suffix = f" (+{len(unmatched_leagues) - 10} more)" if len(unmatched_leagues) > 10 else ""
        print(f"  fetch: unrecognised BSD leagues (ignored): {shown}{suffix}")

    df = pd.DataFrame(rows)

    if season is not None and not df.empty:
        df = df[df["season"] == season].copy()

    if not df.empty:
        df = df.drop_duplicates(subset=["fixture_id"], keep="last")

    if current and FIXTURES.exists():
        existing = pd.read_csv(FIXTURES)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["fixture_id"], keep="last")

    DATA.mkdir(exist_ok=True)
    df.to_csv(FIXTURES, index=False)
    return df


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
    ap.add_argument("--api-key", dest="api_key",
                    help="BSD API key (overrides env / api_keys.json)")
    args = ap.parse_args()
    try:
        df = fetch_fixtures(
            season=args.season,
            current=args.current,
            api_key=args.api_key,
            status=args.status,
        )
    except Exception as e:
        sys.exit(str(e))
    print(f"Wrote {len(df)} fixture rows -> {FIXTURES}")


if __name__ == "__main__":
    main()
