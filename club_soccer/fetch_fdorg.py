#!/usr/bin/env python3
"""Fetch Club Soccer fixtures from football-data.org (supplementary source).

football-data.org is a free REST API covering 10 major European competitions
with live upcoming fixtures and historical results — no rate limits beyond
10 requests/minute on the free tier.

Register at https://www.football-data.org/client/register and add your key:
  data/api_keys.json -> "football-data-org": "YOUR_KEY"
  or set env FOOTBALL_DATA_ORG_KEY

Free-tier competitions covered
-------------------------------
PL   Premier League
ELC  Championship
EL1  League One
EL2  League Two
FAC  FA Cup
DFB  DFB-Pokal
BL1  Bundesliga
SA   Serie A
FL1  Ligue 1
PD   La Liga
CL   UEFA Champions League
EL   UEFA Europa League

Usage
-----
python3 -m club_soccer.fetch_fdorg --merge
python3 -m club_soccer.fetch_fdorg --status SCHEDULED --merge
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from api_keys import get_key
from .competitions import COMPETITIONS, BY_NAME, FDORG_COMPETITIONS

DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
RAW = DATA / "fdorg_cache"

FDORG_BASE = "https://api.football-data.org/v4"
_RATE_PAUSE = 0.7          # seconds between requests (free tier: 10 req/min)
_TIMEOUT = 30


def _get(path: str, api_key: str) -> dict:
    url = f"{FDORG_BASE}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "X-Auth-Token": api_key,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"football-data.org HTTP {exc.code} for {path}: {exc.reason}"
        ) from exc


def _fdorg_to_row(match: dict, comp_name: str, comp_api_id: int,
                  country: str, kind: str) -> dict:
    """Map a football-data.org match object to our fixtures.csv schema."""
    home_obj = match.get("homeTeam") or {}
    away_obj = match.get("awayTeam") or {}
    home = home_obj.get("name") or home_obj.get("shortName") or ""
    away = away_obj.get("name") or away_obj.get("shortName") or ""

    date_raw = str(match.get("utcDate") or "")[:10]   # YYYY-MM-DD
    season_obj = match.get("season") or {}
    start_date = str(season_obj.get("startDate") or date_raw)
    try:
        season = int(start_date[:4])
    except ValueError:
        season = None

    score_obj = match.get("score") or {}
    ft = score_obj.get("fullTime") or {}
    status = str(match.get("status") or "").upper()
    finished = status in ("FINISHED",)

    home_goals = ft.get("home") if finished else None
    away_goals = ft.get("away") if finished else None

    return {
        "fixture_id": match.get("id"),
        "date": date_raw,
        "season": season,
        "competition": comp_name,
        "competition_id": comp_api_id,
        "country": country,
        "type": kind,
        "home_id": home_obj.get("id") or "",
        "home": home,
        "away_id": away_obj.get("id") or "",
        "away": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "status": status[:3],
        "neutral": 0,
        "home_shots": "",
        "away_shots": "",
        "home_sot": "",
        "away_sot": "",
        "home_corners": "",
        "away_corners": "",
    }


def fetch_competition(comp_name: str, api_key: str,
                      status: str | None = None,
                      season: int | None = None) -> list[dict]:
    """Fetch matches for one competition from football-data.org.

    Parameters
    ----------
    comp_name:  Competition name from our registry (e.g. "Premier League").
    api_key:    football-data.org API key.
    status:     "SCHEDULED", "FINISHED", or None (all).
    season:     Season start year (e.g. 2025 for 2025/26).
    """
    comp = BY_NAME.get(comp_name)
    if comp is None or not comp.fdorg_code:
        return []

    path = f"/competitions/{comp.fdorg_code}/matches"
    params: list[str] = []
    if status:
        params.append(f"status={status}")
    if season:
        params.append(f"season={season}")
    if params:
        path += "?" + "&".join(params)

    RAW.mkdir(parents=True, exist_ok=True)
    cache_key = f"{comp.fdorg_code}_{season or 'all'}_{status or 'all'}"
    cache_file = RAW / f"{cache_key}.json"

    try:
        data = _get(path, api_key)
        cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"  fdorg: {comp_name} fetch failed ({exc}); "
              f"{'using cache' if cache_file.exists() else 'skipping'}")
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
        else:
            return []

    matches = data.get("matches") or []
    rows = [_fdorg_to_row(m, comp.name, comp.api_id, comp.country, comp.kind)
            for m in matches]
    return rows


def fetch_all(api_key: str,
              competitions: list[str] | None = None,
              status: str | None = None,
              season: int | None = None,
              pause: float = _RATE_PAUSE) -> pd.DataFrame:
    """Fetch all football-data.org-covered competitions.

    Parameters
    ----------
    competitions:  List of competition names to fetch; defaults to all covered.
    status:        "SCHEDULED", "FINISHED", or None (all statuses).
    season:        Season start year filter.
    pause:         Seconds between API requests (respect 10 req/min free tier).
    """
    comps = competitions or list(FDORG_COMPETITIONS.keys())
    all_rows: list[dict] = []
    for i, name in enumerate(comps):
        if name not in FDORG_COMPETITIONS:
            print(f"  fdorg: {name!r} not covered by free tier — skipping")
            continue
        rows = fetch_competition(name, api_key, status=status, season=season)
        print(f"  fdorg: {name}: {len(rows)} match(es)")
        all_rows.extend(rows)
        if i < len(comps) - 1:
            time.sleep(pause)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["fixture_id"], keep="last")
    return df


def fetch_and_merge(api_key: str,
                    competitions: list[str] | None = None,
                    status: str | None = None,
                    season: int | None = None) -> pd.DataFrame:
    """Fetch from football-data.org and merge onto existing fixtures.csv.

    Uses fixture_id de-duplication so BSD rows (which have different IDs)
    are preserved; fdorg rows only update where fixture_id matches.
    """
    df_new = fetch_all(api_key, competitions=competitions,
                       status=status, season=season)
    if df_new.empty:
        print("  fdorg: no rows fetched — fixtures.csv unchanged")
        return pd.read_csv(FIXTURES) if FIXTURES.exists() else df_new

    if FIXTURES.exists():
        existing = pd.read_csv(FIXTURES)
        merged = pd.concat([existing, df_new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["fixture_id"], keep="last")
    else:
        merged = df_new

    DATA.mkdir(exist_ok=True)
    merged.to_csv(FIXTURES, index=False)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch club soccer fixtures from football-data.org (free tier).",
        epilog=(
            "Register at https://www.football-data.org/client/register "
            "and add 'football-data-org' key to data/api_keys.json."
        ),
    )
    ap.add_argument(
        "--competitions", nargs="+",
        help="subset of competition names to pull (default: all free-tier covered)",
    )
    ap.add_argument(
        "--status", choices=["SCHEDULED", "FINISHED"],
        help="filter by match status (default: fetch all)",
    )
    ap.add_argument(
        "--season", type=int,
        help="season start year (e.g. 2025 for 2025/26)",
    )
    ap.add_argument(
        "--merge", action="store_true",
        help="merge onto existing fixtures.csv (recommended; keeps BSD rows intact)",
    )
    ap.add_argument("--api-key", dest="api_key")
    args = ap.parse_args()

    key = args.api_key or get_key("football-data-org", env="FOOTBALL_DATA_ORG_KEY")
    if not key:
        sys.exit(
            "No football-data.org key. Register at "
            "https://www.football-data.org/client/register and add "
            "'football-data-org' to data/api_keys.json, "
            "or set FOOTBALL_DATA_ORG_KEY."
        )

    print(f"\nFetching from football-data.org "
          f"({len(args.competitions or list(FDORG_COMPETITIONS))} competitions)...")
    if args.merge:
        df = fetch_and_merge(key, competitions=args.competitions,
                             status=args.status, season=args.season)
    else:
        df = fetch_all(key, competitions=args.competitions,
                       status=args.status, season=args.season)
        DATA.mkdir(exist_ok=True)
        df.to_csv(FIXTURES, index=False)

    played = df[df["home_goals"].notna() & df["away_goals"].notna()] if not df.empty else df
    print(f"\nWrote {len(df)} fixtures ({len(played)} played) -> {FIXTURES}")


if __name__ == "__main__":
    main()
