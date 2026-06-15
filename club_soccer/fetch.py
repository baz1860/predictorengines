#!/usr/bin/env python3
"""Fetch/cache Club Soccer fixtures from API-Football, with CSV fallback."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_keys import get_key
from competitions import COMPETITIONS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
RAW = DATA / "api_cache"
BASE = "https://v3.football.api-sports.io"


def _request(path: str, params: dict, api_key: str) -> dict:
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"x-apisports-key": api_key})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _fixture_rows(payload: dict, competition) -> list[dict]:
    rows = []
    for item in payload.get("response", []):
        fx = item.get("fixture", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        status = fx.get("status", {})
        rows.append({
            "fixture_id": fx.get("id"),
            "date": str(fx.get("date", ""))[:10],
            "season": item.get("league", {}).get("season"),
            "competition": competition.name,
            "competition_id": competition.api_id,
            "country": competition.country,
            "type": competition.kind,
            "home_id": teams.get("home", {}).get("id"),
            "home": teams.get("home", {}).get("name"),
            "away_id": teams.get("away", {}).get("id"),
            "away": teams.get("away", {}).get("name"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "status": status.get("short"),
            "neutral": 0,
            "home_shots": "",
            "away_shots": "",
            "home_sot": "",
            "away_sot": "",
            "home_corners": "",
            "away_corners": "",
        })
    return rows


def fetch_fixtures(season: int, current: bool = False, api_key: str | None = None) -> pd.DataFrame:
    key = api_key or get_key("api-football", env="API_FOOTBALL_KEY")
    if not key:
        raise ValueError("No API-Football key. Add data/api_keys.json or pass --api-key.")
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for comp in COMPETITIONS:
        payload = _request("/fixtures", {"league": comp.api_id, "season": season}, key)
        (RAW / f"fixtures_{comp.api_id}_{season}.json").write_text(json.dumps(payload))
        rows.extend(_fixture_rows(payload, comp))
    df = pd.DataFrame(rows)
    if current and FIXTURES.exists():
        existing = pd.read_csv(FIXTURES)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["fixture_id"], keep="last")
    DATA.mkdir(exist_ok=True)
    df.to_csv(FIXTURES, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--current", action="store_true",
                    help="merge fetched rows into existing fixtures.csv")
    ap.add_argument("--api-key")
    args = ap.parse_args()
    try:
        df = fetch_fixtures(args.season, args.current, args.api_key)
    except Exception as e:
        sys.exit(str(e))
    print(f"Wrote {len(df)} fixture rows -> {FIXTURES}")


if __name__ == "__main__":
    main()
