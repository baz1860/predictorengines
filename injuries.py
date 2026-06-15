#!/usr/bin/env python3
"""Pull current World Cup injury/unavailability reports from API-Football.

Writes data/absences_api.csv (team,player,note), which squads.py merges
with your manual data/absences.csv. Manual entries always survive; rerunning
this script only replaces previously API-sourced rows.

Needs a (free) API key from https://www.api-football.com - either pass
--api-key, set API_FOOTBALL_KEY, or add it to data/api_keys.json. One request
per run (league 1 = FIFA World Cup), so the free 100/day quota is never a
concern.

Like edge.py, this needs internet access - run it from your own Terminal,
not the sandbox:

  python3 injuries.py --api-key YOUR_KEY
  python3 injuries.py --api-key YOUR_KEY --dry-run   # show, don't write
  python3 squads.py                                  # then refresh ratings
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

from api_keys import get_key

HERE = Path(__file__).parent
OUT = HERE / "data" / "absences_api.csv"

API_KEY = get_key("api-football", env="API_FOOTBALL_KEY")
LEAGUE_WC = 1
SEASON = 2026

# API-Football team names -> dataset names (extend as mismatches appear)
TEAM_ALIAS = {
    "USA": "United States", "South Korea": "South Korea",
    "Korea Republic": "South Korea", "Iran": "Iran", "IR Iran": "Iran",
    "Ivory Coast": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde", "Curacao": "Curaçao",
    "Congo DR": "DR Congo", "Turkiye": "Turkey", "Czechia": "Czech Republic",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def fetch(api_key):
    url = (f"https://v3.football.api-sports.io/injuries"
           f"?league={LEAGUE_WC}&season={SEASON}")
    req = urllib.request.Request(url, headers={"x-apisports-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.load(r)
    except Exception as e:
        sys.exit(f"Could not reach API-Football ({e}).\n"
                 "Run this from your own Terminal (the sandbox has no internet), "
                 "or add absences by hand to data/absences.csv.")
    if payload.get("errors"):
        sys.exit(f"API-Football error: {payload['errors']}")
    return payload.get("response", [])


def main():
    ap = argparse.ArgumentParser(description="API-Football injury feed")
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.api_key:
        sys.exit("No API key. Get a free one at https://www.api-football.com "
                 "and pass --api-key, set API_FOOTBALL_KEY, or add "
                 "data/api_keys.json.")

    rows = []
    for item in fetch(args.api_key):
        team = item["team"]["name"]
        team = TEAM_ALIAS.get(team, team)
        player = item["player"]["name"]
        why = item["player"].get("reason") or item["player"].get("type") or "injury"
        rows.append((team, player, f"api: {why}"))

    if not rows:
        print("API returned no current World Cup injury reports.")
    for t, p, n in rows:
        print(f"  {t}: {p} ({n})")
    if args.dry_run:
        return
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("team,player,note\n")
        for t, p, n in rows:
            f.write(f'"{t}","{p}","{n}"\n')
    print(f"\nWrote {len(rows)} rows -> {OUT.relative_to(HERE)}")
    print("Now run: python3 squads.py   (to refresh squad_ratings.csv)")


if __name__ == "__main__":
    main()
