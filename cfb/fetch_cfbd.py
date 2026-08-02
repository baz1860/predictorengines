#!/usr/bin/env python3
"""Pull the CFBD roster/schedule inputs for a season into data/cfbd/ and data/.

Fetches for the given year (default: the upcoming season):
  /talent            -> data/cfbd/talent_<yr>.json      (247 composite)
  /player/returning  -> data/cfbd/returning_<yr>.json   (returning production)
  /games             -> data/schedule_<yr>.json         (full season schedule)

These feed the Elo preseason priors (priors.py) and win_totals.py. An empty
API response never overwrites an existing non-empty file, and a year that CFBD
has not published yet is reported, not written.

Key: 'collegefootballdata' in data/api_keys.json (or CFBD_API_KEY).

Usage: python3 -m cfb.fetch_cfbd [year]
"""
import argparse
import json
import os
import sys
import tempfile
import urllib.request
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
CFBD_DIR = os.path.join(HERE, "data", "cfbd")


def _key() -> str:
    k = os.environ.get("CFBD_API_KEY", "").strip()
    if k:
        return k
    sys.path.insert(0, os.path.dirname(HERE))
    from api_keys import get_key
    return (get_key("collegefootballdata") or "").strip()


def pull(path: str, key: str):
    req = urllib.request.Request(f"https://api.collegefootballdata.com{path}",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def save(data, dest: str, label: str) -> None:
    if not data:
        state = "kept existing" if (
            os.path.exists(dest) and os.path.getsize(dest) > 2) else "not written"
        print(f"  {label}: CFBD has no data yet ({state})")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(dest)}.",
                               dir=os.path.dirname(dest))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        with open(tmp) as f:
            staged = json.load(f)
        if not isinstance(staged, list) or not staged:
            raise ValueError(f"staged {label} payload is empty or not a list")
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print(f"  {label}: {len(data)} rows -> {os.path.relpath(dest, HERE)}")


def prepare_schedule(data: object, year: int) -> list[dict]:
    """Validate CFBD schedule identity and retain the model's FBS scope."""
    if not isinstance(data, list):
        raise ValueError("CFBD schedule payload is not a list")
    rows: list[dict] = []
    ids: set[str] = set()
    for game in data:
        if not isinstance(game, dict):
            raise ValueError("CFBD schedule contains a non-object row")
        if int(game.get("season", -1)) != int(year):
            raise ValueError(f"CFBD schedule contains a non-{year} row")
        game_id = str(game.get("id") or "").strip()
        if not game_id or game_id in ids:
            raise ValueError("CFBD schedule has a missing or duplicate event ID")
        ids.add(game_id)
        if not game.get("startDate") or not game.get("homeTeam") or not game.get("awayTeam"):
            raise ValueError(f"CFBD schedule event {game_id} lacks kickoff/team identity")
        divisions = {
            str(game.get("homeClassification") or "").lower(),
            str(game.get("awayClassification") or "").lower(),
        }
        if "fbs" in divisions:
            rows.append(game)
    if len(rows) < 100:
        raise ValueError(f"CFBD schedule has inadequate FBS coverage: {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("year", nargs="?", type=int)
    parser.add_argument("--schedule-only", action="store_true",
                        help="refresh only the validated FBS-involved schedule")
    args = parser.parse_args()
    year = args.year or (
        date.today().year if date.today().month >= 2 else date.today().year - 1)
    key = _key()
    if not key:
        raise SystemExit("No CFBD key. Set CFBD_API_KEY or add "
                         "data/api_keys.json key 'collegefootballdata'.")
    os.makedirs(CFBD_DIR, exist_ok=True)
    print(f"CFBD pulls for {year}:")
    if not args.schedule_only:
        save(pull(f"/talent?year={year}", key),
             os.path.join(CFBD_DIR, f"talent_{year}.json"), "talent (247 composite)")
        save(pull(f"/player/returning?year={year}", key),
             os.path.join(CFBD_DIR, f"returning_{year}.json"), "returning production")
    schedule = prepare_schedule(pull(f"/games?year={year}", key), year)
    save(schedule,
         os.path.join(HERE, "data", f"schedule_{year}.json"), "season schedule")


if __name__ == "__main__":
    main()
