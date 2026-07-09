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
import json
import os
import sys
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
    with open(dest, "w") as f:
        json.dump(data, f)
    print(f"  {label}: {len(data)} rows -> {os.path.relpath(dest, HERE)}")


def main() -> None:
    year = int(sys.argv[1]) if len(sys.argv) > 1 else (
        date.today().year if date.today().month >= 2 else date.today().year - 1)
    key = _key()
    if not key:
        raise SystemExit("No CFBD key. Set CFBD_API_KEY or add "
                         "data/api_keys.json key 'collegefootballdata'.")
    os.makedirs(CFBD_DIR, exist_ok=True)
    print(f"CFBD pulls for {year}:")
    save(pull(f"/talent?year={year}", key),
         os.path.join(CFBD_DIR, f"talent_{year}.json"), "talent (247 composite)")
    save(pull(f"/player/returning?year={year}", key),
         os.path.join(CFBD_DIR, f"returning_{year}.json"), "returning production")
    save(pull(f"/games?year={year}", key),
         os.path.join(HERE, "data", f"schedule_{year}.json"), "season schedule")


if __name__ == "__main__":
    main()
