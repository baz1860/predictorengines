#!/usr/bin/env python3
"""Add UEFA club competition data (Champions/Europa/Conference League, incl. the
Swiss-model league phase) to club_soccer from openfootball — FREE, CC0, no key.

    python3 club_soccer/seed_openfootball.py --seasons 2024 2025 --merge

Source: github.com/openfootball/champions-league (public domain football.txt).
Covers 2011-12 .. current, INCLUDING the 36-team league phase from 2024-25 on.

Why bother: UEFA ties are the only matches that connect teams from different
domestic leagues, so they sharpen cross-league strength in elo + goals. NOTE:
openfootball has results only — no shots — so the xg/xgf ensemble members fall
back to the goals/elo maps for these matches (handled in model._lambdas_xg).

Use --merge to APPEND onto the existing football-data league fixtures (which keep
their shot stats); without it, writes a UEFA-only fixtures file to --out.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as M
from competitions import BY_NAME
from names import make_canon

DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
RAW = "https://raw.githubusercontent.com/openfootball/champions-league/master"

FILES = {"cl.txt": "Champions League", "el.txt": "Europa League",
         "conf.txt": "Conference League"}

MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
          "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

# openfootball dates: "Tue Sep 17 2024" (space-sep) or older "Tue Sep/17 2024"
DATE_RE = re.compile(r"^\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})[/ ](\d{1,2})(?:\s+(\d{4}))?")
MATCH_RE = re.compile(r"^(?:\s*\d{1,2}[.:]\d{2}\s+)?(.+?)\s+v\s+(.+?)\s{2,}(\d.*)$")

def fid(comp_id, date, home, away):
    h = hashlib.md5(f"of|{comp_id}|{date}|{home}|{away}".encode()).hexdigest()
    return int(h[:8], 16)


def parse(text, comp_name, canon):
    comp = BY_NAME[comp_name]
    rows, year, date = [], None, None
    for line in text.splitlines():
        if line.startswith("="):
            m = re.search(r"(\d{4})/(\d{2})", line)
            if m:
                year = int(m.group(1))
            continue
        dm = DATE_RE.match(line)
        if dm:
            mon, day = MONTHS[dm.group(2)], int(dm.group(3))
            if dm.group(4):
                year = int(dm.group(4))
            yr = year if mon >= 7 else (year + 1 if year else None)  # season spans 2 cal years
            date = f"{yr:04d}-{mon:02d}-{day:02d}" if yr else None
            continue
        mm = MATCH_RE.match(line)
        if not mm or date is None:
            continue
        home_raw, away_raw, res = mm.group(1), mm.group(2), mm.group(3)
        res_noht = re.sub(r"\([^)]*\)", "", res)
        if "a.e.t." in res_noht:
            sm = re.search(r"(\d+)-(\d+)\s*a\.e\.t\.", res_noht)
        elif "pen." in res_noht:
            sm = re.search(r"(\d+)-(\d+)\s*pen", res_noht) or re.search(r"(\d+)-(\d+)", res_noht)
        else:
            sm = re.search(r"(\d+)-(\d+)", res_noht)
        if not sm:
            continue
        home, away = canon(home_raw), canon(away_raw)
        rows.append({
            "fixture_id": fid(comp.api_id, date, home, away),
            "date": date, "season": int(date[:4]),
            "competition": comp.name, "competition_id": comp.api_id,
            "country": comp.country, "type": "europe",
            "home_id": "", "home": home, "away_id": "", "away": away,
            "home_goals": int(sm.group(1)), "away_goals": int(sm.group(2)),
            "status": "FT", "neutral": 0,
            "home_shots": "", "away_shots": "", "home_sot": "", "away_sot": "",
            "home_corners": "", "away_corners": "",
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025],
                    help="season start years (2024 = 2024-25). default: 2024 2025")
    ap.add_argument("--merge", action="store_true",
                    help="append onto existing fixtures.csv (dedupe by fixture_id)")
    ap.add_argument("--out", default=str(DATA / "uefa_fixtures.csv"),
                    help="output path when not merging")
    ap.add_argument("--revalidate", action="store_true",
                    help="after --merge, refit + rewrite the walk-forward baseline")
    args = ap.parse_args()

    if not FIXTURES.exists():
        sys.exit("No fixtures.csv yet — run seed_footballdata.py first to build the league base.")
    league_teams = set(pd.read_csv(FIXTURES, usecols=["home", "away"]).stack().unique())
    canon = make_canon(league_teams)

    all_rows, mapped, kept_new = [], set(), set()
    for yr in args.seasons:
        for fn, comp_name in FILES.items():
            url = f"{RAW}/{yr:04d}-{(yr + 1) % 100:02d}/{fn}"
            try:
                text = urllib.request.urlopen(url, timeout=20).read().decode("utf-8")
            except Exception as e:
                print(f"  - {yr}-{(yr+1)%100:02d}/{fn}: {str(e)[:40]}")
                continue
            rows = parse(text, comp_name, canon)
            for r in rows:
                for side in (r["home"], r["away"]):
                    (mapped if side in league_teams else kept_new).add(side)
            all_rows.extend(rows)
            print(f"  {comp_name} {yr}-{(yr+1)%100:02d}: {len(rows)} matches")

    if not all_rows:
        sys.exit("No UEFA data fetched.")
    uefa = pd.DataFrame(all_rows).drop_duplicates(subset=["fixture_id"])
    print(f"\n{len(uefa)} UEFA matches | {len(mapped)} teams linked to league data, "
          f"{len(kept_new)} new (UEFA-only) teams")
    print("  sample new teams:", ", ".join(sorted(kept_new)[:12]))

    if args.merge:
        base = pd.read_csv(FIXTURES)
        merged = pd.concat([base, uefa], ignore_index=True).drop_duplicates(
            subset=["fixture_id"], keep="first")
        merged.to_csv(FIXTURES, index=False)
        print(f"\nMerged -> {FIXTURES}  ({len(base)} -> {len(merged)} rows)")
        if args.revalidate:
            print("\nRefitting + revalidating...")
            M.save_params(M.fit())
            # subprocess so club_soccer/validate.py loads (not the root one)
            import subprocess
            subprocess.run([sys.executable, str(HERE / "validate.py"), "--update-baseline"],
                           cwd=str(HERE), check=False)
    else:
        uefa.to_csv(args.out, index=False)
        print(f"\nWrote UEFA-only -> {args.out} (use --merge to add into fixtures.csv)")


if __name__ == "__main__":
    main()
