#!/usr/bin/env python3
"""Seed or extend the Club Soccer fixtures from API-Football.

Two modes:

  # FRESH full reseed (overwrites fixtures.csv with all competitions)
  python3 club_soccer/seed_real.py --seasons 2024 2025
  python3 club_soccer/seed_real.py --seasons 2024 2025 --stats --max-stats 600

  # MERGE just cups (or UEFA) ONTO the existing league base, keeping its shots
  python3 club_soccer/seed_real.py --seasons 2024 2025 --cups --merge
  python3 club_soccer/seed_real.py --seasons 2024 2025 --competitions "FA Cup" "EFL Cup" --merge

`--cups` selects every cup competition, `--uefa` every European one, or name them
explicitly with `--competitions`. In `--merge` mode the fetched rows are appended
to fixtures.csv (deduped by fixture_id, so league rows and their shot stats are
preserved) and team names are reconciled to the football-data canon via names.py;
a mapping report flags any club that stayed unlinked.

API-Football key: --api-key, then API_FOOTBALL_KEY, then data/api_keys.json. Free
tier = 100 req/day: fixtures cost 1 req per competition per season (cheap, even for
all cups + UEFA); --stats costs 1 req per finished match (expensive — cap it).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fetch as F          # reuse request + row helpers
import model as M
from api_keys import get_key
from competitions import COMPETITIONS, BY_API_ID
from names import make_canon

DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
BACKUP = DATA / "fixtures_prev.bak.csv"
STATS_CACHE = DATA / "api_cache"

# API-Football statistics "type" -> our column suffix
STAT_MAP = {
    "Total Shots": "shots",
    "Shots on Goal": "sot",
    "Corner Kicks": "corners",
}


def fetch_fixtures(seasons: list[int], key: str, comps=None, canon=None) -> pd.DataFrame:
    """Pull fixtures for the given competitions/seasons. If `canon` is given,
    reconcile team names onto the league canon (used in --merge mode)."""
    comps = comps or COMPETITIONS
    F.RAW.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for season in seasons:
        for comp in comps:
            try:
                payload = F._request("/fixtures",
                                     {"league": comp.api_id, "season": season}, key)
            except Exception as e:
                print(f"  ! {comp.name} {season}: {e}")
                continue
            (F.RAW / f"fixtures_{comp.api_id}_{season}.json").write_text(
                json.dumps(payload))
            new = F._fixture_rows(payload, comp)
            if canon:
                for r in new:
                    if r.get("home"):
                        r["home"] = canon(r["home"])
                    if r.get("away"):
                        r["away"] = canon(r["away"])
            rows.extend(new)
            print(f"  {comp.name} {season}: {len(new)} fixtures")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["fixture_id"], keep="last")
    return df


def select_competitions(names, cups, uefa):
    """Resolve the competition subset from --competitions/--cups/--uefa."""
    if not (names or cups or uefa):
        return list(COMPETITIONS)
    wanted = set(names or [])
    if cups:
        wanted |= {c.name for c in COMPETITIONS if c.kind == "cup"}
    if uefa:
        wanted |= {c.name for c in COMPETITIONS if c.kind == "europe"}
    comps = [c for c in COMPETITIONS if c.name in wanted]
    unknown = wanted - {c.name for c in COMPETITIONS}
    if unknown:
        print(f"  ! unknown competitions ignored: {sorted(unknown)}")
    return comps


def fetch_stats(df: pd.DataFrame, key: str, max_stats: int, pause: float) -> pd.DataFrame:
    """Fill shots/sot/corners for finished matches via /fixtures/statistics."""
    STATS_CACHE.mkdir(parents=True, exist_ok=True)
    played = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    todo = played["fixture_id"].dropna().astype(int).tolist()[:max_stats]
    print(f"\nFetching shot stats for {len(todo)} finished matches "
          f"(of {len(played)} played; cap {max_stats})...")
    by_id = {int(fid): {} for fid in todo}
    for n, fid in enumerate(todo, 1):
        cache = STATS_CACHE / f"stats_{fid}.json"
        if cache.exists():
            payload = json.loads(cache.read_text())
        else:
            try:
                payload = F._request("/fixtures/statistics", {"fixture": fid}, key)
            except Exception as e:
                print(f"  ! stats {fid}: {e}")
                continue
            cache.write_text(json.dumps(payload))
            time.sleep(pause)
        resp = payload.get("response", []) or []
        if len(resp) < 2:
            continue
        # response[0] = home team block, response[1] = away (API-Football order)
        for side, block in (("home", resp[0]), ("away", resp[1])):
            for stat in block.get("statistics", []) or []:
                col = STAT_MAP.get(str(stat.get("type", "")))
                if col is None:
                    continue
                val = stat.get("value")
                by_id[fid][f"{side}_{col}"] = 0 if val in (None, "") else val
        if n % 25 == 0:
            print(f"  ...{n}/{len(todo)}")
    # write enrichment back onto the frame
    for col in ("home_shots", "away_shots", "home_sot", "away_sot",
                "home_corners", "away_corners"):
        df[col] = df.apply(
            lambda r: by_id.get(int(r["fixture_id"]), {}).get(col, r.get(col, ""))
            if pd.notna(r["fixture_id"]) else r.get(col, ""), axis=1)
    return df


def refit_and_baseline():
    """Refit the model and rewrite the walk-forward baseline via club validate.py."""
    print("\nRefitting model...")
    params = M.fit()
    M.save_params(params)
    print(f"  fitted {params['fitted_matches']} matches, {len(params['teams'])} teams")
    print("\nWalk-forward validation (fresh baseline):")
    import subprocess
    subprocess.run([sys.executable, str(HERE / "validate.py"), "--update-baseline"],
                   cwd=str(HERE), check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025],
                    help="API-Football seasons to pull (default: 2024 2025)")
    ap.add_argument("--competitions", nargs="+", help="subset by exact name")
    ap.add_argument("--cups", action="store_true", help="select all cup competitions")
    ap.add_argument("--uefa", action="store_true", help="select all European competitions")
    ap.add_argument("--merge", action="store_true",
                    help="append onto existing fixtures.csv (keeps league shots) "
                         "and reconcile team names to the league canon")
    ap.add_argument("--stats", action="store_true",
                    help="also fetch shot stats for finished matches (req-heavy)")
    ap.add_argument("--max-stats", type=int, default=400,
                    help="cap on per-match stats requests (default 400)")
    ap.add_argument("--pause", type=float, default=0.2,
                    help="seconds between uncached stats requests (default 0.2)")
    ap.add_argument("--api-key")
    ap.add_argument("--keep-backup", action="store_true",
                    help="don't overwrite an existing backup file")
    args = ap.parse_args()

    key = args.api_key or get_key("api-football", env="API_FOOTBALL_KEY")
    if not key:
        sys.exit("No API-Football key. Pass --api-key, set API_FOOTBALL_KEY, "
                 "or add 'api-football' to data/api_keys.json.")

    comps = select_competitions(args.competitions, args.cups, args.uefa)
    if not comps:
        sys.exit("No competitions selected.")
    merge = args.merge and FIXTURES.exists()

    # name reconciliation only in merge mode (map new rows onto the league canon)
    canon, league_teams = None, set()
    if merge:
        league_teams = set(pd.read_csv(FIXTURES, usecols=["home", "away"]).stack().unique())
        canon = make_canon(league_teams)

    # back up current fixtures before any change
    if FIXTURES.exists() and not (args.keep_backup and BACKUP.exists()):
        BACKUP.write_text(FIXTURES.read_text())
        print(f"Backed up current fixtures -> {BACKUP.name}")

    print(f"\nFetching {len(comps)} competition(s) x seasons {args.seasons}...")
    df = fetch_fixtures(args.seasons, key, comps, canon)
    if df.empty:
        sys.exit("No fixtures returned — check the key, plan limits, or season values. "
                 f"Backup left intact; restore with: cp {BACKUP} {FIXTURES}")

    if args.stats:
        df = fetch_stats(df, key, args.max_stats, args.pause)

    if merge:
        base = pd.read_csv(FIXTURES)
        new_teams = sorted((set(df["home"].dropna()) | set(df["away"].dropna())) - league_teams)
        linked = len((set(df["home"].dropna()) | set(df["away"].dropna())) & league_teams)
        merged = pd.concat([base, df], ignore_index=True).drop_duplicates(
            subset=["fixture_id"], keep="first")
        merged.to_csv(FIXTURES, index=False)
        print(f"\nMerged {len(df)} fetched rows -> {FIXTURES}  ({len(base)} -> {len(merged)} rows)")
        print(f"  {linked} teams linked to league data, {len(new_teams)} new (unlinked)")
        if new_teams:
            print("  new/unlinked teams (add to names.OVERRIDES if any are dupes):")
            print("   ", ", ".join(new_teams[:25]) + (" ..." if len(new_teams) > 25 else ""))
    else:
        df.to_csv(FIXTURES, index=False)
        played = df[df["home_goals"].notna() & df["away_goals"].notna()]
        print(f"\nWrote {len(df)} fixtures ({len(played)} played) -> {FIXTURES}")

    refit_and_baseline()
    print("\nDone. From now on use: bash club_soccer/update.sh")


if __name__ == "__main__":
    main()
