#!/usr/bin/env python3
"""Import CFBD /lines JSON files into data/closing_spreads.csv AND
data/closing_totals.csv.

Expects files matching data/lines_<year>*.json (raw responses from
https://api.collegefootballdata.com/lines?year=YYYY&seasonType=...).
Consensus = median closing spread / overUnder across providers. Spread sign
convention is auto-validated against actual margins (flipped if needed);
totals are validated by correlation with actual game totals. CFBD lines carry
no juice, so odds are left blank (-110 assumed downstream).

Usage: python3 import_cfbd_lines.py 2025
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SPREADS_CSV = os.path.join(HERE, "data", "closing_spreads.csv")
TOTALS_CSV = os.path.join(HERE, "data", "closing_totals.csv")
GAMES_CSV = os.path.join(HERE, "data", "games.csv")


def get(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def import_totals(paths, year):
    """Median closing overUnder per game -> data/closing_totals.csv."""
    rows = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for game in data:
            totals = [float(t) for ln in game.get("lines", [])
                      if (t := get(ln, "overUnder", "over_under")) is not None]
            if not totals:
                continue
            rows.append({
                "season": get(game, "season"),
                "week": get(game, "week"),
                "home_team": get(game, "homeTeam", "home_team"),
                "away_team": get(game, "awayTeam", "away_team"),
                "total_line": float(np.median(totals)),
                "over_odds": np.nan, "under_odds": np.nan,
                "n_books": len(totals),
            })
    new = pd.DataFrame(rows).drop_duplicates(
        subset=["season", "week", "home_team", "away_team"])
    if new.empty:
        print("no overUnder values found; closing_totals.csv unchanged")
        return
    games = pd.read_csv(GAMES_CSV)
    m = new.merge(games, on=["season", "week", "home_team", "away_team"], how="inner")
    m = m.dropna(subset=["home_points", "away_points"])
    if len(m) >= 50:
        total = m["home_points"] + m["away_points"]
        corr = float(np.corrcoef(total, m["total_line"])[0, 1])
        mae = (total - m["total_line"]).abs().mean()
        print(f"totals validation on {len(m)} matched games: "
              f"corr(total, line) = {corr:.2f}, closing total MAE = {mae:.2f}")
        if corr < 0.3:
            raise SystemExit("total-line correlation too weak — refusing to import")
    cols = ["season", "week", "home_team", "away_team", "total_line",
            "over_odds", "under_odds", "n_books"]
    if os.path.exists(TOTALS_CSV):
        old = pd.read_csv(TOTALS_CSV)
        old = old[old["season"] != int(year)]
        out = pd.concat([old[cols], new[cols]], ignore_index=True)
    else:
        out = new[cols]
    out = out.sort_values(["season", "week", "home_team", "away_team"]
                          ).reset_index(drop=True)
    out.to_csv(TOTALS_CSV, index=False)
    print(f"closing_totals.csv now has {len(out)} games, "
          f"seasons {int(out['season'].min())}-{int(out['season'].max())}")


def main(year):
    paths = sorted(glob.glob(os.path.join(HERE, "data", f"lines_{year}*.json")))
    if not paths:
        raise SystemExit(f"no data/lines_{year}*.json files found")
    rows = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for game in data:
            spreads = [float(s) for ln in game.get("lines", [])
                       if (s := get(ln, "spread")) is not None]
            if not spreads:
                continue
            rows.append({
                "season": get(game, "season"),
                "week": get(game, "week"),
                "home_team": get(game, "homeTeam", "home_team"),
                "away_team": get(game, "awayTeam", "away_team"),
                "home_line": float(np.median(spreads)),
                "home_odds": np.nan, "away_odds": np.nan,
                "n_books": len(spreads),
            })
    new = pd.DataFrame(rows).drop_duplicates(subset=["season", "week", "home_team", "away_team"])
    print(f"{len(new)} lined games parsed from {len(paths)} file(s)")

    # validate sign convention against actual margins
    games = pd.read_csv(GAMES_CSV)
    m = new.merge(games, on=["season", "week", "home_team", "away_team"], how="inner")
    m = m.dropna(subset=["home_points", "away_points"])
    margin = m["home_points"] - m["away_points"]
    corr = float(np.corrcoef(margin, -m["home_line"])[0, 1])
    if corr < 0:
        new["home_line"] = -new["home_line"]
        m["home_line"] = -m["home_line"]
        corr = -corr
        print("note: spread sign flipped to match home-handicap convention")
    print(f"validation on {len(m)} matched games: corr(margin, -line) = {corr:.2f}, "
          f"closing line MAE = {(margin + m['home_line']).abs().mean():.2f}")
    if corr < 0.5:
        raise SystemExit("correlation too weak — refusing to import, check the file")

    cols = ["season", "week", "home_team", "away_team", "home_line", "home_odds", "away_odds", "n_books"]
    if os.path.exists(SPREADS_CSV):
        old = pd.read_csv(SPREADS_CSV)
        old = old[old["season"] != int(year)]
        out = pd.concat([old[cols], new[cols]], ignore_index=True)
    else:
        out = new[cols]
    out = out.sort_values(["season", "week", "home_team", "away_team"]
                          ).reset_index(drop=True)
    out.to_csv(SPREADS_CSV, index=False)
    print(f"closing_spreads.csv now has {len(out)} games, "
          f"seasons {int(out['season'].min())}-{int(out['season'].max())}")

    import_totals(paths, year)


def fetch_lines(year: int) -> list[str]:
    """Pull CFBD /lines for a season into data/lines_<year>_<type>.json.

    Backfills the closing-line history (the totals file has a 2020-2024 hole
    the sportsdataverse mirror does not cover). Existing files are left alone
    unless they are missing or empty.
    """
    from . import fetch_cfbd

    key = fetch_cfbd._key()
    if not key:
        raise SystemExit("No CFBD key. Set CFBD_API_KEY or add "
                         "data/api_keys.json key 'collegefootballdata'.")
    written = []
    for season_type in ("regular", "postseason"):
        dest = os.path.join(HERE, "data", f"lines_{year}_{season_type}.json")
        if os.path.exists(dest) and os.path.getsize(dest) > 2:
            print(f"  lines_{year}_{season_type}.json present; keeping it")
            written.append(dest)
            continue
        data = fetch_cfbd.pull(
            f"/lines?year={year}&seasonType={season_type}", key)
        if not isinstance(data, list) or not data:
            print(f"  CFBD has no {season_type} lines for {year}")
            continue
        fetch_cfbd.save(data, dest, f"{year} {season_type} lines")
        written.append(dest)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("years", nargs="*", type=int, default=[2025],
                    help="seasons to import (default 2025)")
    ap.add_argument("--fetch", action="store_true",
                    help="pull CFBD /lines for each year first (backfill)")
    args = ap.parse_args()
    for target in (args.years or [2025]):
        if args.fetch:
            fetch_lines(target)
        main(target)
