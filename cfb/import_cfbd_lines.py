#!/usr/bin/env python3
"""Import CFBD /lines JSON files into data/closing_spreads.csv.

Expects files matching data/lines_<year>*.json (raw responses from
https://api.collegefootballdata.com/lines?year=YYYY&seasonType=...).
Consensus = median closing spread across providers. Spread sign convention is
auto-validated against actual margins (flipped if needed). CFBD lines carry no
spread juice, so home/away odds are left blank (-110 assumed downstream).

Usage: python3 import_cfbd_lines.py 2025
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SPREADS_CSV = os.path.join(HERE, "data", "closing_spreads.csv")
GAMES_CSV = os.path.join(HERE, "data", "games.csv")


def get(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


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
    out.to_csv(SPREADS_CSV, index=False)
    print(f"closing_spreads.csv now has {len(out)} games, "
          f"seasons {int(out['season'].min())}-{int(out['season'].max())}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2025)
