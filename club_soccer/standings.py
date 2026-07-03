#!/usr/bin/env python3
"""League tables, point-in-time.

Uniform tiebreak (points, goal difference, goals for) across every league —
per-league head-to-head tiebreak rules are NOT implemented; this is a
documented approximation, not a bug. League matches only (cup/Europe don't
count toward a domestic table).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import model as M

WIN_PTS, DRAW_PTS, LOSS_PTS = 3, 1, 0


def table_asof(competition: str, season: int, date: str,
               fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
    """League table for `competition`/`season` using only league matches
    played strictly before `date`.

    Returns columns: team, played, points, gf, ga, gd, position — sorted by
    points, then goal difference, then goals for (uniform rule; no per-
    league head-to-head tiebreak).
    """
    fx = fixtures if fixtures is not None else M.load_fixtures()
    played = M.played(fx)
    asof_ts = pd.Timestamp(date)
    sub = played[(played["competition"] == competition)
                & (played["season"] == season)
                & (played["type"] == "league")
                & (played["date"] < asof_ts)]

    stats: dict[str, dict] = {}

    def _row(team: str) -> dict:
        return stats.setdefault(team, {"team": team, "played": 0, "points": 0,
                                       "gf": 0, "ga": 0})

    for r in sub.itertuples(index=False):
        h, a = _row(r.home), _row(r.away)
        hg, ag = int(r.home_goals), int(r.away_goals)
        h["played"] += 1; a["played"] += 1
        h["gf"] += hg; h["ga"] += ag
        a["gf"] += ag; a["ga"] += hg
        if hg > ag:
            h["points"] += WIN_PTS; a["points"] += LOSS_PTS
        elif hg < ag:
            a["points"] += WIN_PTS; h["points"] += LOSS_PTS
        else:
            h["points"] += DRAW_PTS; a["points"] += DRAW_PTS

    df = pd.DataFrame(list(stats.values()),
                      columns=["team", "played", "points", "gf", "ga"])
    if df.empty:
        return pd.DataFrame(columns=["team", "played", "points", "gf", "ga", "gd", "position"])
    df["gd"] = df["gf"] - df["ga"]
    df = df.sort_values(["points", "gd", "gf"], ascending=False).reset_index(drop=True)
    df["position"] = df.index + 1
    return df[["team", "played", "points", "gf", "ga", "gd", "position"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("competition")
    ap.add_argument("--season", type=int, default=None,
                    help="season start year (default: current, inferred from --date)")
    ap.add_argument("--date", default=None, help="asof date, default today")
    args = ap.parse_args()
    from datetime import datetime, timezone
    date = args.date or str(datetime.now(timezone.utc).date())
    if args.season is None:
        d = pd.Timestamp(date)
        season = d.year if d.month >= 7 else d.year - 1
    else:
        season = args.season
    df = table_asof(args.competition, season, date)
    if df.empty:
        print(f"No league matches found for {args.competition} season {season} before {date}.")
        return
    print(f"{args.competition} {season} table as of {date}:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
