#!/usr/bin/env python3
"""Against-the-spread backtest vs real closing lines (games.csv carries them
directly — no separate closing-lines file needed, unlike cfb).

Walk-forward (see validate.py's docstring for the discipline): for each lined
game, the blend's predicted margin is compared to the actual closing
`spread_line`. When the model disagrees with the market by at least
`threshold` points, it bets that side at the real closing juice (games.csv
`home_spread_odds`/`away_spread_odds`; falls back to -110 if missing).
Pushes are handled explicitly (3 and 7 are common pushes).

Honesty clause: NFL closing lines are the sharpest in sports. Report this
number straight — 49-52% cover is the expected, HONEST outcome; break-even
at -110 is 52.4%.

Usage:
  python3 -m nfl.ats_backtest                      # seasons 2015-2025
  python3 -m nfl.ats_backtest --since 2021 --until 2025
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import validate as V
from .edge import american_to_decimal


def settle(b: pd.DataFrame) -> tuple[int, int, int, np.ndarray]:
    """ATS settlement for the model-preferred side. Returns (w, l, p, pnl_per_unit)."""
    home_side = (b["edge_pts"] > 0).values
    cover = (b["margin"] - b["spread_line"]).values
    push = cover == 0
    won = np.where(home_side, cover > 0, cover < 0) & ~push
    odds_am = np.where(home_side, b["home_spread_odds"].values, b["away_spread_odds"].values)
    dec = np.array([american_to_decimal(a) or (1.0 + 100.0 / 110.0) for a in odds_am])
    pnl = np.where(push, 0.0, np.where(won, dec - 1.0, -1.0))
    return int(won.sum()), int((~won & ~push).sum()), int(push.sum()), pnl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2015)
    ap.add_argument("--until", type=int, default=2025)
    args = ap.parse_args()

    games = V.load_games()
    df = V.walk_forward(games, since=args.since, until=args.until, quiet=True)
    weights = V._load_blend_weight()
    df["model_margin"] = V.blend_margin(df, weights)

    raw = games.set_index(games.index)[["home_spread_odds", "away_spread_odds"]]
    g = df.join(raw)
    g = g[g["spread_line"].notna()]
    g["edge_pts"] = g["model_margin"] - g["spread_line"]

    closing_mae = (g["margin"] - g["spread_line"]).abs().mean()
    model_mae = (g["margin"] - g["model_margin"]).abs().mean()
    print(f"{len(g)} lined games, seasons {args.since}-{args.until} "
         f"(closing spread MAE {closing_mae:.2f}, model margin MAE {model_mae:.2f})")

    print(f"\n{'edge>=':>7s} {'bets':>6s} {'W-L-P':>14s} {'cover%':>7s} {'ROI':>7s}")
    for thr in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        b = g[g["edge_pts"].abs() >= thr]
        if len(b) < 20:
            continue
        w, l, p, pnl = settle(b)
        print(f"{thr:>6.1f} {len(b):>6d} {f'{w}-{l}-{p}':>14s} "
             f"{w / max(w + l, 1):>6.1%} {pnl.mean():>+7.1%}")

    print("\nper season (edge >= 2 pts):")
    for season, s in g.groupby("season"):
        b = s[s["edge_pts"].abs() >= 2.0]
        if b.empty:
            continue
        w, l, p, pnl = settle(b)
        print(f"  {season}: {w}-{l}-{p}  cover {w / max(w + l, 1):.1%}  "
             f"ROI {pnl.mean():+.1%}  ({len(b)} bets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
