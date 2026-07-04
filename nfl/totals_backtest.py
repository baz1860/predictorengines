#!/usr/bin/env python3
"""Over/under backtest vs real closing totals (games.csv `total_line`,
`over_odds`/`under_odds` — no separate file needed).

Same walk-forward discipline as ats_backtest.py. Bets over when the model's
predicted total exceeds the closing total by >= threshold, under when below.

Usage:
  python3 -m nfl.totals_backtest                      # 2015-2025
  python3 -m nfl.totals_backtest --since 2021 --until 2025
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import validate as V
from .edge import american_to_decimal


def settle(b: pd.DataFrame) -> tuple[int, int, int, np.ndarray]:
    over_side = (b["edge_pts"] > 0).values
    diff = (b["total"] - b["total_line"]).values
    push = diff == 0
    won = np.where(over_side, diff > 0, diff < 0) & ~push
    odds_am = np.where(over_side, b["over_odds"].values, b["under_odds"].values)
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
    df["model_total"] = V.blend_total(df, weights)

    raw = games.set_index(games.index)[["over_odds", "under_odds"]]
    g = df.join(raw)
    g = g[g["total_line"].notna()]
    g["edge_pts"] = g["model_total"] - g["total_line"]

    closing_mae = (g["total"] - g["total_line"]).abs().mean()
    model_mae = (g["total"] - g["model_total"]).abs().mean()
    print(f"{len(g)} lined games, seasons {args.since}-{args.until} "
         f"(closing total MAE {closing_mae:.2f}, model total MAE {model_mae:.2f})")

    print(f"\n{'edge>=':>7s} {'bets':>6s} {'W-L-P':>14s} {'win%':>7s} {'ROI':>7s} {'over%':>7s}")
    for thr in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        b = g[g["edge_pts"].abs() >= thr]
        if len(b) < 20:
            continue
        w, l, p, pnl = settle(b)
        print(f"{thr:>6.1f} {len(b):>6d} {f'{w}-{l}-{p}':>14s} "
             f"{w / max(w + l, 1):>6.1%} {pnl.mean():>+7.1%} {(b['edge_pts'] > 0).mean():>6.1%}")

    print("\nper season (edge >= 3 pts):")
    for season, s in g.groupby("season"):
        b = s[s["edge_pts"].abs() >= 3.0]
        if b.empty:
            continue
        w, l, p, pnl = settle(b)
        print(f"  {season}: {w}-{l}-{p}  win {w / max(w + l, 1):.1%}  "
             f"ROI {pnl.mean():+.1%}  ({len(b)} bets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
