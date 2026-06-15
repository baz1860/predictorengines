#!/usr/bin/env python3
"""Over/under backtest vs real consensus closing totals.

Walk-forward: power ratings refit before each week; the model's predicted
total is compared to the consensus closing O/U (data/closing_totals.csv —
mirror 2006-2019 with juice, CFBD 2025 without, -110 assumed). Bets over when
model total exceeds the line by >= threshold, under when below by the same.

Usage:
  python3 totals_backtest.py                       # 2015-2019
  python3 totals_backtest.py --since 2025 --until 2025
"""
import argparse
import os

import numpy as np
import pandas as pd

import elo as E
import power as P
from ats_backtest import american_to_decimal

HERE = os.path.dirname(os.path.abspath(__file__))
TOTALS_CSV = os.path.join(HERE, "data", "closing_totals.csv")


def model_totals(games, since, until):
    ev = games[(games["season"] >= since) & (games["season"] <= until)
               & (games["home"] != E.FCS) & (games["away"] != E.FCS)]
    out = {}
    for (_, _, _), wk in ev.groupby(["season", "week", "season_type"], sort=False):
        try:
            pparams = P.fit(games, asof=wk["date"].min())
        except ValueError:
            continue
        for r in wk.itertuples():
            if r.home not in pparams["teams"] or r.away not in pparams["teams"]:
                continue
            pp = P.predict(pparams, r.home, r.away, neutral=bool(r.neutral))
            out[r.Index] = pp["total"]
    return pd.Series(out, name="model_total")


def settle(b):
    over_side = (b["edge_pts"] > 0).values
    diff = (b["total"] - b["total_line"]).values
    push = diff == 0
    won = np.where(over_side, diff > 0, diff < 0) & ~push
    odds = np.where(over_side, b["over_odds"].values, b["under_odds"].values)
    dec = np.array([american_to_decimal(a) for a in odds])
    pnl = np.where(push, 0.0, np.where(won, dec - 1.0, -1.0))
    return int(won.sum()), int((~won & ~push).sum()), int(push.sum()), pnl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2015)
    ap.add_argument("--until", type=int, default=2019)
    args = ap.parse_args()

    games = E.load_games()
    lines = pd.read_csv(TOTALS_CSV)
    lines = lines[(lines["season"] >= args.since) & (lines["season"] <= args.until)]

    mt = model_totals(games, args.since, args.until)
    g = games.loc[mt.index].copy()
    g["model_total"] = mt
    g = g.merge(lines, on=["season", "week", "home_team", "away_team"], how="inner")
    g["total"] = g["home_points"] + g["away_points"]
    g["edge_pts"] = g["model_total"] - g["total_line"]  # >0: model says over
    print(f"{len(g)} lined games, seasons {args.since}-{args.until} "
          f"(closing total MAE {(g['total'] - g['total_line']).abs().mean():.2f}, "
          f"model total MAE {(g['total'] - g['model_total']).abs().mean():.2f})")

    print(f"\n{'edge>=':>7s} {'bets':>6s} {'W-L-P':>14s} {'win%':>7s} {'ROI':>7s} {'over%':>7s}")
    for thr in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        b = g[g["edge_pts"].abs() >= thr]
        if len(b) < 30:
            continue
        w, l, p, pnl = settle(b)
        print(f"{thr:>6.1f} {len(b):>6d} {f'{w}-{l}-{p}':>14s} "
              f"{w / (w + l):>6.1%} {pnl.mean():>+7.1%} {(b['edge_pts'] > 0).mean():>6.1%}")

    print("\nper season (edge >= 3 pts):")
    for season, s in g.groupby("season"):
        b = s[s["edge_pts"].abs() >= 3.0]
        if b.empty:
            continue
        w, l, p, pnl = settle(b)
        print(f"  {season}: {w}-{l}-{p}  win {w / max(w + l, 1):.1%}  ROI {pnl.mean():+.1%}  ({len(b)} bets)")


if __name__ == "__main__":
    main()
