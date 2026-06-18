#!/usr/bin/env python3
"""Against-the-spread backtest vs real consensus closing lines.

Walk-forward: Elo updated game by game, power ratings refit before each week,
Elo spread map fitted only on pre-evaluation data. For each lined game the
blend's predicted margin is compared to the consensus closing spread; when the
model disagrees with the market by at least `threshold` points it bets that
side at the median closing juice (fallback -110). Pushes returned.

Closing lines: data/closing_spreads.csv (from fetch_data.py; mirror covers
2006-2019). Break-even at -110 juice = 52.38%.

Usage:
  python3 ats_backtest.py                      # seasons 2015-2019
  python3 ats_backtest.py --since 2010 --until 2019
"""
import argparse
import os

import numpy as np
import pandas as pd

from . import elo as E
from . import power as P

HERE = os.path.dirname(os.path.abspath(__file__))
SPREADS_CSV = os.path.join(HERE, "data", "closing_spreads.csv")


def american_to_decimal(a):
    """American -> decimal; invalid values (|a| < 100, data glitches) fall back to -110."""
    if pd.isna(a) or abs(float(a)) < 100.0:
        return 1.0 + 100.0 / 110.0
    a = float(a)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def settle(b):
    """ATS settlement for the model-preferred side. Returns (w, l, p, pnl_per_unit)."""
    home_side = (b["edge_pts"] > 0).values
    cover = (b["margin"] + b["home_line"]).values
    push = cover == 0
    won = np.where(home_side, cover > 0, cover < 0) & ~push
    odds = np.where(home_side, b["home_odds"].values, b["away_odds"].values)
    dec = np.array([american_to_decimal(a) for a in odds])
    pnl = np.where(push, 0.0, np.where(won, dec - 1.0, -1.0))
    return int(won.sum()), int((~won & ~push).sum()), int(push.sum()), pnl


def model_margins(games, since, until):
    """Walk-forward blend margins for eval games. Returns df indexed like games."""
    carry, offs = E.season_priors()
    _, history = E.run_elo(games, record_pregame=True, carry=carry, prior_offsets=offs)
    diffs = np.array([h[2] for h in history])
    pre = (games["season"] < since).values
    m_all = (games["home_points"] - games["away_points"]).values
    slope = float((diffs[pre] * m_all[pre]).sum() / (diffs[pre] ** 2).sum())

    ev = games[(games["season"] >= since) & (games["season"] <= until)
               & (games["home"] != E.FCS) & (games["away"] != E.FCS)]
    out = {}
    for (_, _, _), wk in ev.groupby(["season", "week", "season_type"], sort=False):
        pparams = P.fit(games, asof=wk["date"].min())
        for r in wk.itertuples():
            if r.home not in pparams["teams"] or r.away not in pparams["teams"]:
                continue
            pp = P.predict(pparams, r.home, r.away, neutral=bool(r.neutral))
            out[r.Index] = 0.5 * (slope * diffs[r.Index] + pp["margin"])
    return pd.Series(out, name="model_margin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2015)
    ap.add_argument("--until", type=int, default=2019)
    args = ap.parse_args()

    games = E.load_games()
    lines = pd.read_csv(SPREADS_CSV)
    lines = lines[(lines["season"] >= args.since) & (lines["season"] <= args.until)]

    mm = model_margins(games, args.since, args.until)
    g = games.loc[mm.index].copy()
    g["model_margin"] = mm
    g = g.merge(lines, left_on=["season", "week", "home_team", "away_team"],
                right_on=["season", "week", "home_team", "away_team"], how="inner")
    g["margin"] = g["home_points"] - g["away_points"]
    g["edge_pts"] = g["model_margin"] + g["home_line"]  # >0: model likes home vs line
    print(f"{len(g)} lined games, seasons {args.since}-{args.until} "
          f"(closing spread MAE {(g['margin'] + g['home_line']).abs().mean():.2f}, "
          f"model margin MAE {(g['margin'] - g['model_margin']).abs().mean():.2f})")

    print(f"\n{'edge>=':>7s} {'bets':>6s} {'W-L-P':>14s} {'cover%':>7s} {'ROI':>7s}")
    for thr in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        b = g[g["edge_pts"].abs() >= thr]
        if b.empty:
            continue
        w, l, p, pnl = settle(b)
        print(f"{thr:>6.1f} {len(b):>6d} {f'{w}-{l}-{p}':>14s} "
              f"{w / (w + l):>6.1%} {pnl.mean():>+7.1%}")

    print("\nper season (edge >= 2 pts):")
    for season, s in g.groupby("season"):
        b = s[s["edge_pts"].abs() >= 2.0]
        w, l, p, pnl = settle(b)
        print(f"  {season}: {w}-{l}-{p}  cover {w / max(w + l, 1):.1%}  ROI {pnl.mean():+.1%}  ({len(b)} bets)")


if __name__ == "__main__":
    main()
