#!/usr/bin/env python3
"""Blended FBS match predictor: Elo (elo.py) + offense/defense power ratings
(power.py), averaged 50/50. Predicts win probability, spread, and total.

Usage:
  python3 predictor.py "Ohio State" "Michigan"             # team 1 at home
  python3 predictor.py "Georgia" "Texas" --neutral
  python3 predictor.py ... --model elo|power|blend         # default blend
  python3 predictor.py --backtest [--since 2023]           # walk-forward eval
"""
import argparse
import math

import numpy as np
import pandas as pd

import elo as E
import power as P


def blend_predict(eparams, pparams, t1, t2, neutral=False, model="blend"):
    games, ratings, slope, sigma_e = eparams
    pe = E.predict(ratings, slope, sigma_e, t1, t2, neutral)
    pp = P.predict(pparams, t1, t2, neutral)
    if model == "elo":
        return {"p1": pe["p1"], "margin": pe["margin"], "total": pp["total"]}
    if model == "power":
        return {"p1": pp["p1"], "margin": pp["margin"], "total": pp["total"]}
    return {"p1": 0.5 * (pe["p1"] + pp["p1"]),
            "margin": 0.5 * (pe["margin"] + pp["margin"]),
            "total": pp["total"]}


def backtest(since=2023):
    games = E.load_games()
    carry, offs = E.season_priors()
    _, history = E.run_elo(games, record_pregame=True, carry=carry, prior_offsets=offs)
    diffs = np.array([h[2] for h in history])
    # spread map fitted only on pre-`since` data (no leakage)
    pre = games["season"] < since
    m_all = (games["home_points"] - games["away_points"]).values
    x, y = diffs[pre.values], m_all[pre.values]
    slope = float((x * y).sum() / (x * x).sum())

    ev = games[(games["season"] >= since) & (games["home"] != E.FCS) & (games["away"] != E.FCS)]
    print(f"Backtest: {len(ev)} FBS-vs-FBS games, seasons {since}-{int(games['season'].max())}")
    print("Power ratings refit before each week (walk-forward)...")

    rows = []
    pparams = None
    for (season, week, stype), wk in ev.groupby(["season", "week", "season_type"], sort=False):
        asof = wk["date"].min()
        try:
            pparams = P.fit(games, asof=asof)
        except ValueError:
            continue
        for r in wk.itertuples():
            if r.home not in pparams["teams"] or r.away not in pparams["teams"]:
                continue
            d = diffs[r.Index]
            p_elo = E.win_prob(d)
            m_elo = slope * d
            pp = P.predict(pparams, r.home, r.away, neutral=bool(r.neutral))
            actual_m = r.home_points - r.away_points
            actual_t = r.home_points + r.away_points
            rows.append({
                "p_elo": p_elo, "p_pow": pp["p1"], "m_elo": m_elo, "m_pow": pp["margin"],
                "t_pow": pp["total"], "actual_m": actual_m, "actual_t": actual_t,
            })
    df = pd.DataFrame(rows)
    df["p_blend"] = 0.5 * (df["p_elo"] + df["p_pow"])
    df["m_blend"] = 0.5 * (df["m_elo"] + df["m_pow"])
    res = (df["actual_m"] > 0).astype(float)

    print(f"\n{'model':<14s}{'accuracy':>9s}{'Brier':>8s}{'margin MAE':>12s}{'total MAE':>11s}")
    for name, pcol, mcol in [("Elo", "p_elo", "m_elo"), ("Power", "p_pow", "m_pow"),
                             ("50/50 blend", "p_blend", "m_blend")]:
        acc = ((df[pcol] > 0.5) == (res > 0.5)).mean()
        brier = ((df[pcol] - res) ** 2).mean()
        mae = (df[mcol] - df["actual_m"]).abs().mean()
        tmae = (df["t_pow"] - df["actual_t"]).abs().mean() if "pow" in mcol or "blend" in mcol else float("nan")
        t = f"{tmae:>11.2f}" if not math.isnan(tmae) else f"{'-':>11s}"
        print(f"{name:<14s}{acc:>9.1%}{brier:>8.4f}{mae:>12.2f}{t}")
    print(f"\n(binary Brier: 0.25 = coin flip, lower is better; "
          f"favourite-picks-all baseline acc = {max(res.mean(), 1 - res.mean()):.1%} for home side)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", nargs="*")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--model", choices=["elo", "power", "blend"], default="blend")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--since", type=int, default=2023)
    args = ap.parse_args()

    if args.backtest:
        backtest(args.since)
        return
    if len(args.teams) != 2:
        raise SystemExit(__doc__)
    t1, t2 = args.teams
    eparams = E.build()
    pparams = P.load_params()
    out = blend_predict(eparams, pparams, t1, t2, args.neutral, args.model)
    venue = "neutral site" if args.neutral else f"{t1} at home"
    print(f"{t1} vs {t2} ({venue}, model={args.model})")
    print(f"  P({t1} win) = {out['p1']:.1%}   P({t2} win) = {1 - out['p1']:.1%}")
    print(f"  Spread: {t1} {-out['margin']:+.1f}   Total: {out['total']:.1f}")


if __name__ == "__main__":
    main()
