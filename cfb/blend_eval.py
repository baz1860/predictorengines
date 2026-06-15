#!/usr/bin/env python3
"""Walk-forward comparison of blend combinations (Elo, points-power, EPA-power).

Weights are chosen by looking at 2023-24; 2025 is the untouched validation
season. Prints accuracy / Brier / margin MAE per candidate blend and season
group.

Usage: python3 blend_eval.py [--since 2023 --until 2025]
"""
import argparse

import numpy as np
import pandas as pd

import elo as E
import epa as X
import power as P


def collect(since, until):
    games = E.load_games()
    carry, offs = E.season_priors()
    _, history = E.run_elo(games, record_pregame=True, carry=carry, prior_offsets=offs)
    diffs = np.array([h[2] for h in history])
    pre = (games["season"] < since).values
    m_all = (games["home_points"] - games["away_points"]).values
    slope = float((diffs[pre] * m_all[pre]).sum() / (diffs[pre] ** 2).sum())

    data = X.load_ppa()
    ev = games[(games["season"] >= since) & (games["season"] <= until)
               & (games["home"] != E.FCS) & (games["away"] != E.FCS)]
    rows = []
    for (_, _, _), wk in ev.groupby(["season", "week", "season_type"], sort=False):
        asof = wk["date"].min()
        try:
            pp = P.fit(games, asof=asof)
            xp = X.fit(asof=asof, data=data)
        except ValueError:
            continue
        for r in wk.itertuples():
            if any(t not in pp["teams"] or t not in xp["teams"] for t in (r.home, r.away)):
                continue
            a = P.predict(pp, r.home, r.away, neutral=bool(r.neutral))
            b = X.predict(xp, r.home, r.away, neutral=bool(r.neutral))
            rows.append({
                "season": r.season,
                "p_elo": E.win_prob(diffs[r.Index]), "m_elo": slope * diffs[r.Index],
                "p_pow": a["p1"], "m_pow": a["margin"], "t_pow": a["total"],
                "p_epa": b["p1"], "m_epa": b["margin"], "t_epa": b["total"],
                "margin": r.home_points - r.away_points,
                "total": r.home_points + r.away_points,
            })
    return pd.DataFrame(rows)


CANDIDATES = {
    "elo only":        {"p_elo": 1.0},
    "power only":      {"p_pow": 1.0},
    "epa only":        {"p_epa": 1.0},
    "elo+power (old)": {"p_elo": .5, "p_pow": .5},
    "elo+epa":         {"p_elo": .5, "p_epa": .5},
    "power+epa":       {"p_pow": .5, "p_epa": .5},
    "equal thirds":    {"p_elo": 1 / 3, "p_pow": 1 / 3, "p_epa": 1 / 3},
    "elo+epa heavy":   {"p_elo": .4, "p_pow": .2, "p_epa": .4},
}


def score(df, weights):
    p = sum(w * df[c] for c, w in weights.items())
    m = sum(w * df[c.replace("p_", "m_")] for c, w in weights.items())
    res = (df["margin"] > 0).astype(float)
    return {"acc": ((p > .5) == (res > .5)).mean(), "brier": ((p - res) ** 2).mean(),
            "mae": (m - df["margin"]).abs().mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2023)
    ap.add_argument("--until", type=int, default=2025)
    args = ap.parse_args()
    df = collect(args.since, args.until)
    for label, sub in [("2023-24 (selection)", df[df.season < 2025]),
                       ("2025 (validation)", df[df.season == 2025])]:
        if sub.empty:
            continue
        print(f"\n{label}: {len(sub)} games")
        print(f"{'blend':<18s}{'acc':>7s}{'brier':>8s}{'m MAE':>7s}")
        for name, w in CANDIDATES.items():
            s = score(sub, w)
            print(f"{name:<18s}{s['acc']:>7.1%}{s['brier']:>8.4f}{s['mae']:>7.2f}")
        tm = {"t_pow": (sub.t_pow - sub.total).abs().mean(),
              "t_epa": (sub.t_epa - sub.total).abs().mean(),
              "t_avg": ((sub.t_pow + sub.t_epa) / 2 - sub.total).abs().mean()}
        print("totals MAE: " + "  ".join(f"{k}={v:.2f}" for k, v in tm.items()))


if __name__ == "__main__":
    main()
