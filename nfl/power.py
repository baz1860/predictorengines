#!/usr/bin/env python3
"""Offense/defense power ratings for the NFL (adapted from cfb/power.py).

Per-team offense and defense ratings in points, fitted by weighted ridge
regression on points scored/allowed: exponential time decay (0.8-season
half-life, 2.5-season window — NFL rosters churn faster than CFB and a
32-team/17-game league needs a tighter window than FBS's 130+ teams),
fitted home-field advantage, L2 shrinkage. Predicts expected points for
each side, hence margin AND total.

Usage:
  python3 -m nfl.power --fit                                  # refit, save data/power_params.json
  python3 -m nfl.power "Kansas City Chiefs" "Buffalo Bills"    # team 1 at home
  python3 -m nfl.power --ratings                              # offense/defense table
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES_CSV = os.path.join(HERE, "data", "games.csv")
PARAMS_JSON = os.path.join(HERE, "data", "power_params.json")

HALF_LIFE_DAYS = 0.8 * 365
WINDOW_DAYS = 2.5 * 365
RIDGE = 8.0

# Re-evaluated from cfb/power.py's TOTAL_SHRINK: out-of-sample the additive
# points model over-disperses the total; shrinking it toward the window mean
# (margin untouched) lowers held-out total MAE. Fit on 2003-2014 in
# validate.py; this default (0.90) sits in the plan's expected 0.85-0.95
# band and is overridden by data/power_params.json's fitted value once
# --fit has run.
TOTAL_SHRINK_DEFAULT = 0.90


def load_games(path: str = GAMES_CSV) -> pd.DataFrame:
    g = pd.read_csv(path, parse_dates=["date"])
    return g


def fit(games: pd.DataFrame, asof=None, total_shrink: float = TOTAL_SHRINK_DEFAULT) -> dict:
    if asof is None:
        asof = games["date"].max() + pd.Timedelta(days=1)
    asof = pd.Timestamp(asof)
    g = games[(games["date"] < asof) & (games["date"] >= asof - pd.Timedelta(days=WINDOW_DAYS))]
    if len(g) < 150:
        raise ValueError(f"only {len(g)} games before {asof.date()}")

    teams = sorted(set(g["home"]) | set(g["away"]))
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    age = (asof - g["date"]).dt.days.values
    w = np.sqrt(0.5 ** (age / HALF_LIFE_DAYS))

    mu = float(np.average(np.r_[g["home_score"], g["away_score"]], weights=np.r_[w, w] ** 2))

    rows, cols, vals, y, wts = [], [], [], [], []
    k = 0
    for r, wi in zip(g.itertuples(), w):
        h, a = ti[r.home], ti[r.away]
        hfa_on = 0.0 if r.neutral else 1.0
        rows += [k, k, k]; cols += [h, n + a, 2 * n]; vals += [1.0, -1.0, hfa_on]
        y.append(r.home_score - mu); wts.append(wi); k += 1
        rows += [k, k]; cols += [a, n + h]; vals += [1.0, -1.0]
        y.append(r.away_score - mu); wts.append(wi); k += 1

    A = np.zeros((k + 2 * n, 2 * n + 1))
    for rr, cc, vv in zip(rows, cols, vals):
        A[rr, cc] = vv
    b = np.array(y + [0.0] * (2 * n), dtype=float)
    wfull = np.array(wts + [math.sqrt(RIDGE)] * (2 * n), dtype=float)
    for j in range(2 * n):
        A[k + j, j] = 1.0
    x, *_ = np.linalg.lstsq(A * wfull[:, None], b * wfull, rcond=None)

    off, dfn, hfa = x[:n], x[n:2 * n], float(x[2 * n])

    pred_m, act_m, pred_t, act_t, wm = [], [], [], [], []
    for r, wi in zip(g.itertuples(), w):
        h, a = ti[r.home], ti[r.away]
        hp = mu + off[h] - dfn[a] + (0.0 if r.neutral else hfa)
        ap = mu + off[a] - dfn[h]
        pred_m.append(hp - ap)
        act_m.append(r.home_score - r.away_score)
        pred_t.append(hp + ap)
        act_t.append(r.home_score + r.away_score)
        wm.append(wi ** 2)
    res = np.array(act_m) - np.array(pred_m)
    sigma = float(np.sqrt(np.average(res ** 2, weights=wm)))
    res_t = np.array(act_t) - np.array(pred_t)
    sigma_total = float(np.sqrt(np.average(res_t ** 2, weights=wm)))
    recent = (np.array([(asof - d).days for d in g["date"]]) <= 365)
    total_bias = float(np.average(res_t[recent], weights=np.array(wm)[recent])) if recent.sum() > 100 else 0.0
    total_mean = float(np.average(np.array(act_t), weights=wm))

    return {
        "asof": str(asof.date()), "mu": mu, "hfa": hfa, "sigma": sigma, "sigma_total": sigma_total,
        "total_bias": total_bias, "total_mean": total_mean, "total_shrink": total_shrink,
        "teams": {t: {"off": float(off[ti[t]]), "def": float(dfn[ti[t]])} for t in teams},
    }


def predict(params: dict, team1: str, team2: str, neutral: bool = False) -> dict:
    for t in (team1, team2):
        if t not in params["teams"]:
            raise SystemExit(f"Unknown team: {t!r}")
    o1, d1 = params["teams"][team1]["off"], params["teams"][team1]["def"]
    o2, d2 = params["teams"][team2]["off"], params["teams"][team2]["def"]
    hfa = 0.0 if neutral else params["hfa"]
    tb = params.get("total_bias", 0.0) / 2.0
    p1 = params["mu"] + o1 - d2 + hfa + tb
    p2 = params["mu"] + o2 - d1 + tb
    margin = p1 - p2
    z = margin / params["sigma"]
    pwin = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    total = p1 + p2
    tmean = params.get("total_mean")
    k = float(params.get("total_shrink", 1.0))
    if tmean is not None and k != 1.0:
        total = tmean + k * (total - tmean)
        p1, p2 = (total + margin) / 2.0, (total - margin) / 2.0
    return {"pts1": p1, "pts2": p2, "margin": margin, "total": total, "p1": pwin}


def load_params(path: str = PARAMS_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", nargs="*")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--ratings", action="store_true")
    args = ap.parse_args()

    if args.fit:
        params = fit(load_games())
        os.makedirs(os.path.dirname(PARAMS_JSON), exist_ok=True)
        with open(PARAMS_JSON, "w") as f:
            json.dump(params, f, indent=1)
        print(f"fitted {len(params['teams'])} teams as of {params['asof']}: "
              f"mu={params['mu']:.1f} hfa={params['hfa']:.2f} sigma={params['sigma']:.1f}/{params['sigma_total']:.1f}")
        return 0

    params = load_params()
    if args.ratings:
        t = [(k, v["off"], v["def"], v["off"] + v["def"]) for k, v in params["teams"].items()]
        print(f"{'team':<25s} {'off':>6s} {'def':>6s} {'net':>6s}")
        for name, o, d, net in sorted(t, key=lambda r: -r[3]):
            print(f"{name:<25s} {o:>+6.1f} {d:>+6.1f} {net:>+6.1f}")
        return 0
    if len(args.teams) != 2:
        raise SystemExit(__doc__)
    t1, t2 = args.teams
    p = predict(params, t1, t2, args.neutral)
    venue = "neutral site" if args.neutral else f"{t1} at home"
    print(f"{t1} vs {t2} ({venue})")
    print(f"  Expected score: {t1} {p['pts1']:.1f} - {p['pts2']:.1f} {t2}")
    print(f"  Margin {p['margin']:+.1f}, total {p['total']:.1f}, P({t1} win) = {p['p1']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
