#!/usr/bin/env python3
"""EPA-based power ratings for the NFL: opponent-adjusted offense/defense in
predicted-points-added per play, fitted by weighted ridge regression on
per-team-week EPA (nfl/data/epa_team_week.csv), then calibrated to points.

Why: final scores are noisy (turnovers, garbage time, field position luck);
per-play EPA measures how well a team actually moved the ball. Same ridge
structure as power.py (exponential time decay, fitted HFA, L2 shrinkage), but
the response is a composite EPA/play rate instead of points. Passing is
weighted ~1.6x rushing when combining the two into the composite rate — pass
EPA is the more predictive signal play-for-play.

This is a *candidate* for the margin/total blend, not a guaranteed member:
validate.py --tune-blend decides its weight from walk-forward MAE. Unlike
CFB (where EPA/PPA was rejected), NFL pbp-derived EPA is much cleaner and is
expected to earn a positive weight — but it must prove it.

Usage:
  python3 -m nfl.epa --fit                                  # refit, save data/epa_params.json
  python3 -m nfl.epa "Kansas City Chiefs" "Buffalo Bills"   # EPA-only prediction
  python3 -m nfl.epa --ratings                              # adjusted off/def EPA table
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
EPA_CSV = os.path.join(HERE, "data", "epa_team_week.csv")
PARAMS_JSON = os.path.join(HERE, "data", "epa_params.json")

HALF_LIFE_DAYS = 0.8 * 365
WINDOW_DAYS = 2.5 * 365
RIDGE = 10.0
PASS_WEIGHT = 1.6   # passing EPA is weighted more heavily than rushing EPA


def load_epa_games() -> pd.DataFrame:
    """Team-week EPA rows joined to games.csv for date/home-away/neutral."""
    epa = pd.read_csv(EPA_CSV)
    games = pd.read_csv(GAMES_CSV, parse_dates=["date"])
    g = games[["game_id", "date", "neutral", "home", "away"]]
    m = epa.merge(g, on="game_id", how="inner")
    m["is_home"] = m["team"] == m["home"]
    m["opponent"] = np.where(m["is_home"], m["away"], m["home"])

    off_vol = PASS_WEIGHT * m["dropbacks"] + m["off_carries"]
    off_epa = PASS_WEIGHT * m["off_pass_epa"] + m["off_rush_epa"]
    m["off_rate"] = np.where(off_vol > 0, off_epa / off_vol.replace(0, np.nan), 0.0)
    def_vol = PASS_WEIGHT * m["def_dropbacks_faced"] + m["def_carries_faced"]
    def_epa = PASS_WEIGHT * m["def_pass_epa_allowed"] + m["def_rush_epa_allowed"]
    m["def_rate_allowed"] = np.where(def_vol > 0, def_epa / def_vol.replace(0, np.nan), 0.0)
    m["off_vol"] = off_vol
    return m


def fit(asof=None, data: pd.DataFrame | None = None) -> dict:
    m = data if data is not None else load_epa_games()
    if asof is None:
        asof = m["date"].max() + pd.Timedelta(days=1)
    asof = pd.Timestamp(asof)
    g = m[(m["date"] < asof) & (m["date"] >= asof - pd.Timedelta(days=WINDOW_DAYS))]
    if len(g) < 300:
        raise ValueError(f"only {len(g)} EPA rows before {asof.date()}")

    teams = sorted(set(g["team"]) | set(g["opponent"]))
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    age = (asof - g["date"]).dt.days.values
    w = np.sqrt(0.5 ** (age / HALF_LIFE_DAYS)) * np.sqrt(np.clip(g["off_vol"].values, 1.0, None))
    mu = float(np.average(g["off_rate"], weights=w ** 2))

    k = len(g)
    A = np.zeros((k + 2 * n, 2 * n + 1))
    b = np.zeros(k + 2 * n)
    wfull = np.concatenate([w, np.full(2 * n, math.sqrt(RIDGE))])
    for i, r in enumerate(g.itertuples()):
        A[i, ti[r.team]] = 1.0
        A[i, n + ti[r.opponent]] = -1.0
        A[i, 2 * n] = 0.0 if r.neutral else (1.0 if r.is_home else -1.0)
        b[i] = r.off_rate - mu
    for j in range(2 * n):
        A[k + j, j] = 1.0
    x, *_ = np.linalg.lstsq(A * wfull[:, None], b * wfull, rcond=None)
    off, dfn, hfa = x[:n], x[n:2 * n], float(x[2 * n])

    # calibrate composite rate -> points/game
    games = pd.read_csv(GAMES_CSV, parse_dates=["date"])
    gg = games[(games["date"] < asof) & (games["date"] >= asof - pd.Timedelta(days=WINDOW_DAYS))]
    gg = gg[gg["home"].isin(ti) & gg["away"].isin(ti)]
    agew = (asof - gg["date"]).dt.days.values
    ww = 0.5 ** (agew / HALF_LIFE_DAYS)
    rate_h = mu + off[[ti[t] for t in gg["home"]]] - dfn[[ti[t] for t in gg["away"]]] \
        + np.where(gg["neutral"], 0.0, hfa)
    rate_a = mu + off[[ti[t] for t in gg["away"]]] - dfn[[ti[t] for t in gg["home"]]] \
        - np.where(gg["neutral"], 0.0, hfa)
    X = np.concatenate([rate_h, rate_a])
    Y = np.concatenate([gg["home_score"].values, gg["away_score"].values]).astype(float)
    W = np.concatenate([ww, ww])
    c1 = float(np.cov(X, Y, aweights=W)[0, 1] / np.cov(X, aweights=W))
    c0 = float(np.average(Y, weights=W) - c1 * np.average(X, weights=W))

    pred_m = (c0 + c1 * rate_h) - (c0 + c1 * rate_a)
    act_m = (gg["home_score"] - gg["away_score"]).values
    sigma = float(np.sqrt(np.average((act_m - pred_m) ** 2, weights=ww)))
    pred_t = (c0 + c1 * rate_h) + (c0 + c1 * rate_a)
    act_t = (gg["home_score"] + gg["away_score"]).values
    sigma_total = float(np.sqrt(np.average((act_t - pred_t) ** 2, weights=ww)))

    return {
        "asof": str(asof.date()), "mu": mu, "hfa": hfa, "c0": c0, "c1": c1,
        "sigma": sigma, "sigma_total": sigma_total, "pass_weight": PASS_WEIGHT,
        "teams": {t: {"off": float(off[ti[t]]), "def": float(dfn[ti[t]])} for t in teams},
    }


def predict(params: dict, team1: str, team2: str, neutral: bool = False) -> dict:
    for t in (team1, team2):
        if t not in params["teams"]:
            raise SystemExit(f"Unknown team: {t!r}")
    t1, t2 = params["teams"][team1], params["teams"][team2]
    hfa = 0.0 if neutral else params["hfa"]
    rate1 = params["mu"] + t1["off"] - t2["def"] + hfa
    rate2 = params["mu"] + t2["off"] - t1["def"] - hfa
    p1 = params["c0"] + params["c1"] * rate1
    p2 = params["c0"] + params["c1"] * rate2
    margin = p1 - p2
    z = margin / params["sigma"]
    pwin = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return {"pts1": p1, "pts2": p2, "margin": margin, "total": p1 + p2, "p1": pwin}


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
        params = fit()
        os.makedirs(os.path.dirname(PARAMS_JSON), exist_ok=True)
        with open(PARAMS_JSON, "w") as f:
            json.dump(params, f, indent=1)
        print(f"fitted {len(params['teams'])} teams as of {params['asof']}: "
              f"mu={params['mu']:.4f} epa/play, hfa={params['hfa']:.4f}, "
              f"pts = {params['c0']:.1f} + {params['c1']:.1f}*rate, "
              f"sigma={params['sigma']:.1f}/{params['sigma_total']:.1f}")
        return 0
    params = load_params()
    if args.ratings:
        t = [(k, v["off"], v["def"], v["off"] + v["def"])
             for k, v in params["teams"].items()]
        print(f"{'team':<25s} {'off':>7s} {'def':>7s} {'net':>7s}  (adj EPA/play)")
        for name, o, d, net in sorted(t, key=lambda r: -r[3]):
            print(f"{name:<25s} {o:>+7.3f} {d:>+7.3f} {net:>+7.3f}")
        return 0
    if len(args.teams) != 2:
        raise SystemExit(__doc__)
    t1, t2 = args.teams
    p = predict(params, t1, t2, args.neutral)
    venue = "neutral site" if args.neutral else f"{t1} at home"
    print(f"{t1} vs {t2} ({venue}, EPA model)")
    print(f"  Expected score: {t1} {p['pts1']:.1f} - {p['pts2']:.1f} {t2}")
    print(f"  Margin {p['margin']:+.1f}, total {p['total']:.1f}, P({t1} win) = {p['p1']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
