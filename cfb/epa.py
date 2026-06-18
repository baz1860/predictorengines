#!/usr/bin/env python3
"""EPA-based power ratings (SP+-style): opponent-adjusted offense/defense in
predicted-points-added per play, fitted by weighted ridge regression on CFBD
per-game PPA (data/cfbd/ppa_games_*.json), then calibrated to points.

Why: final scores are noisy (turnovers, garbage time, field position luck);
per-play EPA measures how well a team actually moved the ball. Same structure
as power.py — exponential time decay, fitted HFA, L2 shrinkage — but the
response is PPA/play instead of points.

Usage:
  python3 epa.py --fit                      # refit, save data/epa_params.json
  python3 epa.py "Ohio State" "Michigan"    # EPA-only prediction
  python3 epa.py --ratings                  # adjusted off/def PPA table
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from .elo import load_games, FCS

HERE = os.path.dirname(os.path.abspath(__file__))
CFBD_DIR = os.path.join(HERE, "data", "cfbd")
PARAMS_JSON = os.path.join(HERE, "data", "epa_params.json")

HALF_LIFE_DAYS = 1.5 * 365
WINDOW_DAYS = 4 * 365
RIDGE = 8.0


def _get(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


PPA_FIELDS = ("overall", "passing", "rushing", "firstDown", "secondDown", "thirdDown")


def load_ppa():
    """Per-game team PPA joined to games.csv -> one row per (game, side)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(CFBD_DIR, "ppa_games_*.json"))):
        try:
            data = json.load(open(path))
        except json.JSONDecodeError:
            continue
        for r in data:
            off = _get(r, "offense") or {}
            dfn = _get(r, "defense") or {}
            if _get(off, "overall") is None:
                continue
            row = {
                "game_id": _get(r, "gameId", "game_id"),
                "team": _get(r, "team"),
                "opponent": _get(r, "opponent"),
            }
            for field in PPA_FIELDS:
                row[f"off_{field}"] = float(_get(off, field) or 0.0)
                row[f"def_{field}"] = float(_get(dfn, field) or 0.0)
            row["off_ppa"] = row["off_overall"]
            row["def_ppa"] = row["def_overall"]
            rows.append(row)
    ppa = pd.DataFrame(rows).drop_duplicates(subset=["game_id", "team"])
    games = load_games()
    g = games.merge(ppa, on="game_id", how="inner")
    # keep rows where the PPA side is the FBS team name (home or away)
    g["side_home"] = g["team"] == g["home_team"]
    g = g[(g["team"] == g["home_team"]) | (g["team"] == g["away_team"])]
    g["off_team"] = np.where(g["side_home"], g["home"], g["away"])
    g["def_team"] = np.where(g["side_home"], g["away"], g["home"])
    return g, games


def fit(asof=None, data=None, field: str = "overall"):
    g, games = data if data is not None else load_ppa()
    if field not in PPA_FIELDS:
        raise ValueError(f"unknown PPA field {field!r}; use {', '.join(PPA_FIELDS)}")
    off_col = f"off_{field}"
    if asof is None:
        asof = g["date"].max() + pd.Timedelta(days=1)
    asof = pd.Timestamp(asof)
    g = g[(g["date"] < asof) & (g["date"] >= asof - pd.Timedelta(days=WINDOW_DAYS))]
    if len(g) < 400:
        raise ValueError(f"only {len(g)} PPA rows before {asof.date()}")

    teams = sorted(set(g["off_team"]) | set(g["def_team"]))
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    age = (asof - g["date"]).dt.days.values
    w = np.sqrt(0.5 ** (age / HALF_LIFE_DAYS))
    mu = float(np.average(g[off_col], weights=w ** 2))

    k = len(g)
    A = np.zeros((k + 2 * n, 2 * n + 1))
    b = np.zeros(k + 2 * n)
    wfull = np.concatenate([w, np.full(2 * n, math.sqrt(RIDGE))])
    for i, r in enumerate(g.itertuples()):
        A[i, ti[r.off_team]] = 1.0          # offense rating
        A[i, n + ti[r.def_team]] = -1.0     # opponent defense rating
        A[i, 2 * n] = 0.0 if r.neutral else (1.0 if r.side_home else -1.0)
        b[i] = getattr(r, off_col) - mu
    for j in range(2 * n):
        A[k + j, j] = 1.0
    x, *_ = np.linalg.lstsq(A * wfull[:, None], b * wfull, rcond=None)
    off, dfn, hfa = x[:n], x[n:2 * n], float(x[2 * n])

    # calibrate PPA rate -> points per side on the same window
    gg = games[(games["date"] < asof) & (games["date"] >= asof - pd.Timedelta(days=WINDOW_DAYS))]
    gg = gg[gg["home"].isin(ti) & gg["away"].isin(ti)]
    agew = (asof - gg["date"]).dt.days.values
    ww = 0.5 ** (agew / HALF_LIFE_DAYS)
    rate_h = mu + off[[ti[t] for t in gg["home"]]] - dfn[[ti[t] for t in gg["away"]]] \
        + np.where(gg["neutral"], 0.0, hfa)
    rate_a = mu + off[[ti[t] for t in gg["away"]]] - dfn[[ti[t] for t in gg["home"]]] \
        - np.where(gg["neutral"], 0.0, hfa)
    X = np.concatenate([rate_h, rate_a])
    Y = np.concatenate([gg["home_points"].values, gg["away_points"].values]).astype(float)
    W = np.concatenate([ww, ww])
    c1 = float(np.cov(X, Y, aweights=W)[0, 1] / np.cov(X, aweights=W))
    c0 = float(np.average(Y, weights=W) - c1 * np.average(X, weights=W))

    pred_m = (c0 + c1 * rate_h) - (c0 + c1 * rate_a)
    act_m = (gg["home_points"] - gg["away_points"]).values
    sigma = float(np.sqrt(np.average((act_m - pred_m) ** 2, weights=ww)))
    pred_t = (c0 + c1 * rate_h) + (c0 + c1 * rate_a)
    act_t = (gg["home_points"] + gg["away_points"]).values
    sigma_total = float(np.sqrt(np.average((act_t - pred_t) ** 2, weights=ww)))

    return {
        "asof": str(asof.date()), "field": field, "mu": mu, "hfa": hfa, "c0": c0, "c1": c1,
        "sigma": sigma, "sigma_total": sigma_total,
        "teams": {t: {"off": float(off[ti[t]]), "def": float(dfn[ti[t]])} for t in teams},
    }


def predict(params, team1, team2, neutral=False):
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


def load_params(path=PARAMS_JSON):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", nargs="*")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--ratings", action="store_true")
    args = ap.parse_args()

    if args.fit:
        params = fit()
        with open(PARAMS_JSON, "w") as f:
            json.dump(params, f, indent=1)
        print(f"fitted {len(params['teams'])} teams as of {params['asof']}: "
              f"mu={params['mu']:.3f} ppa/play, hfa={params['hfa']:.3f}, "
              f"pts = {params['c0']:.1f} + {params['c1']:.1f}*rate, "
              f"sigma={params['sigma']:.1f}/{params['sigma_total']:.1f}")
        return
    params = load_params()
    if args.ratings:
        t = [(k, v["off"], v["def"], v["off"] + v["def"])
             for k, v in params["teams"].items() if k != FCS]
        print(f"{'team':<25s} {'off':>7s} {'def':>7s} {'net':>7s}  (adj PPA/play)")
        for name, o, d, net in sorted(t, key=lambda r: -r[3])[:30]:
            print(f"{name:<25s} {o:>+7.3f} {d:>+7.3f} {net:>+7.3f}")
        return
    if len(args.teams) != 2:
        raise SystemExit(__doc__)
    t1, t2 = args.teams
    p = predict(params, t1, t2, args.neutral)
    venue = "neutral site" if args.neutral else f"{t1} at home"
    print(f"{t1} vs {t2} ({venue}, EPA model)")
    print(f"  Expected score: {t1} {p['pts1']:.1f} - {p['pts2']:.1f} {t2}")
    print(f"  Margin {p['margin']:+.1f}, total {p['total']:.1f}, P({t1} win) = {p['p1']:.1%}")


if __name__ == "__main__":
    main()
