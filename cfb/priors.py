#!/usr/bin/env python3
"""Preseason priors for the Elo model: 247 talent composite + returning production.

Loads data/cfbd/talent_<year>.json and returning_<year>.json (CFBD API pulls),
turns them into Elo offsets applied at each team's first game of a season:

    rating = 1500 + carry * (rating_prev - 1500) + b_talent * talent_z
                  + b_ret * (returning_PPA_pct - season_mean) / 10

Coefficients live in data/prior_params.json. Tune with:

  python3 priors.py --tune       # grid search on weeks 1-4, seasons 2016-2024
"""
import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CFBD_DIR = os.path.join(HERE, "data", "cfbd")
PARAMS_JSON = os.path.join(HERE, "data", "prior_params.json")

DEFAULTS = {"carry": 0.70, "b_talent": 0.0, "b_ret": 0.0}


def _get(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_features():
    """-> dict[(team, season)] = {'talent_z': float, 'ret_c': float} (centered)."""
    feats = {}
    for path in glob.glob(os.path.join(CFBD_DIR, "talent_*.json")):
        season = int(re.search(r"(\d{4})", os.path.basename(path)).group(1))
        try:
            data = json.load(open(path))
        except json.JSONDecodeError:
            continue
        vals = {}
        for r in data:
            team = _get(r, "school", "team")
            t = _get(r, "talent")
            if team and t is not None:
                vals[team] = float(t)
        if len(vals) < 20:
            continue
        arr = np.array(list(vals.values()))
        mu, sd = arr.mean(), arr.std()
        for team, t in vals.items():
            feats.setdefault((team, season), {})["talent_z"] = (t - mu) / sd

    for path in glob.glob(os.path.join(CFBD_DIR, "returning_*.json")):
        season = int(re.search(r"(\d{4})", os.path.basename(path)).group(1))
        try:
            data = json.load(open(path))
        except json.JSONDecodeError:
            continue
        vals = {}
        for r in data:
            team = _get(r, "team", "school")
            pct = _get(r, "percentPPA", "percent_ppa")
            if team and pct is not None:
                vals[team] = float(pct)
        if len(vals) < 20:
            continue
        arr = np.array(list(vals.values()))
        if np.nanmax(arr) <= 1.5:  # 0-1 scale -> percent
            vals = {k: v * 100.0 for k, v in vals.items()}
            arr = arr * 100.0
        mu = arr.mean()
        for team, pct in vals.items():
            feats.setdefault((team, season), {})["ret_c"] = pct - mu
    return feats


def load_params():
    if os.path.exists(PARAMS_JSON):
        with open(PARAMS_JSON) as f:
            return json.load(f)
    return dict(DEFAULTS)


def offsets(feats, params):
    """-> dict[(team, season)] = elo offset, plus carry."""
    out = {}
    for key, f in feats.items():
        out[key] = (params["b_talent"] * f.get("talent_z", 0.0)
                    + params["b_ret"] * f.get("ret_c", 0.0) / 10.0)
    return out


def tune(since=2016, until=2024):
    import elo as E

    games = E.load_games()
    feats = load_features()
    early = (games["season"].between(since, until)) & (games["week"] <= 4) \
        & (games["home"] != E.FCS) & (games["away"] != E.FCS)
    actual = (games["home_points"] > games["away_points"]).astype(float).values

    best = None
    for carry in (0.65, 0.70, 0.75, 0.80):
        for b_talent in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0):
            for b_ret in (0.0, 8.0, 12.0, 16.0, 24.0):
                params = {"carry": carry, "b_talent": b_talent, "b_ret": b_ret}
                offs = offsets(feats, params)
                _, hist = E.run_elo(games, record_pregame=True,
                                    carry=carry, prior_offsets=offs)
                diffs = np.array([h[2] for h in hist])
                p = 1.0 / (1.0 + 10.0 ** (-diffs[early.values] / 400.0))
                brier = float(np.mean((p - actual[early.values]) ** 2))
                if best is None or brier < best[0]:
                    best = (brier, params)
                    print(f"  brier {brier:.4f}  {params}")
    brier, params = best
    with open(PARAMS_JSON, "w") as f:
        json.dump(params, f, indent=1)
    print(f"\nbest weeks 1-4 Brier {brier:.4f} ({since}-{until}) -> {PARAMS_JSON}: {params}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--since", type=int, default=2016)
    ap.add_argument("--until", type=int, default=2024)
    args = ap.parse_args()
    if args.tune:
        tune(args.since, args.until)
    else:
        feats = load_features()
        params = load_params()
        offs = offsets(feats, params)
        print(f"{len(feats)} (team, season) feature rows; params {params}")
        top = sorted(((k, v) for k, v in offs.items() if k[1] == max(s for _, s in offs)),
                     key=lambda kv: -kv[1])[:10]
        for (team, season), o in top:
            print(f"  {season} {team:<20s} {o:+.1f}")
