#!/usr/bin/env python3
"""Projected 2026 regular-season win totals.

Per-game win probabilities from the blend (Elo with 2026 preseason priors +
power ratings), combined into each team's exact win distribution via
Poisson-binomial DP. Output: projected_win_totals_2026.csv with expected wins,
quartiles, and P(over) for the half-line nearest the mean.

Caveat: power ratings are end-of-2025 (no roster adjustment); only the Elo
half of the blend carries 2026 returning-production/talent priors.

Usage: python3 win_totals.py [--year 2026]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import elo as E
from . import power as P
from .predictor import load_blend_weight

HERE = os.path.dirname(os.path.abspath(__file__))


def _get(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_schedule(year):
    path = os.path.join(HERE, "data", f"schedule_{year}.json")
    with open(path) as f:
        data = json.load(f)
    rows = []
    for g in data:
        hc = (_get(g, "homeClassification", "home_classification") or "").lower()
        ac = (_get(g, "awayClassification", "away_classification") or "").lower()
        if hc != "fbs" and ac != "fbs":
            continue
        rows.append({
            "week": _get(g, "week"),
            "home_team": _get(g, "homeTeam", "home_team"),
            "away_team": _get(g, "awayTeam", "away_team"),
            "home_fbs": hc == "fbs", "away_fbs": ac == "fbs",
            "neutral": bool(_get(g, "neutralSite", "neutral_site") or False),
            "home_conf": _get(g, "homeConference", "home_conference"),
            "away_conf": _get(g, "awayConference", "away_conference"),
        })
    return pd.DataFrame(rows)


def win_dist(probs):
    """Poisson-binomial: exact P(total wins = k)."""
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, [1.0 - p, p])
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()

    sched = load_schedule(args.year)
    (games, ratings, slope, sigma), state_meta = E.build_as_of(args.year)
    pparams = P.load_params()
    pre = ratings
    w_elo = load_blend_weight()

    team_probs, team_conf = {}, {}
    skipped = 0
    for r in sched.itertuples():
        h = r.home_team if (r.home_fbs or r.home_team in pre) else E.FCS
        a = r.away_team if (r.away_fbs or r.away_team in pre) else E.FCS
        if h not in pre or a not in pre:
            skipped += 1
            continue
        hfa = 0.0 if r.neutral else E.HFA_ELO
        p_elo = E.win_prob(pre[h] + hfa - pre[a])
        try:
            p_pow = P.predict(pparams, h, a, neutral=bool(r.neutral))["p1"]
            p_home = w_elo * p_elo + (1.0 - w_elo) * p_pow
        except SystemExit:
            p_home = p_elo
        if r.home_fbs:
            team_probs.setdefault(r.home_team, []).append(p_home)
            team_conf[r.home_team] = r.home_conf
        if r.away_fbs:
            team_probs.setdefault(r.away_team, []).append(1.0 - p_home)
            team_conf[r.away_team] = r.away_conf

    rows = []
    for team, probs in team_probs.items():
        d = win_dist(probs)
        k = np.arange(len(d))
        exp_w = float((k * d).sum())
        cdf = np.cumsum(d)
        line = round(exp_w) - 0.5 if round(exp_w) >= 1 else 0.5
        rows.append({
            "team": team, "conference": team_conf.get(team) or "",
            "games": len(probs), "exp_wins": round(exp_w, 2),
            "sd": round(float(np.sqrt(((k - exp_w) ** 2 * d).sum())), 2),
            "p25": int(np.searchsorted(cdf, 0.25)),
            "median": int(np.searchsorted(cdf, 0.50)),
            "p75": int(np.searchsorted(cdf, 0.75)),
            "nearest_line": line,
            "p_over_line": round(float(1.0 - cdf[int(line)]), 3),
            "p_bowl_6plus": round(float(1.0 - cdf[5]) if len(cdf) > 5 else 1.0, 3),
        })
    out = pd.DataFrame(rows).sort_values("exp_wins", ascending=False).reset_index(drop=True)
    dest = os.path.join(HERE, f"projected_win_totals_{args.year}.csv")
    out.to_csv(dest, index=False)
    print(f"{len(out)} teams, {sum(len(v) for v in team_probs.values()) // 2}+ games, "
          f"{skipped} skipped -> {dest}")
    print(f"model season {state_meta['model_season']} · state {state_meta['prior_mode']} · "
          f"snapshot {state_meta['snapshot_hash']}\n")
    print(out.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
