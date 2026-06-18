#!/usr/bin/env python3
"""Backtest the TOURNAMENT title-odds model on World Cup 2022.

Same engine as simulate.py (Elo+Poisson / Dixon-Coles blend → scoreline matrices
→ Monte Carlo bracket), but every model object is fitted ONLY on matches before
the 2022-11-20 kickoff, so the title odds are exactly what the model would have
produced going into Qatar 2022 — no hindsight.

All matches neutral (Qatar 2022 was a single-venue tournament). Output: each
team's pre-tournament champion / finalist / semifinalist probability, compared
to what actually happened (Argentina champions, France runners-up, Croatia &
Morocco semi-finalists).

    python3 wc2022_sim_backtest.py [-n 20000] [--seed 42] [--model blend]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from predictor import (load_matches, compute_elo, fit_goal_model,
                       expected_goals, score_matrix, HOME_ADV, DC_RHO, MAX_GOALS)
from dixoncoles import fit_dc

CUTOFF = "2022-11-20"

GROUPS = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}
TEAM_GROUP = {t: g for g, ts in GROUPS.items() for t in ts}

# R16 slot pairings (FIFA 2022 layout): (winnerX, runnerupY)
R16 = [("1A", "2B"), ("1C", "2D"), ("1E", "2F"), ("1G", "2H"),
       ("1B", "2A"), ("1D", "2C"), ("1F", "2E"), ("1H", "2G")]
# QF/SF/Final fold the bracket in pairs.
QF = [(0, 1), (2, 3), (4, 5), (6, 7)]
SF = [(0, 1), (2, 3)]

# What actually happened (for scoring the backtest).
ACTUAL = {"champion": "Argentina", "runner_up": "France",
          "semis": {"Argentina", "France", "Croatia", "Morocco"},
          "quarters": {"Argentina", "France", "Croatia", "Morocco",
                       "Netherlands", "Brazil", "England", "Portugal"}}


class Model:
    """Neutral-venue scoreline sampler over one or more lambda sources."""

    def __init__(self, sources):
        self.sources = sources
        self.cache = {}

    def _dist(self, t1, t2):
        if (t1, t2) not in self.cache:
            Ms, l1s, l2s = [], [], []
            for fn, rho in self.sources:
                lam1, lam2 = fn(t1, t2, 0.0, 0.0)
                Ms.append(score_matrix(lam1, lam2, rho))
                l1s.append(lam1); l2s.append(lam2)
            M = np.mean(Ms, axis=0)
            self.cache[(t1, t2)] = (np.cumsum(M.ravel()),
                                    float(np.mean(l1s)), float(np.mean(l2s)))
        return self.cache[(t1, t2)]

    def sample(self, t1, t2, rng):
        cum, _, _ = self._dist(t1, t2)
        return divmod(int(np.searchsorted(cum, rng.random())), MAX_GOALS + 1)

    def knockout(self, t1, t2, rng):
        g1, g2 = self.sample(t1, t2, rng)
        if g1 != g2:
            return t1 if g1 > g2 else t2
        _, lam1, lam2 = self._dist(t1, t2)            # extra time at 1/3 intensity
        e1, e2 = rng.poisson(lam1 / 3), rng.poisson(lam2 / 3)
        if e1 != e2:
            return t1 if e1 > e2 else t2
        return t1 if rng.random() < 0.5 else t2       # penalties: 50/50


def build_pit_sources(model: str):
    """Point-in-time (leak-free) lambda sources as of CUTOFF."""
    played, _ = load_matches()
    _, played = compute_elo(played)                   # adds pre-game elo cols
    train = played[played["date"] < CUTOFF]
    ratings, _ = compute_elo(train)                   # ratings AS OF cutoff
    sources = []
    if model in ("elo", "blend"):
        beta = fit_goal_model(train)
        sources.append((lambda t1, t2, h1=0.0, h2=0.0:
                        expected_goals(ratings[t1], ratings[t2], beta, (h1 - h2) * HOME_ADV),
                        DC_RHO))
    if model in ("dc", "blend"):
        dc = fit_dc(train, anchor=CUTOFF, verbose=False)
        sources.append((dc.lambdas, dc.rho))
    return sources, ratings


def rank_group(teams, model, rng):
    pts = dict.fromkeys(teams, 0); gd = dict.fromkeys(teams, 0); gf = dict.fromkeys(teams, 0)
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            ga, gb = model.sample(a, b, rng)
            gf[a] += ga; gf[b] += gb; gd[a] += ga - gb; gd[b] += gb - ga
            if ga > gb: pts[a] += 3
            elif gb > ga: pts[b] += 3
            else: pts[a] += 1; pts[b] += 1
    order = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t], rng.random()), reverse=True)
    return order[0], order[1]


def simulate_once(model, rng):
    slot = {}
    for g, teams in GROUPS.items():
        w, r = rank_group(teams, model, rng)
        slot[f"1{g}"], slot[f"2{g}"] = w, r
    r16w = [model.knockout(slot[a], slot[b], rng) for a, b in R16]
    qfw = [model.knockout(r16w[a], r16w[b], rng) for a, b in QF]
    sfw = [model.knockout(qfw[a], qfw[b], rng) for a, b in SF]
    champ = model.knockout(sfw[0], sfw[1], rng)
    return set(r16w), set(qfw), set(sfw), champ


def main():
    ap = argparse.ArgumentParser(description="WC2022 tournament-model backtest")
    ap.add_argument("-n", "--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["elo", "dc", "blend"], default="blend")
    args = ap.parse_args()

    sources, ratings = build_pit_sources(args.model)
    model = Model(sources)
    rng = np.random.default_rng(args.seed)

    c = defaultdict(lambda: dict(reach_QF=0, reach_SF=0, reach_final=0, champion=0))
    for _ in range(args.sims):
        qfists, sfists, finalists, champ = simulate_once(model, rng)
        for t in qfists: c[t]["reach_QF"] += 1     # 8 quarter-finalists
        for t in sfists: c[t]["reach_SF"] += 1     # 4 semi-finalists
        for t in finalists: c[t]["reach_final"] += 1  # 2 finalists
        c[champ]["champion"] += 1

    rows = []
    for t in TEAM_GROUP:
        n = args.sims
        rows.append({"team": t, "group": TEAM_GROUP[t], "elo": round(ratings[t]),
                     "reach_QF": round(c[t]["reach_QF"] / n, 4),
                     "reach_SF": round(c[t]["reach_SF"] / n, 4),
                     "reach_final": round(c[t]["reach_final"] / n, 4),
                     "champion": round(c[t]["champion"] / n, 4)})
    df = pd.DataFrame(rows).sort_values("champion", ascending=False)

    pd.set_option("display.width", 140)
    print(f"\nWC2022 pre-tournament model ({args.model}, {args.sims:,} sims, "
          f"Elo/DC fitted only on data < {CUTOFF})\n")
    print(df.head(12).to_string(index=False))

    print("\n— How it actually finished —")
    a = ACTUAL
    rank = {t: i + 1 for i, t in enumerate(df["team"])}
    champ_p = df.set_index("team")["champion"]
    sf_p = df.set_index("team")["reach_SF"]
    print(f"Champion:   {a['champion']}  → model had {champ_p[a['champion']]:.1%} "
          f"(rank #{rank[a['champion']]} of 32 for title)")
    print(f"Runner-up:  {a['runner_up']}  → model title {champ_p[a['runner_up']]:.1%} "
          f"(rank #{rank[a['runner_up']]})")
    print("Semifinalists: " + ", ".join(
        f"{t} {sf_p[t]:.0%}" for t in sorted(a['semis'], key=lambda x: -sf_p[x])))
    # calibration-style checks
    fav = df.iloc[0]["team"]
    print(f"\nModel's pre-tournament favourite: {fav} ({champ_p[fav]:.1%}). "
          f"Actual winner Argentina was rank #{rank['Argentina']}.")
    top4 = set(df.head(4)["team"])
    hit = top4 & a['semis']
    print(f"Model top-4 by title odds: {sorted(top4)}")
    print(f"  → {len(hit)}/4 actually reached the semis: {sorted(hit)}")
    return df


if __name__ == "__main__":
    main()
