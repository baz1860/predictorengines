#!/usr/bin/env python3
"""Backtest the tournament title-odds model on World Cup 2018 (Russia).

Same engine as simulate.py / wc2022_sim_backtest.py (Elo+Poisson / Dixon-Coles
blend → Monte-Carlo bracket), with every model object fitted ONLY on matches
before the 2018-06-14 kickoff — the odds the model would have produced going in,
no hindsight. 32-team format, identical FIFA bracket layout to 2022.

    python3 wc2018_sim_backtest.py [-n 20000] [--seed 42] [--model blend]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root, for engines.worldcup / core imports
from engines.worldcup.predictor import (load_matches, compute_elo, fit_goal_model,
                       expected_goals, HOME_ADV, DC_RHO)
from engines.worldcup.dixoncoles import fit_dc
from wc2022_sim_backtest import Model, rank_group   # reuse the exact engine

CUTOFF = "2018-06-14"

GROUPS = {
    "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
    "B": ["Portugal", "Spain", "Morocco", "Iran"],
    "C": ["France", "Australia", "Peru", "Denmark"],
    "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
    "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
    "F": ["Germany", "Mexico", "Sweden", "South Korea"],
    "G": ["Belgium", "Panama", "Tunisia", "England"],
    "H": ["Poland", "Senegal", "Colombia", "Japan"],
}
TEAM_GROUP = {t: g for g, ts in GROUPS.items() for t in ts}

R16 = [("1A", "2B"), ("1C", "2D"), ("1E", "2F"), ("1G", "2H"),
       ("1B", "2A"), ("1D", "2C"), ("1F", "2E"), ("1H", "2G")]
QF = [(0, 1), (2, 3), (4, 5), (6, 7)]
SF = [(0, 1), (2, 3)]

ACTUAL = {"champion": "France", "runner_up": "Croatia",
          "semis": {"France", "Croatia", "Belgium", "England"}}


def build_pit_sources(model: str):
    played, _ = load_matches()
    _, played = compute_elo(played)
    train = played[played["date"] < CUTOFF]
    ratings, _ = compute_elo(train)
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
    ap = argparse.ArgumentParser(description="WC2018 tournament-model backtest")
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
        for t in qfists: c[t]["reach_QF"] += 1
        for t in sfists: c[t]["reach_SF"] += 1
        for t in finalists: c[t]["reach_final"] += 1
        c[champ]["champion"] += 1

    n = args.sims
    df = pd.DataFrame([{"team": t, "group": TEAM_GROUP[t], "elo": round(ratings[t]),
                        "reach_QF": round(c[t]["reach_QF"] / n, 4),
                        "reach_SF": round(c[t]["reach_SF"] / n, 4),
                        "reach_final": round(c[t]["reach_final"] / n, 4),
                        "champion": round(c[t]["champion"] / n, 4)}
                       for t in TEAM_GROUP]).sort_values("champion", ascending=False)

    pd.set_option("display.width", 140)
    print(f"\nWC2018 pre-tournament model ({args.model}, {args.sims:,} sims, "
          f"Elo/DC fitted only on data < {CUTOFF})\n")
    print(df.head(12).to_string(index=False))

    a = ACTUAL
    rank = {t: i + 1 for i, t in enumerate(df["team"])}
    champ_p = df.set_index("team")["champion"]
    sf_p = df.set_index("team")["reach_SF"]
    print("\n— How it actually finished —")
    print(f"Champion:   {a['champion']}  → model had {champ_p[a['champion']]:.1%} "
          f"(rank #{rank[a['champion']]} of 32 for title)")
    print(f"Runner-up:  {a['runner_up']}  → model title {champ_p[a['runner_up']]:.1%} "
          f"(rank #{rank[a['runner_up']]})")
    print("Semifinalists: " + ", ".join(
        f"{t} {sf_p[t]:.0%}" for t in sorted(a['semis'], key=lambda x: -sf_p[x])))
    fav = df.iloc[0]["team"]
    print(f"\nModel's pre-tournament favourite: {fav} ({champ_p[fav]:.1%}). "
          f"Actual winner {a['champion']} was rank #{rank[a['champion']]}.")
    top4 = set(df.head(4)["team"])
    print(f"Model top-4 by title odds: {sorted(top4)}")
    print(f"  → {len(top4 & a['semis'])}/4 actually reached the semis: "
          f"{sorted(top4 & a['semis'])}")

    # calibration vs a no-skill baseline (same as the 2022 backtest)
    teams = list(TEAM_GROUP)
    def brier(key, actual_set):
        y = {t: 1.0 if t in actual_set else 0.0 for t in teams}
        P = df.set_index("team")[key]
        model_b = np.mean([(P[t] - y[t]) ** 2 for t in teams])
        base = len(actual_set) / len(teams)
        base_b = np.mean([(base - y[t]) ** 2 for t in teams])
        return model_b, base_b
    print("\nCalibration (Brier, lower = better):")
    for key, act, lbl in [("reach_SF", a["semis"], "reach SF (4/32)"),
                          ("champion", {a["champion"]}, "champion (1/32)")]:
        mo, ba = brier(key, act)
        print(f"  {lbl:18s} model {mo:.4f} vs no-skill {ba:.4f}  "
              f"-> {'BETTER' if mo < ba else 'worse'} by {ba - mo:+.4f}")
    return df


if __name__ == "__main__":
    main()
