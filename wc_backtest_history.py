#!/usr/bin/env python3
"""Backtest the tournament title-odds model across multiple World Cups.

Same engine as simulate.py (Elo+Poisson / Dixon-Coles blend → Monte-Carlo
bracket), fitted leak-free as of each edition's kickoff. 32-team format with the
identical FIFA bracket cross used 1998–2022. Reports, per edition and in
aggregate, how the model's pre-tournament odds compare to what happened.

    python3 wc_backtest_history.py [-n 20000] [--seed 42] [--years 2006,2010,2014]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from predictor import (load_matches, compute_elo, fit_goal_model,
                       expected_goals, HOME_ADV, DC_RHO)
from dixoncoles import fit_dc
from wc2022_sim_backtest import Model, rank_group   # reuse the exact engine

# Identical bracket layout for every 32-team edition.
R16 = [("1A", "2B"), ("1C", "2D"), ("1E", "2F"), ("1G", "2H"),
       ("1B", "2A"), ("1D", "2C"), ("1F", "2E"), ("1H", "2G")]
QF, SF = [(0, 1), (2, 3), (4, 5), (6, 7)], [(0, 1), (2, 3)]

# "Serbia and Montenegro" (WC2006) is carried as "Serbia" in the results data.
EDITIONS = {
    2006: dict(cutoff="2006-06-09", champion="Italy", runner_up="France",
               semis={"Italy", "France", "Germany", "Portugal"}, groups={
        "A": ["Germany", "Costa Rica", "Poland", "Ecuador"],
        "B": ["England", "Paraguay", "Trinidad and Tobago", "Sweden"],
        "C": ["Argentina", "Ivory Coast", "Serbia", "Netherlands"],
        "D": ["Mexico", "Iran", "Angola", "Portugal"],
        "E": ["Italy", "Ghana", "United States", "Czech Republic"],
        "F": ["Brazil", "Croatia", "Australia", "Japan"],
        "G": ["France", "Switzerland", "South Korea", "Togo"],
        "H": ["Spain", "Ukraine", "Tunisia", "Saudi Arabia"]}),
    2010: dict(cutoff="2010-06-11", champion="Spain", runner_up="Netherlands",
               semis={"Spain", "Netherlands", "Germany", "Uruguay"}, groups={
        "A": ["South Africa", "Mexico", "Uruguay", "France"],
        "B": ["Argentina", "Nigeria", "South Korea", "Greece"],
        "C": ["England", "United States", "Algeria", "Slovenia"],
        "D": ["Germany", "Australia", "Serbia", "Ghana"],
        "E": ["Netherlands", "Denmark", "Japan", "Cameroon"],
        "F": ["Italy", "Paraguay", "New Zealand", "Slovakia"],
        "G": ["Brazil", "North Korea", "Ivory Coast", "Portugal"],
        "H": ["Spain", "Switzerland", "Honduras", "Chile"]}),
    2014: dict(cutoff="2014-06-12", champion="Germany", runner_up="Argentina",
               semis={"Germany", "Argentina", "Netherlands", "Brazil"}, groups={
        "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
        "B": ["Spain", "Netherlands", "Chile", "Australia"],
        "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
        "D": ["Uruguay", "Costa Rica", "England", "Italy"],
        "E": ["Switzerland", "Ecuador", "France", "Honduras"],
        "F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
        "G": ["Germany", "Portugal", "Ghana", "United States"],
        "H": ["Belgium", "Algeria", "Russia", "South Korea"]}),
    2018: dict(cutoff="2018-06-14", champion="France", runner_up="Croatia",
               semis={"France", "Croatia", "Belgium", "England"}, groups={
        "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        "B": ["Portugal", "Spain", "Morocco", "Iran"],
        "C": ["France", "Australia", "Peru", "Denmark"],
        "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
        "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        "F": ["Germany", "Mexico", "Sweden", "South Korea"],
        "G": ["Belgium", "Panama", "Tunisia", "England"],
        "H": ["Poland", "Senegal", "Colombia", "Japan"]}),
    2022: dict(cutoff="2022-11-20", champion="Argentina", runner_up="France",
               semis={"Argentina", "France", "Croatia", "Morocco"}, groups={
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"]}),
}


def fit_goal_model_asof(train, cutoff, years=8):
    """Elo→goals Poisson fit (same IRLS as predictor.fit_goal_model) but on a
    trailing window BEFORE the cutoff. The live model hardcodes 'since 2010',
    which is empty/too-sparse for pre-2014 editions; an 8-year trailing window is
    the leak-free analogue and keeps every edition comparable."""
    since = (pd.Timestamp(cutoff) - pd.DateOffset(years=years))
    recent = train[train["date"] >= since]
    adv = np.where(recent["neutral"], 0.0, HOME_ADV)
    diff = (recent["elo_h"] + adv - recent["elo_a"]).to_numpy() / 400.0
    x = np.concatenate([diff, -diff])
    y = np.concatenate([recent["home_score"].to_numpy(), recent["away_score"].to_numpy()])
    beta = np.zeros(2); X = np.column_stack([np.ones_like(x), x])
    for _ in range(25):
        mu = np.exp(X @ beta); z = X @ beta + (y - mu) / mu
        XtW = X.T * mu
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new; break
        beta = beta_new
    return beta


def build_pit_sources(cutoff, model="blend"):
    played, _ = load_matches()
    _, played = compute_elo(played)
    train = played[played["date"] < cutoff]
    ratings, _ = compute_elo(train)
    sources = []
    if model in ("elo", "blend"):
        beta = fit_goal_model_asof(train, cutoff)
        sources.append((lambda t1, t2, h1=0.0, h2=0.0: expected_goals(
            ratings[t1], ratings[t2], beta, (h1 - h2) * HOME_ADV), DC_RHO))
    if model in ("dc", "blend"):
        dc = fit_dc(train, anchor=cutoff, verbose=False)
        sources.append((dc.lambdas, dc.rho))
    return sources, ratings


def simulate_once(model, groups, rng):
    slot = {}
    for g, teams in groups.items():
        slot[f"1{g}"], slot[f"2{g}"] = rank_group(teams, model, rng)
    r16w = [model.knockout(slot[a], slot[b], rng) for a, b in R16]
    qfw = [model.knockout(r16w[a], r16w[b], rng) for a, b in QF]
    sfw = [model.knockout(qfw[a], qfw[b], rng) for a, b in SF]
    return set(qfw), set(sfw), model.knockout(sfw[0], sfw[1], rng)


def run_edition(year, cfg, n, seed, model_name):
    groups = cfg["groups"]
    team_group = {t: g for g, ts in groups.items() for t in ts}
    sources, ratings = build_pit_sources(cfg["cutoff"], model_name)
    model = Model(sources)
    rng = np.random.default_rng(seed)
    c = defaultdict(lambda: dict(reach_SF=0, champion=0))
    for _ in range(n):
        _, sf, champ = simulate_once(model, groups, rng)
        for t in sf:
            c[t]["reach_SF"] += 1
        c[champ]["champion"] += 1
    df = pd.DataFrame([{"team": t, "elo": round(ratings[t]),
                        "reach_SF": c[t]["reach_SF"] / n,
                        "champion": c[t]["champion"] / n} for t in team_group]
                      ).sort_values("champion", ascending=False).reset_index(drop=True)

    teams = list(team_group)
    def brier(col, actual):
        y = {t: 1.0 if t in actual else 0.0 for t in teams}
        P = df.set_index("team")[col]
        base = len(actual) / len(teams)
        return (np.mean([(P[t] - y[t]) ** 2 for t in teams]),
                np.mean([(base - y[t]) ** 2 for t in teams]))

    rank = {t: i + 1 for i, t in enumerate(df["team"])}
    cp = df.set_index("team")["champion"]; sp = df.set_index("team")["reach_SF"]
    top4 = set(df.head(4)["team"])
    b_sf = brier("reach_SF", cfg["semis"]); b_ch = brier("champion", {cfg["champion"]})
    return {
        "year": year, "fav": df.iloc[0]["team"], "fav_p": cp.iloc[0],
        "champ": cfg["champion"], "champ_rank": rank[cfg["champion"]],
        "champ_p": cp[cfg["champion"]], "fav_won": df.iloc[0]["team"] == cfg["champion"],
        "ru": cfg["runner_up"], "ru_rank": rank[cfg["runner_up"]],
        "top4_semis": len(top4 & cfg["semis"]),
        "sf_brier": b_sf[0], "sf_base": b_sf[1],
        "ch_brier": b_ch[0], "ch_base": b_ch[1],
        "df": df, "rank": rank, "sp": sp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="blend", choices=["elo", "dc", "blend"])
    ap.add_argument("--years", default="2006,2010,2014,2018,2022")
    args = ap.parse_args()
    years = [int(y) for y in args.years.split(",")]

    res = []
    for y in years:
        r = run_edition(y, EDITIONS[y], args.sims, args.seed, args.model)
        res.append(r)
        print(f"\n{'='*72}\nWC{y}  (model fitted on data < {EDITIONS[y]['cutoff']}, "
              f"{args.sims:,} sims)\n")
        top = r["df"].head(6).copy()
        top["reach_SF"] = (top["reach_SF"] * 100).round(0).astype(int).astype(str) + "%"
        top["champion"] = (top["champion"] * 100).round(1).astype(str) + "%"
        print(top.to_string(index=False))
        print(f"\n  Actual champion : {r['champ']}  → model {r['champ_p']:.1%} (title rank #{r['champ_rank']})")
        print(f"  Runner-up       : {r['ru']}  → title rank #{r['ru_rank']}")
        print(f"  Model favourite : {r['fav']} ({r['fav_p']:.1%})  "
              f"{'✓ won' if r['fav_won'] else '✗ did not win'}")
        print(f"  Top-4 → semis   : {r['top4_semis']}/4")
        print(f"  Brier reach-SF  : {r['sf_brier']:.4f} vs no-skill {r['sf_base']:.4f}"
              f" ({'+' if r['sf_brier']<r['sf_base'] else '-'} info)")

    print(f"\n{'='*72}\nAGGREGATE over {len(res)} editions ({', '.join(map(str,years))})\n")
    hdr = f"{'year':>6}{'fav':>13}{'fav%':>7}{'won?':>6}{'champion':>12}{'rank':>6}{'champ%':>8}{'top4→sf':>9}"
    print(hdr)
    for r in res:
        print(f"{r['year']:>6}{r['fav']:>13}{r['fav_p']*100:>6.1f}%"
              f"{'yes' if r['fav_won'] else 'no':>6}{r['champ']:>12}{r['champ_rank']:>6}"
              f"{r['champ_p']*100:>7.1f}%{r['top4_semis']:>6}/4")
    fav_wins = sum(r["fav_won"] for r in res)
    champ_in_top4 = sum(r["champ_rank"] <= 4 for r in res)
    mean_rank = np.mean([r["champ_rank"] for r in res])
    sf_better = sum(r["sf_brier"] < r["sf_base"] for r in res)
    ch_better = sum(r["ch_brier"] < r["ch_base"] for r in res)
    print(f"\n  Model favourite won the title : {fav_wins}/{len(res)}")
    print(f"  Actual champion in model top-4: {champ_in_top4}/{len(res)}")
    print(f"  Mean title-rank of actual champ: {mean_rank:.1f} (of 32)")
    print(f"  Beat no-skill Brier (reach-SF) : {sf_better}/{len(res)} editions")
    print(f"  Beat no-skill Brier (champion) : {ch_better}/{len(res)} editions")


if __name__ == "__main__":
    main()
