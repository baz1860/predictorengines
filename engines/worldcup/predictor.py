#!/usr/bin/env python3
"""World Cup match prediction engine.

Pipeline:
  1. Elo ratings computed over all international matches (1872-present),
     with K scaled by tournament importance and goal margin.
  2. Poisson goal model: expected goals for each side fitted as a function
     of Elo difference (Poisson regression on matches since 2010).
  3. Dixon-Coles low-score adjustment to fix the draw underestimate of
     independent Poisson.

Usage:
  python predictor.py "Brazil" "Morocco"          # one match (neutral venue)
  python predictor.py "Mexico" "South Africa" --home  # home advantage for team 1
  python predictor.py --worldcup                  # predict all unplayed WC 2026 fixtures
  python predictor.py --backtest                  # evaluate on matches since 2024
  python predictor.py --ratings                   # top 30 current Elo ratings
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "results.csv"
HOME_ADV = 65.0          # Elo points for non-neutral home side
BASE_RATING = 1500.0
MAX_GOALS = 10           # scoreline grid size
DC_RHO = -0.10           # Dixon-Coles correlation for low scores

K_BY_TOURNAMENT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50, "Copa América": 50, "African Cup of Nations": 50,
    "AFC Asian Cup": 50, "CONCACAF Championship": 50, "Gold Cup": 50,
    "UEFA Nations League": 40, "CONCACAF Nations League": 40,
    "Confederations Cup": 50,
    "Friendly": 20,
}
DEFAULT_K = 30


def load_matches():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    played = df.dropna(subset=["home_score", "away_score"]).copy()
    upcoming = df[df["home_score"].isna()].copy()
    return played, upcoming


def compute_elo(played):
    """Iterate chronologically; return final ratings dict and per-match pre-Elo columns."""
    ratings = {}
    pre_h, pre_a = np.empty(len(played)), np.empty(len(played))
    rows = played[["home_team", "away_team", "home_score", "away_score",
                   "tournament", "neutral"]].itertuples(index=False)
    for i, (h, a, hs, as_, tour, neutral) in enumerate(rows):
        rh = ratings.get(h, BASE_RATING)
        ra = ratings.get(a, BASE_RATING)
        pre_h[i], pre_a[i] = rh, ra
        adv = 0.0 if neutral else HOME_ADV
        exp_h = 1.0 / (1.0 + 10 ** ((ra - (rh + adv)) / 400.0))
        score_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        k = K_BY_TOURNAMENT.get(tour, DEFAULT_K)
        margin = abs(hs - as_)
        g = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8.0)
        delta = k * g * (score_h - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    played = played.copy()
    played["elo_h"], played["elo_a"] = pre_h, pre_a
    return ratings, played


def fit_goal_model(played):
    """Poisson regression: log(goals) = alpha + beta * elo_diff/400.
    Fit on matches since 2010, stacking home and away observations."""
    recent = played[played["date"] >= "2010-01-01"]
    adv = np.where(recent["neutral"], 0.0, HOME_ADV)
    diff_h = (recent["elo_h"] + adv - recent["elo_a"]).to_numpy() / 400.0
    x = np.concatenate([diff_h, -diff_h])
    y = np.concatenate([recent["home_score"].to_numpy(),
                        recent["away_score"].to_numpy()])
    # IRLS for Poisson GLM with intercept
    beta = np.zeros(2)
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(25):
        mu = np.exp(X @ beta)
        W = mu
        z = X @ beta + (y - mu) / mu
        XtW = X.T * W
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    return beta  # [alpha, slope]


def expected_goals(elo1, elo2, beta, home_adv=0.0):
    d = (elo1 + home_adv - elo2) / 400.0
    lam1 = math.exp(beta[0] + beta[1] * d)
    lam2 = math.exp(beta[0] - beta[1] * d)
    return lam1, lam2


def score_matrix(lam1, lam2, rho=DC_RHO):
    g = np.arange(MAX_GOALS + 1)
    p1 = np.exp(-lam1) * lam1 ** g / np.array([math.factorial(i) for i in g])
    p2 = np.exp(-lam2) * lam2 ** g / np.array([math.factorial(i) for i in g])
    M = np.outer(p1, p2)
    # Dixon-Coles adjustment on 0-0, 1-0, 0-1, 1-1
    M[0, 0] *= 1 - lam1 * lam2 * rho
    M[1, 0] *= 1 + lam2 * rho
    M[0, 1] *= 1 + lam1 * rho
    M[1, 1] *= 1 - rho
    return M / M.sum()


def predict(team1, team2, ratings, beta, home_adv=0.0):
    for t in (team1, team2):
        if t not in ratings:
            sys.exit(f"Unknown team: {t!r}. Check spelling against data/results.csv.")
    lam1, lam2 = expected_goals(ratings[team1], ratings[team2], beta, home_adv)
    M = score_matrix(lam1, lam2)
    p_win = np.tril(M, -1).sum()   # rows = team1 goals
    p_draw = np.trace(M)
    p_loss = np.triu(M, 1).sum()
    return lam1, lam2, p_win, p_draw, p_loss, M


def top_scorelines(M, n=5):
    flat = [(i, j, M[i, j]) for i in range(M.shape[0]) for j in range(M.shape[1])]
    return sorted(flat, key=lambda t: -t[2])[:n]


def _gated_ratings(t1, t2, ratings, conf_adjs, conf_threshold):
    """Return ratings dict copy with confederation adjustment applied to t1/t2,
    gated on their Elo gap exceeding conf_threshold."""
    if not conf_adjs:
        return ratings
    from .confederation_adj import apply_match_adj
    e1, e2 = apply_match_adj(
        ratings.get(t1, BASE_RATING), ratings.get(t2, BASE_RATING),
        conf_adjs.get(t1, 0.0), conf_adjs.get(t2, 0.0),
        conf_threshold)
    return {**ratings, t1: e1, t2: e2}


def cmd_match(args, ratings, beta, conf_adjs=None, conf_threshold=0):
    adv = HOME_ADV if args.home else 0.0
    r = _gated_ratings(args.team1, args.team2, ratings,
                       conf_adjs or {}, conf_threshold)
    lam1, lam2, w, d, l, M = predict(args.team1, args.team2, r, beta, adv)
    print(f"\n{args.team1} (Elo {r[args.team1]:.0f}) vs "
          f"{args.team2} (Elo {r[args.team2]:.0f})"
          f"{'  [home advantage: ' + args.team1 + ']' if args.home else '  [neutral]'}")
    print(f"Expected goals: {lam1:.2f} - {lam2:.2f}")
    print(f"  {args.team1} win: {w:6.1%}")
    print(f"  Draw:        {d:6.1%}")
    print(f"  {args.team2} win: {l:6.1%}")
    print(f"  BTTS:        {M[1:, 1:].sum():6.1%}")
    print("Most likely scorelines:")
    for i, j, p in top_scorelines(M):
        print(f"  {i}-{j}  {p:5.1%}")


def cmd_worldcup(upcoming, ratings, beta, conf_adjs=None, conf_threshold=0):
    wc = upcoming[upcoming["tournament"] == "FIFA World Cup"].copy()
    rows = []
    for r in wc.itertuples(index=False):
        adv = 0.0 if r.neutral else HOME_ADV
        gr = _gated_ratings(r.home_team, r.away_team, ratings,
                            conf_adjs or {}, conf_threshold)
        lam1, lam2, w, d, l, M = predict(r.home_team, r.away_team, gr, beta, adv)
        i, j, _ = top_scorelines(M, 1)[0]
        p_btts = M[1:, 1:].sum()
        rows.append({"date": r.date.date(), "home": r.home_team, "away": r.away_team,
                     "xg_home": round(lam1, 2), "xg_away": round(lam2, 2),
                     "p_home": round(w, 3), "p_draw": round(d, 3), "p_away": round(l, 3),
                     "p_btts": round(p_btts, 3),
                     "likely_score": f"{i}-{j}"})
    out = pd.DataFrame(rows).sort_values("date")
    dest = Path(__file__).resolve().parents[2] / "predictions_worldcup_2026.csv"
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {len(out)} predictions -> {dest.name}")


def cmd_backtest(played, beta):
    """Walk-forward: Elo is already point-in-time (pre-match). Score matches since 2024."""
    test = played[played["date"] >= "2024-01-01"]
    brier, naive_brier, correct, n = 0.0, 0.0, 0, 0
    for r in test.itertuples(index=False):
        adv = 0.0 if r.neutral else HOME_ADV
        lam1, lam2 = expected_goals(r.elo_h, r.elo_a, beta, adv)
        M = score_matrix(lam1, lam2)
        w, d, l = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
        actual = (1, 0, 0) if r.home_score > r.away_score else \
                 ((0, 1, 0) if r.home_score == r.away_score else (0, 0, 1))
        brier += (w - actual[0])**2 + (d - actual[1])**2 + (l - actual[2])**2
        naive_brier += (1/3 - actual[0])**2 + (1/3 - actual[1])**2 + (1/3 - actual[2])**2
        pred = ["H", "D", "A"][int(np.argmax([w, d, l]))]
        act = ["H", "D", "A"][int(np.argmax(actual))]
        correct += pred == act
        n += 1
    print(f"Backtest on {n} matches since 2024-01-01:")
    print(f"  Accuracy (3-way):     {correct/n:.1%}")
    print(f"  Brier score (model):  {brier/n:.4f}")
    print(f"  Brier score (chance): {naive_brier/n:.4f}  (lower is better)")


def main():
    ap = argparse.ArgumentParser(description="World Cup match predictor")
    ap.add_argument("team1", nargs="?")
    ap.add_argument("team2", nargs="?")
    ap.add_argument("--home", action="store_true", help="team1 has home advantage")
    ap.add_argument("--worldcup", action="store_true", help="predict all unplayed WC fixtures")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--ratings", action="store_true", help="show top 30 Elo ratings")
    ap.add_argument("--conf-adj", action="store_true",
                    help="apply confederation strength adjustment to Elo ratings "
                         "(fraction loaded from data/conf_adj.json; calibrate with "
                         "python confederation_adj.py --backtest)")
    args = ap.parse_args()

    played, upcoming = load_matches()
    ratings, played = compute_elo(played)
    beta = fit_goal_model(played)

    conf_adjs = {}
    conf_threshold = 0
    if getattr(args, "conf_adj", False):
        from .confederation_adj import (conf_adjustments, load_params,
                                       _wc_teams_2026)
        fraction, conf_threshold = load_params()
        wc_teams = _wc_teams_2026(played, upcoming)
        conf_adjs, global_mean, conf_means = conf_adjustments(
            ratings, wc_teams, fraction)
        print(f"[conf-adj] fraction={fraction:.2f}  threshold={conf_threshold}  "
              f"WC-field mean Elo={global_mean:.0f}  "
              f"confederations: " +
              ", ".join(f"{c} {v:+.0f}" for c, v in
                        sorted(conf_means.items(),
                               key=lambda kv: -(global_mean - kv[1]))))

    if args.ratings:
        top = sorted(ratings.items(), key=lambda kv: -kv[1])[:30]
        for i, (t, r) in enumerate(top, 1):
            print(f"{i:3d}. {t:<25s} {r:7.0f}")
    elif args.worldcup:
        cmd_worldcup(upcoming, ratings, beta, conf_adjs, conf_threshold)
    elif args.backtest:
        cmd_backtest(played, beta)
    elif args.team1 and args.team2:
        cmd_match(args, ratings, beta, conf_adjs, conf_threshold)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
