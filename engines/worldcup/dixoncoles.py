#!/usr/bin/env python3
"""Dixon-Coles attack/defense model.

Each team gets an attack and a defense parameter, fitted by maximum
likelihood with exponential time-decay (recent matches count more):

    lambda_1 = exp(mu + att_1 - def_2 + gamma * home_1)
    lambda_2 = exp(mu + att_2 - def_1 + gamma * home_2)

This separates HOW teams are strong (scoring vs conceding), unlike the
single Elo number, and fits home advantage (gamma) and the low-score
correlation (rho) from data instead of using fixed constants.

Usage:
  python dixoncoles.py --fit              # fit on all data, save params
  python dixoncoles.py "Brazil" "Morocco" # predict (auto-fits if needed)
  python dixoncoles.py --backtest         # DC vs Elo on matches since 2024
  python dixoncoles.py --ratings          # top 30 by attack - opponent-proof
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .predictor import (load_matches, compute_elo, fit_goal_model,
                       expected_goals, score_matrix, HOME_ADV, DC_RHO)

PARAMS_FILE = Path(__file__).resolve().parents[2] / "data" / "dc_params.json"
HALF_LIFE_YEARS = 2.5     # weight halves every 2.5 years
WINDOW_YEARS = 12         # ignore matches older than this (weight < 4%)
ADAM_ITERS = 2500
ADAM_LR = 0.05
L2_REG = 0.001   # shrinks att/def toward 0; mainly disciplines teams with
                 # few matches (their data can't outweigh the prior)


def _decay_weights(dates, anchor):
    age_years = (anchor - dates).dt.days.to_numpy() / 365.25
    return np.exp(-math.log(2) / HALF_LIFE_YEARS * age_years)


def fit_dc(played, anchor=None, verbose=True):
    """Weighted Poisson ML fit via Adam, then 1-D grid search for rho."""
    anchor = pd.Timestamp(anchor) if anchor else played["date"].max()
    df = played[(played["date"] <= anchor) &
                (played["date"] >= anchor - pd.DateOffset(years=WINDOW_YEARS))]
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    T, N = len(teams), len(df)

    hi = df["home_team"].map(idx).to_numpy()
    ai = df["away_team"].map(idx).to_numpy()
    yh = df["home_score"].to_numpy(float)
    ya = df["away_score"].to_numpy(float)
    home = (~df["neutral"].astype(bool)).to_numpy(float)
    w = _decay_weights(df["date"], anchor)

    # params: [att (T), def (T), mu, gamma]
    p = np.zeros(2 * T + 2)
    p[2 * T] = math.log(1.3)
    p[2 * T + 1] = 0.25
    m = np.zeros_like(p); v = np.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-8

    for it in range(1, ADAM_ITERS + 1):
        att, dfn, mu, gamma = p[:T], p[T:2 * T], p[2 * T], p[2 * T + 1]
        lh = np.exp(mu + att[hi] - dfn[ai] + gamma * home)
        la = np.exp(mu + att[ai] - dfn[hi])
        rh, ra = w * (yh - lh), w * (ya - la)

        g = np.zeros_like(p)
        np.add.at(g, hi, rh); np.add.at(g, ai, ra)              # attack
        np.add.at(g, T + ai, -rh); np.add.at(g, T + hi, -ra)    # defense
        g[2 * T] = rh.sum() + ra.sum()                          # mu
        g[2 * T + 1] = (rh * home).sum()                        # gamma
        g /= w.sum()
        g[:2 * T] -= L2_REG * p[:2 * T]                         # L2 prior

        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        p += ADAM_LR * (m / (1 - b1**it)) / (np.sqrt(v / (1 - b2**it)) + eps)

        if verbose and it % 500 == 0:
            ll = np.sum(w * (yh * np.log(lh) - lh + ya * np.log(la) - la))
            print(f"  iter {it:5d}  weighted LL/match {ll / w.sum():.4f}")

    att, dfn, mu, gamma = p[:T], p[T:2 * T], p[2 * T], p[2 * T + 1]
    # identifiability: center att/def, absorb shift into mu
    ca, cd = att.mean(), dfn.mean()
    att, dfn, mu = att - ca, dfn - cd, mu + ca - cd

    # fit rho on the Dixon-Coles tau term, lambdas held fixed
    lh = np.exp(mu + att[hi] - dfn[ai] + gamma * home)
    la = np.exp(mu + att[ai] - dfn[hi])
    is00 = (yh == 0) & (ya == 0); is10 = (yh == 1) & (ya == 0)
    is01 = (yh == 0) & (ya == 1); is11 = (yh == 1) & (ya == 1)
    best_rho, best_ll = 0.0, -np.inf
    for rho in np.arange(-0.25, 0.151, 0.005):
        t = np.ones(N)
        t[is00] = 1 - lh[is00] * la[is00] * rho
        t[is10] = 1 + la[is10] * rho
        t[is01] = 1 + lh[is01] * rho
        t[is11] = 1 - rho
        if (t <= 0).any():
            continue
        ll = np.sum(w * np.log(t))
        if ll > best_ll:
            best_ll, best_rho = ll, float(rho)

    if verbose:
        print(f"  fitted: mu={mu:.3f} (base xg {math.exp(mu):.2f}), "
              f"home gamma={gamma:.3f} (x{math.exp(gamma):.2f}), rho={best_rho:.3f}")
    return DCModel(dict(zip(teams, att)), dict(zip(teams, dfn)),
                   float(mu), float(gamma), best_rho)


class DCModel:
    def __init__(self, att, dfn, mu, gamma, rho):
        self.att, self.dfn = att, dfn
        self.mu, self.gamma, self.rho = mu, gamma, rho

    def lambdas(self, t1, t2, h1=0, h2=0):
        for t in (t1, t2):
            if t not in self.att:
                sys.exit(f"Team {t!r} not in fitted model.")
        l1 = math.exp(self.mu + self.att[t1] - self.dfn[t2] + self.gamma * h1)
        l2 = math.exp(self.mu + self.att[t2] - self.dfn[t1] + self.gamma * h2)
        return l1, l2

    def save(self, path=PARAMS_FILE):
        path.write_text(json.dumps({
            "mu": self.mu, "gamma": self.gamma, "rho": self.rho,
            "teams": {t: [self.att[t], self.dfn[t]] for t in self.att}}))

    @classmethod
    def load(cls, path=PARAMS_FILE):
        d = json.loads(path.read_text())
        att = {t: v[0] for t, v in d["teams"].items()}
        dfn = {t: v[1] for t, v in d["teams"].items()}
        return cls(att, dfn, d["mu"], d["gamma"], d["rho"])

    @classmethod
    def load_or_fit(cls):
        if PARAMS_FILE.exists():
            return cls.load()
        played, _ = load_matches()
        model = fit_dc(played, verbose=False)
        model.save()
        return model


def build_sources(model="blend"):
    """Lambda sources for downstream tools (simulate.py, edge.py).

    Returns ([(predict_fn, rho), ...], elo_ratings). Each predict_fn maps
    (team1, team2, h1, h2) -> (lambda1, lambda2) where h1/h2 are 1.0 when
    that side has home advantage."""
    played, _ = load_matches()
    ratings, played = compute_elo(played)
    sources = []
    if model in ("elo", "blend"):
        beta = fit_goal_model(played)
        sources.append((lambda t1, t2, h1=0.0, h2=0.0: expected_goals(
            ratings[t1], ratings[t2], beta, (h1 - h2) * HOME_ADV), DC_RHO))
    if model in ("dc", "blend"):
        dc = DCModel.load_or_fit()
        sources.append((dc.lambdas, dc.rho))
    if not sources:
        sys.exit(f"Unknown model {model!r}: use elo, dc, or blend.")
    return sources, ratings


def outcome_probs(lam1, lam2, rho):
    M = score_matrix(lam1, lam2, rho)
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum(), M


def cmd_backtest():
    """Fit both models on pre-2024 data only; score 2024+ matches."""
    played, _ = load_matches()
    ratings, played = compute_elo(played)          # elo_h/elo_a are pre-match
    cutoff = pd.Timestamp("2024-01-01")
    train = played[played["date"] < cutoff]
    test = played[played["date"] >= cutoff]
    beta = fit_goal_model(train)                    # Elo->goals map, no leakage
    print(f"Fitting Dixon-Coles on {len(train)} matches before {cutoff.date()}...")
    dc = fit_dc(train, anchor=cutoff)

    stats = {"elo": [0.0, 0, 0], "dc": [0.0, 0, 0],
             "blend": [0.0, 0, 0]}  # brier, correct, n
    skipped = 0
    for r in test.itertuples(index=False):
        if r.home_team not in dc.att or r.away_team not in dc.att:
            skipped += 1
            continue
        h = 0.0 if r.neutral else 1.0
        actual = np.zeros(3)
        actual[0 if r.home_score > r.away_score else
               (1 if r.home_score == r.away_score else 2)] = 1
        # Elo model (point-in-time pre-match ratings)
        le1, le2 = expected_goals(r.elo_h, r.elo_a, beta, h * HOME_ADV)
        # DC model (static params as of cutoff)
        ld1, ld2 = dc.lambdas(r.home_team, r.away_team, h1=h)
        all_probs = {}
        for key, (l1, l2, rho) in (("elo", (le1, le2, DC_RHO)),
                                   ("dc", (ld1, ld2, dc.rho))):
            pw, pd_, pl, _ = outcome_probs(l1, l2, rho)
            all_probs[key] = np.array([pw, pd_, pl])
        all_probs["blend"] = (all_probs["elo"] + all_probs["dc"]) / 2
        for key, probs in all_probs.items():
            stats[key][0] += np.sum((probs - actual) ** 2)
            stats[key][1] += int(np.argmax(probs) == np.argmax(actual))
            stats[key][2] += 1

    print(f"\nOut-of-sample comparison, {stats['dc'][2]} matches since "
          f"{cutoff.date()} ({skipped} skipped, teams unseen in training):\n")
    print(f"{'model':<12}{'accuracy':>10}{'Brier':>10}")
    for key, label in (("elo", "Elo+Poisson"), ("dc", "Dixon-Coles"),
                       ("blend", "50/50 blend")):
        br, c, n = stats[key]
        print(f"{label:<12}{c / n:>9.1%}{br / n:>10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("team1", nargs="?")
    ap.add_argument("team2", nargs="?")
    ap.add_argument("--home", action="store_true", help="team1 at home")
    ap.add_argument("--fit", action="store_true", help="refit and save params")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--ratings", action="store_true",
                    help="top 30 teams by net strength (att + def)")
    args = ap.parse_args()

    if args.backtest:
        cmd_backtest()
        return
    if args.fit:
        played, _ = load_matches()
        model = fit_dc(played)
        model.save()
        print(f"Saved -> {PARAMS_FILE.name} ({len(model.att)} teams)")
        return

    model = DCModel.load_or_fit()
    if args.ratings:
        net = {t: model.att[t] + model.dfn[t] for t in model.att}
        for i, (t, s) in enumerate(sorted(net.items(), key=lambda kv: -kv[1])[:30], 1):
            print(f"{i:3d}. {t:<25s} net {s:+.2f}  "
                  f"att {model.att[t]:+.2f}  def {model.dfn[t]:+.2f}")
    elif args.team1 and args.team2:
        l1, l2 = model.lambdas(args.team1, args.team2,
                               h1=1.0 if args.home else 0.0)
        w, d, l, M = outcome_probs(l1, l2, model.rho)
        print(f"\n{args.team1} vs {args.team2}"
              f"{' [home: ' + args.team1 + ']' if args.home else ' [neutral]'}")
        print(f"Expected goals: {l1:.2f} - {l2:.2f}")
        print(f"  {args.team1} win: {w:6.1%}\n  Draw:        {d:6.1%}\n"
              f"  {args.team2} win: {l:6.1%}")
        flat = sorted(((i, j, M[i, j]) for i in range(M.shape[0])
                       for j in range(M.shape[1])), key=lambda t: -t[2])[:5]
        print("Most likely scorelines:")
        for i, j, pr in flat:
            print(f"  {i}-{j}  {pr:5.1%}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
