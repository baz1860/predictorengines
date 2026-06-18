#!/usr/bin/env python3
"""Find the Dixon-Coles rho that best calibrates draws without hurting accuracy.

Negative rho moves probability into the 0-0 and 1-1 cells (draws). The Elo/Poisson
half of the blend uses a hardcoded DC_RHO = -0.10, while the DC half fits rho from
data (currently -0.045) — so the blend applies a stronger draw correction than the
data supports, which is a prime suspect for the draw over-prediction seen in the
WC2018/WC2022/club calibration.

This fits the model leak-free at a cutoff, then sweeps rho over a large
out-of-sample international test set, reporting for each rho:
  - draw calibration (mean predicted draw vs actual draw rate, overall + high bucket)
  - overall Brier and multiclass log-loss (guards against a draw fix that hurts 1X2)
Then prints WC2018 / WC2022 draw behaviour at the incumbent vs best rho.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from engines.worldcup.predictor import (load_matches, compute_elo, fit_goal_model,
                       expected_goals, score_matrix, HOME_ADV)
from engines.worldcup.dixoncoles import fit_dc

EPS = 1e-12


def probs_for(lam1, lam2, rho):
    M = score_matrix(lam1, lam2, rho)
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()


def build_test(cutoff, test_start, test_end):
    played, _ = load_matches()
    ratings_full, played = compute_elo(played)
    train = played[played["date"] < cutoff]
    beta = fit_goal_model(train)
    dc = fit_dc(train, anchor=cutoff, verbose=False)
    ratings_cut, _ = compute_elo(train)
    test = played[(played["date"] >= test_start) & (played["date"] <= test_end)].copy()
    rows = []
    for r in test.itertuples(index=False):
        h, a = r.home_team, r.away_team
        if h not in dc.att or a not in dc.att or h not in ratings_cut or a not in ratings_cut:
            continue
        hh = 0.0 if r.neutral else 1.0
        le1, le2 = expected_goals(ratings_cut[h], ratings_cut[a], beta, hh * HOME_ADV)
        ld1, ld2 = dc.lambdas(h, a, h1=hh)
        outcome = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        rows.append((le1, le2, ld1, ld2, outcome))
    return pd.DataFrame(rows, columns=["le1", "le2", "ld1", "ld2", "outcome"]), dc.rho


def evaluate(df, rho):
    P = np.zeros((len(df), 3))
    for i, r in enumerate(df.itertuples(index=False)):
        pe = np.array(probs_for(r.le1, r.le2, rho))
        pdc = np.array(probs_for(r.ld1, r.ld2, rho))
        P[i] = (pe + pdc) / 2
    y = df.outcome.to_numpy()
    onehot = np.zeros((len(df), 3)); onehot[np.arange(len(df)), y] = 1
    brier = np.mean(np.sum((P - onehot) ** 2, axis=1))
    logloss = -np.mean(np.log(P[np.arange(len(df)), y] + EPS))
    acc = np.mean(P.argmax(1) == y)
    pdraw = P[:, 1]
    is_draw = (y == 1).astype(float)
    # high-draw bucket (model's most draw-confident games — where 'value' draws live)
    hi = pdraw >= 0.28
    hi_pred = pdraw[hi].mean() if hi.any() else np.nan
    hi_act = is_draw[hi].mean() if hi.any() else np.nan
    return dict(rho=rho, draw_pred=pdraw.mean(), draw_act=is_draw.mean(),
                brier=brier, logloss=logloss, acc=acc,
                hi_n=int(hi.sum()), hi_pred=hi_pred, hi_act=hi_act)


print("Building leak-free out-of-sample international test set "
      "(fit < 2017, test 2017-2024)...")
df, fitted_rho = build_test("2017-01-01", "2017-01-01", "2024-12-31")
print(f"test matches: {len(df):,} | DC rho fitted at cutoff: {fitted_rho:.3f}\n")

print(f"{'rho':>6} {'draw_pred':>10} {'draw_act':>9} {'gap':>7} "
      f"{'hi_pred':>8} {'hi_act':>7} {'hi_gap':>7} {'brier':>8} {'logloss':>8} {'acc':>6}")
results = []
for rho in [0.0, -0.02, -0.045, -0.06, -0.08, -0.10, -0.12]:
    m = evaluate(df, rho)
    results.append(m)
    hg = (m['hi_act'] - m['hi_pred']) * 100 if not np.isnan(m['hi_pred']) else float('nan')
    print(f"{rho:>6.3f} {m['draw_pred']:>9.1%} {m['draw_act']:>8.1%} "
          f"{(m['draw_act']-m['draw_pred'])*100:>+6.1f} {m['hi_pred']:>7.1%} "
          f"{m['hi_act']:>6.1%} {hg:>+6.1f} {m['brier']:>8.4f} {m['logloss']:>8.4f} "
          f"{m['acc']:>5.1%}")

best = min(results, key=lambda m: m["logloss"])
print(f"\nBest log-loss at rho = {best['rho']:.3f} "
      f"(incumbent blend uses Elo rho -0.10 + DC rho {fitted_rho:.3f})")
print(f"Best Brier   at rho = {min(results, key=lambda m: m['brier'])['rho']:.3f}")
# draw-calibration optimum: smallest |gap| in the high bucket
cal = min((m for m in results if not np.isnan(m['hi_pred'])),
          key=lambda m: abs(m['hi_act'] - m['hi_pred']))
print(f"Best high-bucket draw calibration at rho = {cal['rho']:.3f} "
      f"(pred {cal['hi_pred']:.1%} vs actual {cal['hi_act']:.1%})")
