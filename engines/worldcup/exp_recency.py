"""Quantify how much old international data helps the 1X2 goal model.
Compares equal-weight-all-history vs time-decay vs hard cutoffs on held-out
3-way log-loss (competitive matches 2018+). Run: python -m engines.worldcup.exp_recency
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import engines.worldcup.predictor as P
from engines.worldcup.features_1x2 import (build_matrix, fit_poisson, decay_weights,
                                           _ctx_factory, _logloss_1x2, REGISTRY)

TRAIN_TO = "2018-01-01"
feats = [f for f in REGISTRY if f.name in ("intercept", "elo_diff")]

played, _ = P.load_matches()
_, played = P.compute_elo(played)
train = played[played["date"] < TRAIN_TO]
eval_ = played[(played["date"] >= TRAIN_TO) & (played["tournament"] != "Friendly")]
X, y, dates = build_matrix(train, feats, _ctx_factory)
dts = pd.to_datetime(pd.Series(dates))
age = (pd.Timestamp(TRAIN_TO) - dts).dt.days.to_numpy() / 365.25

def brier_acc(theta):
    br = acc = n = 0
    for r in eval_.itertuples(index=False):
        from engines.worldcup.features_1x2 import design_row
        lh = float(np.exp(design_row(r.home_team, r.away_team, r.date, True,  _ctx_factory(r, True),  feats) @ theta))
        la = float(np.exp(design_row(r.away_team, r.home_team, r.date, False, _ctx_factory(r, False), feats) @ theta))
        M = P.score_matrix(lh, la)
        p = [np.tril(M,-1).sum(), np.trace(M), np.triu(M,1).sum()]
        k = 0 if r.home_score>r.away_score else (1 if r.home_score==r.away_score else 2)
        br += sum((p[j]-(1 if j==k else 0))**2 for j in range(3))
        acc += int(np.argmax(p)==k); n+=1
    return br/n, acc/n

def run(label, w, eff_n):
    theta = fit_poisson(X, y, w)
    ll = _logloss_1x2(eval_, theta, feats, _ctx_factory)
    br, acc = brier_acc(theta)
    print(f"  {label:22s}  logloss {ll:.4f}   brier {br:.4f}   acc {acc:5.1%}   "
          f"eff.matches {eff_n:>7,.0f}   theta {np.round(theta,3)}")

print(f"Train < {TRAIN_TO}  ({len(train):,} matches)   eval = competitive 2018+ ({len(eval_):,})\n")
print("Scheme                    held-out 1X2")
# equal weight, all history
run("equal (all history)", None, len(train))
# time-decay half-lives
for hl in (20, 12, 8, 6, 4):
    w = decay_weights(dates, TRAIN_TO, hl)
    run(f"half-life {hl}y", w, w.sum()/2)
# hard cutoffs (equal weight within window)
for cut in (2000, 2010, 2014):
    w = (age <= (pd.Timestamp(TRAIN_TO).year - cut)).astype(float)
    run(f"cutoff >={cut} only", w, w.sum()/2)
