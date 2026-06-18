#!/usr/bin/env python3
"""Is the draw over-predicted specifically in LOPSIDED matches?

The WC2022 +edge draws that lost were longshot draws (avg odds 5.72) — i.e.
matches with a clear favourite where the model still gave the draw ~25-31%. The
full-sample calibration is fine, so the question is whether a bias hides in the
mismatch subset. Buckets the same 7.7k leak-free out-of-sample internationals by
favourite strength (max of home/away win prob) and checks draw calibration in
each, at the incumbent blend rho.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from engines.worldcup.predictor import (load_matches, compute_elo, fit_goal_model,
                       expected_goals, score_matrix, HOME_ADV, DC_RHO)
from engines.worldcup.dixoncoles import fit_dc


def probs_for(lam1, lam2, rho):
    M = score_matrix(lam1, lam2, rho)
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()


played, _ = load_matches()
_, played = compute_elo(played)
train = played[played["date"] < "2017-01-01"]
beta = fit_goal_model(train)
dc = fit_dc(train, anchor="2017-01-01", verbose=False)
ratings_cut, _ = compute_elo(train)
test = played[(played["date"] >= "2017-01-01") & (played["date"] <= "2024-12-31")]

rows = []
for r in test.itertuples(index=False):
    h, a = r.home_team, r.away_team
    if h not in dc.att or a not in dc.att or h not in ratings_cut or a not in ratings_cut:
        continue
    hh = 0.0 if r.neutral else 1.0
    pe = np.array(probs_for(*expected_goals(ratings_cut[h], ratings_cut[a], beta, hh*HOME_ADV), DC_RHO))
    pdc = np.array(probs_for(*dc.lambdas(h, a, h1=hh), dc.rho))
    p = (pe + pdc) / 2
    drew = int(r.home_score == r.away_score)
    rows.append((max(p[0], p[2]), p[1], drew))
d = pd.DataFrame(rows, columns=["fav_prob", "p_draw", "is_draw"])

print(f"n = {len(d):,} out-of-sample internationals (blend, incumbent rho)\n")
print(f"{'favourite strength':>22} {'n':>6} {'model_draw':>11} {'actual_draw':>12} {'gap':>7}")
for lo, hi, lbl in [(0, .45, "even (<45% fav)"), (.45, .55, "lean (45-55%)"),
                    (.55, .65, "clear (55-65%)"), (.65, .75, "strong (65-75%)"),
                    (.75, 1.01, "lopsided (>75%)")]:
    g = d[(d.fav_prob >= lo) & (d.fav_prob < hi)]
    if g.empty:
        continue
    mp, ap = g.p_draw.mean(), g.is_draw.mean()
    print(f"{lbl:>22} {len(g):>6} {mp:>10.1%} {ap:>11.1%} {(ap-mp)*100:>+6.1f}pp")

# the specific cell that matters: lopsided games, model still likes the draw
print("\nLopsided games (fav>70%) where model gives draw >= 22%:")
sub = d[(d.fav_prob > .70) & (d.p_draw >= .22)]
print(f"  n={len(sub)} | model draw {sub.p_draw.mean():.1%} | actual {sub.is_draw.mean():.1%} "
      f"| gap {(sub.is_draw.mean()-sub.p_draw.mean())*100:+.1f}pp")
