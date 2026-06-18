#!/usr/bin/env python3
"""How would the engine have done at World Cup 2022?

Leak-free: the Elo->goals map and the Dixon-Coles fit use only matches
played before the tournament (Nov 20, 2022); Elo ratings are pre-match
by construction. Scores all 64 matches on the 90/120-minute result
(the dataset records draws before shootouts, so 3-way is well-defined).
"""
import numpy as np
import pandas as pd

from engines.worldcup.predictor import (load_matches, compute_elo, fit_goal_model,
                       expected_goals, HOME_ADV, DC_RHO)
from engines.worldcup.dixoncoles import fit_dc, outcome_probs

START, END = "2022-11-20", "2022-12-19"

played, _ = load_matches()
ratings, played = compute_elo(played)
train = played[played["date"] < START]
test = played[(played["tournament"] == "FIFA World Cup") &
              (played["date"].between(START, END))]
print(f"Training on {len(train)} matches before {START}; "
      f"testing on {len(test)} WC 2022 matches.")

beta = fit_goal_model(train)
dc = fit_dc(train, anchor=START, verbose=False)

stats = {k: [0.0, 0, 0.0] for k in ("elo", "dc", "blend")}  # brier, acc, logloss
rows = []
for r in test.itertuples(index=False):
    h = 0.0 if r.neutral else 1.0   # Qatar's own matches are non-neutral
    actual_idx = (0 if r.home_score > r.away_score
                  else (1 if r.home_score == r.away_score else 2))
    actual = np.zeros(3); actual[actual_idx] = 1

    le = expected_goals(r.elo_h, r.elo_a, beta, h * HOME_ADV)
    ld = dc.lambdas(r.home_team, r.away_team, h1=h)
    probs = {}
    pw, pd_, pl, _ = outcome_probs(*le, DC_RHO); probs["elo"] = np.array([pw, pd_, pl])
    pw, pd_, pl, _ = outcome_probs(*ld, dc.rho); probs["dc"] = np.array([pw, pd_, pl])
    probs["blend"] = (probs["elo"] + probs["dc"]) / 2

    for k, p in probs.items():
        stats[k][0] += np.sum((p - actual) ** 2)
        stats[k][1] += int(np.argmax(p) == actual_idx)
        stats[k][2] += -np.log(max(p[actual_idx], 1e-12))
    rows.append({"date": r.date.date(), "match": f"{r.home_team} {int(r.home_score)}"
                 f"-{int(r.away_score)} {r.away_team}",
                 "p_actual_blend": round(probs["blend"][actual_idx], 3),
                 "p_top_blend": round(float(probs["blend"].max()), 3),
                 "predicted": ["home", "draw", "away"][int(np.argmax(probs['blend']))],
                 "outcome": ["home", "draw", "away"][actual_idx]})

n = len(test)
print(f"\n{'model':<12}{'accuracy':>10}{'Brier':>9}{'log-loss':>10}"
      f"   (chance: 33.3%, 0.667, 1.099)")
for k, label in (("elo", "Elo+Poisson"), ("dc", "Dixon-Coles"),
                 ("blend", "50/50 blend")):
    br, acc, ll = stats[k]
    print(f"{label:<12}{acc / n:>9.1%}{br / n:>9.4f}{ll / n:>10.4f}")

df = pd.DataFrame(rows)
df.to_csv("wc2022_backtest.csv", index=False)
print("\nBiggest surprises (lowest blend probability on the actual outcome):")
print(df.nsmallest(6, "p_actual_blend").to_string(index=False))
print("\nMost confident correct calls:")
ok = df[df.predicted == df.outcome]
print(ok.nlargest(5, "p_actual_blend").to_string(index=False))
print(f"\nPer-match detail -> wc2022_backtest.csv")
