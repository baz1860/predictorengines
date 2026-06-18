#!/usr/bin/env python3
"""Large-sample draw-calibration test (the odds-free way to judge draw value).

Realized ROI on +edge draws is negative exactly when the model OVER-states the
draw probability: a "+edge" draw is one where p_model(draw) > market(draw), so if
the model is systematically too high on draws — especially the ones it likes most
— then that 'edge' is phantom and excluding draws is correct. With no historical
draw odds on disk for these competitions (network is blocked here), calibration is
the most powerful test available, and the on-disk walk-forward validation set is
huge (16.8k leak-free predictions).

Outputs:
  1. Draw calibration by model-probability bucket — overall club sample.
  2. Same, restricted to the 2025/26 Premier League season.
  3. WC2018 leak-free replay calibration (international model).
Outcome codes in validation_predictions.csv: 0=home, 1=draw, 2=away (verified:
predicted draw mean 26.0% vs actual 25.2%).
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
CS = HERE / "club_soccer" / "data"


def calib_table(df, pcol="p_draw", drawcol="is_draw", edges=(0, .15, .20, .25, .30, .35, 1.0)):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = df[(df[pcol] >= lo) & (df[pcol] < hi)]
        if g.empty:
            continue
        pred = g[pcol].mean()
        act = g[drawcol].mean()
        out.append(dict(bucket=f"{lo:.0%}-{hi:.0%}", n=len(g),
                        model_says=f"{pred:.1%}", actually_drew=f"{act:.1%}",
                        gap=f"{(act-pred)*100:+.1f}pp"))
    return pd.DataFrame(out)


def naive_value_roi(df, pcol="p_draw", drawcol="is_draw"):
    """ROI proxy with NO odds: bet draw at FAIR model odds (1/p_model) whenever the
    model's draw prob beats the sample's base draw rate (a stand-in for 'beats the
    market'). Returns realized ROI if those draws are graded at fair price 1/p.
    Negative => the model's preferred draws under-perform their own price."""
    base = df[drawcol].mean()
    pick = df[df[pcol] > base].copy()
    if pick.empty:
        return None
    fair_odds = 1.0 / pick[pcol]
    pnl = np.where(pick[drawcol] == 1, fair_odds - 1.0, -1.0)
    return len(pick), pick[drawcol].mean(), pnl.sum() / len(pick)


print("=" * 70)
print("1) DRAW CALIBRATION — full club validation sample (leak-free walk-forward)")
print("=" * 70)
v = pd.read_csv(CS / "validation_predictions.csv")
v["date"] = pd.to_datetime(v["date"])
v["is_draw"] = (v.actual == 1).astype(int)
print(f"n = {len(v):,} matches | model draw mean {v.p_draw.mean():.1%} "
      f"| actual draw rate {v.is_draw.mean():.1%}\n")
print(calib_table(v).to_string(index=False))
r = naive_value_roi(v)
print(f"\nFair-price ROI proxy on model's preferred draws (p_draw > base rate): "
      f"{r[0]} bets, drew {r[1]:.1%}, ROI {r[2]:+.1%}")

print("\n" + "=" * 70)
print("2) DRAW CALIBRATION — 2025/26 Premier League only")
print("=" * 70)
fx = pd.read_csv(CS / "fixtures.csv")
epl = fx[(fx.competition == "Premier League") & (fx.season == 2025)
         & (fx.status == "FT")].copy()
epl["date"] = pd.to_datetime(epl["date"])
key = lambda d: d.home.astype(str) + "|" + d.away.astype(str) + "|" + d.date.dt.date.astype(str)
v["k"] = key(v)
epl["k"] = key(epl)
m = v[v.k.isin(set(epl.k))].copy()
if len(m):
    print(f"matched {len(m)} of {len(epl)} EPL 2025/26 played fixtures "
          f"to model predictions")
    print(f"model draw mean {m.p_draw.mean():.1%} | actual draw rate "
          f"{m.is_draw.mean():.1%}\n")
    print(calib_table(m, edges=(0, .22, .27, .32, 1.0)).to_string(index=False))
    r2 = naive_value_roi(m)
    if r2:
        print(f"\nFair-price ROI proxy on preferred draws: {r2[0]} bets, "
              f"drew {r2[1]:.1%}, ROI {r2[2]:+.1%}")
else:
    print("No EPL 2025/26 fixtures matched the validation set by name+date.")

print("\n" + "=" * 70)
print("3) WC2018 — leak-free international replay calibration")
print("=" * 70)
import sys
sys.path.insert(0, str(HERE))
from engines.worldcup.predictor import (load_matches, compute_elo, fit_goal_model,
                       expected_goals, DC_RHO)
from engines.worldcup.dixoncoles import fit_dc, outcome_probs

played, _ = load_matches()
ratings, played = compute_elo(played)
CUT = "2018-06-14"   # WC2018 opened 14 June 2018
train = played[played["date"] < CUT]
beta = fit_goal_model(train)
dc = fit_dc(train, anchor=CUT, verbose=False)
ratings_cut, _ = compute_elo(train)

# WC2018 group+knockout matches from results.csv
wc = played[(played["date"] >= "2018-06-14") & (played["date"] <= "2018-07-15")
            & (played["tournament"].str.contains("World Cup", case=False, na=False))].copy()
rows = []
for r in wc.itertuples(index=False):
    h, a = r.home_team, r.away_team
    if h not in ratings_cut or a not in ratings_cut:
        continue
    le = expected_goals(ratings_cut[h], ratings_cut[a], beta, 0.0)
    ld = dc.lambdas(h, a)
    pe = np.array(outcome_probs(*le, DC_RHO)[:3])
    pdc = np.array(outcome_probs(*ld, dc.rho)[:3])
    p = (pe + pdc) / 2
    drew = int(r.home_score == r.away_score)
    rows.append(dict(match=f"{h} v {a}", p_draw=float(p[1]), is_draw=drew))
w = pd.DataFrame(rows)
print(f"WC2018 matches modelled: {len(w)} | model draw mean {w.p_draw.mean():.1%} "
      f"| actual draw rate {w.is_draw.mean():.1%}\n")
print(calib_table(w, edges=(0, .25, .30, 1.0)).to_string(index=False))
r3 = naive_value_roi(w)
if r3:
    print(f"\nFair-price ROI proxy on preferred draws: {r3[0]} bets, "
          f"drew {r3[1]:.1%}, ROI {r3[2]:+.1%}")
w.to_csv(HERE / "wc2018_draw_detail.csv", index=False)
