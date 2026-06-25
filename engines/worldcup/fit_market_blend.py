"""Refit the logit-space 1X2 market blend on ALL available World Cup odds
(wc2018 + wc2022) instead of the 64-game single-tournament fit in market_blend.json.

Blend (multiclass geometric pooling, = logit blend):
    p_blend ∝ p_model^(1-w) * p_market^w   (renormalised)

Picks w by leave-one-tournament-out CV, reports model-only / market-only / blend
held-out log-loss, and writes data/market_blend.json (active only if blend wins).
Run: python -m engines.worldcup.fit_market_blend
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import engines.worldcup.predictor as P

NAME = {"USA": "United States", "South Korea": "South Korea", "China PR": "China"}
def nm(x): return NAME.get(x, x)

def frac_to_dec(s):
    s = str(s).strip()
    if "/" in s:
        a, b = s.split("/"); return 1 + float(a)/float(b)
    return float(s)

def implied_probs(oh, od, oa):
    raw = np.array([1/oh, 1/od, 1/oa])
    return raw / raw.sum()            # de-vig (proportional)

def model_probs(elo_h, elo_a, beta, neutral):
    adv = 0.0 if neutral else P.HOME_ADV
    l1, l2 = P.expected_goals(elo_h, elo_a, beta, adv)
    M = P.score_matrix(l1, l2)
    return np.array([np.tril(M,-1).sum(), np.trace(M), np.triu(M,1).sum()])

def blend(pm, pk, w):
    p = (pm**(1-w)) * (pk**w)
    return p / p.sum()

def logloss(rows, w=None, which="blend"):
    ll = 0.0
    for pm, pk, k in rows:
        if which=="model": p=pm
        elif which=="market": p=pk
        else: p=blend(pm,pk,w)
        ll += -np.log(max(p[k], 1e-12))
    return ll/len(rows)

# ---- build (model, market, outcome) rows per tournament, point-in-time ----
played, _ = P.load_matches()
_, played = P.compute_elo(played)
OUT = {"H":0,"D":1,"A":2}
TOURN = {2018: ("data/wc2018_odds.csv","2018-06-01"),
         2022: ("data/wc2022_odds.csv","2022-11-01")}
HOST = {2018:"Russia", 2022:"Qatar"}

bank = {}
for yr,(fp,asof) in TOURN.items():
    ratings,_ = P.compute_elo(played[played["date"] < asof])     # ratings as of pre-tournament
    beta = P.fit_goal_model(played[played["date"] < asof])       # no leakage
    odds = pd.read_csv(ROOT/fp)
    rows=[]; missed=[]
    for r in odds.itertuples(index=False):
        h,a = nm(r.home), nm(r.away)
        if h not in ratings or a not in ratings: missed.append((h,a)); continue
        neutral = (h != HOST[yr])
        pm = model_probs(ratings[h], ratings[a], beta, neutral)
        pk = implied_probs(frac_to_dec(r.odds_home), frac_to_dec(r.odds_draw), frac_to_dec(r.odds_away))
        res = str(r.result90).strip().lower()
        k = {"home":0,"draw":1,"away":2,"h":0,"d":1,"a":2}[res]
        rows.append((pm, pk, k))
    bank[yr]=rows
    if missed: print(f"[{yr}] unmatched names skipped: {missed}")

allrows=[r for rs in bank.values() for r in rs]
print(f"\nMatches used: {len(allrows)}  (2018:{len(bank[2018])}, 2022:{len(bank[2022])})\n")

# grid of w, full-sample + leave-one-tournament-out CV
grid=np.round(np.arange(0,1.01,0.05),2)
full=[(w, logloss(allrows,w)) for w in grid]
best_w_full=min(full,key=lambda t:t[1])[0]

# CV: train w on one tournament, test on the other
def best_w(rows):
    return min(((w,logloss(rows,w)) for w in grid), key=lambda t:t[1])[0]
cv_rows=[]
for test_yr in (2018,2022):
    train_yr=2022 if test_yr==2018 else 2018
    w=best_w(bank[train_yr])
    cv_rows.append((test_yr,w,logloss(bank[test_yr],w)))

print("held-out log-loss (lower=better):")
print(f"  model only   {logloss(allrows,which='model'):.4f}")
print(f"  market only  {logloss(allrows,which='market'):.4f}")
print(f"  blend w*={best_w_full:.2f} (full) {logloss(allrows,best_w_full):.4f}")
print("\nleave-one-tournament-out CV:")
for yr,w,ll in cv_rows:
    print(f"  train {'2022' if yr==2018 else '2018'} -> test {yr}:  w={w:.2f}  test logloss {ll:.4f}")
cv_mean=np.mean([ll for _,_,ll in cv_rows]); cv_w=np.mean([w for _,w,_ in cv_rows])
print(f"  CV mean logloss {cv_mean:.4f}   mean w {cv_w:.2f}")

ll_model=logloss(allrows,which='model'); ll_blend=logloss(allrows,best_w_full)
# A near-1.0 optimal weight means the market dominates and the model adds no edge:
# that is a NO-EDGE result, not a deployable blend (w=1.0 = copy the market + pay vig).
degenerate = best_w_full >= 0.85
out={"w":round(float(best_w_full),3),"w_cv":round(float(cv_w),3),
     "n":len(allrows),"source":"WC2018+WC2022 (de-vigged 1X2), logit/geometric blend, LOTO-CV",
     "logloss_model_only":round(ll_model,4),
     "logloss_market_only":round(logloss(allrows,which='market'),4),
     "logloss_blend":round(ll_blend,4),
     "cv_mean_logloss":round(float(cv_mean),4),
     "active": bool((ll_blend < ll_model - 1e-4) and not degenerate),
     "note": ("optimal w>=0.85: market dominates, model adds no WC 1X2 edge; "
              "not deployable (w->1 = copy the market). Seek softer markets."
              if degenerate else "blend beats model-only; safe to activate.")}
(Path(ROOT/"data/market_blend.json")).write_text(json.dumps(out,indent=2))
print("\nwrote data/market_blend.json:",json.dumps(out))
