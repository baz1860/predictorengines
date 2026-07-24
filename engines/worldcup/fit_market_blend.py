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
from engines.worldcup.market_blend import current_input_hashes

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

# NOTE ON CONVENTION: internally `w` here is the weight ON THE MARKET
# (w=1 -> pure market, w=0 -> pure model), because geometric pooling is written
# p_model^(1-w) * p_market^w. The consumer (market_blend.py) uses the OPPOSITE
# field convention: `model_weight` = weight ON THE MODEL. We therefore convert
# once at write time (model_weight = 1 - w_market) so the fitter and the consumer
# never disagree about what the stored number means.
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

OUT = {"H":0,"D":1,"A":2}
TOURN = {2018: ("data/wc2018_odds.csv","2018-06-01"),
         2022: ("data/wc2022_odds.csv","2022-11-01")}
HOST = {2018:"Russia", 2022:"Qatar"}
# blend must strictly beat BOTH endpoints on held-out data by this margin.
MIN_MARGIN = 1e-4


def _provenance(n_rows: int) -> dict:
    """Hash the exact odds inputs the fit consumed, so a stored artifact can be
    tied to the data that produced it (an unverifiable metric is not evidence)."""
    from datetime import datetime, timezone
    # Use the consumer's input map so the fitter cannot emit hashes that the
    # deploy gate interprets differently.
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_rows": n_rows, "inputs": current_input_hashes()}


def _build_bank():
    played, _ = P.load_matches()
    _, played = P.compute_elo(played)
    bank = {}
    for yr,(fp,asof) in TOURN.items():
        ratings,_ = P.compute_elo(played[played["date"] < asof])   # pre-tournament
        beta = P.fit_goal_model(played[played["date"] < asof])     # no leakage
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
    return bank


def compute(verbose=True):
    """Return the refit result dict (consumer convention) WITHOUT writing."""
    bank = _build_bank()
    allrows=[r for rs in bank.values() for r in rs]
    if verbose:
        print(f"\nMatches used: {len(allrows)}  (2018:{len(bank[2018])}, 2022:{len(bank[2022])})\n")

    grid=np.round(np.arange(0,1.01,0.05),2)
    full=[(w, logloss(allrows,w)) for w in grid]
    best_w_full=min(full,key=lambda t:t[1])[0]          # weight on MARKET (IN-SAMPLE)

    def best_w(rows):
        return min(((w,logloss(rows,w)) for w in grid), key=lambda t:t[1])[0]

    # ── leave-one-tournament-out: the ONLY evidence allowed to open the gate ──
    # For each fold the weight is selected on the OTHER tournament and scored on
    # the held-out one, and BOTH endpoints are scored on those same held-out
    # points, so blend and endpoints are compared like-for-like out of sample.
    folds=[]
    for test_yr in (2018,2022):
        train_yr=2022 if test_yr==2018 else 2018
        w=best_w(bank[train_yr])                        # selected WITHOUT the test fold
        rows=bank[test_yr]
        folds.append({
            "test_tournament": test_yr,
            "train_tournament": train_yr,
            "n": len(rows),
            "model_weight": 1.0 - float(w),
            "logloss_blend": float(logloss(rows, w)),
            "logloss_model_only": float(logloss(rows, which='model')),
            "logloss_market_only": float(logloss(rows, which='market')),
        })
    # Pool by sample size: every row is held out exactly once, so this is the
    # honest out-of-sample loss over all n points.
    tot = sum(f["n"] for f in folds)
    def _pooled(key):
        return sum(f[key] * f["n"] for f in folds) / tot
    ho_blend  = _pooled("logloss_blend")
    ho_model  = _pooled("logloss_model_only")
    ho_market = _pooled("logloss_market_only")
    cv_w = float(np.mean([1.0 - f["model_weight"] for f in folds]))  # market-weight

    # In-sample values are retained for DIAGNOSIS ONLY and are named as such.
    is_model=float(logloss(allrows,which='model'))
    is_market=float(logloss(allrows,which='market'))
    is_blend=float(logloss(allrows,best_w_full))
    # Convert market-weight -> model-weight (consumer convention) at the boundary.
    model_weight = 1.0 - float(best_w_full)
    model_weight_cv = 1.0 - cv_w
    # Gate on the HOLDOUT numbers only. Selecting and scoring on the same rows
    # cannot justify deployment, however large the apparent margin.
    active = (ho_blend < ho_model - MIN_MARGIN) and (ho_blend < ho_market - MIN_MARGIN)

    if verbose:
        print("HOLDOUT (leave-one-tournament-out) log-loss — gate evidence:")
        print(f"  model only   {ho_model:.6f}")
        print(f"  market only  {ho_market:.6f}")
        print(f"  blend        {ho_blend:.6f}")
        for f in folds:
            print(f"    train {f['train_tournament']} -> test {f['test_tournament']} "
                  f"(n={f['n']}): model_weight={f['model_weight']:.2f} "
                  f"blend {f['logloss_blend']:.6f} vs market "
                  f"{f['logloss_market_only']:.6f}")
        print("\nIN-SAMPLE (diagnostic only, never gates):")
        print(f"  model {is_model:.6f} | market {is_market:.6f} | blend {is_blend:.6f}")
        print(f"\nactive={active}")

    return {"model_weight": model_weight,               # canonical: 1==model, 0==market
            "model_weight_cv": model_weight_cv,
            "n": len(allrows),
            "source": "WC2018+WC2022 (de-vigged 1X2), logit/geometric blend, LOTO-CV",
            # ── gate evidence: out-of-sample, like-for-like ──
            "holdout_method": "leave_one_tournament_out",
            "holdout_logloss_blend": ho_blend,
            "holdout_logloss_model_only": ho_model,
            "holdout_logloss_market_only": ho_market,
            "holdout_folds": folds,
            # ── diagnostics: selected AND scored on the same rows; NOT evidence ──
            "in_sample_logloss_blend": is_blend,
            "in_sample_logloss_model_only": is_model,
            "in_sample_logloss_market_only": is_market,
            "provenance": _provenance(len(allrows)),
            "active": bool(active),
            "note": ("Holdout blend does not beat both endpoints by the preregistered "
                     "margin: the model adds no WC 1X2 edge out of sample. Not deployable."
                     if not active else
                     "Holdout blend strictly beats both endpoints; safe to activate.")}


def main(write=True):
    out = compute(verbose=True)
    if write:
        (ROOT/"data/market_blend.json").write_text(json.dumps(out,indent=2))
        print("\nwrote data/market_blend.json:",json.dumps(out))
    return out


if __name__ == "__main__":
    main()
