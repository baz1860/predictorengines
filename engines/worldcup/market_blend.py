#!/usr/bin/env python3
"""Market-anchored probability blend (v2 M3).

The model beats noise but not the closing line. Anchoring the model's 1X2
probabilities toward the de-vigged market removes most fake edges. The blend is
done in logit space, per outcome (one-vs-rest), then renormalised to sum to 1:

    logit(p_final_k) = w * logit(p_model_k) + (1 - w) * logit(p_market_k)
    p_final = softmax-free renormalise( sigmoid(logit) )   for k in {H, D, A}

w (the weight on the model) is fitted once by maximum likelihood on the WC2022
sample (data/wc2022_odds.csv + the same leak-free blend model the replay uses) and
stored in data/market_blend.json. edge.py applies it behind --market-blend; edges
are still computed against the raw de-vigged market, so the blend only moves
p_model. Expect w ≈ 0.2-0.4 and far fewer >=3% edges — that is the point.

Usage:
  python3 market_blend.py --fit     # fit w on WC2022, save data/market_blend.json
  python3 market_blend.py           # show the stored w
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from .predictor import (load_matches, compute_elo, fit_goal_model,
                       expected_goals, DC_RHO)
from .dixoncoles import fit_dc, outcome_probs
from .edge import devig

HERE = Path(__file__).resolve().parents[2]
BLEND_FILE = HERE / "data" / "market_blend.json"
WC2022_ODDS = HERE / "data" / "wc2022_odds.csv"
EPS = 1e-6
_SIDE_IDX = {"home": 0, "draw": 1, "away": 2}


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def blend(p_model, p_market, w):
    """Logit-space blend of two 3-vectors (H,D,A), renormalised to sum 1."""
    z = w * _logit(p_model) + (1.0 - w) * _logit(p_market)
    p = _sigmoid(z)
    s = p.sum()
    return p / s if s > 0 else np.asarray(p_model, float)


def _wc2022_samples():
    """(p_model, p_market, actual_idx) per WC2022 match, model leak-free as of
    the tournament start (same construction as wc2022_replay.py)."""
    played, _ = load_matches()
    _, played = compute_elo(played)              # adds point-in-time elo_h/elo_a
    train = played[played["date"] < "2022-11-20"]
    beta = fit_goal_model(train)
    dc = fit_dc(train, anchor="2022-11-20", verbose=False)
    ratings_cut, _ = compute_elo(train)          # point-in-time Elo at kickoff
    names = {"USA": "United States"}
    to_dec = lambda s: float(Fraction(s)) + 1.0  # fractional -> decimal

    odds = pd.read_csv(WC2022_ODDS)
    samples = []
    for r in odds.itertuples(index=False):
        home = names.get(r.home, r.home)
        away = names.get(r.away, r.away)
        if home not in ratings_cut or home not in dc.att or away not in dc.att:
            continue
        le = expected_goals(ratings_cut[home], ratings_cut[away], beta, 0.0)
        ld = dc.lambdas(home, away)              # neutral venue (Qatar)
        pe = np.array(outcome_probs(*le, DC_RHO)[:3])
        pdc = np.array(outcome_probs(*ld, dc.rho)[:3])
        p_model = (pe + pdc) / 2
        p_market, _ = devig([to_dec(r.odds_home), to_dec(r.odds_draw),
                             to_dec(r.odds_away)])
        samples.append((p_model, np.asarray(p_market, float),
                        _SIDE_IDX[r.result90]))
    return samples


def _mean_logloss(w, samples):
    ll = 0.0
    for pm, pk, a in samples:
        p = blend(pm, pk, w)
        ll += np.log(max(p[a], EPS))
    return -ll / len(samples)


def fit_w(verbose=True):
    samples = _wc2022_samples()
    ws = np.linspace(0.0, 1.0, 1001)
    losses = np.array([_mean_logloss(w, samples) for w in ws])
    i = int(np.argmin(losses))
    w = float(ws[i])
    ll_blend = float(losses[i])
    ll_model = _mean_logloss(1.0, samples)       # pure model
    ll_market = _mean_logloss(0.0, samples)      # pure market
    res = {"w": round(w, 3), "n": len(samples),
           "logloss_blend": round(ll_blend, 4),
           "logloss_model_only": round(ll_model, 4),
           "logloss_market_only": round(ll_market, 4),
           "source": "WC2022 (data/wc2022_odds.csv), logit-space 1X2 blend"}
    BLEND_FILE.write_text(json.dumps(res, indent=2))
    if verbose:
        better = ll_blend < ll_model and ll_blend < ll_market
        print(f"Fitted market blend on {len(samples)} WC2022 matches:")
        print(f"  w (weight on model) = {w:.3f}")
        print(f"  log-loss  model-only {ll_model:.4f} | market-only {ll_market:.4f}"
              f" | blend {ll_blend:.4f}")
        print(f"  blend strictly better than BOTH extremes: {better}")
        print(f"  saved -> {BLEND_FILE.relative_to(HERE)}")
    return res


def load_w():
    """Stored model weight w, or None if not yet fitted."""
    if BLEND_FILE.exists():
        return json.loads(BLEND_FILE.read_text())["w"]
    return None


def main():
    ap = argparse.ArgumentParser(description="Market-anchored 1X2 blend (v2 M3)")
    ap.add_argument("--fit", action="store_true",
                    help="fit w on WC2022 and write data/market_blend.json")
    args = ap.parse_args()
    if args.fit:
        fit_w()
    else:
        w = load_w()
        print(f"market blend w = {w}" if w is not None
              else "Not fitted yet. Run: python3 market_blend.py --fit")


if __name__ == "__main__":
    main()
