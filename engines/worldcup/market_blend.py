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
import hashlib
import json
import math
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
# The exact raw inputs whose hashes the deployable artifact must carry.  Keep
# this map in the consumer (not just the fitter): a loader that merely checks
# that an artifact contains hash-shaped strings cannot establish that the
# evidence still describes the data on disk.
PROVENANCE_INPUTS = {
    "data/wc2018_odds.csv": HERE / "data" / "wc2018_odds.csv",
    "data/wc2022_odds.csv": WC2022_ODDS,
}
EPS = 1e-6
_SIDE_IDX = {"home": 0, "draw": 1, "away": 2}


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def current_input_hashes() -> dict[str, str] | None:
    """SHA-256 digests of every raw input required by a deployable artifact.

    ``None`` is deliberately distinct from a partial mapping: missing or
    unreadable evidence is not evidence and must fail closed at load time.
    """
    out: dict[str, str] = {}
    try:
        for name, path in PROVENANCE_INPUTS.items():
            out[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None
    return out


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
    w = float(ws[i])                             # weight ON the model (this module's convention)
    ll_blend = float(losses[i])
    ll_model = _mean_logloss(1.0, samples)       # pure model
    ll_market = _mean_logloss(0.0, samples)      # pure market
    # This legacy single-tournament fitter selects w and scores it on the SAME
    # WC2022 rows, so it produces no admissible evidence: it emits no holdout_*
    # fields and is hard-wired inactive. Use engines.worldcup.fit_market_blend
    # (leave-one-tournament-out) for anything that could ever be deployed.
    res = {"model_weight": w,                    # canonical: 1==pure model, 0==pure market
           "n": len(samples),
           "in_sample_logloss_blend": ll_blend,  # full precision (a tie must be visible)
           "in_sample_logloss_model_only": ll_model,
           "in_sample_logloss_market_only": ll_market,
           "active": False,
           "note": "Legacy in-sample single-tournament fit: diagnostic only, never "
                   "promotable. Run engines.worldcup.fit_market_blend for holdout "
                   "evidence.",
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


# The blend must beat BOTH endpoints ON HELD-OUT DATA by at least this
# preregistered log-loss margin before it may be deployed. A tie (as the WC2018+
# WC2022 refit produced) is a NO-EDGE result, not a weight to ship.
BLEND_MIN_MARGIN = 1e-4
# Only genuinely out-of-sample evidence may open the gate. Metrics whose weight
# was selected on the same rows it is scored on are diagnostics, never evidence:
# the artifact stores those separately under in_sample_* and they are IGNORED here.
REQUIRED_HOLDOUT_METHODS = frozenset({"leave_one_tournament_out"})


def _valid_blend_artifact(d):
    """True only if the artifact carries a usable model_weight AND out-of-sample
    holdout metrics that strictly beat both endpoints by BLEND_MIN_MARGIN.

    Deliberately reads ONLY holdout_* fields. An artifact carrying just the older
    in-sample logloss_* keys fails closed — it has no admissible evidence."""
    if not isinstance(d, dict):
        return False
    needed = ("model_weight", "holdout_method", "holdout_logloss_blend",
              "holdout_logloss_model_only", "holdout_logloss_market_only")
    if not all(k in d for k in needed):
        return False
    if d.get("holdout_method") not in REQUIRED_HOLDOUT_METHODS:
        return False
    # Provenance must tie the metrics to the data that produced them.
    prov = d.get("provenance")
    if not isinstance(prov, dict) or not prov.get("inputs"):
        return False
    if not all(isinstance(v, str) and v for v in prov["inputs"].values()):
        return False
    # Require real JSON numbers. A numeric STRING ("0.3") or a bool is a schema
    # violation, not a value to coerce — signed provenance must not be type-loose.
    def _num(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        return float(v) if math.isfinite(v) else None

    w = _num(d["model_weight"])
    lb = _num(d["holdout_logloss_blend"])
    lm = _num(d["holdout_logloss_model_only"])
    lk = _num(d["holdout_logloss_market_only"])
    if any(x is None for x in (w, lb, lm, lk)):
        return False
    if not (0.0 <= w <= 1.0):
        return False
    return (lb < lm - BLEND_MIN_MARGIN) and (lb < lk - BLEND_MIN_MARGIN)


def _provenance_matches_current_inputs(d: dict) -> bool:
    """True only when the artifact names and hashes the exact local inputs.

    This makes the stored metrics invalid as soon as either odds source changes.
    It does not claim to authenticate the artifact's author; that requires the
    signed-manifest work tracked separately.
    """
    prov = d.get("provenance")
    recorded = prov.get("inputs") if isinstance(prov, dict) else None
    current = current_input_hashes()
    return isinstance(recorded, dict) and current is not None and recorded == current


def load_w():
    """Model weight (1 == pure model, 0 == pure market), or None.

    Fail-closed: returns None unless the artifact is explicitly ``active: true``,
    schema- and provenance-valid, and its LEAVE-ONE-TOURNAMENT-OUT holdout
    strictly beats BOTH endpoints by the preregistered margin. In-sample metrics
    can never open the gate. A demoted or degenerate (tie-with-market) artifact
    therefore yields None so ``--market-blend`` refuses to deploy it.
    """
    if not BLEND_FILE.exists():
        return None
    try:
        d = json.loads(BLEND_FILE.read_text())
    except (ValueError, OSError):
        return None
    if not isinstance(d, dict) or d.get("active") is not True:
        return None
    if not _valid_blend_artifact(d):
        return None
    if not _provenance_matches_current_inputs(d):
        return None
    return float(d["model_weight"])


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
