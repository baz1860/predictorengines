"""1X2 feature scaffold — multivariate Poisson goal model + promotion harness.

DRAFT / SCAFFOLD. Generalises `predictor.fit_goal_model` (2-param: intercept + eloΔ)
to an arbitrary oriented feature vector, so new signals (squad availability, rest,
attack/defence form, context) feed the SAME expected-goals layer and keep 1X2 / totals
/ BTTS coherent. Every feature is gated on held-out 3-way log-loss before promotion,
mirroring `predictor.fit_dc_params` / `dc_params.json`.

Design:
  * Each match contributes TWO oriented observations: (attacker=home, defender=away)
    and (attacker=away, defender=home). `oriented_features(att, dfd, asof, is_home)`
    returns the covariate row; the response is the attacker's goals.
  * log λ_attacker = θ · features.  λ feeds `predictor.score_matrix`.
  * Incumbent = elo-only [intercept, eloΔ/400]. Candidate = incumbent + new columns.

Run:  python -m engines.worldcup.features_1x2 --selfcheck
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import engines.worldcup.predictor as P  # noqa: E402

DATA = ROOT / "data"


# ──────────────────────────────────────────────────────────────────────────
# Feature registry. Each extractor returns a float for the ATTACKING team,
# computed point-in-time (only info available before `asof`). Keep them cheap
# and side-effect free. `None`/NaN => treated as 0 (feature off for that row).
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Feature:
    name: str
    fn: Callable[..., float]
    active: bool = False          # flip on once it passes the gate
    note: str = ""


def f_intercept(att, dfd, asof, is_home, ctx): return 1.0


def f_elo_diff(att, dfd, asof, is_home, ctx):
    adv = 0.0 if ctx.get("neutral", True) else (P.HOME_ADV if is_home else -P.HOME_ADV)
    return (ctx["elo_att"] + adv - ctx["elo_dfd"]) / 400.0


# ---- #2 squad availability (partly implemented) --------------------------
_SQUAD = None
def _squad():
    global _SQUAD
    if _SQUAD is None:
        try:
            _SQUAD = pd.read_csv(DATA / "squad_ratings.csv").set_index("team")
        except Exception:
            _SQUAD = pd.DataFrame()
    return _SQUAD

def f_avail_gap(att, dfd, asof, is_home, ctx):
    """power_avail - power_full for the attacking team (<=0; injuries hurt).
    TODO: make point-in-time by recomputing power_avail from DATED absences.csv
    rather than the current static snapshot."""
    s = _squad()
    if att not in getattr(s, "index", []):
        return 0.0
    row = s.loc[att]
    return float(row.get("power_avail", row.get("power_full", 0.0)) - row.get("power_full", 0.0))


# ---- #4 rest differential (partly implemented) ---------------------------
def f_rest_diff(att, dfd, asof, is_home, ctx):
    """clip(rest_att - rest_dfd, +/-7). Expects ctx to carry rest days from the
    feature store (rest_days_h/a). Returns 0 when unavailable."""
    ra, rd = ctx.get("rest_att"), ctx.get("rest_dfd")
    if ra is None or rd is None or np.isnan(ra) or np.isnan(rd):
        return 0.0
    return float(np.clip(ra - rd, -7, 7)) / 7.0


# ---- #3 attack/defence form (STUB) ---------------------------------------
def f_ad_form(att, dfd, asof, is_home, ctx):
    """TODO: trailing-window Dixon-Coles att_att - def_dfd (half-life weighted),
    as a RESIDUAL on top of Elo. Wire to wc_v4.matchup / engines.worldcup.dixoncoles.
    Return 0.0 until implemented so the column is inert."""
    return 0.0


# ---- #5 match context / motivation (STUB) --------------------------------
def f_dead_rubber(att, dfd, asof, is_home, ctx):
    """TODO: 1.0 if attacker's result is dead (already through/out) at a final
    group game, else 0.0; derive from tournaments.py standings. Mainly corrects
    goals + draw rate."""
    return float(ctx.get("dead_rubber_att", 0.0))


def f_knockout(att, dfd, asof, is_home, ctx):
    """TODO: 1.0 for knockout fixtures (cagier, drawier in regulation)."""
    return float(ctx.get("knockout", 0.0))


REGISTRY = [
    Feature("intercept", f_intercept, active=True),
    Feature("elo_diff",  f_elo_diff,  active=True),
    Feature("avail_gap", f_avail_gap, active=False, note="#2 wire dated absences"),
    Feature("rest_diff", f_rest_diff, active=False, note="#4 ctx rest days"),
    Feature("ad_form",   f_ad_form,   active=False, note="#3 STUB"),
    Feature("dead_rub",  f_dead_rubber,active=False, note="#5 STUB"),
    Feature("knockout",  f_knockout,  active=False, note="#5 STUB"),
]


def design_row(att, dfd, asof, is_home, ctx, feats):
    return np.array([f.fn(att, dfd, asof, is_home, ctx) for f in feats], float)


# ──────────────────────────────────────────────────────────────────────────
# Multivariate Poisson IRLS (generalises predictor.fit_goal_model)
# ──────────────────────────────────────────────────────────────────────────
def fit_poisson(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                iters: int = 50, tol: float = 1e-10):
    """Weighted Poisson IRLS. `w` are per-observation sample weights (e.g. time
    decay); None => equal weight (the historical behaviour)."""
    if w is None:
        w = np.ones(len(y))
    theta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = np.exp(np.clip(X @ theta, -10, 10))
        W = mu * w                       # weight enters the IRLS weight matrix
        z = X @ theta + (y - mu) / np.maximum(mu, 1e-9)
        XtW = X.T * W
        new = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(new - theta)) < tol:
            theta = new
            break
        theta = new
    return theta


def decay_weights(dates, asof, half_life_years: float | None) -> np.ndarray | None:
    """Exponential time-decay: weight = 0.5 ** (age_years / half_life). A 1873 match
    with an 8-year half-life carries weight 0.5**(~150/8) ≈ 2e-6 — effectively zero,
    which is the point. None => equal weight."""
    if half_life_years is None:
        return None
    a = pd.Timestamp(asof)
    age = np.array([(a - pd.Timestamp(d)).days / 365.25 for d in dates])
    return np.power(0.5, np.clip(age, 0, None) / half_life_years)


def build_matrix(played: pd.DataFrame, feats, ctx_of):
    """Stack two oriented observations per match. `ctx_of(row, attacker_is_home)`
    returns the ctx dict (elo, rest, neutral, context flags). Returns (X, y, dates)
    where dates is one entry per observation (for time-decay weighting)."""
    rows, ys, dates = [], [], []
    for r in played.itertuples(index=False):
        ch = ctx_of(r, True)
        rows.append(design_row(r.home_team, r.away_team, r.date, True, ch, feats))
        ys.append(r.home_score); dates.append(r.date)
        ca = ctx_of(r, False)
        rows.append(design_row(r.away_team, r.home_team, r.date, False, ca, feats))
        ys.append(r.away_score); dates.append(r.date)
    return np.array(rows, float), np.array(ys, float), dates


def _ctx_factory(r, attacker_is_home):
    """Minimal ctx from the elo-tagged results frame. Extend to carry rest days and
    context flags once joined from the feature store."""
    if attacker_is_home:
        return {"elo_att": r.elo_h, "elo_dfd": r.elo_a, "neutral": bool(r.neutral),
                "rest_att": getattr(r, "rest_days_h", np.nan),
                "rest_dfd": getattr(r, "rest_days_a", np.nan)}
    return {"elo_att": r.elo_a, "elo_dfd": r.elo_h, "neutral": bool(r.neutral),
            "rest_att": getattr(r, "rest_days_a", np.nan),
            "rest_dfd": getattr(r, "rest_days_h", np.nan)}


# ──────────────────────────────────────────────────────────────────────────
# Promotion harness: held-out 3-way log-loss, candidate vs elo-only incumbent
# ──────────────────────────────────────────────────────────────────────────
def _logloss_1x2(played_eval, theta, feats, ctx_of) -> float:
    ll, n = 0.0, 0
    for r in played_eval.itertuples(index=False):
        lam_h = float(np.exp(design_row(r.home_team, r.away_team, r.date, True,  ctx_of(r, True),  feats) @ theta))
        lam_a = float(np.exp(design_row(r.away_team, r.home_team, r.date, False, ctx_of(r, False), feats) @ theta))
        M = P.score_matrix(lam_h, lam_a)
        p = [np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()]
        k = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        ll += -np.log(max(p[k], 1e-12)); n += 1
    return ll / n


def evaluate(candidate_names: list[str], train_to="2018-01-01", margin=0.001,
             half_life_years: float | None = 8.0):
    """Fit incumbent (elo-only) and candidate (elo-only + candidate cols) on
    matches before `train_to`; compare held-out log-loss after it. Promote only on
    margin. `half_life_years` applies time-decay weighting (None = equal weight,
    the legacy behaviour). Returns a dict you can serialise next to dc_params.json."""
    played, _ = P.load_matches()
    _, played = P.compute_elo(played)
    train = played[played["date"] < train_to]
    eval_ = played[(played["date"] >= train_to) & (played["tournament"] != "Friendly")]

    incumbent = [f for f in REGISTRY if f.name in ("intercept", "elo_diff")]
    chosen = set(["intercept", "elo_diff"]) | set(candidate_names)
    candidate = [f for f in REGISTRY if f.name in chosen]

    Xi, yi, di = build_matrix(train, incumbent, _ctx_factory)
    Xc, yc, dc = build_matrix(train, candidate, _ctx_factory)
    wi = decay_weights(di, train_to, half_life_years)
    wc = decay_weights(dc, train_to, half_life_years)
    ti, tc = fit_poisson(Xi, yi, wi), fit_poisson(Xc, yc, wc)
    lli = _logloss_1x2(eval_, ti, incumbent, _ctx_factory)
    llc = _logloss_1x2(eval_, tc, candidate, _ctx_factory)
    return {"candidate": candidate_names, "half_life_years": half_life_years,
            "n_train": len(train), "n_eval": len(eval_),
            "logloss_incumbent": round(lli, 5), "logloss_candidate": round(llc, 5),
            "improvement": round(lli - llc, 5), "margin": margin,
            "promote": bool((lli - llc) >= margin),
            "theta_candidate": dict(zip([f.name for f in candidate], np.round(tc, 4).tolist()))}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--features", nargs="*", default=["rest_diff"],
                    help="candidate feature names to test against the elo-only incumbent")
    ap.add_argument("--train-to", default="2018-01-01")
    a = ap.parse_args()
    if a.selfcheck:
        # tiny sanity: incumbent recovers a sensible intercept/slope
        played, _ = P.load_matches(); _, played = P.compute_elo(played)
        sub = played[played["date"] >= "2015-01-01"]
        feats = [f for f in REGISTRY if f.name in ("intercept", "elo_diff")]
        X, y, _ = build_matrix(sub, feats, _ctx_factory)
        print("theta(elo-only) =", np.round(fit_poisson(X, y), 4),
              "(expect ~[0.1-0.3, 0.6-0.9])")
    else:
        print(json.dumps(evaluate(a.features, train_to=a.train_to), indent=2))
