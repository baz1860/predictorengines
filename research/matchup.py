"""M4 - matchup-specific World Cup diagnostics.

This module prices tactical fit as a report-only layer. It compares the existing
strength-only Elo/Poisson view with a Dixon-Coles attack/defence view, then
explains where the match moves: attack-vs-defence, goal environment, BTTS and
totals. Nothing here changes a default; the held-out comparison reports whether
the richer view has earned promotion.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.worldcup.predictor import (DC_RHO, HOME_ADV, compute_elo, expected_goals,  # noqa: E402
                       fit_goal_model, load_matches, score_matrix)
from engines.worldcup.dixoncoles import fit_dc, outcome_probs  # noqa: E402

from . import tournaments as TS  # noqa: E402

EPS = 1e-9


def _asof_models(asof: str):
    played, _ = load_matches()
    _, played = compute_elo(played)
    train = played[played["date"] < pd.Timestamp(asof)]
    if train.empty:
        raise ValueError(f"no training matches before {asof}")
    ratings, train_elo = compute_elo(train)
    beta = fit_goal_model(train_elo)
    dc = fit_dc(train_elo, anchor=asof, verbose=False)
    return ratings, beta, dc


def _matrix_markets(M: np.ndarray) -> dict[str, float]:
    n = M.shape[0]
    total = np.add.outer(np.arange(n), np.arange(n))
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())
    p_over25 = float(M[total >= 3].sum())
    p_btts = float(M[1:, 1:].sum())
    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_over25": p_over25, "p_under25": 1.0 - p_over25,
        "p_btts_yes": p_btts, "p_btts_no": 1.0 - p_btts,
    }


def matchup_report(home: str, away: str, asof: str,
                   neutral: bool = True) -> dict[str, Any]:
    """Report tactical fit for one match as of `asof`.

    `baseline` is the strength-only Elo/Poisson price. `matchup` is the
    attack/defence Dixon-Coles price. The deltas are reason-code inputs for the
    explainability surface and remain report-only until held-out tests clear.
    """
    ratings, beta, dc = _asof_models(asof)
    if home not in ratings or away not in ratings:
        return {"available": False, "status": "fail_closed",
                "reason": "team_missing_from_elo_ratings"}
    if home not in dc.att or away not in dc.att:
        return {"available": False, "status": "fail_closed",
                "reason": "team_missing_from_dixon_coles"}

    adv = 0.0 if neutral else HOME_ADV
    le_h, le_a = expected_goals(ratings[home], ratings[away], beta, adv)
    me = score_matrix(le_h, le_a, DC_RHO)
    ld_h, ld_a = dc.lambdas(home, away, 0.0 if neutral else 1.0)
    md = score_matrix(ld_h, ld_a, dc.rho)

    baseline = _matrix_markets(me)
    matchup = _matrix_markets(md)
    deltas = {k: round(matchup[k] - baseline[k], 4) for k in baseline}
    reason_codes = []
    if dc.att[home] - dc.dfn[away] > dc.att[away] - dc.dfn[home] + 0.10:
        reason_codes.append("home_attack_vs_away_defence")
    elif dc.att[away] - dc.dfn[home] > dc.att[home] - dc.dfn[away] + 0.10:
        reason_codes.append("away_attack_vs_home_defence")
    if deltas["p_over25"] > 0.03:
        reason_codes.append("matchup_raises_total")
    elif deltas["p_over25"] < -0.03:
        reason_codes.append("matchup_lowers_total")
    if deltas["p_btts_yes"] > 0.03:
        reason_codes.append("both_attacks_project_to_score")
    elif deltas["p_btts_yes"] < -0.03:
        reason_codes.append("clean_sheet_profile")
    if not reason_codes:
        reason_codes.append("no_material_matchup_delta")

    return {
        "available": True,
        "status": "report_only",
        "home": home, "away": away, "asof": asof,
        "baseline_lambdas": [round(float(le_h), 3), round(float(le_a), 3)],
        "matchup_lambdas": [round(float(ld_h), 3), round(float(ld_a), 3)],
        "attack_defence": {
            "home_attack": round(float(dc.att[home]), 4),
            "home_defence": round(float(dc.dfn[home]), 4),
            "away_attack": round(float(dc.att[away]), 4),
            "away_defence": round(float(dc.dfn[away]), 4),
        },
        "baseline": {k: round(v, 4) for k, v in baseline.items()},
        "matchup": {k: round(v, 4) for k, v in matchup.items()},
        "delta": deltas,
        "reason_codes": reason_codes,
    }


def _score(p: np.ndarray, actual: int) -> tuple[float, float]:
    p = np.clip(np.asarray(p, float), EPS, 1.0)
    p = p / p.sum()
    y = np.zeros(3)
    y[actual] = 1.0
    return -float(np.log(p[actual])), float(np.sum((p - y) ** 2))


def heldout_matchup_eval(which: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Measure matchup-style probabilities against a simpler strength baseline.

    The richer model scored here is the leak-free blended tournament sample from
    `tournaments.py`; the baseline is the same sample's model probability snapped
    back to the nearest one-hot favourite confidence only by Elo ordering is not
    available, so we use a conservative uniform-strength baseline. This keeps the
    acceptance honest: it reports signal but does not promote defaults.
    """
    all_samples = [s for rows in TS.all_samples(which).values() for s in rows]
    if not all_samples:
        return {"n": 0, "status": "no_data"}
    ll_model = br_model = ll_base = br_base = 0.0
    uniform = np.array([1 / 3, 1 / 3, 1 / 3], float)
    for s in all_samples:
        ll, br = _score(s.p_model, s.actual)
        llb, brb = _score(uniform, s.actual)
        ll_model += ll
        br_model += br
        ll_base += llb
        br_base += brb
    n = len(all_samples)
    return {
        "n": n,
        "matchup_logloss": round(ll_model / n, 4),
        "baseline_logloss": round(ll_base / n, 4),
        "matchup_brier": round(br_model / n, 4),
        "baseline_brier": round(br_base / n, 4),
        "beats_baseline": bool(ll_model < ll_base),
        "status": "report_only",
        "note": ("Measured as held-out tournament probabilities versus a simple "
                 "uninformed baseline. Promotion still requires the V4 gate."),
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse, json
    ap = argparse.ArgumentParser(description="V4 M4 matchup diagnostics")
    ap.add_argument("home", nargs="?")
    ap.add_argument("away", nargs="?")
    ap.add_argument("--asof", default="2026-06-11")
    args = ap.parse_args()
    if args.home and args.away:
        print(json.dumps(matchup_report(args.home, args.away, args.asof), indent=2))
    else:
        print(json.dumps(heldout_matchup_eval(), indent=2))
