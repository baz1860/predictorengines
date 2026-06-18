"""M7 - uncertainty-aware staking recommendations.

This is a report-only wrapper around the existing Kelly/portfolio discipline. It
haircuts candidate size for lineup uncertainty, stale data, market movement, weak
CLV context, and reason codes. If uncertainty overwhelms the edge, it returns a
pass recommendation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from edge import KELLY_FRACTION, kelly

from . import availability as AV
from . import market_model as MM
from .probability import confidence_range, fair_odds


def _haircut_from_reasons(reasons: list[str]) -> float:
    haircut = 1.0
    for r in reasons:
        if r.startswith("edge_below_threshold"):
            haircut *= 0.0
        elif r in ("market_already_moved_to_model", "close_moves_against_pick"):
            haircut *= 0.50
        elif r in ("low_lineup_confidence", "high_availability_uncertainty"):
            haircut *= 0.60
        elif r in ("negative_clv_context", "stale_or_missing_line_history"):
            haircut *= 0.75
    return float(np.clip(haircut, 0.0, 1.0))


def recommendation(match: dict[str, Any], side: str, model_prob: float,
                   market_odds: float, bankroll: float,
                   market_line: dict[str, Any] | None = None,
                   clv_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return fair odds, confidence range, haircuts and final stake.

    `match` expects at least home/away and optionally congestion_h/congestion_a.
    The caller supplies the model probability from a coherent board.
    """
    side_key = side[0].lower()
    home = str(match.get("home", ""))
    away = str(match.get("away", ""))
    team = home if side_key in ("h", "o", "b") else away
    av = AV.availability_adjustment(
        team,
        match.get("congestion_h") if team == home else match.get("congestion_a"),
    )
    uncertainty_sd = 0.04 + min(float(av.get("uncertainty_sd", 0.0) or 0.0) / 200.0, 0.12)
    ci = confidence_range(float(model_prob), uncertainty_sd)

    reasons: list[str] = []
    if float(av.get("lineup_confidence", 1.0)) < 0.75:
        reasons.append("low_lineup_confidence")
    if float(av.get("uncertainty_sd", 0.0) or 0.0) >= 8.0:
        reasons.append("high_availability_uncertainty")

    if market_line:
        dnb = MM.do_not_bet(side, float(model_prob), market_line)
        reasons.extend([r for r in dnb["reasons"] if r != "clear"])
    else:
        reasons.append("stale_or_missing_line_history")

    mean_clv = (clv_context or {}).get("mean_clv")
    if mean_clv is not None and float(mean_clv) < -0.005:
        reasons.append("negative_clv_context")

    raw_kelly = KELLY_FRACTION * kelly(float(model_prob), float(market_odds))
    haircut = _haircut_from_reasons(reasons)
    stake = round(float(bankroll) * raw_kelly * haircut, 2)
    edge = float(model_prob) - (1.0 / float(market_odds))
    pass_rec = stake <= 0.0 or ci["low"] <= 1.0 / float(market_odds)
    if pass_rec and "uncertainty_overwhelms_edge" not in reasons:
        reasons.append("uncertainty_overwhelms_edge")
    if pass_rec:
        stake = 0.0

    return {
        "status": "report_only",
        "recommendation": "pass" if pass_rec else "bet",
        "side": side,
        "model_prob": round(float(model_prob), 4),
        "market_odds": round(float(market_odds), 3),
        "market_implied": round(1.0 / float(market_odds), 4),
        "edge": round(edge, 4),
        "fair_odds": fair_odds(float(model_prob)),
        "confidence": ci,
        "raw_kelly_frac": round(float(raw_kelly), 4),
        "haircut": round(haircut, 4),
        "stake_gbp": stake,
        "availability": av,
        "clv_context": clv_context or {},
        "reason_codes": reasons or ["clear"],
    }
