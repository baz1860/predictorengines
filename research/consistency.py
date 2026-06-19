"""M6 - cross-market consistency checks for coherent boards."""
from __future__ import annotations

from typing import Any

import numpy as np

from .probability import coherent_board

TOL = 1e-6


def check_board(board: dict[str, Any]) -> dict[str, Any]:
    """Flag impossible or contradictory model prices."""
    if not board.get("available", True):
        return {"ok": False, "issues": ["board_unavailable"], "status": "fail_closed"}
    m = board.get("markets", {})
    issues: list[str] = []
    if abs(sum(float(m.get(k, 0.0)) for k in ("home", "draw", "away")) - 1.0) > 5e-4:
        issues.append("1x2_probabilities_do_not_sum_to_one")
    for a, b, label in (("over25", "under25", "totals"),
                        ("btts_yes", "btts_no", "btts")):
        if abs(float(m.get(a, 0.0)) + float(m.get(b, 0.0)) - 1.0) > 5e-4:
            issues.append(f"{label}_complements_do_not_sum_to_one")
    for k, v in m.items():
        if not np.isfinite(float(v)) or float(v) < -TOL or float(v) > 1.0 + TOL:
            issues.append(f"invalid_probability:{k}")
    return {"ok": not issues, "issues": issues, "status": "report_only"}


def market_diagnostics(board: dict[str, Any],
                       market_odds: dict[str, float] | None = None) -> dict[str, Any]:
    """Compare a coherent model board to bookmaker prices and identify stale legs."""
    base = check_board(board)
    market_odds = market_odds or {}
    rows = []
    for k, odds in market_odds.items():
        try:
            o = float(odds)
        except (TypeError, ValueError):
            continue
        if o <= 1.0:
            continue
        model_p = (board.get("markets") or {}).get(k)
        if model_p is None:
            continue
        implied = 1.0 / o
        rows.append({
            "market": k,
            "model_prob": round(float(model_p), 4),
            "raw_implied": round(implied, 4),
            "gap": round(float(model_p) - implied, 4),
            "book_odds": round(o, 3),
            "fair_odds": (board.get("fair_odds") or {}).get(k),
            "diagnostic": "stale_or_generous" if float(model_p) - implied > 0.03 else "in_line",
        })
    return {**base, "market_rows": rows,
            "stale_markets": [r for r in rows if r["diagnostic"] == "stale_or_generous"]}


def diagnostics_for_match(home: str, away: str, asof: str,
                          market_odds: dict[str, float] | None = None) -> dict[str, Any]:
    board = coherent_board(home, away, asof)
    return {"board": board, "diagnostics": market_diagnostics(board, market_odds)}
