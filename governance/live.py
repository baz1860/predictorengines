"""P1/M4 - narrow live advisory prototype for soccer 1X2."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from research.probability import coherent_board, fair_odds


def soccer_live_1x2(home: str, away: str, asof: str, minute: int,
                    home_score: int, away_score: int,
                    red_cards_home: int = 0, red_cards_away: int = 0,
                    state_fetched_at: str | None = None,
                    max_latency_seconds: int = 120) -> dict[str, Any]:
    """Advisory live 1X2 update from score/minute/red-card state.

    Missing/delayed state returns pass, not a stale recommendation.
    """
    if state_fetched_at:
        fetched = datetime.fromisoformat(state_fetched_at.replace("Z", "+00:00"))
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - fetched).total_seconds()
        if age > max_latency_seconds:
            return {"status": "pass", "reason": "stale_live_state",
                    "latency_seconds": round(age, 1)}
    else:
        return {"status": "pass", "reason": "missing_live_state_timestamp"}

    board = coherent_board(home, away, asof)
    if not board.get("available"):
        return {"status": "pass", "reason": "prematch_board_unavailable"}
    remaining = max(0.0, (90.0 - float(minute)) / 90.0)
    pre = board["markets"]
    score_edge = float(home_score - away_score)
    red_edge = float(red_cards_away - red_cards_home) * 0.35
    shock = score_edge * (1.8 - remaining) + red_edge
    h = pre["home"] * np.exp(shock)
    a = pre["away"] * np.exp(-shock)
    d = pre["draw"] * (0.65 + 0.70 * remaining)
    s = h + d + a
    probs = {"home": float(h / s), "draw": float(d / s), "away": float(a / s)}
    return {
        "status": "advisory",
        "validation": "live_unvalidated",
        "home": home, "away": away, "minute": minute,
        "score": f"{home_score}-{away_score}",
        "latency_seconds": 0,
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "fair_odds": {k: fair_odds(v) for k, v in probs.items()},
        "confidence": "low",
        "note": "Prototype only; live markets require archived-state validation before promotion.",
    }
