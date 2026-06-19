"""P5/M5 - deterministic what-if line lab."""
from __future__ import annotations

from typing import Any

import numpy as np

from research.probability import coherent_board, fair_odds


def worldcup_line_lab(home: str, away: str, asof: str,
                      home_elo_delta: float = 0.0,
                      away_elo_delta: float = 0.0,
                      market_move: dict[str, float] | None = None) -> dict[str, Any]:
    """Synthetic sensitivity analysis that never writes production features."""
    base = coherent_board(home, away, asof)
    if not base.get("available"):
        return {"status": "fail_closed", "base": base}
    m = dict(base["markets"])
    # Translate Elo shocks into a small logit shift on the 1X2 win legs.
    shock = (float(home_elo_delta) - float(away_elo_delta)) / 400.0
    h = m["home"] * np.exp(shock)
    a = m["away"] * np.exp(-shock)
    d = m["draw"]
    s = h + d + a
    synth = {**m, "home": float(h / s), "draw": float(d / s), "away": float(a / s)}
    for k, delta in (market_move or {}).items():
        if k in synth:
            synth[k] = float(np.clip(synth[k] + float(delta), 1e-6, 0.999999))
    synth["under25"] = 1.0 - synth["over25"]
    synth["btts_no"] = 1.0 - synth["btts_yes"]
    return {
        "status": "synthetic",
        "home": home, "away": away, "asof": asof,
        "inputs": {
            "home_elo_delta": home_elo_delta,
            "away_elo_delta": away_elo_delta,
            "market_move": market_move or {},
        },
        "base": base["markets"],
        "scenario": {k: round(v, 4) for k, v in synth.items()},
        "delta": {k: round(synth[k] - m[k], 4) for k in m},
        "fair_odds": {k: fair_odds(v) for k, v in synth.items()},
        "note": "Synthetic what-if only; production feature snapshots are untouched.",
    }
