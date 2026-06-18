"""M5/M6 - coherent score-distribution board for World Cup markets.

The score matrix is the single backbone. 1X2, totals, BTTS, correct score, fair
odds, and confidence ranges all derive from the same distribution so downstream
recommendations cannot mix incompatible prices.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predictor import DC_RHO, HOME_ADV, score_matrix  # noqa: E402

from . import feature_store as FS  # noqa: E402

EPS = 1e-9


def markets_from_matrix(M: np.ndarray) -> dict[str, float]:
    n = M.shape[0]
    total = np.add.outer(np.arange(n), np.arange(n))
    out = {
        "home": float(np.tril(M, -1).sum()),
        "draw": float(np.trace(M)),
        "away": float(np.triu(M, 1).sum()),
        "over25": float(M[total >= 3].sum()),
        "under25": float(M[total <= 2].sum()),
        "btts_yes": float(M[1:, 1:].sum()),
    }
    out["btts_no"] = 1.0 - out["btts_yes"]
    return out


def correct_scores(M: np.ndarray, n: int = 8) -> list[dict[str, Any]]:
    rows = []
    for h in range(M.shape[0]):
        for a in range(M.shape[1]):
            rows.append({"score": f"{h}-{a}", "prob": float(M[h, a])})
    rows.sort(key=lambda r: -r["prob"])
    return [{"score": r["score"], "prob": round(r["prob"], 4)} for r in rows[:n]]


def fair_odds(prob: float) -> float | None:
    if prob <= EPS:
        return None
    return round(1.0 / prob, 3)


def confidence_range(prob: float, uncertainty_sd: float = 0.04) -> dict[str, float]:
    """Simple probability interval widened by model-risk SD.

    This is intentionally conservative and report-only; it gives M7 a common
    shape for "fair odds with a confidence range" before richer Bayesian
    intervals are validated.
    """
    width = float(np.clip(uncertainty_sd, 0.01, 0.18))
    lo = float(np.clip(prob - width, EPS, 1.0))
    hi = float(np.clip(prob + width, EPS, 1.0))
    return {"low": round(lo, 4), "mid": round(float(prob), 4),
            "high": round(hi, 4),
            "fair_odds_low": fair_odds(hi),
            "fair_odds_mid": fair_odds(prob),
            "fair_odds_high": fair_odds(lo)}


def coherent_board(home: str, away: str, asof: str,
                   neutral: bool = True,
                   uncertainty_sd: float = 0.04) -> dict[str, Any]:
    """Build a coherent market board from the M1 as-of feature row.

    If the teams cannot be priced from the point-in-time feature store, fail
    closed rather than guessing.
    """
    import pandas as pd

    fixtures = pd.DataFrame([{
        "date": pd.Timestamp(asof),
        "home_team": home,
        "away_team": away,
        "tournament": "FIFA World Cup",
        "neutral": neutral,
    }])
    rows = FS.build_asof(asof, fixtures=fixtures)
    if rows.empty:
        return {"available": False, "status": "fail_closed",
                "reason": "feature_store_could_not_price_fixture"}
    r = rows.iloc[0]
    M = score_matrix(float(r.lam_h), float(r.lam_a), DC_RHO)
    markets = markets_from_matrix(M)
    intervals = {k: confidence_range(v, uncertainty_sd) for k, v in markets.items()}
    return {
        "available": True,
        "status": "report_only",
        "home": home, "away": away, "asof": asof,
        "lambdas": [round(float(r.lam_h), 3), round(float(r.lam_a), 3)],
        "markets": {k: round(v, 4) for k, v in markets.items()},
        "fair_odds": {k: fair_odds(v) for k, v in markets.items()},
        "confidence": intervals,
        "correct_scores": correct_scores(M),
        "source_event_id": str(r.event_id),
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse, json
    ap = argparse.ArgumentParser(description="V4 coherent World Cup board")
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--asof", default="2026-06-11")
    args = ap.parse_args()
    print(json.dumps(coherent_board(args.home, args.away, args.asof), indent=2))
