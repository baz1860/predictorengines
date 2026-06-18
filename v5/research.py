"""P4/M6 - automated research backlog from drift and review signals."""
from __future__ import annotations

from typing import Any

from . import drift, review, store


def generate_backlog(engine: str | None = None) -> dict[str, Any]:
    d = drift.recommendation_drift(engine)
    r = review.analytics()
    items = []
    for alert in d.get("alerts", []):
        items.append({
            "id": f"research:{alert}",
            "source": "drift",
            "hypothesis": f"Investigate {alert.replace('_', ' ')}",
            "metric": "heldout_logloss_clv_drawdown",
            "status": "open",
            "sample_size": d.get("n", 0),
        })
    for state, n in (r.get("states") or {}).items():
        if state in ("rejected", "manually_adjusted") and n:
            items.append({
                "id": f"research:human:{state}",
                "source": "human_review",
                "hypothesis": f"Review common tags behind {state} decisions",
                "metric": "override_clv_vs_model_clv",
                "status": "open",
                "sample_size": int(n),
            })
    report = {
        "generated_at": store.now_iso(),
        "engine": engine or "all",
        "items": items,
        "graveyard": [],
        "note": "Backlog items are hypotheses; promotion requires comparable experiment reports.",
    }
    store.write_json(store.RESEARCH_BACKLOG, report)
    return report
