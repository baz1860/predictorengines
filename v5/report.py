"""Convenience summary for the V5 layer."""
from __future__ import annotations

from . import drift, portfolio, registry, research, review


def build() -> dict:
    return {
        "registry": registry.registry_summary(),
        "drift": drift.recommendation_drift(),
        "portfolio": portfolio.optimize_from_recommendations(),
        "reviews": review.analytics(),
        "research": research.generate_backlog(),
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(build(), indent=2))
