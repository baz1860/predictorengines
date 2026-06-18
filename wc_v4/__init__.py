"""World Cup V4 — bookmaker-grade line engine (modelling-power release).

V4 extends the V3 suite (common engine contract, security, validation gates,
market-blend/CLV, provenance) with *modelling* power. This package implements the
time-boxed V4 slice for the World Cup engine:

  * M1  feature_store  — point-in-time, leak-free per-event feature rows.
  * M2  market_model   — closing-line teacher + market-movement model.
  * M3  availability   — player availability / replacement value with uncertainty.
  * M4  matchup        — report-only tactical matchup diagnostics.
  * M5  probability    — coherent score-distribution market board.
  * M6  consistency    — cross-market contradiction/stale-price checks.
  * M7  staking        — uncertainty-aware stake haircut recommendations.

Design rules (from V4_PLAN.md, section 1 "Guardrails"):
  - Nothing here changes a V3 *default*. Every feature is point-in-time and
    report-only until it clears the held-out gate in `validate_v4.py`.
  - Closing lines are a *teacher*, never a feature: see `schema.OUTCOME_COLUMNS`.
  - No new third-party dependency — numpy + pandas only, like the rest of the suite.

Nothing in this package touches `app/` defaults, engine settlement, security, or
the bankroll store, so V3's safeguards are untouched (guardrail #5).
"""
from __future__ import annotations

from . import schema  # noqa: F401  (re-exported for convenience)

SCHEMA_VERSION = schema.SCHEMA_VERSION

__all__ = ["schema", "SCHEMA_VERSION"]
