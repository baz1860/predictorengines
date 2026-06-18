"""Sport-agnostic shared infrastructure (refactor Phase 2).

Betting-ledger bankroll management and closing-line-value (CLV) tracking operate on
the shared ledger across every engine, so they live here rather than at the repo root.
Will relocate under src/predictors/core/ in Phase 3.
"""
