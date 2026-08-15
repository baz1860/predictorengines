"""Explicit compatibility contract for Club Soccer betting evidence.

The old evidence cohort was the byte hash of several whole source files.  That
made comments, logging and unrelated identity edits reset the betting clock.
This module makes compatibility a deliberate, reviewable decision instead.

Changing pricing, selection, execution, staking eligibility, or calibration
semantics requires a new ``STRATEGY_VERSION``.  Refactors that preserve those
semantics do not.  The manifest is stored with every new decision in a sidecar
ledger so the append-only legacy decision CSV never needs an in-place rewrite.
"""
from __future__ import annotations

import hashlib
import json


STRATEGY_VERSION = "club_soccer_market_value_v1"

# The last byte-hash cohort before explicit versioning.  It used the same
# pricing/selection semantics as v1; declaring that once lets its settled rows
# remain compatible without making all future compatibility depend on bytes.
LEGACY_COMPATIBLE_CODE_HASHES = frozenset({"c635f75b50e5beb7"})

STRATEGY_MANIFEST = {
    "probability_model": "ensemble_with_active_gated_calibration",
    "market_benchmark": "executing_book_proportional_devig",
    "execution": "best_price_from_complete_market_book",
    "decision_window_minutes": [60, 120],
    "selection_edge": "p_model_minus_executing_book_devig",
    "eligibility": "market_model_do_not_bet_when_warmed",
    "stake_fraction": "quarter_kelly_times_lineup_confidence",
}


def manifest_hash() -> str:
    payload = json.dumps(STRATEGY_MANIFEST, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def version_for_legacy_code_hash(code_hash: object) -> str:
    """Map pre-contract rows to a compatible version or an isolated legacy cohort."""
    value = str(code_hash or "")
    if value in LEGACY_COMPATIBLE_CODE_HASHES:
        return STRATEGY_VERSION
    return f"legacy-code:{value or 'unknown'}"
