"""Engine contract layer (refactor Phase 2b).

The neutral interface every engine speaks: the adapter base class + registry, and the
contract vocabulary (canonical fixture/market identity, edge-row normalisation, JSON
finiteness checks). Lifted out of app/engines/ so non-app layers (wc_v4, v5) and the
suite scripts depend on the contract directly instead of importing "up" into app.

app/engines/__init__.py remains the single wiring point: it imports `registry` from
here and registers the concrete adapters into it.
"""
from .registry import EngineAdapter, Registry, registry
from .protocol import (
    CANONICAL_EDGE_FIELDS,
    ContractError,
    assert_finite_json,
    is_finite_number,
    market_id,
    fixture_key,
    validate_prediction,
    validate_table,
    normalize_edge_row,
    normalize_edge_result,
    enrich_template_result,
    validate_edge_rows,
)

__all__ = [
    "EngineAdapter", "Registry", "registry",
    "CANONICAL_EDGE_FIELDS", "ContractError", "assert_finite_json", "is_finite_number",
    "market_id", "fixture_key", "validate_prediction", "validate_table",
    "normalize_edge_row", "normalize_edge_result", "enrich_template_result",
    "validate_edge_rows",
]
