"""International football module — scope contracts, identity and fixture stores.

Built to the plan in plans/international_football_module_plan.md (revision 3.1).
This package deliberately contains NO modelling code. The Elo / goal model /
Dixon-Coles path stays in engines/worldcup until the compatibility harness in
tests/international/test_legacy_goldens.py proves a move is safe.

What lives here, in dependency order:

  taxonomy   -- the competition registry: every label mapped to a category and
                an explicit importance weight, replacing the 12-entry table that
                silently sent 34.3% of matches to a generic default.
  registry   -- the effective-dated team registry: who is a FIFA member, which
                confederation, and whether a team is in product scope.
  identity   -- canonical fixture identity that survives UTC/local date splits,
                which is the defect that duplicated two 2026 World Cup matches.
  fixtures   -- reconciliation and the invariants that must hold over any
                fixture table (no duplicates, no scheduled-and-played pairs).
  store      -- append-only raw observation store + canonical fixture store.

Guardrails:
  * Nothing here mutates data/results.csv as a side effect of import.
  * Scope filtering FAILS CLOSED: an unclassified active team raises rather than
    being silently included or excluded.
  * Taxonomy weights are data, not code constants, and are versioned.
"""
from __future__ import annotations

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION"]
