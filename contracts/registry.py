"""Engine adapter interface + registry.

Every prediction engine ships ONE adapter subclassing EngineAdapter. The adapter
declares its identity and which capabilities it supports, and wraps the engine's
existing code. The UI is built entirely from what adapters report here, so adding
a new engine = drop a new adapter file in this package and register it (see
engines/__init__.py). No UI or server changes required.

Capabilities (a subset of these per engine):
    "predict"   -> predict(params) : match or field prediction
    "refresh"   -> refresh(params) : local data/provider refresh
    "simulate"  -> simulate(params): Monte Carlo tournament / event   (Phase 2)
    "edge"      -> edge(params)    : edges, EV, Kelly stakes           (Phase 2)
    "round_3balls" -> round_3balls(params): round-specific golf 3-ball pricing
    "bankroll"  -> handled at suite level, not per-engine              (Phase 2)
"""
from __future__ import annotations

from typing import Any


class EngineAdapter:
    # --- identity (override in subclass) ---
    id: str = ""            # stable slug, e.g. "worldcup"
    name: str = ""          # display name, e.g. "World Cup 2026"
    sport: str = ""         # "soccer" | "cfb" | "golf" | ...
    capabilities: set[str] = set()

    # Retired ids this adapter still answers to. Settled and open bets record the
    # engine id at the time they were placed, and `app/bankroll_store.py` settles
    # them by calling `registry.get(row["engine"])`. Renaming an engine without
    # declaring the old id here orphans every one of those rows — as of August 2026
    # that is 21 live rows carrying engine="worldcup" in data/suite_ledger.csv.
    #
    # Aliases resolve for get() but are NOT returned by all(), so a renamed engine
    # appears exactly once in the UI while old API calls and old ledger rows keep
    # working. Declare them as a frozenset on the subclass:
    #
    #     class InternationalAdapter(EngineAdapter):
    #         id = "international"
    #         legacy_ids = frozenset({"worldcup"})
    legacy_ids: frozenset[str] = frozenset()

    # --- metadata for the UI ---
    def info(self) -> dict[str, Any]:
        schemas: dict[str, Any] = {}
        for cap in self.capabilities:
            fn = getattr(self, f"{cap}_schema", None)
            if callable(fn):
                schemas[cap] = fn()
        return {
            "id": self.id,
            "name": self.name,
            "sport": self.sport,
            "capabilities": sorted(self.capabilities),
            "legacy_ids": sorted(self.legacy_ids),
            "predict_schema": self.predict_schema(),  # kept for back-compat
            "schemas": schemas,
        }

    def predict_schema(self) -> dict[str, Any]:
        """Describe the inputs the Predict tab should render for this engine.

        kind: "match" (two competitors + home/neutral) or "field" (whole field).
        names: valid competitor names for typeahead validation.
        models: selectable model variants (first is default).
        """
        return {"kind": "match", "names": [], "models": []}

    # --- capability methods (override the ones you declare) ---
    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def simulate(self, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def edge(self, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class AliasConflict(ValueError):
    """Two adapters claim the same id or alias. Fail at startup, not at settlement."""


class Registry:
    """Holds the registered engine adapters, in display order.

    Two namespaces:
      * primary ids  -- one per adapter; these are what `all()` exposes and what
                        the UI renders.
      * legacy ids   -- retired ids that still resolve through `get()`, so historical
                        ledger rows and old API paths keep working after a rename.

    Collisions raise at registration. A silent overwrite here would mean an engine
    quietly shadowing another's settlement path, which is the kind of failure that
    only surfaces when money is being graded.
    """

    def __init__(self) -> None:
        self._engines: dict[str, EngineAdapter] = {}
        self._aliases: dict[str, str] = {}      # legacy id -> primary id

    def register(self, adapter: EngineAdapter) -> None:
        if not adapter.id:
            raise ValueError("adapter missing id")
        if adapter.id in self._engines:
            raise AliasConflict(
                f"engine id {adapter.id!r} already registered by "
                f"{type(self._engines[adapter.id]).__name__}")
        if adapter.id in self._aliases:
            raise AliasConflict(
                f"engine id {adapter.id!r} is already a legacy alias of "
                f"{self._aliases[adapter.id]!r}")

        for legacy in adapter.legacy_ids:
            if legacy == adapter.id:
                raise AliasConflict(
                    f"{adapter.id!r} lists itself in legacy_ids")
            if legacy in self._engines:
                raise AliasConflict(
                    f"legacy id {legacy!r} of {adapter.id!r} collides with the "
                    f"primary id of a registered engine")
            if legacy in self._aliases and self._aliases[legacy] != adapter.id:
                raise AliasConflict(
                    f"legacy id {legacy!r} already aliases "
                    f"{self._aliases[legacy]!r}, cannot also alias {adapter.id!r}")

        self._engines[adapter.id] = adapter
        for legacy in adapter.legacy_ids:
            self._aliases[legacy] = adapter.id

    def get(self, engine_id: str) -> EngineAdapter:
        """Resolve a primary id, or a legacy id via its alias."""
        if engine_id in self._engines:
            return self._engines[engine_id]
        primary = self._aliases.get(engine_id)
        if primary is not None:
            return self._engines[primary]
        raise KeyError(engine_id)

    def resolve_id(self, engine_id: str) -> str:
        """Canonical primary id for any id the registry answers to.

        Use when writing NEW records so the ledger converges on current ids, while
        `get()` keeps reading the old ones.
        """
        if engine_id in self._engines:
            return engine_id
        primary = self._aliases.get(engine_id)
        if primary is None:
            raise KeyError(engine_id)
        return primary

    def is_alias(self, engine_id: str) -> bool:
        return engine_id in self._aliases

    def aliases(self) -> dict[str, str]:
        """legacy id -> primary id, for diagnostics and migration reporting."""
        return dict(self._aliases)

    def known_ids(self) -> set[str]:
        """Every id that resolves, primary and legacy."""
        return set(self._engines) | set(self._aliases)

    def all(self) -> list[EngineAdapter]:
        """Primary adapters only — aliases must not duplicate the UI listing."""
        return list(self._engines.values())


registry = Registry()
