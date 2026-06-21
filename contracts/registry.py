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


class Registry:
    """Holds the registered engine adapters, in display order."""

    def __init__(self) -> None:
        self._engines: dict[str, EngineAdapter] = {}

    def register(self, adapter: EngineAdapter) -> None:
        if not adapter.id:
            raise ValueError("adapter missing id")
        self._engines[adapter.id] = adapter

    def get(self, engine_id: str) -> EngineAdapter:
        if engine_id not in self._engines:
            raise KeyError(engine_id)
        return self._engines[engine_id]

    def all(self) -> list[EngineAdapter]:
        return list(self._engines.values())


registry = Registry()
