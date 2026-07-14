"""Horse-racing engine adapter.

V1 exposes race-level win prediction and analytical edge comparison. Recording
stakes into the shared bankroll is deliberately disabled until the suite ledger
can represent racing dead-heat settlement correctly.
"""
from __future__ import annotations

from typing import Any

from contracts import EngineAdapter, normalize_edge_result

from ._inproc import run_inprocess


class HorseRacingAdapter(EngineAdapter):
    id = "horse_racing"
    name = "Horse Racing (UK & Ireland Flat)"
    sport = "horse_racing"
    capabilities = {"predict", "edge"}

    def _run(self, command: str, params: dict | None = None):
        from horse_racing import engine as racing_engine
        return run_inprocess(racing_engine.COMMANDS, command, params)

    def predict_schema(self) -> dict[str, Any]:
        from .. import provenance
        schema = self._run("schema")
        return {**schema, "freshness": provenance.freshness_warnings(self.id)}

    def edge_schema(self) -> dict[str, Any]:
        races = self._run("schema").get("names", [])
        return {
            "models": [],
            "odds_sources": [{"id": "manual",
                              "label": "Manual horse_racing/data/odds.csv (win only)"}],
            "needs_sim_first": False,
            "options": [],
            "filters": [{"id": "race_id", "label": "Race",
                         "options": [""] + races}],
        }

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        race_id = params.get("race_id") or params.get("team1") or params.get("home")
        return self._run("predict", {**params, "race_id": race_id})

    def edge(self, params: dict[str, Any]) -> dict[str, Any]:
        # No record path in V1: shared bankroll settlement cannot correctly
        # represent dead-heat reductions, so silently accepting record=True would
        # create false P&L.
        result = self._run("edge", {**params, "record": False})
        from .. import bankroll_store
        result["bankroll"] = round(bankroll_store.current_bankroll(), 2)
        result["recorded"] = 0
        result["staking_enabled"] = False
        return normalize_edge_result(result, source="manual",
                                     model="regularized_conditional_logit_v1",
                                     sport=self.sport)
