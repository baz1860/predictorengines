"""Club Soccer engine adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import EngineAdapter
from ._subprocess import run_engine

ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "club_soccer"
RUNNER = Path(__file__).resolve().parent / "runners" / "club_soccer_runner.py"


class ClubSoccerAdapter(EngineAdapter):
    id = "club_soccer"
    name = "Club Soccer"
    sport = "soccer"
    capabilities = {"predict", "edge"}

    def __init__(self) -> None:
        self._schema = None

    def _run(self, command: str, params: dict | None = None):
        return run_engine(ENGINE_DIR, RUNNER, command, params, timeout=180)

    def predict_schema(self) -> dict[str, Any]:
        if self._schema is None:
            self._schema = self._run("schema")
        from .. import provenance
        return {**self._schema, "freshness": provenance.freshness_warnings(self.id)}

    def edge_schema(self) -> dict[str, Any]:
        return {"models": ["ensemble", "goals", "elo"],
                "odds_sources": [
                    {"id": "manual", "label": "Manual club_soccer/data/odds.csv"},
                    {"id": "api", "label": "API-Football odds"},
                    {"id": "the-odds-api", "label": "The Odds API"}],
                "has_template": True,
                "options": [{"id": "market_blend",
                             "label": "Market blend (experimental)",
                             "default": False}],
                "filters": self.predict_schema().get("filters", [])}

    KELLY_FRACTION = 0.25

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._run("predict", params)

    def edge(self, params: dict[str, Any]) -> dict[str, Any]:
        from .. import bankroll_store
        from .contracts import normalize_edge_result
        model = str(params.get("model") or "ensemble")
        odds_source = str(params.get("odds_source") or "manual")
        bankroll = bankroll_store.current_bankroll()
        result = self._run("edge", {**params, "bankroll": bankroll})
        result["bankroll"] = round(bankroll, 2)
        result["recorded"] = 0
        if odds_source == "manual":
            from .. import provenance
            issues = provenance.validate_odds_file(self.id)
            if issues:
                result["odds_issues"] = [e["message"] for e in issues]
        rows = result.get("rows") or []
        if params.get("market_blend"):
            from .. import market_blend as MB
            w = MB.apply_blend_to_rows(rows, self.id, bankroll,
                                       self.KELLY_FRACTION, kelly_key="kelly_stake")
            result["market_blend"] = {"applied": True, "w": w, "experimental": True}
            result["note"] = (result.get("note", "")
                              + f" · market-blended (experimental, w={w:.2f})")
        self._mark_recommended(rows)
        if params.get("record"):
            recs = [r for r in rows if r.get("recommended")]
            if recs:
                df = pd.DataFrame(recs).rename(columns={"date": "match_date"})
                df["source"] = odds_source
                df["model"] = model
                placed = bankroll_store.place_bets(self.id, self.sport, df)
                result["recorded"] = len(placed)
        return normalize_edge_result(result, source=odds_source, model=model,
                                     sport=self.sport)

    @staticmethod
    def _mark_recommended(rows: list[dict]) -> None:
        """Flag the bets recording would place: best edge per (home, away, market)
        with edge > 0 and model prob ≥ 0.40. Recording places exactly these."""
        best: dict[tuple, dict] = {}
        for r in rows:
            r["recommended"] = False
            if float(r.get("edge", 0.0)) > 0 and float(r.get("p_model", 0.0)) >= 0.40:
                k = (r.get("home"), r.get("away"), r.get("market"))
                if k not in best or float(r["edge"]) > float(best[k]["edge"]):
                    best[k] = r
        for r in best.values():
            r["recommended"] = True

    def write_odds_template(self) -> dict[str, Any]:
        from .contracts import enrich_template_result
        return enrich_template_result(self._run("edge_template"))

    def grade_open_bets(self, rows: pd.DataFrame) -> dict[int, tuple]:
        import importlib.util
        import sys
        if str(ENGINE_DIR) not in sys.path:
            sys.path.insert(0, str(ENGINE_DIR))
        spec = importlib.util.spec_from_file_location("club_soccer_edge", ENGINE_DIR / "edge.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load Club Soccer edge grader")
        CE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(CE)
        fixtures = pd.read_csv(ENGINE_DIR / "data" / "fixtures.csv")
        fixtures["home_goals"] = pd.to_numeric(fixtures["home_goals"], errors="coerce")
        fixtures["away_goals"] = pd.to_numeric(fixtures["away_goals"], errors="coerce")
        played = fixtures.dropna(subset=["home_goals", "away_goals"])
        out = {}
        for i, r in rows.iterrows():
            match = played[(played["date"].astype(str) == str(r["match_date"]))
                           & (played["home"] == r["home"])
                           & (played["away"] == r["away"])]
            if match.empty:
                continue
            g = match.iloc[0]
            bet = str(r.get("bet", "")).lower()
            side = str(r["side"])
            market = "btts" if "btts" in bet or "both teams" in bet.lower() else (
                "total" if "over" in bet or "under" in bet else "1x2")
            line = 2.5 if market == "total" else ""
            status = CE.grade(side, market, line, g["home_goals"], g["away_goals"])
            if status:
                out[i] = (status, f"{int(g['home_goals'])}-{int(g['away_goals'])}")
        return out
