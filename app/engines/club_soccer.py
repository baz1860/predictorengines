"""Club Soccer engine adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from contracts import EngineAdapter
from ._inproc import run_inprocess

ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "club_soccer"


class ClubSoccerAdapter(EngineAdapter):
    id = "club_soccer"
    name = "Club Soccer"
    sport = "soccer"
    capabilities = {"predict", "edge"}

    def __init__(self) -> None:
        self._schema = None

    def _run(self, command: str, params: dict | None = None):
        from club_soccer import engine as cs_engine
        return run_inprocess(cs_engine.COMMANDS, command, params)

    def predict_schema(self) -> dict[str, Any]:
        if self._schema is None:
            self._schema = self._run("schema")
        from .. import provenance
        return {**self._schema, "freshness": provenance.freshness_warnings(self.id)}

    def edge_schema(self) -> dict[str, Any]:
        from ..market_blend import is_default_on
        blend_default = is_default_on(self.id)
        return {"models": ["ensemble", "goals", "elo"],
                "odds_sources": [
                    {"id": "manual", "label": "Manual club_soccer/data/odds.csv"},
                    {"id": "bsd", "label": "BSD odds (free)"},
                    {"id": "the-odds-api", "label": "The Odds API"}],
                "has_template": True,
                "options": [{"id": "market_blend",
                             "label": ("Market blend" if blend_default
                                       else "Market blend (experimental)"),
                             "default": blend_default}],
                "filters": self.predict_schema().get("filters", [])}

    KELLY_FRACTION = 0.25

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._run("predict", params)

    def edge(self, params: dict[str, Any]) -> dict[str, Any]:
        from .. import bankroll_store
        from contracts import normalize_edge_result
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
        # Defense in depth: the engine command applies the same policy, but the
        # adapter is the final user-facing boundary and must not trust a future
        # internal refactor to preserve it implicitly.
        from club_soccer.prediction_scope import filter_rows
        rows = filter_rows(rows)
        result["rows"] = rows
        self._mark_recommended(rows)
        if params.get("record"):
            today = pd.Timestamp.now(tz="UTC").date().isoformat()
            recs = [r for r in rows if r.get("recommended")
                    and not r.get("suppressed_reason")
                    and float(r.get("kelly_stake", 0) or 0) > 0
                    and str(r.get("date", ""))[:10] >= today]
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
        with edge > 0, model prob ≥ 0.40, a nonzero stake, and no suppression
        (do-not-bet filter or evidence gate). Recording places exactly these."""
        best: dict[tuple, dict] = {}
        for r in rows:
            r["recommended"] = False
            if (float(r.get("edge", 0.0)) > 0 and float(r.get("p_model", 0.0)) >= 0.40
                    and float(r.get("kelly_stake", 0.0) or 0.0) > 0
                    and not r.get("suppressed_reason")):
                k = (r.get("home"), r.get("away"), r.get("market"))
                if k not in best or float(r["edge"]) > float(best[k]["edge"]):
                    best[k] = r
        for r in best.values():
            r["recommended"] = True

    def write_odds_template(self) -> dict[str, Any]:
        from contracts import enrich_template_result
        return enrich_template_result(self._run("edge_template"))

    def grade_open_bets(self, rows: pd.DataFrame) -> dict[int, tuple]:
        from club_soccer import edge as CE
        from club_soccer.club_identity import canonical_name
        from club_soccer.schema import (OFFICIAL_RESULT_STATUSES,
                                        normalize_status)
        fixtures = pd.read_csv(ENGINE_DIR / "data" / "fixtures.csv")
        fixtures["home_goals"] = pd.to_numeric(fixtures["home_goals"], errors="coerce")
        fixtures["away_goals"] = pd.to_numeric(fixtures["away_goals"], errors="coerce")
        played = fixtures.dropna(subset=["home_goals", "away_goals"])
        # Settlement deliberately bypasses model.played() so an AWARDED (AWD)
        # result still settles. It nevertheless requires a terminal official
        # result: live scores, postponed rows and unknown statuses cannot grade.
        if "status" in played.columns:
            played = played[
                played["status"].map(normalize_status)
                .isin(OFFICIAL_RESULT_STATUSES)
            ]
        played = played.copy()
        played["_home_identity"] = played["home"].map(canonical_name)
        played["_away_identity"] = played["away"].map(canonical_name)
        out = {}
        for i, r in rows.iterrows():
            home = canonical_name(str(r["home"]))
            away = canonical_name(str(r["away"]))
            match = played[(played["date"].astype(str) == str(r["match_date"]))
                           & (played["_home_identity"] == home)
                           & (played["_away_identity"] == away)]
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
