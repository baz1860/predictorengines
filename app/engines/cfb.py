"""College Football engine adapter.

Drives cfb/ via an isolated subprocess runner (cfb_runner.py) to avoid module
name collisions with the root engine. Predict (win/spread/total) and Edge
(ml/spread/total from a manual odds.csv). Settlement is done in-process with
plain pandas against cfb/data/games.csv, so no engine import is needed here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import EngineAdapter
from ._subprocess import run_engine

ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "cfb"
RUNNER = Path(__file__).resolve().parent / "runners" / "cfb_runner.py"


class CFBAdapter(EngineAdapter):
    id = "cfb"
    name = "College Football"
    sport = "cfb"
    capabilities = {"predict", "edge"}

    def __init__(self) -> None:
        self._schema = None

    def _run(self, command, params=None):
        return run_engine(ENGINE_DIR, RUNNER, command, params)

    def predict_schema(self) -> dict[str, Any]:
        if self._schema is None:
            self._schema = self._run("schema")
        return self._schema

    def edge_schema(self) -> dict[str, Any]:
        return {"models": ["blend", "elo", "power"],
                "odds_sources": [{"id": "manual", "label": "Manual cfb/odds.csv"}],
                "options": [{"id": "market_blend",
                             "label": "Market blend (experimental)",
                             "default": False}],
                "has_template": True}

    KELLY_FRACTION = 0.25

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._run("predict", params)

    EDGE_THRESHOLD = 0.03

    def edge(self, params: dict[str, Any]) -> dict[str, Any]:
        from .. import bankroll_store
        from .contracts import normalize_edge_result
        model = str(params.get("model", "blend"))
        bankroll = bankroll_store.current_bankroll()
        result = self._run("edge", {**params, "bankroll": bankroll})
        result["bankroll"] = round(bankroll, 2)
        result["recorded"] = 0
        rows = result.get("rows") or []
        if params.get("market_blend"):
            from .. import market_blend as MB
            w = MB.apply_blend_to_rows(rows, self.id, bankroll,
                                       self.KELLY_FRACTION, kelly_key="kelly_frac")
            result["market_blend"] = {"applied": True, "w": w, "experimental": True}
            result["note"] = (result.get("note", "")
                              + f" · market-blended (experimental, w={w:.2f})")
        self._mark_recommended(rows)
        if params.get("record"):
            recs = [r for r in rows if r.get("recommended")]
            if recs:
                df = pd.DataFrame(recs).rename(
                    columns={"date": "match_date", "kelly_frac": "kelly_stake"})
                df["source"] = "manual"
                df["model"] = model
                placed = bankroll_store.place_bets(self.id, self.sport, df)
                result["recorded"] = len(placed)
        return normalize_edge_result(result, source="manual", model=model,
                                     sport=self.sport)

    def _mark_recommended(self, rows: list[dict]) -> None:
        """Flag the bets recording would place: best edge per (home, away, market)
        clearing the edge threshold. Recording then places exactly these rows —
        no separate ad-hoc filter."""
        best: dict[tuple, dict] = {}
        for r in rows:
            r["recommended"] = False
            if float(r.get("edge", 0.0)) >= self.EDGE_THRESHOLD:
                k = (r.get("home"), r.get("away"), r.get("market"))
                if k not in best or float(r["edge"]) > float(best[k]["edge"]):
                    best[k] = r
        for r in best.values():
            r["recommended"] = True

    def write_odds_template(self) -> dict[str, Any]:
        return self._run("edge_template")

    # ── settlement (pure pandas; no cfb import) ──────────────────────────────
    def grade_open_bets(self, rows: pd.DataFrame) -> dict[int, tuple]:
        games_path = ENGINE_DIR / "data" / "games.csv"
        if not games_path.exists():
            return {}
        games = pd.read_csv(games_path)
        out: dict[int, tuple] = {}
        for i, r in rows.iterrows():
            g = games[(games["home_team"] == r["home"]) & (games["away_team"] == r["away"])
                      & (games["date"].astype(str) >= str(r["match_date"]))]
            if g.empty:
                continue
            g = g.iloc[0]
            margin = g["home_points"] - g["away_points"]
            total = g["home_points"] + g["away_points"]
            side = str(r["side"])
            market, line = self._market_line(r)
            if market == "ml":
                won, push = (margin > 0) if side == "home" else (margin < 0), margin == 0
            elif market == "spread":
                if pd.isna(line):
                    continue
                adj = (margin + line) if side == "home" else (-margin + line)
                won, push = adj > 0, adj == 0
            elif market == "total":
                if pd.isna(line):
                    continue
                won = (total > line) if side == "over" else (total < line)
                push = total == line
            else:
                continue
            status = "push" if push else ("won" if won else "lost")
            out[i] = (status, f"{int(g['home_points'])}-{int(g['away_points'])}")
        return out

    @staticmethod
    def _market_line(r) -> tuple[str, float | None]:
        """Parse market + line from the ledger 'bet' text, e.g.
        'ML home', 'SPREAD home -6.5', 'TOTAL over +48.5'. The suite ledger has
        no dedicated line column, so the line lives in the bet string."""
        import re
        bet = str(r.get("bet", ""))
        low = bet.lower()
        market = "ml" if low.startswith("ml") else (
            "spread" if low.startswith("spread") else (
                "total" if low.startswith("total") else ""))
        m = re.search(r"[-+]?\d+(?:\.\d+)?\s*$", bet)
        line = float(m.group().strip()) if m else None
        return market, line
