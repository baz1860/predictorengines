"""NHL engine adapter."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from contracts import EngineAdapter, normalize_edge_result
from ._inproc import run_inprocess

ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "nhl"
RESULTS_CSV = ENGINE_DIR / "data" / "results.csv"
RESULTS_FILES = [ENGINE_DIR / "data" / "results_2025_26.csv", RESULTS_CSV]


def _fold_team(name: Any) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    ascii_text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(ascii_text.lower().replace(".", "").split())


def _date_key(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%d")
    return str(value or "")[:10]


def _id_key(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


class NHLAdapter(EngineAdapter):
    id = "nhl"
    name = "NHL"
    sport = "nhl"
    capabilities = {"predict", "edge"}

    EDGE_THRESHOLD = 0.03
    KELLY_FRACTION = 0.25

    def __init__(self) -> None:
        self._schema = None

    def _run(self, command: str, params: dict | None = None):
        from nhl import engine as nhl_engine
        return run_inprocess(nhl_engine.COMMANDS, command, params)

    def predict_schema(self) -> dict[str, Any]:
        if self._schema is None:
            self._schema = self._run("schema")
        from .. import provenance
        return {**self._schema, "freshness": provenance.freshness_warnings(self.id)}

    def edge_schema(self) -> dict[str, Any]:
        from nhl import gate as nhl_gate
        return {
            "models": ["blend", "power", "form"],
            "odds_sources": [{"id": "manual", "label": "Manual nhl/data/odds.csv"}],
            "options": [{"id": "market_blend",
                         "label": "Market blend (experimental)",
                         "default": False}],
            "has_template": True,
            "staking_gate": nhl_gate.load_gate(),
        }

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._run("predict", params)

    def edge(self, params: dict[str, Any]) -> dict[str, Any]:
        from .. import bankroll_store
        from .. import provenance
        from nhl import gate as nhl_gate
        model = str(params.get("model") or "blend")
        bankroll = bankroll_store.current_bankroll()
        result = self._run("edge", {**params, "bankroll": bankroll})
        result["bankroll"] = round(bankroll, 2)
        result["recorded"] = 0
        issues = provenance.validate_odds_file(self.id)
        if issues:
            result["odds_issues"] = [e["message"] for e in issues]

        rows = result.get("rows") or []
        if params.get("market_blend"):
            from .. import market_blend as MB
            w = MB.apply_blend_to_rows(rows, self.id, bankroll,
                                       self.KELLY_FRACTION, kelly_key="kelly_frac")
            result["market_blend"] = {"applied": True, "w": w, "experimental": True}
            result["note"] = (result.get("note", "")
                              + f" · market-blended (experimental, w={w:.2f})")
        gate = nhl_gate.load_gate()
        result["staking_gate"] = gate
        if nhl_gate.apply_staking_gate(rows, gate):
            if "staking disabled" not in str(result.get("note", "")).lower():
                result["note"] = (
                    result.get("note", "")
                    + f" · staking disabled: {gate.get('reason', 'validation gate failed')}"
                )
        else:
            self._mark_recommended(rows)

        if params.get("record"):
            recs = [r for r in rows if r.get("recommended")]
            if recs:
                df = pd.DataFrame(recs).rename(columns={"date": "match_date",
                                                        "kelly_frac": "kelly_stake"})
                df["source"] = "manual"
                df["model"] = model
                placed = bankroll_store.place_bets(self.id, self.sport, df)
                result["recorded"] = len(placed)
        return normalize_edge_result(result, source="manual", model=model,
                                     sport=self.sport)

    def _mark_recommended(self, rows: list[dict]) -> None:
        best: dict[tuple, dict] = {}
        for r in rows:
            r["recommended"] = False
            if float(r.get("edge", 0.0)) >= self.EDGE_THRESHOLD and float(r.get("ev_per_unit", 0.0)) > 0:
                k = (
                    r.get("match_date") or r.get("date"),
                    r.get("home"),
                    r.get("away"),
                    r.get("market"),
                    r.get("line", ""),
                    r.get("bookmaker") or r.get("source", "manual"),
                )
                if k not in best or float(r["edge"]) > float(best[k]["edge"]):
                    best[k] = r
        for r in best.values():
            r["recommended"] = True

    def write_odds_template(self) -> dict[str, Any]:
        from contracts import enrich_template_result
        return enrich_template_result(self._run("edge_template"))

    def grade_open_bets(self, rows: pd.DataFrame) -> dict[int, tuple]:
        frames = []
        for path in RESULTS_FILES:
            if path.exists():
                df = pd.read_csv(path)
                df["_result_source"] = str(path)
                frames.append(df)
        if not frames:
            return {}
        games = pd.concat(frames, ignore_index=True, sort=False)
        games["home_goals"] = pd.to_numeric(games["home_goals"], errors="coerce")
        games["away_goals"] = pd.to_numeric(games["away_goals"], errors="coerce")
        games = games.dropna(subset=["home_goals", "away_goals"]).copy()
        games["_date_key"] = games["date"].map(_date_key)
        games["_home_key"] = games["home"].map(_fold_team)
        games["_away_key"] = games["away"].map(_fold_team)
        if "game_id" not in games.columns:
            games["game_id"] = ""
        if "event_id" not in games.columns:
            games["event_id"] = ""
        games["_game_id_key"] = games["game_id"].map(_id_key)
        games["_event_id_key"] = games["event_id"].map(_id_key)
        games = games.drop_duplicates(
            subset=["_game_id_key", "_date_key", "_home_key", "_away_key",
                    "home_goals", "away_goals"],
            keep="first",
        )
        out: dict[int, tuple] = {}
        for idx, r in rows.iterrows():
            event_id = _id_key(r.get("event_id") or r.get("game_id") or "")
            if event_id:
                match = games[
                    (games["_game_id_key"] == event_id) | (games["_event_id_key"] == event_id)
                ]
            else:
                match_date = _date_key(r.get("match_date") or r.get("date") or "")
                match = games[
                    (games["_date_key"] == match_date)
                    & (games["_home_key"] == _fold_team(r.get("home")))
                    & (games["_away_key"] == _fold_team(r.get("away")))
                ]
            if len(match) != 1:
                continue
            g = match.iloc[0]
            home_goals = int(g["home_goals"])
            away_goals = int(g["away_goals"])
            status = self._grade_row(r, home_goals, away_goals)
            if status:
                out[idx] = (status, f"{home_goals}-{away_goals}")
        return out

    @classmethod
    def _grade_row(cls, r, home_goals: int, away_goals: int) -> str | None:
        market, line = cls._market_line(r)
        side = str(r.get("side", "")).lower()
        margin = home_goals - away_goals
        total = home_goals + away_goals
        if market == "ml":
            if side == "home":
                return "won" if margin > 0 else "lost"
            if side == "away":
                return "won" if margin < 0 else "lost"
            return None
        if market == "spread":
            if line is None:
                return None
            adjusted = margin + line if side == "home" else -margin + line
            if adjusted == 0:
                return "push"
            return "won" if adjusted > 0 else "lost"
        if market == "total":
            if line is None:
                return None
            if total == line:
                return "push"
            if side == "over":
                return "won" if total > line else "lost"
            if side == "under":
                return "won" if total < line else "lost"
        return None

    @staticmethod
    def _market_line(r) -> tuple[str, float | None]:
        market = str(r.get("market", "") or "").lower().replace("puck_line", "spread")
        bet = str(r.get("bet", ""))
        low = bet.lower()
        if not market:
            if low.startswith("ml"):
                market = "ml"
            elif low.startswith("puck") or low.startswith("spread"):
                market = "spread"
            elif low.startswith("total"):
                market = "total"
        raw_line = r.get("line", "")
        line = None
        if raw_line is not None and str(raw_line).strip() not in {"", "nan"}:
            try:
                line = float(raw_line)
            except (TypeError, ValueError):
                line = None
        if line is None:
            m = re.search(r"[-+]?\d+(?:\.\d+)?\s*$", bet)
            if m:
                line = float(m.group().strip())
        return market, line
