"""Tennis (ATP + WTA) engine adapter.

Drives the tennis package in-process via tennis.engine.COMMANDS. Capabilities:
Predict (head-to-head match winner + set/games sub-markets from the Markov
chain), Simulate (draw Monte-Carlo → outright win/final/SF/QF), and Edge
(two-way de-vigged EV across tennis/data/odds.csv, fractional-Kelly staked).

Bets settle against completed matches in tennis/data/matches.csv (refreshed by
`fetch.py --accumulate`): a match-winner bet grades the earliest completed match
on/after the bet date whose two participants are the bet's player and opponent.
A bet for a match not yet in matches.csv stays open until the result lands.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from contracts import EngineAdapter, normalize_edge_result
from ._inproc import run_inprocess

ROOT = Path(__file__).resolve().parents[2]
MATCHES_CSV = ROOT / "tennis" / "data" / "matches.csv"


def _fold(name: str) -> str:
    from tennis.model import fold_name
    return fold_name(str(name or ""))


class TennisAdapter(EngineAdapter):
    id = "tennis"
    name = "Tennis (ATP + WTA)"
    sport = "tennis"
    capabilities = {"predict", "simulate", "edge", "draw"}

    def _run(self, command, params=None):
        from tennis import engine as tennis_engine
        return run_inprocess(tennis_engine.COMMANDS, command, params)

    # ── schemas ──
    @staticmethod
    def _tour_ids(tours_raw: list) -> list[str]:
        """Extract plain string IDs from a tours list that may contain {id, label}
        dicts (as returned by tennis/engine.py TOURS). The UI fillSelect and model
        select elements expect strings; objects render as '[object Object]'."""
        out = []
        for t in tours_raw:
            if isinstance(t, dict):
                out.append(str(t.get("id") or t.get("label") or t).upper())
            else:
                out.append(str(t).upper())
        return out or ["ATP", "WTA"]

    def predict_schema(self) -> dict[str, Any]:
        from .. import provenance
        s = self._run("schema")
        return {"kind": "match", "names": s.get("names", []),
                "models": self._tour_ids(s.get("tours", [])),
                "label_a": "Player A", "label_b": "Player B",
                "surfaces": s.get("surfaces", []),
                "freshness": provenance.freshness_warnings(self.id)}

    def simulate_schema(self) -> dict[str, Any]:
        s = self._run("schema")
        return {"models": self._tour_ids(s.get("tours", [])),
                "default_sims": s.get("default_sims", 50000),
                "sim_options": s.get("sim_options", [10000, 50000, 100000]),
                "field_based": True}

    def edge_schema(self) -> dict[str, Any]:
        return {"models": ["ATP", "WTA"],
                "odds_sources": [{"id": "manual",
                                  "label": "Manual tennis/data/odds.csv "
                                           "(tour, surface, best_of, player_a, player_b, odds_a, odds_b)"}],
                "needs_sim_first": False, "options": []}

    def draw_schema(self) -> dict[str, Any]:
        s = self._run("schema")
        return {
            "tours": s.get("tours", []),
            "surfaces": s.get("surfaces", []),
            "names": s.get("names", []),
        }

    # ── capabilities ──
    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        a = params.get("player_a") or params.get("home")
        b = params.get("player_b") or params.get("away")
        return self._run("predict", {**params, "player_a": a, "player_b": b})

    def simulate(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._run("simulate", params)

    def edge(self, params: dict[str, Any]) -> dict[str, Any]:
        from .. import bankroll_store
        bankroll = bankroll_store.current_bankroll()
        peak = bankroll_store.current_peak()
        result = self._run("edge", {**params, "bankroll": bankroll, "peak": peak})
        result["bankroll"] = round(bankroll, 2)
        result["recorded"] = 0
        if params.get("record") and result.get("rows"):
            from datetime import date
            df = pd.DataFrame(result["rows"])
            picks = df[df.get("recommended", False) & (df["stake_gbp"] > 0)].copy()
            if not picks.empty:
                picks["match_date"] = str(date.today())
                picks["source"] = "manual"
                picks = picks.rename(columns={"stake_gbp": "stake"})
                placed = bankroll_store.place_bets(self.id, self.sport, picks)
                result["recorded"] = len(placed)
        return normalize_edge_result(result, source="manual", model="",
                                     sport=self.sport)

    # ── settlement ──
    def grade_open_bets(self, rows: pd.DataFrame) -> dict[int, tuple]:
        """Settle open match-winner bets against matches.csv. A bet on `home`
        (vs `away`) grades against the earliest completed match on/after the
        bet's reference date whose participants are exactly {home, away}.
        Returns {ledger_index: (status, detail)}."""
        if not MATCHES_CSV.exists():
            return {}
        df = pd.read_csv(MATCHES_CSV)
        if df.empty:
            return {}
        df = df.dropna(subset=["winner", "loser", "date"])
        df["_wf"] = df["winner"].map(_fold)
        df["_lf"] = df["loser"].map(_fold)
        df = df.sort_values("date")

        out: dict[int, tuple] = {}
        for idx, r in rows.iterrows():
            if str(r.get("market", "")) not in ("match_winner", "win", ""):
                continue
            home = _fold(r.get("home", ""))
            away = _fold(r.get("away", ""))
            if not home or not away:
                continue
            ref = str(r.get("match_date") or "").strip() or str(r.get("placed_on") or "").strip()
            pair = df[((df["_wf"] == home) & (df["_lf"] == away)) |
                      ((df["_wf"] == away) & (df["_lf"] == home))]
            if ref:
                pair = pair[pair["date"].astype(str) >= ref]
            if pair.empty:
                continue  # match not played / not yet in matches.csv → stay open
            m = pair.iloc[0]
            won = m["_wf"] == home
            detail = f"beat {m['loser']}" if won else f"lost to {m['winner']}"
            out[idx] = ("won" if won else "lost", detail)
        return out
