"""Golf engine adapter.

Drives golf/ via an isolated subprocess runner (golf_runner.py). Capabilities
are Simulate (field projection: win/T5/T10/T20/cut), Edge (outright, place, cut,
matchup & 3-ball markets, calibrated + market-blended), and Predict (head-to-head
matchup probabilities). The UI is capability-driven.

Bets auto-settle against the latest completed event in golf/data/rounds.csv
(refreshed by `fetch.py --accumulate`): win / top-N / make-cut grade off the
recomputed finish, matchups & 3-balls off finishing order with missed-cut
players ranked behind survivors. Outrights for an event not yet in rounds.csv
stay open until the results land.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import EngineAdapter
from ._subprocess import run_engine

ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "golf"
RUNNER = Path(__file__).resolve().parent / "runners" / "golf_runner.py"
ROUNDS_CSV = ENGINE_DIR / "data" / "rounds.csv"


def _event_results(ev: pd.DataFrame) -> dict[str, dict]:
    """Per-player results for one event's rounds slice:
    {player: {made_cut, finish, rank_score}}. rank_score ranks survivors by
    72-hole total, missed-cut players behind everyone by their 36-hole score."""
    g = ev.groupby("player")
    total = g["score_to_par"].sum()
    made = g["made_cut"].max()
    r36 = (ev[ev["round"].isin([1, 2])].groupby("player")["score_to_par"].sum())
    finishers = total[made == 1]
    rank = finishers.rank(method="min")
    out = {}
    for p in total.index:
        mc = int(made.loc[p])
        fin = int(rank.loc[p]) if (mc == 1 and p in rank.index) else 999
        rs = float(total.loc[p]) if mc == 1 else 1e6 + float(r36.get(p, 0.0))
        out[p] = {"made_cut": mc, "finish": fin, "rank_score": rs}
    return out


def _completed_events() -> list[tuple[str, str, dict[str, dict]]]:
    """Every completed event in rounds.csv as (end_date, tournament_id, results),
    sorted ascending by end date. Empty if rounds.csv is missing/empty."""
    if not ROUNDS_CSV.exists():
        return []
    df = pd.read_csv(ROUNDS_CSV)
    if df.empty:
        return []
    out = []
    for tid, ev in df.groupby("tournament_id"):
        end_date = str(ev["date"].max())
        out.append((end_date, str(tid), _event_results(ev)))
    out.sort(key=lambda t: t[0])
    return out


def _grade_one(side: str, player: str, res: dict[str, dict]):
    """(status, detail) for one bet, or None to leave it open."""
    side = str(side)
    if side.startswith("matchup:"):
        a, _, b = side[len("matchup:"):].partition("|")
        if a not in res or b not in res:
            return None
        sa, sb = res[a]["rank_score"], res[b]["rank_score"]
        if abs(sa - sb) < 1e-9:
            return ("push", "tie")
        return ("won", f"beat {b}") if sa < sb else ("lost", f"lost to {b}")
    if side.startswith("3ball:"):
        members = side[len("3ball:"):].split("|")
        if any(m not in res for m in members):
            return None
        scores = {m: res[m]["rank_score"] for m in members}
        best = min(scores.values())
        if list(scores.values()).count(best) > 1:
            return ("push", "tie")
        return ("won", "won group") if scores[player] == best else ("lost", "lost group")
    # outright / place / cut markets
    pr = res.get(player)
    if pr is None:
        return None
    if side == "win":
        return ("won", "winner") if pr["finish"] == 1 else ("lost", f"fin {pr['finish']}")
    if side == "cut":
        return ("won", "made cut") if pr["made_cut"] else ("lost", "missed cut")
    n = {"top5": 5, "top10": 10, "top20": 20}.get(side)
    if n is not None:
        ok = pr["made_cut"] == 1 and pr["finish"] <= n
        return ("won", f"fin {pr['finish']}") if ok else ("lost", f"fin {pr['finish']}")
    return None


class GolfAdapter(EngineAdapter):
    id = "golf"
    name = "Golf (PGA Tour)"
    sport = "golf"
    capabilities = {"simulate", "edge", "predict"}

    def _run(self, command, params=None, timeout=180):
        return run_engine(ENGINE_DIR, RUNNER, command, params, timeout=timeout)

    def predict_schema(self) -> dict[str, Any]:
        # head-to-head matchup: pick two players from the field
        s = self._run("schema")
        return {"kind": "match", "names": s.get("names", []), "models": [],
                "label_a": "Player A", "label_b": "Player B"}

    def simulate_schema(self) -> dict[str, Any]:
        s = self._run("schema")
        return {"models": [], "default_sims": s.get("default_sims", 50000),
                "sim_options": s.get("sim_options", [10000, 50000, 100000]),
                "field_based": True}

    def edge_schema(self) -> dict[str, Any]:
        return {"models": [],
                "odds_sources": [{"id": "manual",
                                  "label": "Manual golf/data/odds.csv (+ matchups.csv / threeballs.csv)"}],
                "needs_sim_first": False,
                "options": [{"id": "calibrated", "label": "Calibrated", "default": True},
                            {"id": "market_blend", "label": "Market blend", "default": True}]}

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        # base interface passes the two picks as home/away or player_a/player_b
        a = params.get("player_a") or params.get("home")
        b = params.get("player_b") or params.get("away")
        return self._run("predict", {**params, "player_a": a, "player_b": b})

    def simulate(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._run("simulate", params)

    def edge(self, params: dict[str, Any]) -> dict[str, Any]:
        from .. import bankroll_store
        from .contracts import normalize_edge_result
        bankroll = bankroll_store.current_bankroll()
        peak = bankroll_store.current_peak()
        result = self._run("edge", {**params, "bankroll": bankroll, "peak": peak})
        result["bankroll"] = round(bankroll, 2)
        result["recorded"] = 0
        if params.get("record") and result.get("rows"):
            from datetime import date
            df = pd.DataFrame(result["rows"])
            # record only the portfolio-recommended bets, at their staked amount
            picks = df[df.get("recommended", False) & (df["stake_gbp"] > 0)].copy()
            if not picks.empty:
                today = str(date.today())
                picks["home"] = picks["player"]
                picks["away"] = ""
                # Stamp the placement date so each week's bet is a distinct event
                # (event_id = date|player) and settlement has a date floor; this
                # is what keeps a stale outright from grading against a later
                # event the player also happens to be in.
                picks["match_date"] = today
                picks["source"] = "manual"
                picks = picks.rename(columns={"stake_gbp": "stake"})
                placed = bankroll_store.place_bets(self.id, self.sport, picks)
                result["recorded"] = len(placed)
        return normalize_edge_result(result, source="manual", model="",
                                     sport=self.sport)

    def grade_open_bets(self, rows: pd.DataFrame) -> dict[int, tuple]:
        """Settle open golf bets EVENT-SAFELY. Each bet is graded only against the
        earliest completed event on/after the bet's reference date whose field
        actually contains the bet's participant(s) — never simply "the latest
        event". A bet placed for an event not yet in rounds.csv (or whose event
        hasn't happened yet) stays open. Returns {ledger_index: (status, detail)}.
        """
        events = _completed_events()
        if not events:
            return {}
        out: dict[int, tuple] = {}
        for idx, r in rows.iterrows():
            side = r.get("side", "")
            player = str(r.get("home", ""))
            ref = str(r.get("match_date") or "").strip() or str(r.get("placed_on") or "").strip()
            for end_date, _tid, res in events:
                if ref and end_date < ref:
                    continue  # event finished before this bet was for — not its event
                verdict = _grade_one(side, player, res)
                if verdict is not None:
                    out[idx] = verdict
                    break      # earliest qualifying event = the bet's event
            # no qualifying event yet → leave open (stale/future outright)
        return out
