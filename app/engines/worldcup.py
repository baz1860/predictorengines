"""World Cup 2026 engine adapter.

Wraps the existing modules by importing their functions directly (no subprocess):
  predict  -> predictor.py            (Elo+Poisson scoreline matrix)
  simulate -> simulate.py             (Monte Carlo group + knockout)
  edge     -> edge.py                 (de-vig, EV, quarter-Kelly)
Settlement of recorded bets is delegated here from the suite bankroll store via
grade_open_bets(), reusing bankroll.py's grading against data/results.csv.

Heavy objects (Elo ratings, fitted goal model, blend sources) are computed once
and cached for the session.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import EngineAdapter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = ["blend", "elo", "dc"]
SPORT = "soccer"


class WorldCupAdapter(EngineAdapter):
    id = "worldcup"
    name = "World Cup 2026"
    sport = SPORT
    capabilities = {"predict", "simulate", "edge"}

    def __init__(self) -> None:
        self._loaded = False
        self._ratings = None
        self._beta = None
        self._names: list[str] = []

    # ── predict (Elo+Poisson) ────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import predictor as p
        played, _ = p.load_matches()
        ratings, played = p.compute_elo(played)
        self._ratings = ratings
        self._beta = p.fit_goal_model(played)
        self._names = sorted(ratings.keys())
        self._loaded = True

    def predict_schema(self) -> dict[str, Any]:
        self._ensure_loaded()
        from .. import provenance
        return {"kind": "match", "names": self._names, "models": ["blend"],
                "supports_home": True, "team_label": "Team",
                "freshness": provenance.freshness_warnings(self.id)}

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        import predictor as p
        team1 = (params.get("team1") or "").strip()
        team2 = (params.get("team2") or "").strip()
        home = bool(params.get("home", False))
        if not team1 or not team2:
            raise ValueError("Pick two teams.")
        if team1 == team2:
            raise ValueError("Pick two different teams.")
        for t in (team1, team2):
            if t not in self._ratings:
                raise ValueError(f"Unknown team: {t!r}")
        adv = p.HOME_ADV if home else 0.0
        lam1, lam2, w, d, l, M = p.predict(team1, team2, self._ratings, self._beta, adv)
        scorelines = [{"score": f"{i}-{j}", "prob": round(float(pr), 4)}
                      for i, j, pr in p.top_scorelines(M, 5)]
        venue = f"home advantage: {team1}" if home else "neutral venue"
        return {
            "competitors": [
                {"name": team1, "sub": f"Elo {round(float(self._ratings[team1]))}"},
                {"name": team2, "sub": f"Elo {round(float(self._ratings[team2]))}"}],
            "headline": f"{lam1:.2f} – {lam2:.2f} expected goals · {venue}",
            "outcomes": [
                {"label": f"{team1} win", "prob": round(float(w), 4), "kind": "win"},
                {"label": "Draw", "prob": round(float(d), 4), "kind": "draw"},
                {"label": f"{team2} win", "prob": round(float(l), 4), "kind": "loss"}],
            "table": {
                "title": "Most likely scorelines",
                "columns": [{"key": "score", "label": "Score", "fmt": "text"},
                            {"key": "prob", "label": "Prob", "fmt": "pct"}],
                "rows": scorelines, "bar": "prob"}}

    # ── simulate (Monte Carlo tournament) ────────────────────────────────────
    def simulate_schema(self) -> dict[str, Any]:
        return {"models": MODELS, "default_sims": 10000,
                "sim_options": [2000, 10000, 50000],
                "stages": ["win_group", "reach_R32", "reach_R16", "reach_QF",
                           "reach_SF", "reach_final", "champion"]}

    def simulate(self, params: dict[str, Any]) -> dict[str, Any]:
        from dixoncoles import build_sources
        from simulate import (MatchModel, load_group_matches, simulate_once,
                              GROUPS, TEAM_GROUP)
        model = str(params.get("model", "blend"))
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model}")
        try:
            n = int(params.get("sims", 10000))
        except (TypeError, ValueError):
            raise ValueError("sims must be a number")
        n = max(500, min(n, 100000))

        sources, ratings = build_sources(model)
        mm = MatchModel(sources)
        group_matches = load_group_matches()
        rng = np.random.default_rng(int(params.get("seed", 42)))

        stages = self.simulate_schema()["stages"]
        counts = {t: dict.fromkeys(stages, 0) for ts in GROUPS.values() for t in ts}
        for _ in range(n):
            gw, adv, r16, qf, sf, fin, champ = simulate_once(mm, group_matches, rng)
            for t in gw: counts[t]["win_group"] += 1
            for t in adv: counts[t]["reach_R32"] += 1
            for t in r16: counts[t]["reach_R16"] += 1
            for t in qf: counts[t]["reach_QF"] += 1
            for t in sf: counts[t]["reach_SF"] += 1
            for t in fin: counts[t]["reach_final"] += 1
            counts[champ]["champion"] += 1

        rows = [{"team": t, "group": TEAM_GROUP[t], "elo": round(ratings[t]),
                 **{s: round(counts[t][s] / n, 4) for s in stages}}
                for t in counts]
        rows.sort(key=lambda r: -r["champion"])
        stage_labels = {"win_group": "Win grp", "reach_R32": "R32", "reach_R16": "R16",
                        "reach_QF": "QF", "reach_SF": "SF", "reach_final": "Final",
                        "champion": "Champion"}
        columns = ([{"key": "team", "label": "Team", "fmt": "text"},
                    {"key": "group", "label": "Grp", "fmt": "text"},
                    {"key": "elo", "label": "Elo", "fmt": "num"}] +
                   [{"key": s, "label": stage_labels[s], "fmt": "pct"} for s in stages])
        return {"note": f"{n:,} simulations · {model} model",
                "columns": columns, "rows": rows[:40]}

    # ── edge (value vs bookmaker odds) ───────────────────────────────────────
    def edge_schema(self) -> dict[str, Any]:
        return {"models": MODELS,
                "odds_sources": [
                    {"id": "api", "label": "Live odds (The Odds API)"},
                    {"id": "manual", "label": "Manual odds.csv"}],
                "adjustments": [
                    {"id": "calibrated", "label": "Calibrated"},
                    {"id": "market_blend", "label": "Market blend"},
                    {"id": "context", "label": "Context"},
                    {"id": "squad_adj", "label": "Squad availability"},
                    {"id": "no_portfolio", "label": "Disable portfolio caps"}],
                "api_source_id": "the-odds-api", "has_template": True}

    def edge(self, params: dict[str, Any]) -> dict[str, Any]:
        import edge as E
        from dixoncoles import build_sources
        from predictor import load_matches
        from .. import settings_store, bankroll_store

        model = str(params.get("model", "blend"))
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model}")
        odds_source = str(params.get("odds_source", "api"))
        record = bool(params.get("record", False))
        calibrated = bool(params.get("calibrated", False))
        market_blend = bool(params.get("market_blend", False))
        context = bool(params.get("context", False))
        squad_adj = bool(params.get("squad_adj", False))
        no_portfolio = bool(params.get("no_portfolio", False))

        if squad_adj:
            from squads import adjusted_sources
            sources, ratings, _adj = adjusted_sources(model)
        else:
            sources, ratings = build_sources(model)
        _, upcoming = load_matches()
        neutral_lookup = {(r.home_team, r.away_team): bool(r.neutral)
                          for r in upcoming.itertuples(index=False)}

        if odds_source == "api":
            key = settings_store.odds_api_key("the-odds-api")
            if not key:
                raise ValueError("No Odds API key set. Add one under Settings.")
            odds = E.fetch_api_odds(key, exit_on_error=False)
            src_note = f"Live odds for {len(odds)} matches (The Odds API)"
        else:
            odds = E.load_manual_odds()
            if odds is None:
                raise ValueError("odds.csv not found. Use 'Write template' first, "
                                 "then fill in decimal odds.")
            if odds.empty:
                raise ValueError("odds.csv has no filled-in rows.")
            src_note = f"Manual odds for {len(odds)} matches (odds.csv)"

        bankroll = bankroll_store.current_bankroll()
        modifiers = E.load_edge_modifiers(calibrated, market_blend, context)
        rows = E.edge_rows(odds, sources, ratings, neutral_lookup, modifiers,
                           strict_names=True)
        for row in rows:
            row["date"] = str(row["date"])
            row["stake_gbp"] = round(float(row["kelly_stake"]) * bankroll, 2)

        rows.sort(key=lambda x: -x["ev_per_unit"])

        # Compute the portfolio-recommended picks once, flag the matching rows,
        # and (if recording) place exactly those — no separate ad-hoc filter.
        recorded = 0
        for row in rows:
            row["recommended"] = False
        if rows:
            df = pd.DataFrame(rows)
            ledger = bankroll_store.load_ledger()
            picks = E.top_confident_picks(df, ledger=ledger)
            picks = E.auto_bet_candidates(
                picks, bankroll, portfolio=not no_portfolio,
                peak=bankroll_store.current_peak(), ledger=ledger, verbose=False)
            if not picks.empty:
                rec_keys = set(zip(picks["home"], picks["away"], picks["side"]))
                for row in rows:
                    if (row["home"], row["away"], row["side"]) in rec_keys:
                        row["recommended"] = True
                if record:
                    picks = picks.copy()
                    picks["source"] = odds_source
                    picks["model"] = model
                    placed = bankroll_store.place_bets(self.id, self.sport, picks)
                    recorded = len(placed)

        columns = [
            {"key": "date", "label": "Date", "fmt": "text"},
            {"key": "match", "label": "Match", "fmt": "text"},
            {"key": "bet", "label": "Bet", "fmt": "text"},
            {"key": "odds", "label": "Odds", "fmt": "num"},
            {"key": "p_model", "label": "Model", "fmt": "pct"},
            {"key": "p_book", "label": "Book", "fmt": "pct"},
            {"key": "edge", "label": "Edge", "fmt": "signed_pct"},
            {"key": "ev_per_unit", "label": "EV", "fmt": "num"},
            {"key": "stake_gbp", "label": "Stake", "fmt": "gbp"}]
        active = []
        if calibrated:
            active.append("calibrated")
        if market_blend:
            active.append("market blend")
        if context:
            active.append("context")
        if squad_adj:
            active.append("squad availability")
        if no_portfolio:
            active.append("portfolio caps off")
        if active:
            src_note += " · " + ", ".join(active)
        from .contracts import normalize_edge_result
        result = {"model": model, "odds_source": odds_source, "note": src_note,
                  "bankroll": round(bankroll, 2), "recorded": recorded,
                  "columns": columns, "rows": rows[:200]}
        return normalize_edge_result(result, source=odds_source, model=model,
                                     sport=self.sport)

    def write_odds_template(self) -> dict[str, Any]:
        import edge as E
        from predictor import load_matches
        from .contracts import enrich_template_result
        _, upcoming = load_matches()
        E.write_template(upcoming)
        return enrich_template_result({"path": E.ODDS_CSV.name})

    # ── settlement (called by the suite bankroll store) ──────────────────────
    def grade_open_bets(self, rows: pd.DataFrame) -> dict[int, tuple]:
        """rows: open ledger rows (a slice with the original index preserved).
        Returns {ledger_index: (won: bool|None, "hs-as")}."""
        import bankroll as B
        results = pd.read_csv(ROOT / "data" / "results.csv")
        results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
        results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
        played = results.dropna(subset=["home_score", "away_score"])
        lookup = {(str(r.date), r.home_team, r.away_team): (r.home_score, r.away_score)
                  for r in played.itertuples(index=False)}
        ko90 = B._load_ko_overrides()
        out: dict[int, tuple] = {}
        for i, r in rows.iterrows():
            key = (str(r["match_date"]), r["home"], r["away"])
            if key in ko90:
                hs, as_ = ko90[key]
            elif key in lookup:
                hs, as_ = lookup[key]
            else:
                continue
            won = B.grade(r["side"], hs, as_)
            if won is None:
                continue  # not auto-gradable (outright/special) — leave open
            out[i] = ("won" if won else "lost", f"{int(hs)}-{int(as_)}")
        return out
