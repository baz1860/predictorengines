#!/usr/bin/env python3
"""Regression tests for the Club Soccer engine.

Run: python3 test_club_soccer.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CLUB = ROOT / "club_soccer"
if str(CLUB) not in sys.path:
    sys.path.insert(0, str(CLUB))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import competitions as C
import edge as E
import model as M
from api_keys import get_key
from app.engines.club_soccer import ClubSoccerAdapter

_fails: list[str] = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def test_registry():
    print("1. competition registry")
    expected = {
        "Premier League", "Championship", "League One", "League Two",
        "Scottish Premiership", "Scottish Championship", "Scottish League One",
        "Scottish League Two", "Bundesliga", "Serie A", "Ligue 1", "La Liga",
        "Champions League", "Europa League", "Conference League", "UEFA Super Cup",
        "FA Cup", "EFL Cup", "Scottish Cup", "Scottish League Cup",
        "DFB-Pokal", "Coppa Italia", "Coupe de France", "Copa del Rey",
    }
    names = set(C.names())
    check("contains every requested competition", expected <= names)
    check("public rows include API-Football IDs", all(r["api_football_id"] for r in C.public_rows()))


def test_api_key_lookup():
    print("2. API key lookup")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "api_keys.json"
        p.write_text(json.dumps({"api-football": "file-key", "api_football": "alias-key"}))
        old = os.environ.get("API_FOOTBALL_KEY")
        os.environ["API_FOOTBALL_KEY"] = "env-key"
        try:
            check("environment wins", get_key("api-football", env="API_FOOTBALL_KEY", path=p) == "env-key")
        finally:
            if old is None:
                os.environ.pop("API_FOOTBALL_KEY", None)
            else:
                os.environ["API_FOOTBALL_KEY"] = old
        check("file lookup works", get_key("api-football", path=p) in {"file-key", "alias-key"})
        check("alias lookup works", get_key("football", path=p) in {"file-key", "alias-key"})


def test_model_math():
    print("3. model probabilities")
    params = M.fit()
    pred = M.predict("Arsenal", "Chelsea", "Premier League", params=params)
    p = pred["probs"]
    check("1X2 probabilities sum to one", abs(p["home"] + p["draw"] + p["away"] - 1.0) < 0.002)
    check("totals probabilities sum to one", abs(p["over25"] + p["under25"] - 1.0) < 0.002)
    check("BTTS probabilities sum to one", abs(p["btts_yes"] + p["btts_no"] - 1.0) < 0.002)
    check("score matrix normalizes", abs(float(pred["matrix"].sum()) - 1.0) < 1e-9)
    try:
        M.predict("Not A Club", "Chelsea", "Premier League", params=params)
        check("unknown team raises", False)
    except ValueError as e:
        check("unknown team raises", "Unknown team" in str(e))


def test_edge_and_settlement():
    print("4. edge and settlement")
    check("1X2 de-vig sums to one", abs(float(E.devig([2.2, 3.3, 3.1]).sum()) - 1.0) < 1e-12)
    odds = E.load_odds()
    rows = E.rows_from_odds(odds, bankroll=100)
    markets = {r["market"] for r in rows}
    check("edge covers 1X2 totals BTTS", {"1x2", "total", "btts"} <= markets)
    check("API odds mapper handles home team names",
          E._map_api_bet("Match Winner", "Arsenal", "2.20", "Arsenal", "Chelsea") == ("1x2", "home", "", 2.2))
    check("API odds mapper handles over 2.5",
          E._map_api_bet("Goals Over/Under", "Over 2.5", "1.90") == ("total", "over", 2.5, 1.9))
    check("API odds mapper handles BTTS",
          E._map_api_bet("Both Teams Score", "Yes", "1.80") == ("btts", "yes", "", 1.8))
    try:
        json.dumps({"rows": rows}, allow_nan=False)
        check("edge rows are strict JSON safe", True)
    except ValueError:
        check("edge rows are strict JSON safe", False)
    check("home win grades won", E.grade("home", "1x2", "", 2, 1) == "won")
    check("draw grades won", E.grade("draw", "1x2", "", 1, 1) == "won")
    check("away win grades won", E.grade("away", "1x2", "", 0, 2) == "won")
    check("over grades won", E.grade("over", "total", 2.5, 2, 1) == "won")
    check("under grades won", E.grade("under", "total", 2.5, 1, 0) == "won")
    check("BTTS grades won", E.grade("yes", "btts", "", 2, 1) == "won")
    check("BTTS no grades won", E.grade("no", "btts", "", 2, 0) == "won")


def test_runner_and_adapter():
    print("5. runner and adapter")
    adapter = ClubSoccerAdapter()
    schema = adapter.predict_schema()
    edge_schema = adapter.edge_schema()
    check("schema exposes filters", bool(schema.get("filters")))
    check("schema exposes date range filters", {"date_from", "date_to"} <= {f["id"] for f in schema.get("filters", [])})
    check("edge schema exposes The Odds API fallback",
          "the-odds-api" in {s["id"] for s in edge_schema.get("odds_sources", [])})
    pred = adapter.predict({"team1": "Arsenal", "team2": "Chelsea", "competition": "Premier League"})
    check("adapter predict returns outcomes", len(pred.get("outcomes", [])) == 3)
    edge = adapter.edge({"odds_source": "manual", "model": "ensemble"})
    check("adapter edge returns columns", bool(edge.get("columns")))
    filtered = adapter.edge({"odds_source": "manual", "model": "ensemble", "date_from": "2026-06-21"})
    check("adapter edge honors date_from", all(r["date"] >= "2026-06-21" for r in filtered.get("rows", [])))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "app" / "engines" / "runners" / "club_soccer_runner.py"), "schema"],
        input="{}",
        cwd=str(CLUB),
        env={**os.environ, "PYTHONPATH": str(CLUB) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    check("runner returns valid JSON", "names" in data and "error" not in data)


if __name__ == "__main__":
    test_registry()
    test_api_key_lookup()
    test_model_math()
    test_edge_and_settlement()
    test_runner_and_adapter()
    print()
    if _fails:
        print(f"{len(_fails)} FAILURE(S): " + ", ".join(_fails))
        raise SystemExit(1)
    print("All Club Soccer tests passed.")
