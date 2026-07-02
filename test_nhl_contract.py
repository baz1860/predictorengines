#!/usr/bin/env python3
"""NHL engine contract and settlement smoke tests.

Run: python3 test_nhl_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engines import registry
from contracts import validate_edge_rows, validate_prediction
from nhl import backtest as B
from nhl import model as M

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_model_probabilities():
    pred = M.predict_match("Toronto Maple Leafs", "Boston Bruins")
    check("moneyline sums to one",
          abs(pred["p_home"] + pred["p_away"] - 1.0) < 1e-9, str(pred))
    check("expected total is NHL-like",
          4.0 <= pred["total"] <= 8.0, str(pred["total"]))
    over, push = M.market_probs(pred, "total", "over", 6.5)
    under, push_under = M.market_probs(pred, "total", "under", 6.5)
    check("total market partitions probability",
          abs(over + under + push + push_under - 1.0) < 1e-6,
          f"{over} {under} {push} {push_under}")


def test_adapter_contracts():
    ad = registry.get("nhl")
    info = ad.info()
    check("adapter registered", info["id"] == "nhl" and "predict" in info["capabilities"], str(info))
    pred = ad.predict({"team1": "Toronto Maple Leafs", "team2": "Boston Bruins", "model": "blend"})
    try:
        validate_prediction(pred)
        pred_ok = True
    except Exception as e:  # noqa: BLE001
        pred_ok = False
        detail = str(e)
    else:
        detail = ""
    check("prediction contract", pred_ok, detail)

    edge = ad.edge({"odds_source": "manual", "record": False})
    rows = edge.get("rows") or []
    try:
        validate_edge_rows(rows)
        edge_ok = True
    except Exception as e:  # noqa: BLE001
        edge_ok = False
        detail = str(e)
    else:
        detail = ""
    check("edge contract", edge_ok, detail)
    check("edge covers NHL markets",
          {"ml", "spread", "total"}.issubset({r["market"] for r in rows}),
          str({r["market"] for r in rows}))


def test_settlement():
    ad = registry.get("nhl")
    rows = pd.DataFrame([
        {"match_date": "2026-04-12", "home": "Toronto Maple Leafs",
         "away": "Boston Bruins", "market": "ml", "side": "home",
         "line": "", "bet": "ML home"},
        {"match_date": "2026-04-12", "home": "Toronto Maple Leafs",
         "away": "Boston Bruins", "market": "spread", "side": "home",
         "line": "-1.5", "bet": "PUCK LINE home -1.5"},
        {"match_date": "2026-04-12", "home": "Toronto Maple Leafs",
         "away": "Boston Bruins", "market": "total", "side": "under",
         "line": "6.5", "bet": "TOTAL under 6.5"},
    ])
    graded = ad.grade_open_bets(rows)
    check("grades NHL moneyline", graded.get(0, ("",))[0] == "won", str(graded))
    check("grades NHL puck line", graded.get(1, ("",))[0] == "won", str(graded))
    check("grades NHL total", graded.get(2, ("",))[0] == "won", str(graded))


def test_backtest():
    report = B.run_backtest(B.load_results(), model="blend", min_edge=0.0)
    summary = report["summary"]
    check("backtest sees completed games", summary["games"] >= 2, str(summary))
    check("backtest has forecast metrics",
          all(k in summary for k in ("accuracy", "brier", "logloss", "margin_mae", "total_mae")),
          str(summary))
    check("backtest emits row details", len(report["rows"]) == summary["games"])
    check("odds-backed backtest can produce bets",
          report["betting"]["bets"] >= 1, str(report["betting"]))


def main() -> int:
    print("NHL engine tests")
    test_model_probabilities()
    test_adapter_contracts()
    test_settlement()
    test_backtest()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
