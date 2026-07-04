#!/usr/bin/env python3
"""NFL engine contract and settlement smoke tests.

Run: python3 test_nfl_contract.py
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
from nfl.predictor import Models, blend_predict

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
    models = Models.load()
    pred = blend_predict(models, "Kansas City Chiefs", "Buffalo Bills")
    check("moneyline sums to one",
          abs(pred["p1"] + pred["p2"] + pred["p_tie"] - 1.0) < 1e-9, str(pred))
    check("expected total is NFL-like", 30.0 <= pred["total"] <= 65.0, str(pred["total"]))
    check("tie prob is small (NFL ties are rare)", pred["p_tie"] < 0.03, str(pred["p_tie"]))


def test_adapter_contracts():
    ad = registry.get("nfl")
    info = ad.info()
    check("adapter registered", info["id"] == "nfl" and "predict" in info["capabilities"], str(info))
    pred = ad.predict({"team1": "Kansas City Chiefs", "team2": "Buffalo Bills", "model": "blend"})
    try:
        validate_prediction(pred)
        pred_ok = True
        detail = ""
    except Exception as e:  # noqa: BLE001
        pred_ok = False
        detail = str(e)
    check("prediction contract", pred_ok, detail)

    edge = ad.edge({"odds_source": "manual", "record": False})
    rows = edge.get("rows") or []
    try:
        validate_edge_rows(rows)
        edge_ok = True
        detail = ""
    except Exception as e:  # noqa: BLE001
        edge_ok = False
        detail = str(e)
    check("edge contract", edge_ok, detail)
    check("edge covers NFL markets",
          {"ml", "spread", "total"}.issubset({r["market"] for r in rows}),
          str({r["market"] for r in rows}))


def test_settlement():
    ad = registry.get("nfl")
    rows = pd.DataFrame([
        {"match_date": "2020-01-01", "home": "Kansas City Chiefs",
         "away": "Buffalo Bills", "market": "ml", "side": "home",
         "line": "", "bet": "ML home"},
        {"match_date": "2020-01-01", "home": "Kansas City Chiefs",
         "away": "Buffalo Bills", "market": "spread", "side": "home",
         "line": "-3.0", "bet": "SPREAD home -3.0"},
        {"match_date": "2020-01-01", "home": "Kansas City Chiefs",
         "away": "Buffalo Bills", "market": "spread", "side": "home",
         "line": "-14.0", "bet": "SPREAD home -14.0"},
        {"match_date": "2020-01-01", "home": "Kansas City Chiefs",
         "away": "Buffalo Bills", "market": "total", "side": "under",
         "line": "60.5", "bet": "TOTAL under 60.5"},
    ])
    # Find a real, played KC-Buffalo game to graft the rows onto so grading has
    # something concrete (and deterministic) to settle against.
    import pandas as _pd
    games = _pd.read_csv(ROOT / "nfl" / "data" / "games.csv")
    g = games[(games["home"] == "Kansas City Chiefs") & (games["away"] == "Buffalo Bills")]
    check("have at least one historical KC-Buffalo game to grade against", not g.empty, "")
    if g.empty:
        return
    g = g.iloc[0]
    margin = int(g["home_score"] - g["away_score"])
    total = int(g["home_score"] + g["away_score"])
    rows["match_date"] = "1999-01-01"  # before any known game, so date filter passes

    graded = ad.grade_open_bets(rows)
    check("grades NFL moneyline",
          graded.get(0, ("",))[0] == ("won" if margin > 0 else "lost"), str(graded.get(0)))
    check("grades NFL spread (non-push line)",
          graded.get(1, ("",))[0] in ("won", "lost"), str(graded.get(1)))
    # push case: pick a spread line exactly equal to the actual margin
    rows.loc[2, "line"] = str(-float(margin))
    graded2 = ad.grade_open_bets(rows)
    check("grades NFL spread push", graded2.get(2, ("",))[0] == "push", str(graded2.get(2)))
    check("grades NFL total",
          graded.get(3, ("",))[0] == ("won" if total < 60.5 else "lost"), str(graded.get(3)))

    # ML tie case: a synthetic 20-20 game (NFL ties are graded as a push on
    # the moneyline in this engine's convention — verify via _grade_row directly).
    from app.engines.nfl import NFLAdapter
    tie_row = {"market": "ml", "side": "home", "line": "", "bet": "ML home"}
    status = NFLAdapter._grade_row(tie_row, margin=0, total=40)
    check("ML tie is not graded as a win", status != "won", str(status))


def main() -> int:
    print("NFL engine tests")
    test_model_probabilities()
    test_adapter_contracts()
    test_settlement()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
