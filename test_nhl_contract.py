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
from nhl import odds_history as OH
from nhl import oddspapi as OPA
from nhl import the_odds_api as TOA

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def expect_raises(name, fn, needle: str):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(name, needle in str(e), str(e))
    else:
        check(name, False, "expected exception")


def _history_results() -> pd.DataFrame:
    return B.load_results(ROOT / "nhl/data/results_2025_26.csv").head(1)


def _valid_history_rows() -> pd.DataFrame:
    event_id = OH.id_key(_history_results().iloc[0]["game_id"])
    return pd.DataFrame([
        {"event_id": event_id, "game_date": "2025-10-07",
         "start_time_utc": "2025-10-07T21:00:00Z",
         "captured_at_utc": "2025-10-07T15:00:00Z",
         "bookmaker": "unit_book", "market": "ml", "side": "home",
         "line": "", "decimal_odds": 2.40, "source": "unit_test"},
        {"event_id": event_id, "game_date": "2025-10-07",
         "start_time_utc": "2025-10-07T21:00:00Z",
         "captured_at_utc": "2025-10-07T15:00:00Z",
         "bookmaker": "unit_book", "market": "ml", "side": "away",
         "line": "", "decimal_odds": 1.60, "source": "unit_test"},
    ])


def test_model_probabilities():
    pred = M.predict_match("Toronto Maple Leafs", "Boston Bruins")
    check("moneyline sums to one",
          abs(pred["p_home"] + pred["p_away"] - 1.0) < 1e-9, str(pred))
    check("expected total is NHL-like",
          4.0 <= pred["total"] <= 8.0, str(pred["total"]))
    check("total includes OT/SO deciding goal expectation",
          pred["total"] > pred["regulation_total"]
          and abs(pred["total"] - pred["regulation_total"] - pred["p_reg_tie"]) < 1e-9,
          str(pred))
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
    check("staking gate disables NHL recommendations",
          all(not r.get("recommended") and float(r.get("stake_gbp", 0.0)) == 0.0 for r in rows),
          str(rows[:2]))


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
          all(k in summary for k in ("accuracy", "brier", "logloss", "margin_mae", "total_mae",
                                     "home_rate_brier", "beats_trivial_baselines")),
          str(summary))
    check("backtest default is walk-forward", summary["mode"] == "walk_forward", str(summary))
    check("backtest emits row details", len(report["rows"]) == summary["games"])
    check("odds-backed backtest flags manual ROI as non-decision-grade",
          report["betting"].get("decision_grade") is False, str(report["betting"]))


def test_odds_history():
    results = _history_results()
    valid = _valid_history_rows()
    odds = OH.validate_odds_history(valid, results_df=results)
    latest = OH.latest_pre_game_quotes(odds)
    check("valid odds history loads", len(odds) == 2 and len(latest) == 2, str(odds))

    bad_time = valid.copy()
    bad_time.loc[0, "captured_at_utc"] = "2025-10-07T21:00:00Z"
    expect_raises("rejects post-start odds", lambda: OH.validate_odds_history(bad_time, results_df=results),
                  "before start_time")

    missing_side = valid.iloc[[0]].copy()
    expect_raises("rejects missing complementary side",
                  lambda: OH.validate_odds_history(missing_side, results_df=results),
                  "incomplete")

    duplicate_side = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
    expect_raises("rejects duplicate side",
                  lambda: OH.validate_odds_history(duplicate_side, results_df=results),
                  "duplicate")

    unknown_event = valid.copy()
    unknown_event["event_id"] = "unknown-game"
    expect_raises("rejects unknown event id",
                  lambda: OH.validate_odds_history(unknown_event, results_df=results),
                  "unknown event_id")

    report = B.run_backtest(results, model="blend", min_edge=0.0, odds_history=valid)
    betting = report["betting"]
    check("odds-history backtest is decision-grade",
          betting.get("decision_grade") is True and betting.get("odds_source") == "odds_history",
          str(betting))
    check("odds-history backtest segments markets",
          betting["bets"] >= 1 and "ml" in betting.get("by_market", {}),
          str(betting))


def test_the_odds_api_provider():
    results = _history_results()
    payload = {
        "timestamp": "2025-10-07T15:00:00Z",
        "_snapshot_requested_at_utc": "2025-10-07T15:00:00Z",
        "data": [{
            "id": "oddsapi_event_1",
            "sport_key": "icehockey_nhl",
            "commence_time": "2025-10-07T21:00:00Z",
            "home_team": "Florida Panthers",
            "away_team": "Chicago Blackhawks",
            "bookmakers": [{
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": "2025-10-07T14:58:00Z",
                "markets": [
                    {"key": "h2h", "last_update": "2025-10-07T14:58:00Z",
                     "outcomes": [
                         {"name": "Florida Panthers", "price": 2.4},
                         {"name": "Chicago Blackhawks", "price": 1.6},
                     ]},
                    {"key": "spreads", "last_update": "2025-10-07T14:58:00Z",
                     "outcomes": [
                         {"name": "Florida Panthers", "price": 4.0, "point": -1.5},
                         {"name": "Chicago Blackhawks", "price": 1.28, "point": 1.5},
                     ]},
                    {"key": "totals", "last_update": "2025-10-07T14:58:00Z",
                     "outcomes": [
                         {"name": "Over", "price": 2.05, "point": 5.5},
                         {"name": "Under", "price": 1.85, "point": 5.5},
                     ]},
                ],
            }],
        }],
    }
    rows, unmatched = TOA.rows_from_snapshot(payload, results)
    odds = OH.validate_odds_history(pd.DataFrame(rows), results_df=results)
    check("The Odds API snapshot maps to NHL game id",
          not unmatched and len(odds) == 6
          and set(odds["market"]) == {"ml", "spread", "total"}
          and set(odds["event_id"]) == {OH.id_key(results.iloc[0]["game_id"])},
          f"rows={rows} unmatched={unmatched}")


def test_oddspapi_provider():
    results = _history_results()
    match = {
        "event_id": OH.id_key(results.iloc[0]["game_id"]),
        "game_date": str(results.iloc[0]["date"])[:10],
    }
    fixture = {
        "fixtureId": "id1500000000000001",
        "sportId": 15,
        "tournamentId": 1234,
        "tournamentName": "NHL",
        "startTime": "2025-10-07T21:00:00.000Z",
        "participant1Name": "Florida Panthers",
        "participant2Name": "Chicago Blackhawks",
    }
    payload = {
        "fixtureId": "id1500000000000001",
        "bookmakers": {
            "pinnacle": {
                "markets": {
                    "151": {"outcomes": {
                        "151": {"players": {"0": [
                            {"createdAt": "2025-10-07T14:00:00+00:00", "price": 2.50, "active": True},
                            {"createdAt": "2025-10-07T16:00:00+00:00", "price": 2.40, "active": True},
                        ]}},
                        "152": {"players": {"0": [
                            {"createdAt": "2025-10-07T14:05:00+00:00", "price": 1.60, "active": True},
                        ]}},
                    }},
                    "15228": {"outcomes": {
                        "15228": {"players": {"0": [
                            {"createdAt": "2025-10-07T15:10:00+00:00", "price": 4.00, "active": True},
                        ]}},
                    }},
                    "15240": {"outcomes": {
                        "15241": {"players": {"0": [
                            {"createdAt": "2025-10-07T15:11:00+00:00", "price": 1.28, "active": True},
                        ]}},
                    }},
                    "15178": {"outcomes": {
                        "15178": {"players": {"0": [
                            {"createdAt": "2025-10-07T15:00:00+00:00", "price": 2.05, "active": True},
                        ]}},
                        "15179": {"players": {"0": [
                            {"createdAt": "2025-10-07T15:01:00+00:00", "price": 1.85, "active": True},
                        ]}},
                    }},
                },
            },
        },
    }
    rows = OPA.rows_from_history(payload, fixture, match)
    odds = OH.validate_odds_history(pd.DataFrame(rows), results_df=results)
    latest = OH.latest_pre_game_quotes(odds)
    check("OddsPapi reconstructs paired snapshots",
          len(odds) == 8 and len(latest) == 6
          and set(odds["source"]) == {"oddspapi"}
          and set(latest["market"]) == {"ml", "spread", "total"},
          f"rows={rows}")
    ml_latest = latest[(latest["market"] == "ml") & (latest["side"] == "home")]
    check("OddsPapi carries forward latest active side prices",
          len(ml_latest) == 1 and abs(float(ml_latest.iloc[0]["decimal_odds"]) - 2.40) < 1e-9
          and str(ml_latest.iloc[0]["captured_at_utc"]) == "2025-10-07T16:00:00Z",
          str(ml_latest))


def main() -> int:
    print("NHL engine tests")
    test_model_probabilities()
    test_adapter_contracts()
    test_settlement()
    test_backtest()
    test_odds_history()
    test_the_odds_api_provider()
    test_oddspapi_provider()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
