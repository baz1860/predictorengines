from __future__ import annotations

import pandas as pd
import pytest

from club_soccer import market_diagnostics as D


def _frame():
    return pd.DataFrame([
        {"fixture_identity": "a", "match_date": "2026-01-01",
         "actual_1x2": "home", "over25_actual": 1,
         "p_home": .60, "p_draw": .25, "p_away": .15,
         "p_over25": .60, "p_under25": .40,
         "p_market_home": .50, "p_market_draw": .30, "p_market_away": .20,
         "p_market_prop_home": .50, "p_market_prop_draw": .30,
         "p_market_prop_away": .20,
         "p_market_over": .55, "p_market_under": .45,
         "p_market_prop_over": .55, "p_market_prop_under": .45},
        {"fixture_identity": "b", "match_date": "2026-01-08",
         "actual_1x2": "away", "over25_actual": 0,
         "p_home": .40, "p_draw": .30, "p_away": .30,
         "p_over25": .45, "p_under25": .55,
         "p_market_home": .35, "p_market_draw": .30, "p_market_away": .35,
         "p_market_prop_home": .35, "p_market_prop_draw": .30,
         "p_market_prop_away": .35,
         "p_market_over": .40, "p_market_under": .60,
         "p_market_prop_over": .40, "p_market_prop_under": .60},
    ])


def test_paired_comparison_scores_model_and_market_on_same_rows():
    report = D._paired_summary(_frame(), "1x2")
    assert report["n"] == 2
    assert report["model_log_loss"] == pytest.approx(
        (-__import__("math").log(.60) - __import__("math").log(.30)) / 2,
        abs=1e-6,
    )
    assert report["market_log_loss"] == pytest.approx(
        (-__import__("math").log(.50) - __import__("math").log(.35)) / 2,
        abs=1e-6,
    )


def test_blend_curve_includes_pure_market_and_pure_model():
    curve = D._blend_curve(_frame(), "1x2")
    assert curve[0]["model_weight"] == 0.0
    assert curve[-1]["model_weight"] == 1.0
    assert len(curve) == 21


def test_latest_complete_snapshot_is_strictly_before_kickoff(tmp_path, monkeypatch):
    odds = pd.DataFrame([
        {"snapshot_time": stamp, "match_date": "2026-01-01",
         "competition": "L", "home": "H", "away": "A", "market": "1x2",
         "side": side, "odds_median": price, "n_books": 1, "disp": 0}
        for stamp, prices in [
            ("2026-01-01T10:00:00+00:00", {"home": 2, "draw": 3, "away": 4}),
            ("2026-01-01T13:00:00+00:00", {"home": 9, "draw": 9, "away": 9}),
        ]
        for side, price in prices.items()
    ])
    path = tmp_path / "odds.csv"
    odds.to_csv(path, index=False)
    monkeypatch.setattr(D, "ODDS", path)
    forecast = pd.DataFrame([{
        "fixture_identity": "f", "match_date": "2026-01-01", "competition": "L",
        "home": "H", "away": "A", "kickoff_utc": "2026-01-01T12:00:00+00:00",
        "actual_1x2": "home", "over25_actual": 1,
        "p_home": .5, "p_draw": .3, "p_away": .2,
    }])
    joined = D._join_market(forecast, "1x2")
    assert len(joined) == 1
    assert joined.iloc[0]["odds_home"] == 2
    assert joined.iloc[0]["snapshot_time"].hour == 10


def test_challenger_has_no_promotion_authority():
    challenger = D._challenger_split(_frame(), "1x2")
    assert challenger["promotion_authority"] is False


def test_run_is_read_only_unless_write_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "REPORT", tmp_path / "report.json")
    monkeypatch.setattr(D, "FORECASTS", tmp_path / "missing_forecasts.csv")
    monkeypatch.setattr(D, "RESULTS", tmp_path / "missing_results.csv")
    monkeypatch.setattr(D, "ODDS", tmp_path / "missing_odds.csv")
    monkeypatch.setattr(D, "_selection_diagnostics", lambda: {"n": 0})
    report = D.run(write=False)
    assert report["status"] == "diagnostic_only_no_promotion_authority"
    assert not D.REPORT.exists()
    D.run(write=True)
    assert D.REPORT.exists()
