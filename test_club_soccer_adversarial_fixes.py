"""Regression coverage for the 2026-07-25 adversarial review fixes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from club_soccer import decision_time_backtest as DTB
from club_soccer import edge as E
from club_soccer import fetch as F
from club_soccer import model as M


def test_rows_from_odds_predicts_each_fixture_once(monkeypatch):
    odds = pd.DataFrame([
        {"date": "2030-01-01", "competition": "Premier League",
         "home": "Arsenal", "away": "Chelsea", "market": market,
         "side": side, "line": line, "odds": price, "bookmaker": "book"}
        for market, line, prices in (
            ("1x2", "", (("home", 2.2), ("draw", 3.4), ("away", 3.2))),
            ("total", 2.5, (("over", 1.9), ("under", 1.9))),
            ("btts", "", (("yes", 1.8), ("no", 2.0))),
        )
        for side, price in prices
    ])
    calls = 0

    def fake_predict(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "probs": {
                "home": 0.45, "draw": 0.27, "away": 0.28,
                "over25": 0.52, "under25": 0.48,
                "btts_yes": 0.54, "btts_no": 0.46,
            },
            "coverage": {"tier": "full", "reliable": True},
        }

    monkeypatch.setattr(M, "predict_match", fake_predict)
    monkeypatch.setattr(M, "load_params", lambda: {})
    E.rows_from_odds(odds)
    assert calls == 1


def test_write_boundary_canonicalises_and_deduplicates(tmp_path):
    rows = pd.DataFrame([
        {"fixture_id": "one", "date": "2026-01-01", "competition": "Bundesliga",
         "home": "FC Bayern München", "away": "Borussia Dortmund", "status": "FIN",
         "home_goals": 2, "away_goals": 1},
        {"fixture_id": "two", "date": "2026-01-01", "competition": "Bundesliga",
         "home": "Bayern Munich", "away": "Borussia Dortmund", "status": "FIN",
         "home_goals": 2, "away_goals": 1},
    ])
    out = F.write_fixtures(rows, tmp_path / "fixtures.csv")
    assert len(out) == 1
    assert out.iloc[0]["home"] == "Bayern Munich"


def test_dormant_competition_adjustments_are_not_fitted(monkeypatch):
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    rows = pd.DataFrame([
        {"fixture_id": str(i), "date": d, "competition": "Test League",
         "type": "league", "home": "A" if i % 2 else "B",
         "away": "B" if i % 2 else "A", "status": "FIN", "neutral": 0,
         "home_goals": i % 4, "away_goals": (i + 1) % 3}
        for i, d in enumerate(dates)
    ])
    calls = 0
    original = M._fit_comp_rho

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(M, "_fit_comp_rho", counted)
    inactive = M.fit(rows)
    assert calls == 0
    assert inactive["comp_adj"] == {}
    assert inactive["league_env"] == {}
    assert inactive["comp_adj_active"] is False

    active = M.fit(rows, competition_adjustments=True)
    assert calls == 1
    assert active["comp_adj_active"] is True
    assert "Test League" in active["comp_adj"]


def test_gate_note_describes_current_per_market_gate():
    note = json.loads(DTB.ARTIFACT.read_text())["note"]
    assert "per-market" in note
    assert "global-AND" not in note
