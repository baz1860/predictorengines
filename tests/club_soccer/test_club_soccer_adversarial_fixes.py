"""Regression coverage for the 2026-07-25 adversarial review fixes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from club_soccer import decision_time_backtest as DTB
from club_soccer import edge as E
from club_soccer import fetch as F
from club_soccer import health as H
from club_soccer import model as M
from club_soccer.club_identity import canonical_id


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


def test_market_blend_is_applied_once_at_the_pricing_boundary(monkeypatch):
    odds = pd.DataFrame([
        {"date": "2030-01-01", "competition": "Premier League",
         "home": "Arsenal", "away": "Chelsea", "market": "1x2",
         "side": side, "line": "", "odds": price, "bookmaker": "book"}
        for side, price in (("home", 2.2), ("draw", 3.4), ("away", 3.2))
    ])
    monkeypatch.setattr(M, "load_params", lambda: {})
    monkeypatch.setattr(M, "predict_match", lambda *_args, **_kwargs: {
        "probs": {"home": 0.45, "draw": 0.27, "away": 0.28,
                  "over25": 0.5, "under25": 0.5,
                  "btts_yes": 0.5, "btts_no": 0.5},
        "coverage": {"tier": "full", "reliable": True},
    })
    monkeypatch.setattr(E, "apply_evidence_gate", lambda _rows: False)
    from app import market_blend as blend
    calls = 0

    def apply(rows, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        for row in rows:
            row["market_blend_w"] = 0.5
        return 0.5

    monkeypatch.setattr(blend, "apply_blend_to_rows", apply)
    E.rows_from_odds(odds, market_blend=True)
    assert calls == 1


def test_invalid_pricing_model_fails_before_loading_artifacts(monkeypatch):
    monkeypatch.setattr(
        M, "load_params",
        lambda: pytest.fail("invalid configuration reached artifact loading"),
    )
    with pytest.raises(ValueError, match="Unknown model"):
        E.rows_from_odds(pd.DataFrame(), model_name="retired-model")


def test_shot_component_collapse_is_price_preserving():
    """The `xg` component is exactly the 50/50 mixture of the pair it replaced.

    Originally this compared the whole production blend against
    `0.20*goals + 0.40*elo + 0.20*long_run + 0.20*recent`. E1 promoted on
    2026-08-16 and moved the goals weight to the pooled component, so the
    production blend is now deliberately different and that comparison can no
    longer hold.

    What the test was really protecting survives untouched: collapsing the old
    xg/xgf pair into one component was supposed to remove a fake degree of
    freedom WITHOUT changing a price, and that is an identity about the
    component itself, independent of whatever weight the blend gives it. So
    assert the identity directly — it stays meaningful across future blend
    changes, which the old form did not.
    """
    params = M.load_params()
    home, away, competition = "Arsenal", "Chelsea", "Premier League"
    parts = M.component_matrices(params, home, away, competition, False)
    rho = M._comp_rho(params, competition)
    long_run = M.score_matrix(*M._lambdas_xg(
        params, home, away, competition, False, form=False
    ), rho)
    recent = M.score_matrix(*M._lambdas_xg(
        params, home, away, competition, False, form=True
    ), rho)

    assert np.allclose(parts["xg"], (long_run + recent) / 2.0,
                       atol=1e-15, rtol=0)
    # And the weight it carries is still the sum of the two it replaced.
    assert M.DEFAULT_ENSEMBLE_W["xg"] == pytest.approx(0.40)


def test_offline_health_never_calls_remote_freshness(monkeypatch):
    from club_soccer import seed_fdcouk_leagues as source
    monkeypatch.setattr(
        source, "refresh_health",
        lambda: pytest.fail("offline health attempted a network refresh"),
    )
    assert H.run_checks(network=False)["ok"] is True


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
    assert out.iloc[0]["home_club_id"]
    assert out.iloc[0]["away_club_id"]
    assert out.iloc[0]["home_club_id"] != out.iloc[0]["away_club_id"]


def test_continental_club_ids_do_not_hash_missing_country_as_nan(tmp_path):
    rows = pd.DataFrame([{
        "fixture_id": "continental",
        "date": "2026-01-01",
        "competition": "Champions League",
        "home": "Barcelona",
        "away": "Arsenal",
        "status": "FIN",
        "home_goals": 1,
        "away_goals": 1,
    }])
    out = F.write_fixtures(rows, tmp_path / "fixtures.csv")
    assert out.iloc[0]["home_club_id"] == canonical_id("Barcelona")
    assert out.iloc[0]["away_club_id"] == canonical_id("Arsenal")


def test_write_boundary_unifies_display_names_for_one_club_id(tmp_path):
    rows = pd.DataFrame([
        {"fixture_id": "1", "date": "2026-01-01",
         "competition": "League Two", "home": "Accrington",
         "away": "Crewe", "status": "FIN", "home_goals": 1,
         "away_goals": 0},
        {"fixture_id": "2", "date": "2026-01-08",
         "competition": "League Two", "home": "Accrington Stanley",
         "away": "Crewe", "status": "FIN", "home_goals": 0,
         "away_goals": 1},
    ])
    out = F.write_fixtures(rows, tmp_path / "fixtures.csv")
    assert out["home"].nunique() == 1
    assert out["home_club_id"].nunique() == 1


def test_write_boundary_rejects_conflicting_scores(tmp_path):
    rows = pd.DataFrame([
        {"fixture_id": "a", "date": "2026-01-01",
         "competition": "Premier League", "home": "Arsenal",
         "away": "Chelsea", "status": "FIN", "home_goals": 1,
         "away_goals": 0},
        {"fixture_id": "b", "date": "2026-01-01",
         "competition": "Premier League", "home": "Arsenal",
         "away": "Chelsea", "status": "FIN", "home_goals": 0,
         "away_goals": 2},
    ])
    with pytest.raises(ValueError, match="conflicting scores"):
        F.write_fixtures(rows, tmp_path / "fixtures.csv")


def test_gate_note_describes_current_per_market_gate():
    note = json.loads(DTB.ARTIFACT.read_text())["note"]
    assert "per-market" in note
    assert "global-AND" not in note
