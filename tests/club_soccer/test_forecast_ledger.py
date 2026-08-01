from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from club_soccer import forecast_ledger as FL
from club_soccer import season


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(FL, "FORECASTS", tmp_path / "forecasts.csv")
    monkeypatch.setattr(FL, "RESULTS", tmp_path / "results.csv")
    monkeypatch.setattr(FL, "PERFORMANCE", tmp_path / "performance.json")
    return tmp_path


def _forecast(**updates):
    row = {field: "" for field in FL.FORECAST_FIELDS}
    row.update({
        "forecast_id": "f1",
        "forecast_ts": "2026-08-01T12:00:00+00:00",
        "run_id": "r1",
        "run_mode": "production_network",
        "primary_eligible": 1,
        "fixture_identity": "2026-08-02|premier league|home-id|away-id",
        "fixture_id": "provider-old",
        "kickoff_utc": "2026-08-02T18:00:00+00:00",
        "match_date": "2026-08-02",
        "lead_hours": 30,
        "competition": "Premier League",
        "type": "league",
        "season": 2026,
        "home_club_id": "home-id",
        "away_club_id": "away-id",
        "home": "Arsenal",
        "away": "Chelsea",
        "model": "ensemble",
        "model_hash": "model",
        "code_hash": "code",
        "resolver_version": "resolver",
        "p_home": 0.60,
        "p_draw": 0.25,
        "p_away": 0.15,
        "p_over25": 0.55,
        "p_under25": 0.45,
        "p_btts_yes": 0.52,
        "p_btts_no": 0.48,
        "xg_home": 1.7,
        "xg_away": 0.9,
        "evidence_tier": "full",
        "evidence_ok": 1,
        "n_matches_home": 50,
        "n_matches_away": 50,
        "lineup_confidence": 1.0,
        "home_attack_mult": 1.0,
        "home_defense_mult": 1.0,
        "away_attack_mult": 1.0,
        "away_defense_mult": 1.0,
        "n_missing_home": 0,
        "n_missing_away": 0,
    })
    row.update(updates)
    return row


def _finished(**updates):
    row = {
        "fixture_id": "provider-new",
        "date": pd.Timestamp("2026-08-02"),
        "competition": "Premier League",
        "home": "Arsenal",
        "away": "Chelsea",
        "home_club_id": "home-id",
        "away_club_id": "away-id",
        "home_goals": 2.0,
        "away_goals": 1.0,
        "status": "FT",
        "result_scope": "regulation",
    }
    row.update(updates)
    return row


def test_append_is_idempotent_per_run_and_fixture(ledger):
    row = _forecast()
    assert FL.append_forecasts([row, row]) == 1
    assert FL.append_forecasts([row]) == 0
    assert len(pd.read_csv(FL.FORECASTS)) == 1
    report = FL.performance_report()
    assert report["forecast_rows"] == 1
    assert report["forecast_fixtures"] == 1
    assert report["settled_fixtures"] == 0


def test_settlement_uses_canonical_identity_not_provider_id(ledger):
    FL.append_forecasts([_forecast()])
    fixtures = pd.DataFrame([_finished()])
    assert FL.settle(fixtures=fixtures, verbose=False) == 1
    result = pd.read_csv(FL.RESULTS).iloc[0]
    assert result.result_fixture_id == "provider-new"
    assert result.actual_1x2 == "home"
    assert result.over25_actual == 1
    assert result.btts_actual == 1
    assert FL.settle(fixtures=fixtures, verbose=False) == 0


def test_live_score_does_not_settle(ledger):
    FL.append_forecasts([_forecast()])
    fixtures = pd.DataFrame([_finished(status="LIV")])
    assert FL.settle(fixtures=fixtures, verbose=False) == 0
    assert not FL.RESULTS.exists()


def test_cup_markets_use_regulation_score(ledger):
    FL.append_forecasts([_forecast()])
    fixtures = pd.DataFrame([_finished(
        status="AET", result_scope="extra_time", home_goals=2.0, away_goals=1.0,
        home_goals_ft=1.0, away_goals_ft=1.0,
    )])
    assert FL.settle(fixtures=fixtures, verbose=False) == 1
    result = pd.read_csv(FL.RESULTS).iloc[0]
    assert result.actual_1x2 == "draw"
    assert result.over25_actual == 0
    assert result.btts_actual == 1


def test_t24_report_counts_each_fixture_once_and_uses_closest_prior_snapshot(ledger):
    early = _forecast(
        forecast_id="early", forecast_ts="2026-07-31T18:00:00+00:00",
        lead_hours=48, p_home=.10, p_draw=.20, p_away=.70,
    )
    t24 = _forecast(
        forecast_id="t24", forecast_ts="2026-08-01T18:00:00+00:00",
        lead_hours=24, p_home=.70, p_draw=.20, p_away=.10,
    )
    late = _forecast(
        forecast_id="late", forecast_ts="2026-08-02T12:00:00+00:00",
        lead_hours=6, p_home=.80, p_draw=.15, p_away=.05,
    )
    FL.append_forecasts([early, t24, late])
    FL.settle(fixtures=pd.DataFrame([_finished()]), verbose=False)
    report = FL.performance_report(now=datetime(2026, 8, 3, tzinfo=timezone.utc))
    assert report["forecast_rows"] == 3
    assert report["forecast_fixtures"] == 1
    assert report["settled_fixtures"] == 1
    assert report["cohorts"]["first_published"]["n"] == 1
    assert report["cohorts"]["t24"]["n"] == 1
    assert report["cohorts"]["latest_pre_kickoff"]["n"] == 1
    # T-24 chose the .70 home forecast, not the wrong .10 first forecast.
    assert report["cohorts"]["t24"]["brier_1x2"] == pytest.approx(.14)
    assert report["cohorts"]["first_published"]["brier_1x2"] == pytest.approx(1.34)
    assert report["by_evidence_t24"]["full"]["n"] == 1
    assert report["by_fixture_type_t24"]["league"]["n"] == 1
    assert report["by_lead_bucket"]["12_36h"]["n"] == 1


def test_build_rows_are_the_rows_rendered_on_the_card(monkeypatch):
    fixtures = pd.DataFrame([{
        "fixture_id": "1", "kickoff_utc": "2026-08-02T18:00:00Z",
        "date": pd.Timestamp("2026-08-02"), "season": 2026,
        "competition": "Premier League", "competition_id": 39,
        "country": "England", "type": "league",
        "home_club_id": "home-id", "away_club_id": "away-id",
        "home": "Arsenal", "away": "Chelsea", "neutral": 0,
        "home_goals": None, "away_goals": None, "status": "NOT",
    }])
    params = {"teams": ["Arsenal", "Chelsea"], "_training_fingerprint": "train"}
    monkeypatch.setattr(FL.M, "load_params", lambda: params)
    monkeypatch.setattr(FL.M, "predict_match", lambda *a, **k: {
        "probs": {"home": .6000, "draw": .2500, "away": .1500,
                  "over25": .55, "under25": .45,
                  "btts_yes": .52, "btts_no": .48},
        "xg_home": 1.7, "xg_away": .9,
        "coverage": {"tier": "full", "reliable": True,
                     "home": {"n": 50}, "away": {"n": 50}},
    })
    monkeypatch.setattr(FL, "_code_hash", lambda: "code")
    monkeypatch.setattr(FL, "_resolver_version", lambda: "resolver")
    monkeypatch.setattr(FL, "_sha", lambda path: "model")
    rows = FL.build_forecasts(
        pd.Timestamp("2026-08-01"), None, None, run_id="run",
        run_mode="production_network", primary_eligible=True,
        forecast_ts=datetime(2026, 8, 1, 12, tzinfo=timezone.utc),
        fixtures=fixtures,
    )
    assert len(rows) == 1
    rendered = "\n".join(season._upcoming_section(rows))
    assert "Arsenal vs Chelsea — H 60% D 25% A 15%" in rendered
    assert rows[0]["p_home"] == .6
