from __future__ import annotations

import inspect

import pandas as pd
import pytest

from club_soccer import decision_ledger as DL
from club_soccer import edge as E
from club_soccer import engine as ENGINE
from club_soccer import forecast_ledger as FL
from club_soccer import prediction_scope as PS
from club_soccer import season
from app.engines.club_soccer import ClubSoccerAdapter


EXPECTED = {
    "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1",
    "Scottish Premiership", "Champions League", "Europa League",
    "Conference League", "UEFA Super Cup", "Copa Libertadores",
    "Copa Sudamericana",
}


def test_surface_allowlist_is_exact_and_rejects_adjacent_competitions():
    assert set(PS.SURFACED_COMPETITIONS) == EXPECTED
    for competition in EXPECTED:
        assert PS.is_surfaced(competition)
    for competition in (
        "Championship", "Scottish Championship", "FA Cup", "EFL Cup",
        "Eredivisie", "MLS", "CAF Champions League",
    ):
        assert not PS.is_surfaced(competition)


def _forecast(competition: str, home: str) -> dict:
    return {
        "match_date": "2026-08-20", "competition": competition,
        "home": home, "away": "Away", "p_home": .6, "p_draw": .25,
        "p_away": .15, "n_missing_home": 0, "n_missing_away": 0,
        "home_attack_mult": 1, "away_attack_mult": 1,
        "lineup_confidence": 1, "type": "europe", "season": "2026",
    }


def _edge(competition: str, match: str, *, low_evidence: bool = False) -> dict:
    return {
        "date": "2999-08-20", "competition": competition, "match": match,
        "market": "1x2", "bet": "Home", "side": "home", "odds": 2.0,
        "p_model": .60, "p_book": .50, "edge": .10, "ev_per_unit": .20,
        "kelly_stake": .02, "stake_gbp": 2.0, "suppressed_reason": "",
        "evidence_tier": "low" if low_evidence else "full",
        "evidence_note": "shadow only",
    }


def test_card_sections_hide_non_surface_forecasts_and_edges(monkeypatch):
    monkeypatch.setattr(season._MarketCache, "note", lambda *args, **kwargs: "")
    forecasts = [
        _forecast("Premier League", "Visible FC"),
        _forecast("Championship", "Hidden FC"),
    ]
    upcoming = "\n".join(season._upcoming_section(forecasts))
    assert "Visible FC" in upcoming
    assert "Hidden FC" not in upcoming

    edges = [
        _edge("Champions League", "Visible FC vs Away"),
        _edge("EFL Cup", "Hidden FC vs Away"),
    ]
    likely = "\n".join(season._likely_winners_section(edges, "2026-01-01"))
    backed = "\n".join(season._backed_bets_section(edges, "2026-01-01"))
    assert "Visible FC" in likely and "Hidden FC" not in likely
    assert "Visible FC" in backed and "Hidden FC" not in backed

    low = [
        _edge("Copa Libertadores", "Visible Low", low_evidence=True),
        _edge("MLS", "Hidden Low", low_evidence=True),
    ]
    rendered_low = "\n".join(season._low_evidence_section(low))
    assert "Visible Low" in rendered_low
    assert "Hidden Low" not in rendered_low


def test_app_schema_and_edge_rows_are_surface_scoped(monkeypatch):
    fixtures = pd.DataFrame([
        {"competition": "Premier League", "home": "Arsenal", "away": "Chelsea",
         "season": 2026},
        {"competition": "Championship", "home": "Hidden", "away": "Other",
         "season": 2026},
    ])
    monkeypatch.setattr(ENGINE.M, "load_fixtures", lambda: fixtures)
    schema = ENGINE.cmd_schema()
    competition_filter = next(f for f in schema["filters"] if f["id"] == "competition")
    assert set(competition_filter["options"]) == EXPECTED
    assert "Arsenal" in schema["names"]
    assert "Hidden" not in schema["names"]

    odds = pd.DataFrame([{"odds": 2.0}])
    monkeypatch.setattr(ENGINE.E, "load_odds", lambda: odds)
    monkeypatch.setattr(ENGINE.E, "validate_quotes", lambda frame, source: (frame, []))
    monkeypatch.setattr(ENGINE.E, "rows_from_odds", lambda *args, **kwargs: [
        _edge("Premier League", "Visible"), _edge("Championship", "Hidden")
    ])
    result = ENGINE.cmd_edge({"odds_source": "manual"})
    assert [row["match"] for row in result["rows"]] == ["Visible"]


def test_app_adapter_is_a_second_surface_boundary(monkeypatch):
    adapter = ClubSoccerAdapter()
    monkeypatch.setattr(adapter, "_run", lambda command, params=None: {
        "columns": [], "rows": [
            _edge("Europa League", "Visible"),
            _edge("Eredivisie", "Hidden"),
        ]
    })
    monkeypatch.setattr("app.bankroll_store.current_bankroll", lambda: 100.0)
    result = adapter.edge({"odds_source": "bsd", "model": "ensemble"})
    assert [row["match"] for row in result["rows"]] == ["Visible"]


def test_on_demand_prediction_requires_a_surface_competition(monkeypatch):
    called = False

    def _predict(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("model must not run for hidden competitions")

    monkeypatch.setattr(ENGINE.M, "predict", _predict)
    with pytest.raises(ValueError, match="Choose one of the surfaced"):
        ENGINE.cmd_predict({
            "team1": "Hidden", "team2": "Other", "competition": "Championship"
        })
    assert called is False


def test_manual_odds_template_is_surface_scoped(tmp_path, monkeypatch):
    fixtures = pd.DataFrame([
        {"date": pd.Timestamp("2026-08-20"), "competition": "Serie A",
         "home": "Visible", "away": "Away"},
        {"date": pd.Timestamp("2026-08-20"), "competition": "Serie B",
         "home": "Hidden", "away": "Away"},
    ])
    monkeypatch.setattr(E.M, "load_fixtures", lambda: fixtures)
    monkeypatch.setattr(E.M, "upcoming", lambda frame: frame)
    target = tmp_path / "odds.csv"
    E.write_template(target)
    text = target.read_text()
    assert "Visible" in text
    assert "Hidden" not in text


def test_collection_and_forecast_ledgers_do_not_import_surface_policy():
    """The scope must never leak backward into evidence/data accumulation."""
    assert "prediction_scope" not in inspect.getsource(FL)
    assert "prediction_scope" not in inspect.getsource(DL)
    run_source = inspect.getsource(season._run_steps)
    assert "FL.append_forecasts,\n              forecasts" in run_source
    assert "PS.filter_rows(forecasts)" not in run_source
