from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from golf import economic
from golf import engine, refresh
from golf.providers.odds_manual import OddsQuote


def _rounds() -> pd.DataFrame:
    rows = []
    totals = {
        "Winner": [-3, -3, -2, -2],
        "Second": [-3, -2, -2, -2],
        "Third": [-2, -2, -2, -2],
        "Fourth": [-2, -2, -2, -1],
        "Tie Five A": [-2, -2, -1, -1],
        "Tie Five B": [-2, -2, -1, -1],
    }
    finishes = {
        "Winner": 1,
        "Second": 2,
        "Third": 3,
        "Fourth": 4,
        "Tie Five A": 5,
        "Tie Five B": 5,
    }
    for name, scores in totals.items():
        for round_no, score in enumerate(scores, 1):
            rows.append({
                "tournament_id": "E1",
                "event_name": "Test Open",
                "date": "2026-01-08",
                "tour": "pga",
                "is_major": 0,
                "course": "Test Course",
                "course_name": "Test Course",
                "round": round_no,
                "player": name,
                "dg_id": name,
                "score_to_par": score,
                "made_cut": 1,
                "finish": finishes[name],
                "total_rounds": 4,
                "no_cut": 0,
            })
    return pd.DataFrame(rows)


def test_odds_snapshot_is_idempotent_and_devigs_complete_group(tmp_path):
    path = tmp_path / "odds_history.csv"
    quotes = [
        OddsQuote(
            event_id="E1",
            market="tournament_matchup",
            player_name="Winner",
            decimal_odds=1.80,
            group_id="g1",
            book="testbook",
            source="test",
            timestamp="2026-01-07T10:00:00+00:00",
        ),
        OddsQuote(
            event_id="E1",
            market="tournament_matchup",
            player_name="Second",
            decimal_odds=2.10,
            group_id="g1",
            book="testbook",
            source="test",
            timestamp="2026-01-07T10:00:00+00:00",
        ),
    ]
    kwargs = {
        "event_id": "E1",
        "event_name": "Test Open",
        "event_start_date": "2026-01-08",
        "phase": "pre_event",
        "observed_at": "2026-01-07T10:01:00+00:00",
        "path": path,
    }
    assert economic.record_odds_snapshot(quotes, **kwargs) == 2
    assert economic.record_odds_snapshot(quotes, **kwargs) == 0
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert all(row["board_complete"] == "1" for row in rows)
    assert all(row["fair_method"] == "multiplicative_complete" for row in rows)
    assert abs(sum(float(row["fair_prob"]) for row in rows) - 1.0) < 1e-7


def test_partial_outright_is_not_normalized(tmp_path):
    path = tmp_path / "odds_history.csv"
    quotes = [
        OddsQuote(market="win", player_name="Winner", decimal_odds=5.0),
        OddsQuote(market="win", player_name="Second", decimal_odds=8.0),
    ]
    economic.record_odds_snapshot(
        quotes,
        event_id="E1",
        event_name="Test Open",
        phase="pre_event",
        observed_at="2026-01-07T09:00:00+00:00",
        path=path,
    )
    rows = list(csv.DictReader(path.open()))
    assert rows[0]["fair_method"] == "raw_implied_partial"
    assert rows[0]["board_complete"] == "0"
    assert float(rows[0]["fair_prob"]) == 0.2


def test_legacy_odds_schema_is_archived_not_overwritten(tmp_path):
    path = tmp_path / "odds_history.csv"
    path.write_text(
        "ts,event,player,market,odds,fair_odds,fair_prob\n"
        "2025-01-01T00:00:00Z,Old Event,Old Player,win,5,4.5,.222\n"
    )
    economic.record_odds_snapshot(
        [OddsQuote(market="win", player_name="Winner", decimal_odds=5.0)],
        event_id="E1",
        event_name="Test Open",
        phase="pre_event",
        observed_at="2026-01-07T09:00:00+00:00",
        path=path,
    )
    archives = list(tmp_path.glob("odds_history.legacy-*.csv"))
    assert len(archives) == 1
    assert "Old Player" in archives[0].read_text()
    with path.open() as handle:
        assert next(csv.reader(handle)) == economic.ODDS_COLS


def test_settlement_dead_heat_clv_and_collecting_gate(tmp_path, monkeypatch):
    odds_path = tmp_path / "odds_history.csv"
    decisions_path = tmp_path / "decision_history.csv"
    rounds_path = tmp_path / "rounds.csv"
    settlements_path = tmp_path / "settled.csv"
    report_path = tmp_path / "report.json"
    _rounds().to_csv(rounds_path, index=False)

    economic.record_odds_snapshot(
        [
            OddsQuote(
                market="top5",
                player_name="Tie Five A",
                decimal_odds=4.0,
                book="testbook",
                source="test",
            )
        ],
        event_id="E1",
        event_name="Test Open",
        phase="pre_event",
        observed_at="2026-01-07T12:00:00+00:00",
        path=odds_path,
    )
    monkeypatch.setattr(economic.model, "PARAMS_JSON", tmp_path / "params.json")
    assert economic.record_decisions(
        [{
            "player": "Tie Five A",
            "market": "Top 5",
            "side": "top5",
            "odds": 4.0,
            "p_model": 0.30,
            "p_market": 0.25,
            "p_final": 0.30,
            "ev_per_unit": 0.20,
            "thin_sample": False,
            "recommended": False,
            "stake_gbp": 0.0,
        }],
        event_id="E1",
        event_name="Test Open",
        phase="pre_event",
        observed_at="2026-01-07T10:00:00+00:00",
        path=decisions_path,
    ) == 1
    assert economic.record_decisions(
        [{
            "player": "Tie Five A",
            "market": "Top 5",
            "side": "top5",
            "odds": 4.0,
            "p_model": 0.30,
            "p_final": 0.30,
            "ev_per_unit": 0.20,
            "thin_sample": False,
        }],
        event_id="E1",
        event_name="Test Open",
        phase="pre_event",
        observed_at="2026-01-07T10:05:00+00:00",
        path=decisions_path,
    ) == 1

    report = economic.economic_report(
        odds_path=odds_path,
        decisions_path=decisions_path,
        rounds_path=rounds_path,
        settlements_path=settlements_path,
        report_path=report_path,
    )
    settled = list(csv.DictReader(settlements_path.open()))
    assert settled[0]["result_credit"] == "0.5"
    assert settled[0]["unit_return"] == "2.0"
    assert settled[0]["unit_profit"] == "1.0"
    assert settled[0]["clv"] == "0.0"
    assert report["status"] == "collecting"
    assert report["automatic_activation"] is False
    assert report["readiness"]["evidence_pass"] is False
    assert report["coverage"]["decision_rows"] == 2
    assert json.loads(report_path.read_text())["coverage"]["paper_bets"] == 1


def test_matchup_and_threeball_settlement():
    outcomes = economic._event_outcomes(_rounds())["E1"]
    matchup = economic._settle_one({
        "player": "Winner",
        "side": "matchup:Winner|Second",
        "odds": "1.8",
    }, outcomes)
    threeball = economic._settle_one({
        "player": "Winner",
        "side": "3ball:Winner|Second|Third",
        "odds": "2.5",
    }, outcomes)
    assert matchup == (1.0, 1.8)
    assert threeball == (1.0, 2.5)


def test_tee_off_detection_uses_earliest_iso_time(tmp_path, monkeypatch):
    field = tmp_path / "field.csv"
    field.write_text(
        "name,tee_time_r1\n"
        "Late,2026-01-08T12:00:00Z\n"
        "Early,2026-01-08T08:00:00Z\n"
    )
    monkeypatch.setattr(engine, "DATA_DIR", tmp_path)
    now = pd.Timestamp("2026-01-08T09:00:00Z").to_pydatetime()
    assert engine._event_has_teed_off(now)
    rows = [
        {"tee_time_r1": "2026-01-08T12:00:00Z"},
        {"tee_time_r1": "2026-01-08T08:00:00Z"},
    ]
    assert refresh._field_has_teed_off(rows, now)


def test_economic_capture_excludes_settled_round_boards():
    quotes = [
        OddsQuote(market="win", player_name="Winner", decimal_odds=5.0),
        OddsQuote(
            market="3ball",
            player_name="Winner",
            decimal_odds=2.5,
            round_no=1,
        ),
        OddsQuote(
            market="2ball",
            player_name="Second",
            decimal_odds=1.9,
            round_no=4,
        ),
    ]
    between = refresh._economic_quotes_for_phase(
        quotes, rounds_done=3, phase="between_rounds", event_complete=False
    )
    assert [(q.market, q.round_no) for q in between] == [
        ("win", None),
        ("2ball", 4),
    ]
    assert refresh._economic_quotes_for_phase(
        quotes, rounds_done=4, phase="between_rounds", event_complete=True
    ) == []
