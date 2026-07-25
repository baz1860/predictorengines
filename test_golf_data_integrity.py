"""Regression tests for the corrected golf ingestion ground truth."""

from __future__ import annotations

import csv
import datetime as dt
import sqlite3
from pathlib import Path

from golf import model
from golf.providers import legacy
from golf.providers.espn import EspnGolfProvider
from golf.providers.odds_manual import ManualOddsProvider, parse_skybet_threeball_text


def _event_payload() -> dict:
    def player(name: str, pid: str, status: str, rounds: int) -> dict:
        return {
            "athlete": {"id": pid, "displayName": name},
            "status": {
                "type": {"name": status},
                "position": {"id": "1" if status == "STATUS_FINISH" else "99"},
            },
            "linescores": [
                {
                    "period": rnd,
                    "displayValue": "-1" if rnd == 1 else "+1",
                    "value": 71 if rnd == 1 else 73,
                    "teeTime": f"2025-01-{21 + rnd:02d}T12:00Z",
                }
                for rnd in range(1, rounds + 1)
            ],
        }

    return {
        "events": [{
            "id": "E1",
            "name": "Rotating Open",
            "date": "2025-01-22T00:00Z",
            "status": {"type": {"name": "STATUS_FINAL", "completed": True}},
            "tournament": {
                "displayName": "Rotating Open",
                "major": True,
                "numberOfRounds": 4,
                "cutRound": 2,
                "cutScore": 1,
                "cutCount": 70,
            },
            "courses": [
                {"id": "away", "name": "Away Course", "host": False},
                {
                    "id": "host",
                    "name": "Real Host Course",
                    "host": True,
                    "shotsToPar": 72,
                    "totalYards": 7600,
                    "holes": [
                        {"number": hole, "shotsToPar": (
                            3 if hole <= 4 else 5 if hole <= 8 else 4
                        )}
                        for hole in range(1, 19)
                    ],
                },
            ],
            "competitions": [{
                "competitors": [
                    player("Made It", "1", "STATUS_FINISH", 4),
                    player("Missed It", "2", "STATUS_CUT", 2),
                ],
            }],
        }],
    }


def test_history_uses_per_event_status_course_rules_and_tee_times(monkeypatch):
    provider = legacy.EspnProvider(seasons=[2025])
    provider._meta["E1"] = legacy.TournamentMeta(
        "E1", "Rotating Open", "2025-01-22", is_major=True,
    )
    monkeypatch.setattr(provider, "_load_all", lambda: None)
    monkeypatch.setattr(provider, "_event_payload", lambda _event: _event_payload())

    rows = provider.rounds_for("E1")
    by_player = {}
    for row in rows:
        by_player.setdefault(row.player, row.made_cut)
        assert row.course != "Rotating Open"
        assert row.course_name == "Real Host Course"
        assert row.course_id == "host"
        assert row.course_par == 72
        assert row.course_yards == 7600
        assert row.cut_round == 2
        assert row.cut_count == 70
        assert row.total_rounds == 4
        assert row.tee_time.endswith("Z")
    assert by_player == {"Made It": 1, "Missed It": 0}


def test_espn_live_and_history_names_resolve_to_one_provider():
    assert legacy.EspnProvider is EspnGolfProvider


def test_event_exposes_measured_course_hole_mix():
    from golf.providers.espn import _event_from_payload

    event = _event_from_payload(_event_payload()["events"][0])
    assert (event.par3_holes, event.par4_holes, event.par5_holes) == (4, 10, 4)


class _StaticProvider:
    name = "static"
    supports_history = True

    def recent_tournaments(self, since=None):
        return [legacy.TournamentMeta("E1", "Event", "2025-01-01")]

    def rounds_for(self, tournament_id):
        return [legacy.RoundRecord(
            tournament_id="E1",
            event_name="Event",
            date="2025-01-01",
            tour="pga",
            is_major=0,
            course="Actual Course",
            course_name="Actual Course",
            course_id="C1",
            course_par=72,
            course_yards=7200,
            round=1,
            tee_time="2025-01-01T12:00Z",
            player="Player",
            dg_id="P1",
            score_to_par=-2.0,
            field_size=100,
            made_cut=1,
            finish=1,
            cut_round=2,
            cut_score=1.0,
            cut_count=65,
            total_rounds=4,
            no_cut=0,
        )]

    def field_for(self, event=None):
        return []

    def pretournament_preds(self, event=None):
        return None

    def sg_categories(self, player, asof=None):
        return None


def test_accumulation_is_byte_idempotent(monkeypatch, tmp_path):
    path = tmp_path / "rounds.csv"
    monkeypatch.setattr(legacy, "ROUNDS_CSV", path)
    assert legacy.accumulate_rounds(_StaticProvider(), verbose=False) == 1
    first = path.read_bytes()
    assert legacy.accumulate_rounds(_StaticProvider(), verbose=False) == 0
    assert path.read_bytes() == first


def test_bad_threeball_price_rejects_whole_group(tmp_path):
    path = tmp_path / "threeballs.csv"
    path.write_text(
        "group_id,player_a,player_b,player_c,odds_a,odds_b,odds_c\n"
        "manual-3ball-r1:1,A,B,C,2.5,0.5,5.0\n"
    )
    assert ManualOddsProvider().load_threeballs(path) == []

    issues = []
    raw = """
    3 Ball Round 1 - A / B / C
    A
    2.5
    B
    0.5
    C
    5.0
    """
    assert parse_skybet_threeball_text(raw, issues=issues) == []
    assert issues


def test_public_stats_gate_uses_snapshot_capture_date(tmp_path):
    stats = tmp_path / "pgatour_stats.csv"
    stats.write_text(
        "season,stat_id,stat_name,player_name,rank,value,raw_json,source\n"
        '2025,1,sg_total,Player,1,1.25,"{}",test\n'
    )
    capture = dt.datetime(2025, 7, 10, tzinfo=dt.timezone.utc).timestamp()
    stats.touch()
    import os
    os.utime(stats, (capture, capture))

    assert model.load_public_stat_priors(stats, asof="2025-07-09") == {}
    assert model.load_public_stat_priors(stats, asof="2025-07-10")["Player"][
        "sg_total"
    ] == 1.25


def test_manual_timestamp_is_utc():
    from golf.providers.odds_manual import _ts

    assert dt.datetime.fromisoformat(_ts()).utcoffset() == dt.timedelta(0)


def test_major_detection_does_not_drop_dp_world_events():
    assert legacy._is_major("PGA Championship")
    assert legacy._is_major("The Open")
    assert not legacy._is_major("BMW PGA Championship")
    assert not legacy._is_major("Fortinet Australian PGA Championship")
    assert not legacy._is_major("Investec SA Open Championship")


def test_hole_features_derive_par_shape_and_blowups():
    relative = [-1, 0, 1, 2] + [0] * 14
    event = {"competitions": [{"competitors": [{
        "athlete": {"id": "P1", "displayName": "Player"},
        "linescores": [{
            "period": 1,
            "linescores": [
                {
                    "period": i + 1,
                    "value": 4 + rel,
                    "scoreType": {
                        "displayValue": "E" if rel == 0 else f"{rel:+d}",
                    },
                }
                for i, rel in enumerate(relative)
            ],
        }],
    }]}]}
    feat = legacy._hole_features(event)[("Player", 1)]
    assert feat["holes_scored"] == 18
    assert feat["birdies_or_better"] == 1
    assert feat["bogeys"] == 1
    assert feat["double_bogeys_or_worse"] == 1
    assert feat["par4_holes"] == 18
    assert feat["par4_to_par"] == 2.0


def test_empty_calibration_promotion_disables_runtime_layer(monkeypatch, tmp_path):
    from golf import calibrate

    path = tmp_path / "calibration.json"
    path.write_text(
        '{"win":{"x":[0,1],"y":[0,1]},'
        '"_meta":{"schema_version":2,"point_in_time_safe":true,'
        '"promoted_markets":[]}}'
    )
    monkeypatch.setattr(calibrate, "CALIB_FILE", path)
    assert calibrate.load_maps() is None


def test_integrity_rejects_duplicate_round_keys(tmp_path):
    from golf import integrity

    rounds = tmp_path / "rounds.csv"
    row = {column: "" for column in legacy.ROUNDS_COLUMNS}
    row.update({
        "tournament_id": "E1",
        "event_name": "Event",
        "date": "2025-01-01",
        "tour": "pga",
        "course": "Course",
        "course_id": "C1",
        "course_name": "Course",
        "course_par": "72",
        "course_yards": "7200",
        "round": "1",
        "player": "Player",
        "dg_id": "P1",
        "score_to_par": "-1",
        "field_size": "100",
        "made_cut": "1",
        "finish": "1",
        "cut_round": "2",
        "cut_count": "65",
        "total_rounds": "4",
        "no_cut": "0",
    })
    with rounds.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=legacy.ROUNDS_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)
    stats, errors = integrity.check_rounds(rounds)
    assert stats["rows"] == 2
    assert any("duplicate round key" in error for error in errors)


def test_course_profile_uses_par_mix_and_yardage():
    params = {
        "course_profile_weight": 0.5,
        "course_context": {
            "par_mean": 71.0,
            "par_sd": 1.0,
            "yards_mean": 7200.0,
            "yards_sd": 200.0,
        },
    }
    player = {
        "course_profile": {
            "par_slope": 0.2,
            "yards_slope": 0.1,
            "par_center_z": 0.0,
            "yards_center_z": 0.0,
        },
        "par_profile": {
            "par3_skill_per_hole": 0.02,
            "par3_center_holes": 4.0,
            "par4_skill_per_hole": -0.01,
            "par4_center_holes": 10.0,
            "par5_skill_per_hole": 0.03,
            "par5_center_holes": 4.0,
        },
    }
    adjustment = model._course_profile_adjustment(
        player,
        params,
        course_par=72,
        course_yards=7400,
        par3_holes=4,
        par4_holes=9,
        par5_holes=5,
    )
    assert adjustment > 0.15


def test_scoring_shape_consumes_birdie_and_bogey_rates():
    import numpy as np
    from golf import simulate

    common = dict(
        means=np.zeros(1),
        sigmas=np.ones(1),
        n_sims=20_000,
        n=1,
        round_corr=0.3,
        tail_df=None,
        blowup_rates=np.array([0.02]),
        blowup_mix=0.4,
    )
    shaped = simulate._draw_scores(
        np.random.default_rng(4),
        birdie_rates=np.array([0.24]),
        bogey_rates=np.array([0.11]),
        **common,
    )
    doubles_only = simulate._draw_scores(
        np.random.default_rng(4),
        birdie_rates=np.array([0.0]),
        bogey_rates=np.array([0.0]),
        **common,
    )
    assert not np.array_equal(shaped, doubles_only)
    assert abs(float(shaped.mean())) < 0.05


def test_store_migrates_old_round_mirror_and_course_columns(tmp_path):
    from golf import store

    db = tmp_path / "golf.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE rounds(x)")
        con.execute("CREATE TABLE events(event_id TEXT PRIMARY KEY, name TEXT NOT NULL)")
    store.init_db(db)
    with store.connect(db) as con:
        tables = {
            row[0]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        columns = {row[1] for row in con.execute("PRAGMA table_info(events)")}
    assert "rounds" not in tables
    assert {
        "course_id", "course_par", "course_yards",
        "par3_holes", "par4_holes", "par5_holes",
    } <= columns
