from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import preflight

from cfb import elo as E
from cfb import edge as CE
from cfb import engine as CENGINE
from cfb import fetch_data as FETCH_DATA
from cfb import identity as IDENTITY
from cfb import dataset_fingerprint as DATASET_FP
from cfb import power as POWER
from cfb import policy as POLICY
from cfb import run_status as RUN_STATUS
from cfb import season as S
from cfb import validate as VALIDATE
from app.engines import cfb as APP_CFB
from app.engines.cfb import CFBAdapter


def _state():
    return {
        "champion_ratings": {"Alpha": 1700.0, E.FCS: 900.0},
        "champion_last_season": {"Alpha": 2025, E.FCS: 2025},
        "fcs_ratings": {"Bravo": 1050.0, "Charlie": 1400.0, E.SUB_FCS: 800.0},
        "fcs_last_season": {"Bravo": 2025, "Charlie": 2025, E.SUB_FCS: 2025},
    }


def test_division_aware_rollover_and_priors_apply_once():
    ratings, state, audit = E.advance_state(
        _state(), 2026, carry=0.5,
        prior_offsets={("Alpha", 2026): 20.0},
        team_divisions={"Alpha": "fbs", "Bravo": "fcs"},
    )

    assert ratings["Alpha"] == 1620.0  # 1500 + .5*(1700-1500) + 20
    assert ratings["Bravo"] == 950.0   # 850 + .5*(1050-850)
    assert audit["rolled_fbs"] == 1
    assert audit["rolled_fcs"] == 2

    ratings2, _, audit2 = E.advance_state(
        state, 2026, carry=0.5,
        prior_offsets={("Alpha", 2026): 20.0},
        team_divisions={"Alpha": "fbs", "Bravo": "fcs"},
    )
    assert ratings2["Alpha"] == ratings["Alpha"]
    assert ratings2["Bravo"] == ratings["Bravo"]
    assert audit2["rolled_fbs"] == 0
    assert audit2["rolled_fcs"] == 0


def test_fcs_to_fbs_transition_has_explicit_floor():
    ratings, state, audit = E.advance_state(
        _state(), 2026, carry=0.5, prior_offsets={},
        team_divisions={"Charlie": "fbs", "New School": "fbs"},
    )

    # Charlie first regresses around the FCS anchor: 850 + .5*(1400-850)=1125,
    # then enters the champion ledger at the standard new-FBS floor.
    assert ratings["Charlie"] == E.NEW_TEAM_ELO
    assert state["champion_ratings"]["Charlie"] == E.NEW_TEAM_ELO
    assert audit["transitioned_fbs"] == ["Charlie"]
    assert ratings["New School"] == E.NEW_TEAM_ELO
    assert audit["new_fbs"] == ["New School"]


def test_january_maps_to_previous_cfb_season():
    assert E.season_for_date("2027-01-10") == 2026
    assert E.season_for_date("2026-08-29") == 2026


def test_reviewed_team_identity_uses_cfbd_ids_and_rejects_prefix_guesses():
    canonical = IDENTITY.resolve("San José State", 2026, "the-odds-api")
    alias = IDENTITY.resolve("San Jose State Spartans", 2026, "the-odds-api")
    assert canonical == {"team_id": "23", "canonical": "San José State",
                         "match_mode": "canonical"}
    assert alias["team_id"] == "23"
    assert alias["canonical"] == "San José State"
    assert alias["match_mode"] == "reviewed_alias"
    assert IDENTITY.resolve("San Jose State Mystery Team", 2026,
                            "the-odds-api") is None


def test_week_zero_reviewed_provider_aliases_have_unique_canonical_ids():
    names = [
        "TCU Horned Frogs", "North Carolina Tar Heels",
        "San Jose State Spartans", "USC Trojans", "NC State Wolfpack",
        "Virginia Cavaliers", "Jacksonville State Gamecocks",
        "North Dakota State Bison", "Sacramento State Hornets",
        "Eastern Michigan Eagles", "New Mexico State Aggies",
        "Florida State Seminoles", "Hawaii Rainbow Warriors",
        "Stanford Cardinal", "Memphis Tigers", "UNLV Rebels",
    ]
    rows = IDENTITY.review_names(names, 2026, "the-odds-api")
    assert all(row["status"] == "resolved" for row in rows)
    assert all(row["match_mode"] == "reviewed_alias" for row in rows)
    assert len({row["team_id"] for row in rows}) == len(names)


def test_identity_registry_refuses_unverified_targets(tmp_path, monkeypatch):
    bad = tmp_path / "aliases.json"
    bad.write_text(json.dumps({"aliases": [{
        "provider": "the-odds-api", "alias": "Made Up Mascots",
        "team_id": "999999", "canonical": "Made Up", "valid_from": 2026,
        "valid_to": 2026, "reviewed_at": "2026-08-01"}]}))
    monkeypatch.setattr(IDENTITY, "ALIASES_JSON", bad)
    with pytest.raises(ValueError, match="not canonical"):
        IDENTITY.alias_index(2026, "the-odds-api")


def test_validation_dataset_fingerprint_changes_with_content(tmp_path):
    path = tmp_path / "lines.csv"
    path.write_text("season,week,home_team,away_team,line\n2025,1,A,B,-3.5\n")
    first = DATASET_FP.fingerprint(path, source="test", decision_time="close")
    path.write_text(
        "season,week,home_team,away_team,line\n"
        "2025,1,A,B,-3.5\n2025,2,C,D,7.0\n")
    second = DATASET_FP.fingerprint(path, source="test", decision_time="close")
    assert first["sha256"] != second["sha256"]
    assert first["rows"] == 1
    assert second["rows"] == 2
    assert second["season_counts"] == {"2025": 2}


def test_validation_gate_refuses_unreviewed_line_fingerprint(monkeypatch):
    metrics = {
        "ml_brier": 0.18, "margin_mae": 12.0, "total_mae": 13.0,
        "data_fingerprint": {"line_sha256": "new"},
    }
    baseline = {
        "ml_brier": 0.18, "margin_mae": 12.0, "total_mae": 13.0,
        "data_fingerprint": {"line_sha256": "reviewed"},
    }
    monkeypatch.setattr(VALIDATE, "_load_baseline", lambda: baseline)
    assert VALIDATE.gate(metrics) == 1
    baseline["data_fingerprint"]["line_sha256"] = "new"
    assert VALIDATE.gate(metrics) == 0


def test_nested_weight_selection_cannot_see_holdout(monkeypatch):
    monkeypatch.setattr(VALIDATE, "_week_block_ci", lambda *a, **k: {})
    selection = pd.DataFrame({
        "season": [2023, 2023, 2024, 2024], "week": [1, 2, 1, 2],
        "p_elo": [0.9, 0.1, 0.9, 0.1], "p_pow": [0.5] * 4,
        "m_elo": [10, -10, 10, -10], "m_pow": [10, -10, 10, -10],
        "t_pow": [50] * 4, "margin": [10, -10, 10, -10], "total": [50] * 4,
    })
    holdout_a = pd.DataFrame({
        "season": [2025, 2025], "week": [1, 2], "p_elo": [0.99, 0.01],
        "p_pow": [0.01, 0.99], "m_elo": [10, -10], "m_pow": [-10, 10],
        "t_pow": [50, 50], "margin": [10, -10], "total": [50, 50],
    })
    holdout_b = holdout_a.copy()
    holdout_b[["p_elo", "p_pow"]] = holdout_b[["p_pow", "p_elo"]]
    first = VALIDATE.nested_holdout_from_frame(
        pd.concat([selection, holdout_a], ignore_index=True), 2024, 2025)
    second = VALIDATE.nested_holdout_from_frame(
        pd.concat([selection, holdout_b], ignore_index=True), 2024, 2025)
    assert first["locked_w_elo"] == second["locked_w_elo"]
    assert first["locked_w_elo"] > 0.5
    assert first["holdout"]["ml_brier"] != second["holdout"]["ml_brier"]


def test_freeze_nested_holdout_writes_audited_runtime_config(tmp_path, monkeypatch):
    artifact = tmp_path / "nested.json"
    blend = tmp_path / "blend.json"
    monkeypatch.setattr(VALIDATE, "NESTED_ARTIFACT", str(artifact))
    monkeypatch.setattr(VALIDATE, "_BLEND_WEIGHT_FILE", str(blend))
    report = {
        "locked_w_elo": 0.55, "selection_window": "2023-2024",
        "selection_games": 100, "holdout_season": 2025, "holdout_games": 50,
        "holdout": {"ml_brier": 0.19},
        "data_fingerprint": {"line_sha256": "reviewed"},
    }
    VALIDATE.freeze_nested_holdout(report)
    assert json.loads(artifact.read_text()) == report
    config = json.loads(blend.read_text())
    assert config["w_elo"] == 0.55
    assert config["method"] == "nested_season_holdout"
    assert config["data_line_sha256"] == "reviewed"


def test_adapter_never_recommends_when_state_is_ineligible():
    rows = [
        {"home": "Alpha", "away": "Beta", "market": "ml", "edge": 0.20},
        {"home": "Alpha", "away": "Beta", "market": "spread", "edge": 0.15},
    ]
    CFBAdapter()._mark_recommended(rows, betting_eligible=False)
    assert all(r["recommended"] is False for r in rows)


def test_adapter_previews_displayed_stakes_with_suite_caps():
    class Store:
        @staticmethod
        def preview_bets(candidates, bankroll):
            assert bankroll == 100.0
            return [{**candidates[0], "stake": 15.0, "stake_capped": True}]

    rows = [{"recommended": True, "event_id": "cfbd:1",
             "stake_gbp": 30.0, "kelly_frac": 0.30}]
    CFBAdapter._preview_recommended(rows, Store, 100.0)
    assert rows[0]["recommended"] is True
    assert rows[0]["stake_gbp"] == 15.0
    assert rows[0]["kelly_frac"] == 0.15
    assert rows[0]["stake_capped"] is True


def test_in_app_edge_loads_policy_and_does_not_silently_skip(monkeypatch, tmp_path):
    odds_path = tmp_path / "odds.csv"
    pd.DataFrame([{"date": "2026-08-29", "odds": 1.9}]).to_csv(
        odds_path, index=False)
    monkeypatch.setattr(CE, "ODDS_CSV", str(odds_path))
    prepared = pd.DataFrame([
        {"date": "2026-08-29", "home": "Alpha", "away": "Beta",
         "neutral": False, "market": "ml", "side": "home", "line": None,
         "odds": 1.9, "p_implied": 0.5, "event_id": "book-1",
         "cfbd_game_id": "123", "bookmaker": "pinnacle",
         "quote_time": "2026-08-01T12:00:00Z", "quote_eligible": True,
         "identity_version": "test",
         "fixture_matched": True, "pair_complete": True},
        {"date": "2026-08-29", "home": "Alpha", "away": "Beta",
         "neutral": False, "market": "ml", "side": "away", "line": None,
         "odds": 1.9, "p_implied": 0.5, "event_id": "book-1",
         "cfbd_game_id": "123", "bookmaker": "pinnacle",
         "quote_time": "2026-08-01T12:00:00Z", "quote_eligible": True,
         "identity_version": "test",
         "fixture_matched": True, "pair_complete": True},
    ])
    monkeypatch.setattr(CE, "prepare_odds", lambda odds, seasons: prepared)
    monkeypatch.setattr(CENGINE.E, "build_as_of", lambda *a, **k: (
        (pd.DataFrame(), {"Alpha": 1500.0, "Beta": 1500.0}, 0.05, 14.0),
        {"betting_eligible": True, "model_season": 2026,
         "prior_mode": "full_prior", "snapshot_hash": "test"}))
    monkeypatch.setattr(CENGINE.P, "load_params", lambda: {
        "teams": {"Alpha": {}, "Beta": {}}, "sigma": 14.0,
        "sigma_total": 14.0})
    monkeypatch.setattr(CENGINE, "blend_predict", lambda *a, **k: {
        "p1": 0.55, "margin": 1.0, "total": 50.0})

    out = CENGINE.cmd_edge({"bankroll": 100.0})
    assert len(out["rows"]) == 2
    assert all(r["market_status"] == "diagnostic" for r in out["rows"])
    assert all(r["stake_gbp"] == 0.0 for r in out["rows"])


def test_market_policy_keeps_ats_and_ml_non_recordable():
    policy = POLICY.load_policy()
    assert policy == {"ml": "diagnostic", "spread": "diagnostic", "total": "paper"}
    assert not POLICY.recordable("ml", policy)
    assert not POLICY.recordable("spread", policy)
    assert not POLICY.recordable("total", policy)


def test_update_propagates_required_fit_failure(tmp_path):
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *cfb.power*) exit 7 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    fake_python.chmod(0o755)
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ.get('PATH', '')}"}
    run = subprocess.run(
        ["bash", str(root / "cfb" / "update.sh")], cwd=root, env=env,
        text=True, capture_output=True, check=False,
    )
    assert run.returncode == 7
    assert "Update complete" not in run.stdout


def test_update_status_is_machine_readable(tmp_path):
    dest = RUN_STATUS.write_status("failure", "validation_gate", "exit 1",
                                   tmp_path / "status.json")
    payload = json.loads(dest.read_text())
    assert payload["status"] == "failure"
    assert payload["step"] == "validation_gate"
    assert payload["message"] == "exit 1"
    assert payload["updated_at"].endswith("+00:00")


def test_preflight_semantically_builds_current_cfb_snapshot():
    cfb = preflight.build_report()["engines"]["cfb"]
    assert cfb["diagnostic_ready"] is True
    assert cfb["model_state"]["model_season"] == 2026
    assert cfb["model_state"]["prior_mode"] == "regression_only"
    assert not any("schedule is unreadable" in issue for issue in cfb["issues"])
    assert not any("power params could not" in issue for issue in cfb["issues"])


def test_quote_gate_requires_exact_fixture_pair_and_fresh_provenance(monkeypatch):
    kickoff = pd.Timestamp("2026-08-29T16:00:00Z")
    registry = {
        "by_id": {"123": {"cfbd_game_id": "123", "date": "2026-08-29",
                            "home": "TCU", "away": "North Carolina",
                            "kickoff": kickoff}},
        "legacy": {},
    }
    monkeypatch.setattr(CE, "fixture_registry", lambda seasons: registry)
    common = {
        "date": "2026-08-29", "home": "TCU", "away": "North Carolina",
        "neutral": 1, "market": "ml", "line": "", "event_id": "odds-1",
        "cfbd_game_id": "123", "commence_time": "2026-08-29T16:00:00Z",
        "bookmaker": "pinnacle", "quote_time": "2026-08-01T11:30:00Z",
        "source": "the-odds-api", "identity_version": "test",
    }
    odds = pd.DataFrame([
        {**common, "side": "home", "odds": 1.50},
        {**common, "side": "away", "odds": 2.70},
    ])
    out = CE.prepare_odds(odds, {2026}, now="2026-08-01T12:00:00Z")
    assert out["fixture_matched"].all()
    assert out["pair_complete"].all()
    assert out["quote_eligible"].all()
    assert out["p_implied"].sum() == pytest.approx(1.0)

    stale = CE.prepare_odds(odds, {2026}, now="2026-08-03T12:00:00Z")
    assert not stale["quote_eligible"].any()


def test_odds_api_snapshot_is_atomic_and_keeps_book_provenance(tmp_path, monkeypatch):
    dest = tmp_path / "odds.csv"
    monkeypatch.setattr(S, "ODDS_CSV", str(dest))
    monkeypatch.setattr(S, "HERE", str(tmp_path))
    slate = pd.DataFrame([{
        "game_id": 123, "season": 2026, "week": 1,
        "date": pd.Timestamp("2026-08-29"), "neutral": True,
        "home_team": "TCU", "home_div": "fbs",
        "away_team": "North Carolina", "away_div": "fbs",
    }])
    payload = [{
        "id": "odds-1", "commence_time": "2026-08-29T16:00:00Z",
        "home_team": "TCU Horned Frogs", "away_team": "North Carolina Tar Heels",
        "bookmakers": [{
            "key": "pinnacle", "last_update": "2026-08-01T11:30:00Z",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "TCU Horned Frogs", "price": 1.50},
                {"name": "North Carolina Tar Heels", "price": 2.70},
            ]}],
        }],
    }]
    monkeypatch.setattr(
        S.urllib.request, "urlopen",
        lambda *a, **k: io.BytesIO(json.dumps(payload).encode()),
    )
    assert S.fetch_odds_api(slate, "secret") == 1
    written = pd.read_csv(dest)
    assert set(written["bookmaker"]) == {"pinnacle"}
    assert set(written["event_id"]) == {"odds-1"}
    assert set(written["cfbd_game_id"].astype(str)) == {"123"}
    assert written["quote_time"].notna().all()
    assert written["identity_version"].notna().all()

    before = dest.read_bytes()
    monkeypatch.setattr(
        S.urllib.request, "urlopen",
        lambda *a, **k: io.BytesIO(b"[]"),
    )
    with pytest.raises(RuntimeError):
        S.fetch_odds_api(slate, "secret")
    assert dest.read_bytes() == before


def test_atomic_data_publish_retains_last_good_on_validation_failure(tmp_path):
    dest = tmp_path / "games.csv"
    dest.write_text("a,b\n1,2\n")
    before = dest.read_bytes()
    with pytest.raises(ValueError):
        FETCH_DATA.atomic_to_csv(pd.DataFrame(columns=["a", "b"]), str(dest),
                                 ["a", "b"], allow_empty=False)
    assert dest.read_bytes() == before

    params = tmp_path / "power.json"
    params.write_text('{"last_good": true}\n')
    before_params = params.read_bytes()
    with pytest.raises(ValueError):
        POWER.save_params({"teams": {}}, str(params))
    assert params.read_bytes() == before_params


def test_settlement_prefers_cfbd_event_id_over_first_name_match(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame([
        {"game_id": 1, "date": "2026-09-01", "home_team": "Alpha",
         "away_team": "Beta", "home_points": 10, "away_points": 20},
        {"game_id": 2, "date": "2026-12-01", "home_team": "Alpha",
         "away_team": "Beta", "home_points": 30, "away_points": 20},
    ]).to_csv(data / "games.csv", index=False)
    monkeypatch.setattr(APP_CFB, "ENGINE_DIR", tmp_path)
    rows = pd.DataFrame([{
        "event_id": "cfbd:2", "match_date": "2026-09-01",
        "home": "Alpha", "away": "Beta", "side": "home", "bet": "ML home",
    }])
    graded = CFBAdapter().grade_open_bets(rows)
    assert graded[0] == ("won", "30-20")


def test_event_id_settlement_survives_postponement(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame([{
        "game_id": 7, "date": "2026-09-12", "home_team": "Alpha",
        "away_team": "Beta", "home_points": 24, "away_points": 17,
    }]).to_csv(data / "games.csv", index=False)
    monkeypatch.setattr(APP_CFB, "ENGINE_DIR", tmp_path)
    rows = pd.DataFrame([{
        "event_id": "cfbd:7", "match_date": "2026-09-01",
        "home": "Alpha", "away": "Beta", "side": "home", "bet": "ML home",
    }])
    assert CFBAdapter().grade_open_bets(rows)[0] == ("won", "24-17")


@pytest.mark.parametrize("games", [
    [{"game_id": 8, "date": "2026-09-01", "home_team": "Beta",
      "away_team": "Alpha", "home_points": 24, "away_points": 17}],
    [{"game_id": 8, "date": "2026-09-01", "home_team": "Alpha",
      "away_team": "Beta", "home_points": None, "away_points": None}],
    [{"game_id": 8, "date": "2026-09-01", "home_team": "Alpha",
      "away_team": "Beta", "home_points": 24, "away_points": 17},
     {"game_id": 8, "date": "2026-09-02", "home_team": "Alpha",
      "away_team": "Beta", "home_points": 21, "away_points": 20}],
])
def test_event_id_settlement_fails_closed_on_bad_identity_or_result(
        games, tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame(games).to_csv(data / "games.csv", index=False)
    monkeypatch.setattr(APP_CFB, "ENGINE_DIR", tmp_path)
    rows = pd.DataFrame([{
        "event_id": "cfbd:8", "match_date": "2026-09-01",
        "home": "Alpha", "away": "Beta", "side": "home", "bet": "ML home",
    }])
    assert CFBAdapter().grade_open_bets(rows) == {}


def test_legacy_settlement_is_date_bounded_and_unambiguous(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame([
        {"game_id": 1, "date": "2026-09-01", "home_team": "Alpha",
         "away_team": "Beta", "home_points": 24, "away_points": 17},
        {"game_id": 2, "date": "2026-12-01", "home_team": "Alpha",
         "away_team": "Beta", "home_points": 10, "away_points": 20},
    ]).to_csv(data / "games.csv", index=False)
    monkeypatch.setattr(APP_CFB, "ENGINE_DIR", tmp_path)
    rows = pd.DataFrame([{
        "event_id": "", "match_date": "2026-09-01",
        "home": "Alpha", "away": "Beta", "side": "home", "bet": "ML home",
    }])
    assert CFBAdapter().grade_open_bets(rows)[0] == ("won", "24-17")
