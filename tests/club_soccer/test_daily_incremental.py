"""Contracts for the low-power daily Club Soccer pipeline."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from club_soccer import cache_retention as CR
from club_soccer import model as M
from club_soccer import player_features as PF
from club_soccer import season
from club_soccer import snapshot_odds as SO
from club_soccer import validate as V
from club_soccer import walkforward_cache as WFC


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "fixture_id": "1", "date": pd.Timestamp("2026-01-01"),
            "season": 2025, "competition": "Premier League", "type": "league",
            "neutral": 0, "home": "A", "away": "B",
            "home_goals": 1.0, "away_goals": 0.0, "status": "FT",
        },
        {
            "fixture_id": "2", "date": pd.Timestamp("2026-08-01"),
            "season": 2026, "competition": "Premier League", "type": "league",
            "neutral": 0, "home": "B", "away": "A",
            "home_goals": None, "away_goals": None, "status": "NS",
        },
    ])


def test_player_cache_only_ingests_new_or_changed_event_bundles(
    tmp_path, monkeypatch
):
    event_dir = tmp_path / "bsd_cache"
    event_dir.mkdir()
    event = {
        "id": 1, "event_date": "2026-01-01T12:00:00Z",
        "home_team": "A", "away_team": "B",
        "lineups": {
            "home": {"players": [
                {"id": 10, "name": "Player A", "minutes": 90, "position": "FW"}
            ]},
            "away": {"players": [
                {"id": 20, "name": "Player B", "minutes": 90, "position": "DF"}
            ]},
        },
    }
    event_path = event_dir / "event_1.json"
    event_path.write_text(json.dumps(event))
    monkeypatch.setattr(PF, "STATS_CACHE", event_dir)

    store = PF.PlayerFeatureStore(tmp_path / "players.json").load()
    assert store.refresh_from_cache() == 1
    first_mtime = store._path.stat().st_mtime_ns
    assert store.refresh_from_cache() == 0
    assert store._path.stat().st_mtime_ns == first_mtime

    event["lineups"]["home"]["players"][0]["minutes"] = 80
    event["lineups"]["away"]["players"] = []
    event_path.write_text(json.dumps(event))
    assert store.refresh_from_cache() == 1
    player = store._data["id:10"]
    assert len(player["apps"]) == 1
    assert player["apps"][0]["mins"] == 80.0
    assert store._data["id:20"]["apps"] == []


def test_model_fit_is_reused_until_training_inputs_change(tmp_path, monkeypatch):
    frame = _fixtures()
    monkeypatch.setattr(M, "PARAMS", tmp_path / "params.json")
    monkeypatch.setattr(M, "load_fixtures", lambda: frame.copy())
    monkeypatch.setattr(WFC, "code_fingerprint", lambda: "stable-code")
    calls = []

    def fake_fit(fixtures):
        calls.append(1)
        return {"teams": ["A", "B"], "fitted_matches": 1}

    monkeypatch.setattr(M, "fit", fake_fit)

    _params, changed = M.fit_if_changed()
    assert changed and len(calls) == 1
    _params, changed = M.fit_if_changed()
    assert not changed and len(calls) == 1

    # Upcoming-board changes do not affect training and must not force a fit.
    frame.loc[1, "date"] = pd.Timestamp("2026-08-02")
    _params, changed = M.fit_if_changed()
    assert not changed and len(calls) == 1

    frame.loc[0, "home_goals"] = 2.0
    _params, changed = M.fit_if_changed()
    assert changed and len(calls) == 2


def test_fixed_window_gate_fingerprint_ignores_later_results(monkeypatch):
    frame = _fixtures()
    later = frame.iloc[[0]].copy()
    later["fixture_id"] = "3"
    later["date"] = pd.Timestamp("2026-07-15")
    baseline = {"test_from": "2025-01-01", "test_to": "2026-07-01"}
    monkeypatch.setattr(WFC, "code_fingerprint", lambda: "stable-code")
    monkeypatch.setattr(M, "load_fixtures", lambda: frame.copy())
    before = V.gate_input_fingerprint(baseline)

    frame = pd.concat([frame, later], ignore_index=True)
    after_later_result = V.gate_input_fingerprint(baseline)
    assert after_later_result == before

    frame.loc[0, "home_goals"] = 3.0
    assert V.gate_input_fingerprint(baseline) != before


def test_gate_state_reuse_is_exact_input_only(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "GATE_STATE", tmp_path / "state.json")
    metrics = {
        "n": 10, "accuracy": 0.5, "brier": 0.6, "log_loss": 1.0,
        "brier_ou25": 0.25, "brier_btts": 0.25,
    }
    V._write_gate_state("abc", True, metrics, [])
    assert V._load_gate_state("abc")["passed"] is True
    assert V._load_gate_state("different") is None


def test_cache_pruning_is_weekly(tmp_path, monkeypatch):
    monkeypatch.setattr(CR, "STATE", tmp_path / "state.json")
    calls = []
    monkeypatch.setattr(CR, "prune_all", lambda: calls.append(1) or {"ok": {}})
    CR.prune_all_if_due(now=1_000_000)
    CR.prune_all_if_due(now=1_000_100)
    assert calls == [1]
    CR.prune_all_if_due(now=1_000_000 + 8 * 86400)
    assert calls == [1, 1]


def test_daily_network_index_is_fetched_once_and_shared(monkeypatch):
    today = str(datetime.now(timezone.utc).date())
    events = [{
        "id": 1, "status": "notstarted",
        "event_date": f"{today}T12:00:00Z",
    }]
    fetched = []
    seen = {}

    monkeypatch.setattr(
        season.F, "_fetch_bsd_events",
        lambda *a, **k: fetched.append(1) or events,
    )
    monkeypatch.setattr(
        season.F, "fetch_fixtures",
        lambda *a, **k: seen.setdefault("fixtures", k["events"]),
    )
    monkeypatch.setattr(
        season.PF, "pull_absences",
        lambda *a, **k: seen.setdefault("absences", k["events"]),
    )

    class Store:
        def load(self):
            return self

        def refresh_from_cache(self):
            return 0

    monkeypatch.setattr(season.PF, "PlayerFeatureStore", Store)
    monkeypatch.setattr(
        season.SO, "build_snapshot_rows",
        lambda *a, **k: seen.setdefault("snapshots", k["events"]) and [],
    )
    monkeypatch.setattr(season.SO, "append_snapshots", lambda rows: rows)
    monkeypatch.setattr(season, "_fdcouk_is_stale", lambda: False)

    from club_soccer import seed_fdcouk_leagues as SFL
    monkeypatch.setattr(
        SFL, "refresh", lambda verbose=True: {"per_competition": {}}
    )

    upcoming, _odds = season.run_network_steps("key")
    assert fetched == [1]
    assert upcoming is events or upcoming == events
    assert seen == {
        "fixtures": events, "absences": events, "snapshots": events,
    }


def test_far_odds_events_are_not_polled_again_every_day(tmp_path, monkeypatch):
    from club_soccer import competitions

    monkeypatch.setattr(SO, "POLL_STATE", tmp_path / "poll.json")
    monkeypatch.setattr(
        competitions, "comp_from_bsd_league",
        lambda _name: type("Comp", (), {"name": "Premier League"})(),
    )
    calls = []
    monkeypatch.setattr(
        SO, "odds_comparison", lambda *_a, **_k: calls.append(1) or {}
    )
    future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=10)
    events = [{
        "id": 99, "status": "notstarted",
        "event_date": future.isoformat(),
        "league": {"name": "Premier League"},
        "home_team": "A", "away_team": "B",
    }]
    SO.build_snapshot_rows("key", events=events, verbose=False)
    SO.build_snapshot_rows("key", events=events, verbose=False)
    assert calls == [1]


def test_decision_backtest_reuses_unchanged_ledgers(tmp_path, monkeypatch):
    from club_soccer import decision_ledger as DL
    from club_soccer import decision_time_backtest as DTB

    monkeypatch.setattr(DTB, "ARTIFACT", tmp_path / "artifact.json")
    monkeypatch.setattr(DTB, "LEDGER", tmp_path / "replay.csv")
    monkeypatch.setattr(DL, "DECISIONS", tmp_path / "decisions.csv")
    monkeypatch.setattr(DL, "SETTLEMENTS", tmp_path / "settlements.csv")
    first = DTB.run(verbose=False)

    monkeypatch.setattr(
        DTB, "build_bets",
        lambda **_k: (_ for _ in ()).throw(AssertionError("should reuse")),
    )
    second = DTB.run(verbose=False)
    assert second == first
