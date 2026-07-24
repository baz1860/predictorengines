#!/usr/bin/env python3
"""Regression: season-run concurrency safety and monitor state validation.

Covers:
  * an exclusive non-blocking lock rejects a concurrent season run;
  * the UUID run_id is preserved from the running marker into the finished
    record, and status publication is conditional on owning that run_id;
  * the monitor rejects future-dated running markers, unknown state values and
    non-object top-level JSON instead of passing or raising.

Run: python3 -m pytest test_run_status.py -q
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from club_soccer import monitor as MON


# ── monitor state validation ─────────────────────────────────────────────
@pytest.fixture()
def last_run(tmp_path, monkeypatch):
    p = tmp_path / "last_run.json"
    monkeypatch.setattr(MON, "LAST_RUN", p)
    return p


def _write(p, payload):
    p.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)


def _now(offset_hours=0.0):
    return (datetime.now(timezone.utc)
            + timedelta(hours=offset_hours)).isoformat()


def test_future_dated_running_marker_fails(last_run):
    # The review's repro: a marker dated 24h in the future used to report
    # "-1440m elapsed" and stay green.
    _write(last_run, {"state": "running", "run_id": "abc",
                      "started_at_utc": _now(+24)})
    ok, failures, _ = MON.check()
    assert ok is False
    assert any("FUTURE" in f for f in failures)


def test_recent_running_marker_is_ok(last_run):
    _write(last_run, {"state": "running", "run_id": "abc",
                      "started_at_utc": _now(-0.25)})
    ok, failures, warnings = MON.check()
    assert ok is True and not failures
    assert any("in progress" in w for w in warnings)


def test_dead_running_marker_fails(last_run):
    _write(last_run, {"state": "running", "run_id": "abc",
                      "started_at_utc": _now(-3)})
    ok, failures, _ = MON.check()
    assert ok is False
    assert any("never finished" in f for f in failures)


@pytest.mark.parametrize("state", ["weird", "FINISHED", "done", "", 0, True])
def test_unknown_state_fails(last_run, state):
    # A bogus state with an otherwise-green payload must not pass.
    _write(last_run, {"state": state, "run_id": "abc",
                      "finished_at_utc": _now(-1), "ok": True})
    ok, failures, _ = MON.check()
    assert ok is False
    assert any("unknown run state" in f for f in failures)


@pytest.mark.parametrize("raw", ["[]", '[1,2,3]', '"a string"', "123", "null"])
def test_non_object_top_level_fails_without_raising(last_run, raw):
    _write(last_run, raw)
    ok, failures, _ = MON.check()          # must not raise AttributeError
    assert ok is False and failures


def test_finished_ok_passes(last_run):
    _write(last_run, {"state": "finished", "run_id": "abc",
                      "finished_at_utc": _now(-1), "ok": True})
    ok, failures, _ = MON.check()
    assert ok is True and not failures


# ── season run locking + run_id continuity ───────────────────────────────
def test_season_lock_rejects_concurrent_run(tmp_path, monkeypatch):
    from club_soccer import season as S
    monkeypatch.setattr(S, "DATA", tmp_path)
    monkeypatch.setattr(S, "LOCK_FILE", tmp_path / "season.lock")
    with S._SeasonLock(tmp_path / "season.lock"):
        with pytest.raises(SystemExit) as ei:
            with S._SeasonLock(tmp_path / "season.lock"):
                pass
        assert "concurrently" in str(ei.value)
    # released: acquiring again now succeeds
    with S._SeasonLock(tmp_path / "season.lock"):
        pass


def test_run_id_preserved_and_publication_is_owned(tmp_path, monkeypatch):
    from club_soccer import season as S
    monkeypatch.setattr(S, "DATA", tmp_path)
    monkeypatch.setattr(S, "LAST_RUN", tmp_path / "last_run.json")

    S._write_running_marker("run-one")
    marker = json.loads((tmp_path / "last_run.json").read_text())
    assert marker["state"] == "running" and marker["run_id"] == "run-one"

    # A different run must NOT overwrite a marker it does not own.
    S._publish_status({"state": "finished", "run_id": "run-two"}, "run-two")
    still = json.loads((tmp_path / "last_run.json").read_text())
    assert still["run_id"] == "run-one", "foreign run clobbered the marker"

    # The owning run may publish its finished record, keeping the same id.
    S._publish_status({"state": "finished", "run_id": "run-one"}, "run-one")
    done = json.loads((tmp_path / "last_run.json").read_text())
    assert done["state"] == "finished" and done["run_id"] == "run-one"


def test_stale_running_marker_does_not_wedge_future_runs(tmp_path, monkeypatch):
    """A run killed mid-flight leaves state=running. The NEXT run must be able
    to claim the marker — otherwise the dead run's id blocks status writes
    forever and the monitor stays red permanently."""
    from club_soccer import season as S
    monkeypatch.setattr(S, "DATA", tmp_path)
    monkeypatch.setattr(S, "LAST_RUN", tmp_path / "last_run.json")

    S._write_running_marker("dead-run")          # previous run, killed
    S._write_running_marker("fresh-run")         # next run claims ownership
    marker = json.loads((tmp_path / "last_run.json").read_text())
    assert marker["run_id"] == "fresh-run"

    # and it can finalize, because it now owns the marker
    S._publish_status({"state": "finished", "run_id": "fresh-run"}, "fresh-run")
    done = json.loads((tmp_path / "last_run.json").read_text())
    assert done["state"] == "finished" and done["run_id"] == "fresh-run"


def test_unique_tmp_paths_per_run(tmp_path, monkeypatch):
    from club_soccer import season as S
    monkeypatch.setattr(S, "DATA", tmp_path)
    monkeypatch.setattr(S, "LAST_RUN", tmp_path / "last_run.json")
    S._publish_status({"state": "running", "run_id": "aaa"}, "aaa")
    # no shared last_run.json.tmp left behind for another run to collide with
    assert not (tmp_path / "last_run.json.tmp").exists()
