#!/usr/bin/env python3
"""Tests for P6 — run history and operational readiness."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from club_soccer import run_ledger as RL


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(RL, "LEDGER", tmp_path / "run_history.jsonl")
    monkeypatch.setattr(RL, "DATA", tmp_path)
    return tmp_path


def _run(days_ago=0, ok=True, stale=None, full_ev=585, gate=True, run_id="abc12345"):
    when = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {"run_id": run_id, "ok": ok, "finished_at_utc": when.isoformat(),
            "failed_required_steps": [], "stale_leagues_no_bsd": stale or [],
            "teams_full_evidence": full_ev, "gate_pass": gate,
            "gate_brier": 0.6119, "gate_limit": 0.6212, "fixtures_rows": 52074}


def _seed(n, **kw):
    for i in range(n, 0, -1):
        RL.append(_run(days_ago=i, run_id=f"run{i:05d}", **kw))


# ── ledger mechanics ──────────────────────────────────────────────────────

def test_append_and_read_round_trip(ledger):
    RL.append(_run())
    assert len(RL.history()) == 1


def test_corrupt_line_is_skipped_not_fatal(ledger):
    RL.append(_run())
    with RL.LEDGER.open("a") as fh:
        fh.write("{not json\n")
    RL.append(_run(run_id="later"))
    assert len(RL.history()) == 2


def test_append_never_raises(ledger, monkeypatch):
    """Observability must not be able to fail the pipeline it observes."""
    monkeypatch.setattr(RL, "LEDGER", ledger / "nope" / "deep" / "x.jsonl")
    RL.append(_run())          # must not raise


def test_ledger_is_bounded(ledger, monkeypatch):
    monkeypatch.setattr(RL, "MAX_ENTRIES", 10)
    _seed(25)
    assert len(RL.history()) <= 10


def test_history_is_oldest_first(ledger):
    _seed(3)
    entries = RL.history()
    stamps = [e["finished_at_utc"] for e in entries]
    assert stamps == sorted(stamps)


# ── streak logic ──────────────────────────────────────────────────────────

def test_streak_counts_consecutive_healthy_runs(ledger):
    _seed(5)
    streak, _ = RL.green_streak()
    assert streak == 5


def test_a_failure_breaks_the_streak(ledger):
    RL.append(_run(days_ago=4, run_id="old1"))
    RL.append(_run(days_ago=3, ok=False, run_id="bad01"))
    RL.append(_run(days_ago=2, run_id="new1"))
    RL.append(_run(days_ago=1, run_id="new2"))
    streak, reason = RL.green_streak()
    assert streak == 2
    assert "failed" in reason


def test_a_stale_league_breaks_the_streak(ledger):
    RL.append(_run(days_ago=3, run_id="old1"))
    RL.append(_run(days_ago=2, stale=["Austrian Bundesliga"], run_id="stale"))
    RL.append(_run(days_ago=1, run_id="new1"))
    streak, reason = RL.green_streak()
    assert streak == 1
    assert "stale" in reason


def test_a_long_gap_breaks_the_streak(ledger):
    """14 green runs spread over two months is not a healthy daily pipeline."""
    RL.append(_run(days_ago=40, run_id="ancient"))
    RL.append(_run(days_ago=1, run_id="recent"))
    streak, reason = RL.green_streak()
    assert streak == 1
    assert "gap" in reason


def test_failures_are_recorded_not_dropped(ledger):
    """A ledger of successes alone cannot measure a streak."""
    RL.append(_run(ok=False, run_id="failed1"))
    assert RL.history()[0]["ok"] is False


# ── readiness gate ────────────────────────────────────────────────────────

def test_empty_ledger_is_not_ready_and_reports_unknown(ledger):
    """Absence of evidence must not read as evidence of health."""
    result = RL.readiness()
    assert result["ready"] is False
    assert result["runs_recorded"] == 0
    for c in result["checks"]:
        assert not c["pass"], "nothing may pass on an empty ledger"
        if "stale" in c["check"]:
            assert "unknown" in c["detail"]


def test_enough_green_runs_reaches_ready(ledger):
    _seed(RL.REQUIRED_GREEN_RUNS)
    result = RL.readiness()
    assert result["green_streak"] >= RL.REQUIRED_GREEN_RUNS
    assert result["ready"] is True


def test_short_streak_is_not_ready(ledger):
    _seed(3)
    assert RL.readiness()["ready"] is False


def test_failing_gate_blocks_readiness(ledger):
    _seed(RL.REQUIRED_GREEN_RUNS, gate=False)
    result = RL.readiness()
    assert result["ready"] is False
    assert any(not c["pass"] and "gate" in c["check"] for c in result["checks"])


def test_eroding_coverage_blocks_readiness(ledger):
    """The slow failure: every run green while the expansion decays."""
    for i in range(RL.REQUIRED_GREEN_RUNS + 1, 1, -1):
        RL.append(_run(days_ago=i, run_id=f"r{i:05d}", full_ev=585))
    RL.append(_run(days_ago=1, run_id="today", full_ev=300))
    result = RL.readiness()
    assert result["ready"] is False
    assert any(not c["pass"] and "eroding" in c["check"] for c in result["checks"])


def test_stable_coverage_does_not_block_readiness(ledger):
    _seed(RL.REQUIRED_GREEN_RUNS, full_ev=585)
    assert RL.readiness()["ready"] is True


# ── wiring ────────────────────────────────────────────────────────────────

def test_season_appends_to_the_ledger():
    import inspect

    from club_soccer import season
    src = inspect.getsource(season._write_last_run)
    assert "run_ledger" in src, "every run must be recorded, including failures"


def test_monitor_reports_readiness_without_failing_on_it():
    """A young ledger is not an incident — escalating it would train the
    operator to ignore the alert."""
    import inspect

    from club_soccer import monitor
    src = inspect.getsource(monitor.main)
    assert "readiness" in src
    assert "SystemExit" not in src.split("readiness")[1]


def test_snapshot_collects_the_fields_readiness_needs():
    snap = RL.snapshot()
    for key in ("teams_full_evidence", "fixtures_rows", "stale_leagues_no_bsd",
                "gate_pass"):
        assert key in snap, f"snapshot must provide {key}"
