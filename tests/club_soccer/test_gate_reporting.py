#!/usr/bin/env python3
"""The promotion gate must be reported honestly and re-pinned deliberately.

Three defects, one shape: a signal that did not measure the thing it was named
after.

* run_ledger recorded `gate_pass` from the Brier alone, ignoring the population
  hash and row count that `validate --gate` actually enforces. From 2026-08-01
  the hash stopped matching, update.sh exited 1 every day, and run_history.jsonl
  logged gate_pass=true for two weeks.
* Re-baselining is legitimate when the SAMPLE changes (dedupe, identity merge)
  and illegitimate when the MODEL regresses. Nothing distinguished the two.
* seed_fdcouk_leagues re-raised any non-404, but football-data.co.uk answers
  300 Multiple Choices for a season file that is not published yet.
"""
from __future__ import annotations

import json
import urllib.error

import pytest

from club_soccer import run_ledger as RL
from club_soccer import seed_fdcouk_leagues as SFL
from club_soccer import promote_baseline as PB
from club_soccer import validate as V


# --- failure partitioning -------------------------------------------------

def test_population_changes_are_separated_from_regressions():
    population, metric = V.partition_gate_failures([
        "evaluation row count 26969 != baseline 27181",
        "evaluation population hash a03d != baseline 2dfb",
        "brier 0.700000 > baseline 0.611889 + tolerance 0.001000",
    ])
    assert len(population) == 2
    assert metric == ["brier 0.700000 > baseline 0.611889 + tolerance 0.001000"]


def test_no_failures_partitions_to_nothing():
    assert V.partition_gate_failures([]) == ([], [])


# --- promote_baseline -----------------------------------------------------

BASE = {
    "brier": 0.6118889338548972, "log_loss": 1.020562853318792,
    "brier_ou25": 0.24517773729185988, "brier_btts": 0.24690999320297685,
    "n": 27181, "test_from": "2024-07-01", "test_to": "2026-07-01",
    "evaluation_hash": "2dfbba7b5209342cf325",
    "tolerances": {"brier": 0.001, "log_loss": 0.0015,
                   "brier_ou25": 0.001, "brier_btts": 0.001},
}


def _measured(**over):
    m = {"brier": 0.6117778, "log_loss": 1.0204061,
         "brier_ou25": 0.2452018, "brier_btts": 0.2469441, "n": 26969}
    m.update(over)
    return m


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    path = tmp_path / "promotion_baseline.json"
    path.write_text(json.dumps(BASE))
    monkeypatch.setattr(V, "PROMOTION_BASELINE", path)
    monkeypatch.setattr(PB.WFC, "code_fingerprint", lambda: "deadbeef")
    monkeypatch.setattr(PB.WFC, "row_hashes", lambda df: __import__(
        "numpy").array([1, 2, 3], dtype="uint64"))
    monkeypatch.setattr(PB.M, "load_fixtures", lambda *a, **k: __import__(
        "pandas").DataFrame({"date": [], "home_goals": [], "away_goals": []}))
    monkeypatch.setattr(PB.M, "played", lambda df: __import__(
        "pandas").DataFrame({"date": []}))
    monkeypatch.setattr(V, "_evaluation_hash", lambda rows: "a03d8581acefec09")
    return path


def test_population_change_is_repinned(isolated):
    payload = PB.build_payload([{"x": 1}], _measured(), dict(BASE))
    assert payload["n"] == 26969
    assert payload["evaluation_hash"] == "a03d8581acefec09"
    assert payload["superseded"]["n"] == 27181
    assert payload["superseded"]["evaluation_hash"] == "2dfbba7b5209342cf325"
    assert payload["promoted_at_utc"] and payload["promoted_on_host"]


def test_tolerances_and_window_survive_the_repin(isolated):
    payload = PB.build_payload([{"x": 1}], _measured(), dict(BASE))
    assert payload["tolerances"] == BASE["tolerances"]
    assert payload["test_from"] == "2024-07-01"
    assert payload["test_to"] == "2026-07-01"


def test_metric_regression_is_refused(isolated):
    """The one thing re-baselining must never be able to do."""
    with pytest.raises(ValueError, match="metric regressions"):
        PB.build_payload([{"x": 1}], _measured(brier=0.75), dict(BASE))
    assert json.loads(isolated.read_text())["n"] == 27181, \
        "the baseline file must be untouched when refused"
    assert "PROMOTION_BASELINE.write_text" not in \
        (V.__file__ and open(V.__file__).read()), \
        "validate.py must never write the gate reference"


def test_force_overrides_a_regression(isolated):
    payload = PB.build_payload([{"x": 1}], _measured(brier=0.75),
                                 dict(BASE), force=True)
    assert payload["brier"] == 0.75
    assert payload["superseded"]["reason"]


def test_passing_gate_has_nothing_to_repin(isolated, monkeypatch):
    monkeypatch.setattr(V, "_evaluation_hash",
                        lambda rows: BASE["evaluation_hash"])
    with pytest.raises(ValueError, match="nothing to re-baseline"):
        PB.build_payload([{"x": 1}], _measured(n=27181), dict(BASE))


def test_missing_previous_baseline_is_refused(isolated):
    with pytest.raises(ValueError, match="no existing"):
        PB.build_payload([{"x": 1}], _measured(), None)


# --- gate_pass reporting --------------------------------------------------

def test_gate_pass_follows_the_gate_state_not_the_brier(tmp_path, monkeypatch):
    """A good Brier on a different population is not a pass."""
    monkeypatch.setattr(RL, "DATA", tmp_path)
    (tmp_path / "validation_latest.json").write_text(json.dumps(
        {"brier": 0.6117778}))                     # comfortably inside tolerance
    (tmp_path / "promotion_baseline.json").write_text(json.dumps(BASE))
    (tmp_path / "validation_gate_state.json").write_text(json.dumps({
        "passed": False,
        "failures": ["evaluation population hash f993 != baseline 2dfb"],
        "checked_at_utc": "2026-08-01T13:35:46+00:00",
    }))
    out = RL.snapshot()
    assert out["gate_pass"] is False
    assert out["gate_failures"] == \
        ["evaluation population hash f993 != baseline 2dfb"]
    assert out["gate_checked_at_utc"] == "2026-08-01T13:35:46+00:00"


def test_missing_gate_state_is_not_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(RL, "DATA", tmp_path)
    (tmp_path / "validation_latest.json").write_text(json.dumps({"brier": 0.1}))
    (tmp_path / "promotion_baseline.json").write_text(json.dumps(BASE))
    out = RL.snapshot()
    assert out["gate_pass"] is False
    assert "gate_failures" in out


# --- fd.co.uk status handling ---------------------------------------------

def _http_error(code):
    return urllib.error.HTTPError("http://x", code, "boom", {}, None)


@pytest.mark.parametrize("code", [300, 404])
def test_unpublished_season_returns_none(code, tmp_path, monkeypatch):
    monkeypatch.setattr(SFL, "CACHE", tmp_path)

    def raiser(*_a, **_k):
        raise _http_error(code)

    monkeypatch.setattr(SFL.urllib.request, "urlopen", raiser)
    assert SFL._fetch("http://x/I2.csv", "I2.csv") is None


@pytest.mark.parametrize("code", [403, 500, 503])
def test_real_errors_still_raise(code, tmp_path, monkeypatch):
    """A broken source must not be mistaken for an early one."""
    monkeypatch.setattr(SFL, "CACHE", tmp_path)

    def raiser(*_a, **_k):
        raise _http_error(code)

    monkeypatch.setattr(SFL.urllib.request, "urlopen", raiser)
    with pytest.raises(urllib.error.HTTPError):
        SFL._fetch("http://x/I2.csv", "I2.csv")
