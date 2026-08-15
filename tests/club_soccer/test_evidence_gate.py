#!/usr/bin/env python3
"""Malformed-structure fuzz + contract tests for club_soccer.evidence_gate.

The evaluator contract is: evaluate() must NEVER raise on ANY JSON input, and
staking_allowed() must fail CLOSED. These tests hammer the parser with
non-object top levels, duplicate keys (top-level and deeply nested), and random
junk structures, and assert the gate degrades to a reason instead of crashing.

Run: python3 -m pytest test_evidence_gate.py -q
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from club_soccer import evidence_gate as EG


@pytest.fixture()
def gate_file(tmp_path, monkeypatch):
    """Point the gate at a throwaway artifact path we control per test."""
    p = tmp_path / "backtest_market.json"
    monkeypatch.setattr(EG, "BACKTEST_JSON", p)
    return p


def _evaluate_never_raises(text: str):
    """Write raw text as the artifact, evaluate, and assert it did not raise."""
    result = EG.evaluate()
    assert isinstance(result, dict)
    assert result["allowed"] is False           # nothing malformed can open the gate
    assert result["reasons"]                    # and it must say why
    return result


# ── non-object top levels ────────────────────────────────────────────────
@pytest.mark.parametrize("raw", [
    "null", "true", "false", "123", "1e3", "-4.5",
    '"a string"', "[]", "[1, 2, 3]", '{"unterminated": ',
    "", "   ", "not json at all", "NaN", "Infinity",
])
def test_non_object_or_broken_top_level_never_raises(gate_file, raw):
    gate_file.write_text(raw)
    res = _evaluate_never_raises(raw)
    # staking_allowed wraps evaluate and must also fail closed without raising
    allowed, reasons = EG.staking_allowed()
    assert allowed is False and reasons


# ── duplicate keys must be rejected, not last-value-wins ─────────────────
def test_duplicate_top_level_key_rejected(gate_file):
    gate_file.write_text('{"backtest_version": "legacy", '
                         '"backtest_version": "decision_time_v3"}')
    res = EG.evaluate()
    assert res["allowed"] is False
    assert any("duplicate key" in r.lower() for r in res["reasons"])


def test_duplicate_nested_key_rejected(gate_file):
    # A duplicate nested n_bets is exactly the "silently valid" attack: the
    # first value is a legitimate 1500, the second a poisoned 0.
    artifact = (
        '{"backtest_version": "decision_time_v3",'
        ' "simulated_betting": {"1x2": {"2%": '
        '{"n_bets": 1500, "n_bets": 0}}}}'
    )
    gate_file.write_text(artifact)
    res = EG.evaluate()
    assert res["allowed"] is False
    assert any("duplicate key" in r.lower() for r in res["reasons"])


# ── nested non-dict rows must fail as reasons, not crash ─────────────────
@pytest.mark.parametrize("sim", [
    None, [], "x", 3, True,
    {"1x2": None}, {"1x2": []}, {"1x2": "x"},
    {"1x2": {"2%": None}}, {"1x2": {"2%": [1, 2]}},
    {"1x2": {"2%": {"n_bets": None}}},
])
def test_malformed_simulated_betting_never_raises(gate_file, sim):
    gate_file.write_text(json.dumps({
        "backtest_version": "decision_time_v3",
        "selection_method": "latest_quote_at_or_before_decision_time",
        "execution_method": "same_decision_time_quote",
        "clv_reference": "captured_closing_devigged",
        "decision_lead_minutes": 90,
        "generated_at_utc": "2999-01-01T00:00:00+00:00",
        "simulated_betting": sim,
    }))
    res = EG.evaluate()
    assert isinstance(res, dict) and res["allowed"] is False


# ── randomized structural fuzz: evaluate() must never raise ──────────────
def _random_json(rng, depth=0):
    if depth > 4:
        return rng.choice([None, 0, 1.5, "s", True])
    kind = rng.randrange(6)
    if kind == 0:
        return rng.choice([None, True, False, 0, -1, 1e9, "text", float("nan")])
    if kind == 1:
        return [(_random_json(rng, depth + 1)) for _ in range(rng.randrange(4))]
    keys = rng.sample(
        ["backtest_version", "simulated_betting", "1x2", "2%", "4%", "6%",
         "n_bets", "flat_roi", "kelly_roi", "clv_mean", "clv_frac_positive",
         "decision_lead_minutes", "generated_at_utc", "model_log_loss_1x2"],
        k=rng.randrange(1, 5))
    return {k: _random_json(rng, depth + 1) for k in keys}


def test_random_structure_fuzz_never_raises(gate_file):
    rng = random.Random(20260719)
    for _ in range(1000):
        try:
            gate_file.write_text(json.dumps(_random_json(rng)))
        except (ValueError, TypeError):
            continue                              # NaN etc. — skip un-writable
        res = EG.evaluate()                       # MUST NOT raise
        assert isinstance(res, dict)
        assert "allowed" in res and res["allowed"] is False
        allowed, reasons = EG.staking_allowed()
        assert allowed is False


def test_missing_artifact_fails_closed(gate_file):
    assert not gate_file.exists()
    res = EG.evaluate()
    assert res["allowed"] is False
    assert any("no backtest_market.json" in r for r in res["reasons"])
