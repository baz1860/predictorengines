#!/usr/bin/env python3
"""Tests for P5 variance inflation — implemented, measured, and left OFF.

This is a negative result kept honest by tests. The mechanism works; the
measurement says enabling it would make predictions worse, because the
miscalibration it was designed to fix was cured by measuring the clubs
instead (P3 domestic data, P4b league seeding, xg gating).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from club_soccer import model as M


def _params(**counts):
    return {"team_evidence": {t: {"n_recent": n} for t, n in counts.items()}}


# ── the mechanism ─────────────────────────────────────────────────────────

def test_inflation_is_off_by_default():
    assert M.VARIANCE_INFLATION_DEFAULT is False


def test_lambda_falls_as_evidence_grows():
    p = _params(A=0, B=0)
    lots = _params(A=500, B=500)
    assert M.inflation_lambda(p, "A", "B") > M.inflation_lambda(lots, "A", "B")


def test_lambda_is_capped():
    for n in (0, 1, 5, 50, 500):
        lam = M.inflation_lambda(_params(A=n, B=n), "A", "B")
        assert 0.0 <= lam <= M.MAX_INFLATION


def test_lambda_keys_off_the_weaker_side():
    mixed = M.inflation_lambda(_params(A=500, B=0), "A", "B")
    both_thin = M.inflation_lambda(_params(A=0, B=0), "A", "B")
    assert mixed == both_thin


def test_zero_lambda_is_a_no_op():
    m = np.array([[0.4, 0.1], [0.2, 0.3]])
    out = M.apply_variance_inflation(m, 0.0)
    assert np.allclose(out, m)


def test_inflation_preserves_total_probability():
    m = np.array([[0.4, 0.1], [0.2, 0.3]])
    for lam in (0.1, 0.25, 1.0):
        out = M.apply_variance_inflation(m, lam)
        assert out.sum() == pytest.approx(1.0)


def test_inflation_preserves_expected_goals():
    """It should widen the distribution without moving its centre."""
    rng = np.random.default_rng(0)
    m = rng.random((6, 6))
    m = m / m.sum()
    before_h = float(sum(i * m[i, :].sum() for i in range(m.shape[0])))
    out = M.apply_variance_inflation(m, 0.25)
    after_h = float(sum(i * out[i, :].sum() for i in range(out.shape[0])))
    assert after_h == pytest.approx(before_h, abs=1e-9)


def test_inflation_reduces_confidence_in_the_modal_scoreline():
    m = np.array([[0.7, 0.05], [0.05, 0.2]])
    out = M.apply_variance_inflation(m, 0.25)
    assert out.max() < m.max()


# ── predict integration ───────────────────────────────────────────────────

def test_predict_default_matches_explicit_off():
    params = M.load_params()
    teams = params["teams"][:2]
    a = M.predict(teams[0], teams[1], "Premier League", params=params)
    b = M.predict(teams[0], teams[1], "Premier League", params=params,
                  variance_inflation=False)
    assert a["probs"] == b["probs"]


def test_enabling_inflation_changes_a_thin_matchup():
    params = M.load_params()
    store = params.get("team_evidence") or {}
    thin = [t for t, v in store.items() if v.get("n_recent", 0) < 5]
    if len(thin) < 2:
        pytest.skip("no sufficiently thin clubs")
    off = M.predict(thin[0], thin[1], None, params=params, variance_inflation=False)
    on = M.predict(thin[0], thin[1], None, params=params, variance_inflation=True)
    assert off["probs"] != on["probs"]


def test_walk_forward_cache_key_includes_variance_inflation():
    """A fit/predict option missing from the key means the cache serves
    results from a different model."""
    import inspect

    from club_soccer import validate as V
    src = inspect.getsource(V.walk_forward)
    assert '"variance_inflation": variance_inflation' in src


# ── the negative result must stay documented ──────────────────────────────

def test_evidence_artifact_justifies_leaving_it_off():
    from pathlib import Path
    path = Path(M.DATA) / "variance_inflation_evidence.json"
    assert path.exists(), "a rejected feature still needs its evidence recorded"
    doc = json.loads(path.read_text())
    ab = doc["ab_walk_forward_2024_07_onward"]
    assert ab["delta"]["brier"] > 0, \
        "if inflation ever measures BETTER, this test should fail and prompt a rethink"
    assert doc["status"].startswith("IMPLEMENTED BUT OFF")
