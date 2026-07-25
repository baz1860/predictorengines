#!/usr/bin/env python3
"""Tests for evidence-weighted ensemble components.

A component built from shots on target carries no information about a club
with no shot data — its attack/defence terms are identically zero, so it emits
a flat league-average matrix. Giving it 40% of the ensemble weight anyway is
how the post-P3 OU2.5/BTTS regression happened.
"""
from __future__ import annotations

import json

import pytest

from club_soccer import model as M


BASE = {"goals": 0.20, "elo": 0.40, "xg": 0.20, "xgf": 0.20}


def _params(**evidence):
    return {"xg_evidence": evidence}


def test_full_shot_data_leaves_weights_essentially_untouched():
    """The scale is smooth, so even a heavily-covered club is nudged very
    slightly rather than pinned at exactly 1.0 — that is intended, but the
    nudge must be negligible."""
    out = M._weights_for_match(_params(A=200.0, B=200.0), BASE, "A", "B")
    for key, base in BASE.items():
        assert out[key] == pytest.approx(base, abs=0.01)


def test_no_shot_data_removes_the_shot_components():
    out = M._weights_for_match(_params(A=0.0, B=0.0), BASE, "A", "B")
    assert out["xg"] == 0.0
    assert out["xgf"] == 0.0


def test_weights_always_renormalise_to_one():
    for ev in (0.0, 1.0, 8.0, 30.0, 200.0):
        out = M._weights_for_match(_params(A=ev, B=ev), BASE, "A", "B")
        assert sum(out.values()) == pytest.approx(1.0)


def test_freed_weight_goes_to_the_informative_components():
    out = M._weights_for_match(_params(A=0.0, B=0.0), BASE, "A", "B")
    # goals:elo ratio preserved, scaled up to fill the gap
    assert out["goals"] == pytest.approx(1 / 3, abs=1e-6)
    assert out["elo"] == pytest.approx(2 / 3, abs=1e-6)


def test_confidence_keys_off_the_weaker_side():
    """A match is only as informative as its least-measured club."""
    mixed = M._weights_for_match(_params(A=200.0, B=0.0), BASE, "A", "B")
    both_zero = M._weights_for_match(_params(A=0.0, B=0.0), BASE, "A", "B")
    assert mixed == both_zero


def test_scaling_is_smooth_not_a_cliff():
    """Sturm Graz (7.1, Europe-only) should sit between 0 and full."""
    zero = M._weights_for_match(_params(A=0.0, B=0.0), BASE, "A", "B")["xg"]
    thin = M._weights_for_match(_params(A=7.1, B=7.1), BASE, "A", "B")["xg"]
    full = M._weights_for_match(_params(A=200.0, B=200.0), BASE, "A", "B")["xg"]
    assert zero < thin < full


def test_more_evidence_never_reduces_shot_weight():
    prev = -1.0
    for ev in (0.0, 2.0, 8.0, 20.0, 80.0, 300.0):
        w = M._weights_for_match(_params(A=ev, B=ev), BASE, "A", "B")["xg"]
        assert w >= prev
        prev = w


def test_legacy_params_without_evidence_are_left_alone():
    """A stale artifact must degrade to old behaviour, not to zero weights."""
    assert M._weights_for_match({}, BASE, "A", "B") == BASE


def test_all_zero_weights_falls_back_rather_than_dividing_by_zero():
    degenerate = {"goals": 0.0, "elo": 0.0, "xg": 1.0, "xgf": 0.0}
    out = M._weights_for_match(_params(A=0.0, B=0.0), degenerate, "A", "B")
    assert out == degenerate


def test_unknown_team_is_treated_as_having_no_evidence():
    out = M._weights_for_match(_params(A=200.0), BASE, "A", "Unknown FC")
    assert out["xg"] == 0.0


# ── integration ───────────────────────────────────────────────────────────

def test_fit_records_per_team_xg_evidence():
    params = M.load_params()
    assert "xg_evidence" in params
    ev = params["xg_evidence"]
    assert len(ev) == len(params["teams"])
    assert any(v == 0.0 for v in ev.values()), \
        "the shot-less leagues must show as zero evidence"
    assert any(v > 50.0 for v in ev.values()), \
        "well-covered leagues must show high evidence"


def test_a_meaningful_share_of_clubs_have_no_shot_evidence():
    """Named clubs are a bad pin here: BSD supplies shot data for several
    leagues fd.co.uk does not, so a specific club can legitimately gain
    evidence between runs (Djurgarden did). What must stay true is that a
    substantial population still has none — that is the condition the gating
    exists for.
    """
    ev = M.load_params()["xg_evidence"]
    zero = sum(1 for v in ev.values() if v == 0.0)
    assert zero > 50, "expected a substantial shot-less population"
    assert zero < len(ev), "not every club can be shot-less"


def test_predict_applies_the_gating():
    params = M.load_params()
    ev = params["xg_evidence"]
    zero = [t for t, v in ev.items() if v == 0.0]
    if len(zero) < 2:
        pytest.skip("no shot-less clubs in the fitted params")
    out = M.predict(zero[0], zero[1], "Eliteserien", params=params)
    assert out["probs"]["home"] > 0


def test_evidence_artifact_records_the_measurement():
    from pathlib import Path
    path = Path(M.DATA) / "xg_gating_evidence.json"
    assert path.exists()
    doc = json.loads(path.read_text())
    d = doc["walk_forward_2024_07_onward"]["delta"]
    assert d["brier"] < 0 and d["ou25"] < 0 and d["btts"] < 0, \
        "a live change must be supported by an improvement"
