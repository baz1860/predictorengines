#!/usr/bin/env python3
"""Regression: only out-of-sample evidence may deploy the World Cup blend.

Round-7 finding 2: the fitter selected the weight and scored it on the SAME
rows, then the loader called those numbers a "frozen holdout". An artifact whose
weight was chosen on the data it is judged by proves nothing, so the gate now
reads holdout_* fields exclusively and ignores in_sample_*.

Run: python3 -m pytest test_blend_gate.py -q
"""
from __future__ import annotations

import json
import hashlib

import pytest

from engines.worldcup import market_blend as MB

MARGIN = MB.BLEND_MIN_MARGIN


def _artifact(**over):
    """A minimally VALID, promotable artifact; override to break one thing."""
    d = {
        "model_weight": 0.3,
        "holdout_method": "leave_one_tournament_out",
        "holdout_logloss_blend": 0.90,
        "holdout_logloss_model_only": 0.95,
        "holdout_logloss_market_only": 0.94,
        "provenance": {"generated_at_utc": "2026-07-20T00:00:00+00:00",
                       "n_rows": 128,
                       "inputs": {"data/wc2018_odds.csv": "a" * 64,
                                  "data/wc2022_odds.csv": "b" * 64}},
        "active": True,
    }
    d.update(over)
    return d


@pytest.fixture()
def blend_file(tmp_path, monkeypatch):
    p = tmp_path / "market_blend.json"
    inputs = {}
    for name, payload in (("data/wc2018_odds.csv", b"2018 test odds\n"),
                          ("data/wc2022_odds.csv", b"2022 test odds\n")):
        source = tmp_path / name.replace("/", "_")
        source.write_bytes(payload)
        inputs[name] = source
    monkeypatch.setattr(MB, "PROVENANCE_INPUTS", inputs)
    monkeypatch.setattr(MB, "BLEND_FILE", p)
    return p, {name: hashlib.sha256(path.read_bytes()).hexdigest()
               for name, path in inputs.items()}


def _load(blend_file, artifact):
    blend_file, _ = blend_file
    blend_file.write_text(json.dumps(artifact))
    return MB.load_w()


def _artifact_for(blend_file, **over):
    _, inputs = blend_file
    values = {"provenance": {"generated_at_utc": "2026-07-20T00:00:00+00:00",
                              "n_rows": 128, "inputs": inputs}}
    values.update(over)
    return _artifact(**values)


def test_valid_holdout_artifact_deploys(blend_file):
    assert _load(blend_file, _artifact_for(blend_file)) == pytest.approx(0.3)


def test_in_sample_only_artifact_can_never_open_the_gate(blend_file):
    """The exact defect: great-looking in-sample numbers, no holdout."""
    art = _artifact_for(blend_file)
    for k in ("holdout_method", "holdout_logloss_blend",
              "holdout_logloss_model_only", "holdout_logloss_market_only"):
        art.pop(k)
    art.update({"in_sample_logloss_blend": 0.10,      # spectacular, and meaningless
                "in_sample_logloss_model_only": 0.95,
                "in_sample_logloss_market_only": 0.94})
    assert _load(blend_file, art) is None


def test_legacy_logloss_field_names_do_not_qualify(blend_file):
    """Pre-round-7 artifacts used bare logloss_* (in-sample). Fail closed."""
    art = _artifact_for(blend_file)
    art["logloss_blend"] = art.pop("holdout_logloss_blend")
    art["logloss_model_only"] = art.pop("holdout_logloss_model_only")
    art["logloss_market_only"] = art.pop("holdout_logloss_market_only")
    assert _load(blend_file, art) is None


def test_inactive_artifact_is_never_deployed(blend_file):
    assert _load(blend_file, _artifact_for(blend_file, active=False)) is None


@pytest.mark.parametrize("method", ["in_sample", "full_sample", "", None, "loto"])
def test_unrecognised_holdout_method_fails_closed(blend_file, method):
    assert _load(blend_file, _artifact_for(blend_file, holdout_method=method)) is None


def test_tie_with_market_is_not_an_edge(blend_file):
    """The real WC case: holdout blend == market exactly."""
    assert _load(blend_file, _artifact_for(
        blend_file, holdout_logloss_blend=0.94,
        holdout_logloss_market_only=0.94)) is None


def test_margin_must_be_strictly_cleared_on_both_endpoints(blend_file):
    # beats model comfortably, but only ties market within the margin
    assert _load(blend_file, _artifact_for(blend_file,
        holdout_logloss_blend=0.94 - MARGIN / 2)) is None
    # clears both by more than the margin
    assert _load(blend_file, _artifact_for(blend_file,
        holdout_logloss_blend=0.94 - MARGIN * 10)) is not None


def test_missing_or_empty_provenance_fails_closed(blend_file):
    assert _load(blend_file, _artifact_for(blend_file, provenance=None)) is None
    assert _load(blend_file, _artifact_for(blend_file, provenance={"inputs": {}})) is None
    assert _load(blend_file, _artifact_for(blend_file,
        provenance={"inputs": {"data/wc2018_odds.csv": None}})) is None


@pytest.mark.parametrize("w", [-0.1, 1.1, "0.3", None, float("nan")])
def test_out_of_range_or_non_finite_weight_fails_closed(blend_file, w):
    assert _load(blend_file, _artifact_for(blend_file, model_weight=w)) is None


def test_forged_or_stale_input_hashes_can_never_deploy(blend_file):
    _, hashes = blend_file
    forged = dict(hashes)
    forged["data/wc2018_odds.csv"] = "0" * 64
    assert _load(blend_file, _artifact_for(
        blend_file, provenance={"inputs": forged})) is None
    assert _load(blend_file, _artifact_for(
        blend_file, provenance={"inputs": {"not-a-real-input": "a"}})) is None


def test_shipped_artifact_is_demoted_and_yields_none():
    """The real data/market_blend.json must stay non-deployable."""
    d = json.loads(MB.BLEND_FILE.read_text())
    assert d["active"] is False
    assert "w" not in d, "deprecated alias must not reappear as a second source of truth"
    assert d["holdout_method"] == "leave_one_tournament_out"
    # holdout blend ties market exactly -> no edge
    assert d["holdout_logloss_blend"] >= d["holdout_logloss_market_only"] - MARGIN
    assert MB.load_w() is None
