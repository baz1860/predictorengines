#!/usr/bin/env python3
"""Regression: validation is descriptive; the promotion gate is promoter-owned.

A reseed that silently re-baselines can hide a regression by redefining
"normal". These tests pin that club_soccer.validate exposes no baseline writer
and that --update-baseline is gone from the CLI and from every seeder.

Run: python3 -m pytest test_baseline_ownership.py -q
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from club_soccer import validate as V

ROOT = Path(__file__).resolve().parent
SEEDERS = ["club_soccer/seed_footballdata.py",
           "club_soccer/seed_real.py",
           "club_soccer/seed_openfootball.py"]


def test_validate_has_no_update_baseline_flag():
    r = subprocess.run([sys.executable, "-m", "club_soccer.validate",
                        "--update-baseline"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
    assert "unrecognized arguments: --update-baseline" in r.stderr


def test_no_seeder_moves_the_baseline():
    # Look for the flag as a passed ARGUMENT (quoted), not as prose in a
    # comment explaining why it was removed.
    for rel in SEEDERS:
        text = (ROOT / rel).read_text()
        assert '"--update-baseline"' not in text, f"{rel} still moves the baseline"
        assert "'--update-baseline'" not in text, f"{rel} still moves the baseline"


def test_validation_module_never_writes_the_promotion_baseline():
    src = (ROOT / "club_soccer" / "validate.py").read_text()
    # No write of either the promoter-owned file or the legacy baseline.
    assert "PROMOTION_BASELINE.write_text" not in src
    assert "BASELINE.write_text" not in src


def test_promotion_baseline_is_readable_and_separate_from_latest():
    base = V.load_promotion_baseline()
    assert isinstance(base, dict) and "brier" in base
    assert V.PROMOTION_BASELINE.name == "promotion_baseline.json"
    assert V.LATEST.name == "validation_latest.json"
    assert V.PROMOTION_BASELINE != V.LATEST


def test_gate_without_any_baseline_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "PROMOTION_BASELINE", tmp_path / "promotion_baseline.json")
    monkeypatch.setattr(V, "BASELINE", tmp_path / "validation_baseline.json")
    assert V.load_promotion_baseline() is None
