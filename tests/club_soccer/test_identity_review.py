"""Tests for the read-only identity ambiguity report."""
from __future__ import annotations

import pandas as pd

from club_soccer import identity_review as IR


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2025, "date": "2025-09-01",
         "competition": "Champions League",
         "home": "SK Sturm Graz", "away": "Bayern Munich"},
        {"season": 2025, "date": "2025-09-04",
         "competition": "Austrian Bundesliga",
         "home": "Sturm Graz", "away": "Salzburg"},
        {"season": 2025, "date": "2025-09-05",
         "competition": "Bundesliga",
         "home": "Bayern Munich", "away": "Dortmund"},
    ])


def test_report_suggests_but_does_not_expose_a_verdict_field(monkeypatch):
    monkeypatch.setattr(IR, "_load_prior_decisions", lambda: {})
    rows = IR.build_rows(_fixtures())
    sturm = next(row for row in rows if row["europe_only_name"] == "SK Sturm Graz")
    assert sturm["suggested_match"] == "Sturm Graz"
    assert "VERDICT" not in sturm


def test_export_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(IR, "build_rows", lambda: [{
        "assessment": "manual review required",
        "europe_only_name": "A",
        "registry_country": "",
        "suggested_match": "B",
        "suggested_league": "Test League",
        "confidence": 0.8,
        "n_matches": 2,
        "seasons": "2025",
        "reason": "name-only hint",
    }])
    target = tmp_path / "report.csv"
    assert IR.export(target) == target
    assert target.exists()


def test_no_apply_entry_point_exists():
    assert not hasattr(IR, "apply")
    assert not hasattr(IR, "read_review")
