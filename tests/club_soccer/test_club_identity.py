"""Tests for the ingest-time club identity resolver."""
from __future__ import annotations

import pandas as pd

from club_soccer import club_identity as CI


def test_normalisation_and_core_are_shared_and_conservative():
    assert CI._norm("Atlético Madrid") == CI._norm("Atletico Madrid")
    assert CI._core("Bologna FC 1909") == CI._core("Bologna")


def test_reviewed_aliases_resolve():
    CI.reload_resolver()
    assert CI.canonical_name("FC Bayern München") == "Bayern Munich"
    assert CI.canonical_name("Heart of Midlothian") == "Hearts"


def test_unknown_club_passes_through():
    assert CI.canonical_name("Entirely New Club 2026") == "Entirely New Club 2026"


def test_country_guard_refuses_cross_country_collision(monkeypatch):
    monkeypatch.setattr(CI, "team_countries", lambda refresh=False: {})
    assert CI.canonical_name("Athletic Club", country="Brazil") == "Athletic Club"


def test_verified_cross_border_alias_is_allowed(monkeypatch):
    monkeypatch.setattr(CI, "team_countries", lambda refresh=False: {})
    assert CI.canonical_name("AS Monaco", country="France") == "Monaco"
    assert CI.canonical_name("Cardiff City", country="England") == "Cardiff"


def test_registry_aliases_share_a_stable_id():
    assert CI.canonical_id("Aarhus") == CI.canonical_id("AGF Aarhus")


def test_europe_only_report_excludes_domestically_observed_clubs():
    frame = pd.DataFrame([
        {"season": 2025, "competition": "Champions League",
         "home": "Sturm Graz", "away": "Bayern Munich"},
        {"season": 2025, "competition": "Bundesliga",
         "home": "Bayern Munich", "away": "Dortmund"},
    ])
    report = CI.europe_only_teams(frame)
    assert "Sturm Graz" in report
    assert "Bayern Munich" not in report


def test_alias_artifact_has_no_chains():
    alias, _ = CI._load_resolver()
    assert all(target not in alias for target in alias.values())
