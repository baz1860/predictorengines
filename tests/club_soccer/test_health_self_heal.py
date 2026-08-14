#!/usr/bin/env python3
"""Adding a reviewed alias must not strand the pipeline.

Putting a new entry in club_alias_map.json necessarily makes the EXISTING
fixtures.csv non-canonical — the names in the file are exactly the ones the new
alias resolves. health flags that, report["ok"] goes False, and season.py used
to abort at step 1, before the step that would canonicalise them. The guard
protecting identity blocked the only thing that repairs identity.

The Mac mini sat in that state for two days after the 2026-08-12 merge, failing
at 06:30 with noncanonical_team_names = 445 and invalid_club_ids = 1023 without
ever touching its data, while the error message pointed at `fetch --repair` —
which only rewrites future-dated finished rows and returns without writing when
there are none.
"""
from __future__ import annotations

import pytest

from club_soccer import health as H
from club_soccer import season as S


def _clean_report(**overrides):
    report = {name: 0 for name in H.HARD_CHECKS}
    report["expired_experiments"] = []
    report["experiment_registry_errors"] = []
    report.update(overrides)
    report["ok"] = not H.failing_hard_checks(report)
    return report


# --- failing_hard_checks --------------------------------------------------

def test_clean_report_has_no_failures():
    assert H.failing_hard_checks(_clean_report()) == []
    assert _clean_report()["ok"] is True


def test_counts_and_collections_both_register():
    assert H.failing_hard_checks(_clean_report(invalid_club_ids=7)) == \
        ["invalid_club_ids"]
    assert H.failing_hard_checks(_clean_report(expired_experiments=["x"])) == \
        ["expired_experiments"]


def test_failures_are_reported_in_declaration_order():
    report = _clean_report(invalid_club_ids=1, future_ft_rows=1)
    assert H.failing_hard_checks(report) == \
        ["future_ft_rows", "invalid_club_ids"]


def test_every_self_healing_check_is_a_hard_check():
    assert H.SELF_HEALING_CHECKS <= set(H.HARD_CHECKS)


def test_disagreement_checks_are_never_self_healing():
    """These mean two sources disagree about a fact — a rewrite would launder
    the disagreement rather than resolve it."""
    for name in ("conflicting_score_identities", "colliding_club_names",
                 "future_ft_rows", "duplicate_fixture_ids",
                 "void_with_results"):
        assert name not in H.SELF_HEALING_CHECKS


# --- the heal itself ------------------------------------------------------

@pytest.fixture
def spy(monkeypatch):
    """Record whether the write boundary was invoked, without touching data."""
    calls = {"write": 0, "recheck": 0}

    def fake_read_csv(*_a, **_k):
        import pandas as pd
        return pd.DataFrame({"home": ["A"], "away": ["B"]})

    def fake_write(df, *_a, **_k):
        calls["write"] += 1
        return df

    def fake_run_checks(*_a, **_k):
        calls["recheck"] += 1
        return _clean_report()

    monkeypatch.setattr(S.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(S.F, "write_fixtures", fake_write)
    monkeypatch.setattr(S.H, "run_checks", fake_run_checks)
    return calls


def test_alias_only_failure_heals(spy):
    report = _clean_report(noncanonical_team_names=445, invalid_club_ids=1023)
    healed = S._heal_canonicalisation(report)
    assert spy["write"] == 1 and spy["recheck"] == 1
    assert healed["ok"] is True


def test_corrupt_data_is_not_rewritten(spy):
    """A future-dated finished row must never be laundered by a rewrite."""
    report = _clean_report(future_ft_rows=3)
    healed = S._heal_canonicalisation(report)
    assert spy["write"] == 0, "write_fixtures must not run for corrupt data"
    assert healed is report


def test_mixed_failure_heals_nothing(spy):
    """Healable + unhealable together must abort, not partially rewrite."""
    report = _clean_report(noncanonical_team_names=445,
                           conflicting_score_identities=2)
    healed = S._heal_canonicalisation(report)
    assert spy["write"] == 0
    assert healed is report


def test_healthy_report_is_left_alone(spy):
    report = _clean_report()
    assert S._heal_canonicalisation(report) is report
    assert spy["write"] == 0


def test_write_failure_returns_the_original_report(monkeypatch):
    """A failed rewrite must not be mistaken for a successful heal."""
    import pandas as pd
    monkeypatch.setattr(S.pd, "read_csv",
                        lambda *a, **k: pd.DataFrame({"home": ["A"]}))

    def boom(*_a, **_k):
        raise ValueError("refusing to shrink fixtures.csv")

    monkeypatch.setattr(S.F, "write_fixtures", boom)
    report = _clean_report(noncanonical_team_names=445)
    healed = S._heal_canonicalisation(report)
    assert healed is report
    assert healed["ok"] is False


def test_still_failing_after_rewrite_is_reported(monkeypatch):
    """If the rewrite does not clear it, the caller must still abort."""
    import pandas as pd
    monkeypatch.setattr(S.pd, "read_csv",
                        lambda *a, **k: pd.DataFrame({"home": ["A"]}))
    monkeypatch.setattr(S.F, "write_fixtures", lambda df, *a, **k: df)
    monkeypatch.setattr(
        S.H, "run_checks",
        lambda *a, **k: _clean_report(noncanonical_team_names=445))
    healed = S._heal_canonicalisation(_clean_report(noncanonical_team_names=445))
    assert healed["ok"] is False
