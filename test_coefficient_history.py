#!/usr/bin/env python3
"""Time-versioned UEFA coefficients (adversarial finding 2 — the seed leak)."""
from __future__ import annotations

import json

import pytest

from club_soccer import uefa_registry as R


def test_history_artifact_has_snapshots_with_dates():
    doc = json.loads(R.COEFFICIENTS_HISTORY.read_text())
    snaps = doc["snapshots"]
    assert len(snaps) >= 5
    for v in snaps.values():
        assert "published_on" in v and "coefficients" in v
        assert len(v["coefficients"]) >= 50


def test_as_of_selects_a_period_correct_snapshot():
    """A fold in 2021 must not see coefficients published in 2025."""
    early = R.strength_prior("Netherlands", 1, as_of="2021-08-01")
    late = R.strength_prior("Netherlands", 1, as_of="2025-08-01")
    # The Netherlands climbed the coefficient table over this span; an early
    # fold must get the LOWER, period-correct prior.
    assert early < late, "as_of must return the coefficients of its own era"


def test_no_snapshot_used_is_published_after_the_as_of_date():
    """The core anti-leak invariant, checked directly against the history."""
    hist = R._load_history()
    for as_of in ("2021-08-01", "2022-08-01", "2023-08-01", "2024-08-01"):
        snap = R._snapshot_for(as_of)
        # the snapshot returned must be one whose publication date <= as_of
        matching = [c for pub, c in hist if pub <= as_of]
        assert matching, f"no snapshot on/before {as_of}"
        assert snap is matching[-1], "must use the latest snapshot NOT after as_of"


def test_none_uses_the_latest_snapshot():
    hist = R._load_history()
    assert R._snapshot_for(None) is hist[-1][1]


def test_anchors_are_relative_and_scale_invariant():
    """England -> 1.00, Scotland -> 0.58 in EVERY snapshot, regardless of the
    source's absolute rescaling between captures."""
    for as_of in (None, "2021-08-01", "2023-08-01"):
        assert R.strength_prior("England", 1, as_of=as_of) == pytest.approx(1.00, abs=0.001)
        assert R.strength_prior("Scotland", 1, as_of=as_of) == pytest.approx(0.58, abs=0.001)


def test_production_priors_are_ordinally_sane():
    top = [R.strength_prior(c, 1) for c in
           ("England", "Italy", "Spain", "Germany", "Netherlands", "Austria")]
    assert top == sorted(top, reverse=True)


def test_unknown_country_falls_back_to_default():
    assert R.strength_prior("Atlantis", 1) == R.DEFAULT_PRIOR


def test_lower_tiers_discounted():
    assert R.strength_prior("England", 2) < R.strength_prior("England", 1)


# ── the fit / walk-forward wiring ─────────────────────────────────────────

def test_fit_accepts_coef_as_of():
    import inspect
    from club_soccer import model as M
    assert "coef_as_of" in inspect.signature(M.fit).parameters


def test_walk_forward_passes_the_fold_cutoff_as_coef_as_of():
    """Without this the historical folds seed from today's coefficients — the
    exact leak. The cutoff must flow into fit()."""
    import inspect
    from club_soccer import validate as V
    src = inspect.getsource(V.walk_forward)
    assert "coef_as_of=cutoff" in src


def test_coef_history_is_in_the_cache_fingerprint():
    """A coefficient refresh must invalidate cached folds, or a stale fold is
    served after the annual update."""
    from club_soccer import walkforward_cache as WFC
    assert "uefa_coefficients_history.json" in WFC._DATA_FILES


def test_seeding_changes_with_as_of_in_a_real_fit():
    """End-to-end: the same club seeded under two eras gets different Elo."""
    from club_soccer import model as M
    df = M.played(M.load_fixtures())
    early = M.fit(df, league_seed=True, coef_as_of="2021-08-01")
    late = M.fit(df, league_seed=True, coef_as_of="2025-08-01")
    # A Dutch club's seed should differ between eras (Netherlands moved).
    dutch = [t for t in early["elo"] if t in late["elo"]]
    assert any(abs(early["elo"][t] - late["elo"][t]) > 1e-6 for t in dutch), \
        "coef_as_of must actually change the seeds"
