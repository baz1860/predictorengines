#!/usr/bin/env python3
"""Tests for P4 — hierarchical competition-strength estimation."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from club_soccer import competitions as C
from club_soccer import league_strength as LS


def _fx(rows):
    return pd.DataFrame(
        [{"date": d, "season": s, "competition": c, "home": h, "away": a,
          "home_goals": 1.0, "away_goals": 0.0} for d, s, c, h, a in rows])


# ── the plausibility gate ─────────────────────────────────────────────────

def test_gate_rejects_the_incumbent_output():
    """The shipped comp_strength.json ranked three top leagues below the
    Scottish Premiership. Any estimator that does this is not promotable."""
    incumbent = {"Premier League": 1.0, "Bundesliga": 0.5451, "Serie A": 0.6235,
                 "La Liga": 0.5835, "Ligue 1": 0.6270,
                 "Scottish Premiership": 0.6979}
    ok, failures = LS.plausibility(incumbent)
    assert not ok
    assert len(failures) == 4


def test_gate_accepts_a_sane_table():
    sane = {"Premier League": 1.00, "La Liga": 0.90, "Serie A": 0.89,
            "Bundesliga": 0.88, "Ligue 1": 0.80, "Scottish Premiership": 0.58,
            "Championship": 0.70, "Serie B": 0.62}
    ok, failures = LS.plausibility(sane)
    assert ok, failures


def test_gate_catches_a_second_tier_above_its_top_tier():
    broken = {"Premier League": 1.00, "Championship": 1.05,
              "Scottish Premiership": 0.58}
    ok, failures = LS.plausibility(broken)
    assert not ok
    assert any("Championship" in f for f in failures)


def test_gate_catches_an_implausibly_weak_premier_league():
    ok, failures = LS.plausibility({"Premier League": 0.4,
                                    "Scottish Premiership": 0.3})
    assert not ok
    assert any("implausibly low" in f for f in failures)


# ── connectivity ──────────────────────────────────────────────────────────

def test_primary_league_picks_the_most_played_competition():
    df = _fx([("2024-08-01", 2024, "Bundesliga", "Bayern Munich", "Dortmund"),
              ("2024-08-08", 2024, "Bundesliga", "Bayern Munich", "Leipzig"),
              ("2024-09-18", 2024, "Champions League", "Bayern Munich", "Ajax")])
    prim = LS.primary_league(df)
    assert prim[("Bayern Munich", 2024)] == "Bundesliga"


def test_connectivity_counts_cross_league_matches_only():
    df = _fx([("2024-08-01", 2024, "Bundesliga", "Bayern Munich", "Dortmund"),
              ("2024-08-08", 2024, "Eredivisie", "Ajax", "PSV"),
              ("2024-09-18", 2024, "Champions League", "Bayern Munich", "Ajax")])
    n = LS.connectivity(df)
    # Only the Champions League tie links the two leagues.
    assert n.get("Bundesliga") == 1
    assert n.get("Eredivisie") == 1


def test_connectivity_counts_matches_against_unknown_leagues():
    """Clubs from associations we cannot source have no primary league.

    Those matches still measure our league's clubs against outsiders, so they
    must count — otherwise smaller leagues look far less connected than they
    are, and get over-shrunk toward the prior.
    """
    df = _fx([("2024-08-01", 2024, "Ekstraklasa", "Lech Poznan", "Legia"),
              ("2024-09-18", 2024, "Champions League", "Lech Poznan", "Unknown SK")])
    n = LS.connectivity(df)
    assert n.get("Ekstraklasa") == 1


# ── shrinkage behaviour ───────────────────────────────────────────────────

def test_zero_connectivity_league_gets_exactly_its_prior():
    """A league with no inter-league matches has had its strength measured by
    nothing. Returning the external prior is the honest answer; returning a
    fitted number would be fabrication."""
    result = LS.fit(k=10.5)
    by_name = {r["competition"]: r for r in result["rows"]}
    for name in ("Parva Liga", "Liga 3", "Scottish League Two"):
        row = by_name.get(name)
        if row is None or row["n_inter_league"] != 0:
            continue
        assert row["shrink_weight"] == 0.0
        assert row["fitted"] == pytest.approx(row["prior"], abs=1e-4)


def test_well_connected_leagues_follow_their_data():
    result = LS.fit(k=10.5)
    by_name = {r["competition"]: r for r in result["rows"]}
    pl = by_name["Premier League"]
    assert pl["n_inter_league"] > 100
    assert pl["shrink_weight"] > 0.9


def test_shrink_weight_is_monotonic_in_connectivity():
    """Monotonic in connectivity WITHIN a prior-confidence group.

    Leagues on the uninformative default prior use a smaller effective K, so
    globally two leagues with equal connectivity can carry different weights.
    Monotonicity is asserted per group, which is the correct contract after the
    per-league-K change.
    """
    from club_soccer import uefa_registry as R
    result = LS.fit(k=10.5)
    rows = [r for r in result["rows"] if r["observed"] is not None]
    for informative in (True, False):
        group = [r for r in rows
                 if R.prior_is_informative(r["country"]) == informative]
        pairs = sorted((r["n_inter_league"], r["shrink_weight"]) for r in group)
        weights = [w for _n, w in pairs]
        assert weights == sorted(weights), \
            f"weights not monotonic within informative={informative}"


def test_default_prior_leagues_follow_their_own_data():
    """A league on the flat default prior must track its observed strength, not
    be anchored to 0.75 — that was the participant-prior fix."""
    from club_soccer import uefa_registry as R
    result = LS.fit(k=10.5)
    for r in result["rows"]:
        if (r["observed"] is not None and r["n_inter_league"] >= 20
                and not R.prior_is_informative(r["country"])):
            assert abs(r["fitted"] - r["observed"]) < abs(r["fitted"] - r["prior"]), \
                f"{r['competition']} still anchored to the default prior"


def test_larger_k_pulls_everything_closer_to_the_prior():
    loose = LS.fit(k=5.0)
    tight = LS.fit(k=320.0)
    loose_rows = {r["competition"]: r for r in loose["rows"]}
    tight_rows = {r["competition"]: r for r in tight["rows"]}
    for name, lr in loose_rows.items():
        tr = tight_rows[name]
        if lr["observed"] is None:
            continue
        assert abs(tr["fitted"] - tr["prior"]) <= abs(lr["fitted"] - lr["prior"]) + 1e-9


def test_fitted_table_passes_the_gate():
    result = LS.fit(k=LS.DEFAULT_K)
    ok, failures = LS.plausibility(result["strengths"])
    assert ok, f"P4 estimator must pass its own gate: {failures}"


def test_cups_inherit_a_discounted_parent_league():
    result = LS.fit(k=LS.DEFAULT_K)
    s = result["strengths"]
    if "FA Cup" in s and "Premier League" in s:
        assert s["FA Cup"] == pytest.approx(LS.CUP_DISCOUNT * s["Premier League"],
                                            abs=1e-3)


# ── the artifact must stay inert ──────────────────────────────────────────

def test_artifact_is_written_gated_off():
    doc = json.loads(LS.COMP_STRENGTH.read_text())
    assert doc["active"] is False, "comp_strength.json must not self-promote"
    assert doc["_method"].startswith("hierarchical")
    assert "_k" in doc and "_fit_at_utc" in doc


def test_gated_artifact_does_not_change_pricing():
    """The whole point of active=false. If this fails, a report-only artifact
    is silently moving live prices."""
    C.reload_comp_strength()
    for name in ("Bundesliga", "Serie A", "Eredivisie", "Scottish Premiership"):
        comp = C.get(name)
        assert C.strength(name) == comp.strength


def test_promotion_requires_the_gate_to_pass():
    doc = json.loads(LS.COMP_STRENGTH.read_text())
    if doc.get("active"):
        assert doc.get("_plausible") is True, \
            "an active comp_strength.json must have passed the plausibility gate"


# ── K estimation ──────────────────────────────────────────────────────────

def test_default_k_is_in_a_sane_range():
    assert 1.0 <= LS.DEFAULT_K <= 500.0


def test_estimate_k_returns_a_bounded_value():
    out = LS.estimate_k(verbose=False)
    assert 1.0 <= out["k"] <= 500.0
    assert out["method"]


# ── P4b: league-prior team seeding ────────────────────────────────────────

def test_league_seeding_is_promoted():
    """PROMOTED 2026-07-22 on the evidence in league_seed_evidence.json."""
    from club_soccer import model as M
    assert M.LEAGUE_SEED_DEFAULT is True


def test_promotion_is_recorded_in_code_not_toggled_at_runtime():
    """The promotion must be a visible constant, not an env var or data flag."""
    import inspect

    from club_soccer import model as M
    src = inspect.getsource(M)
    assert "LEAGUE_SEED_DEFAULT = True" in src
    assert "PROMOTED" in src


def test_promotion_evidence_artifact_exists_and_supports_it():
    from pathlib import Path
    path = Path(LS.DATA) / "league_seed_evidence.json"
    assert path.exists(), "a promotion must carry its evidence"
    doc = json.loads(path.read_text())
    assert doc["delta_brier"] < 0, "promoted change must improve Brier"
    assert len(doc["results"]) >= 2, "single-window evidence is not enough"
    for r in doc["results"]:
        assert r["brier_seeded"] < r["brier_default"]


def test_validation_default_tracks_the_production_default():
    """The gate must measure the model that actually runs.

    If these two drift apart, validation silently scores a model nobody uses —
    the failure mode is a green gate on an unvalidated production model.
    """
    import inspect

    from club_soccer import model as M
    from club_soccer import validate as V
    assert inspect.signature(V.walk_forward).parameters["league_seed"].default is None
    src = inspect.getsource(V.walk_forward)
    assert "M.LEAGUE_SEED_DEFAULT if league_seed is None" in src


def test_pre_promotion_arm_is_still_reproducible():
    """Explicitly passing False must reproduce the old model, so the A/B that
    justified the promotion can be re-run."""
    from club_soccer import model as M
    seeded = M.fit()
    legacy = M.fit(league_seed=False)
    assert seeded["elo"] != legacy["elo"]
    assert seeded["league_seed_active"] is True
    assert legacy["league_seed_active"] is False


def test_params_record_which_model_produced_them():
    from club_soccer import model as M
    params = M.load_params()
    assert params.get("league_seed_active") is True, \
        "stored production params must be the promoted model"


def test_primary_league_map_ignores_cups_and_europe():
    from club_soccer import model as M
    df = pd.DataFrame([
        {"competition": "Austrian Bundesliga", "home": "Sturm Graz", "away": "Salzburg"},
        {"competition": "Austrian Bundesliga", "home": "Salzburg", "away": "Sturm Graz"},
        {"competition": "Champions League", "home": "Sturm Graz", "away": "Bayern Munich"},
        {"competition": "Champions League", "home": "Sturm Graz", "away": "Inter"},
        {"competition": "Champions League", "home": "Sturm Graz", "away": "Arsenal"},
    ])
    m = M._primary_league_map(df)
    assert m["Sturm Graz"] == "Austrian Bundesliga", \
        "European matches must not become a club's primary league"


def test_seeding_lifts_a_strong_league_above_a_weak_one():
    """The point of the change: a new club should not start life rated as the
    average of a pool that spans the Premier League and the National League."""
    from club_soccer import model as M
    from club_soccer.uefa_registry import strength_prior
    top = strength_prior("England", 1)
    weak = strength_prior("England", 5)
    seed_top = M.BASE_ELO + (top - M.LEAGUE_SEED_ANCHOR) * M.ELO_PER_STRENGTH
    seed_weak = M.BASE_ELO + (weak - M.LEAGUE_SEED_ANCHOR) * M.ELO_PER_STRENGTH
    assert seed_top > seed_weak


def test_walk_forward_cache_key_includes_league_seed():
    """A fit option missing from the cache key means the cache serves results
    produced under different settings — a silent wrong answer."""
    import inspect

    from club_soccer import validate as V
    src = inspect.getsource(V.walk_forward)
    assert '"league_seed": league_seed' in src
