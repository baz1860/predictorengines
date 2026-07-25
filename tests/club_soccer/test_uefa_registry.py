#!/usr/bin/env python3
"""Tests for the P2 UEFA association registry and expanded competition table."""
from __future__ import annotations

import collections
import json

import pytest

from club_soccer import uefa_registry as R
from club_soccer.competitions import (BSD_LEAGUE_ALIASES, COMPETITIONS,
                                      comp_from_bsd_league, get, strength)


# ── coefficients ──────────────────────────────────────────────────────────

def test_all_55_associations_present():
    assert len(R.ASSOCIATIONS) == 55
    assert len({a.country for a in R.ASSOCIATIONS}) == 55


def test_ranks_are_a_complete_1_to_55_sequence():
    assert sorted(a.rank for a in R.ASSOCIATIONS) == list(range(1, 56))


def test_coefficients_are_monotonic_in_rank():
    ordered = sorted(R.ASSOCIATIONS, key=lambda a: a.rank)
    coefs = [a.coefficient for a in ordered]
    assert coefs == sorted(coefs, reverse=True)


def test_coefficient_artifact_records_provenance():
    doc = json.loads(R.COEFFICIENTS.read_text())
    for key in ("source", "source_url", "fetched_at_utc", "refresh_cadence"):
        assert doc.get(key), f"coefficient artifact must record {key}"
    assert len(doc["associations"]) == 55


def test_top_ten_ordering_matches_uefa_published():
    """Cross-check against uefa.com's published 2025/26 ordering."""
    expected = ["England", "Italy", "Spain", "Germany", "France",
                "Netherlands", "Portugal", "Belgium", "Turkey", "Czech Republic"]
    got = [a.country for a in sorted(R.ASSOCIATIONS, key=lambda a: a.rank)[:10]]
    assert got == expected


# ── strength priors ───────────────────────────────────────────────────────

def test_strength_prior_is_anchored_on_the_existing_scale():
    """The anchors must reproduce the hand-set values they were fitted to."""
    assert R.strength_prior("England", 1) == pytest.approx(1.00, abs=0.01)
    assert R.strength_prior("Scotland", 1) == pytest.approx(0.58, abs=0.01)


def test_strength_prior_orders_leagues_sensibly():
    top = [R.strength_prior(c, 1) for c in
           ("England", "Italy", "Netherlands", "Austria", "San Marino")]
    assert top == sorted(top, reverse=True)
    # The failure this whole project started from: Austria must not be
    # indistinguishable from a fourth tier.
    assert R.strength_prior("Austria", 1) > R.strength_prior("England", 4)


def test_lower_tiers_are_weaker_than_their_top_tier():
    for country in ("England", "Italy", "Spain", "Germany", "France"):
        assert R.strength_prior(country, 2) < R.strength_prior(country, 1)


def test_unknown_country_falls_back_without_raising():
    assert R.strength_prior("Atlantis") == 0.75


# ── source coverage honesty ───────────────────────────────────────────────

def test_registry_does_not_claim_sources_it_lacks():
    """Associations with no source must expose no ingestible leagues."""
    for a in R.missing_associations():
        assert a.tiers_available == 0
        assert a.note, f"{a.country} must document why it is unavailable"


def test_austria_is_available_and_not_from_bsd():
    """Austria is the fixture that started this. BSD does not carry it."""
    austria = R.BY_COUNTRY["Austria"]
    assert austria.available
    src = austria.leagues[0]
    assert src.source == R.SRC_FDCOUK_NEW
    assert src.code == "AUT"


def test_known_gaps_are_recorded_not_silently_dropped():
    missing = {a.country for a in R.missing_associations()}
    for country in ("Czech Republic", "Israel", "Ukraine", "Serbia", "Croatia"):
        assert country in missing, f"{country} gap must be tracked"


def test_coverage_summary_is_self_consistent():
    s = R.coverage_summary()
    assert s["associations_available"] + s["associations_missing"] == 55
    assert s["divisions_available"] == len(R.planned_leagues())


# ── competition table ─────────────────────────────────────────────────────

def test_no_duplicate_competition_names_or_ids():
    names = collections.Counter(c.name for c in COMPETITIONS)
    ids = collections.Counter(c.api_id for c in COMPETITIONS)
    assert [n for n, v in names.items() if v > 1] == []
    assert [i for i, v in ids.items() if v > 1] == []


def test_new_competitions_carry_a_usable_source():
    for comp in COMPETITIONS:
        if comp.api_id < 9000:
            continue
        has_source = bool(comp.fdcouk_code or comp.fdcouk_new or comp.bsd_league)
        assert has_source, f"{comp.name} has no ingest source"


def test_expansion_leagues_are_domestic_not_european():
    """coverage.py keys 'has_domestic' off kind != 'europe'.

    Scoped to the UEFA expansion block (9000-9099). The 9100+ range holds the
    non-UEFA competitions, which legitimately include continental ones
    (Copa Libertadores, CAF Champions League) whose kind IS 'europe' — that
    field means "continental", not "in Europe".
    """
    for comp in COMPETITIONS:
        if 9000 <= comp.api_id < 9100:
            assert comp.kind == "league"


def test_non_uefa_continental_competitions_are_marked_continental():
    """Libertadores links the South American leagues to each other the way
    UEFA competition links the European ones, so it must not be treated as a
    club's domestic evidence."""
    for name in ("Copa Libertadores", "Copa Sudamericana", "CAF Champions League"):
        comp = get(name)
        if comp:
            assert comp.kind == "europe"


def test_austrian_bundesliga_registered_with_country():
    comp = get("Austrian Bundesliga")
    assert comp is not None
    assert comp.country == "Austria"
    assert comp.fdcouk_new == "AUT"
    assert comp.fdcouk_league == "Bundesliga"


# ── the collision guard (this is the dangerous part) ──────────────────────

def test_bsd_resolution_is_exact_match_only():
    """A substring rule would merge continents. Pin the exact-match contract.

    UPDATED: these five were previously asserted to resolve to None, because
    they were unsupported and a substring rule would have mis-matched them
    onto England's Championship, Italy's Serie A, the Champions League and the
    Swiss Super League. They are now deliberately registered — which makes
    exact matching load-bearing rather than merely cautious. The contract is
    therefore no longer "these do not resolve" but "these resolve to their OWN
    competition, on their own continent".
    """
    cases = {"USL Championship": "USA", "Brasileirão Serie A": "Brazil",
             "CAF Champions League": "Africa", "Chinese Super League": "China",
             "Categoría Primera A": "Colombia"}
    for name, country in cases.items():
        comp = comp_from_bsd_league(name)
        assert comp is not None, f"{name} should now resolve"
        assert comp.country == country
    # Genuinely unsupported names must still resolve to nothing.
    for name in ("Copa do Brasil", "NPL Queensland", "Liga F", "NWSL"):
        assert comp_from_bsd_league(name) is None


def test_ambiguous_bsd_names_resolve_to_the_right_country():
    """'Super League' and 'Superliga' are the two genuine trap names."""
    assert comp_from_bsd_league("Super League").country == "Switzerland"
    assert comp_from_bsd_league("Stoiximan Super League").country == "Greece"
    assert comp_from_bsd_league("Superliga").country == "Romania"
    assert comp_from_bsd_league("Liga Portugal Betclic").country == "Portugal"


def test_national_team_competitions_do_not_resolve():
    for name in ("UEFA Euro 2024", "UEFA Nations League",
                 "World Cup Qualification UEFA", "UEFA European U19 Championship"):
        assert comp_from_bsd_league(name) is None


def test_every_alias_target_exists():
    for src, target in BSD_LEAGUE_ALIASES.items():
        assert get(target) is not None, f"alias {src!r} points at unknown {target!r}"


def test_strength_lookup_works_for_new_leagues():
    for name in ("Eredivisie", "Austrian Bundesliga", "Super Lig", "Parva Liga"):
        s = strength(name)
        assert 0.1 < s < 1.1, f"{name} strength {s} out of range"
        # Must not silently fall through to the 0.75 unknown-competition default.
        assert s != 0.75


# ── point-in-time snapshot dating (finding 13) ─────────────────────────────

def test_no_prior_before_the_first_snapshot():
    """A date before the earliest snapshot has no point-in-time coefficient, so
    the prior must be the DEFAULT, never a future (leaked) snapshot."""
    assert R.strength_prior("England", as_of="2000-01-01") == R.DEFAULT_PRIOR


def test_as_of_selects_the_period_correct_snapshot():
    """A within-range date uses the latest snapshot published on or before it,
    and yields the informative (non-default) anchored prior."""
    assert R.strength_prior("England", as_of="2023-01-01") != R.DEFAULT_PRIOR


def test_degenerate_anchor_separation_falls_back_to_default(monkeypatch):
    """If the two anchors are not meaningfully separated the linear rescale is
    unstable, so the prior must fall back rather than clip to junk."""
    monkeypatch.setattr(R, "_snapshot_for",
                        lambda as_of: {"England": 30.0, "Scotland": 29.99,
                                       "Ruritania": 29.98})
    assert R.strength_prior("Ruritania") == R.DEFAULT_PRIOR


# ── association-name normalization (finding 9) ─────────────────────────────

def test_association_name_variants_resolve_to_a_real_prior():
    """History spells them 'Türkiye'/'Czechia'; the registry uses
    'Turkey'/'Czech Republic'. Both must find the coefficient, not fall back to
    the flat default that caused the Turkish clubs' spurious Elo jump."""
    assert R.strength_prior("Turkey") != R.DEFAULT_PRIOR
    assert R.strength_prior("Turkey") == R.strength_prior("Türkiye")
    assert R.strength_prior("Czech Republic") != R.DEFAULT_PRIOR
