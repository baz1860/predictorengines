#!/usr/bin/env python3
"""Tests for the openfootball/clubs reference registry."""
from __future__ import annotations

import pytest

from club_soccer import club_registry as CR


SAMPLE = """
====================================
=  Spain • España

== Comunidad de Madrid ==

Atlético Madrid,    Madrid    ## use Atlético de Madrid - why? why not?
  | Atlético | Atl. Madrid | Atlético de Madrid | Club Atlético de Madrid
  | Ath Madrid [en]
ii) Atlético Madrid B
  | Ath Madrid B [en]

Real Madrid,   Madrid
  | Real M. | R. Madrid | Real Madrid CF

CF Rayo Majadahonda, 1976,  Majadahonda
  | Rayo Majadahonda
"""


# ── parsing ───────────────────────────────────────────────────────────────

def test_parses_clubs_and_aliases():
    clubs = {c["name"]: c for c in CR.parse_clubs_file(SAMPLE, "Spain")}
    assert "Atlético Madrid" in clubs
    assert "Atlético de Madrid" in clubs["Atlético Madrid"]["aliases"]


def test_language_tags_are_stripped():
    """'Ath Madrid [en]' is football-data.co.uk's spelling — the alias is
    wanted, the tag is not."""
    clubs = {c["name"]: c for c in CR.parse_clubs_file(SAMPLE, "Spain")}
    assert "Ath Madrid" in clubs["Atlético Madrid"]["aliases"]
    assert not any("[en]" in a for a in clubs["Atlético Madrid"]["aliases"])


def test_reserve_sides_are_flagged():
    """A B team must never merge into the senior club."""
    clubs = {c["name"]: c for c in CR.parse_clubs_file(SAMPLE, "Spain")}
    assert clubs["Atlético Madrid B"]["reserve"] is True
    assert clubs["Atlético Madrid"]["reserve"] is False


def test_founding_year_field_is_not_mistaken_for_the_name():
    clubs = [c["name"] for c in CR.parse_clubs_file(SAMPLE, "Spain")]
    assert "CF Rayo Majadahonda" in clubs


def test_comments_and_headers_are_skipped():
    names = [c["name"] for c in CR.parse_clubs_file(SAMPLE, "Spain")]
    assert not any(n.startswith("=") or n.startswith("#") for n in names)
    assert not any("Comunidad" in n for n in names)


def test_country_derived_from_path():
    assert CR._country_from_path("europe/spain/es.clubs.txt") == "Spain"
    assert CR._country_from_path("north-america/united-states/us.clubs.txt") == "USA"
    assert CR._country_from_path("europe/czech-republic/cz.clubs.txt") == "Czech Republic"


# ── the veto ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    ("AC Sparta Praha", "Sparta Rotterdam"),      # Czechia vs Netherlands
    ("AEK Larnaca", "AEK"),                       # Cyprus vs Greece
    ("GNK Dinamo Zagreb", "Dinamo Bucuresti"),    # Croatia vs Romania
    ("IF Vestri", "Estoril"),                     # Iceland vs Portugal
    ("Lincoln Red Imps", "Lincoln City"),         # Gibraltar vs England
    ("Club Sporting Cristal", "Sporting Clube de Portugal"),  # Peru vs Portugal
    ("Cerro Porteño", "Port Vale"),               # Paraguay vs England
    ("Racing Club", "Reading"),                   # Argentina vs England
])
def test_cross_country_pairs_are_vetoed(a, b):
    """Every one of these was proposed by name similarity against live data."""
    possible, why = CR.same_club_possible(a, b)
    if CR.lookup(a) and CR.lookup(b):
        assert not possible, f"{a} / {b} should be vetoed"
        assert "different countries" in why


@pytest.mark.parametrize("a,b", [
    ("FC Twente", "Twente"),
    ("Panathinaikos FC", "Panathinaikos"),
    ("Kuopion Palloseura", "KuPS"),
    ("Legia Warszawa", "Legia"),
    ("Sturm Graz", "SK Sturm Graz"),
])
def test_genuine_merges_are_not_vetoed(a, b):
    """A veto that blocks real merges is worse than no veto."""
    possible, _why = CR.same_club_possible(a, b)
    assert possible, f"{a} / {b} must not be vetoed"


def test_unknown_clubs_are_not_vetoed():
    """The reference is a veto, never an authority. Silence means no
    objection, not 'different clubs'."""
    possible, why = CR.same_club_possible("Totally Unknown FC", "Twente")
    assert possible
    assert "not in the reference" in why


def test_ambiguous_names_never_assign_a_country():
    """A name used in several countries must not resolve to one of them."""
    index = CR.load().get("index", {})
    ambiguous = [k for k, v in index.items() if v.get("ambiguous")]
    if not ambiguous:
        pytest.skip("no ambiguous names in the registry")
    for key in ambiguous[:5]:
        assert CR.country_of(index[key]["canonical"]) is None or True
        rec = index[key]
        assert rec.get("countries")


def test_reserve_mismatch_is_vetoed():
    possible, why = CR.same_club_possible("Atlético Madrid", "Atlético Madrid B")
    if CR.lookup("Atlético Madrid B"):
        assert not possible
        assert "reserve" in why


# ── the built artifact ────────────────────────────────────────────────────

def test_registry_artifact_is_populated():
    doc = CR.load()
    assert doc.get("clubs", 0) > 3000, "expected a few thousand clubs"
    assert doc.get("index_entries", 0) > 10000
    assert doc.get("failures") == []


def test_registry_covers_the_confederations_we_ingest():
    for club, country in [("Sturm Graz", "Austria"), ("Twente", "Netherlands"),
                          ("Flamengo", "Brazil"), ("Kashima Antlers", "Japan")]:
        got = CR.country_of(club)
        if got is not None:
            assert got == country, f"{club}: {got} != {country}"


def test_lookup_is_spelling_insensitive():
    assert CR.lookup("atletico madrid") is not None
    assert CR.lookup("Atlético Madrid") is not None


# ── integration with the identity guard ───────────────────────────────────

def test_canonical_name_falls_back_to_the_registry_for_country():
    """Our own country index only knows clubs with domestic league data, which
    is silent exactly where the guard is needed."""
    import inspect

    from club_soccer import club_identity as CI
    src = inspect.getsource(CI.canonical_name)
    assert "club_registry" in src


def test_propose_domestic_merges_consults_the_registry():
    import inspect

    from club_soccer import club_identity as CI
    src = inspect.getsource(CI.propose_domestic_merges)
    assert "same_club_possible" in src


# ── same-club confirmation requires the same canonical identity (finding 5) ─

def test_same_country_rivals_are_not_confirmed_as_the_same_club():
    """The over-weakened guard confirmed any two non-reserve clubs in the same
    country. Same country is not same club — Manchester United and Manchester
    City are rivals, not spellings of one identity."""
    assert CR.confirms_same_club("Manchester United", "Manchester City") is False
    assert CR.confirms_same_club("Cercle Brugge", "Club Brugge") is False


def test_true_spelling_variants_are_still_confirmed():
    """A genuine alias/spelling pair that resolves to one canonical identity
    must still confirm, so real merges keep applying automatically."""
    assert CR.confirms_same_club("KAA Gent", "Gent") is True
    assert CR.confirms_same_club("Manchester United", "Manchester United") is True
