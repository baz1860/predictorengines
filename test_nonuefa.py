#!/usr/bin/env python3
"""Tests for non-UEFA league support and the guards it required."""
from __future__ import annotations

import collections

import pandas as pd
import pytest

from club_soccer import club_identity as CI
from club_soccer import fetch as F
from club_soccer.competitions import COMPETITIONS, comp_from_bsd_league, get


# ── country-scoped identity ───────────────────────────────────────────────

def test_cross_confederation_collision_is_refused():
    """The demonstrated case: Brazil's 'Athletic Club' (Athletic Club de São
    João del-Rei) otherwise resolves onto Spain's Athletic Bilbao, welding a
    Brazilian club's results onto Athletic Bilbao's rating history."""
    CI.reset_country_index()
    assert CI.canonical_name("Athletic Club", country="Brazil") == "Athletic Club"
    assert CI.canonical_name("Athletic Club", country="Spain") == "Athletic Bilbao"


def test_unscoped_calls_keep_the_old_behaviour():
    CI.reset_country_index()
    assert CI.canonical_name("FC Bayern München") == "Bayern Munich"
    assert CI.canonical_name("Heart of Midlothian") == "Hearts"


def test_matching_country_still_maps():
    CI.reset_country_index()
    assert CI.canonical_name("FC Bayern München", country="Germany") == "Bayern Munich"


def test_unknown_target_country_never_blocks():
    """Continental-only clubs have no domestic country; they must not be
    blocked by a guard that has nothing to compare against."""
    CI.reset_country_index()
    out = CI.canonical_name("Totally Unknown SK", country="Japan")
    assert out == "Totally Unknown SK"


def test_country_index_is_built_from_domestic_leagues():
    CI.reset_country_index()
    index = CI.team_countries()
    assert index.get("Athletic Bilbao") == "Spain"
    assert index.get("Bayern Munich") == "Germany"


# ── registry ──────────────────────────────────────────────────────────────

def test_non_uefa_competitions_registered():
    for name in ("MLS", "J1 League", "K League 1", "Brasileirao Serie A",
                 "Saudi Pro League", "Chinese Super League",
                 "Liga MX Apertura", "Liga MX Clausura"):
        assert get(name) is not None, f"{name} missing from the registry"


def test_no_duplicate_ids_or_names():
    ids = collections.Counter(c.api_id for c in COMPETITIONS)
    names = collections.Counter(c.name for c in COMPETITIONS)
    assert [i for i, v in ids.items() if v > 1] == []
    assert [n for n, v in names.items() if v > 1] == []


def test_calendar_season_set_for_calendar_year_leagues():
    for name in ("MLS", "J1 League", "K League 1", "Brasileirao Serie A",
                 "Chinese Super League"):
        assert get(name).calendar_season is True, f"{name} runs Mar-Nov"


def test_mls_has_no_relegation():
    """MLS uses conferences and playoffs. 0 is correct, not unknown —
    motivation bands built on a relegation battle do not apply."""
    assert get("MLS").releg_spots == 0


def test_non_uefa_leagues_claim_no_european_places():
    for c in COMPETITIONS:
        if c.api_id >= 9100 and c.kind == "league":
            assert c.euro_spots == 0


# ── the collision guard that made this possible ───────────────────────────

def test_historic_collision_names_resolve_to_the_right_confederation():
    """These are the exact names the original substring rule collided on.
    Registering their real counterparts makes exact matching load-bearing."""
    cases = [("Championship", "England"), ("USL Championship", "USA"),
             ("Serie A", "Italy"), ("Brasileirão Serie A", "Brazil"),
             ("Super League", "Switzerland"), ("Chinese Super League", "China"),
             ("Champions League", "Europe"), ("CAF Champions League", "Africa"),
             ("Copa Libertadores", "South America")]
    for bsd_name, expected_country in cases:
        comp = comp_from_bsd_league(bsd_name)
        assert comp is not None, f"{bsd_name} no longer resolves"
        assert comp.country == expected_country, \
            f"{bsd_name} -> {comp.name} ({comp.country}), expected {expected_country}"


def test_national_team_competitions_still_do_not_resolve():
    for name in ("World Cup 2026", "Copa America 2024", "UEFA Nations League",
                 "AFC Asian Cup 2023", "World Cup Qualification CONMEBOL"):
        assert comp_from_bsd_league(name) is None


# ── the shrink guard ──────────────────────────────────────────────────────

def test_write_fixtures_refuses_a_catastrophic_shrink(tmp_path):
    """fetch_fixtures(current=False) reaches write_fixtures with only the
    fetched slice and replaced a 55,000-row file with 2,704 rows. The write
    was atomic — it atomically wrote the WRONG file. Nothing downstream would
    have noticed: the next refit would train on 5% of the data and report a
    plausible Brier."""
    path = tmp_path / "fixtures.csv"
    big = pd.DataFrame({"fixture_id": range(1000), "home": ["A"] * 1000,
                        "away": ["B"] * 1000, "status": ["FT"] * 1000})
    F.write_fixtures(big, path)
    with pytest.raises(ValueError, match="refusing to shrink"):
        F.write_fixtures(big.head(50), path)


def test_shrink_guard_allows_legitimate_dedupe(tmp_path):
    """The largest legitimate removal in this project was ~13% (the P1
    duplicate-row merge). The guard must not block that."""
    path = tmp_path / "fixtures.csv"
    big = pd.DataFrame({"fixture_id": range(1000), "home": ["A"] * 1000,
                        "away": ["B"] * 1000, "status": ["FT"] * 1000})
    F.write_fixtures(big, path)
    out = F.write_fixtures(big.head(870), path)
    assert len(out) == 870


def test_shrink_guard_can_be_overridden_deliberately(tmp_path, monkeypatch):
    path = tmp_path / "fixtures.csv"
    big = pd.DataFrame({"fixture_id": range(1000), "home": ["A"] * 1000,
                        "away": ["B"] * 1000, "status": ["FT"] * 1000})
    F.write_fixtures(big, path)
    monkeypatch.setenv("CLUB_SOCCER_ALLOW_FIXTURE_SHRINK", "1")
    out = F.write_fixtures(big.head(10), path)
    assert len(out) == 10


def test_small_files_are_not_guarded(tmp_path):
    """A fresh seed starts from nothing; the guard must not block bootstrap."""
    path = tmp_path / "fixtures.csv"
    small = pd.DataFrame({"fixture_id": range(20), "home": ["A"] * 20,
                          "away": ["B"] * 20, "status": ["FT"] * 20})
    F.write_fixtures(small, path)
    F.write_fixtures(small.head(2), path)      # must not raise


# ── self-match guard ──────────────────────────────────────────────────────

def test_self_matches_are_dropped_at_the_write_boundary(tmp_path):
    """BSD emits corrupt home == away rows ('Samsunspor v Samsunspor',
    'Stade Rennais v Stade Rennais'). A team cannot play itself, and such a
    row would train the model on a fabricated result, so it must never survive
    a write regardless of which ingest path produced it."""
    path = tmp_path / "fixtures.csv"
    df = pd.DataFrame({
        "fixture_id": [1, 2, 3],
        "home": ["Samsunspor", "Reims", "Arsenal"],
        "away": ["Samsunspor", "Rennes", "Chelsea"],
        "status": ["FT", "FT", "FT"],
    })
    out = F.write_fixtures(df, path)
    assert len(out) == 2
    assert not (out["home"] == out["away"]).any()
    assert "Reims" in set(out["home"])       # the legitimate row survives


# ── league-label reclassification (BSD name collisions) ───────────────────

def test_danish_clubs_in_romanian_superliga_are_reclassified():
    """The reclassifier moves a label to the AGREED country's league. The
    caller supplies the country only when both clubs are known and agree."""
    from club_soccer.competitions import get
    ro = get("Romanian Superliga")
    assert F.reclassify_by_club_country(ro, "Denmark").name == "Danish Superliga"


def test_agreed_country_matching_the_label_is_a_no_op():
    from club_soccer.competitions import get
    ro = get("Romanian Superliga")
    assert F.reclassify_by_club_country(ro, "Romania").name == "Romanian Superliga"


def test_reclassify_needs_a_registered_league_for_the_country():
    from club_soccer.competitions import get
    ro = get("Romanian Superliga")
    # A country with no registered tier-1 league leaves the label untouched.
    assert F.reclassify_by_club_country(ro, "Andorra").name == "Romanian Superliga"


def test_continental_competitions_are_never_reclassified():
    from club_soccer.competitions import get
    cl = get("Champions League")
    assert F.reclassify_by_club_country(cl, "Denmark").name == "Champions League"


def test_ambiguous_single_club_never_reclassifies_the_row():
    """Regression for fixture 10034: a Brazilian 'Athletic Club' (openfootball
    lists only the Spanish one) vs a Brazilian opponent must stay in the
    Brazilian league and must NOT become Athletic Bilbao / La Liga.

    Exercises the row builder, where the both-clubs-agree guard lives."""
    from club_soccer.club_registry import country_of
    # Only meaningful if the registry indeed mis-places raw "Athletic Club".
    ac_country = country_of("Athletic Club")
    ev = {"home_team": "Athletic Club", "away_team": "São Bernardo",
          "id": 999001, "status": "notstarted", "event_date": "2026-08-01T20:00:00Z"}
    row = F._bsd_to_fixture_row(ev, "Brasileirao Serie B", 9106, "Brazil", "league")
    assert row is not None
    assert row["competition"] == "Brasileirao Serie B", "must not leave Brazil"
    assert row["home"] != "Athletic Bilbao", \
        "an ambiguous single club must not be welded onto a Spanish identity"


def test_live_romanian_superliga_has_no_foreign_clubs():
    from club_soccer.club_registry import country_of
    df = pd.read_csv(CI.FIXTURES, low_memory=False)
    ro = df[df["competition"] == "Romanian Superliga"]
    if ro.empty:
        pytest.skip("Romanian Superliga not present")
    teams = set(ro["home"].dropna()) | set(ro["away"].dropna())
    foreign = {t: country_of(t) for t in teams
               if country_of(t) and country_of(t) != "Romania"}
    assert not foreign, f"foreign clubs still in Romanian Superliga: {foreign}"


# ── live artifact ─────────────────────────────────────────────────────────

def test_athletic_bilbao_is_uncontaminated():
    df = pd.read_csv(CI.FIXTURES, low_memory=False)
    rows = df[(df["home"] == "Athletic Bilbao") | (df["away"] == "Athletic Bilbao")]
    if rows.empty:
        pytest.skip("Athletic Bilbao not in fixtures")
    comps = set(rows["competition"])
    assert not any("Brasileirao" in c for c in comps), \
        "a Brazilian club has been merged into Athletic Bilbao"
