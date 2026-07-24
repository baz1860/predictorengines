#!/usr/bin/env python3
"""Tests for P3 — expansion-league ingest and cross-source reconciliation.

The reconciliation guards carry the risk here. A wrong merge silently welds
two clubs' histories together and there is no error to notice afterwards, so
each guard below pins a false positive that was actually proposed against live
data during the build.
"""
from __future__ import annotations

import pandas as pd
import pytest

from club_soccer import club_identity as CI
from club_soccer import seed_fdcouk_leagues as S
from club_soccer.competitions import get as comp_get


def _fx(rows):
    return pd.DataFrame(
        [{"date": d, "season": s, "competition": c, "home": h, "away": a,
          "home_goals": 1.0, "away_goals": 0.0} for d, s, c, h, a in rows])


def _euro(team, opponent, date="2024-09-18", season=2024):
    return (date, season, "Champions League", team, opponent)


def _domestic(team, opponent, comp, date="2024-09-21", season=2024):
    return (date, season, comp, team, opponent)


# ── wave configuration ────────────────────────────────────────────────────

def test_every_wave_references_real_competitions():
    for wave in S.WAVES:
        for comp in S.wave_competitions(wave):
            assert comp is not None


def test_waves_do_not_overlap():
    seen = set()
    for wave in sorted(S.WAVES):
        for name in S.WAVES[wave]:
            assert name not in seen, f"{name} appears in more than one wave"
            seen.add(name)


def test_every_waved_competition_has_a_source():
    for wave in S.WAVES:
        for comp in S.wave_competitions(wave):
            assert comp.fdcouk_code or comp.fdcouk_new, \
                f"{comp.name} has no fd.co.uk source"


def test_fixture_ids_are_deterministic_and_distinct():
    a = S._fid("Eredivisie", "2024-09-21", "Ajax", "PSV")
    b = S._fid("Eredivisie", "2024-09-21", "Ajax", "PSV")
    c = S._fid("Eredivisie", "2024-09-21", "PSV", "Ajax")
    assert a == b, "re-ingest must update rows, not duplicate them"
    assert a != c


# ── cross-source reconciliation ───────────────────────────────────────────

def test_europe_only_teams_excludes_clubs_with_domestic_data():
    df = _fx([_euro("SK Sturm Graz", "Bayern Munich"),
              _domestic("Bayern Munich", "Dortmund", "Bundesliga")])
    euro = CI.europe_only_teams(df)
    assert "SK Sturm Graz" in euro
    assert "Bayern Munich" not in euro


def test_the_motivating_case_sturm_graz_is_reconciled():
    existing = _fx([_euro("SK Sturm Graz", "Bayern Munich"),
                    _domestic("Bayern Munich", "Dortmund", "Bundesliga")])
    incoming = _fx([_domestic("Sturm Graz", "Salzburg", "Austrian Bundesliga")])
    res = CI.propose_domestic_merges(existing, incoming)
    assert res["alias"].get("SK Sturm Graz") == "Sturm Graz"


def test_unconfirmed_merge_with_a_collision_goes_to_review_not_auto_merge():
    """Fail-closed contract (restored after an adversarial review).

    A same-day collision was briefly treated as a mere signal, which let
    "CSKA Sofia" / "FC CSKA 1948 Sofia" auto-merge — identical core after the
    founding year is stripped, and the registry cannot object to a club it does
    not list. Now: auto-merge requires POSITIVE registry confirmation, and an
    unconfirmed pair — with or without a collision — routes to review rather
    than merging silently. Two invented clubs the registry cannot confirm must
    therefore NOT auto-merge."""
    existing = _fx([_euro("AC Sparta Testville", "Bayern Munich", date="2024-09-18"),
                    _domestic("Bayern Munich", "Dortmund", "Bundesliga")])
    incoming = _fx([_domestic("Sparta Testville", "Ajax", "Eredivisie",
                              date="2024-09-18")])
    res = CI.propose_domestic_merges(existing, incoming)
    assert "AC Sparta Testville" not in res["alias"], \
        "an unconfirmed pair must not auto-merge"
    assert any(r["existing"] == "AC Sparta Testville" for r in res["review"]), \
        "it must be routed to human review instead"


def test_cross_country_false_positives_are_still_rejected():
    """Removing the date veto must not let the real false positives through —
    they are caught by the affinity floor and the blocklist."""
    existing = _fx([_euro("AC Sparta Praha", "Bayern Munich"),
                    _domestic("Bayern Munich", "Dortmund", "Bundesliga")])
    incoming = _fx([_domestic("Sparta Rotterdam", "Ajax", "Eredivisie")])
    res = CI.propose_domestic_merges(existing, incoming)
    assert "AC Sparta Praha" not in res["alias"]


def test_season_overlap_is_required():
    existing = _fx([_euro("Old Club FC", "Bayern Munich", date="2021-09-18",
                          season=2021),
                    _domestic("Bayern Munich", "Dortmund", "Bundesliga")])
    incoming = _fx([_domestic("Old Club", "Rival", "Eredivisie",
                              date="2025-09-21", season=2025)])
    res = CI.propose_domestic_merges(existing, incoming)
    assert "Old Club FC" not in res["alias"]
    assert any("season overlap" in r["reason"] for r in res["rejected"])


def test_club_already_present_in_incoming_is_not_absorbed_by_a_neighbour():
    """Cercle Brugge must not be swallowed by Club Brugge."""
    existing = _fx([_euro("Cercle Brugge", "Bayern Munich"),
                    _domestic("Bayern Munich", "Dortmund", "Bundesliga")])
    incoming = _fx([_domestic("Club Brugge", "Anderlecht", "Belgian Pro League"),
                    _domestic("Cercle Brugge", "Genk", "Belgian Pro League")])
    res = CI.propose_domestic_merges(existing, incoming)
    assert "Cercle Brugge" not in res["alias"]


def test_mid_confidence_pairs_go_to_review_not_applied():
    """Between MIN_SCORE and AUTO_SCORE nothing is applied automatically."""
    existing = _fx([_euro("Dinamo Someplace", "Bayern Munich"),
                    _domestic("Bayern Munich", "Dortmund", "Bundesliga")])
    incoming = _fx([_domestic("Dinamo Elsewhere", "Rival", "Romanian Superliga")])
    res = CI.propose_domestic_merges(existing, incoming)
    assert "Dinamo Someplace" not in res["alias"]


def test_blocklisted_pairs_are_never_reproposed():
    for euro_name, new_name in CI.DOMESTIC_MERGE_BLOCKLIST:
        assert CI.DOMESTIC_MANUAL_MERGES.get(euro_name) != new_name, \
            f"{euro_name} is both blocklisted and manually merged to {new_name}"


def test_manual_merges_and_blocklist_do_not_contradict():
    for euro_name, new_name in CI.DOMESTIC_MANUAL_MERGES.items():
        assert (euro_name, new_name) not in CI.DOMESTIC_MERGE_BLOCKLIST


def test_reconciliation_is_one_to_one():
    """Two Europe-only spellings may collapse onto one club, but a single
    domestic identity must not absorb two genuinely different clubs."""
    existing = _fx([_euro("Olympiacos FC", "Bayern Munich"),
                    _euro("Olympiakos Piraeus", "Bayern Munich", date="2024-10-02"),
                    _domestic("Bayern Munich", "Dortmund", "Bundesliga")])
    incoming = _fx([_domestic("Olympiakos", "PAOK", "Greek Super League")])
    res = CI.propose_domestic_merges(existing, incoming)
    assert set(res["alias"].values()) <= {"Olympiakos"}


# ── cumulative alias map ──────────────────────────────────────────────────

def test_alias_map_on_disk_is_cumulative():
    """Regression: rebuilding must not shrink the map.

    build_alias_map only proposes merges it can still see evidence for, and an
    applied merge erases its own evidence. A non-cumulative write would drop
    the P1 entries, and canonical_name() reads this file on every daily fetch —
    so the duplication would quietly return.
    """
    import json
    doc = json.loads(CI.ALIAS_MAP.read_text())
    alias = doc["alias"]
    for src in ("FC Bayern München", "Heart of Midlothian", "FC Internazionale Milano"):
        assert src in alias, f"P1 entry {src!r} lost from the alias map"
    assert len(alias) >= 93


def test_alias_map_has_no_chains():
    import json
    alias = json.loads(CI.ALIAS_MAP.read_text())["alias"]
    for src, dst in alias.items():
        assert dst not in alias, f"chain {src} -> {dst} -> {alias.get(dst)}"


# ── the live artifact ─────────────────────────────────────────────────────

def test_expansion_leagues_are_present_in_fixtures():
    df = pd.read_csv(CI.FIXTURES, low_memory=False)
    comps = set(df["competition"].dropna())
    for name in ("Eredivisie", "Austrian Bundesliga", "Liga Portugal",
                 "Super Lig", "Serie B", "Ekstraklasa"):
        assert name in comps, f"{name} missing from fixtures.csv"


def test_sturm_graz_now_has_domestic_data():
    """The fixture that started this: a single identity, domestically rated."""
    df = pd.read_csv(CI.FIXTURES, low_memory=False)
    names = set(df["home"].dropna()) | set(df["away"].dropna())
    assert "Sturm Graz" in names
    assert "SK Sturm Graz" not in names, "old Europe-only identity must be merged away"
    rows = df[(df["home"] == "Sturm Graz") | (df["away"] == "Sturm Graz")]
    assert set(rows["competition"]) & {"Austrian Bundesliga"}


def test_no_self_matches_in_fixtures():
    df = pd.read_csv(CI.FIXTURES, low_memory=False)
    assert int((df["home"] == df["away"]).sum()) == 0


def test_no_duplicate_match_keys():
    df = pd.read_csv(CI.FIXTURES, low_memory=False)
    key = (df["date"].astype(str).str[:10] + "|" + df["competition"].astype(str)
           + "|" + df["home"].astype(str) + "|" + df["away"].astype(str))
    assert int(key.duplicated().sum()) == 0


def test_all_seed_writers_use_the_canonical_season_helper():
    """Finding 12: season derivation must live in one place. Every seed writer
    must call fetch.season_for_date rather than inlining the July-boundary rule
    (which silently mislabels calendar-year leagues)."""
    import inspect

    from club_soccer import seed_real, seed_openfootball, seed_fdcouk_leagues
    for mod in (seed_real, seed_openfootball, seed_fdcouk_leagues):
        src = inspect.getsource(mod)
        assert "season_for_date" in src, f"{mod.__name__} must use season_for_date"
        # No inline 'month >= 7 else ... - 1' season rule outside the helper.
        assert "if month >= 7 else year - 1" not in src, \
            f"{mod.__name__} still inlines the season rule"
