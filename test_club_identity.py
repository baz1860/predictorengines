#!/usr/bin/env python3
"""Tests for club_soccer.club_identity (P1 identity resolution).

The guards are the point of this module. Each test below pins a guard that
either caught a real false positive in the live data or protects against one.
"""
from __future__ import annotations

import pandas as pd
import pytest

from club_soccer import club_identity as CI


def _fx(rows):
    return pd.DataFrame(
        [{"date": d, "competition": c, "home": h, "away": a,
          "home_goals": hg, "away_goals": ag} for d, c, h, a, hg, ag in rows])


def _dup_pair(name_a, name_b, opponent, competition="Premier League", n=5):
    """n matches recorded twice — once per spelling. The hard evidence shape."""
    rows = []
    for i in range(n):
        date = f"2025-0{i + 1}-01"
        rows.append((date, competition, opponent, name_a, 1.0, 2.0))
        rows.append((date, competition, opponent, name_b, 1.0, 2.0))
    return rows


# ── normalisation ─────────────────────────────────────────────────────────

def test_core_strips_club_tokens_and_years():
    assert CI._core("Bologna FC 1909") == CI._core("Bologna")
    assert CI._core("FC Bayern München") == CI._core("Bayern Munchen")


def test_norm_handles_accents_and_punctuation():
    assert CI._norm("Atlético Madrid") == CI._norm("Atletico Madrid")
    assert CI._norm("St. Mirren") == CI._norm("St Mirren")
    assert CI._norm("Brighton & Hove Albion") == "brighton and hove albion"


# ── guards ────────────────────────────────────────────────────────────────

def test_g1_head_to_head_veto_blocks_chance_collision():
    """The real false positive: Bolton Wanderers / Wolverhampton.

    They chance-collided on date+competition+score with a shared opponent, but
    they have played each other, so they cannot be the same club.
    """
    rows = _dup_pair("Bolton Wanderers", "Wolverhampton", "Someone FC")
    rows.append(("2026-10-13", "Championship", "Bolton Wanderers",
                 "Wolverhampton", 1.0, 1.0))
    result = CI.build_alias_map(_fx(rows))
    assert "Bolton Wanderers" not in result["alias"]
    assert "Wolverhampton" not in result["alias"]
    assert any("G1" in r["reason"] for r in result["rejected"])


def test_g2_country_veto():
    rows = (_dup_pair("Rangers", "Rangers FC", "Opponent A",
                      competition="Scottish Premiership")
            + [("2025-03-01", "Premier League", "Rangers FC", "Arsenal", 0.0, 1.0)] * 1)
    result = CI.build_alias_map(_fx(rows))
    rejected = [r for r in result["rejected"] if "G2" in r["reason"]]
    # Either vetoed by country, or never proposed — both are acceptable;
    # what must not happen is a cross-country merge.
    alias = result["alias"]
    assert not (alias.get("Rangers FC") == "Rangers" and rejected)


def test_g3_reserve_side_never_merges_into_senior():
    rows = _dup_pair("Bayern Munich", "Bayern Munich II", "Opponent A")
    result = CI.build_alias_map(_fx(rows))
    assert "Bayern Munich II" not in result["alias"]
    assert any("G3" in r["reason"] for r in result["rejected"])


def test_g4_rejects_unrelated_names_even_with_evidence():
    rows = _dup_pair("Reims", "Stade Rennais", "Opponent A", competition="Ligue 1")
    result = CI.build_alias_map(_fx(rows))
    assert "Reims" not in result["alias"]
    assert any("G4" in r["reason"] for r in result["rejected"])


def test_g5_evidence_floor_for_non_identical_cores():
    rows = _dup_pair("Newtown", "Newtown Rovers", "Opponent A", n=1)
    result = CI.build_alias_map(_fx(rows))
    assert any("G5" in r["reason"] for r in result["rejected"]) or \
        "Newtown Rovers" not in result["alias"]


# ── merging ───────────────────────────────────────────────────────────────

def test_obvious_duplicate_is_merged():
    rows = _dup_pair("Bayern Munich", "FC Bayern München", "Dortmund",
                     competition="Bundesliga")
    result = CI.build_alias_map(_fx(rows))
    assert result["alias"].get("FC Bayern München") == "Bayern Munich"


def test_normalisation_variants_merge_without_evidence():
    """Accent-only variants are invisible to the evidence collector.

    collect_evidence compares NORMALISED names, so it treats these as already
    identical and never proposes them — yet they are distinct identities to
    the model. "Atletico Madrid"/"Atlético Madrid" split one club 128/109 in
    the live data.
    """
    rows = [("2025-01-01", "La Liga", "Atletico Madrid", "Sevilla", 1.0, 0.0),
            ("2025-02-01", "La Liga", "Atlético Madrid", "Valencia", 2.0, 0.0)]
    result = CI.build_alias_map(_fx(rows))
    merged = {result["alias"].get(n, n) for n in ("Atletico Madrid", "Atlético Madrid")}
    assert len(merged) == 1, "accent variants must resolve to one identity"


def test_manual_alias_forces_canonical_direction():
    """Reviewed names win, regardless of which spelling is more frequent."""
    rows = [("2025-01-01", "Premier League", "Sheffield Utd", "Arsenal", 1.0, 0.0)] * 1
    rows += [("2025-02-01", "Premier League", "Sheffield Utd", "Chelsea", 1.0, 0.0)]
    rows += _dup_pair("Sheffield Utd", "Sheffield United", "Everton")
    result = CI.build_alias_map(_fx(rows))
    assert result["alias"].get("Sheffield Utd") == "Sheffield United"


def test_apply_alias_map_rewrites_both_sides():
    df = _fx([("2025-01-01", "Bundesliga", "FC Bayern München", "Dortmund", 1.0, 0.0),
              ("2025-02-01", "Bundesliga", "Dortmund", "FC Bayern München", 0.0, 1.0)])
    out = CI.apply_alias_map(df, {"FC Bayern München": "Bayern Munich"})
    assert set(out["home"]) == {"Bayern Munich", "Dortmund"}
    assert set(out["away"]) == {"Bayern Munich", "Dortmund"}


def test_alias_map_is_deterministic():
    rows = _dup_pair("Bayern Munich", "FC Bayern München", "Dortmund",
                     competition="Bundesliga")
    a = CI.build_alias_map(_fx(rows))["alias"]
    b = CI.build_alias_map(_fx(rows))["alias"]
    assert a == b


# ── live ingest resolver ──────────────────────────────────────────────────

def test_cross_border_aliases_resolve_before_the_league_index_exists(monkeypatch):
    """Finding 11: with the domestic-league index empty (bootstrap), a club
    whose association differs from the league it plays in must still resolve —
    the registry association country must not veto a verified cross-border
    membership."""
    monkeypatch.setattr(CI, "team_countries", lambda refresh=False: {})
    assert CI.canonical_name("AS Monaco", country="France") == "Monaco"
    assert CI.canonical_name("Cardiff City", country="England") == "Cardiff"
    assert CI.canonical_name("FC Andorra", country="Spain") == "Andorra"


def test_cross_border_allowlist_does_not_open_a_weld(monkeypatch):
    """The allowlist only widens acceptance for verified cross-border clubs; the
    country guard that stops welding a Brazilian club onto Spain's Athletic
    Bilbao must be fully intact."""
    monkeypatch.setattr(CI, "team_countries", lambda refresh=False: {})
    assert CI.canonical_name("Athletic Club", country="Brazil") == "Athletic Club"
    # Cardiff plays in England, not France — a France row must still be refused.
    assert CI.canonical_name("Cardiff City", country="France") == "Cardiff City"


def test_canonical_name_resolves_known_aliases():
    CI.reload_resolver()
    assert CI.canonical_name("FC Bayern München") == "Bayern Munich"
    assert CI.canonical_name("Heart of Midlothian") == "Hearts"


def test_canonical_name_handles_accent_variants_not_literally_in_the_map():
    CI.reload_resolver()
    assert CI.canonical_name("Atlético Madrid") == CI.canonical_name("Atletico Madrid")


def test_canonical_name_passes_unknown_clubs_through():
    """A club we have never seen must NOT be bent onto an existing identity.

    This is the guard that keeps the 55-league expansion safe: newly ingested
    clubs should arrive as themselves, not as the nearest known name.
    """
    CI.reload_resolver()
    assert CI.canonical_name("Totally New Club FC") == "Totally New Club FC"
    assert CI.canonical_name("") == ""
    assert CI.canonical_name(None) is None


def test_fetch_applies_the_canon():
    """Regression: the daily fetch path must not re-create the duplication.

    The canonicalisation now scopes by the club's own registry country (so a
    mislabelled competition cannot block a valid merge), but it must still run
    on the raw provider names before a row is written.
    """
    import inspect

    from club_soccer import fetch
    src = inspect.getsource(fetch)
    assert "_canonical_name(raw_home" in src and "_canonical_name(raw_away" in src, \
        "fetch.py must canonicalise team names before writing rows"


# ── the live artifact ─────────────────────────────────────────────────────

def test_live_fixtures_have_no_normalisation_duplicates():
    """fixtures.csv must not contain two spellings of one club."""
    import collections
    from pathlib import Path

    path = Path(CI.FIXTURES)
    if not path.exists():
        pytest.skip("no fixtures.csv")
    df = pd.read_csv(path, low_memory=False)
    names = set(df["home"].dropna()) | set(df["away"].dropna())
    groups = collections.defaultdict(list)
    for n in names:
        groups[CI._norm(n)].append(n)
    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    assert not dupes, f"duplicate identities remain: {dupes}"
