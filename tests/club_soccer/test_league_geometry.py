#!/usr/bin/env python3
"""League size is a per-season property, and some misattributions are
competition-scoped.

`Competition.teams_n` carried one number per league, which asserts a division
has always had its current shape. Ligue 1 ran 20 clubs until 2023/24, Ligue 2
until 2024/25, and the Süper Lig moved between 19, 20 and 21 — 16 league-seasons
read as corrupt purely because the expectation had no season dimension.

Separately, Premier League 2023/24 and 2024/25 held 56 rows naming "Bolton"
that duplicate Wolverhampton fixtures score-for-score. Bolton is a real club in
the Championship, League One and both cups, so the correction has to be scoped
to the competition it is wrong in.
"""
from __future__ import annotations

import pandas as pd
import pytest

from club_soccer import club_identity as CI
from club_soccer import identity_review as IR
from club_soccer.competitions import TEAMS_N_BY_SEASON, get as comp_get
from club_soccer.competitions import teams_n_for


# --- season-aware league size --------------------------------------------

def test_unlisted_seasons_fall_back_to_the_static_size():
    assert teams_n_for("Premier League", 2026) == comp_get("Premier League").teams_n
    assert teams_n_for("Premier League", 2019) == comp_get("Premier League").teams_n


def test_listed_seasons_use_the_override():
    assert teams_n_for("Ligue 1", 2022) == 20
    assert teams_n_for("Ligue 1", 2026) == 18
    assert teams_n_for("Ligue 2", 2023) == 20
    assert teams_n_for("Super Lig", 2022) == 19


def test_unknown_competition_and_bad_season_are_safe():
    assert teams_n_for("Not A League", 2026) == 0
    assert teams_n_for(None, 2026) == 0
    assert teams_n_for("Ligue 1", None) == comp_get("Ligue 1").teams_n
    assert teams_n_for("Ligue 1", "not-a-year") == comp_get("Ligue 1").teams_n


@pytest.mark.parametrize("key", sorted(TEAMS_N_BY_SEASON))
def test_every_override_names_a_real_league_and_changes_something(key):
    name, season = key
    comp = comp_get(name)
    assert comp is not None, f"{name!r} is not a registered competition"
    assert comp.kind == "league"
    assert TEAMS_N_BY_SEASON[key] != comp.teams_n, \
        f"{key} restates the static teams_n and should be deleted"


def test_overrides_are_corroborated_by_a_clean_round_robin():
    """Each override must survive the arithmetic that justified it: N member
    clubs playing an exact double round-robin of 2*(N-1) matches. This is what
    separates a real size change from an unmerged duplicate identity."""
    from collections import Counter

    from club_soccer import model as M

    league = M.load_fixtures()
    league = league[league["type"] == "league"]
    for (name, season), expected in sorted(TEAMS_N_BY_SEASON.items()):
        group = league[(league["competition"] == name)
                       & (league["season"] == season)]
        if group.empty:
            continue
        counts = Counter()
        for row in group.itertuples(index=False):
            counts[row.home] += 1
            counts[row.away] += 1
        values = sorted(counts.values())
        median = values[len(values) // 2]
        members = [t for t, n in counts.items() if n >= median * 0.5]
        assert len(members) == expected, (
            f"{name} {season}: override says {expected} clubs, data shows "
            f"{len(members)} playing a full schedule"
        )
        assert median == 2 * (expected - 1), (
            f"{name} {season}: {expected} clubs implies {2 * (expected - 1)} "
            f"matches each, data shows a median of {median}"
        )


def test_oversized_detector_uses_the_override():
    rows = {(r["competition"], r["season"])
            for r in IR.oversized_league_seasons()}
    for name, season in TEAMS_N_BY_SEASON:
        assert (name, season) not in rows, \
            f"{name} {season} still flagged despite a season override"


# --- competition-scoped aliases ------------------------------------------

def test_bolton_is_rewritten_only_inside_the_premier_league():
    assert CI.canonical_name("Bolton", country="England",
                             competition="Premier League") == "Wolverhampton"
    for competition in ("Championship", "League One", "FA Cup", "EFL Cup"):
        assert CI.canonical_name("Bolton", country="England",
                                 competition=competition) == "Bolton", \
            f"the real Bolton must survive in {competition}"
    assert CI.canonical_name("Bolton", country="England") == "Bolton"


def test_scope_precedence_is_competition_then_country_then_global():
    """A competition entry must win over a country entry for the same name."""
    resolver = CI._load_competition_resolver()
    assert "Premier League" in resolver
    literal, _ = resolver["Premier League"]
    assert literal["Bolton"] == "Wolverhampton"


def test_no_bolton_rows_remain_in_the_premier_league():
    from club_soccer import model as M

    fixtures = M.load_fixtures()
    premier = fixtures[fixtures["competition"] == "Premier League"]
    names = set(premier["home"]) | set(premier["away"])
    assert "Bolton" not in names
    # …and the real club is untouched elsewhere.
    elsewhere = fixtures[(fixtures["home"] == "Bolton")
                         | (fixtures["away"] == "Bolton")]
    assert set(elsewhere["competition"]) >= {"Championship", "League One"}


def test_premier_league_seasons_have_twenty_clubs():
    from club_soccer import model as M

    fixtures = M.load_fixtures()
    league = fixtures[(fixtures["competition"] == "Premier League")
                      & (fixtures["type"] == "league")]
    for season, group in league.groupby("season"):
        names = set(group["home"]) | set(group["away"])
        assert len(names) <= 20, f"Premier League {season}: {sorted(names)}"


def test_alias_map_provenance_is_reported():
    """A run on a host with a stale alias map is otherwise invisible."""
    from club_soccer import health as H

    report = H.run_checks(network=False)
    assert report["alias_map_sha256"]
    assert report["alias_map_entries"] > 0
