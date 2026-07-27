#!/usr/bin/env python3
"""Tests for P6a — keeping the expansion leagues fresh.

The failure this guards against is silent. A league that stops updating still
prices fixtures, just from frozen ratings, and the global "days since last
result" stays healthy because 40 other leagues are flowing.
"""
from __future__ import annotations

import pandas as pd
import pytest

from club_soccer import seed_fdcouk_leagues as SFL
from club_soccer.competitions import BSD_LEAGUE_ALIASES, COMPETITIONS


def test_bsd_less_set_is_derived_not_hardcoded():
    """Adding a BSD alias later must automatically drop a league from the set."""
    import inspect
    src = inspect.getsource(SFL.needs_fdcouk_refresh)
    assert "BSD_LEAGUE_ALIASES" in src


def test_the_eight_bsd_less_leagues_are_identified():
    names = {c.name for c in SFL.needs_fdcouk_refresh()}
    expected = {"Serie B", "2. Bundesliga", "Ligue 2", "National League",
                "Austrian Bundesliga", "Danish Superliga",
                "League of Ireland", "Russian Premier League"}
    assert names == expected


def test_austria_has_no_bsd_path():
    """The league the whole expansion existed to fix must be in the refresh
    set — BSD's catalogue does not carry it."""
    names = {c.name for c in SFL.needs_fdcouk_refresh()}
    assert "Austrian Bundesliga" in names


def test_bsd_served_leagues_are_excluded():
    """These resolve through the P2 aliases on the daily fetch already."""
    names = {c.name for c in SFL.needs_fdcouk_refresh()}
    for served in ("Eredivisie", "Liga Portugal", "Super Lig", "Ekstraklasa",
                   "Segunda División", "Premier League", "Bundesliga"):
        assert served not in names


def test_every_refresh_league_has_a_fetchable_source():
    for comp in SFL.needs_fdcouk_refresh():
        assert comp.fdcouk_code or comp.fdcouk_new, \
            f"{comp.name} is in the refresh set with no fd.co.uk source"


def test_seasons_include_the_upcoming_one():
    """A refresh that only knows about past seasons cannot pick up new results."""
    assert "2627" in SFL.SEASONS


# ── staleness reporting ───────────────────────────────────────────────────

def test_staleness_reports_every_competition_with_results():
    rows = SFL.staleness()
    assert rows
    for r in rows:
        assert {"competition", "last_result", "days_stale", "upcoming",
                "has_bsd_path", "warn"} <= set(r)


def test_staleness_is_sorted_worst_first():
    rows = SFL.staleness()
    days = [r["days_stale"] for r in rows]
    assert days == sorted(days, reverse=True)


def test_a_league_with_upcoming_fixtures_is_not_flagged_stale():
    """Off-season gaps are normal; a league with fixtures scheduled is fine."""
    for r in SFL.staleness():
        if r["upcoming"] > 0:
            assert not r["warn"]


def test_out_of_season_leagues_do_not_warn():
    """A winter league idle in July must not alarm — warning on every summer
    dormancy trains the operator to ignore the alert."""
    for r in SFL.staleness():
        if not r.get("in_season", True):
            assert not r["warn"], f"{r['competition']} warned while out of season"


def test_active_months_are_proportional_not_any_match():
    """A stray fixture in a month must not make it count as in-season — 14
    seasons accumulate one somewhere in almost every month."""
    import pandas as pd
    # A league that plays Aug-May with one rescheduled July tie across history.
    dates = pd.Series(
        [f"2023-{m:02d}-10" for m in (8, 9, 10, 11, 12, 1, 2, 3, 4, 5)] * 6
        + ["2022-07-15"])
    active = SFL._active_months(dates)
    assert 7 not in active, "a single July fixture must not count as in-season"
    assert 9 in active


def test_refresh_health_flags_only_genuine_lag(monkeypatch):
    """The authoritative check warns only when the source has results we lack,
    not on pre-season gaps where the source is also empty."""
    from club_soccer.competitions import get

    comp = get("Austrian Bundesliga")
    monkeypatch.setattr(
        SFL, "load_competition",
        lambda c, refresh=False, refresh_seasons=None: [{"date": "2026-05-17"}],
    )
    # our fixtures.csv latest for Austria is 2026-05-17 too -> not behind
    rows = SFL.refresh_health([comp])
    assert rows and rows[0]["behind"] is False

    # source jumps ahead -> genuinely behind
    monkeypatch.setattr(
        SFL, "load_competition",
        lambda c, refresh=False, refresh_seasons=None: [{"date": "2026-08-30"}],
    )
    rows = SFL.refresh_health([comp])
    assert rows[0]["behind"] is True


def test_staleness_marks_bsd_path_correctly():
    rows = {r["competition"]: r for r in SFL.staleness()}
    if "Austrian Bundesliga" in rows:
        assert rows["Austrian Bundesliga"]["has_bsd_path"] is False
    if "Eredivisie" in rows:
        assert rows["Eredivisie"]["has_bsd_path"] is True


# ── pipeline wiring ───────────────────────────────────────────────────────

def test_refresh_is_wired_into_the_daily_pipeline():
    """A refresh nobody calls is not a refresh."""
    import inspect

    from club_soccer import season
    src = inspect.getsource(season.run_network_steps)
    assert "seed_fdcouk_leagues" in src
    assert "refresh" in src


def test_health_reports_per_league_staleness():
    import inspect

    from club_soccer import health
    src = inspect.getsource(health)
    assert "stale_leagues" in src


def test_refresh_returns_a_summary_without_writing_when_asked():
    comps = [c for c in SFL.needs_fdcouk_refresh() if c.name == "Danish Superliga"]
    if not comps:
        pytest.skip("Danish Superliga not registered")
    before = len(pd.read_csv(SFL.FIXTURES, low_memory=False))
    result = SFL.refresh(comps, write=False, verbose=False)
    after = len(pd.read_csv(SFL.FIXTURES, low_memory=False))
    assert after == before, "write=False must not touch fixtures.csv"
    assert "per_competition" in result


def test_fixture_ids_are_stable_so_refresh_updates_rather_than_duplicates():
    a = SFL._fid("Austrian Bundesliga", "2026-08-01", "Sturm Graz", "Salzburg")
    b = SFL._fid("Austrian Bundesliga", "2026-08-01", "Sturm Graz", "Salzburg")
    assert a == b
