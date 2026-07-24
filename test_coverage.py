#!/usr/bin/env python3
"""Tests for club_soccer.coverage (P0 evidence instrumentation)."""
from __future__ import annotations

import pandas as pd
import pytest

from club_soccer import coverage as C


def _frame(rows):
    return pd.DataFrame(rows, columns=["date", "competition", "home", "away"])


def _league_rows(team, n, competition="Premier League", start="2026-01-01"):
    base = pd.Timestamp(start)
    return [{"date": base + pd.Timedelta(days=3 * i), "competition": competition,
             "home": team, "away": f"Opp {i}"} for i in range(n)]


def test_well_evidenced_team_is_full():
    df = _frame(_league_rows("Arsenal", 30))
    ev = C.build_team_evidence(df)
    tier, reasons = C.team_tier(ev["Arsenal"])
    assert tier == C.TIER_FULL
    assert reasons == []


def test_euro_only_team_is_never_full_however_many_matches():
    """The Sturm Graz signature: plenty of matches, all European.

    A naive match-count threshold would call this fully evidenced. It is not —
    those matches are against opposition the model also cannot rate, so they
    do not identify a rating.
    """
    df = _frame(_league_rows("SK Sturm Graz", 40, competition="Champions League"))
    ev = C.build_team_evidence(df)
    tier, reasons = C.team_tier(ev["SK Sturm Graz"])
    assert tier == C.TIER_THIN
    assert any("no domestic-league data" in r for r in reasons)


def test_few_matches_is_thin():
    df = _frame(_league_rows("Newcomer FC", 4))
    ev = C.build_team_evidence(df)
    tier, reasons = C.team_tier(ev["Newcomer FC"])
    assert tier == C.TIER_THIN
    assert any("near-negligible" in r for r in reasons)


def test_absent_team_is_defaulted():
    tier, reasons = C.team_tier(None)
    assert tier == C.TIER_DEFAULTED
    assert any("defaults" in r for r in reasons)


def test_stale_matches_do_not_count_as_recent():
    df = _frame(_league_rows("Old Team", 30, start="2015-01-01")
                + _league_rows("Current Team", 30, start="2026-01-01"))
    ev = C.build_team_evidence(df)
    # as_of defaults to the frame max, so 2015 falls outside the 730-day window
    assert C.team_tier(ev["Old Team"])[0] == C.TIER_THIN
    assert C.team_tier(ev["Current Team"])[0] == C.TIER_FULL


def test_unknown_competition_counts_as_domestic():
    """Expansion leagues may appear in fixtures before the registry.

    Treating an unrecognised competition as European would wrongly hold a
    well-evidenced team at `thin`; the conservative reading is domestic.
    """
    df = _frame(_league_rows("Sturm Graz", 30, competition="Austrian Bundesliga"))
    ev = C.build_team_evidence(df)
    assert ev["Sturm Graz"]["has_domestic"] is True
    assert C.team_tier(ev["Sturm Graz"])[0] == C.TIER_FULL


def test_match_tier_is_the_worse_of_the_two_sides():
    df = _frame(_league_rows("Hearts", 30, competition="Scottish Premiership")
                + _league_rows("Sturm", 20, competition="Champions League"))
    params = {"teams": ["Hearts", "Sturm"],
              "elo": {"Hearts": 1700.0, "Sturm": 1500.0},
              "team_evidence": C.build_team_evidence(df)}
    cov = C.match_coverage(params, "Hearts", "Sturm")
    assert cov["home"]["tier"] == C.TIER_FULL
    assert cov["away"]["tier"] == C.TIER_THIN
    assert cov["tier"] == C.TIER_THIN
    assert cov["reliable"] is False
    assert any("Sturm" in n for n in cov["notes"])


def test_worst_tier_ordering():
    assert C.worst_tier("full", "full") == "full"
    assert C.worst_tier("full", "thin") == "thin"
    assert C.worst_tier("thin", "defaulted") == "defaulted"


def test_unknown_team_in_params_is_defaulted():
    params = {"teams": [], "elo": {}, "team_evidence": {}}
    cov = C.match_coverage(params, "Nobody A", "Nobody B")
    assert cov["tier"] == C.TIER_DEFAULTED


def test_legacy_params_without_evidence_degrade_to_thin_not_defaulted():
    """A stale artifact should raise caution, not a false 'no data' alarm."""
    params = {"teams": ["Arsenal"], "elo": {"Arsenal": 1700.0}}
    rec = C.team_coverage(params, "Arsenal")
    assert rec["tier"] == C.TIER_THIN
    assert any("predates coverage" in r for r in rec["reasons"])


def test_summarise_counts_euro_only_teams():
    df = _frame(_league_rows("Arsenal", 30)
                + _league_rows("Sturm", 30, competition="Champions League"))
    params = {"team_evidence": C.build_team_evidence(df)}
    s = C.summarise(params)
    # 2 named teams + the shared "Opp 0..29" pool both fixtures draw from
    assert s["teams"] == 32
    assert s["euro_only_teams"] >= 1


# ── integration: the instrumentation must reach the real pricing path ──────

def test_predict_attaches_coverage():
    from club_soccer import model as M
    params = M.load_params()
    teams = params["teams"]
    if len(teams) < 2:
        pytest.skip("no fitted params available")
    out = M.predict(teams[0], teams[1], "Premier League", params=params)
    assert "coverage" in out
    assert out["coverage"]["tier"] in (C.TIER_FULL, C.TIER_THIN, C.TIER_DEFAULTED)


def test_fit_emits_team_evidence():
    from club_soccer import model as M
    params = M.load_params()
    assert "team_evidence" in params, "fit() must record per-team evidence"
    assert len(params["team_evidence"]) > 0
