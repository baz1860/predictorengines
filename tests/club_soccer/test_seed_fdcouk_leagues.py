"""Tests for the fd.co.uk expansion seeder."""
from __future__ import annotations

import inspect
import json

import pandas as pd

from club_soccer import club_identity as CI
from club_soccer import seed_fdcouk_leagues as S


def test_every_wave_references_real_sourced_competitions():
    seen = set()
    for wave in S.WAVES:
        for comp in S.wave_competitions(wave):
            assert comp.name not in seen
            assert comp.fdcouk_code or comp.fdcouk_new
            seen.add(comp.name)


def test_fixture_ids_are_deterministic_and_directional():
    first = S._fid("Eredivisie", "2024-09-21", "Ajax", "PSV")
    assert first == S._fid("Eredivisie", "2024-09-21", "Ajax", "PSV")
    assert first != S._fid("Eredivisie", "2024-09-21", "PSV", "Ajax")


def test_seeder_has_no_secondary_fuzzy_mutation_workflow():
    source = inspect.getsource(S)
    assert "propose_domestic_merges" not in source
    assert "_persist_aliases" not in source
    assert ".bak.pre_wave" not in source


def test_alias_artifact_is_cumulative_and_chain_free():
    alias = json.loads(CI.ALIAS_MAP.read_text())["alias"]
    for source in (
        "FC Bayern München",
        "Heart of Midlothian",
        "FC Internazionale Milano",
    ):
        assert source in alias
    assert all(target not in alias for target in alias.values())


def test_expansion_leagues_are_present_in_fixtures():
    frame = pd.read_csv(CI.FIXTURES, low_memory=False)
    competitions = set(frame["competition"].dropna())
    for name in (
        "Eredivisie",
        "Austrian Bundesliga",
        "Liga Portugal",
        "Super Lig",
        "Serie B",
        "Ekstraklasa",
    ):
        assert name in competitions


def test_sturm_graz_has_one_domestically_rated_identity():
    frame = pd.read_csv(CI.FIXTURES, low_memory=False)
    names = set(frame["home"].dropna()) | set(frame["away"].dropna())
    assert "Sturm Graz" in names
    assert "SK Sturm Graz" not in names
    rows = frame[(frame["home"] == "Sturm Graz") | (frame["away"] == "Sturm Graz")]
    assert "Austrian Bundesliga" in set(rows["competition"])


def test_live_fixtures_have_no_self_matches_or_duplicate_keys():
    frame = pd.read_csv(CI.FIXTURES, low_memory=False)
    assert int((frame["home"] == frame["away"]).sum()) == 0
    key = (
        frame["date"].astype(str).str[:10]
        + "|" + frame["competition"].astype(str)
        + "|" + frame["home"].astype(str)
        + "|" + frame["away"].astype(str)
    )
    assert int(key.duplicated().sum()) == 0


def test_all_seed_writers_use_the_canonical_season_helper():
    from club_soccer import seed_openfootball, seed_real

    for module in (seed_real, seed_openfootball, S):
        source = inspect.getsource(module)
        assert "season_for_date" in source
        assert "if month >= 7 else year - 1" not in source
