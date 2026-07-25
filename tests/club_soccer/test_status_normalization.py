#!/usr/bin/env python3
"""Regression: canonical status normalization (no naive [:3] truncation).

The old truncation mislabeled AWARDED->AWA and ABANDONED->ABA, so awarded
results never settled and abandoned rows were not treated as void. These tests
pin the canonical map and the training/void semantics that depend on it.

Run: python3 -m pytest test_status_normalization.py -q
"""
from __future__ import annotations

import pandas as pd
import pytest

from club_soccer import schema
from club_soccer import model as M


@pytest.mark.parametrize("raw,want", [
    ("FINISHED", "FIN"), ("finished", "FIN"), ("FT", "FT"),
    ("AET", "AET"), ("PEN", "PEN"),
    ("AWARDED", "AWD"), ("AWA", "AWD"),          # heals old truncation
    ("ABANDONED", "ABD"), ("ABA", "ABD"),        # heals old truncation
    ("POSTPONED", "POS"), ("CANCELLED", "CAN"), ("CANCELED", "CAN"),
    ("SUSPENDED", "SUS"), ("INTERRUPTED", "INT"),
    ("SCHEDULED", "NOT"), ("TIMED", "NOT"), ("NOT_STARTED", "NOT"),
    ("IN_PLAY", "LIV"), ("PAUSED", "LIV"), ("INPROGRESS", "LIV"),
    ("", ""), ("NaN", ""), (None, ""),
])
def test_normalize_status_map(raw, want):
    assert schema.normalize_status(raw) == want


# ── round-7 finding 5: unknown statuses must quarantine, not truncate ────
@pytest.mark.parametrize("raw", [
    "POSITIVE", "POSTED",        # would have truncated to POS  -> VOID
    "FINALIZED", "FINAL_PENDING",  # would have truncated to FIN  -> OFFICIAL RESULT
    "CANVASSED",                 # -> CAN (void)
    "ABDUCTED",                  # -> ABD (void)
    "SUSPECT",                   # -> SUS (void)
    "INTERIM",                   # -> INT (void)
    "WALKOVER", "TBD", "??", "totally new status",
])
def test_unknown_status_quarantines_and_is_never_void_or_result(raw):
    canon = schema.normalize_status(raw)
    assert canon == schema.QUARANTINE_STATUS
    assert canon not in schema.VOID_STATUSES
    assert canon not in schema.OFFICIAL_RESULT_STATUSES
    assert canon in schema.TRAINING_EXCLUDED_STATUSES   # never trains


def test_quarantined_rows_are_excluded_from_training_and_pricing():
    df = pd.DataFrame([
        {"home": "A", "away": "B", "home_goals": 1, "away_goals": 0,
         "status": "FINALIZED", "date": "2024-01-01"},     # unknown, has a score
        {"home": "C", "away": "D", "home_goals": 2, "away_goals": 1,
         "status": "FINISHED", "date": "2024-01-01"},
    ])
    assert list(M.played(df)["home"]) == ["C"]             # UNK never trains
    up = pd.DataFrame([
        {"home": "E", "away": "F", "home_goals": None, "away_goals": None,
         "status": "POSITIVE", "date": "2030-01-01"},      # unknown, no score
        {"home": "G", "away": "H", "home_goals": None, "away_goals": None,
         "status": "SCHEDULED", "date": "2030-01-01"},
    ])
    assert list(M.upcoming(up)["home"]) == ["G"]           # UNK is never priced


def test_write_fixtures_preserves_status_raw_for_quarantined_rows(tmp_path):
    from club_soccer import fetch
    df = pd.DataFrame([
        {"fixture_id": "1", "status": "SOME_NEW_STATE", "home": "A", "away": "B",
         "home_goals": 1, "away_goals": 0, "date": "2024-01-01"},
        {"fixture_id": "2", "status": "FINISHED", "home": "C", "away": "D",
         "home_goals": 2, "away_goals": 1, "date": "2024-01-01"},
    ])
    out = fetch.write_fixtures(df, path=tmp_path / "fixtures.csv")
    by_id = {str(r.fixture_id): r for r in out.itertuples()}
    assert by_id["1"].status == schema.QUARANTINE_STATUS
    assert by_id["1"].status_raw == "SOME_NEW_STATE"       # recoverable
    assert int(by_id["1"].home_goals) == 1                 # data NOT destroyed
    assert by_id["2"].status == "FIN"

    # A later recognised provider response clears the old quarantine detail.
    healed = out.copy()
    healed.loc[healed["fixture_id"] == "1", "status"] = "FINISHED"
    healed = fetch.write_fixtures(healed, path=tmp_path / "healed.csv")
    assert healed.loc[healed["fixture_id"] == "1", "status_raw"].item() == ""


def test_source_mappers_preserve_the_real_unknown_status(tmp_path):
    """Normalising inside a mapper must not turn status_raw into literal UNK."""
    from club_soccer import fetch, seed_real

    bsd_event = {"id": "bsd-1", "home_team": "A", "away_team": "B",
                 "event_date": "2027-01-01T12:00:00Z",
                 "status": "SOME_NEW_STATE"}
    bsd = fetch._bsd_to_fixture_row(bsd_event, "Test", 1, "X", "league")
    assert bsd["status"] == schema.QUARANTINE_STATUS
    assert bsd["status_raw"] == "SOME_NEW_STATE"

    seed = seed_real._bsd_to_row(
        {**bsd_event, "date": bsd_event["event_date"]}, "Test", 1, "X", "league")
    assert seed["status"] == schema.QUARANTINE_STATUS
    assert seed["status_raw"] == "SOME_NEW_STATE"

    out = fetch.write_fixtures(pd.DataFrame([bsd, seed]),
                               path=tmp_path / "fixtures.csv")
    assert set(out["status_raw"]) == {"SOME_NEW_STATE"}


def test_current_fixture_data_has_no_quarantined_rows():
    """The quarantine must be preventive: no existing row may reclassify."""
    from club_soccer import fetch as F
    df = pd.read_csv(F.FIXTURES, low_memory=False)
    st = df["status"].map(schema.normalize_status)
    bad = sorted(set(df.loc[st.isin(schema.QUARANTINE_STATUSES), "status"].astype(str)))
    assert not bad, f"unmapped statuses in production data: {bad}"


def test_awarded_is_official_result_but_not_trainable():
    assert "AWD" in schema.OFFICIAL_RESULT_STATUSES
    assert "AWD" in schema.TRAINING_EXCLUDED_STATUSES
    assert "AWD" not in schema.VOID_STATUSES
    assert "ABD" in schema.VOID_STATUSES


def test_played_excludes_awarded_and_void_including_legacy_codes():
    df = pd.DataFrame([
        {"home": "A", "away": "B", "home_goals": 3, "away_goals": 0,
         "status": "AWARDED", "date": "2024-01-01"},
        {"home": "C", "away": "D", "home_goals": 2, "away_goals": 1,
         "status": "FINISHED", "date": "2024-01-01"},
        {"home": "E", "away": "F", "home_goals": 1, "away_goals": 1,
         "status": "ABANDONED", "date": "2024-01-01"},
        {"home": "G", "away": "H", "home_goals": 2, "away_goals": 2,
         "status": "AWA", "date": "2024-01-01"},      # legacy truncation
    ])
    trainable = list(M.played(df)["home"])
    assert trainable == ["C"]                          # only the clean FINISHED row


def test_live_and_scheduled_rows_never_train(tmp_path=None):
    """Finding 10: a live in-play row can carry a non-final score. Admitting it
    to training grades an unfinished match as if it were over, so LIV (and NOT)
    must be excluded even though they are 'not void'."""
    assert "LIV" in schema.NON_TERMINAL_STATUSES
    assert "LIV" in schema.TRAINING_EXCLUDED_STATUSES
    assert "NOT" in schema.TRAINING_EXCLUDED_STATUSES
    df = pd.DataFrame([
        {"home": "A", "away": "B", "home_goals": 1, "away_goals": 0,
         "status": "IN_PLAY", "date": "2024-01-01"},          # live, non-final
        {"home": "C", "away": "D", "home_goals": 2, "away_goals": 1,
         "status": "FINISHED", "date": "2024-01-01"},
    ])
    assert list(M.played(df)["home"]) == ["C"]                # LIV never trains


def test_app_settlement_requires_an_official_terminal_status(
        tmp_path, monkeypatch):
    """The app must not grade an in-play score as a finished bet."""
    from app.engines import club_soccer as adapter_module

    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame([
        {
            "date": "2026-08-01", "home": "Arsenal", "away": "Chelsea",
            "home_goals": 1, "away_goals": 0, "status": "IN_PLAY",
        },
        {
            "date": "2026-08-02", "home": "Arsenal", "away": "Chelsea",
            "home_goals": 3, "away_goals": 0, "status": "AWARDED",
        },
    ]).to_csv(data / "fixtures.csv", index=False)
    monkeypatch.setattr(adapter_module, "ENGINE_DIR", tmp_path)
    open_bets = pd.DataFrame([
        {
            "match_date": "2026-08-01", "home": "Arsenal",
            "away": "Chelsea", "bet": "1X2", "side": "home",
        },
        {
            "match_date": "2026-08-02", "home": "Arsenal",
            "away": "Chelsea", "bet": "1X2", "side": "home",
        },
    ])
    graded = adapter_module.ClubSoccerAdapter().grade_open_bets(open_bets)
    assert 0 not in graded
    assert graded[1] == ("won", "3-0")


def test_write_fixtures_normalizes_and_clears_void(tmp_path):
    from club_soccer import fetch
    df = pd.DataFrame([
        {"fixture_id": "1", "status": "ABANDONED", "home": "A", "away": "B",
         "home_goals": 2, "away_goals": 1, "date": "2024-01-01"},
        {"fixture_id": "2", "status": "AWARDED", "home": "C", "away": "D",
         "home_goals": 3, "away_goals": 0, "date": "2024-01-01"},
    ])
    out = fetch.write_fixtures(df, path=tmp_path / "fixtures.csv")
    by_id = {str(r.fixture_id): r for r in out.itertuples()}
    assert by_id["1"].status == "ABD"                  # not "ABA"
    assert pd.isna(by_id["1"].home_goals)              # void result cleared
    assert by_id["2"].status == "AWD"                  # not "AWA"
    assert int(by_id["2"].home_goals) == 3             # awarded score retained
