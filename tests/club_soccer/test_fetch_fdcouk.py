"""Closing-odds feed (market_history.csv) refresh + persistence."""
from __future__ import annotations

import inspect

import pandas as pd

from club_soccer import fetch_fdcouk as FD


def _sample(match_date: str = "2026-08-10") -> pd.DataFrame:
    row = {c: "" for c in FD.MARKET_COLUMNS}
    row.update({"match_date": match_date, "competition": "Premier League",
                "home": "Arsenal", "away": "Chelsea",
                "psc_h": 2.0, "psc_d": 3.4, "psc_a": 3.8})
    return pd.DataFrame([row])


def test_refresh_persists_market_history(tmp_path, monkeypatch):
    """build() only returns a frame; the pipeline needs it WRITTEN to disk, or
    the closing feed silently never updates and no bet ever gets a CLV score."""
    monkeypatch.setattr(FD, "DATA", tmp_path)
    monkeypatch.setattr(FD, "MARKET_HISTORY", tmp_path / "market_history.csv")
    monkeypatch.setattr(FD, "build", lambda **_: _sample())

    FD.refresh(verbose=False)

    assert FD.MARKET_HISTORY.exists()
    out = pd.read_csv(FD.MARKET_HISTORY)
    assert len(out) == 1
    assert str(out["match_date"].iloc[0]) == "2026-08-10"


def test_refresh_never_clobbers_with_an_empty_frame(tmp_path, monkeypatch):
    """A transient network failure (build returns nothing) must not wipe the
    accumulated closing history."""
    mh = tmp_path / "market_history.csv"
    mh.write_text("match_date,competition\n2025-05-01,Premier League\n")
    monkeypatch.setattr(FD, "DATA", tmp_path)
    monkeypatch.setattr(FD, "MARKET_HISTORY", mh)
    monkeypatch.setattr(FD, "build",
                        lambda **_: pd.DataFrame(columns=FD.MARKET_COLUMNS))

    FD.refresh(verbose=False)

    assert "2025-05-01,Premier League" in mh.read_text()   # preserved, not wiped


def test_daily_pipeline_persists_market_history_not_just_builds():
    """The pipeline step must call the persisting refresh(), not the
    non-writing build() it used to be wired to."""
    from club_soccer import season
    src = inspect.getsource(season)
    assert "FD.refresh" in src
    assert 'FD.build' not in src, "pipeline still references the non-writing build()"


def test_current_season_string_covers_the_new_campaign():
    """The refresh must know about the season now starting, or European fixtures
    captured from August onward would have no closing reference."""
    assert FD._current_season_string() in FD.SEASON_STRINGS
