#!/usr/bin/env python3
"""Fixture rows show the last-known book price next to the model's `fair`.

`fair` is 1/p_model: it sums to exactly 100% and carries no margin, so it is
not comparable to a book price without de-vigging. Printing it alone left the
operator with nothing to check it against, and when the odds fetch failed the
card dropped the comparison entirely instead of falling back to the snapshot
already on disk. Display only — a stale median must never reach an EV
calculation.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from club_soccer import season as S
from club_soccer import snapshot_odds as SO

NOW = datetime(2026, 8, 10, 7, 0, tzinfo=timezone.utc)


def _snapshot(rows):
    return pd.DataFrame(rows)


def _odds_rows(snapshot_time="2026-08-09T06:31:58+00:00",
               home="Silkeborg IF", away="Odense Boldklub"):
    base = {"snapshot_time": snapshot_time, "match_date": "2026-08-10",
            "competition": "Danish Superliga", "home": home, "away": away,
            "market": "1x2", "n_books": 16}
    return [dict(base, side=side, odds_median=price)
            for side, price in (("home", 2.625), ("draw", 3.6), ("away", 2.4))]


@pytest.fixture
def market(monkeypatch):
    def _build(rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "odds_history_club.csv"
            _snapshot(rows).to_csv(path, index=False)
            monkeypatch.setattr(SO, "ODDS_HISTORY_CSV", path)
            cache = S._MarketCache()
            cache._load()
            return cache
    return _build


def test_note_renders_price_and_age(market):
    cache = market(_odds_rows())
    note = cache.note("Danish Superliga", "2026-08-10",
                      "Silkeborg", "Odense", now=NOW)
    assert "mkt 2.62/3.60/2.40" in note
    assert "16 books" in note
    assert "24h old" in note


def test_fuller_provider_names_still_join(market):
    """Odds feeds carry 'Odense Boldklub' where fixtures.csv says 'Odense'."""
    cache = market(_odds_rows(home="Silkeborg IF", away="Odense Boldklub"))
    assert cache.note("Danish Superliga", "2026-08-10",
                      "Silkeborg", "Odense", now=NOW)


def test_stale_snapshots_are_withheld(market):
    """Eight days old is not a usable comparison, so print nothing rather
    than a number the operator might trust."""
    cache = market(_odds_rows(snapshot_time="2026-08-01T06:00:00+00:00"))
    assert cache.note("Danish Superliga", "2026-08-10",
                      "Silkeborg", "Odense", now=NOW) == ""


def test_unknown_fixture_returns_empty(market):
    cache = market(_odds_rows())
    assert cache.note("Danish Superliga", "2026-08-10",
                      "Viborg", "Aarhus", now=NOW) == ""


def test_partial_market_is_skipped(market):
    """A fixture missing the draw price cannot render a 1x2 line."""
    rows = [row for row in _odds_rows() if row["side"] != "draw"]
    cache = market(rows)
    assert cache.note("Danish Superliga", "2026-08-10",
                      "Silkeborg", "Odense", now=NOW) == ""


def test_missing_odds_file_is_not_an_error(monkeypatch):
    monkeypatch.setattr(SO, "ODDS_HISTORY_CSV", Path("/nonexistent/odds.csv"))
    cache = S._MarketCache()
    assert cache.note("Danish Superliga", "2026-08-10",
                      "Silkeborg", "Odense", now=NOW) == ""


def test_market_note_is_display_only():
    """The renderer may read the snapshot store; the pricing path must not.

    Scans executable code only — the docstrings deliberately discuss edge and
    staking in order to explain why this class stays out of both.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(S._MarketCache))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                body[0].value.value = ""
    code = ast.unparse(tree).lower()
    for banned in ("kelly", "edge", "stake"):
        assert banned not in code, \
            f"_MarketCache must not participate in staking ({banned!r} found)"


def test_upcoming_section_is_the_only_caller():
    """If the staking path ever starts consulting the snapshot cache, this
    fails — that is the boundary the class exists to hold."""
    import inspect
    for name in ("_likely_winners_section", "_backed_bets_section"):
        src = inspect.getsource(getattr(S, name))
        assert "_MarketCache" not in src and "market.note" not in src
