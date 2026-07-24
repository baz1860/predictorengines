#!/usr/bin/env python3
"""Tests for the walk-forward fold cache.

The whole value of a validation gate is that it can fail. A cache with an
incomplete key produces a gate that always passes on stale results while
looking like it ran — worse than no gate at all. Most of these tests are
therefore about INVALIDATION, not about speed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from club_soccer import walkforward_cache as WFC


def _hashes(values):
    return np.array(values, dtype="uint64")


def _opts(**kw):
    base = {"min_train": 200, "league_adjustments": False}
    base.update(kw)
    return base


# ── key sensitivity ───────────────────────────────────────────────────────

def test_identical_inputs_give_identical_keys():
    a = WFC.fold_key("2025-01", _hashes([1, 2, 3]), _hashes([4]), _opts())
    b = WFC.fold_key("2025-01", _hashes([1, 2, 3]), _hashes([4]), _opts())
    assert a == b


def test_key_changes_when_training_data_changes():
    a = WFC.fold_key("2025-01", _hashes([1, 2, 3]), _hashes([4]), _opts())
    b = WFC.fold_key("2025-01", _hashes([1, 2, 9]), _hashes([4]), _opts())
    assert a != b


def test_key_changes_when_a_row_is_added():
    a = WFC.fold_key("2025-01", _hashes([1, 2, 3]), _hashes([4]), _opts())
    b = WFC.fold_key("2025-01", _hashes([1, 2, 3, 4]), _hashes([4]), _opts())
    assert a != b


def test_key_changes_when_test_data_changes():
    """A backfilled score changes the label, so the fold must recompute."""
    a = WFC.fold_key("2025-01", _hashes([1, 2]), _hashes([4]), _opts())
    b = WFC.fold_key("2025-01", _hashes([1, 2]), _hashes([5]), _opts())
    assert a != b


def test_key_is_order_sensitive():
    """Elo is sequential: the same matches in a different order can produce
    different ratings, so an order-insensitive key could give a false hit."""
    a = WFC.fold_key("2025-01", _hashes([1, 2, 3]), _hashes([4]), _opts())
    b = WFC.fold_key("2025-01", _hashes([3, 2, 1]), _hashes([4]), _opts())
    assert a != b


def test_key_changes_with_month():
    a = WFC.fold_key("2025-01", _hashes([1, 2]), _hashes([4]), _opts())
    b = WFC.fold_key("2025-02", _hashes([1, 2]), _hashes([4]), _opts())
    assert a != b


def test_key_changes_with_fit_options():
    a = WFC.fold_key("2025-01", _hashes([1, 2]), _hashes([4]), _opts())
    b = WFC.fold_key("2025-01", _hashes([1, 2]), _hashes([4]),
                     _opts(league_adjustments=True))
    c = WFC.fold_key("2025-01", _hashes([1, 2]), _hashes([4]),
                     _opts(min_train=500))
    assert len({a, b, c}) == 3


def test_key_changes_with_code_fingerprint(monkeypatch):
    a = WFC.fold_key("2025-01", _hashes([1, 2]), _hashes([4]), _opts())
    monkeypatch.setattr(WFC, "code_fingerprint", lambda: "deadbeefdeadbeef")
    b = WFC.fold_key("2025-01", _hashes([1, 2]), _hashes([4]), _opts())
    assert a != b


# ── the fingerprint must cover behaviour-changing data files ──────────────

def test_code_fingerprint_covers_model_and_competition_sources():
    """comp_strength.json and ensemble_weights.json change predictions with no
    code edit at all — the classic way a cache goes quietly stale."""
    assert "model.py" in WFC._CODE_FILES
    assert "competitions.py" in WFC._CODE_FILES
    assert "comp_strength.json" in WFC._DATA_FILES
    assert "ensemble_weights.json" in WFC._DATA_FILES


def test_code_fingerprint_is_stable_within_a_process():
    assert WFC.code_fingerprint() == WFC.code_fingerprint()


def test_fingerprint_detects_a_file_change_without_manual_reset(tmp_path, monkeypatch):
    """Finding 14: an on-disk edit to a tracked file mid-process must change the
    fingerprint, or the cache serves stale folds. The mtime/size signature
    catches it without anyone remembering to call reset_code_fingerprint()."""
    import os

    f = tmp_path / "tracked.py"
    f.write_text("a")
    os.utime(f, (1000, 1000))
    monkeypatch.setattr(WFC, "_tracked_paths", lambda: [f])
    WFC.reset_code_fingerprint()
    fp1 = WFC.code_fingerprint()

    f.write_text("bb")                     # different content AND size
    os.utime(f, (2000, 2000))              # and a later mtime
    fp2 = WFC.code_fingerprint()           # no manual reset
    assert fp1 != fp2


def test_key_columns_cover_the_model_inputs():
    for col in ("home", "away", "home_goals", "away_goals", "date",
                "competition", "neutral", "home_sot", "away_sot",
                "home_corners", "away_corners", "home_xg", "away_xg"):
        assert col in WFC.KEY_COLUMNS


def test_row_hashes_detect_a_changed_score():
    df = pd.DataFrame({"date": ["2025-01-01", "2025-01-02"],
                       "home": ["A", "B"], "away": ["B", "A"],
                       "home_goals": [1.0, 2.0], "away_goals": [0.0, 1.0]})
    before = WFC.row_hashes(df)
    df.loc[0, "home_goals"] = 3.0
    after = WFC.row_hashes(df)
    assert before[0] != after[0]
    assert before[1] == after[1]


# ── store / load ──────────────────────────────────────────────────────────

def test_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(WFC, "CACHE_DIR", tmp_path)
    rows = [{"home": "A", "away": "B", "p_home": 0.5}]
    WFC.store("2025-01", "abc", rows)
    assert WFC.load("2025-01", "abc") == rows


def test_missing_entry_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(WFC, "CACHE_DIR", tmp_path)
    assert WFC.load("2025-01", "nope") is None


def test_corrupt_entry_is_a_miss_not_an_empty_fold(tmp_path, monkeypatch):
    """Returning [] for a corrupt file would silently drop a month from the
    metric — a quiet way to make the gate pass."""
    monkeypatch.setattr(WFC, "CACHE_DIR", tmp_path)
    (tmp_path / "2025-01_abc.json").write_text("{not json")
    assert WFC.load("2025-01", "abc") is None


def test_prune_only_removes_superseded_entries_for_processed_months(tmp_path, monkeypatch):
    """A windowed run must not wipe folds outside its window.

    walk_forward(test_from=...) is used by the eval harness and tuning
    helpers; if those wiped the rest of the cache, the next full run would
    recompute everything.
    """
    monkeypatch.setattr(WFC, "CACHE_DIR", tmp_path)
    WFC.store("2025-01", "current", [{"a": 1}])
    WFC.store("2025-01", "stale", [{"a": 2}])
    WFC.store("2024-06", "untouched", [{"a": 3}])
    removed = WFC.prune({("2025-01", "current")})
    assert removed == 1
    assert WFC.load("2025-01", "current") is not None
    assert WFC.load("2025-01", "stale") is None
    assert WFC.load("2024-06", "untouched") is not None, \
        "a month outside this run's window must survive"


def test_prune_survives_an_unlinkable_file(tmp_path, monkeypatch):
    monkeypatch.setattr(WFC, "CACHE_DIR", tmp_path)
    WFC.store("2025-01", "stale", [{"a": 1}])

    def _boom(self):
        raise OSError("read-only")

    monkeypatch.setattr("pathlib.Path.unlink", _boom)
    WFC.prune({("2025-01", "current")})     # must not raise


# ── equivalence with the uncached path ────────────────────────────────────

def test_cached_and_uncached_walk_forward_agree():
    """The cache must be exact, not approximate."""
    import math

    from club_soccer import validate as V

    cached_rows, cached_m = V.walk_forward(test_from="2026-03-01", use_cache=True)
    live_rows, live_m = V.walk_forward(test_from="2026-03-01", use_cache=False)
    if not live_rows:
        pytest.skip("no fixtures in the test window")

    assert cached_m == live_m
    assert len(cached_rows) == len(live_rows)

    def _eq(x, y):
        if isinstance(x, float) and isinstance(y, float):
            if math.isnan(x) and math.isnan(y):
                return True
        return x == y

    for a, b in zip(cached_rows, live_rows):
        assert all(_eq(a[k], b[k]) for k in a)
