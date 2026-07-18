#!/usr/bin/env python3
"""Offline correctness tests for the horse-racing V2 feature/state engine."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horse_racing.experiments import LADDER, run_ladder
from horse_racing.features import (ALL_FAMILIES, FAMILIES, FEATURES,
                                   EntityState, HistoryState, build_feature_frame,
                                   feature_names, going_group)
from horse_racing.registry import BLOCKED_FEATURES, REGISTRY, verify_registry
from horse_racing.schema import load_bundle
from horse_racing.validate import walk_forward
from test_horse_racing import _synthetic


def _state_engine_checks() -> None:
    # Short horizons must forget faster than long horizons.
    state = EntityState((30.0, 365.0))
    t0 = pd.Timestamp("2024-01-01T14:00:00Z")
    state.update(t0, 1.0, True, True, True, True)
    later = t0 + pd.Timedelta(days=120)
    short = state.form(later, 30.0)
    long_ = state.form(later, 365.0)
    assert 0 <= short < long_, (short, long_)
    # Confidence measure decays too, and is 0 for unseen entities.
    assert state.effective_n(later, 30.0) < state.effective_n(t0, 30.0)
    fresh = EntityState((30.0,))
    assert fresh.effective_n(t0, 30.0) == 0.0
    # DNF updates count toward exposure but not toward form.
    dnf_state = EntityState((365.0,))
    dnf_state.update(t0, None, False, False, False, False)
    assert dnf_state.form(t0, 365.0) == 0.0
    assert dnf_state.rate(t0, 365.0, "dnf", 0.05, 5.0) > 0.05
    # Going normalization groups jurisdiction-specific labels.
    assert going_group("Good To Soft") == "soft"
    assert going_group("yielding") == "soft"
    assert going_group("standard to slow") == "aw_standard"
    # Draw prior stays conservative with no evidence.
    empty = HistoryState()
    from horse_racing.features import _draw_effect
    assert _draw_effect(empty, "c1", "turf", 1200, t0, 1.0) == 0.0


def _schema_checks() -> None:
    verify_registry()
    assert feature_names(ALL_FAMILIES) == FEATURES
    # Families are disjoint and each contributes at least one feature.
    seen: set[str] = set()
    for family, cols in FAMILIES.items():
        assert cols, family
        assert not (set(cols) & seen), f"family overlap in {family}"
        seen |= set(cols)
    # The ladder covers every family exactly once in order.
    assert LADDER[-1][1] == ALL_FAMILIES
    assert [name for name, _ in LADDER][0] == "core"
    # Blocked features stay blocked.
    assert "market_odds" in BLOCKED_FEATURES
    assert not set(BLOCKED_FEATURES) & set(REGISTRY)


def _half_life_scale_checks(root: Path) -> None:
    bundle = load_bundle(root)
    base = build_feature_frame(bundle, ["race-040"], include_labels=False)
    slow = build_feature_frame(bundle, ["race-040"], include_labels=False,
                               half_life_scale=2.0)
    merged = base.merge(slow, on="runner_id", suffixes=("_1", "_2"))
    changed = any(not np.allclose(merged[f"{f}_1"], merged[f"{f}_2"])
                  for f in FEATURES)
    assert changed, "half_life_scale must alter decayed state features"
    try:
        build_feature_frame(bundle, ["race-040"], half_life_scale=0.0)
        raise AssertionError("invalid half_life_scale accepted")
    except ValueError:
        pass


def _ablation_and_ladder_checks(root: Path) -> None:
    # Feature-subset walk-forward runs and reports the shared scheme.
    subset = feature_names(("core",))
    _preds, report = walk_forward(root, min_train=30, test_size=14,
                                  features=subset)
    assert report["evaluation"]["features"] == subset
    assert report["evaluation"]["scheme"] == "expanding_chronological_walk_forward"
    assert "slices" in report
    assert report["paired_logloss"]["model_minus_uniform"]["bootstrap"] \
        == "day_clustered"
    # Lockbox withholds races from both fitting and scoring.
    _preds_l, report_l = walk_forward(root, min_train=30, test_size=14,
                                      lockbox_frac=0.2)
    assert report_l["evaluation"]["lockbox_races_withheld"] > 0
    assert report_l["metrics"]["p_model"]["races"] \
        < report["metrics"]["p_model"]["races"]
    # Full ladder with gates produces a report and accepts some rung.
    result = run_ladder(root, min_train=30, test_size=14, skip_hl_search=True)
    assert result["accepted"] in [name for name, _ in LADDER]
    assert result["rungs"][0]["gates"]["promoted"] is True
    for entry in result["rungs"][1:]:
        assert "paired_delta" in entry["gates"]
        assert isinstance(entry["gates"]["promoted"], bool)


def main() -> int:
    _state_engine_checks()
    _schema_checks()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _synthetic(root, n_completed=72)
        synthetic_odds = pd.read_csv(root / "odds.csv")
        synthetic_odds[["race_id", "runner_id", "decimal_odds"]].assign(
            price_type="official_starting_price",
            available_at="2025-01-01T00:00:00+00:00", source="synthetic"
        ).to_csv(root / "starting_prices.csv", index=False)
        _half_life_scale_checks(root)
        _ablation_and_ladder_checks(root)
    print("horse_racing_v2: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
