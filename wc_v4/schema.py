"""V4 feature schema + leakage registry (the spine of M1).

Every V4 feature row is described here ONCE, so the feature store, the market
model, the availability layer and the validation harness all agree on:

  * which columns are point-in-time FEATURES (legal model inputs at prediction
    time), and
  * which columns are OUTCOME / TEACHER columns (the result, the *closing* line,
    settlement P&L, CLV) that must NEVER be fed to a point-in-time prediction.

This separation is the whole point of guardrail #2 ("no future injury status,
closing line, lineup, result or post-event statistic can leak into a historical
prediction") and guardrail #3 ("closing lines are a teacher, not an oracle").

`assert_no_leakage()` is the single chokepoint the leakage tests drive.
"""
from __future__ import annotations

from typing import Iterable

# Bump when the row shape changes in a way that invalidates cached matrices.
SCHEMA_VERSION = 2

# Provenance fields every row carries (M1 acceptance criterion).
PROVENANCE_COLUMNS = ["asof", "event_id", "source", "fetched_at", "schema_version"]

# Identity / descriptive columns — safe, but not model inputs on their own.
ID_COLUMNS = ["match_date", "home", "away", "competition", "neutral"]

# Point-in-time FEATURES: everything legal to feed a prediction made at `asof`.
# Each is knowable strictly before kickoff from data dated < asof.
FEATURE_COLUMNS = [
    # fundamental strength (predictor.compute_elo writes PRE-match elo -> leak-free)
    "elo_h", "elo_a", "elo_diff",
    "lam_h", "lam_a",                 # expected goals from the as-of goal model
    "p_model_h", "p_model_d", "p_model_a",
    # schedule / fatigue (computed only from matches dated < the event)
    "rest_days_h", "rest_days_a", "congestion_h", "congestion_a",
    # market state KNOWN before kickoff: opening + current odds only (NOT close)
    "odds_open_h", "odds_open_d", "odds_open_a",
    "odds_curr_h", "odds_curr_d", "odds_curr_a",
    "p_market_h", "p_market_d", "p_market_a",
    "move_open_curr_h", "move_open_curr_d", "move_open_curr_a",
    "book_dispersion",
    # availability (M3) — point-in-time from absences known at asof
    "avail_adj_h", "avail_adj_a", "lineup_conf_h", "lineup_conf_a",
    "confirmed_xi_power_h", "confirmed_xi_power_a",
    "bench_power_h", "bench_power_a",
    "formation_known_h", "formation_known_a",
    "market_dispersion_h", "market_dispersion_d", "market_dispersion_a",
]

# OUTCOME / TEACHER columns. Legal as *labels* and as a *teacher* signal during
# training/validation, but injecting any of these into FEATURE_COLUMNS is leakage.
OUTCOME_COLUMNS = [
    "home_score", "away_score", "result",         # the result itself
    "shots_h", "shots_a", "shots_on_target_h", "shots_on_target_a",
    "corners_h", "corners_a", "cards_h", "cards_a", "xg_h", "xg_a",
    "odds_close_h", "odds_close_d", "odds_close_a",  # the CLOSING line (teacher)
    "p_close_h", "p_close_d", "p_close_a",
    "clv", "settled_pnl",                          # post-event
]

_FEATURE_SET = set(FEATURE_COLUMNS)
_OUTCOME_SET = set(OUTCOME_COLUMNS)


class LeakageError(AssertionError):
    """Raised when an outcome/teacher column is used as a model feature."""


def assert_no_leakage(feature_cols: Iterable[str]) -> None:
    """Reject any attempt to treat an OUTCOME/TEACHER column as a feature.

    This is the chokepoint the leakage tests drive: a walk-forward harness asks
    for its feature columns, and if a future-only column (the result, the closing
    line, CLV, settlement P&L) has crept in, we fail loudly here rather than
    silently training on the future.
    """
    cols = list(feature_cols)
    bad = [c for c in cols if c in _OUTCOME_SET]
    if bad:
        raise LeakageError(
            f"outcome/teacher columns used as features (leakage): {sorted(set(bad))}. "
            "Closing line, result, CLV and settlement P&L are teachers, not inputs."
        )


def feature_columns(df_columns: Iterable[str]) -> list[str]:
    """The subset of `df_columns` that are legal point-in-time features.

    Used by the validation harness to assemble a model matrix and by the leakage
    test, which adds a poisoned column and confirms it is excluded / rejected.
    """
    present = [c for c in df_columns if c in _FEATURE_SET]
    assert_no_leakage(present)  # belt and braces
    return present


def is_outcome_column(name: str) -> bool:
    return name in _OUTCOME_SET
