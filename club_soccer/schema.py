"""Club Soccer feature schema + leakage registry.

Every feature-store row is described here ONCE so the feature store, the
market layer and the validation harness all agree on which columns are
point-in-time FEATURES (legal model inputs at prediction time) versus
OUTCOME/TEACHER columns (result, closing odds, settlement) that must never
leak into a historical prediction. Mirrors `wc_v4/schema.py`.
"""
from __future__ import annotations

from typing import Iterable

# Bump when the row shape changes in a way that invalidates cached matrices.
SCHEMA_VERSION = 5

# Canonical fixture shape. Older CSVs are intentionally accepted by
# model.load_fixtures(), but every new fetch/merge writes these columns so
# competition context and BSD detail data are not silently discarded.
# Statuses that VOID a result (postponed / cancelled / abandoned / suspended /
# interrupted). A fixture transitioning into one of these must have its
# scores and match stats cleared, and must never count as played: a postponed
# match that keeps its pre-postponement score silently trains the model on a
# result that officially never happened.
#
# AWARDED (AWD) is deliberately NOT here: an awarded match carries a legal
# official result (typically 3-0) per football-data.org's status model. It
# counts as an official result for settlement — though whether awarded
# scorelines should train the goals model is a separate open question.
VOID_STATUSES = {"POS", "CAN", "ABD", "SUS", "INT"}

# Statuses that carry a standing official result.
OFFICIAL_RESULT_STATUSES = {"FT", "FIN", "AET", "PEN", "AWD"}

# Statuses excluded from MODEL TRAINING. Superset of VOID_STATUSES: an
# awarded (AWD) 3-0 is a legal official result — it settles bets — but it is
# an administrative scoreline, not evidence about either team's goal-scoring
# process, so it must not train the goals/Elo models.
# QUARANTINE: a status we do not recognise. It is deliberately NOT void and NOT
# an official result — we do not know what it means, so the row must not train,
# must not settle, and must not be priced until a human maps it. The original
# provider string is preserved in `status_raw` so it can be mapped later.
QUARANTINE_STATUS = "UNK"
QUARANTINE_STATUSES = {QUARANTINE_STATUS}

# Non-terminal match states: the match is scheduled ("NOT") or in progress
# ("LIV"). A live row can carry a CURRENT (non-final) score, so admitting it to
# training or settlement grades an unfinished match as if it were over. These
# are excluded from BOTH — only a terminal status may train or settle.
NON_TERMINAL_STATUSES = {"NOT", "LIV"}

TRAINING_EXCLUDED_STATUSES = (VOID_STATUSES | {"AWD"} | QUARANTINE_STATUSES
                              | NON_TERMINAL_STATUSES)

# ── canonical status normalization ────────────────────────────────────────
# Providers emit either full words (FINISHED, AWARDED, ABANDONED) or their own
# short codes. Naive first-three-characters truncation MISLABELS the important
# ones — AWARDED->"AWA" (not AWD) and ABANDONED->"ABA" (not ABD) — so awarded
# results never settle and abandoned rows are not recognised as void. Map
# explicitly instead. This is the single source of truth every writer uses;
# `AWA`/`ABA` are healed so any legacy truncated rows re-normalize correctly.
_STATUS_CANON = {
    "FINISHED": "FIN", "FIN": "FIN", "FT": "FT", "FULLTIME": "FT",
    "AET": "AET", "AFTEREXTRATIME": "AET",
    "PEN": "PEN", "PENALTIES": "PEN", "PENALTYSHOOTOUT": "PEN",
    "AWARDED": "AWD", "AWD": "AWD", "AWA": "AWD",
    "NOTSTARTED": "NOT", "NOT": "NOT", "SCHEDULED": "NOT", "TIMED": "NOT",
    "POSTPONED": "POS", "POS": "POS",
    "CANCELLED": "CAN", "CANCELED": "CAN", "CAN": "CAN",
    "ABANDONED": "ABD", "ABD": "ABD", "ABA": "ABD",
    "SUSPENDED": "SUS", "SUS": "SUS",
    "INTERRUPTED": "INT", "INT": "INT",
    "INPLAY": "LIV", "INPROGRESS": "LIV", "LIVE": "LIV", "PAUSED": "LIV",
    "HALFTIME": "LIV", "LIV": "LIV",   # self-map: normalize_status must be idempotent
    # BSD in-play phase labels (the "1st_half" that was quarantining as UNK).
    # All are live-match states: inert for training/settlement, but recognised
    # rather than quarantined. Alnum-stripping already removes the underscore.
    "1STHALF": "LIV", "2NDHALF": "LIV", "FIRSTHALF": "LIV", "SECONDHALF": "LIV",
    "1H": "LIV", "2H": "LIV", "ET": "LIV", "EXTRATIME": "LIV",
    "BREAKTIME": "LIV", "PENALTIESLIVE": "LIV",
}
_STATUS_EMPTY = {"", "NAN", "NONE", "NON", "NULL"}


def normalize_status(raw) -> str:
    """Canonical status code (FIN/FT/AET/PEN/AWD/NOT/POS/CAN/ABD/SUS/INT/LIV).

    Empty/NaN -> "". An UNRECOGNISED value -> QUARANTINE_STATUS ("UNK").

    It must NOT fall back to the first three characters: truncation is exactly
    the bug this map replaced, and it silently manufactures meaning — "POSITIVE"
    and "POSTED" would both become "POS" (a VOID status, clearing results),
    "FINALIZED" would become "FIN" (an OFFICIAL RESULT, admitting it to
    training). Quarantining instead keeps an unknown status inert everywhere
    until a human maps it."""
    if raw is None:
        return ""
    s = str(raw).strip().upper()
    if s in _STATUS_EMPTY:
        return ""
    key = "".join(ch for ch in s if ch.isalnum())
    if key in _STATUS_CANON:
        return _STATUS_CANON[key]
    return QUARANTINE_STATUS

# Result/stat columns cleared on a void transition.
RESULT_COLUMNS = [
    "home_goals", "away_goals", "home_goals_ht", "away_goals_ht",
    "home_goals_ft", "away_goals_ft",
    "extra_time_home_goals", "extra_time_away_goals",
    "shootout_home", "shootout_away", "shootout_winner",
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners", "home_xg", "away_xg", "xg_source",
    "home_possession", "away_possession",
    "home_yellow_cards", "away_yellow_cards",
    "home_red_cards", "away_red_cards",
]

FIXTURE_COLUMNS = [
    # kickoff_utc: full kickoff instant (ISO, UTC) — `date` is a derived
    # date-only display/partition field. Staking-time freshness checks should
    # prefer kickoff_utc; legacy rows without it fall back to date-only.
    "fixture_id", "kickoff_utc", "date", "season", "competition", "competition_id",
    "country", "type",
    # Provider IDs are source-local; *_club_id is the deterministic
    # cross-provider identity generated after canonical name resolution.
    "home_id", "home_club_id", "home", "away_id", "away_club_id", "away",
    # status_raw: the provider's original string, preserved whenever the status
    # could not be mapped (status == QUARANTINE_STATUS) so an operator can add
    # the mapping later. Empty for recognised statuses.
    "home_goals", "away_goals", "status", "status_raw", "result_scope", "neutral",
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners", "home_xg", "away_xg", "xg_source",
    "home_possession", "away_possession",
    "home_yellow_cards", "away_yellow_cards",
    "home_red_cards", "away_red_cards",
    "home_goals_ht", "away_goals_ht", "home_goals_ft", "away_goals_ft",
    "extra_time_home_goals", "extra_time_away_goals",
    "shootout_home", "shootout_away", "shootout_winner",
    "round_name", "round_number", "group_name", "venue",
]

FIXTURE_NUMERIC_COLUMNS = [
    "home_goals", "away_goals", "neutral", "home_shots", "away_shots",
    "home_sot", "away_sot", "home_corners", "away_corners", "home_xg",
    "away_xg", "home_possession", "away_possession", "home_yellow_cards",
    "away_yellow_cards", "home_red_cards", "away_red_cards", "home_goals_ht",
    "away_goals_ht", "home_goals_ft", "away_goals_ft",
    "extra_time_home_goals", "extra_time_away_goals", "shootout_home",
    "shootout_away", "round_number",
]

# Provenance fields every row carries.
PROVENANCE_COLUMNS = ["asof", "source", "fetched_at", "schema_version"]

# Identity / descriptive columns — safe, but not model inputs on their own.
ID_COLUMNS = ["event_id", "match_date", "home", "away", "competition",
              "season", "neutral"]

# Point-in-time FEATURES: everything legal to feed a prediction made at `asof`.
# Each is knowable strictly before kickoff from data dated < asof.
FEATURE_COLUMNS = [
    "elo_h", "elo_a", "elo_diff",
    "lam_h", "lam_a",
    "p_model_h", "p_model_d", "p_model_a",
    "rest_days_h", "rest_days_a",
    "matches_7d_h", "matches_7d_a",
    "matches_14d_h", "matches_14d_a",
    "matches_30d_h", "matches_30d_a",
    "xi_load_7d_h", "xi_load_7d_a",
    "xi_load_14d_h", "xi_load_14d_a",
    "xi_load_30d_h", "xi_load_30d_a",
]

# OUTCOME / TEACHER columns. Legal as *labels* and as a *teacher* signal
# during training/validation, but injecting any of these into
# FEATURE_COLUMNS is leakage.
OUTCOME_COLUMNS = [
    "home_goals", "away_goals", "result",
    "p_close_h", "p_close_d", "p_close_a",
    "odds_close_h", "odds_close_d", "odds_close_a",
    "odds_close_over25", "odds_close_under25",
]

_FEATURE_SET = set(FEATURE_COLUMNS)
_OUTCOME_SET = set(OUTCOME_COLUMNS)


class LeakageError(AssertionError):
    """Raised when an outcome/teacher column is used as a model feature."""


def assert_no_leakage(feature_cols: Iterable[str]) -> None:
    """Reject any attempt to treat an OUTCOME/TEACHER column as a feature."""
    cols = list(feature_cols)
    bad = [c for c in cols if c in _OUTCOME_SET]
    if bad:
        raise LeakageError(
            f"outcome/teacher columns used as features (leakage): {sorted(set(bad))}. "
            "Closing odds and results are teachers, not inputs."
        )


def feature_columns(df_columns: Iterable[str]) -> list[str]:
    """The subset of `df_columns` that are legal point-in-time features."""
    present = [c for c in df_columns if c in _FEATURE_SET]
    assert_no_leakage(present)  # belt and braces
    return present


def is_outcome_column(name: str) -> bool:
    return name in _OUTCOME_SET
