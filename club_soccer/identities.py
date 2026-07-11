"""Stable fixture identities and duplicate reconciliation.

Provider IDs are not a sufficient identity key: the same league match can be
present once from football-data and once from BSD with different IDs. This
module keeps the source CSV clean and prevents double-weighting in fitting and
ambiguous joins in validation/market diagnostics.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd


def _norm(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.casefold()).strip()


def match_identity(row) -> str:
    """Canonical identity: date + competition + home + away."""
    raw_date = row.get("date")
    try:
        date = "" if raw_date is None or bool(pd.isna(raw_date)) else pd.Timestamp(raw_date).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        date = ""
    return "|".join((date, _norm(row.get("competition")),
                     _norm(row.get("home")), _norm(row.get("away"))))


def identity_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(index=df.index, dtype="string")
    return df.apply(match_identity, axis=1)


def duplicate_identity_count(df: pd.DataFrame) -> int:
    keys = identity_series(df)
    return int(keys.duplicated().sum())


def conflicting_score_identity_count(df: pd.DataFrame) -> int:
    """Count identity groups containing more than one distinct score."""
    if df.empty or not {"home_goals", "away_goals"}.issubset(df.columns):
        return 0
    work = df.copy()
    work["_identity"] = identity_series(work)
    scores = work.dropna(subset=["home_goals", "away_goals"]).groupby("_identity") \
        [["home_goals", "away_goals"]].nunique()
    return int(((scores["home_goals"] > 1) | (scores["away_goals"] > 1)).sum())


def _present(value) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return not (isinstance(value, str) and not value.strip())


def _richness(row: pd.Series) -> int:
    """Score rows so a BSD detail row wins over a sparse legacy row."""
    fields = (
        "home_shots", "away_shots", "home_sot", "away_sot",
        "home_corners", "away_corners", "home_xg", "away_xg",
        "home_possession", "away_possession", "round_name", "venue",
        "result_scope", "shootout_winner",
    )
    score = sum(_present(row.get(c)) for c in fields)
    if str(row.get("xg_source") or "").casefold() == "bsd":
        score += 10
    if str(row.get("source") or "").casefold() == "bsd":
        score += 5
    return int(score)


def dedupe_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Reconcile duplicate provider rows by canonical match identity.

    The richest row is the base; missing fields from its siblings are filled
    without replacing non-empty observations. Conflicting scores are left
    visible in the selected row and counted by ``conflicting_score_identity_count``
    for health reporting rather than silently averaged.
    """
    if df.empty or not {"date", "competition", "home", "away"}.issubset(df.columns):
        return df.copy()
    work = df.copy()
    keys = identity_series(work)
    if not keys.duplicated().any():
        return df.copy()
    work["_identity"] = keys
    rows: list[dict] = []
    for _, group in work.groupby("_identity", sort=False, dropna=False):
        ordered = group.copy()
        ordered["_identity_richness"] = group.apply(_richness, axis=1).to_numpy()
        ordered = ordered.sort_values("_identity_richness", ascending=False)
        base = ordered.iloc[0].drop(labels=["_identity", "_identity_richness"]).to_dict()
        for _, sibling in ordered.iloc[1:].iterrows():
            for col in work.columns:
                if col in {"_identity", "_identity_richness"}:
                    continue
                if not _present(base.get(col)) and _present(sibling.get(col)):
                    base[col] = sibling[col]
        rows.append(base)
    out = pd.DataFrame(rows)
    return out.reindex(columns=[c for c in df.columns if c in out.columns] +
                       [c for c in out.columns if c not in df.columns]) \
        .reset_index(drop=True)
