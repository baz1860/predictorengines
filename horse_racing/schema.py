"""Canonical input schema and as-of snapshot helpers.

The engine deliberately consumes provider-neutral CSVs. Provider adapters may
write these tables later, but modelling code never depends on provider names or
retrospective profile pages.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (ALLOWED_CODES, ALLOWED_JURISDICTIONS, DEFAULT_CUTOFF_MINUTES,
                     SURFACE_ALIASES)

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"

RACE_COLUMNS = [
    "race_id", "meeting_date", "scheduled_off_utc", "course_id", "course_name",
    "jurisdiction", "code", "surface", "going", "distance_metres", "race_class",
    "handicap_flag", "prediction_cutoff", "source_updated_at", "record_version",
]
RUNNER_COLUMNS = [
    "race_id", "runner_id", "horse_id", "horse_name", "trainer_id", "trainer_name",
    "jockey_id", "jockey_name", "draw", "age", "weight_carried_kg",
    "official_rating", "declared_status", "non_runner_status", "field_version",
    "source_updated_at", "record_version",
]
RESULT_COLUMNS = [
    "race_id", "runner_id", "finish_position", "completion_status",
    "result_published_at", "result_updated_at", "record_version",
]
ODDS_COLUMNS = [
    "race_id", "runner_id", "market_id", "source", "decimal_odds", "available_size",
    "captured_at", "field_version", "market_status",
]

REQUIRED = {
    "races": {"race_id", "scheduled_off_utc", "course_id", "jurisdiction", "code",
              "surface", "distance_metres", "source_updated_at"},
    "runners": {"race_id", "runner_id", "horse_id", "horse_name", "trainer_id",
                "jockey_id", "declared_status", "source_updated_at"},
    "results": {"race_id", "runner_id", "finish_position", "completion_status",
                "result_published_at"},
    "odds": {"race_id", "runner_id", "market_id", "source", "decimal_odds",
             "captured_at", "market_status"},
}

ACTIVE_STATUSES = {"", "active", "declared", "runner"}
NON_RUNNER_STATUSES = {"non_runner", "non-runner", "nr", "withdrawn", "scratched"}


class DataError(ValueError):
    """Raised when canonical data is unsafe or internally inconsistent."""


@dataclass(frozen=True)
class DataBundle:
    races: pd.DataFrame
    runners: pd.DataFrame
    results: pd.DataFrame
    odds: pd.DataFrame
    data_dir: Path


def _read(path: Path, columns: list[str], required: set[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(required - set(df.columns))
    if missing:
        raise DataError(f"{path.name} missing required columns: {', '.join(missing)}")
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df


def _timestamps(df: pd.DataFrame, columns: list[str], table: str) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out:
            continue
        raw = out[col].astype(str).str.strip()
        parsed = pd.to_datetime(raw.where(raw != ""), utc=True, errors="coerce")
        bad = (raw != "") & parsed.isna()
        if bad.any():
            rows = ", ".join(str(i + 2) for i in out.index[bad][:5])
            raise DataError(f"{table}.{col} has invalid UTC timestamp(s) at CSV row(s) {rows}")
        out[col] = parsed
    return out


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _nonempty_ids(df: pd.DataFrame, columns: list[str], table: str) -> None:
    for col in columns:
        if col not in df:
            continue
        bad = df[col].astype(str).str.strip() == ""
        if bad.any():
            rows = ", ".join(str(i + 2) for i in df.index[bad][:5])
            raise DataError(f"{table}.{col} is blank at CSV row(s) {rows}")


def load_bundle(data_dir: str | Path | None = None) -> DataBundle:
    root = Path(data_dir) if data_dir else DATA_DIR
    races = _read(root / "races.csv", RACE_COLUMNS, REQUIRED["races"])
    runners = _read(root / "runners.csv", RUNNER_COLUMNS, REQUIRED["runners"])
    results = _read(root / "results.csv", RESULT_COLUMNS, REQUIRED["results"])
    odds = _read(root / "odds.csv", ODDS_COLUMNS, REQUIRED["odds"])

    races = _timestamps(races, ["scheduled_off_utc", "prediction_cutoff",
                                "source_updated_at"], "races")
    runners = _timestamps(runners, ["source_updated_at"], "runners")
    results = _timestamps(results, ["result_published_at", "result_updated_at"], "results")
    odds = _timestamps(odds, ["captured_at"], "odds")

    races = _numeric(races, ["distance_metres", "race_class", "handicap_flag",
                             "record_version"])
    runners = _numeric(runners, ["draw", "age", "weight_carried_kg", "official_rating",
                                 "field_version", "record_version"])
    results = _numeric(results, ["finish_position", "record_version"])
    odds = _numeric(odds, ["decimal_odds", "available_size", "field_version"])

    _nonempty_ids(races, ["race_id", "course_id"], "races")
    _nonempty_ids(runners, ["race_id", "runner_id", "horse_id", "trainer_id", "jockey_id"],
                  "runners")
    _nonempty_ids(results, ["race_id", "runner_id"], "results")
    _nonempty_ids(odds, ["race_id", "runner_id"], "odds")

    if not races.empty and races["race_id"].duplicated().any():
        dup = races.loc[races["race_id"].duplicated(), "race_id"].iloc[0]
        raise DataError(f"races.csv has duplicate race_id {dup!r}; race metadata must be canonical")
    if not races.empty:
        races["jurisdiction"] = races["jurisdiction"].astype(str).str.strip().str.upper()
        races["code"] = races["code"].astype(str).str.strip().str.lower()
        raw_surface = races["surface"].astype(str).str.strip().str.lower()
        unknown_surface = sorted(set(raw_surface) - set(SURFACE_ALIASES))
        if unknown_surface:
            raise DataError(f"unsupported surface {unknown_surface[0]!r}")
        races["surface"] = raw_surface.map(SURFACE_ALIASES)
        bad_jurisdiction = sorted(set(races["jurisdiction"]) - ALLOWED_JURISDICTIONS)
        bad_code = sorted(set(races["code"]) - ALLOWED_CODES)
        if bad_jurisdiction:
            raise DataError(f"V1 supports GB/IE only; found {bad_jurisdiction[0]!r}")
        if bad_code:
            raise DataError(f"V1 supports flat racing only; found {bad_code[0]!r}")
    known = set(races["race_id"].astype(str))
    for name, df in (("runners", runners), ("results", results), ("odds", odds)):
        unknown = sorted(set(df["race_id"].astype(str)) - known) if not df.empty else []
        if unknown:
            raise DataError(f"{name}.csv references unknown race_id {unknown[0]!r}")

    if not races.empty:
        missing_ts = races["scheduled_off_utc"].isna() | races["source_updated_at"].isna()
        if missing_ts.any():
            rid = races.loc[missing_ts, "race_id"].iloc[0]
            raise DataError(f"race {rid!r} is missing scheduled_off_utc/source_updated_at")
    if not runners.empty and runners["source_updated_at"].isna().any():
        raise DataError("every runner declaration requires source_updated_at")
    if not results.empty and results["result_published_at"].isna().any():
        raise DataError("every result requires result_published_at")
    if not odds.empty:
        bad_odds = odds["decimal_odds"].isna() | (odds["decimal_odds"] <= 1.0)
        if bad_odds.any():
            raise DataError("every populated odds row requires decimal_odds > 1.0")
        if odds["captured_at"].isna().any():
            raise DataError("every odds row requires captured_at")

    return DataBundle(races=races, runners=runners, results=results, odds=odds,
                      data_dir=root)


def race_cutoff(race: pd.Series, cutoff_minutes: int = DEFAULT_CUTOFF_MINUTES) -> pd.Timestamp:
    explicit = race.get("prediction_cutoff")
    if pd.notna(explicit):
        return pd.Timestamp(explicit)
    return pd.Timestamp(race["scheduled_off_utc"]) - pd.Timedelta(minutes=cutoff_minutes)


def race_row(bundle: DataBundle, race_id: str) -> pd.Series:
    rows = bundle.races[bundle.races["race_id"].astype(str) == str(race_id)]
    if rows.empty:
        raise DataError(f"unknown race_id {race_id!r}")
    return rows.iloc[0]


def runner_snapshot(bundle: DataBundle, race_id: str, cutoff: pd.Timestamp,
                    active_only: bool = True) -> pd.DataFrame:
    rows = bundle.runners[
        (bundle.runners["race_id"].astype(str) == str(race_id))
        & (bundle.runners["source_updated_at"] <= cutoff)
    ].copy()
    if rows.empty:
        return rows
    rows["_version"] = rows["record_version"].fillna(0)
    rows = rows.sort_values(["source_updated_at", "_version"]).drop_duplicates(
        "runner_id", keep="last")
    rows = rows.drop(columns="_version")
    if active_only:
        declared = rows["declared_status"].astype(str).str.strip().str.lower()
        nr = rows["non_runner_status"].astype(str).str.strip().str.lower()
        rows = rows[declared.isin(ACTIVE_STATUSES) & ~nr.isin(NON_RUNNER_STATUSES)]
    if rows["runner_id"].duplicated().any():
        raise DataError(f"race {race_id!r} has duplicate active runner_id")
    if rows["horse_id"].duplicated().any():
        raise DataError(f"race {race_id!r} has duplicate active horse_id")
    return rows.reset_index(drop=True)


def final_results(bundle: DataBundle, race_id: str) -> pd.DataFrame:
    rows = bundle.results[bundle.results["race_id"].astype(str) == str(race_id)].copy()
    if rows.empty:
        return rows
    effective = rows["result_updated_at"].where(rows["result_updated_at"].notna(),
                                                  rows["result_published_at"])
    rows["_effective"] = effective
    rows["_version"] = rows["record_version"].fillna(0)
    return (rows.sort_values(["_effective", "_version"])
            .drop_duplicates("runner_id", keep="last")
            .drop(columns=["_effective", "_version"]).reset_index(drop=True))


def latest_odds_snapshot(bundle: DataBundle, race_id: str, cutoff: pd.Timestamp,
                         source: str | None = None) -> pd.DataFrame:
    rows = bundle.odds[
        (bundle.odds["race_id"].astype(str) == str(race_id))
        & (bundle.odds["captured_at"] <= cutoff)
        & (bundle.odds["market_id"].astype(str).str.lower() == "win")
    ].copy()
    if source:
        rows = rows[rows["source"].astype(str).str.lower() == source.lower()]
    elif not rows.empty:
        latest_by_source = rows.groupby(rows["source"].astype(str))["captured_at"].max()
        source = str(latest_by_source.idxmax())
        rows = rows[rows["source"].astype(str) == source]
    if rows.empty:
        return rows
    # Select the latest state per runner BEFORE checking status. A suspended or
    # closed state invalidates the board; never fall back to an older open quote.
    latest = (rows.sort_values("captured_at").drop_duplicates("runner_id", keep="last")
              .reset_index(drop=True))
    open_ = latest["market_status"].astype(str).str.strip().str.lower().isin(
        {"", "open", "active"})
    if not bool(open_.all()):
        return latest.iloc[0:0].copy()
    return latest


def validate_training_races(bundle: DataBundle, cutoff_minutes: int = 15) -> list[str]:
    """Return completed race IDs safe to use as labelled training examples."""
    valid: list[str] = []
    for race in bundle.races.sort_values("scheduled_off_utc").itertuples(index=False):
        rid = str(race.race_id)
        row = race_row(bundle, rid)
        cutoff = race_cutoff(row, cutoff_minutes)
        runners = runner_snapshot(bundle, rid, cutoff)
        results = final_results(bundle, rid)
        if len(runners) < 2 or results.empty:
            continue
        labelled = runners[["runner_id"]].merge(results, on="runner_id", how="left")
        winners = pd.to_numeric(labelled["finish_position"], errors="coerce") == 1
        if int(winners.sum()) != 1:
            continue
        published = pd.to_datetime(labelled.loc[winners, "result_published_at"], utc=True)
        if published.isna().any() or (published <= cutoff).any():
            raise DataError(f"race {rid!r} has a winner result available before prediction cutoff")
        valid.append(rid)
    return valid


def init_templates(data_dir: str | Path | None = None, overwrite: bool = False) -> list[Path]:
    root = Path(data_dir) if data_dir else DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for name, columns in (("races", RACE_COLUMNS), ("runners", RUNNER_COLUMNS),
                          ("results", RESULT_COLUMNS), ("odds", ODDS_COLUMNS)):
        path = root / f"{name}.csv"
        if path.exists() and not overwrite:
            continue
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        written.append(path)
    return written


def finite_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
