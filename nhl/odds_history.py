"""Strict historical odds snapshots for NHL ROI backtests."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from . import model as M

DATA_DIR = Path(__file__).resolve().parent / "data"
ODDS_HISTORY_CSV = DATA_DIR / "odds_history.csv"

REQUIRED_COLUMNS = {
    "event_id", "game_date", "start_time_utc", "captured_at_utc",
    "bookmaker", "market", "side", "line", "decimal_odds", "source",
}
VALID_SIDES = {
    "ml": {"home", "away"},
    "spread": {"home", "away"},
    "total": {"over", "under"},
}


def id_key(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _line_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if text == "":
        return None
    try:
        x = float(text)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def line_key(market: str, line: float | None) -> str:
    market = M.normalize_market(market)
    if market == "ml":
        return ""
    if line is None or not math.isfinite(float(line)):
        raise ValueError(f"{market} quote needs a numeric line")
    value = abs(float(line)) if market == "spread" else float(line)
    return f"{value:.2f}"


def quote_group_key(row: pd.Series) -> tuple[str, str, pd.Timestamp, str, str]:
    return (
        id_key(row["event_id"]),
        str(row["bookmaker"]),
        row["_captured_at_utc"],
        str(row["market"]),
        str(row["_line_key"]),
    )


def quote_family_key(row: pd.Series) -> tuple[str, str, str, str]:
    return (
        id_key(row["event_id"]),
        str(row["bookmaker"]),
        str(row["market"]),
        str(row["_line_key"]),
    )


def load_odds_history(path: str | Path = ODDS_HISTORY_CSV,
                      results_df: pd.DataFrame | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NHL odds history file not found: {p}")
    return validate_odds_history(pd.read_csv(p), results_df=results_df, source_path=p)


def validate_odds_history(df: pd.DataFrame, *,
                          results_df: pd.DataFrame | None = None,
                          source_path: str | Path = "odds_history") -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{source_path} missing required columns: {missing}")

    out = df.copy()
    if "_row_num" not in out.columns:
        out["_row_num"] = out.index + 2
    out["event_id"] = out["event_id"].map(id_key)
    out["bookmaker"] = out["bookmaker"].fillna("").astype(str).str.strip().str.lower()
    out["source"] = out["source"].fillna("").astype(str).str.strip()
    out["side"] = out["side"].fillna("").astype(str).str.lower().str.strip()
    out["decimal_odds"] = pd.to_numeric(out["decimal_odds"], errors="coerce")
    out["_line_value"] = out["line"].map(_line_number)
    out["_game_date_key"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["_start_time_utc"] = pd.to_datetime(out["start_time_utc"], utc=True, errors="coerce")
    out["_captured_at_utc"] = pd.to_datetime(out["captured_at_utc"], utc=True, errors="coerce")

    errors: list[str] = []
    _collect_blank_errors(out, "event_id", errors)
    _collect_blank_errors(out, "bookmaker", errors)
    _collect_blank_errors(out, "source", errors)
    _collect_bad_time_errors(out, "_game_date_key", "game_date", errors)
    _collect_bad_time_errors(out, "_start_time_utc", "start_time_utc", errors)
    _collect_bad_time_errors(out, "_captured_at_utc", "captured_at_utc", errors)

    bad_odds = out[out["decimal_odds"].isna() | (out["decimal_odds"] <= 1.0)]
    if not bad_odds.empty:
        errors.append(f"decimal_odds must be > 1.0 on row(s): {_rows(bad_odds)}")

    market_errors: list[int] = []
    normalized_markets: list[str] = []
    for row in out.itertuples(index=False):
        try:
            normalized_markets.append(M.normalize_market(getattr(row, "market")))
        except ValueError:
            market_errors.append(int(getattr(row, "_row_num")))
            normalized_markets.append("")
    out["market"] = normalized_markets
    if market_errors:
        errors.append(f"market invalid on row(s): {market_errors}")

    bad_sides = [
        int(r["_row_num"])
        for r in out.to_dict("records")
        if r["market"] in VALID_SIDES and str(r["side"]) not in VALID_SIDES[str(r["market"])]
    ]
    if bad_sides:
        errors.append(f"side invalid on row(s): {bad_sides}")

    out["_line_key"] = ""
    for idx, row in out.iterrows():
        market = str(row["market"])
        line = row["_line_value"]
        row_num = int(row["_row_num"])
        if market == "ml":
            if line is not None and math.isfinite(float(line)):
                errors.append(f"moneyline must have blank line on row {row_num}")
            continue
        if line is None or not math.isfinite(float(line)):
            errors.append(f"{market} must have numeric line on row {row_num}")
            continue
        out.at[idx, "_line_key"] = line_key(market, line)

    not_pregame = out[
        out["_captured_at_utc"].notna()
        & out["_start_time_utc"].notna()
        & (out["_captured_at_utc"] >= out["_start_time_utc"])
    ]
    if not not_pregame.empty:
        errors.append(f"captured_at_utc must be before start_time_utc on row(s): {_rows(not_pregame)}")

    if errors:
        raise ValueError("; ".join(errors))

    out["_quote_group_key"] = out.apply(quote_group_key, axis=1)
    out["_quote_family_key"] = out.apply(quote_family_key, axis=1)
    _validate_quote_groups(out)
    if results_df is not None:
        _validate_against_results(out, results_df)
    return out.reset_index(drop=True)


def quote_groups(df: pd.DataFrame):
    return df.groupby("_quote_group_key", sort=False)


def devig_group(group: pd.DataFrame) -> dict[int, float]:
    inv = 1.0 / group["decimal_odds"].astype(float)
    overround = float(inv.sum())
    if overround <= 0:
        raise ValueError("Cannot de-vig quote group with non-positive overround")
    return {int(idx): float(prob) for idx, prob in (inv / overround).items()}


def latest_pre_game_quotes(df: pd.DataFrame, *,
                           hours_before_start: float | None = None) -> pd.DataFrame:
    if hours_before_start is not None and hours_before_start < 0:
        raise ValueError("hours_before_start must be >= 0")
    eligible = df[df["_captured_at_utc"] < df["_start_time_utc"]].copy()
    if hours_before_start is not None:
        cutoff = eligible["_start_time_utc"] - pd.to_timedelta(float(hours_before_start), unit="h")
        eligible = eligible[eligible["_captured_at_utc"] <= cutoff].copy()
    if eligible.empty:
        return eligible

    selected_groups: set[tuple[Any, ...]] = set()
    for _family, group in eligible.groupby("_quote_family_key", sort=False):
        latest_capture = group["_captured_at_utc"].max()
        for key in group.loc[group["_captured_at_utc"] == latest_capture, "_quote_group_key"].unique():
            selected_groups.add(key)
    return eligible[eligible["_quote_group_key"].isin(selected_groups)].reset_index(drop=True)


def _validate_quote_groups(df: pd.DataFrame) -> None:
    errors = []
    for key, group in df.groupby("_quote_group_key", sort=False):
        market = str(group.iloc[0]["market"])
        expected = VALID_SIDES[market]
        side_counts = group["side"].value_counts()
        duplicate_sides = sorted(side for side, count in side_counts.items() if int(count) > 1)
        if duplicate_sides:
            errors.append(f"duplicate side(s) {duplicate_sides} in group {key} on row(s): {_rows(group)}")
        sides = set(group["side"].astype(str))
        if sides != expected:
            errors.append(
                f"incomplete {market} group {key} on row(s): {_rows(group)}; "
                f"missing={sorted(expected - sides)}, extra={sorted(sides - expected)}"
            )
        if market == "spread" and {"home", "away"}.issubset(sides):
            home_line = float(group.loc[group["side"] == "home", "_line_value"].iloc[0])
            away_line = float(group.loc[group["side"] == "away", "_line_value"].iloc[0])
            if abs(home_line + away_line) > 1e-9:
                errors.append(f"spread home/away lines must be opposite in group {key}")
    if errors:
        raise ValueError("; ".join(errors))


def _validate_against_results(odds: pd.DataFrame, results: pd.DataFrame) -> None:
    id_col = "game_id" if "game_id" in results.columns else "event_id" if "event_id" in results.columns else None
    if id_col is None:
        raise ValueError("results_df needs game_id or event_id to validate odds history")
    games = results.copy()
    games["_event_id_key"] = games[id_col].map(id_key)
    games["_date_key"] = pd.to_datetime(games["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    date_by_id = dict(zip(games["_event_id_key"], games["_date_key"]))
    unknown = sorted(set(odds["event_id"]) - set(date_by_id))
    if unknown:
        raise ValueError(f"odds history references unknown event_id(s): {unknown[:10]}")
    bad_dates = odds[odds.apply(lambda r: date_by_id.get(r["event_id"]) != r["_game_date_key"], axis=1)]
    if not bad_dates.empty:
        raise ValueError(f"odds history game_date mismatches results on row(s): {_rows(bad_dates)}")


def _collect_blank_errors(df: pd.DataFrame, column: str, errors: list[str]) -> None:
    bad = df[df[column].astype(str).str.strip() == ""]
    if not bad.empty:
        errors.append(f"{column} is blank on row(s): {_rows(bad)}")


def _collect_bad_time_errors(df: pd.DataFrame, parsed_col: str, label: str,
                             errors: list[str]) -> None:
    bad = df[df[parsed_col].isna()]
    if not bad.empty:
        errors.append(f"{label} invalid on row(s): {_rows(bad)}")


def _rows(df: pd.DataFrame) -> list[int]:
    return df["_row_num"].astype(int).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate NHL historical odds snapshots")
    ap.add_argument("--odds-history", default=str(ODDS_HISTORY_CSV))
    ap.add_argument("--results", default=str(DATA_DIR / "results_2025_26.csv"))
    args = ap.parse_args()

    results = pd.read_csv(args.results)
    odds = load_odds_history(args.odds_history, results_df=results)
    latest = latest_pre_game_quotes(odds)
    print(f"validated {len(odds)} odds row(s); {len(latest)} latest pre-game row(s)")


if __name__ == "__main__":
    main()
