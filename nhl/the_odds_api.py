"""The Odds API historical NHL odds ingestion.

Provider docs:
  https://the-odds-api.com/liveapi/guides/v4/#get-historical-odds

The Odds API event IDs are provider IDs, not NHL game IDs. This adapter maps
provider events back to local NHL results by home/away teams and nearby game
date, then emits rows in nhl.odds_history's strict schema.
"""
from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from api_keys import get_key

from . import model as M
from . import odds_history as OH

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "icehockey_nhl"
DEFAULT_MARKETS = "h2h,spreads,totals"
DEFAULT_REGIONS = "us"
DEFAULT_SNAPSHOT_TIME = "12:00:00Z"
DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = DATA_DIR / "api_cache" / "the_odds_api"

MARKET_MAP = {
    "h2h": "ml",
    "spreads": "spread",
    "totals": "total",
}


@dataclass(frozen=True)
class FetchResult:
    rows: list[dict[str, Any]]
    unmatched: list[dict[str, str]]
    snapshots: int


def api_key(explicit: str | None = None) -> str:
    key = (explicit or get_key("the-odds-api", env="THE_ODDS_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "No The Odds API key. Set THE_ODDS_API_KEY or add data/api_keys.json "
            "with key 'the-odds-api'."
        )
    return key


def fetch_historical_snapshot(*, snapshot_at_utc: str, key: str,
                              regions: str = DEFAULT_REGIONS,
                              markets: str = DEFAULT_MARKETS,
                              use_cache: bool = True) -> dict[str, Any]:
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "date": snapshot_at_utc,
    }
    cache = _cache_path(snapshot_at_utc, regions, markets)
    if use_cache and cache.exists():
        try:
            return json.loads(cache.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    url = f"{BASE_URL}/historical/sports/{SPORT_KEY}/odds?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "nhl-engine/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload, indent=2))
    return payload


def rows_from_snapshot(payload: dict[str, Any],
                       results_df: pd.DataFrame) -> tuple[list[dict[str, Any]],
                                                          list[dict[str, str]]]:
    snapshot_ts = str(payload.get("timestamp") or "")
    requested_ts = str(payload.get("_snapshot_requested_at_utc") or "")
    data = payload.get("data") or []
    results_index = _results_index(results_df)
    rows: list[dict[str, Any]] = []
    unmatched: list[dict[str, str]] = []
    for event in data:
        match = _match_result(event, results_index)
        if match is None:
            unmatched.append({
                "provider_event_id": str(event.get("id") or ""),
                "home": str(event.get("home_team") or ""),
                "away": str(event.get("away_team") or ""),
                "commence_time": str(event.get("commence_time") or ""),
                "reason": "no unique local game match",
            })
            continue
        rows.extend(_event_rows(event, match, snapshot_ts, requested_ts))
    return rows, unmatched


def fetch_from_results_dates(results_df: pd.DataFrame, *, key: str,
                             regions: str = DEFAULT_REGIONS,
                             markets: str = DEFAULT_MARKETS,
                             snapshot_time: str = DEFAULT_SNAPSHOT_TIME,
                             date_from: str | None = None,
                             date_to: str | None = None,
                             use_cache: bool = True) -> FetchResult:
    dates = _snapshot_dates(results_df, date_from=date_from, date_to=date_to)
    rows: list[dict[str, Any]] = []
    unmatched: list[dict[str, str]] = []
    for day in dates:
        snapshot_at = _snapshot_timestamp(day, snapshot_time)
        payload = fetch_historical_snapshot(
            snapshot_at_utc=snapshot_at,
            key=key,
            regions=regions,
            markets=markets,
            use_cache=use_cache,
        )
        payload["_snapshot_requested_at_utc"] = snapshot_at
        r, u = rows_from_snapshot(payload, results_df)
        rows.extend(r)
        unmatched.extend(u)
    rows = _dedupe_rows(rows)
    return FetchResult(rows=rows, unmatched=unmatched, snapshots=len(dates))


def write_odds_history(rows: list[dict[str, Any]], path: str | Path = OH.ODDS_HISTORY_CSV,
                       *, append: bool = False) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id", "game_date", "start_time_utc", "captured_at_utc",
        "bookmaker", "market", "side", "line", "decimal_odds", "source",
        "provider_event_id", "provider_bookmaker_title", "provider_last_update_utc",
        "snapshot_requested_at_utc", "snapshot_timestamp_utc",
    ]
    existing: list[dict[str, Any]] = []
    if append and out.exists():
        with out.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    combined = _dedupe_rows([*existing, *rows])
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return out


def _event_rows(event: dict[str, Any], match: dict[str, Any],
                snapshot_ts: str, requested_ts: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    home = str(event.get("home_team") or "")
    away = str(event.get("away_team") or "")
    start_time = _iso_utc(event.get("commence_time"))
    for book in event.get("bookmakers") or []:
        bookmaker = str(book.get("key") or book.get("title") or "").strip().lower()
        if not bookmaker:
            continue
        book_title = str(book.get("title") or "")
        for market in book.get("markets") or []:
            provider_market = str(market.get("key") or "")
            local_market = MARKET_MAP.get(provider_market)
            if not local_market:
                continue
            provider_last_update = str(market.get("last_update") or book.get("last_update") or "")
            for outcome in market.get("outcomes") or []:
                mapped = _map_outcome(local_market, outcome, home, away)
                if mapped is None:
                    continue
                side, line = mapped
                try:
                    odds = float(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                if odds <= 1.0:
                    continue
                rows.append({
                    "event_id": match["event_id"],
                    "game_date": match["game_date"],
                    "start_time_utc": start_time,
                    "captured_at_utc": snapshot_ts,
                    "bookmaker": bookmaker,
                    "market": local_market,
                    "side": side,
                    "line": "" if line is None else line,
                    "decimal_odds": odds,
                    "source": "the_odds_api",
                    "provider_event_id": str(event.get("id") or ""),
                    "provider_bookmaker_title": book_title,
                    "provider_last_update_utc": provider_last_update,
                    "snapshot_requested_at_utc": requested_ts,
                    "snapshot_timestamp_utc": snapshot_ts,
                })
    return rows


def _map_outcome(local_market: str, outcome: dict[str, Any],
                 home: str, away: str) -> tuple[str, float | None] | None:
    name = str(outcome.get("name") or "")
    folded = _fold(name)
    if local_market in {"ml", "spread"}:
        if folded == _fold(home):
            side = "home"
        elif folded == _fold(away):
            side = "away"
        else:
            return None
        point = outcome.get("point")
        line = None if local_market == "ml" else _float_or_none(point)
        return side, line
    if local_market == "total":
        if folded == "over":
            side = "over"
        elif folded == "under":
            side = "under"
        else:
            return None
        return side, _float_or_none(outcome.get("point"))
    return None


def _results_index(results_df: pd.DataFrame) -> list[dict[str, Any]]:
    id_col = "game_id" if "game_id" in results_df.columns else "event_id" if "event_id" in results_df.columns else None
    if id_col is None:
        raise ValueError("results_df needs game_id or event_id for The Odds API mapping")
    out = []
    for r in results_df.itertuples(index=False):
        row = r._asdict()
        out.append({
            "event_id": OH.id_key(row.get(id_col)),
            "game_date": str(row.get("date") or "")[:10],
            "home_key": _fold(row.get("home")),
            "away_key": _fold(row.get("away")),
        })
    return out


def _match_result(event: dict[str, Any], results_index: list[dict[str, Any]]) -> dict[str, Any] | None:
    home_key = _fold(event.get("home_team"))
    away_key = _fold(event.get("away_team"))
    commence = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
    if pd.isna(commence):
        return None
    utc_day = commence.date()
    possible_dates = {
        utc_day.isoformat(),
        (utc_day - timedelta(days=1)).isoformat(),
        (utc_day + timedelta(days=1)).isoformat(),
    }
    candidates = [
        row for row in results_index
        if row["home_key"] == home_key
        and row["away_key"] == away_key
        and row["game_date"] in possible_dates
    ]
    exact = [row for row in candidates if row["game_date"] == utc_day.isoformat()]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _snapshot_dates(results_df: pd.DataFrame, *,
                    date_from: str | None,
                    date_to: str | None) -> list[date]:
    dates = pd.to_datetime(results_df["date"], errors="coerce").dropna().dt.date
    if date_from:
        lo = pd.to_datetime(date_from, errors="raise").date()
        dates = dates[dates >= lo]
    if date_to:
        hi = pd.to_datetime(date_to, errors="raise").date()
        dates = dates[dates <= hi]
    return sorted(set(dates))


def _snapshot_timestamp(day: date, snapshot_time: str) -> str:
    value = str(snapshot_time).strip()
    if value.endswith("Z"):
        value = value[:-1]
    if len(value.split(":")) == 2:
        value = f"{value}:00"
    return f"{day.isoformat()}T{value}Z"


def _cache_path(snapshot_at_utc: str, regions: str, markets: str) -> Path:
    safe = (
        f"{SPORT_KEY}_{snapshot_at_utc}_{regions}_{markets}"
        .replace(":", "")
        .replace(",", "-")
        .replace("/", "-")
    )
    return CACHE_DIR / f"{safe}.json"


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out = []
    for row in rows:
        key = (
            OH.id_key(row.get("event_id")),
            str(row.get("bookmaker") or ""),
            str(row.get("captured_at_utc") or ""),
            str(row.get("market") or ""),
            str(row.get("side") or ""),
            str(row.get("line") or ""),
            str(row.get("decimal_odds") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _fold(value: Any) -> str:
    try:
        return M.canonical_team_name(str(value))
    except Exception:  # noqa: BLE001
        return " ".join(str(value or "").lower().replace(".", "").split())


def _float_or_none(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_utc(value: Any) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch NHL historical odds from The Odds API")
    ap.add_argument("--results", default=str(DATA_DIR / "results_2025_26.csv"))
    ap.add_argument("--out", default=str(OH.ODDS_HISTORY_CSV))
    ap.add_argument("--api-key", help="The Odds API key; defaults to THE_ODDS_API_KEY/data/api_keys.json")
    ap.add_argument("--regions", default=DEFAULT_REGIONS)
    ap.add_argument("--markets", default=DEFAULT_MARKETS)
    ap.add_argument("--snapshot-time", default=DEFAULT_SNAPSHOT_TIME,
                    help="UTC time queried for each game date, e.g. 12:00:00Z")
    ap.add_argument("--date-from")
    ap.add_argument("--date-to")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    results = pd.read_csv(args.results)
    fetched = fetch_from_results_dates(
        results,
        key=api_key(args.api_key),
        regions=args.regions,
        markets=args.markets,
        snapshot_time=args.snapshot_time,
        date_from=args.date_from,
        date_to=args.date_to,
        use_cache=not args.no_cache,
    )
    if not fetched.rows:
        raise SystemExit(
            f"No The Odds API rows mapped from {fetched.snapshots} snapshot(s); "
            f"unmatched events: {len(fetched.unmatched)}"
        )
    # Validate before writing; this enforces exact NHL game_id mapping and pre-game timestamps.
    OH.validate_odds_history(pd.DataFrame(fetched.rows), results_df=results)
    path = write_odds_history(fetched.rows, args.out, append=args.append)
    print(
        f"wrote {len(fetched.rows)} row(s) from {fetched.snapshots} snapshot(s) -> {path}; "
        f"unmatched provider events: {len(fetched.unmatched)}"
    )


if __name__ == "__main__":
    main()
