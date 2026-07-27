"""OddsPapi historical NHL odds ingestion.

Provider docs:
  https://oddspapi.io/us/docs/get-fixtures
  https://oddspapi.io/us/docs/get-historical-odds

OddsPapi historical odds are a per-outcome change log. This adapter reconstructs
complete pre-game market snapshots by carrying forward the latest active price
for both sides of each supported market, then emits nhl.odds_history's strict
provider-neutral schema.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from api_keys import get_key

from . import model as M
from . import odds_history as OH

BASE_URL = "https://api.oddspapi.io/v4"
SPORT_ID = 15
HISTORICAL_START_DATE = date(2026, 1, 1)
DEFAULT_BOOKMAKERS = "pinnacle,bet365,draftkings"
DEFAULT_MARKETS = "ml,total,spread"
DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = DATA_DIR / "api_cache" / "oddspapi"


@dataclass(frozen=True)
class SideSpec:
    side: str
    market_id: str
    outcome_id: str
    line: float | None


@dataclass(frozen=True)
class MarketSpec:
    market: str
    sides: tuple[SideSpec, SideSpec]


@dataclass(frozen=True)
class FetchResult:
    rows: list[dict[str, Any]]
    unmatched: list[dict[str, str]]
    fixtures: int
    history_calls: int


MARKET_SPECS = (
    MarketSpec(
        market="ml",
        sides=(
            SideSpec("home", "151", "151", None),
            SideSpec("away", "151", "152", None),
        ),
    ),
    MarketSpec(
        market="spread",
        sides=(
            SideSpec("home", "15228", "15228", -1.5),
            SideSpec("away", "15240", "15241", 1.5),
        ),
    ),
    MarketSpec(
        market="total",
        sides=(
            SideSpec("over", "15174", "15174", 5.5),
            SideSpec("under", "15174", "15175", 5.5),
        ),
    ),
    MarketSpec(
        market="total",
        sides=(
            SideSpec("over", "15176", "15176", 6.0),
            SideSpec("under", "15176", "15177", 6.0),
        ),
    ),
    MarketSpec(
        market="total",
        sides=(
            SideSpec("over", "15178", "15178", 6.5),
            SideSpec("under", "15178", "15179", 6.5),
        ),
    ),
)


def api_key(explicit: str | None = None) -> str:
    key = (explicit or get_key("oddspapi", env="ODDSPAPI_KEY") or "").strip()
    if not key:
        raise ValueError(
            "No OddsPapi key. Set ODDSPAPI_KEY or add data/api_keys.json "
            "with key 'oddspapi'."
        )
    return key


def fetch_fixtures(*, from_utc: str, to_utc: str, key: str,
                   sport_id: int = SPORT_ID,
                   use_cache: bool = True) -> list[dict[str, Any]]:
    params = {
        "apiKey": key,
        "sportId": str(sport_id),
        "from": from_utc,
        "to": to_utc,
        "language": "en",
    }
    cache = _cache_path("fixtures", f"{sport_id}_{from_utc}_{to_utc}")
    if use_cache and cache.exists():
        try:
            payload = json.loads(cache.read_text())
            return payload if isinstance(payload, list) else []
        except (OSError, json.JSONDecodeError):
            pass
    payload = _get_json("/fixtures", params)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected OddsPapi fixtures payload: {type(payload).__name__}")
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload, indent=2))
    return payload


def fetch_historical_odds(*, fixture_id: str, bookmakers: str, key: str,
                          use_cache: bool = True) -> dict[str, Any]:
    params = {
        "apiKey": key,
        "fixtureId": fixture_id,
        "bookmakers": bookmakers,
    }
    cache = _cache_path("historical", f"{fixture_id}_{bookmakers}")
    if use_cache and cache.exists():
        try:
            payload = json.loads(cache.read_text())
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            pass
    payload = _get_json("/historical-odds", params)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected OddsPapi historical payload: {type(payload).__name__}")
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload, indent=2))
    return payload


def rows_from_history(payload: dict[str, Any], fixture: dict[str, Any],
                      match: dict[str, Any], *,
                      markets: str = DEFAULT_MARKETS) -> list[dict[str, Any]]:
    selected = {M.normalize_market(m.strip()) for m in str(markets).split(",") if m.strip()}
    start_time = _iso_utc(fixture.get("startTime"))
    start_ts = pd.to_datetime(start_time, utc=True, errors="coerce")
    if pd.isna(start_ts):
        return []
    rows: list[dict[str, Any]] = []
    for bookmaker, book_payload in (payload.get("bookmakers") or {}).items():
        book = book_payload if isinstance(book_payload, dict) else {}
        for spec in MARKET_SPECS:
            if spec.market not in selected:
                continue
            rows.extend(_reconstruct_market_rows(
                fixture=fixture,
                match=match,
                bookmaker=str(bookmaker),
                book=book,
                spec=spec,
                start_time=start_time,
                start_ts=pd.Timestamp(start_ts),
            ))
    return rows


def fetch_from_results(results_df: pd.DataFrame, *, key: str,
                       bookmakers: str = DEFAULT_BOOKMAKERS,
                       markets: str = DEFAULT_MARKETS,
                       date_from: str | None = None,
                       date_to: str | None = None,
                       use_cache: bool = True,
                       fixture_sleep_seconds: float = 0.0,
                       history_sleep_seconds: float = 0.0,
                       max_fixtures: int | None = None) -> FetchResult:
    windows = _fixture_windows(results_df, date_from=date_from, date_to=date_to)
    results_index = _results_index(results_df)
    fixtures: list[dict[str, Any]] = []
    for i, (start, end) in enumerate(windows):
        if i and fixture_sleep_seconds > 0:
            time.sleep(fixture_sleep_seconds)
        fixtures.extend(fetch_fixtures(
            from_utc=f"{start.isoformat()}T00:00:00Z",
            to_utc=f"{end.isoformat()}T23:59:59Z",
            key=key,
            use_cache=use_cache,
        ))

    rows: list[dict[str, Any]] = []
    unmatched: list[dict[str, str]] = []
    matched_fixtures = 0
    history_calls = 0
    chunks = _bookmaker_chunks(bookmakers)
    for fixture in _dedupe_fixtures(_nhl_fixtures(fixtures)):
        match = _match_result(fixture, results_index)
        if match is None:
            unmatched.append({
                "provider_fixture_id": str(fixture.get("fixtureId") or ""),
                "home": str(fixture.get("participant1Name") or ""),
                "away": str(fixture.get("participant2Name") or ""),
                "startTime": str(fixture.get("startTime") or ""),
                "reason": "no unique local game match",
            })
            continue
        matched_fixtures += 1
        if max_fixtures is not None and matched_fixtures > max_fixtures:
            break
        for chunk in chunks:
            if history_calls and history_sleep_seconds > 0:
                time.sleep(history_sleep_seconds)
            payload = fetch_historical_odds(
                fixture_id=str(fixture.get("fixtureId") or ""),
                bookmakers=",".join(chunk),
                key=key,
                use_cache=use_cache,
            )
            history_calls += 1
            rows.extend(rows_from_history(payload, fixture, match, markets=markets))
    rows = _dedupe_rows(rows)
    return FetchResult(
        rows=rows,
        unmatched=unmatched,
        fixtures=matched_fixtures,
        history_calls=history_calls,
    )


def write_odds_history(rows: list[dict[str, Any]], path: str | Path = OH.ODDS_HISTORY_CSV,
                       *, append: bool = False) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id", "game_date", "start_time_utc", "captured_at_utc",
        "bookmaker", "market", "side", "line", "decimal_odds", "source",
        "provider_fixture_id", "provider_tournament_id", "provider_market_id",
        "provider_outcome_id", "provider_created_at_utc", "provider_limit",
        "provider_active",
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


def _reconstruct_market_rows(*, fixture: dict[str, Any], match: dict[str, Any],
                             bookmaker: str, book: dict[str, Any],
                             spec: MarketSpec, start_time: str,
                             start_ts: pd.Timestamp) -> list[dict[str, Any]]:
    series_by_side = {
        side.side: _history_series(book, side.market_id, side.outcome_id)
        for side in spec.sides
    }
    if any(not series for series in series_by_side.values()):
        return []

    timeline = sorted({
        ts for series in series_by_side.values()
        for ts, _snap in series
        if ts < start_ts
    })
    rows: list[dict[str, Any]] = []
    last_prices: tuple[float, ...] | None = None
    for captured_ts in timeline:
        side_states: dict[str, tuple[SideSpec, dict[str, Any]]] = {}
        for side_spec in spec.sides:
            latest = _latest_at(series_by_side[side_spec.side], captured_ts)
            if latest is None or not _is_active(latest):
                side_states = {}
                break
            price = _float_or_none(latest.get("price"))
            if price is None or price <= 1.0:
                side_states = {}
                break
            side_states[side_spec.side] = (side_spec, latest)
        if len(side_states) != len(spec.sides):
            continue
        prices = tuple(float(side_states[s.side][1]["price"]) for s in spec.sides)
        if prices == last_prices:
            continue
        last_prices = prices
        captured = _iso_utc(captured_ts)
        for side_spec in spec.sides:
            snap = side_states[side_spec.side][1]
            rows.append({
                "event_id": match["event_id"],
                "game_date": match["game_date"],
                "start_time_utc": start_time,
                "captured_at_utc": captured,
                "bookmaker": bookmaker.strip().lower(),
                "market": spec.market,
                "side": side_spec.side,
                "line": "" if side_spec.line is None else side_spec.line,
                "decimal_odds": float(snap["price"]),
                "source": "oddspapi",
                "provider_fixture_id": str(fixture.get("fixtureId") or ""),
                "provider_tournament_id": str(fixture.get("tournamentId") or ""),
                "provider_market_id": side_spec.market_id,
                "provider_outcome_id": side_spec.outcome_id,
                "provider_created_at_utc": _iso_utc(snap.get("createdAt")),
                "provider_limit": snap.get("limit", ""),
                "provider_active": _is_active(snap),
            })
    return rows


def _history_series(book: dict[str, Any], market_id: str,
                    outcome_id: str) -> list[tuple[pd.Timestamp, dict[str, Any]]]:
    players = (
        book.get("markets", {})
        .get(str(market_id), {})
        .get("outcomes", {})
        .get(str(outcome_id), {})
        .get("players", {})
    )
    raw = players.get("0") if isinstance(players, dict) else None
    if not isinstance(raw, list):
        return []
    series: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for snap in raw:
        if not isinstance(snap, dict):
            continue
        ts = pd.to_datetime(snap.get("createdAt"), utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        series.append((pd.Timestamp(ts), snap))
    return sorted(series, key=lambda item: item[0])


def _latest_at(series: list[tuple[pd.Timestamp, dict[str, Any]]],
               captured_ts: pd.Timestamp) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for ts, snap in series:
        if ts > captured_ts:
            break
        latest = snap
    return latest


def _is_active(snap: dict[str, Any]) -> bool:
    value = snap.get("active", True)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _get_json(path: str, params: dict[str, str]) -> Any:
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "nhl-engine/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _nhl_fixtures(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for fixture in fixtures:
        tournament = str(fixture.get("tournamentName") or "").strip().lower()
        slug = str(fixture.get("tournamentSlug") or "").strip().lower()
        if tournament == "nhl" or slug == "nhl":
            out.append(fixture)
    return out


def _results_index(results_df: pd.DataFrame) -> list[dict[str, Any]]:
    id_col = "game_id" if "game_id" in results_df.columns else "event_id" if "event_id" in results_df.columns else None
    if id_col is None:
        raise ValueError("results_df needs game_id or event_id for OddsPapi mapping")
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


def _match_result(fixture: dict[str, Any], results_index: list[dict[str, Any]]) -> dict[str, Any] | None:
    home_key = _fold(fixture.get("participant1Name"))
    away_key = _fold(fixture.get("participant2Name"))
    start = pd.to_datetime(fixture.get("startTime"), utc=True, errors="coerce")
    if pd.isna(start):
        return None
    utc_day = start.date()
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


def _fixture_windows(results_df: pd.DataFrame, *,
                     date_from: str | None,
                     date_to: str | None) -> list[tuple[date, date]]:
    result_dates = pd.to_datetime(results_df["date"], errors="coerce").dropna().dt.date
    if result_dates.empty:
        return []
    start = max(min(result_dates), HISTORICAL_START_DATE)
    end = max(result_dates)
    if date_from:
        start = pd.to_datetime(date_from, errors="raise").date()
    if date_to:
        end = pd.to_datetime(date_to, errors="raise").date()
    if end < start:
        return []

    windows: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        win_end = min(cur + timedelta(days=8), end)
        windows.append((cur, win_end))
        cur = win_end + timedelta(days=1)
    return windows


def _bookmaker_chunks(bookmakers: str) -> list[list[str]]:
    books = [b.strip().lower() for b in str(bookmakers).split(",") if b.strip()]
    if not books:
        raise ValueError("At least one OddsPapi bookmaker is required")
    return [books[i:i + 3] for i in range(0, len(books), 3)]


def _dedupe_fixtures(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for fixture in fixtures:
        fixture_id = str(fixture.get("fixtureId") or "")
        if not fixture_id or fixture_id in seen:
            continue
        seen.add(fixture_id)
        out.append(fixture)
    return out


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


def _cache_path(kind: str, value: str) -> Path:
    safe = (
        str(value)
        .replace(":", "")
        .replace(",", "-")
        .replace("/", "-")
        .replace("?", "-")
        .replace("&", "-")
    )
    return CACHE_DIR / kind / f"{safe}.json"


def _fold(value: Any) -> str:
    try:
        return M.canonical_team_name(str(value))
    except Exception:  # noqa: BLE001
        text = str(value or "")
        text = text.replace(".", "")
        return " ".join(text.lower().split())


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
    ap = argparse.ArgumentParser(description="Fetch NHL historical odds from OddsPapi")
    ap.add_argument("--results", default=str(DATA_DIR / "results_2025_26.csv"))
    ap.add_argument("--out", default=str(OH.ODDS_HISTORY_CSV))
    ap.add_argument("--api-key", help="OddsPapi key; defaults to ODDSPAPI_KEY/data/api_keys.json")
    ap.add_argument("--bookmakers", default=DEFAULT_BOOKMAKERS,
                    help="Comma-separated OddsPapi bookmaker slugs; historical endpoint is chunked in threes")
    ap.add_argument("--markets", default=DEFAULT_MARKETS,
                    help="Comma-separated local markets to import: ml,total,spread")
    ap.add_argument("--date-from",
                    help="Defaults to max(results min date, 2026-01-01) because OddsPapi history starts Jan 2026")
    ap.add_argument("--date-to")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--fixture-sleep-seconds", type=float, default=2.1)
    ap.add_argument("--history-sleep-seconds", type=float, default=5.1)
    ap.add_argument("--max-fixtures", type=int,
                    help="Development throttle; fetch at most this many matched fixtures")
    args = ap.parse_args()

    results = pd.read_csv(args.results)
    fetched = fetch_from_results(
        results,
        key=api_key(args.api_key),
        bookmakers=args.bookmakers,
        markets=args.markets,
        date_from=args.date_from,
        date_to=args.date_to,
        use_cache=not args.no_cache,
        fixture_sleep_seconds=max(args.fixture_sleep_seconds, 0.0),
        history_sleep_seconds=max(args.history_sleep_seconds, 0.0),
        max_fixtures=args.max_fixtures,
    )
    if not fetched.rows:
        raise SystemExit(
            f"No OddsPapi rows mapped from {fetched.fixtures} matched fixture(s) "
            f"and {fetched.history_calls} historical call(s); unmatched fixtures: {len(fetched.unmatched)}"
        )
    OH.validate_odds_history(pd.DataFrame(fetched.rows), results_df=results)
    path = write_odds_history(fetched.rows, args.out, append=args.append)
    print(
        f"wrote {len(fetched.rows)} row(s) from {fetched.fixtures} fixture(s) "
        f"and {fetched.history_calls} historical call(s) -> {path}; "
        f"unmatched provider fixtures: {len(fetched.unmatched)}"
    )


if __name__ == "__main__":
    main()
