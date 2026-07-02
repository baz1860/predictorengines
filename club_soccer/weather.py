#!/usr/bin/env python3
"""Weather features (totals-oriented) from Open-Meteo — free, no key.

Backfill: archive-api.open-meteo.com/v1/archive for played fixtures since
2022-08 with a venues.csv row, taking the 15:00 local value (kickoff time
isn't in fixtures.csv — this is a documented approximation).
Forecast: api.open-meteo.com/v1/forecast for upcoming fixtures <=16 days out
(the free forecast endpoint's horizon).

Output data/weather.csv: fixture_id, temp_c, precip_mm, wind_kmh.
Features (symmetric — they shift TOTALS, not a side):
  wind_high = max(0, wind_kmh - 25) / 10
  precip    = min(precip_mm, 10) / 5
  temp_cold = max(0, 0 - temp_c) / 5
  temp_hot  = max(0, temp_c - 28) / 5
Fit through the P3 GLM machinery (context.py) — both sides' lambda equally.
Gate: held-out OU2.5 Brier (see backtest via validate.py's OU2.5 metric);
1X2 must not regress beyond the plan Sec 12 tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import model as M

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
VENUES_CSV = DATA / "venues.csv"
WEATHER_CACHE = DATA / "weather_cache"
WEATHER_CSV = DATA / "weather.csv"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
KICKOFF_LOCAL_HOUR = 15   # documented approximation — no kickoff time in fixtures.csv
FORECAST_HORIZON_DAYS = 16
BACKFILL_SINCE = "2022-08-01"
SLEEP_BETWEEN_CALLS = 0.2

WIND_HIGH_THRESHOLD = 25.0
PRECIP_CAP_MM = 10.0
TEMP_HOT_THRESHOLD = 28.0


def load_venues() -> pd.DataFrame:
    if not VENUES_CSV.exists():
        return pd.DataFrame(columns=["team", "city", "lat", "lon", "approx"])
    return pd.read_csv(VENUES_CSV)


def missing_venues() -> list[str]:
    """Teams appearing in fixtures.csv with no venues.csv row."""
    fx = M.load_fixtures()
    teams = set(fx["home"].dropna()) | set(fx["away"].dropna())
    known = set(load_venues()["team"])
    return sorted(teams - known)


def _cache_path(lat: float, lon: float, date: str) -> Path:
    WEATHER_CACHE.mkdir(parents=True, exist_ok=True)
    return WEATHER_CACHE / f"{lat:.4f}_{lon:.4f}_{date}.json"


def _fetch_one(url: str, lat: float, lon: float, date: str,
               is_forecast: bool = False) -> dict | None:
    """One GET, cached to disk. Returns {"temp_c","precip_mm","wind_kmh"} or
    None on any failure (offline-first — never raises)."""
    path = _cache_path(lat, lon, date)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass

    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "auto",
    }
    if is_forecast:
        params["start_date"] = date
        params["end_date"] = date
    else:
        params["start_date"] = date
        params["end_date"] = date
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{qs}", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"  weather: fetch failed for {lat},{lon} {date} ({exc}) — skipped")
        return None

    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    target = f"{date}T{KICKOFF_LOCAL_HOUR:02d}:00"
    if target not in times:
        return None
    idx = times.index(target)
    try:
        result = {
            "temp_c": hourly["temperature_2m"][idx],
            "precip_mm": hourly["precipitation"][idx],
            "wind_kmh": hourly["wind_speed_10m"][idx],
        }
    except (KeyError, IndexError):
        return None
    if all(v is not None for v in result.values()):
        path.write_text(json.dumps(result))
        time.sleep(SLEEP_BETWEEN_CALLS)
    return result if all(v is not None for v in result.values()) else None


def build(verbose: bool = True) -> pd.DataFrame:
    """Backfill (played, >= BACKFILL_SINCE) + forecast (upcoming, <=16d out)
    weather for every fixture with a venues.csv row. Writes data/weather.csv."""
    venues = load_venues()
    if venues.empty:
        if verbose:
            print("weather: no venues.csv rows — nothing to fetch")
        return pd.DataFrame(columns=["fixture_id", "temp_c", "precip_mm", "wind_kmh"])
    vmap = {r.team: (float(r.lat), float(r.lon)) for r in venues.itertuples(index=False)}

    fx = M.load_fixtures()
    played = M.played(fx)
    played = played[played["date"] >= pd.Timestamp(BACKFILL_SINCE)]
    upcoming = M.upcoming(fx)
    today = datetime.now(timezone.utc).date()
    horizon = pd.Timestamp(today + timedelta(days=FORECAST_HORIZON_DAYS))
    upcoming = upcoming[upcoming["date"] <= horizon]

    rows = []
    skipped_no_venue = 0
    fetched_ok = 0
    for is_forecast, subset in ((False, played), (True, upcoming)):
        url = FORECAST_URL if is_forecast else ARCHIVE_URL
        for r in subset.itertuples(index=False):
            latlon = vmap.get(r.home)
            if latlon is None:
                skipped_no_venue += 1
                continue
            date_str = pd.Timestamp(r.date).strftime("%Y-%m-%d")
            wx = _fetch_one(url, latlon[0], latlon[1], date_str, is_forecast)
            if wx is None:
                continue
            rows.append({"fixture_id": r.fixture_id, **wx})
            fetched_ok += 1

    if verbose:
        print(f"  weather: {fetched_ok} fixtures with weather, "
              f"{skipped_no_venue} skipped (no venues.csv row)")
    df = pd.DataFrame(rows, columns=["fixture_id", "temp_c", "precip_mm", "wind_kmh"])
    DATA.mkdir(exist_ok=True)
    df.to_csv(WEATHER_CSV, index=False)
    return df


def load_weather() -> pd.DataFrame:
    if not WEATHER_CSV.exists():
        return pd.DataFrame(columns=["fixture_id", "temp_c", "precip_mm", "wind_kmh"])
    return pd.read_csv(WEATHER_CSV)


def features(temp_c: float, precip_mm: float, wind_kmh: float) -> dict[str, float]:
    """The four symmetric totals-oriented terms from raw weather values."""
    return {
        "wind_high": max(0.0, wind_kmh - WIND_HIGH_THRESHOLD) / 10.0,
        "precip": min(precip_mm, PRECIP_CAP_MM) / 5.0,
        "temp_cold": max(0.0, 0.0 - temp_c) / 5.0,
        "temp_hot": max(0.0, temp_c - TEMP_HOT_THRESHOLD) / 5.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing-venues", action="store_true",
                    help="list fixtures.csv teams with no venues.csv row")
    ap.add_argument("--build", action="store_true",
                    help="fetch backfill + forecast weather, write data/weather.csv")
    args = ap.parse_args()
    if args.missing_venues:
        missing = missing_venues()
        print(f"{len(missing)} teams with no venues.csv row:")
        for t in missing:
            print(f"  {t}")
        return
    if args.build:
        df = build()
        print(f"Wrote {len(df)} rows -> {WEATHER_CSV}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
