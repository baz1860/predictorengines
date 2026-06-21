"""Open-Meteo weather provider for golf course and tee-wave features."""

from __future__ import annotations

import csv
import datetime as dt
import json
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "api_cache" / "open_meteo"
COURSE_LOCATIONS = DATA_DIR / "course_locations.csv"


@dataclass(frozen=True)
class CourseLocation:
    course_name: str
    latitude: float
    longitude: float
    timezone: str = "auto"


class OpenMeteoProvider:
    name = "open_meteo"

    def __init__(self, cache_dir: Path | None = None, ttl_seconds: int = 3600):
        self.cache_dir = cache_dir or CACHE_DIR
        self.ttl_seconds = ttl_seconds

    def load_course_locations(self, path: Path | None = None) -> dict[str, CourseLocation]:
        path = path or COURSE_LOCATIONS
        if not path.exists():
            return {}
        out = {}
        with path.open() as f:
            for row in csv.DictReader(f):
                name = (row.get("course") or row.get("course_name") or "").strip()
                try:
                    lat = float(row.get("latitude") or row.get("lat"))
                    lon = float(row.get("longitude") or row.get("lon"))
                except (TypeError, ValueError):
                    continue
                out[_key(name)] = CourseLocation(
                    course_name=name,
                    latitude=lat,
                    longitude=lon,
                    timezone=row.get("timezone") or "auto",
                )
        return out

    def forecast(self, location: CourseLocation, start_date: str, days: int = 4,
                 use_cache: bool = True) -> dict:
        start = dt.date.fromisoformat(start_date[:10])
        end = start + dt.timedelta(days=max(1, days) - 1)
        return self._request(
            "forecast",
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "timezone": location.timezone,
                "hourly": ",".join([
                    "temperature_2m", "precipitation", "wind_speed_10m",
                    "wind_gusts_10m", "wind_direction_10m",
                ]),
            },
            use_cache=use_cache,
        )

    def historical(self, location: CourseLocation, start_date: str, end_date: str,
                   use_cache: bool = True) -> dict:
        return self._request(
            "archive",
            "https://archive-api.open-meteo.com/v1/archive",
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "start_date": start_date[:10],
                "end_date": end_date[:10],
                "timezone": location.timezone,
                "hourly": ",".join([
                    "temperature_2m", "precipitation", "wind_speed_10m",
                    "wind_gusts_10m", "wind_direction_10m",
                ]),
            },
            use_cache=use_cache,
        )

    def summarize_wave(self, payload: dict, start_hour: int = 7, end_hour: int = 19) -> dict:
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        keep = []
        for i, ts in enumerate(times):
            try:
                hour = int(str(ts)[11:13])
            except ValueError:
                continue
            if start_hour <= hour <= end_hour:
                keep.append(i)
        return {
            "temperature_2m": _mean(hourly.get("temperature_2m"), keep),
            "precipitation": _sum(hourly.get("precipitation"), keep),
            "wind_speed_10m": _mean(hourly.get("wind_speed_10m"), keep),
            "wind_gusts_10m": _mean(hourly.get("wind_gusts_10m"), keep),
            "wind_direction_10m": _circular_mean(hourly.get("wind_direction_10m"), keep),
            "hours": len(keep),
        }

    def _request(self, label: str, url: str, params: dict, use_cache: bool = True) -> dict:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = "_".join([
            label,
            str(params.get("latitude")),
            str(params.get("longitude")),
            str(params.get("start_date")),
            str(params.get("end_date")),
        ]).replace("/", "-")
        cache = self.cache_dir / f"{key}.json"
        if use_cache and cache.exists() and time.time() - cache.stat().st_mtime <= self.ttl_seconds:
            return json.loads(cache.read_text())
        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
        cache.write_text(json.dumps(payload))
        return payload


def _key(name: str) -> str:
    return " ".join(str(name or "").lower().split())


def _mean(values, idx: list[int]) -> float | None:
    vals = [float(values[i]) for i in idx if values and i < len(values) and values[i] is not None]
    return round(statistics.fmean(vals), 3) if vals else None


def _sum(values, idx: list[int]) -> float | None:
    vals = [float(values[i]) for i in idx if values and i < len(values) and values[i] is not None]
    return round(sum(vals), 3) if vals else None


def _circular_mean(values, idx: list[int]) -> float | None:
    import math

    vals = [math.radians(float(values[i])) for i in idx if values and i < len(values) and values[i] is not None]
    if not vals:
        return None
    sin_m = statistics.fmean(math.sin(v) for v in vals)
    cos_m = statistics.fmean(math.cos(v) for v in vals)
    return round((math.degrees(math.atan2(sin_m, cos_m)) + 360) % 360, 1)
