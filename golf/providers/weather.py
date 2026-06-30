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
    aliases: tuple[str, ...] = ()


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
                aliases = tuple(
                    a.strip() for a in str(row.get("aliases") or "").split(";")
                    if a.strip()
                )
                loc = CourseLocation(
                    course_name=name,
                    latitude=lat,
                    longitude=lon,
                    timezone=row.get("timezone") or "auto",
                    aliases=aliases,
                )
                for key in (name, *aliases):
                    out[_key(key)] = loc
        return out

    def resolve_location(self, course_name: str = "",
                         event_name: str = "") -> tuple[CourseLocation | None, str]:
        """Resolve ESPN course/event text to a configured course location."""
        locs = self.load_course_locations()
        for raw in (course_name, event_name):
            key = _key(raw)
            if key and key in locs:
                return locs[key], raw
        return None, course_name or event_name

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

    def wave_features(self, payload: dict, split_hour: int = 12,
                      start_hour: int = 7, end_hour: int = 19) -> dict:
        """Round/day early-vs-late weather penalties in strokes.

        This is intentionally conservative. It uses wind speed, gusts, and rain
        to estimate relative wave difficulty; only the early/late difference is
        consumed by the model and field-centering removes any tournament-level
        scoring effect.
        """
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        per_date: dict[str, dict[str, list[int]]] = {}
        for i, ts in enumerate(times):
            text = str(ts)
            if len(text) < 13:
                continue
            date = text[:10]
            try:
                hour = int(text[11:13])
            except ValueError:
                continue
            if not start_hour <= hour <= end_hour:
                continue
            side = "late" if hour >= split_hour else "early"
            per_date.setdefault(date, {"early": [], "late": []})[side].append(i)

        rounds = {}
        for rnd, date in enumerate(sorted(per_date), 1):
            sides = per_date[date]
            early = _difficulty(hourly, sides["early"])
            late = _difficulty(hourly, sides["late"])
            if early is None or late is None:
                continue
            avg = (early + late) / 2.0
            rounds[str(rnd)] = {
                "date": date,
                "wave_penalty": {
                    "split_hour": split_hour,
                    "early_penalty": round(early - avg, 3),
                    "late_penalty": round(late - avg, 3),
                    "early_raw": round(early, 3),
                    "late_raw": round(late, 3),
                },
            }
        return {"rounds": rounds}

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


def _difficulty(hourly: dict, idx: list[int]) -> float | None:
    if not idx:
        return None
    wind = _mean(hourly.get("wind_speed_10m"), idx) or 0.0
    gust = _mean(hourly.get("wind_gusts_10m"), idx) or wind
    rain = _sum(hourly.get("precipitation"), idx) or 0.0
    # Open-Meteo wind is usually km/h for the default endpoint. A 10 km/h wind
    # wave gap should matter, but not swamp skill.
    return 0.018 * wind + 0.010 * max(0.0, gust - wind) + 0.030 * rain
