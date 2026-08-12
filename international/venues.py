"""Venue table with real timezones, derived from coordinates.

The bug this replaces
---------------------
`scripts/worldcup/live_data.py::_local_match_date()` converts every kick-off to a
single hardcoded timezone (`_TZ_PDT`) and takes the date. For a World Cup played
across three North American countries that is roughly right. For a global module it
is wrong: a 19:00 kick-off in Tokyo converted to US Pacific lands on the previous
calendar day, and a fixture whose local date is wrong is a fixture that duplicates
against any source that dated it correctly — the exact failure documented in
international/identity.py.

Approach: BSD's venue endpoint carries latitude and longitude for 1,786 venues.
Coordinates resolve to an IANA timezone, which gives a correct local date anywhere.
Resolution is cached to data/international/venues.csv so a fixture refresh does not
re-derive it.

`timezonefinder` is an optional dependency. Without it, venues resolve to an empty
timezone and every affected fixture is flagged in its `conflict` column rather than
silently taking a wrong local date. Degraded, visible, not wrong.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import timeutil

ROOT = Path(__file__).resolve().parents[1]
VENUES_CSV = ROOT / "data" / "international" / "venues.csv"

COLUMNS = ["venue_id", "name", "city", "country", "country_code",
           "latitude", "longitude", "timezone", "tz_source", "capacity"]


@lru_cache(maxsize=1)
def _finder():
    try:
        from timezonefinder import TimezoneFinder
    except ImportError:
        return None
    return TimezoneFinder()


def timezone_for(lat: object, lon: object) -> str:
    """IANA timezone for coordinates, or "" when it cannot be determined."""
    tf = _finder()
    if tf is None:
        return ""
    try:
        return tf.timezone_at(lat=float(lat), lng=float(lon)) or ""
    except (TypeError, ValueError):
        return ""


def timezonefinder_available() -> bool:
    return _finder() is not None


@lru_cache(maxsize=1)
def load() -> dict[int, dict]:
    """venue_id -> venue record. Empty dict when the cache has not been built."""
    if not VENUES_CSV.exists():
        return {}
    df = pd.read_csv(VENUES_CSV)
    return {int(r["venue_id"]): dict(r) for _, r in df.iterrows()}


def get(venue_id: object) -> dict:
    try:
        return load().get(int(venue_id), {})
    except (TypeError, ValueError):
        return {}


def tz_of(venue_id: object) -> str:
    """IANA zone for a venue, or "".

    Note the NaN trap: a blank CSV cell loads as float NaN, and `NaN or ""`
    evaluates to NaN (NaN is truthy), so the naive version of this returned the
    string "nan" — which then fails tz_convert and silently falls back to UTC.
    """
    tz = get(venue_id).get("timezone")
    return str(tz).strip() if _nonempty(tz) else ""


def build(records: list[dict]) -> pd.DataFrame:
    """Normalise provider venue records and resolve each timezone.

    Only about 44% of BSD venues carry coordinates. For the rest we infer the
    timezone from the country, but ONLY when every coordinate-bearing venue in
    that country agrees on one zone. Multi-timezone countries (USA, Russia,
    Brazil, Australia…) are left unresolved and flagged, because guessing a zone
    there would produce exactly the wrong-local-date bug this module exists to
    prevent. The map is derived from the data rather than hardcoded, so it
    improves automatically as the provider fills in coordinates.
    """
    rows = []
    for v in records:
        lat, lon = v.get("latitude"), v.get("longitude")
        rows.append({
            "venue_id": v.get("id"),
            "name": v.get("name") or "",
            "city": v.get("city") or "",
            "country": v.get("country") or "",
            "country_code": v.get("country_code") or "",
            "latitude": lat, "longitude": lon,
            "timezone": timezone_for(lat, lon),
            "capacity": v.get("capacity") or "",
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timezone"] = df.timezone.fillna("").astype(str)
    df["tz_source"] = ""
    df.loc[df.timezone != "", "tz_source"] = "coordinates"

    resolved = df[(df.timezone != "") & df.country.astype(bool)]
    by_country = resolved.groupby("country").timezone.nunique()
    single = {c: resolved[resolved.country == c].timezone.iloc[0]
              for c in by_country[by_country == 1].index}
    fill = (df.timezone == "") & df.country.isin(single)
    df.loc[fill, "timezone"] = df.loc[fill, "country"].map(single)
    df.loc[fill, "tz_source"] = "country"
    return df


def save(df: pd.DataFrame) -> Path:
    VENUES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(VENUES_CSV, index=False)
    load.cache_clear()
    return VENUES_CSV


def _nonempty(value: object) -> bool:
    """True for a real string. NaN is falsy in intent but truthy in Python."""
    return pd.notna(value) and str(value).strip() != ""


def coverage() -> dict[str, int]:
    v = load()
    return {"venues": len(v),
            "with_coords": sum(1 for r in v.values()
                               if pd.notna(r.get("latitude"))),
            "with_timezone": sum(1 for r in v.values()
                                 if _nonempty(r.get("timezone"))),
            "tz_from_coords": sum(1 for r in v.values()
                                  if str(r.get("tz_source", "")) == "coordinates"),
            "tz_from_country": sum(1 for r in v.values()
                                   if str(r.get("tz_source", "")) == "country")}


def local_date(kickoff_utc: object, venue_id: object = None,
               fallback_tz: str = "") -> tuple[str, str]:
    """(local_date, timezone_used). Timezone is "" when it could not be resolved.

    Callers must treat an empty timezone as a flag, not a default: the returned
    date is then the UTC date, which is right for roughly half the world and wrong
    for the rest.
    """
    tz = (tz_of(venue_id) if venue_id is not None else "") or fallback_tz
    return timeutil.local_date(kickoff_utc, tz)
