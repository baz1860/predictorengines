"""Where each national team actually plays at home.

The problem
-----------
Two measured gaps share one root cause — we do not know where a fixture is played:

  * **202 of 255** BSD fixtures carry no venue, so their local date is the UTC date
    and may be a day out (see international/timeutil.py);
  * the altitude adjustment in `engines/worldcup/context.py` runs off a hand-typed
    dictionary of **48 cities**, so a match in Addis Ababa or Cusco is treated as
    sea level unless someone remembered to add it.

Why "national stadium" is the wrong question
--------------------------------------------
Some countries have one. Many do not. Measured over the last decade of non-neutral
home matches:

    Spain          22 distinct home cities
    United States  39
    Germany        21
    Kyrgyzstan      1
    Liechtenstein   1

So a single "national stadium" column would be right for roughly half the world and
quietly wrong for the rest — exactly the failure mode of the 48-city altitude dict.

What this does instead
----------------------
Builds a *distribution* per team from `results.csv`: every city they have hosted a
match in over the window, ranked by frequency, with the share each represents. A
team with one venue gets a certainty of 1.0; Spain gets its most-used ground and an
honest 20-something percent share.

Callers must respect `share`. `primary_venue()` returns the modal venue AND how
often it is actually used, so a caller can decide whether a guess is good enough —
using Spain's most likely ground to date a fixture is a coin flip, and the code
should say so rather than imply precision it does not have.

Geocoding is Open-Meteo: free, no key, and it returns latitude, longitude,
elevation and the IANA timezone in a single call. Spot-checked against known values
(Quito 2854m vs 2850 actual, Madrid 665 vs 667, La Paz 3782).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results.csv"
GEOCODE_CSV = ROOT / "data" / "international" / "geocode_cache.csv"
HOME_VENUES_CSV = ROOT / "data" / "international" / "team_home_venues.csv"

WINDOW_YEARS = 10
GEOCODE_COLUMNS = ["city", "country", "latitude", "longitude", "elevation_m",
                   "timezone", "resolved_country", "source", "resolved_at"]
HOME_COLUMNS = ["team", "city", "country", "matches", "share", "rank",
                "latitude", "longitude", "elevation_m", "timezone"]

# Below this share, calling a venue the team's "home" is not defensible.
CONFIDENT_SHARE = 0.50


@dataclass(frozen=True)
class Venue:
    city: str
    country: str
    matches: int
    share: float
    latitude: float | None
    longitude: float | None
    elevation_m: float | None
    timezone: str

    @property
    def confident(self) -> bool:
        return self.share >= CONFIDENT_SHARE

    @property
    def located(self) -> bool:
        return self.latitude is not None and self.longitude is not None


def home_match_history(results: pd.DataFrame | None = None,
                       window_years: int = WINDOW_YEARS) -> pd.DataFrame:
    """Non-neutral home matches per (team, city, country) over the window."""
    df = results if results is not None else pd.read_csv(RESULTS, parse_dates=["date"])
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.assign(date=pd.to_datetime(df["date"], errors="coerce"))
    played = df.dropna(subset=["home_score", "away_score"])

    # Neutral-venue matches say nothing about where a team plays at home. They are
    # the majority of tournament football, so including them would put every World
    # Cup participant's "home" in the host country.
    neutral = played["neutral"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    home = played[~neutral]
    home = home[home.date >= home.date.max() - pd.DateOffset(years=window_years)]

    grouped = (home.groupby(["home_team", "city", "country"])
                   .size().reset_index(name="matches")
                   .rename(columns={"home_team": "team"}))
    totals = grouped.groupby("team").matches.transform("sum")
    grouped["share"] = (grouped.matches / totals).round(4)
    grouped["rank"] = (grouped.groupby("team").matches
                              .rank(method="first", ascending=False).astype(int))
    return grouped.sort_values(["team", "rank"]).reset_index(drop=True)


def pending_geocodes(history: pd.DataFrame | None = None) -> pd.DataFrame:
    """(city, country) pairs not yet in the geocode cache."""
    hist = history if history is not None else home_match_history()
    want = hist[["city", "country"]].drop_duplicates()
    have = load_geocode_cache()
    if have.empty:
        return want
    merged = want.merge(have[["city", "country"]].assign(_seen=1),
                        on=["city", "country"], how="left")
    return merged[merged._seen.isna()].drop(columns=["_seen"])


def load_geocode_cache() -> pd.DataFrame:
    if not GEOCODE_CSV.exists():
        return pd.DataFrame(columns=GEOCODE_COLUMNS)
    return pd.read_csv(GEOCODE_CSV)


def save_geocode_cache(df: pd.DataFrame) -> Path:
    GEOCODE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=GEOCODE_COLUMNS).to_csv(GEOCODE_CSV, index=False)
    load_geocode_cache.cache_clear() if hasattr(load_geocode_cache, "cache_clear") else None
    return GEOCODE_CSV


def build(history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join the home-match history to the geocode cache."""
    hist = history if history is not None else home_match_history()
    geo = load_geocode_cache()
    if geo.empty:
        for col in ("latitude", "longitude", "elevation_m", "timezone"):
            hist[col] = None
        return hist.reindex(columns=HOME_COLUMNS)
    out = hist.merge(
        geo[["city", "country", "latitude", "longitude", "elevation_m", "timezone"]],
        on=["city", "country"], how="left")
    return out.reindex(columns=HOME_COLUMNS)


def save(df: pd.DataFrame) -> Path:
    HOME_VENUES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HOME_VENUES_CSV, index=False)
    _load.cache_clear()
    return HOME_VENUES_CSV


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    if not HOME_VENUES_CSV.exists():
        return pd.DataFrame(columns=HOME_COLUMNS)
    return pd.read_csv(HOME_VENUES_CSV)


def _to_venue(row: pd.Series) -> Venue:
    def num(value):
        return None if pd.isna(value) else float(value)
    tz = row.get("timezone")
    return Venue(str(row["city"]), str(row["country"]), int(row["matches"]),
                 float(row["share"]), num(row.get("latitude")),
                 num(row.get("longitude")), num(row.get("elevation_m")),
                 "" if pd.isna(tz) else str(tz))


def venues_for(team: object) -> list[Venue]:
    """Every known home venue for a team, most used first."""
    df = _load()
    rows = df[df.team == str(team)].sort_values("rank")
    return [_to_venue(r) for _, r in rows.iterrows()]


def primary_venue(team: object) -> Venue | None:
    """The modal home venue, or None. CHECK `.share` before trusting it."""
    vs = venues_for(team)
    return vs[0] if vs else None


def venue_for_city(city: object, country: object = None) -> Venue | None:
    """Look up a specific city, for when the fixture DOES name a venue."""
    geo = load_geocode_cache()
    if geo.empty:
        return None
    hit = geo[geo.city.astype(str).str.casefold() == str(city).strip().casefold()]
    if country:
        narrowed = hit[hit.country.astype(str).str.casefold()
                       == str(country).strip().casefold()]
        hit = narrowed if len(narrowed) else hit
    if hit.empty:
        return None
    r = hit.iloc[0]
    return Venue(str(r.city), str(r.country), 0, 1.0,
                 None if pd.isna(r.latitude) else float(r.latitude),
                 None if pd.isna(r.longitude) else float(r.longitude),
                 None if pd.isna(r.elevation_m) else float(r.elevation_m),
                 "" if pd.isna(r.timezone) else str(r.timezone))


def timezone_for(team: object = None, city: object = None,
                 country: object = None) -> tuple[str, str]:
    """(timezone, basis) — the fallback chain for dating a fixture.

    Order: the named city, then the team's modal home venue, then nothing. The
    basis string is returned so callers can record HOW a date was derived rather
    than presenting a guess and a fact identically.
    """
    if city:
        v = venue_for_city(city, country)
        if v and v.timezone:
            return v.timezone, "venue city"
    if team:
        v = primary_venue(team)
        if v and v.timezone:
            basis = (f"home venue {v.city} ({v.share:.0%} of home matches)"
                     if v.confident else
                     f"MODAL home venue {v.city}, only {v.share:.0%} of home "
                     f"matches — low confidence")
            return v.timezone, basis
    return "", "unresolved"


def elevation_for(team: object = None, city: object = None,
                  country: object = None) -> tuple[float | None, str]:
    """(metres, basis). Replaces the 48-city hand-typed altitude dictionary."""
    if city:
        v = venue_for_city(city, country)
        if v and v.elevation_m is not None:
            return v.elevation_m, "venue city"
    if team:
        v = primary_venue(team)
        if v and v.elevation_m is not None:
            return v.elevation_m, f"modal home venue {v.city} ({v.share:.0%})"
    return None, "unresolved"


def coverage() -> dict:
    df = _load()
    if df.empty:
        return {"teams": 0}
    primary = df[df["rank"] == 1]
    return {
        "teams": int(df.team.nunique()),
        "venues": len(df),
        "geocoded": int(df.latitude.notna().sum()),
        "with_timezone": int(df.timezone.notna().sum()),
        "with_elevation": int(df.elevation_m.notna().sum()),
        "teams_single_venue": int((primary.share >= 0.999).sum()),
        "teams_confident_primary": int((primary.share >= CONFIDENT_SHARE).sum()),
        "teams_rotating": int((primary.share < CONFIDENT_SHARE).sum()),
    }


if __name__ == "__main__":
    for k, v in coverage().items():
        print(f"  {k:<26}{v}")
    df = _load()
    if not df.empty:
        p = df[df["rank"] == 1].sort_values("share")
        print("\nmost venue-rotating teams (modal venue share):")
        print(p.head(10)[["team", "city", "matches", "share"]].to_string(index=False))
        print("\nhighest-altitude modal home venues:")
        hi = p.dropna(subset=["elevation_m"]).sort_values("elevation_m", ascending=False)
        print(hi.head(10)[["team", "city", "elevation_m", "timezone"]].to_string(index=False))
