#!/usr/bin/env python3
"""Geocode home venues and build the per-team venue table.

Source: Open-Meteo's geocoding API — free, no key, and it returns latitude,
longitude, **elevation and the IANA timezone** in one call, which is exactly the
set of things missing. Spot-checked: Quito 2854m (2850 actual), Madrid 665 (667),
La Paz 3782, Bishkek 767.

Matching is deliberately conservative. The API is asked for a city name and returns
candidates worldwide; we accept one only if its country matches the country
`results.csv` recorded for that match. "Springfield" resolves to dozens of places
and picking the first would silently relocate a fixture — and a wrong venue means a
wrong timezone, which means a wrong local date, which is how fixtures duplicate.
Unmatched cities are left blank and reported, never guessed.

Results are cached to data/international/geocode_cache.csv, so a rebuild costs no
network. Be polite: the API is free and there is no reason to hammer it.

Usage:
  python3 -m scripts.international.build_home_venues --pending      # what's missing
  python3 -m scripts.international.build_home_venues --geocode      # fetch + cache
  python3 -m scripts.international.build_home_venues --build        # write the table
  python3 -m scripts.international.build_home_venues --geocode --build --limit 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from international import home_venues as HV     # noqa: E402
from international import timeutil as TU        # noqa: E402

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
SLEEP_SECONDS = 0.35
SOURCE = "open-meteo"

# results.csv records the country as it was AT THE TIME of the match, while the
# geocoder knows only present-day countries. These are the cases that matter.
COUNTRY_ALIASES = {
    # Home nations: results.csv records "England"/"Scotland"/"Wales"/"Northern
    # Ireland" as countries (they are separate football associations), while any
    # geocoder returns "United Kingdom". Without these, England's home venues —
    # 16 cities including London — resolve to nothing at all.
    "england": {"england", "united kingdom"},
    "scotland": {"scotland", "united kingdom"},
    "wales": {"wales", "united kingdom"},
    "northern ireland": {"northern ireland", "united kingdom"},
    # Territories and dependencies whose football association is separate from the
    # sovereign state the geocoder reports.
    "jersey": {"jersey", "united kingdom"},
    "guernsey": {"guernsey", "united kingdom"},
    "isle of man": {"isle of man", "united kingdom"},
    "gibraltar": {"gibraltar", "united kingdom"},
    "puerto rico": {"puerto rico", "united states"},
    "guadeloupe": {"guadeloupe", "france"},
    "martinique": {"martinique", "france"},
    "french guiana": {"french guiana", "france"},
    "réunion": {"réunion", "reunion", "france"},
    "new caledonia": {"new caledonia", "france"},
    "tahiti": {"tahiti", "french polynesia", "france"},
    "macau": {"macau", "china", "macao"},
    "hong kong": {"hong kong", "china"},
    "northern cyprus": {"northern cyprus", "cyprus"},
    "palestine": {"palestine", "palestinian territory", "state of palestine",
                  "israel"},
    "curaçao": {"curaçao", "curacao", "netherlands"},
    "aruba": {"aruba", "netherlands"},
    "bonaire": {"bonaire", "netherlands", "caribbean netherlands"},
    "sint maarten": {"sint maarten", "netherlands"},

    "united states": {"united states", "united states of america", "usa"},
    "south korea": {"south korea", "korea, republic of", "republic of korea"},
    "north korea": {"north korea", "democratic people's republic of korea"},
    "ivory coast": {"ivory coast", "côte d'ivoire", "cote d'ivoire"},
    "czech republic": {"czech republic", "czechia"},
    "turkey": {"turkey", "türkiye", "turkiye"},
    "dr congo": {"dr congo", "democratic republic of the congo", "congo-kinshasa"},
    "congo": {"congo", "republic of the congo", "congo-brazzaville"},
    "cape verde": {"cape verde", "cabo verde"},
    "east timor": {"east timor", "timor-leste"},
    "swaziland": {"swaziland", "eswatini"},
    "macedonia": {"macedonia", "north macedonia"},
    "burma": {"burma", "myanmar"},
    "russia": {"russia", "russian federation"},
    "iran": {"iran", "islamic republic of iran"},
    "syria": {"syria", "syrian arab republic"},
    "tanzania": {"tanzania", "united republic of tanzania"},
    "vietnam": {"vietnam", "viet nam"},
    "laos": {"laos", "lao people's democratic republic"},
    "bolivia": {"bolivia", "plurinational state of bolivia"},
    "venezuela": {"venezuela", "bolivarian republic of venezuela"},
    "moldova": {"moldova", "republic of moldova"},
    "netherlands": {"netherlands", "the netherlands", "holland"},
}


# Open-Meteo returns 9999.0 as a no-data sentinel, not an elevation. Suva and
# Thessaloniki both came back "9999m" on the first run — which would have made Fiji
# the highest-altitude side in world football and fed a nonsense altitude
# adjustment. The highest ground ever used for a senior international is around
# 4,000m (El Alto, Bolivia), so anything above this is data, not terrain.
MAX_PLAUSIBLE_ELEVATION_M = 5000


def _clean_elevation(value: object) -> float | None:
    try:
        metres = float(value)
    except (TypeError, ValueError):
        return None
    if metres < -500 or metres > MAX_PLAUSIBLE_ELEVATION_M:
        return None
    return metres


def _country_matches(want: object, got: object) -> bool:
    a = str(want or "").strip().casefold()
    b = str(got or "").strip().casefold()
    if not a or not b:
        return False
    if a == b:
        return True
    for group in COUNTRY_ALIASES.values():
        if a in group and b in group:
            return True
    return False


def geocode(city: str, country: str, timeout: int = 20) -> dict | None:
    """Best candidate for this city IN THIS COUNTRY, or None."""
    url = (f"{GEOCODE_URL}?name={urllib.parse.quote(str(city))}"
           f"&count=10&language=en&format=json")
    req = urllib.request.Request(url, headers={"User-Agent": "soccer-predictor"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
    except Exception:
        return None

    for hit in payload.get("results") or []:
        if _country_matches(country, hit.get("country")):
            return {
                "city": city, "country": country,
                "latitude": hit.get("latitude"), "longitude": hit.get("longitude"),
                "elevation_m": _clean_elevation(hit.get("elevation")),
                "timezone": hit.get("timezone") or "",
                "resolved_country": hit.get("country") or "",
                "source": SOURCE, "resolved_at": TU.now_iso(),
            }
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pending", action="store_true", help="list what is missing")
    ap.add_argument("--geocode", action="store_true", help="fetch missing coordinates")
    ap.add_argument("--build", action="store_true", help="write the venue table")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--window-years", type=int, default=HV.WINDOW_YEARS)
    a = ap.parse_args()

    history = HV.home_match_history(window_years=a.window_years)
    teams = history.team.nunique()
    pairs = history[["city", "country"]].drop_duplicates()
    print(f"home-match history: {teams} teams, {len(history)} team-venue rows, "
          f"{len(pairs)} distinct city/country pairs "
          f"(window {a.window_years}y)")

    pending = HV.pending_geocodes(history)
    print(f"not yet geocoded: {len(pending)}")

    if a.pending:
        for _, r in pending.head(40).iterrows():
            print(f"  {r.city}, {r.country}")
        if len(pending) > 40:
            print(f"  … {len(pending) - 40} more")
        return

    if a.geocode and len(pending):
        cache = HV.load_geocode_cache()
        found, missed = [], []
        todo = pending.head(a.limit)
        print(f"\ngeocoding {len(todo)} pair(s) via Open-Meteo…")
        for i, (_, r) in enumerate(todo.iterrows(), 1):
            hit = geocode(r.city, r.country)
            if hit:
                found.append(hit)
            else:
                missed.append(f"{r.city}, {r.country}")
                found.append({"city": r.city, "country": r.country,
                              "latitude": None, "longitude": None,
                              "elevation_m": None, "timezone": "",
                              "resolved_country": "", "source": SOURCE,
                              "resolved_at": TU.now_iso()})
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}…")
            time.sleep(SLEEP_SECONDS)

        cache = pd.concat([cache, pd.DataFrame(found)], ignore_index=True)
        cache = cache.drop_duplicates(subset=["city", "country"], keep="last")
        HV.save_geocode_cache(cache)
        resolved = len(found) - len(missed)
        print(f"resolved {resolved}/{len(found)}; cache now {len(cache)} rows")
        if missed:
            print(f"\nUNRESOLVED ({len(missed)}) — left blank, never guessed:")
            for m in missed[:25]:
                print(f"  {m}")
            if len(missed) > 25:
                print(f"  … {len(missed) - 25} more")

    if a.build:
        table = HV.build(history)
        HV.save(table)
        print(f"\nwrote {HV.HOME_VENUES_CSV.relative_to(ROOT)} ({len(table)} rows)")
        for k, v in HV.coverage().items():
            print(f"  {k:<26}{v}")


if __name__ == "__main__":
    main()
