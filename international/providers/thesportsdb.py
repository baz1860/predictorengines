"""TheSportsDB fixture provider — CROSS-CHECK ONLY.

Measured limits of the free tier, 8 August 2026
-----------------------------------------------
  * `all_leagues.php` returns **10 soccer leagues**, none of them international.
    There is no league discovery: IDs must be hardcoded.
  * `eventsnextleague.php` returns **1 event** per call on the free key.
  * The documented v2 API, which lifts these caps, is $9/month.

So this provider cannot enumerate a fixture list. What it CAN do is answer
"does an independent source also believe this match exists, on this date?" — which
is precisely what the plan needs to stop BSD becoming the primary source by
default rather than by evidence.

It found one thing BSD did not on the first run: **Azerbaijan v Tajikistan,
23 September 2026**, an international friendly absent from BSD's feed, which had
only one friendly in the whole September window.

Do NOT promote this to a primary source without the paid tier. It is registered
here so that provider comparison is possible at all, and so the cost of upgrading
can be argued from measured need rather than a guess.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Iterable

import pandas as pd

from .. import timeutil

PROVIDER = "thesportsdb"
BASE = "https://www.thesportsdb.com/api/v1/json"
FREE_KEY = "3"          # documented public test key

# Hardcoded because discovery is unavailable on the free tier. Verified 8 Aug 2026.
LEAGUE_IDS = {
    4562: "Friendly",           # "International Friendlies"
}

# Known-but-unverified IDs. Left out of LEAGUE_IDS deliberately: an unverified ID
# that silently returns the wrong competition is worse than no coverage at all.
UNVERIFIED = {
    "UEFA Nations League": None,
    "World Cup qualification": None,
}


def _get(path: str, key: str = FREE_KEY, timeout: int = 25):
    req = urllib.request.Request(f"{BASE}/{key}/{path}",
                                 headers={"User-Agent": "soccer-predictor"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch_next(league_id: int, key: str = FREE_KEY) -> list[dict]:
    payload = _get(f"eventsnextleague.php?id={league_id}", key)
    return payload.get("events") or []


def parse_events(events: Iterable[dict], competition: str) -> list[dict]:
    """Provider events -> comparison rows (NOT canonical fixture rows).

    Deliberately does not build canonical fixtures: without a reliable venue or a
    stable cross-provider identity these should not enter the fixture store. They
    exist to be compared against it.
    """
    from engines.worldcup.names import canonical_team, looks_like_national_team

    out = []
    for e in events or []:
        home = e.get("strHomeTeam") or ""
        away = e.get("strAwayTeam") or ""
        if not home or not away:
            # Some payloads carry only "A vs B" in strEvent.
            title = str(e.get("strEvent") or "")
            if " vs " in title:
                home, away = (s.strip() for s in title.split(" vs ", 1))
        if not home or not away:
            continue
        if not (looks_like_national_team(home) and looks_like_national_team(away)):
            continue

        stamp = e.get("strTimestamp") or e.get("dateEvent")
        ts = timeutil.to_utc(stamp)
        if ts is None:
            continue
        kickoff = ts.isoformat()
        local = str(pd.Timestamp(e.get("dateEvent") or ts).date())

        out.append({
            "provider": PROVIDER,
            "provider_event_id": str(e.get("idEvent") or ""),
            "home_team": canonical_team(home),
            "away_team": canonical_team(away),
            "competition": competition,
            "kickoff_utc": kickoff,
            "local_date": local,
        })
    return out


def fetch_all(key: str = FREE_KEY) -> list[dict]:
    rows = []
    for lid, competition in LEAGUE_IDS.items():
        try:
            rows.extend(parse_events(fetch_next(lid, key), competition))
        except Exception:
            continue
    return rows
