"""BSD (Bzzoiro Sports Data) fixture provider.

Measured coverage, 8 August 2026 (see docs/international_provider_spike.md):
  catalogue        79 leagues, of which 16 are senior men's international
  qualifying       ALL SIX confederations present as separate leagues — coverage
                   The Odds API does not have (it carries Europe and South America
                   only)
  upcoming         280 international fixtures, 180 of them inside the
                   21 Sep - 6 Oct window
  gaps             AFCON 2027 qualifying is ABSENT despite starting 23 September;
                   continental tournaments are per-edition leagues ("UEFA Euro
                   2024", "AFC Asian Cup 2023") that do not roll forward

Two API generations are in play and they differ in ways that matter:

  v1 `/api/events/`     embeds the full league, venue (with lat/lon) and odds, and
                        serves `event_date` with a **+04:00** offset.
  v2 `/api/v2/events/`  returns `league_id` / `venue_id` as integers and serves
                        `event_date` in **+00:00**.

Both are offset-aware, so converting to UTC is unambiguous. What is NOT
unambiguous is the *local* date, which needs the venue timezone — see
international/venues.py.
"""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .. import taxonomy, timeutil, venues
from ..store import ABANDONED, CANCELLED, PLAYED, POSTPONED, SCHEDULED

PROVIDER = "bsd"

# BSD league name -> our competition label (international/taxonomy.py).
# Deliberately explicit: an unmapped league is REPORTED, never guessed, because a
# wrong competition label silently changes a match's rating weight.
LEAGUE_TO_COMPETITION = {
    "World Cup 2026": "FIFA World Cup",
    "International Friendly Games": "Friendly",
    "World Cup Qualification UEFA": "FIFA World Cup qualification",
    "World Cup Qualification CONMEBOL": "FIFA World Cup qualification",
    "World Cup Qualification CAF": "FIFA World Cup qualification",
    "World Cup Qualification AFC": "FIFA World Cup qualification",
    "World Cup Qualification CONCACAF": "FIFA World Cup qualification",
    "World Cup Qualification OFC": "FIFA World Cup qualification",
    "UEFA Nations League": "UEFA Nations League",
    "CONCACAF Nations League": "CONCACAF Nations League",
    "UEFA Euro 2024": "UEFA Euro",
    "Africa Cup of Nations 2023": "African Cup of Nations",
    "AFC Asian Cup 2023": "AFC Asian Cup",
    "CONCACAF Gold Cup 2025": "Gold Cup",
    "Copa America 2024": "Copa América",
}

# Leagues we can see but deliberately exclude (plan §3 scope).
EXCLUDED_LEAGUES = {
    "UEFA European U19 Championship",   # youth
    "Club Friendlies",                  # clubs
    "Liga F", "NWSL",                   # women's
}

STATUS_MAP = {
    "notstarted": SCHEDULED, "scheduled": SCHEDULED, "ns": SCHEDULED,
    "finished": PLAYED, "ft": PLAYED, "aet": PLAYED, "pen": PLAYED,
    "postponed": POSTPONED, "delayed": POSTPONED,
    "cancelled": CANCELLED, "canceled": CANCELLED,
    "abandoned": ABANDONED, "interrupted": ABANDONED, "suspended": ABANDONED,
}


class UnmappedLeague(KeyError):
    """A BSD league that is neither mapped nor explicitly excluded."""


def international_league_ids(leagues: Iterable[dict]) -> dict[int, str]:
    """league_id -> our competition label, for leagues in scope."""
    out: dict[int, str] = {}
    for lg in leagues:
        name = str(lg.get("name") or "").strip()
        if name in EXCLUDED_LEAGUES or lg.get("is_women"):
            continue
        comp = LEAGUE_TO_COMPETITION.get(name)
        if comp is not None:
            out[int(lg["id"])] = comp
    return out


def unmapped_international_leagues(leagues: Iterable[dict]) -> list[str]:
    """International-looking leagues we have not mapped. Should be reviewed."""
    continental = {"International", "Africa", "Asia", "Europe",
                   "North America", "South America", "Oceania"}
    club_markers = ("Champions League", "Europa", "Conference", "Libertadores",
                    "Sudamericana", "CAF Champions")
    out = []
    for lg in leagues:
        name = str(lg.get("name") or "").strip()
        if (str(lg.get("country")) in continental
                and not lg.get("is_women")
                and name not in LEAGUE_TO_COMPETITION
                and name not in EXCLUDED_LEAGUES
                and not any(m in name for m in club_markers)):
            out.append(name)
    return sorted(set(out))


def _score(value: object) -> object:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class NotANationalTeam(ValueError):
    """A club or age-group side appearing in an international league."""


def parse_event(event: dict, competition: str,
                venue_lookup: bool = True) -> dict:
    """One v2 event -> one canonical fixture row. Pure: no network, no clock."""
    from engines.worldcup.names import canonical_team, looks_like_national_team

    from ..store import normalize_fixture

    # BSD files club friendlies and youth internationals inside "International
    # Friendly Games" — the spike found Albirex Niigata, Miami United and
    # Jamaica U20 in there. Reject before anything downstream sees them.
    home_raw, away_raw = event["home_team"], event["away_team"]
    for side in (home_raw, away_raw):
        if not looks_like_national_team(side):
            raise NotANationalTeam(f"{side!r} is not a senior national team")

    # Canonicalise BEFORE the scope test. Without this, "Czechia", "Türkiye",
    # "Ireland" and "Bosnia & Herzegovina" are all read as unknown teams and
    # their fixtures silently dropped — 24 of them in the first spike run.
    home, away = canonical_team(home_raw), canonical_team(away_raw)

    kickoff = timeutil.to_utc(event["event_date"])
    if kickoff is None:
        raise ValueError(f"unparseable event_date {event.get('event_date')!r}")

    venue_id = event.get("venue_id")
    local, tz = venues.local_date(kickoff, venue_id if venue_lookup else None)
    venue = venues.get(venue_id) if venue_lookup else {}
    tz_basis = "venue" if tz else ""

    # Fallback: BSD supplies no venue for most Nations League fixtures (183 of 255
    # on the first run), leaving the local date as the UTC date and possibly a day
    # out. When the match is NOT on neutral ground, the home team's usual ground is
    # a far better guess than UTC.
    #
    # The neutral check is essential and easy to forget: at a neutral venue the
    # home team's own stadium is irrelevant, and using it would confidently produce
    # the wrong timezone rather than an honestly missing one.
    if not tz and venue_lookup and not event.get("is_neutral_ground"):
        from .. import home_venues
        guess, basis = home_venues.timezone_for(team=home)
        if guess:
            local, tz, tz_basis = (
                timeutil.local_date(kickoff, guess)[0], guess, basis)

    status = STATUS_MAP.get(str(event.get("status", "")).lower(), SCHEDULED)
    hs, as_ = _score(event.get("home_score")), _score(event.get("away_score"))
    if status == PLAYED and (hs is None or as_ is None):
        # Finished but unscored: refuse to promote it, or it becomes a blank row
        # that load_matches() will present as an upcoming fixture forever.
        status = SCHEDULED

    row = normalize_fixture(
        home=home, away=away, competition=competition,
        kickoff_utc=kickoff, local_date=local, venue_tz=tz,
        city=str(venue.get("city") or ""), country=str(venue.get("country") or ""),
        neutral=bool(event.get("is_neutral_ground")),
        home_score=hs, away_score=as_, status=status,
        provider=PROVIDER, provider_event_id=event.get("id"),
    )
    if not tz:
        row["conflict"] = ("venue timezone unresolved; local_date is the UTC date "
                           + (f"(venue_id={venue_id})" if venue_id else "(no venue)"))
    elif tz_basis and tz_basis != "venue":
        # Record HOW the date was derived. A date inferred from where a team
        # usually plays is not the same fact as a date from a named venue, and the
        # store should not present them identically.
        row["conflict"] = f"local_date derived from {tz_basis}, not a named venue"
    return row


def parse_events(events: Iterable[dict], league_map: dict[int, str],
                 venue_lookup: bool = True) -> tuple[list[dict], list[dict]]:
    """(rows, skipped). Events outside the league map are skipped, not guessed."""
    rows, skipped = [], []
    for e in events:
        lid = e.get("league_id")
        comp = league_map.get(int(lid)) if lid is not None else None
        if comp is None:
            skipped.append({"id": e.get("id"), "league_id": lid,
                            "home": e.get("home_team"), "away": e.get("away_team"),
                            "reason": "league not in international map"})
            continue
        try:
            rows.append(parse_event(e, comp, venue_lookup=venue_lookup))
        except Exception as exc:                    # never let one row kill a run
            skipped.append({"id": e.get("id"), "league_id": lid,
                            "home": e.get("home_team"), "away": e.get("away_team"),
                            "reason": f"{type(exc).__name__}: {exc}"})
    return rows, skipped


def validate_competitions(league_map: dict[int, str]) -> list[str]:
    """Every mapped competition must exist in our taxonomy."""
    return sorted({c for c in league_map.values() if taxonomy.classify(c) is None})


# ── network ──────────────────────────────────────────────────────────────────
def fetch_leagues(api_key: str) -> list[dict]:
    from bsd_client import get_all_v2_leagues
    return get_all_v2_leagues(api_key)


def fetch_venues(api_key: str) -> list[dict]:
    import json
    import urllib.request
    out, offset = [], 0
    while True:
        url = (f"https://sports.bzzoiro.com/api/v2/venues/?limit=200&offset={offset}")
        req = urllib.request.Request(url, headers={
            "Authorization": f"Token {api_key}", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            page = json.load(r)
        out.extend(page.get("results", []))
        if not page.get("next"):
            return out
        offset += 200


def fetch_events(api_key: str, status: str = "notstarted") -> list[dict]:
    from bsd_client import get_all_v2_events
    return get_all_v2_events(api_key, status=status)
