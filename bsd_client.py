"""BSD (Bzzoiro Sports Data) API client.

Free football data — no rate limits, no credit card required.
Register at https://sports.bzzoiro.com/register/ to get a key.

Authentication: Authorization: Token YOUR_API_KEY (header on every request)
Base URL:       https://sports.bzzoiro.com

Key endpoints
-------------
GET /api/events/           Paginated list of matches (all leagues).
                           Embeds odds, unavailable players, and (when
                           available) confirmed lineups.
GET /api/events/{id}/      Single match with full detail.

Shared pagination convention (all BSD list endpoints):
  ?limit=200&offset=0  ->  {"count": N, "next": "...", "results": [...]}

BSD docs:    https://sports.bzzoiro.com/docs/football/
Swagger UI:  https://sports.bzzoiro.com/api/docs/
OpenAPI:     https://sports.bzzoiro.com/api/schema/
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

BSD_BASE = "https://sports.bzzoiro.com"
_DEFAULT_LIMIT = 200          # BSD max page size
_TIMEOUT = 30                 # seconds


def _get(path: str, api_key: str, **params) -> Any:
    """Single authenticated GET to the BSD API.

    Raises RuntimeError if the HTTP request fails or the response signals
    an error.  All other parsing is left to callers.
    """
    qs = urllib.parse.urlencode(
        {k: v for k, v in params.items() if v is not None}
    )
    url = f"{BSD_BASE}{path}"
    if qs:
        url += "?" + qs
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Token {api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"BSD HTTP {exc.code} for {path}: {exc.reason}"
        ) from exc


def get_events_page(api_key: str, **params) -> dict:
    """Return one page of events.

    Useful params (all optional):
      status   — "upcoming" | "live" | "finished"
      date     — "YYYY-MM-DD"  (filter by match date)
      league   — BSD league id (int/str) if known
      limit    — items per page (default 200, max 200)
      offset   — items to skip (default 0)
    """
    return _get("/api/events/", api_key, **params)


def get_all_events(api_key: str, **filters) -> list[dict]:
    """Fetch ALL pages of /api/events/ matching *filters*.

    De-duplicates by BSD event ``id`` so double-fetches are safe.
    """
    results: list[dict] = []
    seen: set[int | str] = set()
    limit = int(filters.pop("limit", _DEFAULT_LIMIT))
    offset = 0

    while True:
        page = get_events_page(api_key, limit=limit, offset=offset, **filters)
        batch = page.get("results") or []
        for item in batch:
            eid = item.get("id")
            if eid not in seen:
                seen.add(eid)
                results.append(item)
        if not page.get("next") or not batch:
            break
        offset += limit

    return results


def get_event(api_key: str, event_id: int | str) -> dict:
    """Single event by BSD id — includes full lineups and stats when available."""
    return _get(f"/api/events/{event_id}/", api_key)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def league_name(event: dict) -> str:
    """Return the league/competition name from a BSD event dict.

    BSD may return league as a string *or* as a nested object
    {"id": …, "name": …}. This helper normalises both.
    """
    raw = event.get("league") or event.get("competition") or ""
    if isinstance(raw, dict):
        return str(raw.get("name") or raw.get("title") or "").strip()
    return str(raw).strip()


def event_date_utc(event: dict) -> str:
    """ISO-8601 UTC kickoff string, or empty string."""
    return str(event.get("date") or event.get("kickoff") or "").strip()


def unavailable_players(event: dict) -> dict[str, list[dict]]:
    """Return {"home": [...], "away": [...]} injury/suspension dicts.

    Each player dict typically has: name, reason, status.
    """
    raw = event.get("unavailable_players") or {}
    if not isinstance(raw, dict):
        return {"home": [], "away": []}
    return {
        "home": list(raw.get("home") or []),
        "away": list(raw.get("away") or []),
    }


def lineups(event: dict) -> dict[str, dict]:
    """Return {"home": {formation, starters, bench}, "away": {…}}.

    Returns empty dicts if no lineups are available yet.
    """
    raw = event.get("lineups") or event.get("lineup") or {}
    if not isinstance(raw, dict):
        return {"home": {}, "away": {}}
    return {
        "home": dict(raw.get("home") or raw.get("home_team") or {}),
        "away": dict(raw.get("away") or raw.get("away_team") or {}),
    }


def match_statistics(event: dict) -> dict[str, dict]:
    """Return {"home": {shots, xg, …}, "away": {…}}.

    Returns empty dicts if stats are not yet available (pre-match).
    """
    raw = event.get("statistics") or event.get("stats") or {}
    if not isinstance(raw, dict):
        return {"home": {}, "away": {}}
    return {
        "home": dict(raw.get("home") or {}),
        "away": dict(raw.get("away") or {}),
    }
