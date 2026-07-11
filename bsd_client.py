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
_STATUS_ALIASES = {
    "upcoming": "notstarted", "scheduled": "notstarted", "ns": "notstarted",
    "finished": "finished", "ft": "finished", "live": "inprogress",
}


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
      status   — "notstarted" | "inprogress" | "finished"
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
    if isinstance(filters.get("status"), str):
        raw_status = filters["status"].lower()
        filters["status"] = _STATUS_ALIASES.get(raw_status, raw_status)
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


def get_player_stats_page(api_key: str, **params) -> dict:
    """Return one page from BSD's per-match player-statistics endpoint.

    The endpoint is deliberately kept separate from event detail: lineup rows
    describe who was selected, while player-stats rows contain minutes, rating,
    xG/xA and on-ball/defensive actions.
    """
    return _get("/api/player-stats/", api_key, **params)


def get_all_player_stats(api_key: str, **filters) -> list[dict]:
    """Fetch all paginated player-stat rows matching *filters*.

    Typical filters are ``event=<bsd_event_id>`` or ``player=<player_id>``.
    """
    results: list[dict] = []
    limit = int(filters.pop("limit", _DEFAULT_LIMIT))
    offset = 0
    while True:
        page = get_player_stats_page(api_key, limit=limit, offset=offset, **filters)
        batch = page.get("results") or []
        results.extend(batch)
        if not page.get("next") or not batch:
            break
        offset += limit
    return results


def get_standings(api_key: str, league_id: int | str, **params) -> dict:
    """Return BSD's current standings payload for a league."""
    return _get(f"/api/leagues/{league_id}/standings/", api_key, **params)


def odds_comparison(api_key: str, event_id: int | str) -> dict:
    """Multi-bookmaker odds comparison for one event (v2 API).

    Only populated close to kickoff (empirically: same-day for lower-profile
    leagues; "bookmakers_count": 0 / "markets": {} well in advance). Shape:
        {"bookmakers_count": int, "markets": {
            "1x2": {"HOME": {"bookmakers": {slug: {"decimal_odds": float, ...}}}, "DRAW": {...}, "AWAY": {...}},
            "over_under_25": {"over@2.5": {...}, "under@2.5": {...}},
            ...
        }}
    The top-level odds_* fields on /api/events/ list/detail responses are
    NOT a multi-bookmaker list despite earlier assumptions — this endpoint
    is the real source for bookmaker-level data.
    """
    return _get(f"/api/v2/events/{event_id}/odds/comparison/", api_key)


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
    return str(
        event.get("event_date")
        or event.get("date")
        or event.get("kickoff")
        or ""
    ).strip()


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
    raw = (event.get("statistics") or event.get("stats")
           or event.get("live_stats") or event.get("sr_stats") or {})
    if not isinstance(raw, dict):
        return {"home": {}, "away": {}}
    return {
        "home": dict(raw.get("home") or {}),
        "away": dict(raw.get("away") or {}),
    }


def _num(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalized_match_statistics(event: dict) -> dict[str, dict[str, float | None]]:
    """Normalize BSD's evolving team-stat field names.

    BSD currently puts finished-match team statistics in ``live_stats``. Older
    payloads used ``statistics``/``stats`` and some list endpoints expose a
    handful of scalar fields directly. This function gives fixture ingestion a
    single stable shape while retaining only numeric, model-relevant fields.
    """
    raw = match_statistics(event)
    out: dict[str, dict[str, float | None]] = {"home": {}, "away": {}}
    aliases = {
        "shots": ("total_shots", "shots"),
        "sot": ("shots_on_target", "shots_on_goal", "sot"),
        "corners": ("corner_kicks", "corners"),
        # BSD has used all three of these xG shapes in the wild: nested
        # live_stats.expected_goals, side_xg_live, and actual_side_xg.
        "xg": ("expected_goals", "xg", "xg_live", "actual_xg"),
        "possession": ("ball_possession", "possession"),
        "yellow_cards": ("yellow_cards", "yellow_card"),
        "red_cards": ("red_cards", "red_card"),
    }
    for side in ("home", "away"):
        src = raw[side]
        for target, names in aliases.items():
            value = next((src.get(name) for name in names if src.get(name) is not None), None)
            if value is None:
                direct_names = [f"{side}_{name}" for name in names]
                if target == "xg":
                    direct_names = [f"actual_{side}_xg", f"{side}_xg_live",
                                    f"{side}_xg", f"{side}_expected_goals"] + direct_names
                elif target == "shots":
                    direct_names = [f"{side}_shots", f"{side}_total_shots"] + direct_names
                elif target == "sot":
                    direct_names = [f"{side}_sot", f"{side}_shots_on_target"] + direct_names
                elif target == "corners":
                    direct_names = [f"{side}_corners", f"{side}_corner_kicks"] + direct_names
                value = next((event.get(name) for name in direct_names
                              if event.get(name) is not None), None)
            out[side][target] = _num(value)
    return out


def _pair(value) -> tuple[float | None, float | None]:
    if not isinstance(value, dict):
        return None, None
    return _num(value.get("home")), _num(value.get("away"))


def fixture_detail_fields(event: dict) -> dict:
    """Return optional fixture columns available on an event/detail payload."""
    stats = normalized_match_statistics(event)
    home_obj = event.get("home_team_obj") or {}
    away_obj = event.get("away_team_obj") or {}
    neutral_raw = (event.get("is_neutral_ground")
                   if "is_neutral_ground" in event else event.get("neutral"))
    shootout = event.get("penalty_shootout") or event.get("shootout_score")
    et_score = event.get("extra_time_score") or event.get("aet_score")
    so_h, so_a = _pair(shootout)
    et_h, et_a = _pair(et_score)
    explicit_scope = event.get("result_scope") or event.get("score_scope")
    finished = str(event.get("status") or "").lower() in {
        "finished", "ft", "aet", "pen"
    }
    result_scope = (str(explicit_scope).lower() if explicit_scope else
                    "penalties" if so_h is not None and so_a is not None else
                    "extra_time" if et_h is not None and et_a is not None else
                    "regulation" if finished else "")
    explicit_winner = (event.get("shootout_winner") or
                       event.get("penalty_shootout_winner") or "")
    shootout_winner = str(explicit_winner).lower() if explicit_winner else (
        "home" if so_h is not None and so_a is not None and so_h > so_a else
        "away" if so_h is not None and so_a is not None and so_a > so_h else "")
    fields = {
        "home_id": event.get("home_team_id") or home_obj.get("id") or "",
        "away_id": event.get("away_team_id") or away_obj.get("id") or "",
        # Do not infer false when the detail payload omitted the field: a
        # missing value must not erase a known neutral-ground flag on merge.
        "neutral": int(bool(neutral_raw)) if neutral_raw is not None else None,
        "home_goals_ht": _num(event.get("home_score_ht")),
        "away_goals_ht": _num(event.get("away_score_ht")),
        "home_goals_ft": _num(event.get("home_score_ft")
                               if event.get("home_score_ft") is not None
                               else event.get("home_score")),
        "away_goals_ft": _num(event.get("away_score_ft")
                               if event.get("away_score_ft") is not None
                               else event.get("away_score")),
        "extra_time_home_goals": et_h,
        "extra_time_away_goals": et_a,
        "shootout_home": so_h,
        "shootout_away": so_a,
        "shootout_winner": shootout_winner,
        "result_scope": result_scope,
        "round_name": event.get("round_name") or "",
        "round_number": _num(event.get("round_number")),
        "group_name": event.get("group_name") or "",
        "venue": str((event.get("venue") or {}).get("name", "")
                      if isinstance(event.get("venue"), dict)
                      else (event.get("venue") or "")).strip(),
    }
    for side in ("home", "away"):
        fields[f"{side}_shots"] = stats[side]["shots"]
        fields[f"{side}_sot"] = stats[side]["sot"]
        fields[f"{side}_corners"] = stats[side]["corners"]
        fields[f"{side}_xg"] = stats[side]["xg"]
        fields[f"{side}_possession"] = stats[side]["possession"]
        fields[f"{side}_yellow_cards"] = stats[side]["yellow_cards"]
        fields[f"{side}_red_cards"] = stats[side]["red_cards"]
    if stats["home"]["xg"] is not None and stats["away"]["xg"] is not None:
        fields["xg_source"] = "bsd"
    return fields
