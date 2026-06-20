#!/usr/bin/env python3
"""Experimental WhoScored adapter for cached/public match-centre payloads.

This module intentionally does not automate browser sessions, CAPTCHA flows,
cookies, proxies, or other access-control workarounds. It accepts raw JSON or a
plain HTML page that already contains a `matchCentreData` object, then normalizes
that payload into the same canonical World Cup CSV tables as the paid providers.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.worldcup.names import known_teams, require_known_team  # noqa: E402
from scripts.worldcup import live_data as LD  # noqa: E402

SOURCE = "whoscored_scrape"

FIXTURE_KEYS = ["event_id", "source"]
AVAILABILITY_KEYS = ["team", "player", "provider_fixture_id", "source"]
LINEUP_KEYS = ["event_id", "team", "player", "role", "source"]
STATS_KEYS = ["event_id", "team", "source"]

SHOT_TYPES = {"goal", "savedshot", "missedshots", "shotonpost"}


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. Save a WhoScored matchCentreData JSON file "
            "first, or use --input-html/--url with a page that contains one."
        )
    try:
        return coerce_payload(json.loads(p.read_text()))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid JSON: {exc}") from exc


def load_html(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. Save the WhoScored match page HTML first, "
            "or pass a public page URL with --url."
        )
    return extract_match_centre_json(p.read_text())


def load_har(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist.")
    try:
        har = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid HAR/JSON: {exc}") from exc
    payloads = []
    for entry in har.get("log", {}).get("entries", []):
        content = entry.get("response", {}).get("content", {})
        text = content.get("text")
        if not text:
            continue
        if content.get("encoding") == "base64":
            try:
                text = base64.b64decode(text).decode("utf-8", errors="replace")
            except Exception:
                continue
        payload = _payload_from_text(text)
        if payload:
            payloads.append(payload)
    return _dedupe_payloads(payloads)


def load_dir(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist.")
    if not p.is_dir():
        raise ValueError(f"{p} is not a directory.")
    payloads: list[dict[str, Any]] = []
    for child in sorted(p.iterdir()):
        if child.suffix.lower() == ".json":
            try:
                payloads.append(load_json(child))
            except ValueError:
                continue
        elif child.suffix.lower() in (".html", ".htm"):
            try:
                payloads.append(load_html(child))
            except ValueError:
                continue
        elif child.suffix.lower() == ".har":
            payloads.extend(load_har(child))
    return _dedupe_payloads(payloads)


def fetch_url(url: str) -> dict[str, Any]:
    """Fetch a public page once with a plain urllib request and parse its JSON."""
    req = urllib.request.Request(url, headers={
        "Accept": "text/html,application/json",
        "User-Agent": "SoccerPredictionResearch/1.0",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    return extract_match_centre_json(html)


def extract_match_centre_json(html: str) -> dict[str, Any]:
    """Extract a strict JSON `matchCentreData` object from HTML/script text."""
    payload = _payload_from_text(html)
    if payload:
        return payload
    raise ValueError(
        "Could not extract matchCentreData JSON from HTML. Use --input-json "
        "with an object produced by an existing scraper, or --input-har from "
        "a browser network export."
    )


def _payload_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return coerce_payload(json.loads(stripped))
        except Exception:
            pass
    markers = (
        "matchCentreData",
        "matchCentreDataJson",
        "window.__MATCH_CENTRE_DATA__",
    )
    for marker in markers:
        pos = text.find(marker)
        while pos >= 0:
            start = text.find("{", pos)
            if start < 0:
                break
            try:
                blob = _balanced_object(text, start)
                return coerce_payload(_loads_json_like(blob))
            except Exception:
                pos = text.find(marker, pos + len(marker))
    return None


def _loads_json_like(blob: str) -> dict[str, Any]:
    try:
        return json.loads(blob)
    except json.JSONDecodeError as first_exc:
        cleaned = re.sub(r"\b(undefined|NaN|Infinity|-Infinity)\b", "null", blob)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "embedded matchCentreData is not strict JSON. Export the "
                "parsed object from an existing scraper with json.dump(...)."
            ) from exc


def coerce_payload(obj: Any) -> dict[str, Any]:
    """Return the nested WhoScored match payload from common wrappers."""
    if isinstance(obj, str):
        payload = _payload_from_text(obj)
        if payload:
            return payload
        raise ValueError("string input did not contain matchCentreData JSON")
    if not isinstance(obj, dict):
        raise ValueError("input must be a JSON object")
    if _looks_like_payload(obj):
        validate_payload(obj)
        return obj
    if "log" in obj and "entries" in obj.get("log", {}):
        raise ValueError("this looks like a HAR file; use --input-har instead")
    for key in (
        "matchCentreData",
        "matchCentreDataJson",
        "match_data",
        "matchData",
        "data",
        "payload",
    ):
        value = obj.get(key)
        if value is None:
            continue
        try:
            return coerce_payload(value)
        except ValueError:
            continue
    found = _find_payload(obj)
    if found:
        validate_payload(found)
        return found
    raise ValueError(
        "JSON did not contain a recognizable WhoScored match payload "
        "(expected home/homeTeam and away/awayTeam objects)."
    )


def validate_payload(payload: dict[str, Any]) -> None:
    home_obj, away_obj = _team_obj(payload, "home"), _team_obj(payload, "away")
    if not _team_name(home_obj) or not _team_name(away_obj):
        raise ValueError(
            "WhoScored payload is missing home/away team names. Capture the "
            "full matchCentreData object, not just events."
        )


def _looks_like_payload(obj: dict[str, Any]) -> bool:
    return bool(_team_obj(obj, "home") and _team_obj(obj, "away"))


def _find_payload(value: Any, depth: int = 0) -> dict[str, Any] | None:
    if depth > 7:
        return None
    if isinstance(value, dict):
        if _looks_like_payload(value):
            return value
        for child in value.values():
            found = _find_payload(child, depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_payload(child, depth + 1)
            if found:
                return found
    return None


def _dedupe_payloads(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for payload in payloads:
        key = (
            _first(payload, "matchId", "id", "match_id", default=""),
            _team_name(_team_obj(payload, "home")),
            _team_name(_team_obj(payload, "away")),
            _first(payload, "startDate", "kickoff", "date", default=""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(payload)
    return out


def _balanced_object(text: str, start: int) -> str:
    depth = 0
    quote = ""
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced JSON object")


def normalize_payload(payload: dict[str, Any], fetched_at: str | None = None,
                      teams: set[str] | None = None) -> dict[str, pd.DataFrame]:
    payload = coerce_payload(payload)
    fetched = fetched_at or LD.utc_now()
    known = teams if teams is not None else known_teams()
    meta = fixture_meta(payload, fetched, known)
    return {
        "fixtures": pd.DataFrame([meta["fixture"]]),
        "availability": parse_availability(payload, meta, fetched, known),
        "lineups": parse_lineups(payload, meta, fetched, known),
        "match_stats": parse_match_stats(payload, meta, fetched, known),
    }


def fixture_meta(payload: dict[str, Any], fetched_at: str,
                 teams: set[str]) -> dict[str, Any]:
    home_obj, away_obj = _team_obj(payload, "home"), _team_obj(payload, "away")
    home = require_known_team(_team_name(home_obj), teams, "whoscored fixture")
    away = require_known_team(_team_name(away_obj), teams, "whoscored fixture")
    kickoff = _first(payload, "startDate", "kickoff", "date", default="")
    match_date = LD._local_match_date(kickoff)
    competition = _competition(payload)
    event_id = LD._event_id(match_date, home, away, competition)
    provider_id = _first(payload, "matchId", "id", "match_id", default="")
    rnd = _first(payload, "round", "stageName", "tournamentStage", default="")
    fixture = {
        "event_id": event_id,
        "provider_fixture_id": provider_id,
        "match_date": match_date,
        "kickoff_utc": kickoff,
        "home": home,
        "away": away,
        "competition": competition,
        "round": rnd,
        "group": LD._group_from_round(rnd),
        "status_long": _status(payload),
        "status_short": _first(payload, "statusCode", "periodCode",
                               "fixtureStatus", default=""),
        "elapsed": _first(payload, "expandedMaxMinute", "minute", default=""),
        "venue_name": _first(payload, "venueName", "venue", default=""),
        "venue_city": _first(payload, "venueCity", default=""),
        "provider_home_id": _team_id(home_obj),
        "provider_away_id": _team_id(away_obj),
        "source": SOURCE,
        "fetched_at": fetched_at,
    }
    return {
        "event_id": event_id,
        "provider_fixture_id": provider_id,
        "match_date": match_date,
        "home": home,
        "away": away,
        "home_id": _team_id(home_obj),
        "away_id": _team_id(away_obj),
        "fixture": fixture,
    }


def parse_lineups(payload: dict[str, Any], meta: dict[str, Any],
                  fetched_at: str, teams: set[str]) -> pd.DataFrame:
    rows = []
    confirmed = _lineup_confirmed(payload)
    status = "confirmed" if confirmed else "scraped"
    for side in ("home", "away"):
        team_obj = _team_obj(payload, side)
        team = require_known_team(_team_name(team_obj), teams, "whoscored lineup")
        formation = _formation(team_obj)
        for player in _players(team_obj):
            role = _player_role(player)
            if role not in ("starter", "bench"):
                continue
            rows.append({
                "event_id": meta["event_id"],
                "provider_fixture_id": meta["provider_fixture_id"],
                "match_date": meta["match_date"],
                "team": team,
                "player": _player_name(player),
                "provider_team_id": _team_id(team_obj),
                "provider_player_id": _first(player, "playerId", "id",
                                             "player_id", default=""),
                "starter": role == "starter",
                "role": role,
                "position": _first(player, "position", "positionText",
                                   "usualPosition", "playingPosition",
                                   default=""),
                "shirt_number": _first(player, "shirtNo", "shirtNumber",
                                       "number", "jerseyNumber", default=""),
                "formation": formation,
                "lineup_status": status,
                "published_at": _first(payload, "lineupPublishedAt",
                                       "lineupConfirmedAt", default=fetched_at),
                "source": SOURCE,
                "fetched_at": fetched_at,
            })
    return pd.DataFrame(rows)


def parse_availability(payload: dict[str, Any], meta: dict[str, Any],
                       fetched_at: str, teams: set[str]) -> pd.DataFrame:
    rows = []
    for side in ("home", "away"):
        team_obj = _team_obj(payload, side)
        team = require_known_team(_team_name(team_obj), teams,
                                  "whoscored availability")
        for item in _missing_players(team_obj):
            reason = _first(item, "reason", "status", "description", "type",
                            default="")
            status, certainty, affects = LD.classify_availability(
                reason, _first(item, "type", default=""))
            rows.append({
                "team": team,
                "player": _player_name(item),
                "status": status,
                "reason": reason,
                "certainty": certainty,
                "affects_availability": bool(affects),
                "source": SOURCE,
                "fetched_at": fetched_at,
                "provider_fixture_id": meta["provider_fixture_id"],
                "provider_team_id": _team_id(team_obj),
                "provider_player_id": _first(item, "playerId", "id",
                                             "player_id", default=""),
            })
    return pd.DataFrame(rows)


def parse_match_stats(payload: dict[str, Any], meta: dict[str, Any],
                      fetched_at: str, teams: set[str]) -> pd.DataFrame:
    rows_by_team: dict[Any, dict[str, Any]] = {}
    for side in ("home", "away"):
        team_obj = _team_obj(payload, side)
        team = require_known_team(_team_name(team_obj), teams, "whoscored stats")
        tid = _team_id(team_obj)
        rows_by_team[tid] = {
            "event_id": meta["event_id"],
            "provider_fixture_id": meta["provider_fixture_id"],
            "match_date": meta["match_date"],
            "team": team,
            "source": SOURCE,
            "fetched_at": fetched_at,
            "shots": 0,
            "shots_on_target": 0,
            "corners": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "fouls": 0,
            "xg": 0.0,
        }

    for ev in _events(payload):
        tid = _first(ev, "teamId", "team_id", default=None)
        if tid not in rows_by_team:
            continue
        row = rows_by_team[tid]
        etype = _event_type(ev)
        low = etype.lower()
        if _is_shot(ev, low):
            row["shots"] += 1
            if _is_shot_on_target(ev, low):
                row["shots_on_target"] += 1
            xg = _float(_first(ev, "xG", "xg", "expectedGoals", default=None))
            if xg is not None:
                row["xg"] += xg
        if "corner" in low or _has_qualifier(ev, "cornertaken"):
            row["corners"] += 1
        if "foul" in low:
            row["fouls"] += 1
        card = _display(_first(ev, "cardType", "card", default="")).lower()
        if card:
            if "red" in card:
                row["red_cards"] += 1
            elif "yellow" in card:
                row["yellow_cards"] += 1

    return pd.DataFrame(rows_by_team.values())


def write_canonical(tables: dict[str, pd.DataFrame]) -> None:
    if not tables["fixtures"].empty:
        LD._upsert_csv(LD.FIXTURES_CSV, tables["fixtures"], FIXTURE_KEYS)
    if not tables["availability"].empty:
        LD._upsert_csv(LD.AVAILABILITY_CSV, tables["availability"],
                       AVAILABILITY_KEYS)
    if not tables["lineups"].empty:
        LD._upsert_csv(LD.LINEUPS_CSV, tables["lineups"], LINEUP_KEYS)
    if not tables["match_stats"].empty:
        LD._upsert_csv(LD.MATCH_STATS_CSV, tables["match_stats"], STATS_KEYS)


def _team_obj(payload: dict[str, Any], side: str) -> dict[str, Any]:
    obj = payload.get(side) or payload.get(f"{side}Team") or {}
    if isinstance(obj, dict):
        return obj
    return {}


def _team_name(obj: dict[str, Any]) -> str:
    value = _first(obj, "name", "teamName", "title", default="")
    if isinstance(value, dict):
        return str(_first(value, "name", "displayName", default=""))
    return str(value or "")


def _team_id(obj: dict[str, Any]) -> Any:
    return _first(obj, "teamId", "id", "team_id", default="")


def _players(team_obj: dict[str, Any]) -> list[dict[str, Any]]:
    value = _first(team_obj, "players", "lineup", "squad", default=[])
    if isinstance(value, dict):
        value = list(value.values())
    return [p for p in value or [] if isinstance(p, dict)]


def _missing_players(team_obj: dict[str, Any]) -> list[dict[str, Any]]:
    vals = []
    for key in ("missingPlayers", "unavailablePlayers", "absences"):
        value = team_obj.get(key) or []
        if isinstance(value, dict):
            value = list(value.values())
        vals.extend(p for p in value if isinstance(p, dict))
    return vals


def _events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("events") or payload.get("matchCentreEvents") or []
    if isinstance(value, dict):
        value = list(value.values())
    return [e for e in value if isinstance(e, dict)]


def _formation(team_obj: dict[str, Any]) -> str:
    value = _first(team_obj, "formation", "formationName", default="")
    if value:
        return str(value)
    formations = team_obj.get("formations") or []
    if isinstance(formations, list) and formations:
        first = formations[0]
        if isinstance(first, dict):
            return str(_first(first, "formationName", "formation", default=""))
    return ""


def _player_role(player: dict[str, Any]) -> str:
    if _truthy(_first(player, "isFirstEleven", "isStarter", "isStarting",
                     default=False)):
        return "starter"
    if _truthy(_first(player, "isSubstitute", "isBench", default=False)):
        return "bench"
    pos = str(_first(player, "position", "positionText", default="")).lower()
    if pos in ("sub", "substitute", "bench"):
        return "bench"
    return ""


def _player_name(player: dict[str, Any]) -> str:
    value = _first(player, "name", "playerName", "shortName", "fullName",
                   default="")
    if isinstance(value, dict):
        return str(_first(value, "name", "displayName", default=""))
    return str(value or "")


def _lineup_confirmed(payload: dict[str, Any]) -> bool:
    for key in ("isLineupConfirmed", "lineupsConfirmed", "lineupConfirmed"):
        if key in payload:
            return _truthy(payload[key])
    return False


def _competition(payload: dict[str, Any]) -> str:
    for key in ("competition", "league", "tournament", "tournamentName"):
        value = payload.get(key)
        if isinstance(value, dict):
            value = _first(value, "name", "title", default="")
        if value:
            return str(value)
    return "FIFA World Cup"


def _status(payload: dict[str, Any]) -> str:
    value = _first(payload, "status", "statusText", "fixtureStatus", default="")
    if isinstance(value, dict):
        return str(_first(value, "displayName", "name", "value", default=""))
    return str(value or "")


def _event_type(event: dict[str, Any]) -> str:
    return _display(_first(event, "type", "eventType", default=""))


def _display(value: Any) -> str:
    if isinstance(value, dict):
        return str(_first(value, "displayName", "name", "value", default=""))
    return str(value or "")


def _is_shot(event: dict[str, Any], event_type: str) -> bool:
    return _truthy(event.get("isShot")) or event_type in SHOT_TYPES


def _is_shot_on_target(event: dict[str, Any], event_type: str) -> bool:
    return (
        _truthy(event.get("isShotOnTarget"))
        or _truthy(event.get("isGoal"))
        or event_type in {"goal", "savedshot"}
    )


def _has_qualifier(event: dict[str, Any], wanted: str) -> bool:
    quals = event.get("qualifiers") or []
    if isinstance(quals, dict):
        quals = list(quals.values())
    for q in quals:
        name = _display(_first(q, "type", "qualifierType", default=q))
        if name.lower().replace(" ", "") == wanted:
            return True
    return False


def _first(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def _load_inputs(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    for path in args.input_json or []:
        yield load_json(path)
    for path in args.input_html or []:
        yield load_html(path)
    for path in args.input_har or []:
        yield from load_har(path)
    for path in args.input_dir or []:
        yield from load_dir(path)
    for url in args.url or []:
        yield fetch_url(url)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Normalize cached/public WhoScored match data")
    ap.add_argument("--input-json", action="append", default=[],
                    help="Path to a raw matchCentreData JSON file.")
    ap.add_argument("--input-html", action="append", default=[],
                    help="Path to an HTML page containing matchCentreData.")
    ap.add_argument("--input-har", action="append", default=[],
                    help="Path to a browser HAR export containing matchCentreData.")
    ap.add_argument("--input-dir", action="append", default=[],
                    help="Directory of .json, .html, and .har captures.")
    ap.add_argument("--url", action="append", default=[],
                    help="Public page URL to fetch once with plain urllib.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and print row counts without writing CSVs.")
    args = ap.parse_args()

    try:
        payloads = list(_load_inputs(args))
    except Exception as exc:
        ap.error(str(exc))
    if not payloads:
        ap.error(
            "provide at least one --input-json, --input-html, --input-har, "
            "--input-dir, or --url; no recognizable payloads were found"
        )

    fetched = LD.utc_now()
    all_tables = {"fixtures": [], "availability": [], "lineups": [],
                  "match_stats": []}
    for payload in payloads:
        if not args.dry_run:
            LD._save_raw("whoscored", "match", payload, fetched)
        tables = normalize_payload(payload, fetched)
        for key, df in tables.items():
            all_tables[key].append(df)

    merged = {
        key: (pd.concat(frames, ignore_index=True)
              if frames else pd.DataFrame())
        for key, frames in all_tables.items()
    }
    if not args.dry_run:
        write_canonical(merged)
    counts = ", ".join(f"{k}={len(v)}" for k, v in merged.items())
    mode = "dry-run" if args.dry_run else "wrote"
    print(f"whoscored: {mode} {counts}")


if __name__ == "__main__":
    main()
