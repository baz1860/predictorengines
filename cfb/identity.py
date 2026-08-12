"""Reviewed CFB team identity at provider-ingest boundaries.

CFBD schedule team IDs are the canonical identity. Provider names are accepted
only when they are either the canonical CFBD display name or a reviewed alias
in ``data/team_aliases.json``. Fuzzy and prefix matching are deliberately
report-only: an unknown spelling must block quote ingestion instead of being
silently attached to the most plausible team.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ALIASES_JSON = DATA / "team_aliases.json"
ODDS_API_URL = (
    "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds/"
)


def fold(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = text.encode("ascii", "ignore").decode().lower().replace("&", "and")
    return re.sub(r"[^a-z0-9 ]", "", text).strip()


def _id_text(value: object) -> str:
    text = str(value or "").strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


# (season, schedule mtime) -> catalog; (aliases mtime, season, provider) -> index.
# resolve() is called per provider name at odds ingest; without this it parsed
# the full schedule JSON twice per call.
_CATALOG_CACHE: dict = {}
_ALIAS_CACHE: dict = {}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def schedule_catalog(season: int) -> dict:
    """Return canonical CFBD identities indexed by ID and normalised name."""
    path = DATA / f"schedule_{int(season)}.json"
    # NB: distinct name — `key` is reused as a loop variable below.
    catalog_key = (int(season), _mtime(path))
    cached = _CATALOG_CACHE.get(catalog_key)
    if cached is not None:
        return cached
    try:
        games = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable CFB schedule identity source: {path}") from exc
    if not isinstance(games, list) or not games:
        raise ValueError(f"empty CFB schedule identity source: {path}")

    by_id: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    ambiguous: set[str] = set()
    for game in games:
        if int(game.get("season", season)) != int(season):
            continue
        for side in ("home", "away"):
            team_id = _id_text(game.get(f"{side}Id"))
            name = str(game.get(f"{side}Team") or "").strip()
            if not team_id or not name:
                continue
            prior = by_id.get(team_id)
            if prior and prior["canonical"] != name:
                raise ValueError(
                    f"CFBD team ID {team_id} has conflicting names: "
                    f"{prior['canonical']!r}, {name!r}"
                )
            record = {"team_id": team_id, "canonical": name}
            by_id[team_id] = record
            key = fold(name)
            if key in by_name and by_name[key]["team_id"] != team_id:
                ambiguous.add(key)
            else:
                by_name[key] = record
    for key in ambiguous:
        by_name.pop(key, None)
    if len(by_id) < 100:
        raise ValueError(f"CFB schedule identity coverage too small: {len(by_id)} teams")
    catalog = {"by_id": by_id, "by_name": by_name, "ambiguous_names": sorted(ambiguous)}
    _CATALOG_CACHE.clear()   # one season's catalog is in play at a time
    _CATALOG_CACHE[catalog_key] = catalog
    return catalog


def _load_aliases() -> dict:
    try:
        raw = json.loads(ALIASES_JSON.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable CFB alias registry: {ALIASES_JSON}") from exc
    aliases = raw.get("aliases") if isinstance(raw, dict) else None
    if not isinstance(aliases, list):
        raise ValueError("CFB alias registry 'aliases' must be a list")
    return raw


def alias_index(season: int, provider: str | None = None) -> dict[str, dict]:
    """Validated reviewed aliases for a provider and season."""
    cache_key = (int(season), provider,
                 _mtime(ALIASES_JSON), _mtime(DATA / f"schedule_{int(season)}.json"))
    cached = _ALIAS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    catalog = schedule_catalog(season)
    out: dict[str, dict] = {}
    for row in _load_aliases()["aliases"]:
        if not isinstance(row, dict):
            raise ValueError("CFB alias rows must be objects")
        row_provider = str(row.get("provider") or "").strip()
        if provider and row_provider not in (provider, "*"):
            continue
        valid_from = int(row.get("valid_from") or season)
        valid_to = int(row.get("valid_to") or season)
        if not valid_from <= int(season) <= valid_to:
            continue
        alias = str(row.get("alias") or "").strip()
        team_id = _id_text(row.get("team_id"))
        canonical = str(row.get("canonical") or "").strip()
        if not alias or not row_provider or not team_id or not canonical:
            raise ValueError(f"incomplete CFB alias row: {row!r}")
        target = catalog["by_id"].get(team_id)
        if target is None or target["canonical"] != canonical:
            raise ValueError(
                f"CFB alias target is not canonical for {season}: {row!r}"
            )
        key = fold(alias)
        prior = out.get(key)
        if prior and prior["team_id"] != team_id:
            raise ValueError(f"ambiguous reviewed CFB alias: {alias!r}")
        out[key] = {**target, "provider": row_provider,
                    "reviewed_at": str(row.get("reviewed_at") or "")}
    _ALIAS_CACHE.clear()
    _ALIAS_CACHE[cache_key] = out
    return out


def resolve(name: object, season: int, provider: str | None = None) -> dict | None:
    """Resolve one provider spelling without guessing; return canonical ID/name."""
    key = fold(name)
    if not key:
        return None
    catalog = schedule_catalog(season)
    canonical = catalog["by_name"].get(key)
    if canonical:
        return {**canonical, "match_mode": "canonical"}
    reviewed = alias_index(season, provider).get(key)
    if reviewed:
        return {**reviewed, "match_mode": "reviewed_alias"}
    return None


def review_names(names: list[str], season: int,
                 provider: str | None = None) -> list[dict]:
    """Read-only provider-name review report."""
    rows = []
    for name in sorted(set(str(value) for value in names if str(value).strip())):
        match = resolve(name, season, provider)
        rows.append({
            "provider": provider or "",
            "provider_name": name,
            "status": "resolved" if match else "review_required",
            "team_id": "" if match is None else match["team_id"],
            "canonical": "" if match is None else match["canonical"],
            "match_mode": "" if match is None else match["match_mode"],
        })
    return rows


# Fields the model actually acts on. A change to any of these can move a
# prediction, a stake, or a fixture match, and must force human re-review.
# Everything else CFBD ships (its own pregame Elo, excitement index, venue
# metadata, …) is informational and deliberately excluded: a control that
# fires on changes nobody needs to read is a control that stops being read.
SCHEDULE_DECISION_FIELDS = (
    "id", "season", "week", "seasonType", "startDate",
    "homeId", "homeTeam", "homeClassification",
    "awayId", "awayTeam", "awayClassification",
    "neutralSite", "completed",
)


def schedule_identity(season: int, path: Path | None = None) -> dict:
    """Canonical, order-independent view of the decision-relevant schedule."""
    source = path or (DATA / f"schedule_{int(season)}.json")
    games = json.loads(Path(source).read_text())
    if not isinstance(games, list) or not games:
        raise ValueError(f"empty CFB schedule identity source: {source}")
    rows = sorted(
        ({field: game.get(field) for field in SCHEDULE_DECISION_FIELDS}
         for game in games),
        key=lambda row: str(row.get("id")),
    )
    return {"season": int(season), "events": len(rows), "rows": rows}


def schedule_identity_sha256(season: int, path: Path | None = None) -> str:
    """Hash of the decision-relevant schedule content only.

    Replaces hashing the raw provider file: CFBD backfilling its own
    `homePregameElo`/`awayPregameElo` used to trip the reviewed-schedule gate
    even though the model never reads those fields.
    """
    payload = json.dumps(schedule_identity(season, path),
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def registry_version(season: int) -> str:
    h = hashlib.sha256()
    h.update((DATA / f"schedule_{int(season)}.json").read_bytes())
    h.update(ALIASES_JSON.read_bytes())
    return h.hexdigest()[:16]


def _odds_api_key() -> str:
    try:
        from api_keys import get_key
        return (get_key("the-odds-api", env="THE_ODDS_API_KEY") or "").strip()
    except Exception:
        return (os.environ.get("THE_ODDS_API_KEY") or "").strip()


def live_provider_names(through: date) -> list[str]:
    """Fetch current Odds API event names for a bounded identity-only review."""
    key = _odds_api_key()
    if not key:
        raise ValueError("The Odds API key is not configured")
    query = urllib.parse.urlencode({
        "apiKey": key, "regions": "us", "markets": "h2h",
        "oddsFormat": "decimal",
    })
    with urllib.request.urlopen(f"{ODDS_API_URL}?{query}", timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("Odds API identity response is not a list")
    names: list[str] = []
    for event in payload:
        try:
            kickoff = date.fromisoformat(str(event["commence_time"])[:10])
        except (KeyError, TypeError, ValueError):
            continue
        if kickoff <= through:
            names.extend([event.get("home_team", ""), event.get("away_team", "")])
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--provider", default="the-odds-api")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--names", type=Path,
                        help="JSON Odds API payload or newline-delimited names")
    source.add_argument("--live", action="store_true",
                        help="fetch current provider names for read-only review")
    parser.add_argument("--through", type=date.fromisoformat,
                        help="last kickoff date included with --live")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    names: list[str] = []
    if args.live:
        if args.through is None:
            raise SystemExit("--live requires --through YYYY-MM-DD")
        try:
            names = live_provider_names(args.through)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    elif args.names:
        text = args.names.read_text()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            names = [line.strip() for line in text.splitlines() if line.strip()]
        else:
            if not isinstance(payload, list):
                raise SystemExit("provider JSON must be a list of events")
            for event in payload:
                names.extend([event.get("home_team", ""), event.get("away_team", "")])
    rows = review_names(names, args.season, args.provider)
    unresolved = sum(row["status"] != "resolved" for row in rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
                "provider", "provider_name", "status", "team_id", "canonical",
                "match_mode"])
            writer.writeheader()
            writer.writerows(rows)
    print(f"CFB identity registry {registry_version(args.season)}: "
          f"{len(rows) - unresolved}/{len(rows)} resolved")
    if unresolved:
        for row in rows:
            if row["status"] != "resolved":
                print(f"  REVIEW REQUIRED: {row['provider_name']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
