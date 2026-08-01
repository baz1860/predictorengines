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
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ALIASES_JSON = DATA / "team_aliases.json"


def fold(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = text.encode("ascii", "ignore").decode().lower().replace("&", "and")
    return re.sub(r"[^a-z0-9 ]", "", text).strip()


def _id_text(value: object) -> str:
    text = str(value or "").strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def schedule_catalog(season: int) -> dict:
    """Return canonical CFBD identities indexed by ID and normalised name."""
    path = DATA / f"schedule_{int(season)}.json"
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
    return {"by_id": by_id, "by_name": by_name, "ambiguous_names": sorted(ambiguous)}


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


def registry_version(season: int) -> str:
    h = hashlib.sha256()
    h.update((DATA / f"schedule_{int(season)}.json").read_bytes())
    h.update(ALIASES_JSON.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--provider", default="the-odds-api")
    parser.add_argument("--names", type=Path,
                        help="JSON Odds API payload or newline-delimited names")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    names: list[str] = []
    if args.names:
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
