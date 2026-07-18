"""Bzzoiro v2 enrichment collector for shotmaps, lineups and player data.

This is deliberately an enrichment cache, not a silent replacement for the
canonical fixture table.  The v2 feed has richer spatial data than the legacy
BSD event endpoint, so each response is retained with its source, retrieval
time and raw payload.  Downstream experiments can then join the flattened
features to fixtures without changing production predictions.

Typical use::

    python3 -m club_soccer.bsd_enrichment --collect \
        --league-ids 1,3,4,5,6 --date-from 2025-08-01 --date-to 2026-05-31
    python3 -m club_soccer.bsd_enrichment --summary
    python3 -m club_soccer.bsd_enrichment --join

The collector uses the authenticated BSD REST surface behind the Bzzoiro MCP
tools.  It is intentionally resumable: one event is one JSON file, and a
failed endpoint is recorded rather than causing an otherwise useful event to
be discarded.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api_keys import get_key
from bsd_client import (
    get_all_v2_events,
    get_all_v2_leagues,
    get_v2_event,
    get_v2_event_incidents,
    get_v2_event_lineups,
    get_v2_event_player_stats,
    get_v2_event_stats,
)

from . import model as M
from .names import simplify

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ENRICHMENT_DIR = DATA / "bsd_enrichment"
MANIFEST = DATA / "bsd_enrichment_manifest.csv"
FLAT_OUTPUT = DATA / "bsd_enriched_matches.csv"
SCHEMA_VERSION = 1

DEFAULT_LEAGUES = (1, 3, 4, 5, 6)  # PL, La Liga, Serie A, Bundesliga, Ligue 1
DEFAULT_DATE_FROM = "2025-08-01"
DEFAULT_DATE_TO = "2026-05-31"
LEAGUE_NAMES = {
    1: "Premier League", 3: "La Liga", 4: "Serie A",
    5: "Bundesliga", 6: "Ligue 1",
}


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _fixture_key(value: Any) -> str:
    """Normalise integer fixture IDs after pandas nullable merges."""
    try:
        if pd.isna(value):
            return ""
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _event_path(event_id: int | str) -> Path:
    return ENRICHMENT_DIR / f"event_{event_id}.json"


def _json_text(path: Path, payload: dict) -> None:
    """Atomically persist one enrichment record."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _record_complete(record: dict[str, Any], required: tuple[str, ...] | None = None) -> bool:
    if record.get("schema") != SCHEMA_VERSION:
        return False
    if not all(key in record for key in
               ("event", "stats", "lineups", "incidents", "player_stats")):
        return False
    if required is None:
        return True
    available = set(record.get("endpoints") or
                    ("event", "stats", "lineups", "incidents", "player_stats"))
    return set(required).issubset(available)


def _fetch_one(api_key: str, event: dict, force: bool = False,
               endpoints: tuple[str, ...] | None = None) -> tuple[int, bool, str]:
    event_id = int(event["id"])
    endpoints = endpoints or ("event", "stats", "lineups", "incidents", "player_stats")
    path = _event_path(event_id)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}
    if path.exists() and not force:
        try:
            if _record_complete(existing, endpoints):
                return event_id, False, "cached"
        except Exception:
            pass

    record: dict[str, Any] = existing if existing.get("schema") == SCHEMA_VERSION else {
        "schema": SCHEMA_VERSION,
        "source": "bzzoiro_v2",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "errors": {},
    }
    record.update({"source": "bzzoiro_v2", "retrieved_at": datetime.now(timezone.utc).isoformat(),
                  "event": event})
    old_endpoints = set(record.get("endpoints") or [])
    record["endpoints"] = sorted(old_endpoints | set(endpoints))
    record.setdefault("errors", {})
    calls = {
        "event": lambda: get_v2_event(api_key, event_id),
        "stats": lambda: get_v2_event_stats(api_key, event_id),
        "lineups": lambda: get_v2_event_lineups(api_key, event_id),
        "incidents": lambda: get_v2_event_incidents(api_key, event_id),
        "player_stats": lambda: get_v2_event_player_stats(api_key, event_id),
    }
    for name in endpoints:
        call = calls[name]
        try:
            value = call()
            if name == "event" and isinstance(value, dict):
                # Keep the indexed event's league label and any provenance
                # fields if the detail response omits them.
                value = {**event, **value}
            record[name] = value
        except Exception as exc:
            record[name] = {} if name != "player_stats" else []
            record["errors"][name] = str(exc)
    _json_text(path, record)
    return event_id, True, "fetched"


def _league_map(api_key: str) -> dict[int, str]:
    return {int(row["id"]): str(row.get("name") or row["id"])
            for row in get_all_v2_leagues(api_key)}


def collect(api_key: str, league_ids: list[int], date_from: str,
            date_to: str, max_events: int | None = None,
            oldest_first: bool = False, workers: int = 6,
            force: bool = False,
            endpoints: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Collect v2 enrichments for a bounded set of finished events."""
    ENRICHMENT_DIR.mkdir(parents=True, exist_ok=True)
    leagues = _league_map(api_key)
    events: dict[int, dict] = {}
    for league_id in league_ids:
        rows = get_all_v2_events(
            api_key, league_id=int(league_id), date_from=date_from,
            date_to=date_to, status="finished", limit=50)
        for row in rows:
            if row.get("id") is None:
                continue
            item = dict(row)
            item["league_name"] = leagues.get(int(league_id), str(league_id))
            events[int(row["id"])] = item

    ordered = sorted(events.values(),
                     key=lambda row: str(row.get("event_date") or ""),
                     reverse=not oldest_first)
    if max_events is not None:
        ordered = ordered[:max(0, int(max_events))]

    fetched = cached = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(_fetch_one, api_key, event, force, endpoints): event
                   for event in ordered}
        for index, future in enumerate(as_completed(futures), 1):
            event = futures[future]
            try:
                _, did_fetch, status = future.result()
                if did_fetch:
                    fetched += 1
                else:
                    cached += 1
            except Exception as exc:
                failed += 1
                status = f"error: {exc}"
            if index == 1 or index % 25 == 0 or index == len(futures):
                print(f"  enrichment {index}/{len(futures)}: {status}")

    manifest_rows = []
    for event in ordered:
        path = _event_path(event["id"])
        try:
            record = json.loads(path.read_text())
        except Exception:
            continue
        stats = record.get("stats") or {}
        lineups = record.get("lineups") or {}
        manifest_rows.append({
            "event_id": event["id"],
            "date": str(event.get("event_date") or "")[:10],
            "league_id": event.get("league_id"),
            "league": event.get("league_name", ""),
            "season_id": event.get("season_id"),
            "home": event.get("home_team", ""),
            "away": event.get("away_team", ""),
            "path": str(path.relative_to(DATA)),
            "has_shotmap": bool(stats.get("shotmap")),
            "has_momentum": bool(stats.get("momentum")),
            "lineup_status": lineups.get("lineup_status", ""),
            "player_rows": int((record.get("player_stats") or {}).get("count", 0)
                                if isinstance(record.get("player_stats"), dict) else 0),
            "error_count": len(record.get("errors") or {}),
        })
    pd.DataFrame(manifest_rows).to_csv(MANIFEST, index=False)
    return {"events": len(ordered), "fetched": fetched, "cached": cached,
            "failed": failed, "manifest": str(MANIFEST)}


def _refresh_player_one(api_key: str, path: Path) -> tuple[int, str]:
    """Refresh only the v2 player payload for one cached event."""
    try:
        record = json.loads(path.read_text())
        event_id = int((record.get("event") or {}).get("id"))
        record["player_stats"] = get_v2_event_player_stats(api_key, event_id)
        errors = record.setdefault("errors", {})
        errors.pop("player_stats", None)
        record["player_stats_retrieved_at"] = datetime.now(timezone.utc).isoformat()
        _json_text(path, record)
        return event_id, "refreshed"
    except Exception as exc:
        return int(path.stem.split("_")[-1]), f"error: {exc}"


def refresh_players(api_key: str, workers: int = 6,
                    max_events: int | None = None) -> dict[str, Any]:
    """Backfill player stats without refetching the larger event payloads."""
    paths = sorted(ENRICHMENT_DIR.glob("event_*.json"))
    if max_events is not None:
        paths = paths[:max(0, int(max_events))]
    refreshed = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_refresh_player_one, api_key, path) for path in paths]
        for index, future in enumerate(as_completed(futures), 1):
            _, status = future.result()
            if status == "refreshed":
                refreshed += 1
            else:
                failed += 1
            if index == 1 or index % 50 == 0 or index == len(futures):
                print(f"  player stats {index}/{len(futures)}: {status}")
    return {"events": len(paths), "refreshed": refreshed, "failed": failed}


def _side_from_team(event: dict, team_id: Any) -> str | None:
    try:
        value = int(team_id)
        if value == int(event.get("home_team_id")):
            return "home"
        if value == int(event.get("away_team_id")):
            return "away"
    except (TypeError, ValueError):
        return None
    return None


def _shot_features(shots: list[dict], side: str) -> dict[str, Any]:
    rows = [shot for shot in shots if bool(shot.get("home")) == (side == "home")]
    xg = sum(_num(row.get("xg"), 0.0) or 0.0 for row in rows)
    xgot_values = [_num(row.get("xgot")) for row in rows]
    xgot = sum(value for value in xgot_values if value is not None)
    goals = sum(str(row.get("type") or "").casefold() == "goal" for row in rows)
    return {
        f"{side}_shotmap_shots": len(rows),
        f"{side}_shotmap_xg": round(float(xg), 6),
        f"{side}_shotmap_xgot": round(float(xgot), 6),
        f"{side}_shotmap_goals": int(goals),
        f"{side}_shotmap_assisted": sum(str(row.get("sit") or "").casefold() == "assisted"
                                         for row in rows),
        f"{side}_shotmap_set_piece": sum("set" in str(row.get("sit") or "").casefold()
                                          for row in rows),
        f"{side}_shotmap_inside_box": sum(
            _num((row.get("pos") or {}).get("x"), 100.0) <= 18.0
            for row in rows if isinstance(row.get("pos"), dict)),
    }


def _team_stats(stats: dict, side: str) -> dict[str, Any]:
    raw = (stats.get("stats") or {}).get(side) or {}
    out: dict[str, Any] = {}
    fields = (
        "expected_goals", "total_shots", "shots_on_target", "shots_inside_box",
        "shots_outside_box", "ball_possession", "passes", "accurate_passes",
        "pass_accuracy_pct", "corner_kicks", "final_third_entries",
        "touches_in_penalty_area", "big_chances", "fouls", "yellow_cards",
        "interceptions", "clearances", "recoveries", "tackles_won",
    )
    for field in fields:
        value = raw.get(field)
        if isinstance(value, dict):
            value = value.get("actual") or value.get("value")
        if value is not None:
            out[f"{side}_bsd_{field}"] = _num(value, value)
    xg = raw.get("xg")
    if xg is None:
        xg = raw.get("expected_goals")
    if isinstance(xg, dict):
        xg = xg.get("actual")
    if xg is not None:
        out[f"{side}_bsd_xg"] = _num(xg, xg)
    return out


def _team_player_stats(event: dict, rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for side in ("home", "away"):
        team_id = event.get(f"{side}_team_id")
        selected = [row for row in rows if _side_from_team(event, row.get("team_id")) == side]
        for name, field in (("minutes", "minutes_played"), ("xg", "expected_goals"),
                            ("xa", "expected_assists"), ("shots", "total_shots"),
                            ("sot", "shots_on_target"), ("passes", "total_pass"),
                            ("accurate_passes", "accurate_pass"),
                            ("tackles", "total_tackle"), ("interceptions", "interception"),
                            ("clearances", "total_clearance"), ("goals", "goals")):
            values = [_num(row.get(field), 0.0) or 0.0 for row in selected]
            out[f"{side}_player_{name}"] = round(float(sum(values)), 6)
        ratings = [_num(row.get("rating")) for row in selected]
        ratings = [value for value in ratings if value is not None]
        out[f"{side}_player_count"] = len(selected)
        out[f"{side}_player_rating_mean"] = round(float(np.mean(ratings)), 6) if ratings else np.nan
        out[f"{side}_player_xg_rows"] = sum(row.get("expected_goals") is not None
                                            for row in selected)
    return out


def _lineup_features(lineups: dict) -> dict[str, Any]:
    out = {"lineup_status": lineups.get("lineup_status", "")}
    groups = lineups.get("lineups") or {}
    for side in ("home", "away"):
        group = groups.get(side) or {}
        starters = group.get("starters") or group.get("starting_xi") or group.get("players") or []
        substitutes = group.get("substitutes") or group.get("bench") or []
        out[f"{side}_lineup_starters"] = len(starters)
        out[f"{side}_lineup_substitutes"] = len(substitutes)
    return out


def _incident_features(incidents: dict) -> dict[str, Any]:
    rows = incidents.get("incidents") if isinstance(incidents, dict) else []
    rows = rows if isinstance(rows, list) else []
    out = {"incident_count": len(rows)}
    for key, terms in {
        "goals": ("goal",), "yellow_cards": ("yellow",),
        "red_cards": ("red",), "substitutions": ("sub",),
        "penalties": ("penalt",), "var": ("var",),
    }.items():
        out[f"incident_{key}"] = sum(any(term in str(row).casefold() for term in terms)
                                      for row in rows)
    return out


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one raw record into a match-level experimental row."""
    event = record.get("event") or {}
    stats = record.get("stats") or {}
    row: dict[str, Any] = {
        "event_id": event.get("id"),
        "date": str(event.get("event_date") or "")[:10],
        "season_id": event.get("season_id"),
        "league_id": event.get("league_id"),
        "league": event.get("league_name") or
                  LEAGUE_NAMES.get(_num(event.get("league_id")), ""),
        "home": event.get("home_team", ""),
        "away": event.get("away_team", ""),
        "home_id": event.get("home_team_id"),
        "away_id": event.get("away_team_id"),
        "home_goals": event.get("home_score"),
        "away_goals": event.get("away_score"),
        "neutral": bool(event.get("is_neutral_ground", False)),
        "venue_id": event.get("venue_id"),
        "referee_id": event.get("referee_id"),
        "source": "bzzoiro_v2",
    }
    for side in ("home", "away"):
        row.update(_team_stats(stats, side))
        row.update(_shot_features(stats.get("shotmap") or [], side))
    players = record.get("player_stats") or []
    if isinstance(players, dict):
        players = players.get("player_stats") or players.get("results") or []
    row.update(_team_player_stats(event, players if isinstance(players, list) else []))
    row.update(_lineup_features(record.get("lineups") or {}))
    row.update(_incident_features(record.get("incidents") or {}))
    return row


def load_flat() -> pd.DataFrame:
    rows = []
    for path in sorted(ENRICHMENT_DIR.glob("event_*.json")):
        try:
            rows.append(flatten_record(json.loads(path.read_text())))
        except Exception:
            continue
    return pd.DataFrame(rows)


def join_to_fixtures(fixtures: pd.DataFrame | None = None,
                     output: Path = FLAT_OUTPUT) -> dict[str, Any]:
    """Join enriched rows to canonical fixtures without modifying fixtures.csv."""
    enriched = load_flat()
    if fixtures is None:
        fixtures = M.load_fixtures()
    if enriched.empty:
        return {"enriched": 0, "matched": 0, "output": str(output)}
    left = enriched.copy()
    right = fixtures.copy()
    left["_key"] = (left["date"].astype(str).str[:10] + "|" +
                     left["home"].map(simplify) + "|" + left["away"].map(simplify))
    right["_key"] = (right["date"].dt.strftime("%Y-%m-%d") + "|" +
                      right["home"].map(simplify) + "|" + right["away"].map(simplify))
    right["_date_ts"] = right["date"].dt.normalize()
    lookup = right.drop_duplicates("_key")[["_key", "fixture_id", "competition", "type"]]
    out = left.merge(lookup, on="_key", how="left", suffixes=("", "_fixture"))
    out["fixture_joined"] = out["fixture_id"].notna()
    out["join_method"] = np.where(out["fixture_joined"], "exact", "")
    # BSD and football-data occasionally assign a late UTC kickoff to the
    # adjacent local calendar date.  Only use this fallback for the same
    # simplified home/away pair and within one day, never a loose team-only
    # match.  It recovers timezone artefacts without creating new identities.
    used = set(out.loc[out["fixture_joined"], "fixture_id"].astype(str))
    for idx, item in out.loc[~out["fixture_joined"]].iterrows():
        try:
            event_date = pd.Timestamp(item["date"]).normalize()
            candidates = right[
                (right["home"].map(simplify) == simplify(item["home"])) &
                (right["away"].map(simplify) == simplify(item["away"])) &
                ((right["_date_ts"] - event_date).abs() <= pd.Timedelta(days=1)) &
                (~right["fixture_id"].astype(str).isin(used))
            ].copy()
        except Exception:
            candidates = pd.DataFrame()
        if len(candidates) != 1:
            continue
        candidate = candidates.iloc[0]
        for field in ("fixture_id", "competition", "type"):
            out.at[idx, field] = candidate[field]
        out.at[idx, "fixture_joined"] = True
        out.at[idx, "join_method"] = "same_pair_1d"
        used.add(str(candidate["fixture_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["_key"]).to_csv(output, index=False)
    return {"enriched": int(len(out)),
            "matched": int(out["fixture_joined"].sum()),
            "exact": int((out["join_method"] == "exact").sum()),
            "same_pair_1d": int((out["join_method"] == "same_pair_1d").sum()),
            "shotmap": int(out.filter(like="shotmap_shots").notna().any(axis=1).sum()),
            "output": str(output)}


def candidate_fixtures(fixtures: pd.DataFrame | None = None,
                       enriched: pd.DataFrame | None = None,
                       fill_only: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a non-production fixture frame with Bzzoiro xG filled in.

    The candidate only fills training observations: the model never reads a
    test row's post-match xG during prediction.  Existing complete xG pairs
    are preserved by default so this is an incremental coverage experiment,
    not an untracked provider replacement.
    """
    base = M.load_fixtures() if fixtures is None else fixtures.copy(deep=True)
    if enriched is None:
        if not FLAT_OUTPUT.exists():
            return base, {"eligible": 0, "filled": 0, "skipped_existing": 0}
        enriched = pd.read_csv(FLAT_OUTPUT, low_memory=False)
    needed = {"fixture_id", "home_shotmap_xg", "away_shotmap_xg", "fixture_joined"}
    if not needed.issubset(enriched.columns):
        return base, {"eligible": 0, "filled": 0, "skipped_existing": 0}
    eligible = enriched[
        enriched["fixture_joined"].astype(str).str.casefold().eq("true") &
        enriched["home_shotmap_xg"].notna() & enriched["away_shotmap_xg"].notna()
    ].copy()
    if eligible.empty or "fixture_id" not in base.columns:
        return base, {"eligible": int(len(eligible)), "filled": 0,
                      "skipped_existing": 0}
    eligible["_fixture_key"] = eligible["fixture_id"].map(_fixture_key)
    eligible = eligible.drop_duplicates("_fixture_key")
    base["_fixture_key"] = base["fixture_id"].map(_fixture_key)
    lookup = eligible.set_index("_fixture_key")
    filled = skipped = 0
    for idx, row in base.iterrows():
        key = row["_fixture_key"]
        if key not in lookup.index:
            continue
        item = lookup.loc[key]
        existing = (pd.notna(row.get("home_xg")) and pd.notna(row.get("away_xg")))
        if fill_only and existing:
            skipped += 1
            continue
        base.at[idx, "home_xg"] = float(item["home_shotmap_xg"])
        base.at[idx, "away_xg"] = float(item["away_shotmap_xg"])
        if "xg_source" in base.columns:
            base.at[idx, "xg_source"] = "bzzoiro_v2"
        filled += 1
    base.drop(columns=["_fixture_key"], inplace=True)
    return base, {"eligible": int(len(eligible)), "filled": filled,
                  "skipped_existing": skipped}


def summary() -> dict[str, Any]:
    frame = load_flat()
    if frame.empty:
        return {"events": 0, "shotmap": 0, "confirmed_lineups": 0,
                "player_rows": 0}
    by_league = {}
    for league, group in frame.groupby("league", dropna=False):
        label = str(league or "unknown")
        by_league[label] = {
            "events": int(len(group)),
            "shotmap": int((group.filter(like="shotmap_shots").sum(axis=1) > 0).sum()),
            "confirmed_lineups": int((group["lineup_status"] == "confirmed").sum()),
            "player_rows": int(group.filter(like="player_count").sum(axis=1).sum()),
        }
    return {
        "events": int(len(frame)),
        "shotmap": int((frame.filter(like="shotmap_shots").sum(axis=1) > 0).sum()),
        "confirmed_lineups": int((frame["lineup_status"] == "confirmed").sum()),
        "player_rows": int(frame.filter(like="player_count").sum(axis=1).sum()),
        "date_from": str(frame["date"].min()),
        "date_to": str(frame["date"].max()),
        "by_league": by_league,
    }


def evaluate_xg_candidate(min_train: int = 200) -> dict[str, Any]:
    """Compare the fill-only xG candidate on the pulled 2025/26 window."""
    from . import validate as V

    base = M.load_fixtures()
    candidate, coverage = candidate_fixtures(base)
    window = {"test_from": DEFAULT_DATE_FROM, "test_to": "2026-06-01"}
    _, baseline = V.walk_forward(min_train=min_train, fixtures=base, **window)
    _, challenger = V.walk_forward(min_train=min_train, fixtures=candidate, **window)
    return {
        "coverage": coverage,
        "baseline": baseline,
        "candidate": challenger,
        "delta_brier": round(challenger["brier"] - baseline["brier"], 6),
        "delta_log_loss": round(challenger["log_loss"] - baseline["log_loss"], 6),
        "window": window,
        "promotion": False,
        "note": "diagnostic only; Bzzoiro post-match xG is used only in prior training rows",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Bzzoiro v2 spatial football data.")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--join", action="store_true",
                    help="flatten the raw cache and join it to fixtures.csv")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--evaluate", action="store_true",
                    help="compare fill-only Bzzoiro shotmap xG with incumbent walk-forward")
    ap.add_argument("--refresh-players", action="store_true",
                    help="refresh only v2 player stats in the raw event cache")
    ap.add_argument("--league-ids", default=",".join(map(str, DEFAULT_LEAGUES)))
    ap.add_argument("--date-from", default=DEFAULT_DATE_FROM)
    ap.add_argument("--date-to", default=DEFAULT_DATE_TO)
    ap.add_argument("--max-events", type=int)
    ap.add_argument("--oldest-first", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stats-only", action="store_true",
                    help="pull only the fast v2 stats/shotmap endpoint")
    ap.add_argument("--endpoints",
                    help="comma-separated v2 endpoints to fetch incrementally")
    ap.add_argument("--api-key")
    args = ap.parse_args()
    if args.collect:
        key = args.api_key or get_key("bsd", env="BSD_API_KEY")
        if not key:
            raise SystemExit("No BSD key — set BSD_API_KEY or use --api-key.")
        league_ids = [int(value) for value in args.league_ids.split(",") if value.strip()]
        if args.endpoints:
            endpoints = tuple(value.strip() for value in args.endpoints.split(",")
                              if value.strip())
        else:
            endpoints = ("stats",) if args.stats_only else None
        print(collect(key, league_ids, args.date_from, args.date_to,
                      max_events=args.max_events, oldest_first=args.oldest_first,
                      workers=args.workers, force=args.force, endpoints=endpoints))
    if args.refresh_players:
        key = args.api_key or get_key("bsd", env="BSD_API_KEY")
        if not key:
            raise SystemExit("No BSD key — set BSD_API_KEY or use --api-key.")
        print(refresh_players(key, workers=args.workers,
                              max_events=args.max_events))
    if args.join:
        print(join_to_fixtures())
    if args.summary:
        print(summary())
    if args.evaluate:
        print(json.dumps(evaluate_xg_candidate(), indent=2))
    if not (args.collect or args.join or args.summary or args.refresh_players or args.evaluate):
        ap.print_help()


if __name__ == "__main__":
    main()
