"""Player-level availability and form features for the Club Soccer engine.

Data pipeline
-------------
1.  BSD /api/events/?status=finished  →  event detail calls  →  per-player stats
2.  build_player_stats() aggregates into a rolling contribution store
    (JSON cache at club_soccer/data/player_stats_cache.json)
3.  For upcoming matches, unavailable_players is embedded in the BSD events
    response — no extra call needed.
4.  PlayerFeatureStore.adjustments_for_match() converts absences into team
    lambda multipliers that model.predict() can consume.

The two multipliers per team
----------------------------
    attack_mult   = 1 + att_delta     (< 1 when key attackers are out)
    defense_mult  = 1 + def_delta     (> 1 when key defenders are out; opponent
                                       scores more)

Both clamped to [0.80, 1.25] so no single player can swing predictions by more
than 25%.

Usage
-----
    from club_soccer.player_features import PlayerFeatureStore
    store = PlayerFeatureStore()
    store.refresh(api_key="YOUR_BSD_KEY")   # build/update stats cache

    # When running edge analysis on an upcoming BSD event dict:
    adj = store.adjustments_for_match(event)
    # -> {"home": {"attack_mult": 0.88, "defense_mult": 1.04},
    #     "away": {"attack_mult": 1.0,  "defense_mult": 1.0}}

    # Or supply names directly (e.g. from absences.csv / manual list):
    adj = store.adjustments_from_names(
        home_team="Arsenal", unavailable_home=["Saka", "Havertz"],
        away_team="Chelsea", unavailable_away=[],
    )

Stand-alone refresh
-------------------
    python3 -m club_soccer.player_features --refresh [--max-events 500]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import unicodedata
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bsd_client import (
    get_all_events, get_all_player_stats, get_event, event_date_utc,
    league_name as bsd_league_name,
    unavailable_players as bsd_unavailable,
    lineups as bsd_lineups,
)
from api_keys import get_key

CACHE_SCHEMA_VERSION = 3
EVENT_MANIFEST_KEY = "_event_manifest"

DATA = HERE / "data"
PLAYER_CACHE = DATA / "player_stats_cache.json"
STATS_CACHE  = DATA / "bsd_cache"             # shared with seed_real.py
MODEL_PARAMS = DATA / "model_params.json"
ABSENCES_CSV = DATA / "absences_club.csv"
ABSENCES_COLUMNS = ["recorded_at", "match_date", "competition", "team", "player", "reason", "status"]

# Position → fraction of impact on ATTACK (own goals); rest lands on DEFENCE.
# A missing striker mostly reduces the team's own scoring.
# A missing GK/DF mostly increases the opponent's scoring.
POS_ATT_SHARE: dict[str, float] = {
    "GK": 0.05, "DF": 0.15, "MF": 0.45, "FW": 0.90,
}
POS_DEF_SHARE: dict[str, float] = {
    "GK": 0.95, "DF": 0.80, "MF": 0.40, "FW": 0.05,
}

# Maximum total fractional adjustment per side (cap at 25% shift).
ADJ_CAP = 0.25

# Positional fallback xG-per-90 when we have no match-stats history for a player.
# Derived from broad Premier League averages — used only as a last resort.
_POS_XG_DEFAULT: dict[str, float] = {
    "GK": 0.00, "DF": 0.03, "MF": 0.07, "FW": 0.22,
}
_POS_XA_DEFAULT: dict[str, float] = {
    "GK": 0.00, "DF": 0.02, "MF": 0.08, "FW": 0.08,
}

# Player xG is sparse even in a large cache. Use a recency-weighted empirical
# rate with a positional prior equivalent to six 90-minute appearances. This
# keeps one 12-minute cameo from creating an extreme absence adjustment.
PLAYER_XG_HALF_LIFE_DAYS = 180.0
PLAYER_XG_PRIOR_MINUTES = 540.0

# BSD stat field names in event detail (player level)
_PLAYER_XG_FIELDS  = ("xg", "expected_goals", "xGoal", "xgoal")
_PLAYER_XA_FIELDS  = ("xa", "expected_assists", "xAssist", "xassist", "key_passes")
_PLAYER_MIN_FIELDS = ("minutes", "minutes_played", "time", "minutesPlayed")
_PLAYER_POS_FIELDS = ("position", "pos", "positionId", "specific_position")
_PLAYER_ID_FIELDS = ("player_id", "id", "api_id", "provider_player_id")


# ── name normalisation ────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lowercase, strip accents, remove punctuation — used for fuzzy name matching."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def _name_tokens(s: str) -> set[str]:
    return set(_norm(s).split())


def _names_match(a: str, b: str, min_shared: int = 1) -> bool:
    """True if the two names share at least min_shared tokens."""
    return len(_name_tokens(a) & _name_tokens(b)) >= min_shared


# ── BSD per-player stat extraction ────────────────────────────────────────────

def _get_first(d: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _extract_player_entry(player_dict: dict, default_mins: float = 0.0) -> dict:
    """Pull (name, position, xg, xa, minutes) from a BSD player dict.

    BSD returns player stats in one of two shapes:
      A) flat dict with keys like "xg", "position", "minutes"
      B) nested "stats" sub-dict
    """
    base = player_dict if isinstance(player_dict, dict) else {}
    nested_player = base.get("player") if isinstance(base.get("player"), dict) else {}
    stats = base.get("stats") or base.get("player_stats") or base

    name = str(base.get("name") or base.get("player_name") or
               nested_player.get("name") or nested_player.get("short_name") or "")
    pos_raw = str(_get_first(base, _PLAYER_POS_FIELDS, "") or
                  _get_first(nested_player, _PLAYER_POS_FIELDS, "") or
                  _get_first(stats, _PLAYER_POS_FIELDS, "")).upper().strip()

    # Normalise position to GK/DF/MF/FW
    pos = _normalise_pos(pos_raw)

    xg  = _safe_float(_get_first(stats, _PLAYER_XG_FIELDS, 0.0))
    xa  = _safe_float(_get_first(stats, _PLAYER_XA_FIELDS, 0.0))
    mins_raw = _get_first(stats, _PLAYER_MIN_FIELDS, None)
    if mins_raw is not None:
        mins = _safe_float(mins_raw)
    elif "sub_in" in base or "sub_out" in base:
        # Current BSD lineup rows carry substitution minutes rather than
        # minutes_played. Unused substitutes are passed with default_mins=0.
        sub_in = _safe_float(base.get("sub_in"), -1.0)
        sub_out = _safe_float(base.get("sub_out"), -1.0)
        if sub_in >= 0:
            mins = max(0.0, 90.0 - sub_in)
        elif sub_out >= 0:
            mins = max(0.0, sub_out)
        else:
            mins = float(default_mins)
    else:
        mins = float(default_mins)

    player_id = _get_first(base, _PLAYER_ID_FIELDS, None)
    if player_id is None:
        player_id = _get_first(nested_player, _PLAYER_ID_FIELDS, None)
    try:
        player_id = int(player_id) if player_id is not None else None
    except (TypeError, ValueError):
        player_id = None

    metrics = {}
    for key in ("rating", "goals", "goal_assist", "total_shots", "shots_on_target",
                "key_pass", "total_pass", "accurate_pass", "duel_won", "duel_lost",
                "total_tackle", "won_tackle", "total_clearance", "interception",
                "ball_recovery", "possession_lost", "yellow_card", "red_card",
                "saves", "goals_conceded"):
        if key in base:
            value = base[key]
            try:
                metrics[key] = float(value) if value is not None else None
            except (TypeError, ValueError):
                metrics[key] = None

    return {"name": name, "player_id": player_id, "pos": pos, "xg": xg,
            "xa": xa, "mins": mins, "metrics": metrics}


def _normalise_pos(raw: str) -> str:
    raw = raw.upper().strip()
    if raw in ("GK", "G", "GOALKEEPER", "PORTERO"):
        return "GK"
    if raw.startswith("D") or raw in ("CB", "LB", "RB", "LWB", "RWB", "SW", "DEFENDER"):
        return "DF"
    if raw.startswith("M") or raw in ("CM", "DM", "AM", "CDM", "CAM", "MIDFIELDER",
                                       "MEDIOCENTRO"):
        return "MF"
    if raw.startswith("F") or raw in ("ST", "CF", "LW", "RW", "SS", "FORWARD",
                                       "STRIKER", "DELANTERO"):
        return "FW"
    return "MF"   # default to midfield when unknown


def _players_from_event(event_detail: dict) -> list[tuple[dict, str | None]]:
    """Extract all player entries from a BSD event detail response, each
    tagged with the side ("home"/"away") the source shape assigned it to.

    Tries several known BSD response shapes, all of which group players by
    side already:
      1. event_detail["lineups"]["home|away"]["starters|bench"] list
      2. event_detail["players"]["home|away"] list
      3. event_detail["home_players"] / event_detail["away_players"] list
      4. (fallback) a single flat "players" list with no side grouping —
         side is None; the caller falls back to a positional-order guess
         and must mark those entries side_confident=False.
    """
    out: list[tuple[dict, str | None]] = []

    seen: set[tuple[str, str]] = set()

    def _drain(lst, side: str | None, default_mins: float = 0.0):
        if isinstance(lst, list):
            for p in lst:
                entry = _extract_player_entry(p, default_mins=default_mins)
                if entry["name"]:
                    identity = (str(entry.get("player_id") or ""), _norm(entry["name"]))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    out.append((entry, side))

    # Shape 1: lineups
    lineups = event_detail.get("lineups") or {}
    for side in ("home", "away"):
        grp = lineups.get(side) or {}
        # Current BSD shape is players + substitutes. Older payloads used
        # starters/bench; accept both without treating unused substitutes as
        # 90-minute appearances.
        _drain(grp.get("players") or grp.get("starters") or grp.get("starting_xi") or [],
               side, default_mins=90.0)
        _drain(grp.get("substitutes") or grp.get("bench") or [], side, default_mins=0.0)

    # Shape 2: players dict keyed by side
    if not out:
        players_field = event_detail.get("players")
        if isinstance(players_field, dict):
            for side in ("home", "away"):
                _drain(players_field.get(side, []), side)

    # Shape 3: top-level home_players / away_players
    if not out:
        _drain(event_detail.get("home_players", []), "home")
        _drain(event_detail.get("away_players", []), "away")

    # Shape 4: unstructured flat list — no side information available
    if not out:
        players_field = event_detail.get("players")
        if isinstance(players_field, list):
            _drain(players_field, None)

    return out


# ── Player stats cache ────────────────────────────────────────────────────────

class PlayerFeatureStore:
    """Builds and queries the per-player rolling stats cache.

    Cache schema v3 — a JSON file:
        {
          "v": 3,
          "id:123": {
            "player_id": 123,
            "name": "original name",
            "pos": "FW",
            "apps": [{"date": "2026-01-01", "team": "Arsenal", "mins": 90.0,
                      "xg": 0.4, "xa": 0.1, "side_confident": true}, ...]
                     (most recent 60, dated so squads/transfers can be
                     derived by "most recent app wins" — see club_squads.py)
          },
          ...
        }
    Older caches are discarded on load — rebuild with `--from-cache` or
    `--refresh`. Stable provider IDs are preferred; name keys remain a
    deliberate fallback for sources without IDs.
    """

    ROLLING_N = 60   # appearances kept per player

    def __init__(self, cache_path: Path = PLAYER_CACHE):
        self._path = cache_path
        self._data: dict[str, dict] = {}
        self._team_xg: dict[str, float] = {}   # team -> mean xG-for from model_params
        self._team_xga: dict[str, float] = {}  # team -> mean xG-against
        self._loaded = False

    # ── I/O ────────────────────────────────────────────────────────────────

    def _player_records(self) -> "list[tuple[str, dict]]":
        """(key, record) pairs, skipping schema/manifest sentinels."""
        return [
            (k, v) for k, v in self._data.items()
            if not k.startswith("_") and isinstance(v, dict)
        ]

    def _event_manifest(self) -> dict[str, str]:
        """Return the processed-event signature map.

        It is stored as a list rather than a dict so older readers that treat
        every top-level dict as a player cannot mistake metadata for a player.
        """
        raw = self._data.get(EVENT_MANIFEST_KEY, [])
        if not isinstance(raw, list):
            return {}
        out: dict[str, str] = {}
        for item in raw:
            if isinstance(item, list) and len(item) == 2:
                out[str(item[0])] = str(item[1])
        return out

    def _set_event_manifest(self, manifest: dict[str, str]) -> None:
        self._data[EVENT_MANIFEST_KEY] = [
            [event_id, signature]
            for event_id, signature in sorted(manifest.items())
        ]

    def _remove_event_apps(self, event_id: str) -> None:
        """Remove a superseded event before ingesting its changed payload."""
        for _key, record in self._player_records():
            apps = record.get("apps")
            if isinstance(apps, list):
                record["apps"] = [
                    app for app in apps
                    if str(app.get("event_id", "")) != str(event_id)
                ]

    @staticmethod
    def _event_signature(event_path: Path, stats_path: Path) -> str:
        """Cheap change detector for Syncthing-managed immutable responses."""
        parts = []
        for path in (event_path, stats_path):
            try:
                stat = path.stat()
                parts.append(f"{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                parts.append("missing")
        return "|".join(parts)

    def load(self) -> "PlayerFeatureStore":
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
            except Exception:
                raw = {}
            if raw.get("v") != CACHE_SCHEMA_VERSION:
                if raw:
                    print(f"  player cache is schema v{raw.get('v', '1 (unversioned)')}, "
                          f"not v{CACHE_SCHEMA_VERSION} — discarding; rebuild with "
                          f"`python3 -m club_soccer.player_features --from-cache`")
                raw = {}
            self._data = raw
        self._load_team_baselines()
        self._loaded = True
        return self

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data["v"] = CACHE_SCHEMA_VERSION
        tmp = self._path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))
        tmp.replace(self._path)

    def _load_team_baselines(self) -> None:
        """Load team-level expected-goals baselines from model_params.json."""
        if not MODEL_PARAMS.exists():
            return
        try:
            params = json.loads(MODEL_PARAMS.read_text())
        except Exception:
            return
        base = float(params.get("global_avg", 1.3))
        import math
        for side, atk_key, def_key in (("home", "attack_xg", "defence_xg"),
                                        ("home", "attack", "defence")):
            atk_map = params.get(atk_key) or params.get("attack") or {}
            def_map = params.get(def_key) or params.get("defence") or {}
            for team, a in atk_map.items():
                d = def_map.get(team, 0.0)
                self._team_xg[team]  = base * math.exp(float(a))
                self._team_xga[team] = base * math.exp(float(d))
            break  # only need one iteration

    # ── Cache building ──────────────────────────────────────────────────────

    def refresh(self, api_key: str, max_events: int = 500,
                pause: float = 0.1, days_back: int = 60) -> int:
        """Fetch finished BSD events and update the player stats cache.

        Only fetches event detail for events that aren't already in the
        shared bsd_cache/ directory (used by seed_real.py).

        BSD's /api/events/ silently defaults to a ~7-day forward window when
        no date_from/date_to is given (see club_soccer/fetch.py), so a plain
        status="finished" query returns nothing outside a live match week —
        always pass an explicit lookback window.

        Returns the number of events processed.
        """
        if not self._loaded:
            self.load()

        STATS_CACHE.mkdir(parents=True, exist_ok=True)
        print(f"Fetching BSD finished events for player stats cache...")
        from datetime import datetime, timedelta, timezone
        today = datetime.now(timezone.utc).date()
        date_from = str(today - timedelta(days=days_back))
        date_to = str(today)
        try:
            events = get_all_events(api_key, status="finished",
                                    date_from=date_from, date_to=date_to)
        except Exception as exc:
            print(f"  ! BSD fetch failed: {exc}")
            return 0

        # BSD's pagination order is not guaranteed to be chronological. Pick
        # the newest window first, then ingest that selected window oldest to
        # newest so the rolling-appearance truncation really keeps the latest
        # 60 apps rather than whichever IDs happen to sort last.
        events = sorted(events, key=lambda ev: event_date_utc(ev), reverse=True)
        selected_events = sorted(events[:max_events], key=lambda ev: event_date_utc(ev))

        processed = 0
        stats_rows = 0
        manifest = self._event_manifest()
        for ev in selected_events:
            eid = str(ev.get("id") or "")
            if not eid:
                continue
            cache_file = STATS_CACHE / f"event_{eid}.json"
            if cache_file.exists():
                try:
                    detail = json.loads(cache_file.read_text())
                except Exception:
                    continue
            else:
                try:
                    detail = get_event(api_key, eid)
                    cache_file.write_text(json.dumps(detail, indent=2))
                    time.sleep(pause)
                except Exception as exc:
                    print(f"  ! event {eid}: {exc}")
                    continue

            date = (event_date_utc(ev) or event_date_utc(detail))[:10]
            home = str(ev.get("home_team") or "")
            away = str(ev.get("away_team") or "")
            stats_cache = STATS_CACHE / f"player_stats_{eid}.json"
            rows = None
            if stats_cache.exists():
                try:
                    rows = json.loads(stats_cache.read_text())
                except Exception:
                    rows = None
            if rows is None:
                try:
                    rows = get_all_player_stats(api_key, event=eid)
                    stats_cache.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
                    time.sleep(pause)
                except Exception as exc:
                    print(f"  ! player stats event {eid}: {exc}")
                    rows = []
            if rows:
                self._ingest_player_stats(rows, home, away, date, eid)
                stats_rows += len(rows)
            else:
                self._ingest_event_players(detail, home, away, date, eid)
            manifest[eid] = self._event_signature(cache_file, stats_cache)
            self._set_event_manifest(manifest)
            processed += 1
            if processed % 50 == 0:
                print(f"  ...{processed}/{len(selected_events)}")
            # A provider timeout should not discard a long refresh. Persist a
            # checkpoint after each modest batch; the per-event JSON caches are
            # already written independently, so a later run can resume safely.
            if processed % 25 == 0:
                self.save()

        self.save()
        print(f"  Player cache: {len(self._player_records())} players, "
              f"{processed} events processed, {stats_rows} player-stat rows.")
        return processed

    def refresh_from_cache(self) -> int:
        """Ingest only new or changed bsd_cache event bundles.

        The first run after upgrading has no manifest and performs one full
        migration. Later runs stat every event bundle but only read JSON for a
        new/changed event, avoiding thousands of parses and a 23 MiB rewrite.
        """
        if not self._loaded:
            self.load()
        if not STATS_CACHE.exists():
            return 0
        manifest = self._event_manifest()
        pending = []
        for cache_file in STATS_CACHE.glob("event_*.json"):
            eid = cache_file.stem.removeprefix("event_")
            stats_cache = STATS_CACHE / f"player_stats_{eid}.json"
            signature = self._event_signature(cache_file, stats_cache)
            if manifest.get(eid) == signature:
                continue
            try:
                detail = json.loads(cache_file.read_text())
            except Exception:
                continue
            pending.append((
                event_date_utc(detail), eid, cache_file, stats_cache, detail,
                signature,
            ))
        processed = 0
        for _, manifest_id, cache_file, stats_cache, detail, signature in sorted(
            pending, key=lambda item: item[0]
        ):
            date = event_date_utc(detail)[:10]
            home = str(detail.get("home_team") or "")
            away = str(detail.get("away_team") or "")
            eid = str(detail.get("id") or cache_file.stem.removeprefix("event_"))
            if manifest_id in manifest:
                self._remove_event_apps(eid)
            rows = None
            if stats_cache.exists():
                try:
                    rows = json.loads(stats_cache.read_text())
                except Exception:
                    rows = None
            if rows:
                self._ingest_player_stats(rows, home, away, date, eid)
            else:
                self._ingest_event_players(detail, home, away, date, eid)
            manifest[manifest_id] = signature
            processed += 1
        if processed:
            self._set_event_manifest(manifest)
            self.save()
        elif EVENT_MANIFEST_KEY not in self._data and manifest:
            self._set_event_manifest(manifest)
            self.save()
        return processed

    def refresh_player_stats_from_cached_events(
        self, api_key: str, max_events: int = 500,
        pause: float = 0.1, newest_first: bool = False,
    ) -> int:
        """Fetch player-stat rows for event details already in ``bsd_cache``.

        This is deliberately separate from :meth:`refresh`: a wide BSD
        ``/events/`` historical query can be slow or paginate unexpectedly,
        while the local event-detail cache already gives us an exact,
        bounded set of event IDs.  Each response is written before it is
        ingested, so an interrupted run can resume without repeating calls.
        Events are ingested chronologically regardless of request order so
        the per-player rolling cache retains the correct latest appearances.
        """
        if not self._loaded:
            self.load()
        STATS_CACHE.mkdir(parents=True, exist_ok=True)
        cached: list[tuple[str, Path, dict]] = []
        for cache_file in STATS_CACHE.glob("event_*.json"):
            try:
                detail = json.loads(cache_file.read_text())
                date = event_date_utc(detail)
            except Exception:
                continue
            if date:
                cached.append((date, cache_file, detail))
        cached.sort(key=lambda row: row[0], reverse=newest_first)
        selected = cached[:max(0, int(max_events))]
        fetched = 0
        stat_rows = 0
        for date, cache_file, detail in selected:
            eid = str(detail.get("id") or cache_file.stem.removeprefix("event_"))
            stats_cache = STATS_CACHE / f"player_stats_{eid}.json"
            if stats_cache.exists():
                continue
            try:
                rows = get_all_player_stats(api_key, event=eid)
                stats_cache.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
                fetched += 1
                stat_rows += len(rows)
                if pause:
                    time.sleep(pause)
            except Exception as exc:
                print(f"  ! player stats event {eid}: {exc}")
                continue

        # Rebuild from all local player-stat responses, including older ones
        # fetched in prior runs.  This avoids order-dependent results when a
        # run is interrupted between request and checkpoint.
        # Empty historical responses are still cached to avoid retry storms,
        # but they are not evidence that the existing player cache should be
        # discarded.  Rebuild only when at least one non-empty player-stat
        # payload was received.
        if stat_rows:
            self._data = {"v": CACHE_SCHEMA_VERSION}
            manifest: dict[str, str] = {}
            for _, cache_file, detail in sorted(cached, key=lambda row: row[0]):
                eid = str(detail.get("id") or cache_file.stem.removeprefix("event_"))
                stats_cache = STATS_CACHE / f"player_stats_{eid}.json"
                if not stats_cache.exists():
                    self._ingest_event_players(
                        detail, str(detail.get("home_team") or ""),
                        str(detail.get("away_team") or ""),
                        event_date_utc(detail)[:10], eid)
                    continue
                try:
                    rows = json.loads(stats_cache.read_text())
                except Exception:
                    continue
                home = str(detail.get("home_team") or "")
                away = str(detail.get("away_team") or "")
                detail_date = event_date_utc(detail)
                if rows:
                    self._ingest_player_stats(rows, home, away, detail_date[:10], eid)
                else:
                    self._ingest_event_players(detail, home, away,
                                               detail_date[:10], eid)
                manifest[eid] = self._event_signature(cache_file, stats_cache)
            self._set_event_manifest(manifest)
            self.save()
        print(f"  Player-stat responses fetched: {fetched}, rows: {stat_rows}.")
        return fetched

    def _ingest_event_players(self, detail: dict, home: str, away: str, date: str,
                              event_id: str | int | None = None) -> None:
        """Update the cache from one event's player list, using the side each
        BSD response shape already provides. Only entries with no side at all
        (an unstructured flat list — shape 4 in `_players_from_event`) fall
        back to a positional-order guess, tagged side_confident=False."""
        pairs = _players_from_event(detail)
        n = len(pairs)
        for i, (entry, side) in enumerate(pairs):
            if side == "home":
                team, confident = home, True
            elif side == "away":
                team, confident = away, True
            else:
                team, confident = (home if i < n // 2 else away), False
            self._update_player(entry, team, date, confident, event_id=event_id)

    def _ingest_player_stats(self, rows: list[dict], home: str, away: str,
                             date: str, event_id: str | int) -> None:
        """Ingest canonical rows from GET /api/player-stats/."""
        for row in rows:
            player = row.get("player") if isinstance(row.get("player"), dict) else {}
            team = str(player.get("team") or row.get("team") or "")
            if not team:
                team = home
            entry = _extract_player_entry(row, default_mins=0.0)
            if entry["name"] and entry["mins"] > 0:
                self._update_player(entry, team, date, True, event_id=event_id)

    def _update_player(self, entry: dict, team: str, date: str,
                       side_confident: bool = True,
                       event_id: str | int | None = None) -> None:
        player_id = entry.get("player_id")
        key = f"id:{player_id}" if player_id is not None else _norm(entry["name"])
        if not key:
            return
        if key not in self._data:
            self._data[key] = {
                "name": entry["name"],
                "player_id": player_id,
                "pos": entry["pos"],
                "apps": [],
            }
        rec = self._data[key]
        if entry["mins"] > 0:
            app = {
                "event_id": str(event_id) if event_id is not None else "",
                "date": date, "team": team,
                "mins": entry["mins"], "xg": entry["xg"], "xa": entry["xa"],
                "side_confident": side_confident,
                "metrics": entry.get("metrics", {}),
            }
            if event_id is not None:
                rec["apps"] = [a for a in rec["apps"]
                               if str(a.get("event_id", "")) != str(event_id)]
            rec["apps"].append(app)
            rec["apps"] = rec["apps"][-self.ROLLING_N:]
        # Update position if we now have a better signal
        if entry["pos"] != "MF" or rec["pos"] == "MF":
            rec["pos"] = entry["pos"]

    # ── Lookup ──────────────────────────────────────────────────────────────

    def _find_player(self, name: str, player_id: int | str | None = None) -> dict | None:
        """Return the cache entry for name, using fuzzy token matching."""
        if player_id is not None:
            try:
                exact = self._data.get(f"id:{int(player_id)}")
                if isinstance(exact, dict):
                    return exact
            except (TypeError, ValueError):
                pass
        key = _norm(name)
        if isinstance(self._data.get(key), dict):
            return self._data[key]
        # Try shared-token fallback
        toks = _name_tokens(name)
        best, best_score = None, 0
        for k, rec in self._player_records():
            shared = len(toks & _name_tokens(rec["name"]))
            if shared > best_score and shared >= 1:
                best, best_score = rec, shared
        return best if best_score >= 1 else None

    def player_xg_per90(self, name: str, player_id: int | str | None = None) -> float | None:
        """Recency-weighted, position-shrunk xG per 90; None if unknown.

        The prior is deliberately weak enough to preserve real signal after a
        handful of full appearances, while preventing short substitute spells
        from dominating an availability adjustment.
        """
        rec = self._find_player(name, player_id)
        if rec is None or not rec["apps"]:
            return None
        apps = rec["apps"]
        total_mins = sum(float(m.get("mins", 0.0)) for m in apps)
        if total_mins < 10:
            return None
        dates = [pd.Timestamp(m["date"]) for m in apps if m.get("date")]
        anchor = max(dates) if dates else pd.Timestamp.now(tz="UTC").tz_localize(None)
        decay = math.log(2.0) / PLAYER_XG_HALF_LIFE_DAYS
        weighted_xg = weighted_mins = 0.0
        for app in apps:
            mins = float(app.get("mins", 0.0))
            try:
                age = max(0.0, float((anchor - pd.Timestamp(app["date"])).days))
            except Exception:
                age = 0.0
            wt = math.exp(-decay * age)
            weighted_xg += float(app.get("xg", 0.0) or 0.0) * wt
            weighted_mins += mins * wt
        prior_rate = _POS_XG_DEFAULT.get(rec.get("pos", "MF"), _POS_XG_DEFAULT["MF"])
        prior90 = PLAYER_XG_PRIOR_MINUTES / 90.0
        return (weighted_xg + prior_rate * prior90) / (weighted_mins / 90.0 + prior90) * 1.0

    def player_position(self, name: str, player_id: int | str | None = None) -> str:
        """Returns GK/DF/MF/FW; defaults to MF."""
        rec = self._find_player(name, player_id)
        return rec["pos"] if rec else "MF"

    # ── Team baseline ───────────────────────────────────────────────────────

    def _team_avg_xg(self, team: str) -> float:
        """Team's expected-goals-for per match from model params (or fallback)."""
        # Try exact, then fuzzy
        if team in self._team_xg:
            return self._team_xg[team]
        toks = _name_tokens(team)
        for t, v in self._team_xg.items():
            if len(_name_tokens(t) & toks) >= 1:
                return v
        return 1.3  # league average fallback

    def _team_avg_xga(self, team: str) -> float:
        """Team's expected-goals-against per match from model params (or fallback)."""
        if team in self._team_xga:
            return self._team_xga[team]
        toks = _name_tokens(team)
        for t, v in self._team_xga.items():
            if len(_name_tokens(t) & toks) >= 1:
                return v
        return 1.3

    # ── Core adjustment calculation ─────────────────────────────────────────

    def _player_contribution(self, name: str, pos: str | None = None) -> dict:
        """Return estimated per-match xG and xA contribution for a player.

        Uses the player's rolling stats when available; falls back to
        positional defaults when not.
        """
        if pos is None:
            pos = self.player_position(name)
        xg90 = self.player_xg_per90(name)
        if xg90 is not None:
            att_contrib = xg90 / 90.0 * 85.0  # scale: ~85min average game time
            def_contrib = POS_DEF_SHARE.get(pos, 0.4)  # proxy: defensive share
        else:
            # Positional fallback
            att_contrib = _POS_XG_DEFAULT.get(pos, 0.07)
            def_contrib = POS_DEF_SHARE.get(pos, 0.4)
        att_weight = POS_ATT_SHARE.get(pos, 0.45)
        return {
            "pos": pos,
            "att_contrib": att_contrib * att_weight,
            "def_contrib": def_contrib,
            "from_data": xg90 is not None,
        }

    def _compute_team_adj(
        self,
        team: str,
        missing: list[dict],   # each: {"name": str, "reason": str, "pos": str, ...}
    ) -> dict[str, float]:
        """Compute attack_mult and defense_mult for one team given their absentees.

        attack_mult  < 1.0 → team scores fewer goals
        defense_mult > 1.0 → opponent scores more goals
        """
        if not missing:
            return {"attack_mult": 1.0, "defense_mult": 1.0,
                    "n_missing": 0, "detail": []}

        baseline_xg  = self._team_avg_xg(team)
        baseline_xga = self._team_avg_xga(team)

        # Per-player contribution estimates — 11 starters, rough positional split
        # We'll measure the missing player's value relative to an equal-split
        # starting XI (baseline_xg / 11 per attacker, etc.).
        avg_starter_xg  = baseline_xg  / 11.0
        avg_starter_xga = baseline_xga / 11.0   # average defensive "coverage" per player

        att_loss_total = 0.0
        def_loss_total = 0.0
        detail = []

        for miss in missing:
            name = str(miss.get("name") or miss.get("player") or "")
            pos_raw = str(miss.get("position") or miss.get("pos") or "")
            pos = _normalise_pos(pos_raw) if pos_raw else self.player_position(name)
            contrib = self._player_contribution(name, pos)

            # Attack loss: how much less does the team score without this player?
            att_loss = min(contrib["att_contrib"], avg_starter_xg * 2.0)

            # Defense loss: how much more does the OPPONENT score without this player?
            # We scale by the player's positional defensive responsibility.
            def_loss = contrib["def_contrib"] * avg_starter_xga

            att_loss_total += att_loss
            def_loss_total += def_loss
            detail.append({
                "name": name, "pos": pos,
                "att_loss": round(att_loss, 4),
                "def_loss": round(def_loss, 4),
                "from_data": contrib["from_data"],
            })

        # Convert to fractional adjustments (negative = worse)
        att_frac = min(ADJ_CAP, att_loss_total / max(baseline_xg, 0.5))
        def_frac = min(ADJ_CAP, def_loss_total / max(baseline_xga, 0.5))

        return {
            "attack_mult":  round(1.0 - att_frac, 4),    # < 1.0
            "defense_mult": round(1.0 + def_frac, 4),    # > 1.0
            "n_missing": len(missing),
            "att_frac": round(att_frac, 4),
            "def_frac": round(def_frac, 4),
            "detail": detail,
        }

    # ── Public API ──────────────────────────────────────────────────────────

    def adjustments_for_match(self, event: dict) -> dict[str, dict]:
        """Compute player adjustments from a BSD event dict.

        The event should be a live/upcoming BSD event that includes the
        ``unavailable_players`` field (returned by GET /api/events/).

        Returns
        -------
        {
          "home": {"attack_mult": float, "defense_mult": float, "n_missing": int, ...},
          "away": {"attack_mult": float, "defense_mult": float, "n_missing": int, ...},
        }
        """
        if not self._loaded:
            self.load()

        home_team = str(event.get("home_team") or "")
        away_team = str(event.get("away_team") or "")
        unavail   = bsd_unavailable(event)

        home_adj = self._compute_team_adj(home_team, unavail.get("home", []))
        away_adj = self._compute_team_adj(away_team, unavail.get("away", []))
        return {"home": home_adj, "away": away_adj}

    def adjustments_from_names(
        self,
        home_team: str,
        unavailable_home: list[str],
        away_team: str,
        unavailable_away: list[str],
    ) -> dict[str, dict]:
        """Compute adjustments from plain player name lists.

        Useful when you have manual lists (e.g. from absences.csv / press reports)
        rather than structured BSD event data.
        """
        if not self._loaded:
            self.load()
        home_missing = [{"name": n, "pos": ""} for n in unavailable_home]
        away_missing = [{"name": n, "pos": ""} for n in unavailable_away]
        return {
            "home": self._compute_team_adj(home_team, home_missing),
            "away": self._compute_team_adj(away_team, away_missing),
        }

    def adjustments_from_lineups(self, event: dict) -> dict[str, dict] | None:
        """Lineup-based quality signal — only available ~1h before kickoff.

        Compares the expected xG of the confirmed starting XI against the team's
        season baseline. Returns None when lineups aren't confirmed yet.
        """
        if not self._loaded:
            self.load()
        lu = bsd_lineups(event)
        if not lu:
            return None
        result = {}
        home_team = str(event.get("home_team") or "")
        away_team = str(event.get("away_team") or "")
        for side, team in (("home", home_team), ("away", away_team)):
            grp = lu.get(side, {})
            starters = grp.get("players") or grp.get("starters") or grp.get("starting_xi") or []
            if not starters:
                result[side] = None
                continue
            # Sum xG of confirmed starters
            xi_xg = sum(
                self.player_xg_per90(
                    p.get("name") or p.get("player") or "",
                    p.get("player_id") or p.get("id") or p.get("api_id")
                ) or _POS_XG_DEFAULT.get(
                    _normalise_pos(str(p.get("specific_position") or p.get("position") or "")), 0.07
                )
                for p in starters
            )
            baseline = self._team_avg_xg(team)
            # Ratio vs baseline (>1 means stronger-than-usual XI)
            xi_ratio = xi_xg / max(baseline, 0.5)
            result[side] = {
                "xi_xg": round(xi_xg, 3),
                "baseline_xg": round(baseline, 3),
                "xi_ratio": round(xi_ratio, 4),
                "lineup_confirmed": True,
                "n_starters": len(starters),
            }
        return result

    def summary(self) -> dict:
        records = self._player_records()
        n_players = len(records)
        n_with_stats = sum(1 for _, r in records if r["apps"])
        n_entries = sum(len(r["apps"]) for _, r in records)
        return {
            "players": n_players,
            "players_with_stats": n_with_stats,
            "total_match_entries": n_entries,
            "teams_covered": len(self._team_xg),
        }


# ── Module-level convenience singleton ───────────────────────────────────────

_store: PlayerFeatureStore | None = None


def get_store(load: bool = True) -> PlayerFeatureStore:
    """Return (and lazily initialise) the module-level singleton store."""
    global _store
    if _store is None:
        _store = PlayerFeatureStore()
        if load:
            _store.load()
    return _store


def adjustments_for_match(event: dict) -> dict[str, dict]:
    """Module-level shortcut."""
    return get_store().adjustments_for_match(event)


def adjustments_from_names(
    home_team: str, unavailable_home: list[str],
    away_team: str, unavailable_away: list[str],
) -> dict[str, dict]:
    """Module-level shortcut."""
    return get_store().adjustments_from_names(
        home_team, unavailable_home, away_team, unavailable_away)


# ── Market-dispersion helper (BSD multi-bookmaker odds) ───────────────────────

def market_dispersion(event: dict) -> dict[str, float | None]:
    """Measure bookmaker disagreement from BSD multi-bookmaker odds.

    BSD embeds odds from 17+ bookmakers.  The dispersion (std-dev of the
    implied probability distribution across bookmakers) is a signal that
    the market is uncertain.  High dispersion → model edge may be real;
    low dispersion → market is confident, be more cautious.

    Returns {"home_disp": float|None, "draw_disp": float|None, "away_disp": float|None}
    """
    bk_odds = event.get("bookmakers") or event.get("odds_providers") or []
    if not isinstance(bk_odds, list) or len(bk_odds) < 3:
        # Fall back to scalar top-level odds (no dispersion measurable)
        return {"home_disp": None, "draw_disp": None, "away_disp": None}

    home_probs, draw_probs, away_probs = [], [], []
    for bk in bk_odds:
        try:
            oh = float(bk.get("odds_home") or bk.get("home") or 0)
            od = float(bk.get("odds_draw") or bk.get("draw") or 0)
            oa = float(bk.get("odds_away") or bk.get("away") or 0)
            if oh < 1.01 or od < 1.01 or oa < 1.01:
                continue
            total = 1 / oh + 1 / od + 1 / oa
            home_probs.append(1 / oh / total)
            draw_probs.append(1 / od / total)
            away_probs.append(1 / oa / total)
        except (TypeError, ValueError, ZeroDivisionError):
            continue

    def _disp(lst: list) -> float | None:
        return float(np.std(lst)) if len(lst) >= 3 else None

    return {
        "home_disp": _disp(home_probs),
        "draw_disp": _disp(draw_probs),
        "away_disp": _disp(away_probs),
        "n_bookmakers": len(home_probs),
    }


# ── Dated absences (P2.4) ───────────────────────────────────────────────────────

def pull_absences(api_key: str, days_ahead: int = 14,
                  events: list[dict] | None = None) -> int:
    """Pull unavailable_players for upcoming BSD events in our competitions
    and append to data/absences_club.csv.

    Append-only; deduped on (match_date, team, player), keeping the row with
    the latest recorded_at. Point-in-time rule for any consumer: a backtest
    as-of date A may only use rows with recorded_at <= A.
    """
    from datetime import datetime, timedelta, timezone
    from .competitions import comp_from_bsd_league

    today = datetime.now(timezone.utc).date()
    if events is None:
        try:
            events = get_all_events(
                api_key, status="notstarted", date_from=str(today),
                date_to=str(today + timedelta(days=days_ahead))
            )
        except Exception as exc:
            print(f"  pull_absences: BSD fetch failed ({exc}) — skipped")
            return 0
    else:
        from .schema import normalize_status
        end = str(today + timedelta(days=days_ahead))
        events = [
            ev for ev in events
            if normalize_status(ev.get("status")) == "NOT"
            and str(event_date_utc(ev))[:10] >= str(today)
            and str(event_date_utc(ev))[:10] <= end
        ]

    recorded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for ev in events:
        comp = comp_from_bsd_league(bsd_league_name(ev))
        if comp is None:
            continue
        match_date = event_date_utc(ev)[:10]
        home = str(ev.get("home_team") or "")
        away = str(ev.get("away_team") or "")
        unavail = bsd_unavailable(ev)
        for side, team in (("home", home), ("away", away)):
            for p in unavail.get(side, []):
                name = str(p.get("name") or "").strip()
                if not name:
                    continue
                rows.append({"recorded_at": recorded_at, "match_date": match_date,
                            "competition": comp.name, "team": team, "player": name,
                            "reason": str(p.get("reason") or ""),
                            "status": str(p.get("status") or "")})

    new_df = pd.DataFrame(rows, columns=ABSENCES_COLUMNS)
    if ABSENCES_CSV.exists():
        try:
            old = pd.read_csv(ABSENCES_CSV)
        except Exception:
            old = pd.DataFrame(columns=ABSENCES_COLUMNS)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    if not combined.empty:
        combined = (combined.sort_values("recorded_at")
                   .drop_duplicates(subset=["match_date", "team", "player"], keep="last")
                   .sort_values(["match_date", "team", "player"]).reset_index(drop=True))
    DATA.mkdir(exist_ok=True)
    combined.to_csv(ABSENCES_CSV, index=False)
    print(f"  pull_absences: {len(new_df)} rows observed this run "
          f"({len(combined)} total in {ABSENCES_CSV.name})")
    return len(new_df)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build/query the BSD player stats cache for the Club Soccer engine.",
        epilog="Register at https://sports.bzzoiro.com/register/ for a BSD key.",
    )
    ap.add_argument("--refresh", action="store_true",
                    help="Fetch BSD events and rebuild player stats cache "
                         "(requires --api-key or BSD_API_KEY env var)")
    ap.add_argument("--from-cache", action="store_true",
                    help="Rebuild player stats from already-downloaded bsd_cache/ "
                         "files (no API calls)")
    ap.add_argument("--refresh-cached", action="store_true",
                    help="Fetch player stats for locally cached event details "
                         "(requires --api-key or BSD_API_KEY)")
    ap.add_argument("--summary", action="store_true",
                    help="Print cache summary")
    ap.add_argument("--player", metavar="NAME",
                    help="Look up a player's contribution estimate")
    ap.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"),
                    help="Show availability adjustment for two teams "
                         "(uses --missing-home and --missing-away)")
    ap.add_argument("--missing-home", nargs="*", default=[],
                    help="Player names missing from the home side")
    ap.add_argument("--missing-away", nargs="*", default=[],
                    help="Player names missing from the away side")
    ap.add_argument("--max-events", type=int, default=500,
                    help="Max BSD events to process (default 500)")
    ap.add_argument("--oldest-first", action="store_true",
                    help="With --refresh-cached, prioritise the oldest local events")
    ap.add_argument("--days-back", type=int, default=60,
                    help="With --refresh, how many days of finished matches "
                         "to pull (default 60)")
    ap.add_argument("--pull-absences", action="store_true",
                    help="Pull unavailable_players for upcoming BSD events and "
                         "append to data/absences_club.csv (P2.4)")
    ap.add_argument("--pause", type=float, default=0.1,
                    help="Seconds between uncached API calls (default 0.1)")
    ap.add_argument("--api-key", dest="api_key",
                    help="BSD API key (overrides env/api_keys.json)")
    args = ap.parse_args()

    store = PlayerFeatureStore()

    if args.pull_absences:
        key = args.api_key or get_key("bsd", env="BSD_API_KEY")
        if not key:
            sys.exit("No BSD key — set BSD_API_KEY or use --api-key.")
        pull_absences(key)
        if not (args.refresh or args.from_cache or args.summary or args.player or args.match):
            return

    if args.refresh or args.refresh_cached:
        key = args.api_key or get_key("bsd", env="BSD_API_KEY")
        if not key:
            sys.exit("No BSD key — set BSD_API_KEY or use --api-key.")
        store.load()
        if args.refresh:
            store.refresh(key, max_events=args.max_events, pause=args.pause,
                          days_back=args.days_back)
        if args.refresh_cached:
            store.refresh_player_stats_from_cached_events(
                key, max_events=args.max_events, pause=args.pause,
                newest_first=not args.oldest_first)
    elif args.from_cache:
        store.load()
        n = store.refresh_from_cache()
        print(f"Processed {n} cached events.")
    else:
        store.load()

    if args.summary:
        s = store.summary()
        print(f"Players: {s['players']} ({s['players_with_stats']} with match data)")
        print(f"Match entries: {s['total_match_entries']}")
        print(f"Teams in model: {s['teams_covered']}")

    if args.player:
        rec = store._find_player(args.player)
        if rec:
            xg90 = store.player_xg_per90(args.player)
            print(f"{rec['name']}  pos={rec['pos']}  "
                  f"xG/90={xg90:.3f}" if xg90 else f"{rec['name']}  pos={rec['pos']}  "
                  f"(positional default, no match data)")
            c = store._player_contribution(args.player)
            print(f"  att_contrib={c['att_contrib']:.4f}  def_contrib={c['def_contrib']:.4f}")
        else:
            print(f"Player {args.player!r} not found in cache.")

    if args.match:
        home, away = args.match
        adj = store.adjustments_from_names(
            home, args.missing_home, away, args.missing_away)
        for side, team in (("home", home), ("away", away)):
            a = adj[side]
            print(f"\n{team} ({side}):")
            print(f"  attack_mult={a['attack_mult']:.4f}  "
                  f"defense_mult={a['defense_mult']:.4f}  "
                  f"n_missing={a['n_missing']}")
            for d in a.get("detail", []):
                star = " *" if d["from_data"] else ""
                print(f"    {d['name']} ({d['pos']})  "
                      f"att_loss={d['att_loss']:.4f}  def_loss={d['def_loss']:.4f}{star}")


if __name__ == "__main__":
    main()
