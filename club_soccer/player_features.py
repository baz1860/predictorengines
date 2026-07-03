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
    get_all_events, get_event, event_date_utc,
    league_name as bsd_league_name,
    unavailable_players as bsd_unavailable,
    lineups as bsd_lineups,
)
from api_keys import get_key

CACHE_SCHEMA_VERSION = 2

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

# BSD stat field names in event detail (player level)
_PLAYER_XG_FIELDS  = ("xg", "expected_goals", "xGoal", "xgoal")
_PLAYER_XA_FIELDS  = ("xa", "expected_assists", "xAssist", "xassist", "key_passes")
_PLAYER_MIN_FIELDS = ("minutes", "minutes_played", "time", "minutesPlayed")
_PLAYER_POS_FIELDS = ("position", "pos", "positionId")


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


def _extract_player_entry(player_dict: dict) -> dict:
    """Pull (name, position, xg, xa, minutes) from a BSD player dict.

    BSD returns player stats in one of two shapes:
      A) flat dict with keys like "xg", "position", "minutes"
      B) nested "stats" sub-dict
    """
    base   = player_dict if isinstance(player_dict, dict) else {}
    stats  = base.get("stats") or base.get("player_stats") or base

    name = str(base.get("name") or base.get("player_name") or base.get("player") or "")
    pos_raw = str(_get_first(base, _PLAYER_POS_FIELDS, "") or
                  _get_first(stats, _PLAYER_POS_FIELDS, "")).upper().strip()

    # Normalise position to GK/DF/MF/FW
    pos = _normalise_pos(pos_raw)

    xg  = _safe_float(_get_first(stats, _PLAYER_XG_FIELDS, 0.0))
    xa  = _safe_float(_get_first(stats, _PLAYER_XA_FIELDS, 0.0))
    mins = _safe_float(_get_first(stats, _PLAYER_MIN_FIELDS, 90.0))

    return {"name": name, "pos": pos, "xg": xg, "xa": xa, "mins": mins}


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

    def _drain(lst, side: str | None):
        if isinstance(lst, list):
            for p in lst:
                entry = _extract_player_entry(p)
                if entry["name"]:
                    out.append((entry, side))

    # Shape 1: lineups
    lineups = event_detail.get("lineups") or {}
    for side in ("home", "away"):
        grp = lineups.get(side) or {}
        _drain(grp.get("starters") or grp.get("starting_xi") or [], side)
        _drain(grp.get("bench") or grp.get("substitutes") or [], side)

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

    Cache schema v2 — a JSON file:
        {
          "v": 2,
          "player_name_norm": {
            "name": "original name",
            "pos": "FW",
            "apps": [{"date": "2026-01-01", "team": "Arsenal", "mins": 90.0,
                      "xg": 0.4, "xa": 0.1, "side_confident": true}, ...]
                     (most recent 60, dated so squads/transfers can be
                     derived by "most recent app wins" — see club_squads.py)
          },
          ...
        }
    v1 caches (no "v" key, "matches"/"teams" instead of dated "apps") are
    discarded on load — rebuild with `--from-cache` (bsd_cache event files
    carry their own dates via event_date_utc, no API calls needed).
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
        """(key, record) pairs, skipping the "v" schema-version sentinel."""
        return [(k, v) for k, v in self._data.items() if isinstance(v, dict)]

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
        self._path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))

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

        processed = 0
        for ev in events[:max_events]:
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
            self._ingest_event_players(detail, home, away, date)
            processed += 1
            if processed % 50 == 0:
                print(f"  ...{processed}/{min(max_events, len(events))}")

        self.save()
        print(f"  Player cache: {len(self._player_records())} players, "
              f"{processed} events processed.")
        return processed

    def refresh_from_cache(self) -> int:
        """Build player stats from already-downloaded bsd_cache/ files (no API calls)."""
        if not self._loaded:
            self.load()
        if not STATS_CACHE.exists():
            return 0
        processed = 0
        for cache_file in sorted(STATS_CACHE.glob("event_*.json")):
            try:
                detail = json.loads(cache_file.read_text())
            except Exception:
                continue
            date = event_date_utc(detail)[:10]
            home = str(detail.get("home_team") or "")
            away = str(detail.get("away_team") or "")
            self._ingest_event_players(detail, home, away, date)
            processed += 1
        if processed:
            self.save()
        return processed

    def _ingest_event_players(self, detail: dict, home: str, away: str, date: str) -> None:
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
            self._update_player(entry, team, date, confident)

    def _update_player(self, entry: dict, team: str, date: str,
                       side_confident: bool = True) -> None:
        key = _norm(entry["name"])
        if not key:
            return
        if key not in self._data:
            self._data[key] = {
                "name": entry["name"],
                "pos": entry["pos"],
                "apps": [],
            }
        rec = self._data[key]
        if entry["mins"] > 0:
            rec["apps"].append({
                "date": date, "team": team,
                "mins": entry["mins"], "xg": entry["xg"], "xa": entry["xa"],
                "side_confident": side_confident,
            })
            rec["apps"] = rec["apps"][-self.ROLLING_N:]
        # Update position if we now have a better signal
        if entry["pos"] != "MF" or rec["pos"] == "MF":
            rec["pos"] = entry["pos"]

    # ── Lookup ──────────────────────────────────────────────────────────────

    def _find_player(self, name: str) -> dict | None:
        """Return the cache entry for name, using fuzzy token matching."""
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

    def player_xg_per90(self, name: str) -> float | None:
        """Rolling avg xG per 90 minutes for a player; None if unknown."""
        rec = self._find_player(name)
        if rec is None or not rec["apps"]:
            return None
        apps = rec["apps"]
        total_xg = sum(m["xg"] for m in apps)
        total_mins = sum(m["mins"] for m in apps)
        if total_mins < 10:
            return None
        return total_xg / total_mins * 90.0

    def player_position(self, name: str) -> str:
        """Returns GK/DF/MF/FW; defaults to MF."""
        rec = self._find_player(name)
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
            starters = grp.get("starters") or []
            if not starters:
                result[side] = None
                continue
            # Sum xG of confirmed starters
            xi_xg = sum(
                self.player_xg_per90(
                    p.get("name") or p.get("player") or ""
                ) or _POS_XG_DEFAULT.get(
                    _normalise_pos(str(p.get("position") or "")), 0.07
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

def pull_absences(api_key: str, days_ahead: int = 14) -> int:
    """Pull unavailable_players for upcoming BSD events in our competitions
    and append to data/absences_club.csv.

    Append-only; deduped on (match_date, team, player), keeping the row with
    the latest recorded_at. Point-in-time rule for any consumer: a backtest
    as-of date A may only use rows with recorded_at <= A.
    """
    from datetime import datetime, timedelta, timezone
    from .competitions import comp_from_bsd_league

    today = datetime.now(timezone.utc).date()
    try:
        events = get_all_events(api_key, status="notstarted",
                                date_from=str(today), date_to=str(today + timedelta(days=days_ahead)))
    except Exception as exc:
        print(f"  pull_absences: BSD fetch failed ({exc}) — skipped")
        return 0

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

    if args.refresh:
        key = args.api_key or get_key("bsd", env="BSD_API_KEY")
        if not key:
            sys.exit("No BSD key — set BSD_API_KEY or use --api-key.")
        store.load()
        store.refresh(key, max_events=args.max_events, pause=args.pause,
                      days_back=args.days_back)
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
