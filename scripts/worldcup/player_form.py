#!/usr/bin/env python3
"""Phase 1 — WC player-form store (proof of concept).

Builds a per-player *form* value for the World Cup field from BSD's player-stats
endpoint, keyed by the global ``player_id`` confirmed in Phase 0.

Pipeline
--------
1.  GET /api/v2/worldcup/squads/        -> the full WC field (player_id, team, pos)
2.  GET /api/v2/players/{id}/stats/     -> that player's per-match history
                                           (xG, xA, rating, minutes, shots)
3.  Collapse the history into a recency- and minutes-weighted form value, shrunk
    toward a position baseline so thin samples fall back to a sensible prior.
4.  Cache to data/worldcup/player_form_cache.json.

This is the *store* only — it does NOT touch predictions. Per-team multipliers
(Phase 2) and any wiring into the edge path (later, behind a validation gate) are
separate steps.

Usage
-----
    python3 -m scripts.worldcup.player_form --teams 489,490 --report
    python3 -m scripts.worldcup.player_form --max-players 40 --report
    python3 -m scripts.worldcup.player_form --all            # whole field (slow)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CACHE = HERE / "data" / "worldcup" / "player_form_cache.json"

# ── form-model constants (transparent + tunable) ──────────────────────────────
RECENCY_DECAY = 0.92        # weight of match i back = DECAY**i (0 = most recent)
MAX_MATCHES = 40            # most-recent N matches used per player
SHRINK_K = 3.0             # prior strength, in full-90 equivalents
RATING_BASELINE = 6.7       # BSD per-match rating ~ league average

# position -> (xg/90, xa/90) baseline a no-history player regresses to
POS_BASELINE = {
    "GK": (0.00, 0.01),
    "DF": (0.06, 0.07),
    "MF": (0.18, 0.20),
    "FW": (0.45, 0.18),
}


def _norm_pos(raw: str) -> str:
    r = (raw or "").upper().strip()
    if r.startswith("G"):
        return "GK"
    if r.startswith("D"):
        return "DF"
    if r.startswith("F") or r in ("ST", "CF", "LW", "RW"):
        return "FW"
    return "MF"


# ── BSD pulls ─────────────────────────────────────────────────────────────────

def _client():
    from api_keys import get_key
    from bsd_client import _get
    key = get_key("bsd", env="BSD_API_KEY")
    if not key:
        sys.exit("no BSD key (set BSD_API_KEY or add to keys file).")
    return _get, key


def fetch_wc_field(get, key) -> list[dict]:
    """All players in the WC squads endpoint (paginated)."""
    field, offset = [], 0
    while True:
        page = get(f"/api/v2/worldcup/squads/?limit=50&offset={offset}", key)
        res = page.get("results") or []
        field.extend(res)
        if not page.get("next") or not res:
            break
        offset += 50
    return field


def fetch_player_stats(get, key, pid) -> list[dict]:
    try:
        return (get(f"/api/v2/players/{pid}/stats/", key).get("results") or [])
    except Exception:  # noqa: BLE001
        return []


def team_name(get, key, tid, _cache={}) -> str:
    if tid in _cache:
        return _cache[tid]
    try:
        t = get(f"/api/v2/teams/{tid}/", key)
        nm = str(t.get("name") or t.get("short_name") or tid)
    except Exception:  # noqa: BLE001
        nm = str(tid)
    _cache[tid] = nm
    return nm


# ── form computation ──────────────────────────────────────────────────────────

def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def compute_form(rows: list[dict], pos: str) -> dict:
    """Collapse a player's per-match rows into a weighted form value.

    Rates (xg90/xa90) are minutes-aware weighted means, then shrunk toward the
    position baseline by SHRINK_K full-90 equivalents. `n_eff` is how many full
    matches of evidence we effectively have (recency- and minutes-discounted)."""
    rows = rows[:MAX_MATCHES]
    base_xg, base_xa = POS_BASELINE.get(pos, POS_BASELINE["MF"])

    num_xg = num_xa = den90 = 0.0      # for rate stats
    w_rating = w_sum = 0.0             # for the rating mean
    raw_minutes = 0.0
    used = 0
    for i, r in enumerate(rows):
        mins = _f(r.get("minutes_played"))
        if mins <= 0:
            continue
        w = RECENCY_DECAY ** i               # recency weight
        num_xg += w * _f(r.get("expected_goals"))
        num_xa += w * _f(r.get("expected_assists"))
        den90 += w * (mins / 90.0)
        rating = _f(r.get("rating"), RATING_BASELINE)
        rel = w * min(mins / 90.0, 1.0)      # rating reliability weight
        w_rating += rel * rating
        w_sum += rel
        raw_minutes += mins
        used += 1

    n_eff = den90
    xg90 = (num_xg + SHRINK_K * base_xg) / (den90 + SHRINK_K)
    xa90 = (num_xa + SHRINK_K * base_xa) / (den90 + SHRINK_K)
    rating = ((w_rating + SHRINK_K * RATING_BASELINE) / (w_sum + SHRINK_K)
              if w_sum else RATING_BASELINE)

    # a single comparable form index: rating z plus attacking over/under-baseline,
    # scaled small. Positive = in form. (Directional for the PoC, not calibrated.)
    form_index = round((rating - RATING_BASELINE)
                       + 0.5 * ((xg90 + xa90) - (base_xg + base_xa)), 3)

    return {
        "matches_used": used,
        "minutes": round(raw_minutes),
        "n_eff": round(n_eff, 2),
        "xg90": round(xg90, 3),
        "xa90": round(xa90, 3),
        "rating": round(rating, 2),
        "form_index": form_index,
        "coverage": "ok" if n_eff >= 2 else "thin",
    }


# ── build ─────────────────────────────────────────────────────────────────────

def build(teams: set[int] | None, max_players: int | None,
          verbose: bool = True) -> dict:
    get, key = _client()
    field = fetch_wc_field(get, key)
    if teams:
        field = [p for p in field if p.get("team_id") in teams]
    if max_players:
        field = field[:max_players]

    store: dict[str, dict] = {}
    n = len(field)
    for k, p in enumerate(field, 1):
        pid = p.get("player_id")
        if pid is None:
            continue
        pos = _norm_pos(p.get("position"))
        rows = fetch_player_stats(get, key, pid)
        form = compute_form(rows, pos)
        store[str(pid)] = {
            "player_id": pid,
            "name": p.get("name", ""),
            "team_id": p.get("team_id"),
            "position": pos,
            "club": p.get("club", ""),
            **form,
        }
        if verbose and (k % 10 == 0 or k == n):
            print(f"  ...{k}/{n} players", file=sys.stderr)
        time.sleep(0.02)   # be polite to the API

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(store, ensure_ascii=False, indent=1))
    return store


# ── report ────────────────────────────────────────────────────────────────────

def report(store: dict) -> None:
    get, key = _client()
    by_team: dict = defaultdict(list)
    for v in store.values():
        by_team[v["team_id"]].append(v)

    print(f"\nPlayer-form store — {len(store)} players, "
          f"{len(by_team)} teams  (cache: {CACHE.name})")
    thin = sum(1 for v in store.values() if v["coverage"] == "thin")
    print(f"coverage: {len(store)-thin} ok / {thin} thin "
          f"(<2 effective matches)\n")

    for tid, players in by_team.items():
        players.sort(key=lambda v: v["form_index"], reverse=True)
        tn = team_name(get, key, tid)
        cov = sum(1 for v in players if v["coverage"] == "ok")
        print(f"── {tn}  (team {tid})  —  {cov}/{len(players)} with real form")
        print(f"   {'player':22} {'pos':3} {'mins':>4} {'n_eff':>5} "
              f"{'xg90':>5} {'xa90':>5} {'rtg':>4} {'form':>6}")
        for v in players[:8]:
            print(f"   {v['name'][:22]:22} {v['position']:3} {v['minutes']:>4} "
                  f"{v['n_eff']:>5} {v['xg90']:>5} {v['xa90']:>5} "
                  f"{v['rating']:>4} {v['form_index']:>+6.2f}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="WC player-form store (PoC)")
    ap.add_argument("--teams", help="comma-separated BSD team_ids to include")
    ap.add_argument("--max-players", type=int, help="cap number of players")
    ap.add_argument("--all", action="store_true", help="whole field (slow)")
    ap.add_argument("--report", action="store_true", help="print team tables")
    args = ap.parse_args()

    teams = None
    if args.teams:
        teams = {int(x) for x in args.teams.split(",") if x.strip()}
    # Default cap only guards a broad scan. An explicit --teams or --all means
    # "take the whole (filtered) field"; --max-players always wins if given.
    if args.max_players:
        max_players = args.max_players
    elif args.all or teams:
        max_players = None
    else:
        max_players = 40

    store = build(teams, max_players)
    if args.report:
        report(store)
    else:
        print(f"built {len(store)} players -> {CACHE}")


if __name__ == "__main__":
    main()
