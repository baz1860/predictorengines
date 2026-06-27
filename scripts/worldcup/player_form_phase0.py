#!/usr/bin/env python3
"""Phase 0 linkage check for the player-form PoC.

Confirms the premise the rest of the PoC depends on: that the
``provider_player_id`` we already store in data/worldcup/lineups.csv is a clean,
stable key we can join BSD per-player *stats* against.

Two halves:
  OFFLINE  — inspect the IDs already in lineups.csv (always runs).
  LIVE     — resolve the BSD key, refetch one event detail, and confirm the
             same player IDs appear in BSD's response (runs only if key +
             network are available; degrades gracefully otherwise).

Usage:
    python3 -m scripts.worldcup.player_form_phase0
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
LINEUPS_CSV = HERE / "data" / "worldcup" / "lineups.csv"

# BSD player dicts may carry the id under any of these keys.
_ID_FIELDS = ("provider_player_id", "player_id", "id", "sr_id", "uid")


# ── OFFLINE half ──────────────────────────────────────────────────────────────

def load_lineup_ids() -> list[dict]:
    if not LINEUPS_CSV.exists():
        sys.exit(f"missing {LINEUPS_CSV}")
    with LINEUPS_CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


def offline_report(rows: list[dict]) -> dict:
    n = len(rows)
    have_id = [r for r in rows if (r.get("provider_player_id") or "").strip()]
    numeric = [r for r in have_id if r["provider_player_id"].strip().isdigit()]
    by_fixture = defaultdict(set)
    by_player = {}
    for r in rows:
        pid = (r.get("provider_player_id") or "").strip()
        fx = (r.get("provider_fixture_id") or "").strip()
        if pid:
            by_fixture[fx].add(pid)
            by_player[pid] = r.get("player", "")

    print("── OFFLINE: lineups.csv ───────────────────────────────────────────")
    print(f"  rows                       {n}")
    print(f"  rows with player_id        {len(have_id)}  ({len(have_id)/n*100:.0f}%)")
    print(f"  player_id is integer       {len(numeric)}  ({len(numeric)/n*100:.0f}%)")
    print(f"  distinct players           {len(by_player)}")
    print(f"  distinct fixtures          {len([k for k in by_fixture if k])}")
    dup = n - len(set((r.get('provider_fixture_id',''), r.get('provider_player_id',''))
                      for r in rows))
    print(f"  (fixture,player) dup rows  {dup}")
    print(f"  sample id->name            " +
          ", ".join(f"{p}={by_player[p]}" for p in list(by_player)[:4]))
    verdict = "STRONG" if len(numeric) == n else (
              "OK" if len(have_id) == n else "WEAK")
    print(f"  offline linkage premise    {verdict}")
    print()
    return {"by_fixture": by_fixture, "by_player": by_player, "verdict": verdict}


# ── LIVE half ─────────────────────────────────────────────────────────────────

def _extract_ids(event_detail: dict) -> set[str]:
    """Pull every player id we can find from a BSD event-detail response,
    across the same shapes _players_from_event handles."""
    ids: set[str] = set()

    def _take(p):
        if isinstance(p, dict):
            for k in _ID_FIELDS:
                v = p.get(k)
                if v is not None and str(v).strip():
                    ids.add(str(v).strip())
                    break
            # id sometimes nested under "player"
            sub = p.get("player")
            if isinstance(sub, dict):
                _take(sub)

    def _drain(lst):
        if isinstance(lst, list):
            for p in lst:
                _take(p)

    lineups = event_detail.get("lineups") or {}
    for side in ("home", "away"):
        grp = lineups.get(side) or {}
        # BSD shape: starting XI under "players", bench under "substitutes".
        _drain(grp.get("players") or grp.get("starters") or grp.get("starting_xi") or [])
        _drain(grp.get("substitutes") or grp.get("bench") or [])
    # legacy/alt shapes (kept as fallbacks)
    for side in ("home", "away"):
        _drain((event_detail.get("players") or {}).get(side, []))
    _drain(event_detail.get("home_players", []))
    _drain(event_detail.get("away_players", []))
    return ids


def live_check(offline: dict) -> None:
    print("── LIVE: BSD cross-check ──────────────────────────────────────────")
    try:
        from api_keys import get_key
        from bsd_client import get_event
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIPPED — import failed: {exc}")
        return

    key = get_key("bsd", env="BSD_API_KEY")
    if not key:
        print("  SKIPPED — no BSD key (set BSD_API_KEY or add to keys file).")
        print("            offline half stands; rerun with a key to finish Phase 0.")
        return

    # pick the fixture with the most known players
    fixtures = {k: v for k, v in offline["by_fixture"].items() if k}
    if not fixtures:
        print("  SKIPPED — no provider_fixture_id in lineups.csv.")
        return
    fx = max(fixtures, key=lambda k: len(fixtures[k]))
    local_ids = fixtures[fx]
    print(f"  probing fixture {fx}  ({len(local_ids)} local player ids)")

    try:
        detail = get_event(key, fx)
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIPPED — BSD fetch failed (network/key): {exc}")
        print("            offline half stands; run where BSD is reachable.")
        return

    remote_ids = _extract_ids(detail)
    overlap = local_ids & remote_ids
    print(f"  remote player ids          {len(remote_ids)}")
    print(f"  overlap with lineups.csv   {len(overlap)} / {len(local_ids)}")
    if local_ids:
        pct = len(overlap) / len(local_ids) * 100
        print(f"  id-join coverage           {pct:.0f}%")
        if pct >= 90:
            print("  LIVE linkage premise       CONFIRMED — IDs join cleanly.")
        elif pct >= 50:
            print("  LIVE linkage premise       PARTIAL — fall back to name match for gaps.")
        else:
            print("  LIVE linkage premise       FAILED — IDs differ; use name matching.")


# ── CROSS-COMPETITION half ────────────────────────────────────────────────────

def club_check(offline: dict, max_events: int = 16) -> None:
    """Confirm a WC player_id is a GLOBAL key that also resolves CLUB matches.

    Phase 1 pulls each WC squad player's club form by player_id. That only works
    if /api/v2/players/{id}/stats/ returns matches across competitions, not just
    internationals. We take one WC player, pull their per-match stats, and
    resolve a sample of event_ids to leagues to see the club/international split.
    """
    print("── CROSS-COMP: club form by player_id ─────────────────────────────")
    try:
        from api_keys import get_key
        from bsd_client import _get, get_event, league_name
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIPPED — import failed: {exc}")
        return
    key = get_key("bsd", env="BSD_API_KEY")
    if not key:
        print("  SKIPPED — no BSD key.")
        return

    by_player = offline["by_player"]
    if not by_player:
        print("  SKIPPED — no player ids in lineups.csv.")
        return
    pid = next(iter(by_player))
    name = by_player[pid]
    try:
        prof = _get(f"/api/v2/players/{pid}/", key)
        stats = _get(f"/api/v2/players/{pid}/stats/", key).get("results") or []
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIPPED — player endpoint fetch failed: {exc}")
        return

    resolved = str(prof.get("name") or prof.get("short_name") or "")
    print(f"  probing player_id {pid}  (lineup: '{name}')")
    print(f"  /players/{pid}/ resolves to '{resolved}'  "
          f"id-match={'OK' if resolved else '??'}")
    print(f"  per-match stat rows         {len(stats)}")

    from collections import Counter
    leagues: Counter = Counter()
    seen: set = set()
    for r in stats:
        eid = r.get("event_id")
        if eid in seen:
            continue
        seen.add(eid)
        try:
            leagues[league_name(get_event(key, eid))] += 1
        except Exception:  # noqa: BLE001
            leagues["(unresolved)"] += 1
        if len(seen) >= max_events:
            break
    intl = sum(n for lg, n in leagues.items()
               if "World Cup" in lg or "International" in lg or "Nations" in lg)
    club = sum(leagues.values()) - intl
    print(f"  sampled {len(seen)} distinct events -> club={club}  international={intl}")
    for lg, n in leagues.most_common():
        print(f"     {n:>2}  {lg}")
    if club > 0:
        print("  CROSS-COMP premise          CONFIRMED — player_id pulls club form.")
    else:
        print("  CROSS-COMP premise          INCONCLUSIVE — no club matches in sample.")
    print()


def main() -> None:
    rows = load_lineup_ids()
    offline = offline_report(rows)
    live_check(offline)
    print()
    club_check(offline)


if __name__ == "__main__":
    main()
