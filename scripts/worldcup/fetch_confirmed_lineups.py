#!/usr/bin/env python3
"""Fetch CONFIRMED lineups for the 2026 World Cup matches played so far.

BSD's list/date/status filters only serve an upcoming window, so finished matches
are reached by event id directly. WC 2026 is league 27; its finished matches sit in
a contiguous id range. This walks that range, keeps league-27 events that are
finished with confirmed lineups, and (re)writes data/worldcup/lineups.csv in the
existing schema — starters (lineups[side]["players"]) and bench
(lineups[side]["substitutes"]) for both teams.

Usage:
    python3 -m scripts.worldcup.fetch_confirmed_lineups            # auto-scan
    python3 -m scripts.worldcup.fetch_confirmed_lineups --from 8287 --to 8352
    python3 -m scripts.worldcup.fetch_confirmed_lineups --dry-run
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

OUT = HERE / "data" / "worldcup" / "lineups.csv"
WC_LEAGUE_ID = 27
SCAN_FROM, SCAN_TO = 8280, 8360        # generous bounds around the WC id block

COLUMNS = ["event_id", "provider_fixture_id", "match_date", "team", "player",
           "provider_team_id", "provider_player_id", "starter", "role",
           "position", "shirt_number", "formation", "lineup_status",
           "published_at", "source", "fetched_at"]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def _team_obj(ev: dict, side: str) -> tuple[str, str]:
    """(name, team_id) for 'home'/'away'."""
    obj = ev.get(f"{side}_team_obj") or {}
    name = str(ev.get(f"{side}_team") or obj.get("name") or "")
    tid = str(obj.get("id") or "")
    return name, tid


def _rows_for_event(ev: dict, eid: int, now: str) -> list[dict]:
    lu = ev.get("lineups") or {}
    if not lu.get("confirmed"):
        return []
    date = str(ev.get("event_date") or "")[:10]
    hn, _ = _team_obj(ev, "home")
    an, _ = _team_obj(ev, "away")
    composite = f"{date}|{_slug(hn)}|{_slug(an)}|fifa-world-cup"
    rows: list[dict] = []
    for side in ("home", "away"):
        grp = lu.get(side) or {}
        team_name, team_id = _team_obj(ev, side)
        formation = grp.get("formation") or ""
        for kind, starter in (("players", True), ("substitutes", False)):
            for p in (grp.get(kind) or []):
                pid = p.get("player_id")
                if pid is None:
                    continue
                rows.append({
                    "event_id": composite,
                    "provider_fixture_id": eid,
                    "match_date": date,
                    "team": team_name,
                    "player": p.get("name", ""),
                    "provider_team_id": team_id,
                    "provider_player_id": pid,
                    "starter": starter,
                    "role": "starter" if starter else "sub",
                    "position": p.get("specific_position") or p.get("position") or "",
                    "shirt_number": p.get("jersey_number") or "",
                    "formation": formation,
                    "lineup_status": "confirmed",
                    "published_at": now,
                    "source": "bsd",
                    "fetched_at": now,
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch confirmed WC2026 lineups")
    ap.add_argument("--from", dest="lo", type=int, default=SCAN_FROM)
    ap.add_argument("--to", dest="hi", type=int, default=SCAN_TO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from api_keys import get_key
    from bsd_client import _get
    key = get_key("bsd", env="BSD_API_KEY")
    if not key:
        sys.exit("no BSD key (set BSD_API_KEY).")

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    all_rows: list[dict] = []
    matches = 0
    skipped_unconfirmed = 0
    for eid in range(args.lo, args.hi + 1):
        try:
            ev = _get(f"/api/events/{eid}/", key)
        except Exception:  # noqa: BLE001
            continue
        lg = ev.get("league") or {}
        if lg.get("id") != WC_LEAGUE_ID:
            continue
        if str(ev.get("status")) != "finished":
            continue
        rows = _rows_for_event(ev, eid, now)
        if not rows:
            skipped_unconfirmed += 1
            continue
        all_rows.extend(rows)
        matches += 1
        hn, _ = _team_obj(ev, "home")
        an, _ = _team_obj(ev, "away")
        print(f"  {eid}  {ev.get('event_date','')[:10]}  "
              f"{hn[:14]:14} v {an[:14]:14}  ({len(rows)} players)",
              file=sys.stderr)

    print(f"\nfinished WC matches with confirmed lineups: {matches}", file=sys.stderr)
    if skipped_unconfirmed:
        print(f"finished but unconfirmed lineups (skipped): {skipped_unconfirmed}",
              file=sys.stderr)

    if args.dry_run:
        print(f"[dry-run] would write {len(all_rows)} rows to {OUT}")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows across {matches} matches -> {OUT}")


if __name__ == "__main__":
    main()
