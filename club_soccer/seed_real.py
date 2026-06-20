#!/usr/bin/env python3
"""Seed or extend the Club Soccer fixtures from BSD (free, no rate limits).

Replaces the former API-Football integration.  BSD provides the same data
(fixtures, results, shot stats) with no daily quota and no cost.

Register at https://sports.bzzoiro.com/register/ and add your key:
  data/api_keys.json -> "bsd": "YOUR_KEY"   or   env BSD_API_KEY

Two modes:

  # FRESH full reseed (overwrites fixtures.csv with all competitions)
  python3 -m club_soccer.seed_real --seasons 2024 2025
  python3 -m club_soccer.seed_real --seasons 2024 2025 --stats

  # MERGE just cups (or UEFA) ONTO the existing league base
  python3 -m club_soccer.seed_real --cups --merge
  python3 -m club_soccer.seed_real --competitions "FA Cup" "EFL Cup" --merge

`--cups` selects every cup competition, `--uefa` every European one, or name
them explicitly with `--competitions`.  In `--merge` mode fetched rows are
appended to fixtures.csv (deduped by fixture_id, league rows and shot stats
preserved) and team names are reconciled to the football-data canon via names.py.

`--stats` fetches per-match shot/corner stats from BSD event detail
(GET /api/events/{id}/).  One request per finished match — no daily cap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from api_keys import get_key
from bsd_client import get_all_events, get_event, league_name as bsd_league_name
from .competitions import COMPETITIONS, comp_from_bsd_league
from . import model as M
from .names import make_canon

DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
BACKUP = DATA / "fixtures_prev.bak.csv"
STATS_CACHE = DATA / "bsd_cache"

# BSD stat field names -> our fixtures.csv column suffixes
STAT_FIELDS = {
    # Direct top-level BSD fields (home_shots, away_shots, etc.)
    "shots":         "shots",
    "shots_on_goal": "sot",
    "shots_on_target": "sot",
    "sot":           "sot",
    "corners":       "corners",
    "corner_kicks":  "corners",
}

# BSD statistics list format: [{"type": "Total Shots", "value": 8}, ...]
STAT_LIST_MAP = {
    "total shots":    "shots",
    "shots on goal":  "sot",
    "shots on target": "sot",
    "corner kicks":   "corners",
    "corners":        "corners",
}

_FINISHED = {"finished", "ft", "aet", "pen"}
_UPCOMING = {"upcoming", "scheduled", "ns"}


def _num(v) -> int | str:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return ""


def _bsd_to_row(event: dict, comp_name: str, comp_api_id: int,
                country: str, kind: str, canon=None) -> dict:
    """Convert a BSD event dict to our fixtures.csv schema."""
    home = str(event.get("home_team") or "")
    away = str(event.get("away_team") or "")
    if canon:
        home = canon(home)
        away = canon(away)

    kickoff = str(event.get("date") or event.get("kickoff") or "")
    date_str = kickoff[:10]

    status_raw = str(event.get("status") or "").lower()
    finished = status_raw in _FINISHED

    # Score
    score = event.get("score") or event.get("result") or {}
    if isinstance(score, dict):
        hg = score.get("home") if score.get("home") is not None else event.get("home_score")
        ag = score.get("away") if score.get("away") is not None else event.get("away_score")
    else:
        hg = event.get("home_score") or event.get("goals_home")
        ag = event.get("away_score") or event.get("goals_away")

    home_goals = hg if finished else None
    away_goals = ag if finished else None

    try:
        year = int(date_str[:4])
        month = int(date_str[5:7])
        season = year if month >= 7 else year - 1
    except (ValueError, IndexError):
        season = None

    return {
        "fixture_id": event.get("id"),
        "date": date_str,
        "season": season,
        "competition": comp_name,
        "competition_id": comp_api_id,
        "country": country,
        "type": kind,
        "home_id": event.get("home_team_id") or "",
        "home": home,
        "away_id": event.get("away_team_id") or "",
        "away": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "status": status_raw.upper()[:3] if status_raw else "",
        "neutral": 0,
        "home_shots": "",
        "away_shots": "",
        "home_sot": "",
        "away_sot": "",
        "home_corners": "",
        "away_corners": "",
    }


def fetch_fixtures(seasons: list[int] | None, key: str,
                   comps=None, canon=None) -> pd.DataFrame:
    """Pull fixtures from BSD for the given competitions/seasons."""
    target_comps = comps or COMPETITIONS
    comp_names = {c.name for c in target_comps}

    # Fetch upcoming and finished
    all_events: list[dict] = []
    for status in ("upcoming", "finished"):
        try:
            all_events.extend(get_all_events(key, status=status))
        except Exception as exc:
            print(f"  ! BSD {status} fetch failed: {exc}")

    # Deduplicate
    seen: set = set()
    unique: list[dict] = []
    for ev in all_events:
        eid = ev.get("id")
        if eid not in seen:
            seen.add(eid)
            unique.append(ev)

    rows: list[dict] = []
    unmatched: set[str] = set()

    for ev in unique:
        lname = bsd_league_name(ev)
        comp = comp_from_bsd_league(lname)
        if comp is None or comp.name not in comp_names:
            if comp is None:
                unmatched.add(lname)
            continue

        # Season filter
        row = _bsd_to_row(ev, comp.name, comp.api_id, comp.country, comp.kind, canon)
        if seasons and row["season"] not in seasons:
            continue
        rows.append(row)
        print(f"  {comp.name} {row['date']}: {row['home']} vs {row['away']}")

    if unmatched:
        shown = sorted(unmatched)[:8]
        print(f"  unrecognised BSD leagues (ignored): {shown}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["fixture_id"], keep="last")
    return df


def select_competitions(names, cups, uefa):
    """Resolve competition subset from --competitions/--cups/--uefa."""
    if not (names or cups or uefa):
        return list(COMPETITIONS)
    wanted = set(names or [])
    if cups:
        wanted |= {c.name for c in COMPETITIONS if c.kind == "cup"}
    if uefa:
        wanted |= {c.name for c in COMPETITIONS if c.kind == "europe"}
    comps = [c for c in COMPETITIONS if c.name in wanted]
    unknown = wanted - {c.name for c in COMPETITIONS}
    if unknown:
        print(f"  ! unknown competitions ignored: {sorted(unknown)}")
    return comps


def _extract_stats(event_detail: dict) -> dict[str, dict[str, int | str]]:
    """Extract per-side shot stats from a BSD event detail response.

    Returns: {"home": {"shots": N, "sot": N, "corners": N}, "away": {...}}
    """
    result: dict[str, dict] = {"home": {}, "away": {}}

    # Format 1: nested statistics dict {"home": [...], "away": [...]}
    stats = event_detail.get("statistics") or event_detail.get("stats") or {}
    if isinstance(stats, dict):
        for side in ("home", "away"):
            side_data = stats.get(side) or {}
            if isinstance(side_data, list):
                for s in side_data:
                    col = STAT_LIST_MAP.get(str(s.get("type", "")).strip().lower())
                    if col:
                        result[side][col] = _num(s.get("value"))
            elif isinstance(side_data, dict):
                for field, col in STAT_FIELDS.items():
                    if field in side_data:
                        result[side][col] = _num(side_data[field])
    # Format 2: top-level home_shots, away_corners, etc.
    for side in ("home", "away"):
        for field, col in STAT_FIELDS.items():
            full = f"{side}_{field}"
            if full in event_detail and col not in result[side]:
                result[side][col] = _num(event_detail[full])

    return result


def fetch_stats(df: pd.DataFrame, key: str, max_stats: int,
                pause: float) -> pd.DataFrame:
    """Fill shots/sot/corners for finished matches via BSD event detail."""
    STATS_CACHE.mkdir(parents=True, exist_ok=True)
    played = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    todo = played["fixture_id"].dropna().astype(str).tolist()[:max_stats]
    print(f"\nFetching shot stats for {len(todo)} finished matches "
          f"(of {len(played)} played; cap {max_stats})...")

    by_id: dict[str, dict] = {}
    for n, fid in enumerate(todo, 1):
        cache = STATS_CACHE / f"event_{fid}.json"
        if cache.exists():
            detail = json.loads(cache.read_text())
        else:
            try:
                detail = get_event(key, fid)
                cache.write_text(json.dumps(detail, indent=2))
                time.sleep(pause)
            except Exception as exc:
                print(f"  ! BSD event {fid}: {exc}")
                continue
        by_id[str(fid)] = _extract_stats(detail)
        if n % 25 == 0:
            print(f"  ...{n}/{len(todo)}")

    for col in ("home_shots", "away_shots", "home_sot", "away_sot",
                "home_corners", "away_corners"):
        side, stat = col.split("_", 1)
        stat_key = "sot" if stat == "sot" else stat.replace("_", "")
        # Fix: map column name to stat key correctly
        stat_key = {"shots": "shots", "sot": "sot", "corners": "corners"}.get(
            stat, stat)
        df[col] = df.apply(
            lambda r: by_id.get(str(r["fixture_id"]), {}).get(side, {}).get(stat_key, "")
            if pd.notna(r.get("fixture_id")) else "",
            axis=1,
        )
    return df


def refit_and_baseline() -> None:
    print("\nRefitting model...")
    params = M.fit()
    M.save_params(params)
    print(f"  fitted {params['fitted_matches']} matches, {len(params['teams'])} teams")
    print("\nWalk-forward validation (fresh baseline):")
    import subprocess
    subprocess.run([sys.executable, str(HERE / "validate.py"), "--update-baseline"],
                   cwd=str(HERE), check=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025],
                    help="season start years to pull (default: 2024 2025)")
    ap.add_argument("--competitions", nargs="+",
                    help="subset by exact competition name")
    ap.add_argument("--cups", action="store_true",
                    help="select all cup competitions")
    ap.add_argument("--uefa", action="store_true",
                    help="select all European competitions")
    ap.add_argument("--merge", action="store_true",
                    help="append onto existing fixtures.csv (keeps league shots) "
                         "and reconcile team names to the league canon")
    ap.add_argument("--stats", action="store_true",
                    help="fetch shot stats for finished matches via BSD event detail")
    ap.add_argument("--max-stats", type=int, default=400,
                    help="cap on per-match stats requests (default 400)")
    ap.add_argument("--pause", type=float, default=0.1,
                    help="seconds between uncached stats requests (default 0.1, "
                         "BSD has no rate limit)")
    ap.add_argument("--api-key", dest="api_key",
                    help="BSD API key (overrides env BSD_API_KEY / api_keys.json)")
    ap.add_argument("--keep-backup", action="store_true",
                    help="don't overwrite an existing backup file")
    args = ap.parse_args()

    key = args.api_key or get_key("bsd", env="BSD_API_KEY")
    if not key:
        sys.exit(
            "No BSD key. Register at https://sports.bzzoiro.com/register/ "
            "and add 'bsd' to data/api_keys.json, or set BSD_API_KEY."
        )

    comps = select_competitions(args.competitions, args.cups, args.uefa)
    if not comps:
        sys.exit("No competitions selected.")

    merge = args.merge and FIXTURES.exists()

    # Name reconciliation — only in merge mode (map new rows onto league canon)
    canon, league_teams = None, set()
    if merge:
        league_teams = set(
            pd.read_csv(FIXTURES, usecols=["home", "away"]).stack().unique()
        )
        canon = make_canon(league_teams)

    # Back up current fixtures before any change
    if FIXTURES.exists() and not (args.keep_backup and BACKUP.exists()):
        BACKUP.write_text(FIXTURES.read_text())
        print(f"Backed up current fixtures -> {BACKUP.name}")

    print(f"\nFetching {len(comps)} competition(s) x seasons {args.seasons} from BSD...")
    df = fetch_fixtures(args.seasons, key, comps, canon)

    if df.empty:
        sys.exit(
            "No fixtures returned — check your BSD key or the competition/season values. "
            f"Backup left intact; restore with: cp {BACKUP} {FIXTURES}"
        )

    if args.stats:
        df = fetch_stats(df, key, args.max_stats, args.pause)

    if merge:
        base = pd.read_csv(FIXTURES)
        new_teams = sorted(
            (set(df["home"].dropna()) | set(df["away"].dropna())) - league_teams
        )
        linked = len(
            (set(df["home"].dropna()) | set(df["away"].dropna())) & league_teams
        )
        merged = (
            pd.concat([base, df], ignore_index=True)
            .drop_duplicates(subset=["fixture_id"], keep="first")
        )
        merged.to_csv(FIXTURES, index=False)
        print(
            f"\nMerged {len(df)} fetched rows -> {FIXTURES} "
            f"({len(base)} -> {len(merged)} rows)"
        )
        print(
            f"  {linked} teams linked to league data, "
            f"{len(new_teams)} new (unlinked)"
        )
        if new_teams:
            print("  new/unlinked teams (add to names.OVERRIDES if dupes):")
            print("   ", ", ".join(new_teams[:25]) + (" ..." if len(new_teams) > 25 else ""))
    else:
        df.to_csv(FIXTURES, index=False)
        played = df[df["home_goals"].notna() & df["away_goals"].notna()]
        print(f"\nWrote {len(df)} fixtures ({len(played)} played) -> {FIXTURES}")

    refit_and_baseline()
    print("\nDone. From now on use: python3 -m club_soccer.fetch --current")


if __name__ == "__main__":
    main()
