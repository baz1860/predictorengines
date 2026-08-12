#!/usr/bin/env python3
"""Fetch international fixtures into the canonical store (plan §8 steps 3-4).

Order of operations is deliberate and non-negotiable:

  1. fetch          network call
  2. RAW WRITE      store the payload verbatim, before anything interprets it
  3. parse          pure functions, no network
  4. scope filter   FIFA members only, fail closed on unknown active teams
  5. upsert         canonical fixture store, keyed on fixture_id

Step 2 before step 3 is what makes the evidence phase auditable: every canonical
row can be regenerated from a stored payload with `--replay`, so a coverage claim
can be rechecked after the window closes without re-querying anyone.

Nothing here writes to data/results.csv. Promotion of a finished fixture into the
results history is a separate, later decision (plan §7.2).

Usage:
  python3 -m scripts.international.fetch_fixtures --venues      # build venue table first
  python3 -m scripts.international.fetch_fixtures --dry-run
  python3 -m scripts.international.fetch_fixtures --write
  python3 -m scripts.international.fetch_fixtures --replay      # offline, from raw
  python3 -m scripts.international.fetch_fixtures --coverage    # report only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from international import registry as R          # noqa: E402
from international import timeutil as TU         # noqa: E402
from international import venues as V            # noqa: E402
from international.providers import bsd          # noqa: E402
from international.store import (FixtureStore, RawObservation,  # noqa: E402
                                 RawStore, SCHEDULED)

WINDOW_START, WINDOW_END = "2026-09-21", "2026-10-06"


def _key() -> str:
    from api_keys import get_key
    k = get_key("bsd", env="BSD_API_KEY")
    if not k:
        sys.exit("no BSD API key (env BSD_API_KEY or data/api_keys.json 'bsd')")
    return k


def build_venues(raw: RawStore) -> None:
    key = _key()
    records = bsd.fetch_venues(key)
    raw.write(RawObservation(bsd.PROVIDER, "venues", records))
    df = V.build(records)
    V.save(df)
    cov = V.coverage()
    print(f"venues: {cov['venues']} stored, {cov['with_coords']} with coordinates, "
          f"{cov['with_timezone']} resolved to a timezone")
    if not V.timezonefinder_available():
        print("  WARNING: timezonefinder not installed — every fixture will fall "
              "back to the UTC date and be flagged in `conflict`.")


def load_source(replay: bool, raw: RawStore) -> tuple[list[dict], list[dict]]:
    """(leagues, events) either from the network or from the raw store."""
    if replay:
        leagues = events = None
        for obs in raw.replay(bsd.PROVIDER, "leagues"):
            leagues = obs["payload"]
        for obs in raw.replay(bsd.PROVIDER, "events"):
            events = obs["payload"]
        if leagues is None or events is None:
            sys.exit("--replay needs stored leagues and events observations; "
                     "run a live fetch first")
        print(f"replaying from raw store: {len(leagues)} leagues, {len(events)} events")
        return leagues, events

    key = _key()
    leagues = bsd.fetch_leagues(key)
    raw.write(RawObservation(bsd.PROVIDER, "leagues", leagues))
    events = bsd.fetch_events(key, status="notstarted")
    raw.write(RawObservation(bsd.PROVIDER, "events", events,
                             request={"status": "notstarted"}))
    return leagues, events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="upsert into the fixture store")
    ap.add_argument("--dry-run", action="store_true", help="fetch and report, write nothing")
    ap.add_argument("--replay", action="store_true", help="reparse from the raw store, offline")
    ap.add_argument("--venues", action="store_true", help="(re)build the venue table and exit")
    ap.add_argument("--coverage", action="store_true", help="coverage report only")
    ap.add_argument("--universe", default="fifa", choices=["fifa", "all"])
    ap.add_argument("--caf", action="store_true",
                    help="also derive AFCON 2027 qualifying from the draw. No "
                         "network: every paid-free route to CAF fixtures is closed, "
                         "so these are computed from the group draw and the "
                         "round-robin template. See international/providers/caf.py")
    a = ap.parse_args()

    raw = RawStore()
    if a.venues:
        build_venues(raw)
        return

    leagues, events = load_source(a.replay, raw)

    league_map = bsd.international_league_ids(leagues)
    bad = bsd.validate_competitions(league_map)
    if bad:
        sys.exit(f"mapped competitions missing from the taxonomy: {bad}")

    unmapped = bsd.unmapped_international_leagues(leagues)
    if unmapped:
        print(f"\nUNMAPPED international-looking leagues ({len(unmapped)}) — "
              f"review before trusting coverage:")
        for n in unmapped:
            print(f"  {n}")

    rows, skipped = bsd.parse_events(events, league_map)
    print(f"\nparsed {len(rows)} international fixtures "
          f"({len(skipped)} events skipped as out of scope or unparseable)")

    # Scope filter, with three outcomes rather than two.
    #
    # A team we have deliberately classified as non-FIFA (Guadeloupe, Sint Maarten)
    # is a normal exclusion. A name we have NEVER SEEN is different: it might be a
    # newly recognised association, a rename, or — as the first spike run found with
    # "Albirex Niigata" — a club that leaked into an international league. Those are
    # quarantined and reported, never silently dropped, because silence here is how
    # a club ends up in a national-team rating model.
    kept, out_of_scope, unknown = [], [], []
    for r in rows:
        names = (r["home_team"], r["away_team"])
        try:
            if R.fixture_in_scope(*names, r["local_date"], universe=a.universe):
                kept.append(r)
                continue
        except R.ScopeError as exc:
            unknown.append({"fixture": f"{names[0]} v {names[1]}", "reason": str(exc)})
            continue
        novel = [n for n in names if R.status(n) == R.UNCLASSIFIED]
        (unknown if novel else out_of_scope).append(
            {"fixture": f"{names[0]} v {names[1]}",
             "reason": f"unrecognised name(s): {', '.join(novel)}"} if novel else r)

    print(f"in scope ({a.universe}): {len(kept)}   "
          f"known out of scope: {len(out_of_scope)}   unrecognised: {len(unknown)}")

    if out_of_scope:
        excluded = sorted({n for r in out_of_scope
                           for n in (r["home_team"], r["away_team"])
                           if R.status(n) == R.NON_FIFA})
        print(f"  excluded as non-FIFA: {', '.join(excluded)}")

    if unknown:
        print(f"\nQUARANTINE — {len(unknown)} fixture(s) with unrecognised team names:")
        for q in unknown[:10]:
            print(f"  {q['fixture']}")
        print("  These are NOT in the team registry. Either they are new/renamed "
              "national sides (add them via\n  scripts/international/seed_team_registry.py) "
              "or non-national teams the provider filed under an\n  international league "
              "(extend NON_NATIONAL_MARKERS in engines/worldcup/names.py).")

    if kept:
        df = pd.DataFrame(kept)
        print("\nby competition:")
        print(df.groupby("competition").size().sort_values(ascending=False).to_string())

        ts = TU.series_to_utc(df.kickoff_utc)
        win = df[(ts >= f"{WINDOW_START}T00:00:00Z") & (ts <= f"{WINDOW_END}T23:59:59Z")]
        print(f"\nSeptember window ({WINDOW_START} to {WINDOW_END}): {len(win)} fixtures")
        if len(win):
            print(win.groupby("competition").size().to_string())

        no_tz = int((df.venue_tz == "").sum())
        print(f"\nvenue timezone unresolved: {no_tz}/{len(df)} "
              f"({'local dates are UTC dates for these' if no_tz else 'all resolved'})")
        print(f"neutral-ground flag set: {int(df.neutral.astype(bool).sum())}/{len(df)}")

    if a.caf:
        from international.providers import caf
        problems = caf.validate_template()
        if problems:
            sys.exit(f"CAF template is invalid, refusing to emit: {problems[:3]}")
        caf_rows = caf.to_fixture_rows()
        print(f"\nCAF derived: {len(caf_rows)} AFCON 2027 qualifying fixtures "
              f"(dated to matchday windows, not published kick-offs)")
        kept.extend(caf_rows)

    if a.coverage or a.dry_run or not a.write:
        print("\n(no write — pass --write to upsert into the fixture store)")
        return

    res = FixtureStore().upsert(kept)
    print(f"\nfixture store: +{res['added']} new, {res['updated']} updated, "
          f"{res['total']} total")


if __name__ == "__main__":
    main()
