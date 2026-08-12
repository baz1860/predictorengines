#!/usr/bin/env python3
"""Poll odds for upcoming international fixtures (plan §8, odds evidence).

Polling horizon
---------------
Fixtures within `POLL_HORIZON_DAYS` of kickoff are polled, nearest first, and
`--max-requests` caps any single run so a misconfigured cron cannot run away.

The horizon is 60 days, not the 14 you would choose if you already knew when prices
appear. We do not: BSD returned nothing at six weeks out, and the decision was to
wait rather than pay. A window that opens after prices first appear cannot measure
when they first appeared — it would report "day 14" whatever the truth. The wider
horizon costs nothing because BSD is free and un-rate-limited.

Absence is recorded
-------------------
A fixture that returns no prices gets a row saying so. Otherwise "no odds existed"
and "we never asked" look identical later, and only one of them is a finding.
`--report` turns those rows into the answer: `absence_profile()` shows the priced
share by time-to-kickoff, and `first_price_timing()` shows when a price first
appeared for each fixture.

Usage:
  python3 -m scripts.international.fetch_odds --dry-run
  python3 -m scripts.international.fetch_odds --write
  python3 -m scripts.international.fetch_odds --write --window-only --max-requests 200
  python3 -m scripts.international.fetch_odds --report        # do prices ever arrive?
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from international import odds as O                          # noqa: E402
from international import timeutil as TU                      # noqa: E402
from international.store import (FixtureStore, RawObservation,  # noqa: E402
                                 RawStore, SCHEDULED)

BSD_BASE = "https://sports.bzzoiro.com"
WINDOW_START, WINDOW_END = "2026-09-21", "2026-10-06"


# Polling cadence, calibrated against MEASURED BSD behaviour rather than guessed.
#
# Recovered from the stored World Cup 2026 raw payloads (557 observations across
# 64 fixtures), BSD populates its inline odds like this:
#
#     < 6h      100% priced        1 - 3 days   96% priced
#     6 - 24h   100% priced        3 - 7 days   21% priced
#     > 7 days  never observed (we never sampled that far out)
#
# Earliest price per fixture: median 60h before kickoff, max 130h. Prices arrive
# around three days out and are universal inside a day. 62 of 64 fixtures were
# eventually priced.
#
# Two consequences for the schedule:
#   1. Polling every fixture inside 60 days spends most calls on fixtures nobody
#      has priced yet.
#   2. Far more importantly, a FLAT cadence samples the 0-72h window — where the
#      price actually moves, and where closing-line value lives — no more densely
#      than the empty weeks before it. Sparse sampling near kickoff means no usable
#      closing price, which is the one measurement this whole exercise exists to
#      produce.
#
# So: keep a long horizon (it is free, and confirmed absence is still evidence),
# but shrink the interval sharply as kickoff approaches.
#
# CAVEAT: those figures come from a World Cup, the most heavily traded international
# event there is. Friendlies and Nations League will likely be priced later and less
# completely, so treat 60h as an optimistic upper bound and let the live window
# correct it.
POLL_HORIZON_DAYS = 60

# (hours_to_kickoff_at_or_below, minimum_hours_between_polls)
POLL_CADENCE = [
    (6, 0.5),         # closing price — every 30 minutes
    (24, 1.0),        # 100% priced here; capture the drift
    (72, 4.0),        # 96% priced; prices arriving and moving
    (168, 12.0),      # 21% priced; catch first appearance
    (24 * 60, 48.0),  # discovery only: confirm absence every couple of days
]


def poll_interval(hours: float) -> float:
    """Minimum hours between polls for a fixture this far from kickoff."""
    for below, interval in POLL_CADENCE:
        if hours <= below:
            return interval
    return float("inf")


def poll_due(hours: float, hours_since_last: float | None = None) -> bool:
    """Is this fixture due a snapshot?

    A missing `hours_since_last` means never polled, which is always due. Note it
    must be tested for NaN as well as None: `Series.map` turns a returned None into
    NaN, and `NaN >= interval` is False — so the naive `is None` check silently
    marked every never-polled fixture as NOT due, which is the exact opposite of
    the intent and made the poller do nothing at all.
    """
    if hours is None or hours != hours or hours < 0:      # NaN or already played
        return False
    if hours > 24 * POLL_HORIZON_DAYS:
        return False
    if hours_since_last is None or hours_since_last != hours_since_last:
        return True
    return hours_since_last >= poll_interval(hours)


def last_poll_times(store) -> dict:
    """fixture_id -> most recent snapshot timestamp, from the odds store."""
    df = store.load()
    if df.empty:
        return {}
    ts = TU.series_to_utc(df.snapshot_at)
    return df.assign(_ts=ts).groupby("fixture_id")._ts.max().to_dict()


def _get(path: str, key: str, timeout: int = 25):
    req = urllib.request.Request(
        BSD_BASE + path,
        headers={"Authorization": f"Token {key}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true", help="coverage report only")
    ap.add_argument("--window-only", action="store_true",
                    help=f"restrict to {WINDOW_START}..{WINDOW_END}")
    ap.add_argument("--max-requests", type=int, default=300)
    ap.add_argument("--ignore-schedule", action="store_true",
                    help="poll every future fixture regardless of time to kickoff. "
                         "Use to record a baseline: 'we asked at T-6 weeks and there "
                         "was nothing' is evidence, and it cannot be gathered later.")
    a = ap.parse_args()

    store = O.OddsStore()
    if a.report:
        cov = store.coverage()
        print("odds store coverage")
        for k, v in cov.items():
            print(f"  {k:<18}{v}")
        df = store.load()
        if not df.empty:
            print("\nby status:")
            print(df.status.value_counts().to_string())
            print("\npricing by time to kickoff (the 'do prices ever arrive' question):")
            print(store.absence_profile().to_string(index=False))
            first = store.first_price_timing()
            if len(first):
                print("\nwhen a price FIRST appeared, per fixture:")
                print(first.to_string(index=False))
                print(f"\nmedian first-price lead time: "
                      f"{first.hours_before_kickoff.median():.1f}h")
            else:
                print("\nNo fixture has EVER carried a price. That is the finding, "
                      "not a bug —\nsee the bucket table above for how far out we "
                      "have asked so far.")
            priced = df[df.status == O.PRICED]
            if len(priced):
                print("\npriced fixtures by competition:")
                print(priced.groupby("competition").fixture_id.nunique().to_string())
        return

    fx = FixtureStore().load()
    if fx.empty:
        sys.exit("fixture store is empty — run fetch_fixtures.py first")

    fx = fx[fx.status == SCHEDULED].copy()
    ko = TU.series_to_utc(fx.kickoff_utc)
    now = TU.now_utc()
    fx["hours"] = (ko - now).dt.total_seconds() / 3600.0

    # Fixtures with no kick-off time cannot be polled on a cadence — there is
    # nothing to measure "hours to kickoff" against. Currently this is the entire
    # CAF derived set, which is dated to matchday windows rather than published
    # kick-offs. Report it rather than let 144 fixtures vanish from the schedule
    # silently.
    undated = fx[ko.isna()]
    if len(undated):
        by_provider = undated.groupby("provider").size().to_dict()
        print(f"NOT POLLABLE: {len(undated)} scheduled fixture(s) have no kick-off "
              f"time {by_provider}.\n  These are window-dated, not published times. "
              f"They can be predicted but not priced\n  until a dated source "
              f"supplies kick-offs.")
        fx = fx[ko.notna()].copy()

    if a.window_only:
        fx = fx[(ko >= f"{WINDOW_START}T00:00:00Z") & (ko <= f"{WINDOW_END}T23:59:59Z")]

    future = fx[fx.hours > 0]

    # Cadence depends on when each fixture was last polled, not just how far off it
    # is. Without this the poller either re-polls everything every run (wasteful and
    # noisy) or applies one interval everywhere (too sparse near kickoff to capture
    # a closing price).
    last = last_poll_times(store)
    now = TU.now_utc()
    since = fx.fixture_id.map(
        lambda fid: None if fid not in last
        else (now - last[fid]).total_seconds() / 3600.0)

    if a.ignore_schedule:
        due = future.sort_values("hours")
    else:
        mask = [poll_due(h, s) for h, s in zip(fx.hours, since)]
        due = fx[mask].sort_values("hours")

    print(f"scheduled fixtures: {len(fx)}   future: {len(future)}   "
          f"due a snapshot: {len(due)}"
          + (" (schedule overridden)" if a.ignore_schedule
             else f" (cadence-based, horizon {POLL_HORIZON_DAYS}d)"))
    if len(due) and not a.ignore_schedule:
        nearest = due.hours.min()
        print(f"  nearest kickoff {nearest/24:.1f}d out — polling every "
              f"{poll_interval(nearest):.1f}h at that range")
    if len(due) > a.max_requests:
        print(f"  budget caps this run at {a.max_requests}; nearest kickoffs first")
        due = due.head(a.max_requests)

    if due.empty:
        if len(future):
            # min of FUTURE hours; fx.hours.min() is the most PAST fixture, which
            # reported a nonsensical negative "nearest kickoff".
            print(f"\nNothing due. Nearest kickoff is {future.hours.min()/24:.1f} "
                  f"days away.")
        else:
            print("\nNothing due — no future fixtures in the store.")
        print("Odds are generally not published this far out; that is expected. "
              "Use --ignore-schedule to record a baseline observation anyway.")
        return

    if not (a.write or a.dry_run):
        print("\n(no write — pass --write or --dry-run)")
        return

    from api_keys import get_key
    key = get_key("bsd", env="BSD_API_KEY")
    if not key:
        sys.exit("no BSD API key")

    raw = RawStore()
    rows: list[dict] = []
    priced = absent = failed = 0

    for _, f in due.iterrows():
        fixture = f.to_dict()
        eid = fixture.get("provider_event_id")
        try:
            detail = _get(f"/api/events/{eid}/", key)
            compare = _get(f"/api/odds/?event={eid}", key)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            rows.append(O.record_absence(fixture, status=O.FETCH_FAILED))
            failed += 1
            print(f"  FAIL {fixture['home_team']} v {fixture['away_team']}: "
                  f"{type(exc).__name__}")
            continue

        got = (O.parse_bsd_comparison(compare, fixture)
               or O.parse_bsd_inline(detail, fixture))
        if got:
            rows.extend(got)
            priced += 1
        else:
            rows.append(O.record_absence(fixture))
            absent += 1

    if a.write and rows:
        raw.write(RawObservation("bsd", "odds_poll",
                                 {"fixtures": len(due), "rows": len(rows)}))

    print(f"\npolled {len(due)} fixture(s): {priced} priced, {absent} returned no "
          f"odds, {failed} failed")
    print(f"rows to store: {len(rows)}")

    if a.dry_run or not a.write:
        print("(dry run — nothing written)")
        return

    n = store.append(rows)
    print(f"appended {n} row(s) to {store.path.relative_to(ROOT)}")
    for k, v in store.coverage().items():
        print(f"  {k:<18}{v}")


if __name__ == "__main__":
    main()
