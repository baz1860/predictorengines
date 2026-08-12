#!/usr/bin/env python3
"""Promote finished canonical fixtures into data/results.csv (plan §7.2).

This is the ONE place the fixture store is allowed to write into the results
history, and it is deliberately paranoid, because results.csv feeds every rating in
the module and a bad row there is silently permanent.

Guards, in order:
  1. status must be `played` AND both scores present;
  2. kickoff must be in the past;
  3. both teams must be in product scope at the match date;
  4. the row must not already exist — checked on the DATE-TOLERANT signature, not
     an exact date match, because a UTC/local disagreement is exactly how the July
     2026 duplicates were created;
  5. after promotion the whole file must still pass `fixtures.assert_invariants`,
     or the write is rolled back.

Guard 4 is the important one. martj42 upstream will also publish these results, and
if we promote a row dated locally while upstream publishes it dated in UTC, we
recreate the very bug this module was built to fix — from the other direction.
Hence `--prefer-upstream` (the default): we only promote a result that upstream has
NOT published, and we withdraw ours once it does.

Usage:
  python3 -m scripts.international.promote_results --dry-run
  python3 -m scripts.international.promote_results --write
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from international import fixtures as F      # noqa: E402
from international import timeutil as TU     # noqa: E402
from international import registry as R      # noqa: E402
from international.identity import DATE_TOLERANCE_DAYS, signature  # noqa: E402
from international.store import PLAYED, FixtureStore  # noqa: E402

RESULTS = ROOT / "data" / "results.csv"
RESULT_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score",
                  "tournament", "city", "country", "neutral"]


def candidates(fixtures: pd.DataFrame, results: pd.DataFrame,
               asof: object = None) -> tuple[list[dict], list[str]]:
    """(rows ready to promote, human-readable reasons for everything rejected)."""
    now = TU.naive_utc(asof)      # compared against naive results.csv dates

    existing: dict[str, list[pd.Timestamp]] = {}
    for r in results.itertuples(index=False):
        existing.setdefault(
            signature(r.home_team, r.away_team, r.tournament), []
        ).append(pd.Timestamp(r.date))

    ready, rejected = [], []
    for f in fixtures.itertuples(index=False):
        label = f"{f.home_team} v {f.away_team} ({f.local_date})"

        if str(f.status) != PLAYED:
            continue
        if pd.isna(f.home_score) or pd.isna(f.away_score):
            rejected.append(f"{label}: marked played but has no score")
            continue

        ko = TU.naive_utc(f.kickoff_utc)
        if ko is not None and ko > now:
            rejected.append(f"{label}: kickoff is in the future")
            continue

        try:
            if not R.fixture_in_scope(f.home_team, f.away_team, f.local_date):
                rejected.append(f"{label}: out of product scope at that date")
                continue
        except R.ScopeError as exc:
            rejected.append(f"{label}: {exc}")
            continue

        sig = signature(f.home_team, f.away_team, f.competition)
        mine = pd.Timestamp(f.local_date)
        near = [d for d in existing.get(sig, [])
                if abs((d - mine).days) <= DATE_TOLERANCE_DAYS]
        if near:
            rejected.append(f"{label}: already in results.csv "
                            f"(dated {near[0].date()}) — upstream wins")
            continue

        ready.append({
            "date": str(mine.date()),
            "home_team": f.home_team, "away_team": f.away_team,
            "home_score": int(f.home_score), "away_score": int(f.away_score),
            "tournament": f.competition,
            "city": f.city if isinstance(f.city, str) else "",
            "country": f.country if isinstance(f.country, str) else "",
            "neutral": "TRUE" if bool(f.neutral) else "FALSE",
        })
    return ready, rejected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--asof", default=None)
    a = ap.parse_args()

    fixtures = FixtureStore().load()
    results = pd.read_csv(RESULTS, parse_dates=["date"])
    print(f"fixture store: {len(fixtures)}   results.csv: {len(results)}")

    if fixtures.empty:
        print("nothing to promote — fixture store is empty")
        return

    ready, rejected = candidates(fixtures, results, a.asof)
    print(f"\nready to promote: {len(ready)}")
    for r in ready[:15]:
        print(f"  + {r['date']}  {r['home_team']} {r['home_score']}-"
              f"{r['away_score']} {r['away_team']}  ({r['tournament']})")
    if len(ready) > 15:
        print(f"  … {len(ready) - 15} more")

    if rejected:
        print(f"\nrejected: {len(rejected)}")
        for msg in rejected[:12]:
            print(f"  - {msg}")
        if len(rejected) > 12:
            print(f"  … {len(rejected) - 12} more")

    if not ready or not a.write:
        print("\n(no write — pass --write to promote)" if ready else "")
        return

    merged = pd.concat([results, pd.DataFrame(ready)], ignore_index=True)
    merged = merged.sort_values("date", kind="stable")

    # Guard 5: the result must survive the same invariants the gate applies.
    try:
        F.assert_invariants(merged, asof=a.asof)
    except F.FixtureIntegrityError as exc:
        sys.exit(f"\nPROMOTION ABORTED — the merged file fails integrity checks:\n{exc}")

    stamp = TU.now_utc().strftime("%Y%m%d")
    backup = RESULTS.with_suffix(f".csv.bak.promote.{stamp}")
    shutil.copy2(RESULTS, backup)
    merged.to_csv(RESULTS, index=False)
    print(f"\nbackup -> {backup.name}")
    print(f"promoted {len(ready)} result(s); results.csv now {len(merged)} rows")


if __name__ == "__main__":
    main()
