#!/usr/bin/env python3
"""One-off audited migration: fix the calendar-year season on legacy
OpenFootball rows.

The seed_openfootball parser now assigns European seasons on the July boundary
(`season = year if month >= 7 else year - 1`), but rows written before that fix
kept the calendar year for every January-June match. This migration repairs
ONLY those rows, and ONLY the `season` column.

Rows are identified deterministically: a row is an OpenFootball row iff its
stored `fixture_id` recomputes exactly from the OpenFootball ID scheme
(md5("of|comp_id|date|home|away")). Names/dates are read as-is, so a row that
does not recompute is left completely untouched. Every other column is
preserved byte-for-byte (the CSV is read as strings with NA disabled).

Usage:
  python3 -m club_soccer.migrate_openfootball_season            # dry-run report
  python3 -m club_soccer.migrate_openfootball_season --apply    # write the fix
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "data" / "fixtures.csv"


def of_fixture_id(comp_id: str, date: str, home: str, away: str) -> str:
    """Reproduce seed_openfootball.fid() as a string for comparison."""
    h = hashlib.md5(f"of|{comp_id}|{date}|{home}|{away}".encode()).hexdigest()
    return str(int(h[:8], 16))


def _expected_season(date: str) -> str | None:
    """August-July season for a YYYY-MM-DD date, or None if unparseable."""
    try:
        yr, mo = int(date[:4]), int(date[5:7])
    except (ValueError, IndexError):
        return None
    return str(yr if mo >= 7 else yr - 1)


def plan(df: pd.DataFrame) -> list[int]:
    """Row indices whose season must change (OpenFootball rows on the wrong
    side of the July boundary)."""
    fix: list[int] = []
    for i, r in df.iterrows():
        try:
            recomputes = of_fixture_id(
                r["competition_id"], r["date"], r["home"], r["away"]
            ) == str(r["fixture_id"])
        except KeyError:
            return []                                   # schema mismatch — do nothing
        if not recomputes:
            continue
        want = _expected_season(r["date"])
        if want is not None and str(r["season"]) != want:
            fix.append(i)
    return fix


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="write the fix (default is a dry run)")
    ap.add_argument("--fixtures", default=str(FIXTURES))
    ap.add_argument("--expect", type=int, default=None,
                    help="assert exactly this many rows change (safety pin)")
    args = ap.parse_args()

    path = Path(args.fixtures)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    of_rows = sum(
        of_fixture_id(r["competition_id"], r["date"], r["home"], r["away"])
        == str(r["fixture_id"]) for _, r in df.iterrows()
    )
    to_fix = plan(df)
    print(f"total rows            : {len(df)}")
    print(f"OpenFootball-ID rows  : {of_rows}")
    print(f"rows needing season fix: {len(to_fix)}")
    if to_fix:
        sample = [(df.at[i, 'date'], df.at[i, 'season'],
                   _expected_season(df.at[i, 'date'])) for i in to_fix[:6]]
        print("  sample (date, old_season, new_season):")
        for d, old, new in sample:
            print(f"    {d}  {old} -> {new}")

    if args.expect is not None and len(to_fix) != args.expect:
        sys.exit(f"ABORT: expected {args.expect} changes, found {len(to_fix)}")

    if not args.apply:
        print("\n(dry run — pass --apply to write)")
        return
    if not to_fix:
        print("nothing to do.")
        return

    before = df.copy()
    for i in to_fix:
        df.at[i, "season"] = _expected_season(df.at[i, "date"])

    # Verify ONLY the season column changed, and only on the planned rows.
    diff = (before != df)
    changed_cols = [c for c in df.columns if diff[c].any()]
    if changed_cols != ["season"]:
        sys.exit(f"ABORT: unexpected columns changed: {changed_cols}")
    changed_rows = sorted(df.index[diff["season"]])
    if changed_rows != sorted(to_fix):
        sys.exit("ABORT: changed rows do not match the plan")

    backup = path.with_suffix(".csv.bak.of_season")
    shutil.copy2(path, backup)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    print(f"\napplied {len(to_fix)} season fixes.")
    print(f"backup -> {backup}")


if __name__ == "__main__":
    main()
