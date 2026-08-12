#!/usr/bin/env python3
"""Repair duplicate fixtures in data/results.csv and seed the adjudication ledger.

Plan §7.1. Two classes of problem, handled differently on purpose:

  AUTO   one row scored + one row blank, same teams/competition/venue, <=1 day
         apart. Unambiguous UTC/local date split. The blank row is dropped: it
         carries no information the scored row lacks, and while it exists
         `load_matches()` presents a finished match as an upcoming fixture.

  REVIEW both rows scored with identical results. Almost certainly the same
         duplication mechanism, but dropping one deletes a historical result and
         shifts every Elo rating computed after it. These are written to
         data/international/fixture_exceptions.csv as `pending_review` so the
         gate stops treating them as new, and a human decides.

Usage:
  python3 -m scripts.international.repair_fixtures              # dry run
  python3 -m scripts.international.repair_fixtures --write      # apply
  python3 -m scripts.international.repair_fixtures --write --backup
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

from international import fixtures as F  # noqa: E402
from international import timeutil as TU  # noqa: E402

RESULTS = ROOT / "data" / "results.csv"
EXCEPTIONS = F.EXCEPTIONS_CSV
EXC_COLUMNS = ["pair_id", "decision", "home", "away", "competition",
               "date_a", "date_b", "reason", "recorded_at", "note"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply changes")
    ap.add_argument("--backup", action="store_true", help="copy results.csv aside first")
    ap.add_argument("--asof", default=None, help="date for the staleness check")
    a = ap.parse_args()

    df = pd.read_csv(RESULTS)
    print(f"results.csv: {len(df)} rows, {df.home_score.notna().sum()} played")

    pairs = F.find_duplicates(df)
    auto = [p for p in pairs
            if any(r in p.reason for r in F.AUTO_RESOLVABLE_REASONS)]
    review = [p for p in pairs if p not in auto]

    print(f"\nAUTO-RESOLVABLE ({len(auto)}) — blank row dropped:")
    for p in auto:
        print(f"  {p.home} v {p.away} ({p.competition})  {p.date_a} / {p.date_b}")

    print(f"\nNEEDS REVIEW ({len(review)}) — both scored, recorded as pending:")
    for p in review:
        print(f"  {p.home} v {p.away} ({p.competition})  {p.date_a} / {p.date_b}")

    # Use the REAL adjudication ledger, not an empty dict. An earlier version passed
    # `exceptions={}` here — correct on the first run when nothing was adjudicated,
    # but it meant that once a pair WAS marked accepted_duplicate the script silently
    # kept ignoring it and reported "0 rows dropped".
    kept, unresolved = F.reconcile(df)
    dropped = len(df) - len(kept)
    print(f"\nrows dropped: {dropped}   rows remaining: {len(kept)}")

    stale = F.stale_blanks(kept, asof=a.asof)
    print(f"stale blank rows AFTER repair: {len(stale)}")
    if not stale.empty:
        print(stale[["date", "home_team", "away_team", "tournament"]].to_string(index=False))

    now = TU.now_iso()
    existing = F.load_exceptions()
    new_rows = []
    for p in review:
        pid = F._pair_id(p)
        if pid in existing:
            continue
        new_rows.append({
            "pair_id": pid, "decision": F.PENDING, "home": p.home, "away": p.away,
            "competition": p.competition, "date_a": p.date_a, "date_b": p.date_b,
            "reason": p.reason, "recorded_at": now,
            "note": "identical score on consecutive days at the same venue; "
                    "confirm against an external source before dropping either row",
        })

    if not a.write:
        print(f"\n(dry run — would drop {dropped} row(s) and record "
              f"{len(new_rows)} pending exception(s); pass --write)")
        return

    if a.backup:
        dest = RESULTS.with_suffix(f".csv.bak.dupes.{now[:10].replace('-', '')}")
        shutil.copy2(RESULTS, dest)
        print(f"backup -> {dest.name}")

    kept.to_csv(RESULTS, index=False)
    print(f"wrote {RESULTS.relative_to(ROOT)} ({len(kept)} rows)")

    if new_rows:
        EXCEPTIONS.parent.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame(new_rows, columns=EXC_COLUMNS)
        if EXCEPTIONS.exists():
            old = pd.read_csv(EXCEPTIONS, dtype=str, keep_default_na=False)
            out = pd.concat([old, out], ignore_index=True)
        out.to_csv(EXCEPTIONS, index=False)
        print(f"wrote {EXCEPTIONS.relative_to(ROOT)} ({len(out)} rows)")


if __name__ == "__main__":
    main()
