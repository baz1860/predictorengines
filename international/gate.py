"""Blocking health gate for international data (plan §7.5).

The problem this fixes
----------------------
`update.sh` ran the validation gate like this:

    # Warn loudly on regression but NEVER block the daily update (|| guard).
    python3 -m engines.worldcup.validate --quiet --gate \
      || echo "   ##### VALIDATION GATE FAILED ..."

So the gate printed and the pipeline carried on. It also scored a single pooled
number across all competitions, which — given that measured skill ranges from 0.09
to 0.35 across competitions — means a regression in one can be masked by another.

This module is the data-integrity half of the replacement. It exits non-zero, and
`update.sh` no longer swallows the exit code.

Checks:
  fixtures   duplicate/ambiguous pairs and past-kickoff blank rows (fixtures.py)
  scope      every ACTIVE team classified in the registry (registry.py)
  taxonomy   every competition label classified (taxonomy.py)

`--strict` additionally fails on exceptions still marked pending_review, so a
backlog cannot be carried indefinitely. Use it in release, not in the daily run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results.csv"


def run(strict: bool = False, asof: object = None) -> list[str]:
    """Return a list of failure messages. Empty means healthy."""
    from . import fixtures as F
    from . import registry as R
    from . import taxonomy as T

    failures: list[str] = []

    try:
        df = pd.read_csv(RESULTS)
    except Exception as exc:                                  # pragma: no cover
        return [f"cannot read {RESULTS.name}: {exc}"]

    try:
        F.assert_invariants(df, asof=asof, allow_pending=not strict)
    except F.FixtureIntegrityError as exc:
        failures.append(str(exc))

    try:
        R.assert_scope_complete()
    except R.ScopeError as exc:
        failures.append(str(exc))

    missing = T.unmapped()
    if missing:
        failures.append(
            f"taxonomy: {len(missing)} competition label(s) unclassified: "
            f"{', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}")

    # A NEW pattern-matched label means a feed introduced a competition we have
    # never classified. It is running on a guessed weight until someone looks.
    novel = T.unacknowledged_provisional()
    if novel:
        failures.append(
            f"taxonomy: {len(novel)} NEW competition label(s) running on a guessed "
            f"weight: {', '.join(novel[:8])}{' …' if len(novel) > 8 else ''}. "
            f"Add them to EXPLICIT or ACKNOWLEDGED_PROVISIONAL in "
            f"international/taxonomy.py.")

    failures.extend(fixture_store_failures(strict=strict, asof=asof))
    return failures


def fixture_store_failures(strict: bool = False, asof: object = None) -> list[str]:
    """Health of the canonical fixture store. Empty when it has not been built.

    Separate from the results.csv checks because the two stores fail differently:
    results.csv accumulates duplicates from merges, while the fixture store
    accumulates *stale scheduled rows* — fixtures a provider announced and then
    never updated. Both end up presenting a finished match as forthcoming.
    """
    from . import fixtures as F
    from .store import SCHEDULED, FixtureStore

    out: list[str] = []
    df = FixtureStore().load()
    if df.empty:
        return out

    dupe_ids = df.fixture_id.duplicated().sum()
    if dupe_ids:
        out.append(f"fixture store: {dupe_ids} duplicated fixture_id(s) — upsert is broken")

    now = F._naive_now(asof)
    dates = pd.to_datetime(df.local_date, errors="coerce")
    stale = df[(df.status == SCHEDULED)
               & (dates < now - pd.Timedelta(days=F.STALE_AFTER_DAYS))]
    if len(stale):
        head = "; ".join(f"{r.home_team} v {r.away_team} ({r.local_date})"
                         for r in stale.head(5).itertuples(index=False))
        msg = (f"fixture store: {len(stale)} fixture(s) still 'scheduled' past "
               f"kickoff: {head}{' …' if len(stale) > 5 else ''}")
        # Warn by default, block in strict: a provider lagging on results is
        # normal for a day or two and should not stop the daily run.
        if strict:
            out.append(msg)
        else:
            print(f"[gate] WARNING: {msg}", file=sys.stderr)

    # UTC storage invariant. Every stored instant must carry an explicit offset.
    # A naive timestamp in the store is not a cosmetic issue: it is ambiguous by
    # exactly the amount that produced the July 2026 duplicates, and it will be
    # silently reinterpreted as UTC by whatever reads it next.
    from . import timeutil as TU

    audit = TU.audit_utc_column(df.kickoff_utc)
    if audit["implicit_or_bad"]:
        out.append(
            f"fixture store: {audit['implicit_or_bad']} kickoff_utc value(s) lack an "
            f"explicit UTC offset. Every stored instant must end '+00:00' — an "
            f"ambiguous timestamp is the defect that duplicated two World Cup "
            f"fixtures.")

    if strict:
        no_tz = int((df.venue_tz.isna() | (df.venue_tz.astype(str).str.strip() == "")).sum())
        if no_tz:
            out.append(
                f"fixture store: {no_tz}/{len(df)} fixture(s) have no venue timezone, "
                f"so their local_date is the UTC date and may be off by one day")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="International data health gate")
    ap.add_argument("--strict", action="store_true",
                    help="also fail on exceptions still pending review")
    ap.add_argument("--asof", default=None, help="date for the staleness check")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    failures = run(strict=a.strict, asof=a.asof)
    if failures:
        print("[gate] INTERNATIONAL DATA GATE FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    if not a.quiet:
        print("[gate] international data healthy "
              f"({'strict' if a.strict else 'daily'} mode)")


if __name__ == "__main__":
    main()
