#!/usr/bin/env python3
"""Cross-provider agreement report (plan §6.2, §8 step 3).

The plan's most-repeated mistake has been pre-committing to a provider before
measuring it: the June plan named API-Football, revision 1 named BSD, and both
were written before any comparison existed. This script is the thing that stops it
happening a third time.

It answers three questions, for every provider registered:

  1. **Recall** — which fixtures does provider B know about that A does not?
  2. **Agreement** — where both know a fixture, do they agree on the date?
  3. **Independence** — is provider B actually a separate observation, or a mirror
     of the same upstream data? (openfootball mirrors martj42, which is already our
     results history, so it cannot corroborate anything.)

Question 3 matters most and is the easiest to get wrong. Two sources agreeing means
nothing if one is a copy of the other.

Usage:
  python3 -m scripts.international.compare_providers
  python3 -m scripts.international.compare_providers --json report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from international.identity import DATE_TOLERANCE_DAYS, signature  # noqa: E402
from international.providers import thesportsdb                     # noqa: E402
from international.store import FixtureStore                        # noqa: E402

# Independence notes, recorded so a future reader does not mistake a mirror for
# corroboration.
INDEPENDENCE = {
    "bsd": "independent commercial feed",
    "thesportsdb": "independent community database",
    "openfootball": "MIRROR of martj42/international_results — the same upstream as "
                    "data/results.csv. Cannot corroborate our own history.",
    "football-data.org": "independent, but only 12 competitions on the free tier",
}


def compare(base: pd.DataFrame, other: list[dict], name: str) -> dict:
    """Agreement between the canonical store and one provider's view."""
    if base.empty:
        return {"provider": name, "error": "fixture store empty"}

    base_sig = {}
    for r in base.itertuples(index=False):
        base_sig.setdefault(
            signature(r.home_team, r.away_team, r.competition), []
        ).append(str(r.local_date))

    both = only_other = date_mismatch = 0
    novel, mismatches = [], []
    for row in other:
        sig = signature(row["home_team"], row["away_team"], row["competition"])
        if sig not in base_sig:
            only_other += 1
            novel.append(f"{row['home_team']} v {row['away_team']} "
                         f"({row['local_date']}, {row['competition']})")
            continue
        both += 1
        theirs = pd.Timestamp(row["local_date"])
        gaps = [abs((theirs - pd.Timestamp(d)).days) for d in base_sig[sig]]
        if min(gaps) > DATE_TOLERANCE_DAYS:
            date_mismatch += 1
            mismatches.append(f"{row['home_team']} v {row['away_team']}: "
                              f"ours {base_sig[sig][0]} vs theirs {row['local_date']}")

    return {
        "provider": name,
        "independence": INDEPENDENCE.get(name, "unknown"),
        "provider_fixtures": len(other),
        "in_both": both,
        "only_in_provider": only_other,
        "date_disagreements": date_mismatch,
        "agreement_rate": round(both / len(other), 3) if other else None,
        "novel_examples": novel[:10],
        "mismatch_examples": mismatches[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    base = FixtureStore().load()
    print(f"canonical fixture store: {len(base)} fixtures "
          f"(provider: {sorted(set(base.provider)) if len(base) else '—'})\n")

    reports = []
    try:
        tsdb = thesportsdb.fetch_all()
        reports.append(compare(base, tsdb, "thesportsdb"))
    except Exception as exc:
        reports.append({"provider": "thesportsdb", "error":
                        f"{type(exc).__name__}: {exc}"})

    for r in reports:
        print(f"── {r['provider']} ──")
        if "error" in r:
            print(f"   unavailable: {r['error']}\n")
            continue
        print(f"   independence      {r['independence']}")
        print(f"   fixtures seen     {r['provider_fixtures']}")
        print(f"   also in our store {r['in_both']}")
        print(f"   ONLY in provider  {r['only_in_provider']}")
        print(f"   date disagreements{r['date_disagreements']:>4}")
        for n in r["novel_examples"]:
            print(f"     + {n}")
        for m in r["mismatch_examples"]:
            print(f"     ! {m}")
        print()

    print("Independence notes:")
    for k, v in INDEPENDENCE.items():
        print(f"  {k:<20} {v}")
    print("\nA provider that mirrors our own upstream cannot corroborate us. "
          "openfootball is listed\nfor completeness and deliberately NOT used as a "
          "cross-check for that reason.")

    if a.json:
        a.json.write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
