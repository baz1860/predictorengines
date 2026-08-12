"""Promotion of finished fixtures into results.csv.

results.csv feeds every rating in the module, and a bad row there is silently
permanent — nothing downstream re-validates it. So each guard is tested
individually rather than trusting that "it returned zero rows" means it works.

The guard that matters most is the duplicate check, because it is the one that can
recreate the original bug from the other direction: if we promote a result dated
locally while martj42 upstream publishes the same match dated in UTC, the next
merge produces the pair of rows this whole module exists to prevent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.international.promote_results import candidates  # noqa: E402

FAIL = 0
NOW = "2026-10-10"


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


def fixture(**kw) -> dict:
    base = {
        "fixture_id": "f1", "provider_event_id": "1", "status": "played",
        "kickoff_utc": "2026-09-24T18:45:00+00:00", "local_date": "2026-09-24",
        "venue_tz": "Europe/Madrid", "home_team": "Spain", "away_team": "France",
        "competition": "UEFA Nations League", "city": "Madrid", "country": "Spain",
        "neutral": False, "home_score": 2, "away_score": 1,
        "provider": "bsd", "observed_at": "", "raw_ref": "", "conflict": "",
        "signature": "", "category": "",
    }
    base.update(kw)
    return base


def results(rows=()) -> pd.DataFrame:
    cols = ["date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "city", "country", "neutral"]
    df = pd.DataFrame(list(rows), columns=cols)
    df["date"] = pd.to_datetime(df["date"]) if len(df) else pd.to_datetime([])
    return df


def test_happy_path() -> None:
    print("\npromotion, happy path")
    ready, rejected = candidates(pd.DataFrame([fixture()]), results(), NOW)
    check("a finished, in-scope, unseen fixture is promoted", len(ready) == 1)
    if ready:
        r = ready[0]
        check("scores carried as integers",
              (r["home_score"], r["away_score"]) == (2, 1))
        check("competition becomes the tournament label",
              r["tournament"] == "UEFA Nations League")
        check("neutral serialised as upstream expects", r["neutral"] == "FALSE")
        check("date is the LOCAL date, matching results.csv convention",
              r["date"] == "2026-09-24")


def test_guards() -> None:
    print("\nguards")
    ready, _ = candidates(pd.DataFrame([fixture(status="scheduled")]), results(), NOW)
    check("a scheduled fixture is not promoted", ready == [])

    ready, rej = candidates(
        pd.DataFrame([fixture(home_score=None, away_score=None)]), results(), NOW)
    check("played-but-unscored is rejected and explained",
          ready == [] and any("no score" in m for m in rej), str(rej))

    ready, rej = candidates(
        pd.DataFrame([fixture(kickoff_utc="2027-01-01T00:00:00+00:00")]),
        results(), NOW)
    check("a future kickoff is rejected",
          ready == [] and any("future" in m for m in rej), str(rej))

    ready, rej = candidates(
        pd.DataFrame([fixture(home_team="Guadeloupe")]), results(), NOW)
    check("an out-of-scope side is rejected",
          ready == [] and any("scope" in m for m in rej), str(rej))

    # Effective dating: Kosovo joined FIFA in 2016, so a 2010 fixture is out of scope.
    ready, rej = candidates(
        pd.DataFrame([fixture(home_team="Kosovo", local_date="2010-06-01",
                              kickoff_utc="2010-06-01T18:00:00+00:00")]),
        results(), NOW)
    check("scope is judged at the MATCH date, not today",
          ready == [] and any("scope" in m for m in rej), str(rej))


def test_duplicate_guard() -> None:
    print("\nduplicate guard (the one that can recreate the original bug)")
    existing = results([("2026-09-24", "Spain", "France", 2, 1,
                         "UEFA Nations League", "Madrid", "Spain", "FALSE")])
    ready, rej = candidates(pd.DataFrame([fixture()]), existing, NOW)
    check("an exact existing result blocks promotion",
          ready == [] and any("already in results" in m for m in rej), str(rej))

    # The real case: upstream dated it one day later (UTC vs local).
    off_by_one = results([("2026-09-25", "Spain", "France", 2, 1,
                           "UEFA Nations League", "Madrid", "Spain", "FALSE")])
    ready, rej = candidates(pd.DataFrame([fixture()]), off_by_one, NOW)
    check("a result dated ONE DAY apart still blocks promotion",
          ready == [] and any("already in results" in m for m in rej), str(rej))

    far = results([("2026-10-02", "Spain", "France", 2, 1,
                    "UEFA Nations League", "Madrid", "Spain", "FALSE")])
    ready, _ = candidates(pd.DataFrame([fixture()]), far, NOW)
    check("a genuinely different meeting is NOT blocked", len(ready) == 1)

    other_comp = results([("2026-09-24", "Spain", "France", 2, 1,
                           "Friendly", "Madrid", "Spain", "FALSE")])
    ready, _ = candidates(pd.DataFrame([fixture()]), other_comp, NOW)
    check("same teams, same day, different competition is not a duplicate",
          len(ready) == 1)


def main() -> int:
    test_happy_path()
    test_guards()
    test_duplicate_guard()
    print(f"\n{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURE(S)'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
