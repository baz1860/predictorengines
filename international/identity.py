"""Canonical fixture identity that survives UTC/local date disagreement.

The defect this fixes
---------------------
On 8 August 2026 `data/results.csv` contained BOTH of these, for each of two
matches:

    2026-07-06,Argentina,Egypt,,,FIFA World Cup,Atlanta,United States,TRUE
    2026-07-07,Argentina,Egypt,3,2,FIFA World Cup,Atlanta,United States,TRUE

Same teams, same competition, same city, one day apart. These are North American
evening kick-offs: **local date 6 July, UTC date 7 July**. One source dated the
fixture locally, the other in UTC.

`merge_results.py` keys on `(date, home_team, away_team)`. Because the dates
differ, the local row is classified "local-only" and appended, so both survive.
The scoreless row then satisfies `load_matches()`'s test for an upcoming fixture
(`home_score` is blank, with no date or status check), so a match played a month
ago is still presented as a forthcoming fixture.

Diagnosing this as "stale data" and adding a date-based staleness check would
have hidden the symptom and left the duplicate generator running. The fix has to
be at the identity layer.

Approach
--------
`signature()` is a **date-free** identity: teams + competition. Two rows sharing a
signature within `DATE_TOLERANCE_DAYS` are candidates for being the same match.
That is deliberately a *candidate* relation, not an equality: two teams can
legitimately play twice in short succession (two-legged ties, tournament group
then knockout). `classify_pair()` separates the cases, and anything ambiguous is
reported for adjudication rather than silently merged — the same fail-closed
stance as the team registry.

When a provider supplies a stable event ID or a real kick-off timestamp, that
wins over any of this heuristic machinery; `canonical_id()` prefers them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

from . import timeutil

DATE_TOLERANCE_DAYS = 1

# Outcomes of comparing two rows that share a signature.
SAME_MATCH = "same_match"            # one real fixture recorded twice
DISTINCT_MATCHES = "distinct"        # genuinely two fixtures (e.g. two-legged tie)
AMBIGUOUS = "ambiguous"              # needs a human; never auto-merged


def norm_team(name: object) -> str:
    return str(name or "").strip()


def signature(home: object, away: object, competition: object) -> str:
    """Date-free fixture identity. Order-sensitive: home/away is meaningful."""
    parts = [norm_team(home).casefold(),
             norm_team(away).casefold(),
             str(competition or "").strip().casefold()]
    return "|".join(parts)


def canonical_id(home: object, away: object, competition: object,
                 kickoff_utc: object = None, provider_event_id: object = None,
                 date: object = None) -> str:
    """Stable ID for a fixture, best available basis first.

    provider event ID > kick-off timestamp > date + signature.
    The last form is the weak one and is exactly what the ±1 day reconciliation
    exists to compensate for.
    """
    if provider_event_id not in (None, "", float("nan")):
        basis = f"pid:{provider_event_id}"
    elif kickoff_utc not in (None, ""):
        ts = timeutil.to_utc(kickoff_utc)
        basis = f"utc:{ts.strftime('%Y%m%dT%H%M')}:" \
                f"{signature(home, away, competition)}"
    else:
        d = pd.Timestamp(date).strftime("%Y%m%d") if date is not None else "unknown"
        basis = f"date:{d}:{signature(home, away, competition)}"
    return hashlib.sha1(basis.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PairVerdict:
    outcome: str
    reason: str


def _has_score(row: pd.Series) -> bool:
    return pd.notna(row.get("home_score")) and pd.notna(row.get("away_score"))


def classify_pair(a: pd.Series, b: pd.Series,
                  tolerance_days: int = DATE_TOLERANCE_DAYS) -> PairVerdict:
    """Are these two rows the same real-world match?

    Rules, in order:
      1. Dates further apart than the tolerance -> distinct fixtures.
      2. Both scored, and the scores differ    -> distinct (a genuine double-header
         or two-legged tie; merging would destroy a result).
      3. Both scored with identical scores     -> same match, duplicated.
      4. Exactly one scored                    -> same match: the scoreless row is
         the pre-match record of the fixture the other row reports the result for.
         This is the Argentina-Egypt case.
      5. Neither scored                        -> same match if the venue agrees,
         otherwise ambiguous.
    """
    da, db = pd.Timestamp(a["date"]), pd.Timestamp(b["date"])
    gap = abs((da - db).days)
    if gap > tolerance_days:
        return PairVerdict(DISTINCT_MATCHES, f"{gap} days apart")

    sa, sb = _has_score(a), _has_score(b)
    if sa and sb:
        if (a["home_score"], a["away_score"]) == (b["home_score"], b["away_score"]):
            return PairVerdict(SAME_MATCH, "identical scores within tolerance")
        return PairVerdict(DISTINCT_MATCHES,
                           "both scored with different results — two real matches")
    if sa != sb:
        city_a, city_b = str(a.get("city", "")), str(b.get("city", ""))
        if city_a and city_b and city_a != city_b:
            return PairVerdict(AMBIGUOUS,
                               f"one scored, venues differ ({city_a} vs {city_b})")
        return PairVerdict(SAME_MATCH,
                           "one scored, one blank, same venue within tolerance "
                           "(UTC/local date split)")

    city_a, city_b = str(a.get("city", "")), str(b.get("city", ""))
    if city_a and city_b and city_a == city_b:
        return PairVerdict(SAME_MATCH, "both blank, same venue within tolerance")
    return PairVerdict(AMBIGUOUS, "both blank, venue unconfirmed")
