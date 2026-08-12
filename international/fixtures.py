"""Fixture-table invariants and duplicate reconciliation.

Two jobs:

  `find_duplicates(df)`  -- report candidate duplicate pairs, classified.
  `assert_invariants(df)` -- the gate hook. Raises if the table is unsound.

Invariants enforced (plan §7.1):
  I1  No two rows describe the same real match (same signature, within the date
      tolerance, classified SAME_MATCH).
  I2  No fixture is simultaneously scheduled (blank score) and played (scored).
      I1 subsumes most of this, but I2 is stated separately because it is the
      symptom that reaches the model: `load_matches()` treats any blank-score row
      as an upcoming fixture.
  I3  No blank-score row is older than STALE_AFTER_DAYS, unless it is an
      unresolved placeholder (blank team names) — those are deliberate, and
      `engines/worldcup/predictor.py` filters them before prediction.

I3 is the check that would have caught the July 2026 rows on the day they
appeared. It is a backstop for I1, not a substitute: I1 removes the cause, I3
catches whatever the cause misses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import timeutil
from .identity import (AMBIGUOUS, DATE_TOLERANCE_DAYS, DISTINCT_MATCHES,
                       SAME_MATCH, classify_pair, signature)

ROOT = Path(__file__).resolve().parents[1]
EXCEPTIONS_CSV = ROOT / "data" / "international" / "fixture_exceptions.csv"

STALE_AFTER_DAYS = 3
UNRESOLVED_TOKENS = {"", "na", "nan", "none", "tbd", "to be decided"}

# Adjudication decisions recorded in EXCEPTIONS_CSV.
ACCEPTED_DUPLICATE = "accepted_duplicate"   # confirmed duplicate, safe to drop
DISTINCT_CONFIRMED = "distinct_confirmed"   # confirmed two real matches, keep both
PENDING = "pending_review"                  # seen, not yet adjudicated

# Only this reason is safe to reconcile without a human. A blank row plus a scored
# row for the same fixture is unambiguous: nothing is lost by dropping the blank.
# "Both scored with identical scores" is *probably* a duplicate but dropping one
# deletes a historical result and shifts every subsequent Elo rating, so it goes
# to a human instead.
AUTO_RESOLVABLE_REASONS = ("one scored, one blank",)


class FixtureIntegrityError(RuntimeError):
    """A fixture table violates an invariant."""


@dataclass(frozen=True)
class DuplicatePair:
    idx_a: int
    idx_b: int
    outcome: str
    reason: str
    date_a: str
    date_b: str
    home: str
    away: str
    competition: str

    def as_row(self) -> dict:
        return {"idx_a": self.idx_a, "idx_b": self.idx_b, "outcome": self.outcome,
                "reason": self.reason, "date_a": self.date_a, "date_b": self.date_b,
                "home": self.home, "away": self.away, "competition": self.competition}


def _is_unresolved(value: object) -> bool:
    return str(value or "").strip().casefold() in UNRESOLVED_TOKENS


def _naive_now(asof: object = None) -> pd.Timestamp:
    """A tz-naive 'now'.

    `results.csv` dates are naive, while `pd.Timestamp.utcnow()` is tz-aware on
    current pandas, so comparing them raises. Normalising here keeps the callers
    free of timezone bookkeeping — and note the irony that a timezone mismatch is
    the defect this module exists to fix.
    """
    return timeutil.naive_utc(asof).normalize()


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ("home_score", "away_score"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    out["_sig"] = [signature(h, a, t) for h, a, t
                   in zip(out["home_team"], out["away_team"], out["tournament"])]
    return out


def find_duplicates(df: pd.DataFrame,
                    tolerance_days: int = DATE_TOLERANCE_DAYS,
                    include_distinct: bool = False) -> list[DuplicatePair]:
    """Candidate duplicate pairs, classified by identity.classify_pair."""
    work = _prepare(df)
    # Placeholder rows have no teams; they cannot be reconciled and are excluded.
    work = work[~(work.home_team.map(_is_unresolved)
                  | work.away_team.map(_is_unresolved))]

    pairs: list[DuplicatePair] = []
    for sig, grp in work.groupby("_sig", sort=False):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("date")
        rows = list(grp.iterrows())
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                ia, a = rows[i]
                ib, b = rows[j]
                if abs((a["date"] - b["date"]).days) > tolerance_days:
                    continue
                v = classify_pair(a, b, tolerance_days)
                if v.outcome == DISTINCT_MATCHES and not include_distinct:
                    continue
                pairs.append(DuplicatePair(
                    int(ia), int(ib), v.outcome, v.reason,
                    str(a["date"].date()), str(b["date"].date()),
                    str(a["home_team"]), str(a["away_team"]), str(a["tournament"])))
    return pairs


def stale_blanks(df: pd.DataFrame, asof: object = None,
                 stale_after_days: int = STALE_AFTER_DAYS) -> pd.DataFrame:
    """Blank-score rows old enough that they cannot still be upcoming."""
    work = _prepare(df)
    now = _naive_now(asof)
    blank = work.home_score.isna() | work.away_score.isna()
    resolved = ~(work.home_team.map(_is_unresolved) | work.away_team.map(_is_unresolved))
    old = work.date < (now - pd.Timedelta(days=stale_after_days))
    return work[blank & resolved & old].drop(columns=["_sig"])


def _pair_id(p: DuplicatePair) -> str:
    lo, hi = sorted((p.date_a, p.date_b))
    return f"{signature(p.home, p.away, p.competition)}|{lo}|{hi}"


def load_exceptions(path: Path | None = None) -> dict[str, str]:
    """Adjudicated duplicate pairs -> decision. Missing file means none."""
    path = path or EXCEPTIONS_CSV
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return dict(zip(df.pair_id, df.decision))


def assert_invariants(df: pd.DataFrame, asof: object = None,
                      check_stale: bool = True,
                      exceptions: dict[str, str] | None = None,
                      allow_pending: bool = True) -> None:
    """Gate hook. Raise FixtureIntegrityError on any violation.

    Pairs recorded in data/international/fixture_exceptions.csv are treated as
    already seen. A NEW duplicate always fails the gate — that is the point: it
    means the ingest has started generating duplicates again. Pairs marked
    `pending_review` are tolerated when `allow_pending=True` (the daily gate) and
    rejected when False (the release gate), so a backlog cannot be ignored forever.
    """
    problems: list[str] = []
    known = load_exceptions() if exceptions is None else exceptions

    dupes = find_duplicates(df)
    new, pending = [], []
    for p in dupes:
        if p.outcome not in (SAME_MATCH, AMBIGUOUS):
            continue
        decision = known.get(_pair_id(p))
        if decision in (ACCEPTED_DUPLICATE, DISTINCT_CONFIRMED):
            continue
        (pending if decision == PENDING else new).append(p)

    if new:
        head = "; ".join(f"{p.home} v {p.away} ({p.date_a}/{p.date_b}): {p.reason}"
                         for p in new[:5])
        problems.append(f"I1: {len(new)} UNRECORDED duplicate/ambiguous pair(s): {head}"
                        f"{' …' if len(new) > 5 else ''}. Adjudicate them into "
                        f"{EXCEPTIONS_CSV.name} or fix the ingest.")
    if pending and not allow_pending:
        head = "; ".join(f"{p.home} v {p.away} ({p.date_a}/{p.date_b})" for p in pending[:5])
        problems.append(f"I2: {len(pending)} pair(s) still pending review: {head}")

    if check_stale:
        stale = stale_blanks(df, asof=asof)
        if not stale.empty:
            head = "; ".join(
                f"{r.home_team} v {r.away_team} ({str(r.date)[:10]})"
                for r in stale.head(5).itertuples(index=False))
            problems.append(f"I3: {len(stale)} blank-score row(s) past kickoff: {head}"
                            f"{' …' if len(stale) > 5 else ''}")

    if problems:
        raise FixtureIntegrityError(
            "fixture table failed integrity checks:\n  - " + "\n  - ".join(problems))


def reconcile(df: pd.DataFrame,
              tolerance_days: int = DATE_TOLERANCE_DAYS,
              exceptions: dict[str, str] | None = None,
              ) -> tuple[pd.DataFrame, list[DuplicatePair]]:
    """Drop rows that duplicate another row describing the same match.

    **Conservative by design.** Only pairs whose reason is in
    AUTO_RESOLVABLE_REASONS (one row scored, one blank) are dropped automatically:
    the blank row is redundant and nothing is lost. Pairs where BOTH rows carry a
    score are returned for adjudication and both rows kept, because dropping one
    deletes a historical result and shifts every Elo rating computed after it.

    Pairs explicitly marked ACCEPTED_DUPLICATE in the exceptions ledger ARE
    dropped, since a human has signed off.
    """
    work = _prepare(df)
    known = load_exceptions() if exceptions is None else exceptions
    pairs = find_duplicates(df, tolerance_days)
    drop: set[int] = set()
    unresolved: list[DuplicatePair] = []

    for p in pairs:
        if p.outcome == DISTINCT_MATCHES:
            continue
        decision = known.get(_pair_id(p))
        if decision == DISTINCT_CONFIRMED:
            continue
        auto = any(r in p.reason for r in AUTO_RESOLVABLE_REASONS)
        if not (auto or decision == ACCEPTED_DUPLICATE):
            unresolved.append(p)
            continue
        a, b = work.loc[p.idx_a], work.loc[p.idx_b]
        a_scored = pd.notna(a["home_score"]) and pd.notna(a["away_score"])
        b_scored = pd.notna(b["home_score"]) and pd.notna(b["away_score"])
        if a_scored and not b_scored:
            drop.add(p.idx_b)
        elif b_scored and not a_scored:
            drop.add(p.idx_a)
        else:
            drop.add(p.idx_b if a["date"] <= b["date"] else p.idx_a)

    kept = df.drop(index=[i for i in drop if i in df.index])
    return kept, unresolved
