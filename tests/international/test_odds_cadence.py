"""Odds polling cadence.

The cadence is calibrated against measured BSD behaviour (World Cup 2026 raw
payloads, 557 observations): 100% priced inside 24h, 96% at 1-3 days, 21% at
3-7 days, median first price 60h out.

Two failure modes this pins:
  * polling too sparsely near kickoff, which means no usable closing price — and
    closing-line value is the only honest measure of edge we will ever compute;
  * treating "never polled" as "not due", which silently disables the poller
    entirely. That is not hypothetical: `Series.map` turns a returned None into
    NaN, `NaN >= interval` is False, and the first version of this scheduled
    exactly zero fixtures.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.international.fetch_odds import (POLL_HORIZON_DAYS,  # noqa: E402
                                              poll_due, poll_interval)

FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


def test_interval_tightens_towards_kickoff() -> None:
    print("\ncadence shape")
    ladder = [poll_interval(h) for h in (3, 12, 48, 120, 600)]
    check("interval is monotonically non-decreasing with time to kickoff",
          all(a <= b for a, b in zip(ladder, ladder[1:])), str(ladder))
    check("closing window sampled sub-hourly", poll_interval(3) <= 0.5)
    check("inside 24h sampled at least hourly", poll_interval(12) <= 1.0)
    check("1-3 days sampled several times a day", poll_interval(48) <= 4.0)
    check("beyond the horizon is never polled",
          poll_interval(24 * (POLL_HORIZON_DAYS + 10)) == float("inf"))

    # The measured facts the ladder is built on.
    check("3-7d band polled more often than the discovery band",
          poll_interval(120) < poll_interval(600))
    check("median first-price lead time (60h) falls in a densely polled band",
          poll_interval(60) <= 4.0, str(poll_interval(60)))


def test_due_logic() -> None:
    print("\ndue logic")
    check("a never-polled fixture is due", poll_due(100, None))
    check("NaN 'since' is treated as never polled, not as not-due",
          poll_due(100, float("nan")))
    check("a just-polled fixture is not due", not poll_due(100, 0.1))
    check("a fixture polled longer ago than its interval is due",
          poll_due(100, poll_interval(100) + 0.1))

    check("a past fixture is never due", not poll_due(-5, None))
    check("a NaN kickoff is never due", not poll_due(float("nan"), None))
    check("beyond the horizon is not due",
          not poll_due(24 * (POLL_HORIZON_DAYS + 1), None))

    check("inside 6h, a 40-minute-old poll is due", poll_due(3, 0.7))
    check("inside 6h, a 10-minute-old poll is not", not poll_due(3, 0.17))
    check("at 5 days, a 24h-old poll is due", poll_due(120, 24))
    check("at 5 days, a 2h-old poll is not", not poll_due(120, 2))
    check("at 40 days, a 12h-old poll is not due", not poll_due(24 * 40, 12))
    check("at 40 days, a 3-day-old poll is due", poll_due(24 * 40, 72))


def main() -> int:
    test_interval_tightens_towards_kickoff()
    test_due_logic()
    print(f"\n{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURE(S)'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
