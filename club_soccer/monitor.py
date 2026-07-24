#!/usr/bin/env python3
"""Run-status monitor: `python3 -m club_soccer.monitor`.

Reads data/last_run.json (written atomically on every season.py exit path)
and exits nonzero unless ALL of:

  - the file exists and parses,
  - finished_at_utc is <= MAX_RUN_AGE_HOURS old,
  - ok is true (no required-step failures, no crash).

Optional freshness warnings (absences/squads/odds-snapshot ages) print but
don't fail the check on their own.

Schedule this daily (cron/launchd) after the season run. To get notified on
failure, set CLUB_SOCCER_NOTIFY_CMD to a shell command; the failure summary
is passed on stdin (e.g. a mail/webhook/osascript one-liner).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAST_RUN = HERE / "data" / "last_run.json"
MAX_RUN_AGE_HOURS = 26.0
# A "running" marker older than this is a dead run. Kept at 2.0h rather than
# guessing a tighter deadline without p99 timing data; the deployment instead
# schedules a SECOND monitor run at 10:00, which catches a 07:30 job killed at
# startup (2.5h old by then) that the 09:00 run (90m) cannot.
MAX_RUNNING_HOURS = 2.0
CLOCK_SKEW_MINUTES = 5.0     # future-dated status beyond this is corrupt
STALE_INPUT_DAYS = 3.0


def _parse_utc(value) -> datetime | None:
    """Timezone-aware parse; naive timestamps are rejected (ambiguous
    provenance must not hold the monitor green)."""
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def check() -> tuple[bool, list[str], list[str]]:
    """(ok, failures, warnings)."""
    failures: list[str] = []
    warnings: list[str] = []
    if not LAST_RUN.exists():
        return False, ["no last_run.json — the daily run has never completed "
                       "(or its status write is broken)"], []
    try:
        status = json.loads(LAST_RUN.read_text())
    except Exception as exc:
        return False, [f"last_run.json unreadable ({exc})"], []

    # A top-level list/str/number must not crash status.get(...) — a malformed
    # status file is a failure, not an exception.
    if not isinstance(status, dict):
        return False, [f"last_run.json top level is {type(status).__name__}, "
                       "not an object"], []

    now = datetime.now(timezone.utc)

    # A run that wrote its "running" marker and never finished (SIGKILL,
    # power loss, hang) must go red after the deadline — the old green
    # status is gone, so this state is what a dead run looks like.
    state = status.get("state")
    if state == "running":
        started = _parse_utc(status.get("started_at_utc"))
        if started is None:
            return False, [f"running marker has invalid started_at_utc: "
                           f"{status.get('started_at_utc')!r}"], []
        run_h = (now - started).total_seconds() / 3600.0
        if run_h < -(CLOCK_SKEW_MINUTES / 60.0):
            # A future-dated running marker would otherwise report a nonsense
            # negative "elapsed" and stay green forever.
            return False, [f"running marker started_at_utc is {-run_h:.1f}h in "
                           "the FUTURE — status provenance corrupt"], []
        if run_h > MAX_RUNNING_HOURS:
            return False, [f"run started {run_h:.1f}h ago and never finished "
                           f"(deadline {MAX_RUNNING_HOURS:g}h) — killed or hung"], []
        return True, [], [f"run in progress ({run_h * 60:.0f}m elapsed, "
                          f"run_id {status.get('run_id')})"]

    # Only "running" and "finished" are legal. An unknown or missing state must
    # fail rather than fall through to the finished-timestamp checks (where a
    # bogus state with a fresh finished_at_utc could hold the monitor green).
    if state not in ("finished", None):
        return False, [f"unknown run state {state!r} in last_run.json"], []

    finished = _parse_utc(status.get("finished_at_utc"))
    if finished is None:
        failures.append(f"finished_at_utc invalid or naive: "
                        f"{status.get('finished_at_utc')!r}")
    else:
        age_h = (now - finished).total_seconds() / 3600.0
        if age_h < -(CLOCK_SKEW_MINUTES / 60.0):
            # A future timestamp is corrupt provenance that would otherwise
            # hold the monitor green forever.
            failures.append(f"finished_at_utc is {-age_h:.1f}h in the FUTURE "
                            "— status provenance corrupt")
        elif age_h > MAX_RUN_AGE_HOURS:
            failures.append(f"last run is {age_h:.1f}h old "
                            f"(limit {MAX_RUN_AGE_HOURS:g}h)")

    if not status.get("ok", False):
        crashed = status.get("crashed")
        steps = status.get("failed_required_steps") or []
        detail = crashed or "; ".join(steps) or "unknown"
        failures.append(f"last run not ok: {detail}")

    for key in ("absences_age_days", "squads_age_days",
                "odds_snapshot_age_days"):
        age = status.get(key)
        if age is None:
            warnings.append(f"{key}: file missing")
        elif age > STALE_INPUT_DAYS:
            warnings.append(f"{key}: {age:.1f}d old (> {STALE_INPUT_DAYS:g}d "
                            "— a swallowed refresh failure looks like a "
                            "quiet day; check the run log)")

    return not failures, failures, warnings


def main() -> None:
    ok, failures, warnings = check()
    print(f"club_soccer monitor: {'OK' if ok else 'FAIL'}")
    for f in failures:
        print(f"  FAIL  {f}")
    for w in warnings:
        print(f"  warn  {w}")

    # Readiness is reported, never allowed to fail the monitor. This check
    # answers a different question — "has this been healthy for 14 days?" —
    # and a young ledger is not an incident. Escalating a not-yet-ready system
    # to an alert would train the operator to ignore the alert.
    try:
        from . import run_ledger as RL
        result = RL.readiness()
        print(f"  readiness: {'READY' if result['ready'] else 'not ready'} "
              f"({result['green_streak']}/{result['required']} green runs, "
              f"{result['runs_recorded']} recorded)")
        for c in result["checks"]:
            if not c["pass"]:
                print(f"    - {c['check']}: {c['detail']}")
    except Exception as exc:
        print(f"  readiness unavailable ({exc})")
    if not ok:
        cmd = os.environ.get("CLUB_SOCCER_NOTIFY_CMD")
        if cmd:
            try:
                subprocess.run(cmd, shell=True, input="\n".join(failures),
                               text=True, timeout=30)
            except Exception as exc:
                print(f"  notify command failed ({exc})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
