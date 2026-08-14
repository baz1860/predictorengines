#!/usr/bin/env python3
"""P6 — run history and operational readiness.

The gap this fills
------------------
`last_run.json` holds exactly one run, and `monitor.py` answers "did the most
recent run work?". That is the right question for an alert, and the wrong
question for the P6 acceptance gate, which is:

    14 consecutive green daily runs with full league coverage
    and no staleness alerts

You cannot evaluate a streak from a single record. Worse, the failure modes
this project introduced are *slow*: a league quietly stops updating, coverage
erodes as clubs fall below the evidence bar, the fitted-match count drifts
down after a bad merge. Each individual run looks fine. Only the sequence
shows it.

So every run appends one line to data/run_history.jsonl, and readiness() reads
the sequence and reports whether the system has actually been healthy — not
whether it happened to be healthy the one time someone looked.

Design notes
------------
* Append-only JSONL. A corrupt line is skipped, never fatal: a monitor that
  crashes on its own history is worse than no monitor.
* Bounded. Trimmed to MAX_ENTRIES so it cannot grow without limit on a machine
  nobody is watching.
* Failures are recorded too. A ledger of only successes cannot measure a
  streak — the whole point is knowing when the run broke.

CLI:
  python3 -m club_soccer.run_ledger                 # readiness verdict
  python3 -m club_soccer.run_ledger --history 20    # recent runs
  python3 -m club_soccer.run_ledger --json
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNTIME = Path(os.environ.get("CLUB_SOCCER_RUNTIME_DIR", str(DATA)))
LEDGER = RUNTIME / "run_history.jsonl"

MAX_ENTRIES = 400
REQUIRED_GREEN_RUNS = 14          # the P6 acceptance gate
MAX_RUN_GAP_HOURS = 30.0          # a "daily" run that skipped a day breaks the streak


def append(record: dict) -> None:
    """Append one run outcome. Never raises — a ledger failure must not fail
    the pipeline it is only observing."""
    try:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(record)
        entry.setdefault("recorded_at_utc", datetime.now(timezone.utc).isoformat())
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        _trim()
    except Exception as exc:      # pragma: no cover - defensive
        print(f"   run ledger not written ({exc})")


def _trim() -> None:
    try:
        lines = LEDGER.read_text(encoding="utf-8").splitlines()
        if len(lines) > MAX_ENTRIES:
            LEDGER.write_text("\n".join(lines[-MAX_ENTRIES:]) + "\n", encoding="utf-8")
    except Exception:
        pass


def history(limit: int | None = None) -> list[dict]:
    """Recorded runs, oldest first. Corrupt lines are skipped, not fatal."""
    if not LEDGER.exists():
        return []
    out: list[dict] = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out[-limit:] if limit else out


def _parse(value) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else None


def green_streak(entries: list[dict] | None = None) -> tuple[int, str]:
    """Consecutive healthy runs, most recent backwards. Returns (n, reason).

    A gap longer than MAX_RUN_GAP_HOURS breaks the streak even if every
    recorded run passed — "14 green runs" spread over two months is not
    evidence that a daily pipeline is healthy.
    """
    entries = history() if entries is None else entries
    if not entries:
        return 0, "no runs recorded"
    streak = 0
    previous: datetime | None = None
    for entry in reversed(entries):
        if not entry.get("ok"):
            return streak, f"run {entry.get('run_id', '?')[:8]} failed"
        if entry.get("stale_leagues_no_bsd"):
            return streak, (f"run {entry.get('run_id', '?')[:8]} had "
                            f"{len(entry['stale_leagues_no_bsd'])} stale league(s)")
        when = _parse(entry.get("finished_at_utc"))
        if when is None:
            # A record with no valid timestamp cannot be placed in the daily
            # cadence, so it cannot count toward "14 consecutive daily runs".
            # Absence of a timestamp must break the streak, not be skipped —
            # otherwise 14 undated records read as READY.
            return streak, f"run {entry.get('run_id', '?')[:8]} has no valid timestamp"
        if previous is not None:
            gap = (previous - when).total_seconds() / 3600.0
            if gap > MAX_RUN_GAP_HOURS:
                return streak, f"gap of {gap:.0f}h between runs"
        previous = when
        streak += 1
    return streak, "all recorded runs healthy"


def readiness(required: int = REQUIRED_GREEN_RUNS) -> dict:
    """The P6 acceptance gate, evaluated against recorded history."""
    entries = history()
    streak, reason = green_streak(entries)
    latest = entries[-1] if entries else None

    checks: list[dict] = []

    checks.append({
        "check": f"{required} consecutive green runs",
        "pass": streak >= required,
        "detail": f"{streak} green ({reason})",
    })

    if latest is None:
        # No history means UNKNOWN, not healthy. Reporting "no stale leagues"
        # off an empty ledger would be absence of evidence dressed up as
        # evidence of absence — the same mistake as a green gate on a model
        # nobody validated.
        for name in ("latest run succeeded",
                     "no stale leagues without a BSD path",
                     "validation gate passing"):
            checks.append({"check": name, "pass": False,
                           "detail": "unknown — no runs recorded yet"})
        return {"ready": False, "green_streak": streak, "required": required,
                "runs_recorded": 0, "checks": checks}

    checks.append({
        "check": "latest run succeeded",
        "pass": bool(latest.get("ok")),
        "detail": latest.get("crashed") or
        (", ".join(latest.get("failed_required_steps") or []) or "ok"),
    })
    stale = latest.get("stale_leagues_no_bsd") or []
    checks.append({
        "check": "no stale leagues without a BSD path",
        "pass": not stale,
        "detail": ", ".join(stale) if stale else "none",
    })
    checks.append({
        "check": "validation gate passing",
        "pass": bool(latest.get("gate_pass", False)),
        "detail": f"brier {latest.get('gate_brier')} vs limit {latest.get('gate_limit')}",
    })

    # Coverage must not be eroding. Compares the latest run against the median
    # of the previous ten: a slow decline in full-evidence clubs means the
    # expansion is decaying even while every individual run reports green.
    prior = [e.get("teams_full_evidence") for e in entries[:-1][-10:]
             if isinstance(e.get("teams_full_evidence"), int)]
    now = latest.get("teams_full_evidence")
    if prior and isinstance(now, int):
        baseline = sorted(prior)[len(prior) // 2]
        checks.append({
            "check": "full-evidence coverage not eroding",
            "pass": now >= baseline * 0.95,
            "detail": f"{now} now vs {baseline} median of last {len(prior)}",
        })

    return {
        "ready": all(c["pass"] for c in checks),
        "green_streak": streak,
        "required": required,
        "runs_recorded": len(entries),
        "checks": checks,
    }


def snapshot() -> dict:
    """Coverage/health fields to attach to a run record.

    Best-effort: a pipeline must not fail because its observability could not
    be computed.
    """
    out: dict = {}
    try:
        import pandas as pd

        from . import coverage as COV
        from . import model as M
        from . import seed_fdcouk_leagues as SFL

        params = M.load_params()
        summary = COV.summarise(params)
        out["teams_total"] = summary.get("teams")
        out["teams_full_evidence"] = summary.get("by_tier", {}).get("full")
        out["euro_only_teams"] = summary.get("euro_only_teams")
        out["fitted_matches"] = params.get("fitted_matches")
        out["league_seed_active"] = params.get("league_seed_active")

        df = pd.read_csv(SFL.FIXTURES, low_memory=False)
        out["fixtures_rows"] = int(len(df))
        out["identities"] = int(len(set(df["home"].dropna()) | set(df["away"].dropna())))

        # Use the authoritative source-freshness check, not the season
        # heuristic: only a league genuinely BEHIND its source should break the
        # green-run streak. Off-season and pre-season gaps must not, or the
        # streak never starts through a normal summer. Falls back to the
        # offline heuristic if the network check is unavailable.
        try:
            behind = [r for r in SFL.refresh_health() if r.get("behind")]
            out["stale_leagues_no_bsd"] = [r["competition"] for r in behind]
        except Exception:
            stale = [r for r in SFL.staleness()
                     if r["warn"] and not r["has_bsd_path"]]
            out["stale_leagues_no_bsd"] = [r["competition"] for r in stale]
    except Exception as exc:
        out["snapshot_error"] = str(exc)

    # gate_pass must mean what `validate --gate` decided, not a private
    # re-derivation of it.
    #
    # This used to be `float(latest["brier"]) <= limit` — Brier alone. The real
    # gate (validate.gate_failures) additionally pins the evaluation window,
    # the row count and an identity/outcome hash of the exact population, so a
    # model can be scored on a DIFFERENT sample and still look green here. That
    # is precisely what happened: from 2026-08-01 the population hash stopped
    # matching the baseline and `update.sh` exited 1 every day, while this
    # field recorded gate_pass=true because the Brier was unchanged. Two weeks
    # of run_history asserting a gate had passed when it had not.
    #
    # validation_gate_state.json is written by validate --gate itself, so it is
    # the authority. Brier is still reported for context, but it no longer gets
    # to decide.
    try:
        latest = json.loads((DATA / "validation_latest.json").read_text())
        baseline = json.loads((DATA / "promotion_baseline.json").read_text())
        limit = float(baseline["brier"]) + float(baseline.get("gate_tol", 0.01))
        out["gate_brier"] = round(float(latest["brier"]), 5)
        out["gate_limit"] = round(limit, 5)
    except Exception:
        pass

    try:
        state = json.loads((DATA / "validation_gate_state.json").read_text())
        out["gate_pass"] = bool(state.get("passed", False))
        failures = state.get("failures") or []
        if failures:
            out["gate_failures"] = list(failures)[:5]
        out["gate_checked_at_utc"] = state.get("checked_at_utc")
    except Exception:
        # No gate state means the gate has not run, which is not a pass.
        out.setdefault("gate_pass", False)
        out.setdefault("gate_failures", ["validation_gate_state.json missing "
                                         "or unreadable — gate never ran"])

    # Staking-evidence accumulation: how far the decision-time backtest is
    # toward the gate's 1,000-bet bar. Surfaced so the operator can watch the
    # gate approach rather than guess when it might open.
    try:
        bt = json.loads((RUNTIME / "backtest_market.json").read_text())
        sim = bt.get("simulated_betting", {})
        n = max((row.get("n_bets", 0) or 0)
                for m in sim.values() for row in m.values()) if sim else 0
        out["staking_bets_accumulated"] = int(n)
        out["staking_bets_target"] = 1000
    except Exception:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", type=int, metavar="N", help="show the last N runs")
    ap.add_argument("--required", type=int, default=REQUIRED_GREEN_RUNS)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.history:
        entries = history(args.history)
        if not entries:
            print("no runs recorded yet")
            return
        print(f"{'finished':<22}{'ok':>4}{'rows':>8}{'full-ev':>9}{'stale':>7}  notes")
        for e in entries:
            note = e.get("crashed") or ", ".join(e.get("failed_required_steps") or [])
            print(f"  {str(e.get('finished_at_utc', ''))[:19]:<20}"
                  f"{'Y' if e.get('ok') else 'N':>4}"
                  f"{e.get('fixtures_rows', '-'):>8}"
                  f"{e.get('teams_full_evidence', '-'):>9}"
                  f"{len(e.get('stale_leagues_no_bsd') or []):>7}  {note}")
        return

    result = readiness(args.required)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"club_soccer readiness: {'READY' if result['ready'] else 'NOT READY'}")
    print(f"  runs recorded : {result['runs_recorded']}")
    print(f"  green streak  : {result['green_streak']} / {result['required']}")
    print()
    for c in result["checks"]:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}")
        print(f"         {c['detail']}")


if __name__ == "__main__":
    main()
