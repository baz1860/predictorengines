#!/usr/bin/env python3
"""M9 release & ops tests.

Covers:
  * run_checks delegates to pytest markers instead of a manual suite list;
  * daily_summary assembles offline (validation/freshness/CLV/bankroll) without
    crashing and exposes no API-key values;
  * the CLV section degrades to a clear local action when there are no snapshots.

Recommendation counting (which shells out to each engine) is stubbed so the test
stays fast; the live path is exercised by `python3 daily_summary.py`.

Run: python3 test_release.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_checks
import daily_summary
from core import clv_suite

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")
        assert cond, detail or name


def test_run_checks_uses_pytest_markers():
    fast_cmd = run_checks._pytest_cmd("fast")
    gates_cmd = run_checks._pytest_cmd("gates")
    check("fast marker command", fast_cmd[-2:] == ["-m", "fast"], str(fast_cmd))
    check("gates marker command", gates_cmd[-2:] == ["-m", "gates"], str(gates_cmd))
    check("manual suite list removed", not hasattr(run_checks, "ORDERED"))


def test_daily_summary_offline():
    orig = daily_summary._recommended_count
    saved_hist = clv_suite.HISTORY
    try:
        daily_summary._recommended_count = lambda eng: 0   # stub the subprocess shell-out
        with tempfile.TemporaryDirectory() as d:
            clv_suite.HISTORY = Path(d) / "no_history.csv"   # force no-snapshots path
            s = daily_summary.build_summary()
        check("has the top-level sections",
              all(k in s for k in ("generated_at", "bankroll", "engines", "clv")), str(s.keys()))
        check("covers all four engines", len(s["engines"]) == 4, str(list(s["engines"])))
        check("each engine has a gate status",
              all(e["validation"] in ("PASS", "FAIL", "unknown") for e in s["engines"].values()))
        check("clv degrades to a local action",
              "action" in s["clv"] or s["clv"].get("status", "").startswith("no"),
              str(s["clv"]))

        import json
        from app import settings_store
        blob = json.dumps(s)
        keys = settings_store.load().get("odds_api_keys", {}) or {}
        leaked = [k for k in keys.values() if k and str(k) in blob]
        check("summary leaks no API-key values", not leaked, str(leaked))
    finally:
        daily_summary._recommended_count = orig
        clv_suite.HISTORY = saved_hist


def main():
    print("M9 release tests")
    test_run_checks_uses_pytest_markers()
    test_daily_summary_offline()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
