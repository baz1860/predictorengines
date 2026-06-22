#!/usr/bin/env python3
"""Offline smoke test for the weekly golf narrative report."""

from __future__ import annotations

import tempfile
from pathlib import Path

from golf.weekly_report import generate_report

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_weekly_report_from_current_outputs():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "weekly_report.md"
        result = generate_report(top=5, edge_top=3, threeball_top=3, output=out)
        text = out.read_text()
        check("writes report", out.exists(), str(result))
        check("includes executive section", "## Executive View" in text, text[:200])
        check("includes forecast table", "## Winner And Placement Forecast" in text, text[:300])
        check("reports row counts", result["predictions"] >= 1, str(result))


def main():
    print("Golf weekly-report tests")
    test_weekly_report_from_current_outputs()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
