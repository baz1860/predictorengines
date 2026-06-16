#!/usr/bin/env python3
"""One command to check the whole suite (V3 M9).

Runs every fast offline test suite (contract, security, bankroll, per-milestone,
and the V3 M5–M8 suites), and — with --gates — the per-engine validation gates
via validate_all.py. Prints a compact pass/fail table and exits non-zero if
anything fails, so it works in CI or as a pre-bet sanity check.

    python3 run_checks.py            # fast tests only (~seconds)
    python3 run_checks.py --gates    # also run validation gates (~1 min)
    python3 run_checks.py --gates --sims 4000   # smaller golf sim for the gate
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Explicit order: foundational contracts first, then milestones, then V3 add-ons.
# Any other test_*.py in the repo is appended so new suites are picked up too.
ORDERED = [
    "test_engines_contract", "test_security", "test_bankroll", "test_club_soccer",
    "test_m2", "test_m3", "test_m4", "test_m5", "test_m6", "test_m7",
    "test_market_blend", "test_clv_suite", "test_cfb_blend", "test_provenance",
    "test_model_audit",
]


def _suites() -> list[str]:
    found = sorted(p.stem for p in ROOT.glob("test_*.py"))
    return [s for s in ORDERED if s in found] + [s for s in found if s not in ORDERED]


def _run(cmd: list[str], cwd: Path) -> tuple[bool, float, str]:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return p.returncode == 0, time.time() - t0, (p.stdout + p.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the suite's checks (V3 M9)")
    ap.add_argument("--gates", action="store_true", help="also run validation gates")
    ap.add_argument("--sims", type=int, default=4000, help="golf gate sims (with --gates)")
    args = ap.parse_args()

    print("Running fast test suites…\n")
    results, failures = [], 0
    for suite in _suites():
        ok, secs, out = _run([sys.executable, f"{suite}.py"], ROOT)
        results.append((suite, ok, secs))
        if not ok:
            failures += 1
            print(f"  FAIL  {suite}  ({secs:.1f}s)")
            print("    " + "\n    ".join(out.strip().splitlines()[-6:]))
        else:
            print(f"  PASS  {suite}  ({secs:.1f}s)")

    if args.gates:
        print("\nRunning validation gates (validate_all.py --gate)…")
        ok, secs, out = _run([sys.executable, "validate_all.py", "--gate",
                              "--sims", str(args.sims)], ROOT)
        results.append(("validation_gates", ok, secs))
        if not ok:
            failures += 1
        tail = "\n    ".join(out.strip().splitlines()[-8:])
        print(f"  {'PASS' if ok else 'FAIL'}  validation_gates  ({secs:.1f}s)\n    {tail}")

    total = sum(s for _, _, s in results)
    print(f"\n{len(results) - failures}/{len(results)} checks passed in {total:.1f}s")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
