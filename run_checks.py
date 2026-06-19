#!/usr/bin/env python3
"""One command to check the suite through pytest markers.

Runs the fast offline pytest suite (`-m fast`) and, with `--gates`, the per-engine
validation gate pytest suite (`-m gates`). Prints a compact pass/fail table and
exits non-zero if anything fails, so it works in CI or as a pre-bet sanity check.

    python3 run_checks.py            # fast tests only (~seconds)
    python3 run_checks.py --gates    # also run validation gates (~1 min)
    python3 run_checks.py --gates --sims 4000   # smaller golf sim for the gate
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _pytest_cmd(marker: str) -> list[str]:
    return [sys.executable, "-m", "pytest", "-m", marker]


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[bool, float, str]:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    return p.returncode == 0, time.time() - t0, (p.stdout + p.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the suite's checks (V3 M9)")
    ap.add_argument("--gates", action="store_true", help="also run validation gates")
    ap.add_argument("--sims", type=int, default=4000, help="golf gate sims (with --gates)")
    args = ap.parse_args()

    print("Running fast pytest suite…\n")
    results, failures = [], 0
    ok, secs, out = _run(_pytest_cmd("fast"), ROOT)
    results.append(("pytest_fast", ok, secs))
    if not ok:
        failures += 1
        print(f"  FAIL  pytest_fast  ({secs:.1f}s)")
        print("    " + "\n    ".join(out.strip().splitlines()[-12:]))
    else:
        print(f"  PASS  pytest_fast  ({secs:.1f}s)")

    if args.gates:
        print("\nRunning validation gates via pytest…")
        env = {**os.environ, "VALIDATION_SIMS": str(args.sims)}
        ok, secs, out = _run(_pytest_cmd("gates"), ROOT, env=env)
        results.append(("pytest_gates", ok, secs))
        if not ok:
            failures += 1
        tail = "\n    ".join(out.strip().splitlines()[-12:])
        print(f"  {'PASS' if ok else 'FAIL'}  pytest_gates  ({secs:.1f}s)\n    {tail}")

    total = sum(s for _, _, s in results)
    print(f"\n{len(results) - failures}/{len(results)} checks passed in {total:.1f}s")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
