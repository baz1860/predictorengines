#!/usr/bin/env python3
"""NFL margin/total PMF sanity-anchor tests (Phase 2 gate).

Run: python3 test_nfl_pmf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfl import margin_dist as MD


def main() -> int:
    print("NFL margin/total PMF tests")
    ok, bad = MD.selftest()
    print(f"\n{ok} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
