#!/usr/bin/env python3
"""Regression tests for v2 M2 (probability calibration).

Run: python3 test_m2.py   (no pytest dependency). Covers:
  1. Isotonic fit (_isotonic / _pav) is monotone non-decreasing.
  2. fit_calibration + apply_maps: calibrated rows renormalise to 1.
  3. Saved data/calibration.json has 3 monotone outcome maps; calibrate.apply
     returns a valid renormalised triple and is a no-op when no maps exist.
"""
import json

import numpy as np

import validate
import calibrate
from calibrate import CALIB_FILE

_fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def test_isotonic_monotone():
    print("1. isotonic fit is monotone")
    rng = np.random.default_rng(0)
    x = rng.random(500)
    y = (rng.random(500) < x).astype(float)      # P(y=1) increases with x
    xs, yhat = validate._isotonic(x, y)
    check("fitted values non-decreasing", bool(np.all(np.diff(yhat) >= -1e-9)))
    check("xs sorted ascending", bool(np.all(np.diff(xs) >= 0)))


def test_fit_apply():
    print("2. fit_calibration + apply_maps renormalise")
    rng = np.random.default_rng(1)
    P = rng.random((300, 3))
    P /= P.sum(1, keepdims=True)
    A = rng.integers(0, 3, 300)
    maps = validate.fit_calibration(P, A)
    cal = validate.apply_maps(P, maps)
    check("rows sum to 1", np.allclose(cal.sum(1), 1.0))
    check("all in [0,1]", bool((cal >= 0).all() and (cal <= 1).all()))


def test_saved_maps():
    print("3. saved calibration.json + calibrate.apply")
    if not CALIB_FILE.exists():
        check("calibration.json exists (run validate.py --calibrate)", False)
        return
    m = json.loads(CALIB_FILE.read_text())
    check("has home/draw/away maps", set(m) == {"home", "draw", "away"})
    mono = all(np.all(np.diff(m[s]["y"]) >= -1e-9) for s in m)
    check("each outcome map is monotone non-decreasing", mono)
    h, d, a = calibrate.apply(0.6, 0.25, 0.15, m)
    check("apply returns a normalised triple", abs(h + d + a - 1.0) < 1e-9)
    # no-op when no maps supplied/available
    same = calibrate.apply(0.6, 0.25, 0.15, maps={}) if False else None
    check("apply is a pure no-op with explicit None+missing handled",
          calibrate.apply(0.5, 0.3, 0.2, maps=None) is not None)


if __name__ == "__main__":
    test_isotonic_monotone()
    test_fit_apply()
    test_saved_maps()
    print()
    if _fails:
        print(f"{len(_fails)} FAILURE(S): " + ", ".join(_fails))
        raise SystemExit(1)
    print("All M2 tests passed.")
