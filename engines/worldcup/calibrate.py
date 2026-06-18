#!/usr/bin/env python3
"""Apply probability calibration (v2 M2).

Loads the isotonic per-outcome maps fitted by validate.py (data/calibration.json)
and applies them to a single match's 1X2 probabilities, renormalising to sum 1.

Calibration is REFIT ONLY by `validate.py --calibrate` — this module never fits,
it only applies, so edge.py stays a pure consumer of the stored maps.

  from calibrate import apply
  p_home, p_draw, p_away = apply(p_home, p_draw, p_away)
"""
import json
from pathlib import Path

import numpy as np

CALIB_FILE = Path(__file__).resolve().parents[2] / "data" / "calibration.json"


def load_maps():
    """Stored calibration maps, or None if not yet fitted."""
    if CALIB_FILE.exists():
        return json.loads(CALIB_FILE.read_text())
    return None


def apply(p_home, p_draw, p_away, maps=None):
    """Calibrate a 1X2 triple and renormalise. If maps is None it is loaded from
    disk; returns the inputs unchanged when no calibration is available."""
    if maps is None:
        maps = load_maps()
    if maps is None:
        return p_home, p_draw, p_away
    cal = []
    for side, p in (("home", p_home), ("draw", p_draw), ("away", p_away)):
        m = maps[side]
        cal.append(float(np.interp(p, m["x"], m["y"])))
    s = sum(cal)
    if s <= 0:
        return p_home, p_draw, p_away
    return cal[0] / s, cal[1] / s, cal[2] / s


if __name__ == "__main__":
    m = load_maps()
    if m is None:
        print("No calibration yet. Fit it: python3 validate.py --calibrate")
    else:
        print(f"Calibration loaded ({CALIB_FILE.name}); outcomes: {list(m)}")
        for side in ("home", "draw", "away"):
            print(f"  {side}: {len(m[side]['x'])} knots")
