#!/usr/bin/env python3
"""Apply 1X2 probability calibration for the Club Soccer engine.

Loads the isotonic per-outcome maps fitted by `validate.py --calibrate`
(data/calibration.json) and applies them to a single match's home/draw/away
probabilities, renormalising to sum 1. Mirrors the World Cup calibrate.py.

This module NEVER fits — it only applies, so edge.py stays a pure consumer.

  from calibrate import apply
  p_home, p_draw, p_away = apply(p_home, p_draw, p_away)
"""
import json
from pathlib import Path

import numpy as np

CALIB_FILE = Path(__file__).resolve().parent / "data" / "calibration.json"


def load_maps():
    """Stored calibration maps, or None if not yet fitted."""
    if CALIB_FILE.exists():
        return json.loads(CALIB_FILE.read_text())
    return None


def apply(p_home, p_draw, p_away, maps=None):
    """Calibrate a 1X2 triple and renormalise. Loads maps from disk if not
    given; returns the inputs unchanged when no calibration is available."""
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
