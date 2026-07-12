#!/usr/bin/env python3
"""Apply 1X2 probability calibration for the Club Soccer engine.

Loads the gated probability transform fitted by `validate.py --calibrate`
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


def _unwrap(payload):
    """Support legacy isotonic maps and the newer gated temperature payload."""
    if isinstance(payload, dict) and payload.get("method") == "temperature":
        return ({"method": "temperature",
                 "temperature": float(payload.get("temperature", 1.0))},
                bool(payload.get("active", False)))
    if isinstance(payload, dict) and isinstance(payload.get("maps"), dict):
        return payload.get("maps"), bool(payload.get("active", False))
    return payload, False


def load_maps(active_only: bool = False):
    """Stored calibration maps, or None if not yet fitted.

    ``active_only`` is used by production card/pricing paths. Calibration is
    deliberately opt-in because a map can improve Brier while worsening
    log-loss, and a raw legacy file has no promotion decision attached.
    """
    if CALIB_FILE.exists():
        try:
            maps, active = _unwrap(json.loads(CALIB_FILE.read_text()))
            if active_only and not active:
                return None
            return maps
        except Exception:
            return None
    return None


def load_active_maps():
    """Return maps only when an explicit ``active: true`` gate is stored."""
    return load_maps(active_only=True)


def apply(p_home, p_draw, p_away, maps=None):
    """Calibrate a 1X2 triple and renormalise.

    Supports both the legacy isotonic-map shape and a temperature transform.
    Loads the stored transform when omitted; returns inputs unchanged when no
    calibration is available.
    """
    if maps is None:
        maps = load_maps()
    if maps is None:
        return p_home, p_draw, p_away
    if maps.get("method") == "temperature":
        temperature = max(0.05, float(maps.get("temperature", 1.0)))
        p = np.clip(np.asarray([p_home, p_draw, p_away], dtype=float), 1e-12, 1.0)
        logits = np.log(p)
        logits -= logits.max()
        q = np.exp(logits / temperature)
        q /= q.sum()
        return float(q[0]), float(q[1]), float(q[2])
    cal = []
    for side, p in (("home", p_home), ("draw", p_draw), ("away", p_away)):
        m = maps[side]
        cal.append(float(np.interp(p, m["x"], m["y"])))
    s = sum(cal)
    if s <= 0:
        return p_home, p_draw, p_away
    return cal[0] / s, cal[1] / s, cal[2] / s


def temperature_probs(P, temperature: float) -> np.ndarray:
    """Apply multiclass temperature scaling to an n x 3 probability array."""
    p = np.clip(np.asarray(P, dtype=float), 1e-12, 1.0)
    t = max(0.05, float(temperature))
    logits = np.log(p)
    logits -= logits.max(axis=1, keepdims=True)
    q = np.exp(logits / t)
    return q / q.sum(axis=1, keepdims=True)


def fit_temperature(P, A, low: float = 0.50, high: float = 1.50,
                    step: float = 0.001) -> float:
    """Fit a single multiclass temperature by minimising training log-loss.

    The one-dimensional grid is deterministic, dependency-free and fine
    enough for the small calibration correction used here.
    """
    P = np.asarray(P, dtype=float)
    A = np.asarray(A, dtype=int)
    if len(A) == 0:
        return 1.0
    grid = np.arange(low, high + step / 2, step)
    losses = []
    for t in grid:
        q = temperature_probs(P, float(t))
        losses.append(float(-np.log(np.clip(q[np.arange(len(A)), A], 1e-12, 1.0)).mean()))
    return float(grid[int(np.argmin(losses))])


if __name__ == "__main__":
    m = load_maps()
    if m is None:
        print("No calibration yet. Fit it: python3 validate.py --calibrate")
    else:
        print(f"Calibration loaded ({CALIB_FILE.name}); outcomes: {list(m)}")
        for side in ("home", "draw", "away"):
            print(f"  {side}: {len(m[side]['x'])} knots")
