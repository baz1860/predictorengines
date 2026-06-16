#!/usr/bin/env python3
"""M6 CFB tunable blend-weight tests.

Covers the gated, default-OFF blend-weight upgrade:
  * default weight is 0.50 (V2 behaviour) when no weight file is opted into;
  * a written weight file is loaded and clamped to [0, 1];
  * blend_predict honours w_elo extremes (pure Elo / pure power);
  * choose_weight minimises ML Brier WITHOUT regressing the 0.5-blend margin MAE
    (the conservative constraint) and is a pure function of the walk frame.

Runs without touching the real cfb/data/blend_weight.json.

Run: python3 test_cfb_blend.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CFB = ROOT / "cfb"
if str(CFB) not in sys.path:
    sys.path.insert(0, str(CFB))

import predictor as PR
import validate as V

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_default_weight():
    saved = PR._BLEND_WEIGHT_FILE
    try:
        PR._BLEND_WEIGHT_FILE = str(ROOT / "does_not_exist_blend.json")
        check("default weight is 0.50", PR.load_blend_weight() == 0.50,
              str(PR.load_blend_weight()))
    finally:
        PR._BLEND_WEIGHT_FILE = saved


def test_weight_file_loaded_and_clamped():
    saved = PR._BLEND_WEIGHT_FILE
    try:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "blend_weight.json"
            f.write_text('{"w_elo": 0.6}')
            PR._BLEND_WEIGHT_FILE = str(f)
            check("loads written weight 0.6", abs(PR.load_blend_weight() - 0.6) < 1e-9)
            f.write_text('{"w_elo": 5.0}')
            check("clamps weight to 1.0", PR.load_blend_weight() == 1.0)
    finally:
        PR._BLEND_WEIGHT_FILE = saved


def test_blend_predict_extremes():
    # stub the two base models so we test only the blend math (no data load)
    se, sp = PR.E.predict, PR.P.predict
    try:
        PR.E.predict = lambda *a, **k: {"p1": 0.80, "margin": 14.0, "total": 0.0}
        PR.P.predict = lambda *a, **k: {"p1": 0.60, "margin": 6.0, "total": 55.0}
        ep = (None, None, None, None)
        pure_elo = PR.blend_predict(ep, None, "A", "B", w_elo=1.0)
        pure_pow = PR.blend_predict(ep, None, "A", "B", w_elo=0.0)
        mid = PR.blend_predict(ep, None, "A", "B", w_elo=0.5)
        check("w_elo=1 → pure Elo p1", abs(pure_elo["p1"] - 0.80) < 1e-9)
        check("w_elo=0 → pure power p1", abs(pure_pow["p1"] - 0.60) < 1e-9)
        check("w_elo=0.5 → midpoint p1", abs(mid["p1"] - 0.70) < 1e-9, str(mid["p1"]))
        check("total always from power", abs(mid["total"] - 55.0) < 1e-9)
    finally:
        PR.E.predict, PR.P.predict = se, sp


def test_choose_weight_constrained():
    # synthetic frame: Elo a bit better-calibrated on win prob; both fine on margin
    rng = np.random.default_rng(0)
    n = 4000
    margin = rng.normal(3.0, 16.0, n)
    res = (margin > 0).astype(float)
    # p_pow biased toward 0.5 (under-confident), p_elo sharper toward truth
    base = 1.0 / (1.0 + np.exp(-margin / 12.0))
    p_elo = np.clip(base, 1e-6, 1 - 1e-6)
    p_pow = np.clip(0.5 + 0.5 * (base - 0.5), 1e-6, 1 - 1e-6)
    df = pd.DataFrame({"p_elo": p_elo, "p_pow": p_pow,
                       "m_elo": margin + rng.normal(0, 9, n),
                       "m_pow": margin + rng.normal(0, 9, n),
                       "margin": margin})
    out = V.choose_weight(df)
    check("chosen weight favours the sharper model (>=0.5)", out["chosen"] >= 0.5,
          str(out["chosen"]))
    check("chosen brier <= baseline brier",
          out["chosen_brier"] <= out["baseline_brier"] + 1e-9,
          f"{out['chosen_brier']} vs {out['baseline_brier']}")
    base_mae = out["baseline_margin_mae"]
    check("chosen never regresses margin MAE", out["chosen_margin_mae"] <= base_mae + 1e-9,
          f"{out['chosen_margin_mae']} vs {base_mae}")
    check("every feasible row respects the margin constraint",
          all(r["margin_mae"] <= base_mae + 1e-9 for r in out["table"] if r["ok"]))


def main():
    print("M6 CFB blend-weight tests")
    for fn in (test_default_weight, test_weight_file_loaded_and_clamped,
               test_blend_predict_extremes, test_choose_weight_constrained):
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
