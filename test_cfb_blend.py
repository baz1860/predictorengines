#!/usr/bin/env python3
"""M6 CFB tunable blend-weight tests.

Covers the gated, default-OFF blend-weight/model-stack upgrade:
  * default weight is 0.50 (V2 behaviour) when no weight file is opted into;
  * a written weight file is loaded and clamped to [0, 1];
  * blend_predict honours w_elo extremes (pure Elo / pure power);
  * EPA/PPA stack weights remain backwards-compatible and default-off;
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


def test_stack_weight_backwards_compatibility():
    saved = PR._BLEND_WEIGHT_FILE
    try:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "blend_weight.json"
            PR._BLEND_WEIGHT_FILE = str(f)
            f.write_text('{"w_elo": 0.65}')
            w = PR.load_blend_weights()
            check("old w_elo file keeps EPA off", abs(w["epa"]) < 1e-9, str(w))
            check("old w_elo file infers power weight", abs(w["power"] - 0.35) < 1e-9, str(w))
            f.write_text('{"weights": {"elo": 2, "power": 1, "epa": 1}}')
            w = PR.load_blend_weights()
            check("new stack weights normalise", abs(sum(w.values()) - 1.0) < 1e-9, str(w))
            check("new stack weights include EPA", abs(w["epa"] - 0.25) < 1e-9, str(w))
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


def test_blend3_predict_uses_epa_only_when_requested():
    se, sp, sx = PR.E.predict, PR.P.predict, PR.X.predict
    try:
        PR.E.predict = lambda *a, **k: {"p1": 0.80, "margin": 14.0, "total": 0.0}
        PR.P.predict = lambda *a, **k: {"p1": 0.60, "margin": 6.0, "total": 55.0}
        PR.X.predict = lambda *a, **k: {"p1": 0.40, "margin": -2.0, "total": 49.0}
        ep = (None, None, None, None)
        out = PR.blend_predict(ep, None, "A", "B", model="blend3",
                               xparams={"teams": {"A": {}, "B": {}}},
                               weights={"elo": 0.25, "power": 0.25, "epa": 0.50})
        check("blend3 combines EPA win prob",
              abs(out["p1"] - (0.25 * 0.80 + 0.25 * 0.60 + 0.50 * 0.40)) < 1e-9,
              str(out["p1"]))
        check("blend3 blends totals over power/EPA only",
              abs(out["total"] - ((0.25 * 55.0 + 0.50 * 49.0) / 0.75)) < 1e-9,
              str(out["total"]))
    finally:
        PR.E.predict, PR.P.predict, PR.X.predict = se, sp, sx


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


def test_choose_stack_rejects_bad_epa():
    rng = np.random.default_rng(1)
    n = 3000
    margin = rng.normal(2.0, 15.0, n)
    res = (margin > 0).astype(float)
    p_true = 1.0 / (1.0 + np.exp(-margin / 12.0))
    df = pd.DataFrame({
        "p_elo": np.clip(p_true + rng.normal(0, 0.03, n), 1e-6, 1 - 1e-6),
        "p_pow": np.clip(p_true + rng.normal(0, 0.04, n), 1e-6, 1 - 1e-6),
        "p_epa": np.clip(1.0 - p_true + rng.normal(0, 0.03, n), 1e-6, 1 - 1e-6),
        "m_elo": margin + rng.normal(0, 8, n),
        "m_pow": margin + rng.normal(0, 8, n),
        "m_epa": -margin + rng.normal(0, 8, n),
        "t_pow": 52 + rng.normal(0, 8, n),
        "t_epa": 30 + rng.normal(0, 20, n),
        "margin": margin,
        "total": 52 + rng.normal(0, 8, n),
    })
    out = V.choose_stack_weights(df)
    check("bad EPA receives no constrained promotion",
          out["chosen"]["weights"]["epa"] == 0.0,
          str(out["chosen"]))
    check("chosen stack is eligible",
          out["chosen"]["margin_mae"] <= out["champion"]["margin_mae"] + 1e-9
          and out["chosen"]["total_mae"] <= out["champion"]["total_mae"] + 1e-9,
          str(out["chosen"]))


def test_ppa_split_grid_rejects_regressing_signal():
    rng = np.random.default_rng(2)
    n = 2500
    margin = rng.normal(2.0, 14.0, n)
    total = 54 + rng.normal(0, 7, n)
    p_true = np.clip(1.0 / (1.0 + np.exp(-margin / 11.0)), 1e-6, 1 - 1e-6)
    df = pd.DataFrame({
        "margin": margin,
        "total": total,
        "p_champ": np.clip(p_true + rng.normal(0, 0.025, n), 1e-6, 1 - 1e-6),
        "m_champ": margin + rng.normal(0, 7, n),
        "t_champ": total + rng.normal(0, 6, n),
    })
    for name in ("pass", "rush", "early", "third"):
        df[f"p_{name}"] = np.clip(1.0 - p_true + rng.normal(0, 0.02, n), 1e-6, 1 - 1e-6)
        df[f"m_{name}"] = -margin + rng.normal(0, 8, n)
        df[f"t_{name}"] = 32 + rng.normal(0, 14, n)
    names = ("champ", "pass", "rush", "early", "third")
    champion = V._score_named_stack(df, {"champ": 1.0}, names)
    rows = []
    for weights in V._simplex_weights(names, step=0.5):
        row = V._score_named_stack(df, weights, names)
        row["eligible"] = (
            row["margin_mae"] <= champion["margin_mae"] + 1e-9
            and row["total_mae"] <= champion["total_mae"] + 1e-9
        )
        rows.append(row)
    feasible = [r for r in rows if r["eligible"]]
    chosen = min(feasible or rows, key=lambda r: (r["ml_brier"], r["margin_mae"], r["total_mae"]))
    check("split-PPA grid keeps regressing signals out",
          chosen["weights"].get("champ") == 1.0,
          str(chosen))


def main():
    print("M6 CFB blend-weight tests")
    for fn in (test_default_weight, test_weight_file_loaded_and_clamped,
               test_stack_weight_backwards_compatibility,
               test_blend_predict_extremes, test_blend3_predict_uses_epa_only_when_requested,
               test_choose_weight_constrained, test_choose_stack_rejects_bad_epa,
               test_ppa_split_grid_rejects_regressing_signal):
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
