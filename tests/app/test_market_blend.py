#!/usr/bin/env python3
"""M5 shared market-blend tests.

Covers the generalised, dependency-light blend used by every priced engine:
  * logit-space blend math (pure model / pure market / midpoint behaviour);
  * blend pulls the model toward the market and shrinks an inflated edge;
  * line anchoring is a linear convex blend;
  * the adapter row-applier recomputes edge/EV/Kelly/stake in place;
  * blends are OFF by default (no engine is a validated default yet).

Run: python3 test_market_blend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import market_blend as MB

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")
        assert cond, detail or name


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_blend_extremes():
    pm = [0.6, 0.25, 0.15]
    pk = [0.4, 0.30, 0.30]
    pure_model = MB.blend_probs(pm, pk, 1.0)
    pure_market = MB.blend_probs(pm, pk, 0.0)
    check("pure model (w=1) == model", all(approx(a, b) for a, b in zip(pure_model, pm)),
          str(pure_model))
    check("pure market (w=0) == market", all(approx(a, b) for a, b in zip(pure_market, pk)),
          str(pure_market))
    blended = MB.blend_probs(pm, pk, 0.5)
    check("blend renormalises to 1", approx(sum(blended), 1.0), str(sum(blended)))
    check("blend sits between model and market on outcome 0",
          pk[0] < blended[0] < pm[0], str(blended))


def test_blend_two_shrinks_edge():
    # model is bullish vs a fairer market; blending must reduce the model prob.
    p_model, p_market = 0.62, 0.50
    blended = MB.blend_two(p_model, p_market, 0.4)
    check("blend_two pulls model toward market",
          p_market < blended < p_model, f"{blended:.4f}")


def test_anchor_line():
    check("anchor_line w=1 == model", approx(MB.anchor_line(7.0, 3.0, 1.0), 7.0))
    check("anchor_line w=0 == market", approx(MB.anchor_line(7.0, 3.0, 0.0), 3.0))
    check("anchor_line midpoint", approx(MB.anchor_line(7.0, 3.0, 0.5), 5.0))


def test_devig():
    p = MB.devig([2.0, 2.0])  # 50/50 book with vig folded out
    check("devig sums to 1", approx(sum(p), 1.0), str(p))
    check("devig symmetric", approx(p[0], 0.5) and approx(p[1], 0.5), str(p))
    check("devig drops invalid odds", MB.devig([1.0, 0.0]) == [], str(MB.devig([1.0, 0.0])))


def test_apply_to_rows():
    rows = [
        {"p_model": 0.62, "p_book": 0.50, "odds": 2.0, "edge": 0.12,
         "ev_per_unit": 0.24, "kelly_frac": 0.05, "stake_gbp": 5.0},
        {"p_model": "n/a", "p_book": 0.5, "odds": 2.0},  # untouched (bad p_model)
    ]
    before_edge = rows[0]["edge"]
    w = MB.apply_blend_to_rows(rows, "cfb", bankroll=100.0, kelly_fraction=0.25, w=0.4)
    check("apply returns weight used", approx(w, 0.4), str(w))
    check("apply shrinks the inflated edge", rows[0]["edge"] < before_edge,
          f"{rows[0]['edge']} !< {before_edge}")
    check("apply stamps market_blend_w", rows[0].get("market_blend_w") == 0.4,
          str(rows[0].get("market_blend_w")))
    check("apply recomputes stake_gbp", rows[0]["stake_gbp"] ==
          round(rows[0]["kelly_frac"] * 100.0, 2), str(rows[0]["stake_gbp"]))
    check("apply leaves un-priceable row untouched", "market_blend_w" not in rows[1],
          str(rows[1]))


def test_defaults_off():
    check("no engine is a validated blend default", len(MB.DEFAULT_BLEND_ON) == 0)
    check("club_soccer blend not default-on", not MB.is_default_on("club_soccer"))
    check("cfb blend not default-on", not MB.is_default_on("cfb"))
    check("weight_for falls back without a file",
          isinstance(MB.weight_for("nonexistent_engine"), float))


def main():
    print("M5 market-blend tests")
    for fn in (test_blend_extremes, test_blend_two_shrinks_edge, test_anchor_line,
               test_devig, test_apply_to_rows, test_defaults_off):
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
