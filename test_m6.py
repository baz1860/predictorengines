#!/usr/bin/env python3
"""Regression tests for v2 M6 (context features: rest, altitude).

Run: python3 test_m6.py   (no pytest dependency). Covers:
  1. Altitude table + threshold (high-altitude venues count, low ones don't).
  2. Fitted coefficients: alt_gap kept (significant, negative), rest_diff dropped
     (|t| < 2); travel not present.
  3. multipliers: lowland side at altitude penalised, altitude side unaffected.
  4. market_probs(ctx=None) is a no-op (default path unchanged).
  5. Held-out validation passes (context not worse than baseline).
"""
import numpy as np

from engines.worldcup import context

from engines.worldcup import edge

from engines.worldcup.dixoncoles import build_sources

_fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def test_altitude_table():
    print("1. altitude table + threshold")
    check("Mexico City ~2.24km", abs(context.venue_alt_km("Mexico City") - 2.24) < 1e-9)
    check("Guadalajara/Zapopan counts (>1km)", context.venue_alt_km("Zapopan") > 1.0)
    check("Monterrey (<1km) -> 0", context.venue_alt_km("Guadalupe") == 0.0)
    check("US lowland (Atlanta) -> 0", context.venue_alt_km("Atlanta") == 0.0)
    check("unknown city -> 0", context.venue_alt_km("Nowhere") == 0.0)


def test_coefficients():
    print("2. fitted coefficients (data/context_coef.json)")
    coef = context.load_coef()
    check("coef file present", bool(coef))
    check("alt_gap kept and negative", coef.get("alt_gap", 0) < 0)
    check("rest_diff dropped (|t|<2)", "rest_diff" not in coef)
    check("travel not present", "travel" not in coef)


def test_multipliers():
    print("3. multipliers direction")
    coef = {"alt_gap": -0.14}
    # lowland side (gap 2.24) at Mexico City vs altitude side (gap 0)
    mh, ma = context.multipliers(0.0, 2.24, 0.0, coef)
    check("lowland side penalised at altitude (mult<1)", mh < 0.99)
    check("altitude side unaffected (mult==1)", abs(ma - 1.0) < 1e-9)
    # no altitude, no rest in coef -> identity
    m = context.multipliers(5.0, 0.0, 0.0, coef)
    check("no active feature -> identity", abs(m[0] - 1.0) < 1e-9)


def test_ctx_none_noop():
    print("4. market_probs(ctx=None) no-op")
    sources, ratings = build_sources("blend")
    nl = {}
    base = edge.market_probs("Brazil", "Argentina", sources, nl)
    none = edge.market_probs("Brazil", "Argentina", sources, nl, ctx=None)
    ident = edge.market_probs("Brazil", "Argentina", sources, nl, ctx=(1.0, 1.0))
    check("ctx=None equals default", all(abs(base[k] - none[k]) < 1e-12 for k in base))
    check("ctx=(1,1) equals default", all(abs(base[k] - ident[k]) < 1e-9 for k in base))


def test_validation():
    print("5. held-out validation")
    check("context not worse than baseline on held-out", context.validate(verbose=False))


if __name__ == "__main__":
    test_altitude_table()
    test_coefficients()
    test_multipliers()
    test_ctx_none_noop()
    test_validation()
    print()
    if _fails:
        print(f"{len(_fails)} FAILURE(S): " + ", ".join(_fails))
        raise SystemExit(1)
    print("All M6 tests passed.")
