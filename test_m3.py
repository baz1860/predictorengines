#!/usr/bin/env python3
"""Regression tests for v2 M3 (market blend + CLV).

Run: python3 test_m3.py   (no pytest dependency). Covers:
  1. Logit blend math: w=1 -> model, w=0 -> market, sums to 1, monotonic in w,
     and the blend sits between model and market.
  2. compute_clv: closing-odds proxy = latest snapshot at/ before kick-off;
     CLV% = bet_odds/closing - 1; no snapshot -> NaN.
  3. Fitted w (data/market_blend.json) is interior and strictly beats both
     pure-model and pure-market log-loss on WC2022.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import clv
from engines.worldcup.market_blend import blend, BLEND_FILE

_fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def test_blend_math():
    print("1. logit blend math")
    pm = np.array([0.50, 0.30, 0.20])
    pk = np.array([0.40, 0.30, 0.30])
    b1 = blend(pm, pk, 1.0)
    b0 = blend(pm, pk, 0.0)
    bm = blend(pm, pk, 0.5)
    check("w=1 returns model", np.allclose(b1, pm, atol=1e-6))
    check("w=0 returns market", np.allclose(b0, pk, atol=1e-6))
    check("blend sums to 1", abs(bm.sum() - 1.0) < 1e-9)
    # home prob: market 0.40 < blend < model 0.50, and increasing w -> toward model
    seq = [blend(pm, pk, w)[0] for w in (0.0, 0.25, 0.5, 0.75, 1.0)]
    check("home prob monotonic increasing in w", all(x <= y + 1e-9
          for x, y in zip(seq, seq[1:])))
    check("blend lies between market and model", pk[0] <= bm[0] <= pm[0])


def test_compute_clv():
    print("2. compute_clv")
    ledger = pd.DataFrame([
        {"match_date": "2026-06-20", "home": "Brazil", "away": "Haiti",
         "side": "home", "odds": 2.00, "status": "won", "stake": 5.0, "pnl": 5.0},
        {"match_date": "2026-06-20", "home": "Spain", "away": "Uruguay",
         "side": "draw", "odds": 3.50, "status": "lost", "stake": 5.0, "pnl": -5.0},
    ])
    hist = pd.DataFrame([
        # two snapshots for Brazil home: latest before kickoff (1.80) is closing
        {"snapshot_time": "2026-06-18T09:00:00", "match_date": "2026-06-20",
         "home": "Brazil", "away": "Haiti", "side": "home", "odds": 1.95},
        {"snapshot_time": "2026-06-20T10:00:00", "match_date": "2026-06-20",
         "home": "Brazil", "away": "Haiti", "side": "home", "odds": 1.80},
        # Spain draw has no snapshot -> NaN CLV
    ])
    s = clv.compute_clv(ledger, hist)
    check("Brazil CLV uses latest snapshot (2.00/1.80-1)",
          abs(s.iloc[0] - (2.00 / 1.80 - 1.0)) < 1e-9)
    check("missing snapshot -> NaN", pd.isna(s.iloc[1]))
    co = clv.closing_odds(hist, "2026-06-20", "Brazil", "Haiti", "home")
    check("closing_odds picks pre-kickoff latest (1.80)", abs(co - 1.80) < 1e-9)


def test_fitted_w():
    print("3. fitted w on WC2022 (data/market_blend.json)")
    if not BLEND_FILE.exists():
        check("market_blend.json exists (run market_blend.py --fit)", False)
        return
    d = json.loads(BLEND_FILE.read_text())
    check("w is interior (0 < w < 1)", 0.0 < d["w"] < 1.0)
    check("blend log-loss < model-only", d["logloss_blend"] < d["logloss_model_only"])
    check("blend log-loss < market-only", d["logloss_blend"] < d["logloss_market_only"])


if __name__ == "__main__":
    test_blend_math()
    test_compute_clv()
    test_fitted_w()
    print()
    if _fails:
        print(f"{len(_fails)} FAILURE(S): " + ", ".join(_fails))
        raise SystemExit(1)
    print("All M3 tests passed.")
