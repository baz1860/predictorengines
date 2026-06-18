#!/usr/bin/env python3
"""Regression tests for v2 M7 (portfolio staking discipline).

Run: python3 test_m7.py   (no pytest dependency). Covers:
  1. Simultaneous-Kelly daily cap + single-match cap (6 big-edge same-day bets).
  2. Correlation guard: shared-team combined exposure (incl. existing open
     outright) capped at 1.5x single-match cap.
  3. Drawdown brake halves Kelly below 70% of peak.
  4. bankroll.json peak tracking is backward-compatible.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import edge
from core import bankroll
from edge import (portfolio_size, SINGLE_MATCH_CAP, DAILY_EXPOSURE_CAP,
                  CORR_CAP_MULT)

_fails = []
EMPTY_LEDGER = pd.DataFrame(columns=["home", "away", "side", "stake", "status"])


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def _auto(rows):
    return pd.DataFrame(rows)


def test_daily_and_single_caps():
    print("1. daily + single-match caps (6 big-edge same-day bets)")
    bank = 100.0
    rows = [{"match_date": "2026-06-20", "home": f"H{i}", "away": f"A{i}",
             "side": "home", "bet": f"H{i} win", "odds": 3.0,
             "ev_per_unit": 1.0 + i * 0.01, "kelly_stake": 0.20}  # £20 each pre
            for i in range(6)]
    out = portfolio_size(_auto(rows), bank, peak=bank, ledger=EMPTY_LEDGER,
                         verbose=False)
    total = out["stake_post"].sum()
    check("total recorded <= daily cap",
          total <= DAILY_EXPOSURE_CAP * bank + 1e-6)
    check("each bet <= single-match cap",
          bool((out["stake_post"] <= SINGLE_MATCH_CAP * bank + 1e-6).all()))
    check("rescaled kelly_stake matches stake_post",
          np.allclose(out["kelly_stake"] * bank, out["stake_post"], atol=0.02))


def test_correlation_guard():
    print("2. correlation guard (shared team incl. open outright)")
    bank = 100.0
    corr_cap = CORR_CAP_MULT * SINGLE_MATCH_CAP * bank      # £15
    # two NEW bets both touching Brazil
    rows = [{"match_date": "2026-06-20", "home": "Brazil", "away": "Haiti",
             "side": "home", "bet": "Brazil win", "odds": 2.0,
             "ev_per_unit": 0.9, "kelly_stake": 0.20},
            {"match_date": "2026-06-20", "home": "Brazil", "away": "Scotland",
             "side": "home", "bet": "Brazil win", "odds": 2.0,
             "ev_per_unit": 0.8, "kelly_stake": 0.20}]
    out = portfolio_size(_auto(rows), bank, peak=bank, ledger=EMPTY_LEDGER,
                         verbose=False)
    brazil_total = out["stake_post"].sum()
    check("two Brazil bets combined <= 1.5x single cap",
          brazil_total <= corr_cap + 1e-6)

    # new Brazil bet vs an existing OPEN outright on Brazil (£12 already on)
    led = pd.DataFrame([{"home": "Argentina", "away": "—OUTRIGHT—",
                         "side": "outright", "stake": 12.0, "status": "open"}])
    rows2 = [{"match_date": "2026-06-20", "home": "Argentina", "away": "Austria",
              "side": "home", "bet": "Argentina win", "odds": 2.0,
              "ev_per_unit": 1.0, "kelly_stake": 0.20}]   # wants £10
    out2 = portfolio_size(_auto(rows2), bank, peak=bank, ledger=led, verbose=False)
    check("new bet capped by existing open exposure on same team",
          out2["stake_post"].iloc[0] <= corr_cap - 12.0 + 1e-6)


def test_drawdown_brake():
    print("3. drawdown brake")
    rows = [{"match_date": "2026-06-20", "home": "H", "away": "A", "side": "home",
             "bet": "H win", "odds": 3.0, "ev_per_unit": 0.5, "kelly_stake": 0.05}]
    bank = 60.0
    no_dd = portfolio_size(_auto(rows), bank, peak=60.0, ledger=EMPTY_LEDGER,
                           verbose=False)["stake_post"].iloc[0]
    dd = portfolio_size(_auto(rows), bank, peak=100.0, ledger=EMPTY_LEDGER,
                        verbose=False)["stake_post"].iloc[0]   # 60 < 70% of 100
    check("no brake when at peak (stake = quarter-Kelly £3)", abs(no_dd - 3.0) < 0.05)
    check("brake halves Kelly in drawdown (£1.50)", abs(dd - 1.5) < 0.05)


def test_peak_backward_compat():
    print("4. bankroll.json peak tracking backward-compatible")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "bankroll.json").write_text(json.dumps({"bankroll": 90.0}))  # v1 shape
        orig = (bankroll.STATE, bankroll.LEDGER)
        bankroll.STATE = d / "bankroll.json"
        bankroll.LEDGER = d / "ledger.csv"          # absent
        try:
            check("v1 json loads bankroll", bankroll.current_bankroll() == 90.0)
            check("peak migrates to >= start/bankroll",
                  bankroll.current_peak() >= 90.0)
            bankroll._save_bankroll(95.0)            # writes peak field
            st = json.loads((d / "bankroll.json").read_text())
            check("peak persisted and never below bankroll", st["peak"] >= 95.0)
        finally:
            bankroll.STATE, bankroll.LEDGER = orig


if __name__ == "__main__":
    test_daily_and_single_caps()
    test_correlation_guard()
    test_drawdown_brake()
    test_peak_backward_compat()
    print()
    if _fails:
        print(f"{len(_fails)} FAILURE(S): " + ", ".join(_fails))
        raise SystemExit(1)
    print("All M7 tests passed.")
