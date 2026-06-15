#!/usr/bin/env python3
"""Regression tests for v2 M4 (knockout correctness).

Run: python3 test_m4.py    (no pytest dependency; prints PASS/FAIL, exits non-zero
on any failure). Covers:
  1. 90-minute knockout settlement: ko_overrides.csv beats results.csv, and 1X2
     grading flips correctly between a 90' draw and an after-extra-time win.
  2. allocate_thirds produces a valid slotting for ALL 495 group combinations
     (no crash, distinct teams, every slot respects FIFA's allowed groups).
  3. Annex C loader: a valid committed table is used deterministically; a
     malformed one is rejected loudly.
"""
import json
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import bankroll
import simulate
from simulate import THIRD_SLOTS, allocate_thirds

_fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


# ── 1. 90-minute knockout settlement ─────────────────────────────────────────
def test_settlement():
    print("1. 90-minute knockout settlement")
    # grade() is the pure decision; verify a draw bet flips between 90' and FT
    check("draw bet wins on 0-0 (90')", bankroll.grade("draw", 0, 0) is True)
    check("draw bet loses on 1-0 (FT after ET)", bankroll.grade("draw", 1, 0) is False)

    # Integration: a knockout that was 0-0 at 90' but 1-0 after extra time.
    # results.csv (FT) would settle a 'draw' bet as LOST; ko_overrides (90') WON.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "results.csv").write_text(
            "date,home_team,away_team,home_score,away_score,tournament,"
            "city,country,neutral\n"
            "2026-06-28,France,Brazil,1,0,FIFA World Cup,Dallas,USA,True\n")
        (d / "ko_overrides.csv").write_text(
            "date,home,away,score90\n2026-06-28,France,Brazil,0-0\n")
        (d / "ledger.csv").write_text(
            "placed_on,match_date,home,away,side,bet,odds,stake,status,pnl,"
            "bankroll_after\n"
            "2026-06-27,2026-06-28,France,Brazil,draw,Draw,3.4,10,open,0.0,\n")
        (d / "bankroll.json").write_text(json.dumps({"bankroll": 100.0}))

        orig = (bankroll.DATA, bankroll.LEDGER, bankroll.STATE,
                bankroll.KO_OVERRIDES)
        bankroll.DATA = d
        bankroll.LEDGER = d / "ledger.csv"
        bankroll.STATE = d / "bankroll.json"
        bankroll.KO_OVERRIDES = d / "ko_overrides.csv"
        try:
            bankroll.settle(verbose=False)
            led = pd.read_csv(d / "ledger.csv")
            row = led.iloc[0]
            check("override path settles draw as WON", row["status"] == "won")
            check("bankroll credited at 90' odds (100 + 10*2.4)",
                  abs(float(json.loads((d / "bankroll.json").read_text())
                            ["bankroll"]) - 124.0) < 1e-6)

            # Now drop the override -> FT score (1-0) settles the same bet as LOST.
            (d / "ko_overrides.csv").write_text("date,home,away,score90\n")
            (d / "ledger.csv").write_text(
                "placed_on,match_date,home,away,side,bet,odds,stake,status,pnl,"
                "bankroll_after\n"
                "2026-06-27,2026-06-28,France,Brazil,draw,Draw,3.4,10,open,0.0,\n")
            (d / "bankroll.json").write_text(json.dumps({"bankroll": 100.0}))
            bankroll.settle(verbose=False)
            led = pd.read_csv(d / "ledger.csv")
            check("no override -> FT path settles draw as LOST",
                  led.iloc[0]["status"] == "lost")
        finally:
            (bankroll.DATA, bankroll.LEDGER, bankroll.STATE,
             bankroll.KO_OVERRIDES) = orig


# ── 2. allocate_thirds valid for all 495 combinations ────────────────────────
def test_all_combos():
    print("2. allocate_thirds over all 495 group combinations")
    rng = np.random.default_rng(42)
    groups = list("ABCDEFGHIJKL")
    bad = 0
    for combo in combinations(groups, 8):
        thirds = {g: f"team_{g}" for g in combo}        # group letter -> a team
        slots = allocate_thirds(thirds, rng)
        teams = set(slots.values())
        ok = (set(slots) == set(THIRD_SLOTS) and len(teams) == 8)
        # every slot must draw from an allowed group
        for slot, team in slots.items():
            g = team.split("_")[1]
            if g not in THIRD_SLOTS[slot]:
                ok = False
        if not ok:
            bad += 1
    check("all 495 combos produced a valid, complete slotting", bad == 0)


# ── 3. Annex C loader ────────────────────────────────────────────────────────
def _one_valid_assignment(combo):
    """Brute-force one valid matching for a combo (to build a tiny test table)."""
    slots = list(THIRD_SLOTS)
    cs = set(combo)
    res = {}

    def bt(i, used):
        if i == len(slots):
            return True
        for g in THIRD_SLOTS[slots[i]]:
            if g in cs and g not in used:
                res[slots[i]] = g
                if bt(i + 1, used | {g}):
                    return True
                del res[slots[i]]
        return False
    bt(0, set())
    return dict(res)


def test_annexc_loader():
    print("3. Annex C loader (use valid table, reject malformed)")
    combo = "ABCDEFGH"
    asg = _one_valid_assignment(combo)
    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "good.json"
        good.write_text(json.dumps({combo: asg}))
        orig = simulate.ANNEXC_FILE
        simulate.ANNEXC_FILE = good
        try:
            table = simulate._load_annexc()
            check("valid table loads", table is not None and combo in table)
            # patch the module-level cache and confirm allocate_thirds uses it
            simulate._ANNEXC = table
            thirds = {g: f"team_{g}" for g in combo}
            slots = allocate_thirds(thirds, np.random.default_rng(1))
            check("allocate_thirds uses the official table",
                  all(slots[s] == f"team_{asg[s]}" for s in asg))

            bad = Path(d) / "bad.json"
            bad.write_text(json.dumps({combo: {"T74": "Z"}}))  # illegal/incomplete
            simulate.ANNEXC_FILE = bad
            rejected = False
            try:
                simulate._load_annexc()
            except ValueError:
                rejected = True
            check("malformed table is rejected", rejected)
        finally:
            simulate.ANNEXC_FILE = orig
            simulate._ANNEXC = None


if __name__ == "__main__":
    test_settlement()
    test_all_combos()
    test_annexc_loader()
    print()
    if _fails:
        print(f"{len(_fails)} FAILURE(S): " + ", ".join(_fails))
        raise SystemExit(1)
    print("All M4 tests passed.")
