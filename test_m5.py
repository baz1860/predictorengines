#!/usr/bin/env python3
"""Regression tests for v2 M5 (squad layer v2).

Run: python3 test_m5.py   (no pytest dependency). Covers:
  1. Starter-weighting in squad_power (ranks 1-11 full, 12-18 half, rest 0).
  2. Position-aware split: att_adj + def_adj == elo_adj; a pure-DF absence skews
     to defence, a pure-FW absence skews to attack.
  3. adjusted_sources applies asymmetrically (attack lowers OWN goals; defence
     raises the OPPONENT's).
  4. Backfill: data/squads.csv has 0 ea_proxy rows and 48 teams.
"""
import numpy as np
import pandas as pd

from engines.worldcup import squads

from engines.worldcup.squads import squad_power, POS_DEF_SHARE, SQUADS_CSV, OUT_CSV

_fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def test_starter_weighting():
    print("1. starter-weighting in squad_power")
    check("18 equal ratings -> that rating", abs(squad_power([80] * 18) - 80) < 1e-9)
    # 11 at 90, 7 at 70: (11*90 + 0.5*7*70) / (11 + 3.5)
    expect = (11 * 90 + 0.5 * 7 * 70) / 14.5
    check("XI vs bench weighting", abs(squad_power([90] * 11 + [70] * 7) - expect) < 1e-9)
    # a weaker 19th player (rank>18) is ignored
    check("rank>18 ignored", abs(squad_power([80] * 18 + [50]) - 80) < 1e-9)


def test_split_invariant_and_direction():
    print("2. att/def split invariant + direction")
    df = pd.read_csv(OUT_CSV)
    inv = np.allclose(df["att_adj"] + df["def_adj"], df["elo_adj"], atol=0.05)
    check("att_adj + def_adj == elo_adj for all teams", inv)
    # direction via POS_DEF_SHARE
    check("DF absence weighted to defence", POS_DEF_SHARE["DF"] > 0.5)
    check("FW absence weighted to attack", POS_DEF_SHARE["FW"] < 0.5)
    nl = df[df["team"] == "Netherlands"]
    if not nl.empty and nl["elo_adj"].iloc[0] != 0:   # only Timber (DF) out
        check("Netherlands (DF out) def_frac == 0.75",
              abs(nl["def_frac"].iloc[0] - 0.75) < 1e-6)


def test_asymmetric_application():
    print("3. adjusted_sources asymmetry")
    from engines.worldcup.dixoncoles import build_sources
    raw, ratings = build_sources("blend")
    t1, t2 = "Brazil", "Argentina"
    orig = squads.load_adj_split
    try:
        # pure ATTACK hit on t1 -> its own goals drop, opponent ~unchanged
        squads.load_adj_split = lambda: ({t1: -25.0}, {t1: 0.0})
        wrapped, _, _ = squads.adjusted_sources("blend")
        r1, r2 = raw[0][0](t1, t2), wrapped[0][0](t1, t2)
        check("attack hit lowers OWN goals", r2[0] < r1[0] - 1e-6)
        check("attack hit barely moves opponent goals", abs(r2[1] - r1[1]) < 1e-6)
        # pure DEFENCE hit on t1 -> opponent goals rise, own ~unchanged
        squads.load_adj_split = lambda: ({t1: 0.0}, {t1: -25.0})
        wrapped2, _, _ = squads.adjusted_sources("blend")
        d1 = wrapped2[0][0](t1, t2)
        check("defence hit raises OPPONENT goals", d1[1] > r1[1] + 1e-6)
        check("defence hit barely moves own goals", abs(d1[0] - r1[0]) < 1e-6)
    finally:
        squads.load_adj_split = orig


def test_backfill():
    print("4. squads.csv backfill")
    s = pd.read_csv(SQUADS_CSV)
    check("0 ea_proxy rows remain", (s["source"] == "ea_proxy").sum() == 0)
    check("48 teams present", s["team"].nunique() == 48)


if __name__ == "__main__":
    test_starter_weighting()
    test_split_invariant_and_direction()
    test_asymmetric_application()
    test_backfill()
    print()
    if _fails:
        print(f"{len(_fails)} FAILURE(S): " + ", ".join(_fails))
        raise SystemExit(1)
    print("All M5 tests passed.")
