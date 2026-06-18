#!/usr/bin/env python3
"""Betting-performance backtest for the World Cup engine.

A betting model is judged on profit at the odds taken, not on 3-way accuracy.
This scores the actual placed bets in the suite ledger -- which records the
price and stake at placement time, so it is already a point-in-time record that
model changes cannot retro-edit -- and reports yield/ROI overall, by market, and
by odds band. Closing-line value (CLV) is pulled from the existing `clv` module
when odds snapshots exist.

Settled bets (won/lost) drive P&L; open bets are reported as live exposure only.

Usage:
  python backtest_betting.py                 # worldcup soccer (default)
  python backtest_betting.py --engine golf   # another engine in the suite ledger
  python backtest_betting.py --csv out.csv   # also write per-bet detail
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SUITE = ROOT / "data" / "suite_ledger.csv"
LEDGER = ROOT / "data" / "ledger.csv"


def market_of(side):
    s = str(side)
    if s in ("home", "away", "draw"):
        return "1X2 (match result)"
    if s in ("over25", "under25"):
        return "Totals (O/U 2.5)"
    if s.startswith("btts"):
        return "BTTS"
    if s == "outright":
        return "Outright"
    return "Other"


def odds_band(o):
    o = float(o)
    if o < 1.8:
        return "Short (<1.8)"
    if o < 2.5:
        return "Even (1.8-2.5)"
    if o < 4.0:
        return "Mid (2.5-4.0)"
    return "Longshot (>=4.0)"


def block(df, by):
    rows = []
    for key, g in df.groupby(by):
        staked = g.stake.sum()
        pnl = g.pnl.sum()
        rows.append(dict(bucket=key, bets=len(g), won=int((g.status == "won").sum()),
                         staked=round(staked, 2), pnl=round(pnl, 2),
                         roi=f"{100*pnl/staked:+.1f}%" if staked else "-"))
    return pd.DataFrame(rows)


def main():
    engine = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else "worldcup"

    if SUITE.exists():
        led = pd.read_csv(SUITE)
        led = led[led["engine"] == engine].copy()
        src = "suite_ledger.csv"
    else:
        led = pd.read_csv(LEDGER); src = "ledger.csv"
    for c in ("odds", "stake", "pnl"):
        led[c] = pd.to_numeric(led[c], errors="coerce")

    settled = led[led.status.isin(["won", "lost"])].copy()
    openb = led[led.status == "open"].copy()
    if settled.empty:
        print("No settled bets yet.")
        return

    settled["market"] = settled.side.map(market_of)
    settled["band"] = settled.odds.map(odds_band)
    n = len(settled); staked = settled.stake.sum(); pnl = settled.pnl.sum()

    print(f"=== Betting backtest: {engine}  (source: {src}) ===\n")
    print(f"Settled bets : {n}  ({(settled.status=='won').sum()}W / "
          f"{(settled.status=='lost').sum()}L, strike {100*(settled.status=='won').mean():.0f}%)")
    print(f"Total staked : {staked:.2f} u")
    print(f"Net P&L      : {pnl:+.2f} u")
    print(f"Yield / ROI  : {100*pnl/staked:+.1f}%")
    print(f"Avg odds     : {settled.odds.mean():.2f}")
    if not openb.empty:
        print(f"Open exposure: {len(openb)} bets, {pd.to_numeric(openb.stake).sum():.2f} u staked")

    big = settled.loc[settled.pnl.idxmax()]; bad = settled.loc[settled.pnl.idxmin()]
    print(f"\nBiggest winner: {big.home} v {big.away} — {big.bet} @ {big.odds} -> {big.pnl:+.2f} u")
    print(f"Biggest loser : {bad.home} v {bad.away} — {bad.bet} @ {bad.odds} -> {bad.pnl:+.2f} u")
    # concentration check: how much of the profit is the single best bet?
    if pnl > 0:
        print(f"Profit concentration: best bet = {100*big.pnl/pnl:.0f}% of net P&L; "
              f"ex-best = {pnl-big.pnl:+.2f} u "
              f"({100*(pnl-big.pnl)/(staked-big.stake):+.1f}% ROI)")

    print("\nBy market:")
    print(block(settled, "market").to_string(index=False))
    print("\nBy odds band:")
    bands = block(settled, "band")
    order = {"Short (<1.8)": 0, "Even (1.8-2.5)": 1, "Mid (2.5-4.0)": 2, "Longshot (>=4.0)": 3}
    print(bands.sort_values("bucket", key=lambda s: s.map(order)).to_string(index=False))

    # CLV via existing module (needs data/odds_history.csv snapshots)
    try:
        from core import clv
        hist = clv._load_history()
        if hist is not None and "snapshot_time" in hist:
            # normalise tz-aware timestamps to naive so closing_odds() can compare
            hist = hist.copy()
            hist["snapshot_time"] = (pd.to_datetime(hist["snapshot_time"], utc=True)
                                     .dt.tz_localize(None))
        clv_vals = clv.compute_clv(settled.reset_index(drop=True), hist)
        have = clv_vals.dropna()
        print("\nClosing-line value (CLV):")
        if len(have):
            print(f"  bets with closing odds: {len(have)}/{n}")
            print(f"  mean CLV  : {have.mean()*100:+.2f}%")
            print(f"  positive-CLV rate: {(have>0).mean():.0%}")
        else:
            print("  no closing-odds snapshots yet — run `python clv.py --snapshot` "
                  "before kickoffs (needs The Odds API).")
    except Exception as e:
        print(f"\nCLV unavailable: {e}")

    if "--csv" in sys.argv:
        dest = sys.argv[sys.argv.index("--csv") + 1]
        settled.to_csv(dest, index=False)
        print(f"\nPer-bet detail -> {dest}")


if __name__ == "__main__":
    main()
