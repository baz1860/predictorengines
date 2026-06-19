#!/usr/bin/env python3
"""M5 suite-level CLV tests.

Covers the V3 M5 CLV acceptance:
  * empty / no-snapshot history produces a clear no-data report, NOT a crash;
  * snapshot reads an engine's odds file and records matching open bets;
  * the closing-odds proxy is matched by (engine, event_id, market, side) and
    CLV% is computed correctly for a settled bet;
  * --write-closing backfills ledger.closing_odds (with a backup).

Runs entirely on temp files; the real ledger/history are never touched.

Run: python3 test_clv_suite.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import clv_suite as CLV
import app.bankroll_store as B

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


def _point_at(tmp: Path):
    B.DATA = tmp
    B.LEDGER = tmp / "suite_ledger.csv"
    B.STATE = tmp / "suite_bankroll.json"
    CLV.DATA = tmp
    CLV.HISTORY = tmp / "clv_history.csv"


def _write_ledger(tmp: Path, rows: list[dict]):
    df = pd.DataFrame(rows)
    for c in B.COLS:
        if c not in df.columns:
            df[c] = ""
    df[B.COLS].to_csv(tmp / "suite_ledger.csv", index=False)


def test_no_data_report():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _point_at(tmp)
        _write_ledger(tmp, [{"engine": "cfb", "status": "won", "odds": 2.0,
                             "match_date": "2026-06-20", "home": "A", "away": "B",
                             "side": "home", "market": "ml", "event_id": "e1"}])
        try:
            CLV.report()           # no history file at all
            ok = True
        except Exception as e:     # must NOT crash
            ok = False
            print("   ", e)
        check("no-snapshot report does not crash", ok)
        check("history file not created by report", not CLV.HISTORY.exists())


def test_snapshot_from_odds_file():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _point_at(tmp)
        odds = tmp / "club_odds.csv"
        pd.DataFrame([
            {"date": "2026-06-20", "competition": "Premier League", "home": "Arsenal",
             "away": "Liverpool", "market": "1x2", "side": "home", "line": "", "odds": 2.30},
            {"date": "2026-06-20", "competition": "Premier League", "home": "Arsenal",
             "away": "Liverpool", "market": "1x2", "side": "away", "line": "", "odds": 3.40},
        ]).to_csv(odds, index=False)
        CLV.CLUB_ODDS = odds
        _write_ledger(tmp, [{"engine": "club_soccer", "status": "open", "odds": 2.20,
                             "match_date": "2026-06-20", "home": "Arsenal",
                             "away": "Liverpool", "side": "home", "market": "1x2",
                             "event_id": "2026-06-20|Arsenal|Liverpool"}])
        n = CLV.snapshot()
        check("snapshot records the matching open bet", n == 1, f"n={n}")
        hist = pd.read_csv(CLV.HISTORY)
        check("snapshot stored current line 2.30",
              abs(float(hist.iloc[0]["odds"]) - 2.30) < 1e-9, str(hist.iloc[0]["odds"]))
        check("snapshot stored the event_id",
              hist.iloc[0]["event_id"] == "2026-06-20|Arsenal|Liverpool")


def test_clv_computed_and_backfill():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _point_at(tmp)
        # a settled WON bet taken at 2.20; closing proxy snapshot at 2.00 → +10% CLV
        _write_ledger(tmp, [{"engine": "cfb", "status": "won", "odds": 2.20, "pnl": 12.0,
                             "match_date": "2026-06-20", "home": "Ohio State",
                             "away": "Michigan", "side": "home", "market": "ml",
                             "event_id": "ev-cfb-1"}])
        pd.DataFrame([
            {"snapshot_time": "2026-06-19T12:00:00+00:00", "engine": "cfb",
             "event_id": "ev-cfb-1", "market": "ml", "side": "home",
             "home": "Ohio State", "away": "Michigan", "match_date": "2026-06-20",
             "odds": 2.00},
        ]).to_csv(CLV.HISTORY, index=False)
        ledger = CLV._load_ledger()
        hist = CLV._load_history()
        have = CLV._clv_table(ledger, hist)
        check("CLV table has the settled bet", len(have) == 1, f"len={len(have)}")
        clv = float(have.iloc[0]["clv"])
        check("CLV% = 2.20/2.00 - 1 = +10%", abs(clv - 0.10) < 1e-9, f"{clv:.4f}")

        # closing proxy must NOT match a different market/side
        ledger2 = ledger.copy()
        ledger2.loc[0, "side"] = "away"
        have2 = CLV._clv_table(ledger2, hist)
        check("no match for a different side → excluded", len(have2) == 0, f"len={len(have2)}")

        # backfill writes closing_odds + a backup
        CLV.report(write_closing=True)
        back = pd.read_csv(B.LEDGER)
        check("backfill wrote closing_odds 2.0",
              abs(float(back.iloc[0]["closing_odds"]) - 2.0) < 1e-9,
              str(back.iloc[0]["closing_odds"]))
        check("backfill made a ledger backup",
              B.LEDGER.with_suffix(".csv.bak.clv").exists())


def main():
    print("M5 suite-CLV tests")
    test_no_data_report()
    test_snapshot_from_odds_file()
    test_clv_computed_and_backfill()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
