#!/usr/bin/env python3
"""Closing line value (CLV) tracking (v2 M3).

CLV is the single most reliable signal that a betting operation is genuinely +EV:
it measures whether the odds you took beat the market's closing odds. Positive
mean CLV over time predicts long-run profit far better than a small P&L sample.

    CLV% (per bet) = bet_odds / closing_odds - 1

This module snapshots live odds for open bets (so a closing line exists to compare
against later) and reports CLV once bets are settled.

  python3 clv.py --snapshot   # record current odds for open ledger bets into
                              #   data/odds_history.csv  (needs The Odds API;
                              #   run on Barrie's machine — degrades offline)
  python3 clv.py --report     # per-settled-bet CLV, rolling mean CLV, win rate,
                              #   actual P&L vs CLV-expected P&L

The last snapshot at/ before kick-off is used as the closing-odds proxy.
Reads data/ledger.csv (never modified) and data/odds_history.csv.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
LEDGER = DATA / "ledger.csv"
ODDS_HISTORY = DATA / "odds_history.csv"
HISTORY_COLS = ["snapshot_time", "match_date", "home", "away", "side", "odds"]
# ledger side -> odds.csv / API column produced by edge.fetch_api_odds
SIDE_COL = {"home": "odds_home", "draw": "odds_draw", "away": "odds_away",
            "over25": "odds_over25", "under25": "odds_under25",
            "btts_yes": "odds_btts_yes", "btts_no": "odds_btts_no"}


def _load_history():
    if not ODDS_HISTORY.exists():
        return None
    try:
        df = pd.read_csv(ODDS_HISTORY)
    except Exception:
        return None
    return df if not df.empty else None


def closing_odds(hist, match_date, home, away, side):
    """Closing-odds proxy: latest snapshot for this outcome at/before kick-off
    (treated as end of match_date). Returns float or None."""
    if hist is None:
        return None
    m = hist[(hist["home"] == home) & (hist["away"] == away)
             & (hist["side"] == side)].copy()
    if m.empty:
        return None
    cutoff = pd.Timestamp(str(match_date)) + pd.Timedelta(days=1)
    m["t"] = pd.to_datetime(m["snapshot_time"], errors="coerce")
    m = m[m["t"] <= cutoff]
    if m.empty:
        return None
    o = float(m.sort_values("t").iloc[-1]["odds"])
    return o if o > 1.0 else None


def compute_clv(ledger, hist=None):
    """Series of CLV% (fraction) per ledger row; NaN when no closing snapshot."""
    if hist is None:
        hist = _load_history()
    vals = []
    for r in ledger.itertuples(index=False):
        co = closing_odds(hist, r.match_date, r.home, r.away, r.side)
        vals.append(float(r.odds) / co - 1.0 if co else np.nan)
    return pd.Series(vals, index=ledger.index)


def snapshot(api_key=None):
    """Record current odds for open ledger bets. Needs network (The Odds API)."""
    if not LEDGER.exists():
        print("No ledger yet — nothing to snapshot.")
        return
    ledger = pd.read_csv(LEDGER)
    open_bets = ledger[ledger["status"] == "open"]
    if open_bets.empty:
        print("No open bets to snapshot.")
        return
    from edge import fetch_api_odds, ALIASES, DEFAULT_API_KEY
    key = api_key or DEFAULT_API_KEY
    try:
        odds = fetch_api_odds(key)
    except SystemExit as e:           # fetch_api_odds exits on network failure
        print(f"Snapshot skipped — could not reach The Odds API ({e}). "
              "Run this on a machine with internet access.")
        return
    # index fetched odds by canonicalised (home, away)
    canon = lambda n: ALIASES.get(str(n).strip(), str(n).strip())
    fetched = {(canon(r.home), canon(r.away)): r for r in odds.itertuples(index=False)}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_rows = []
    for b in open_bets.itertuples(index=False):
        row = fetched.get((b.home, b.away))
        col = SIDE_COL.get(b.side)
        if row is None or col is None:
            continue
        o = getattr(row, col, np.nan)
        if pd.isna(o) or float(o) <= 1.0:
            continue
        new_rows.append({"snapshot_time": now, "match_date": b.match_date,
                         "home": b.home, "away": b.away, "side": b.side,
                         "odds": round(float(o), 3)})
    if not new_rows:
        print("No live odds matched the open bets (names/markets); nothing recorded.")
        return
    hist = _load_history()
    out = (pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)
           if hist is not None else pd.DataFrame(new_rows, columns=HISTORY_COLS))
    out.to_csv(ODDS_HISTORY, index=False)
    print(f"Recorded {len(new_rows)} odds snapshot(s) at {now} -> "
          f"{ODDS_HISTORY.name} ({len(out)} rows total).")


def report():
    hist = _load_history()
    if hist is None:
        print("No CLV snapshots yet. Run 'python3 clv.py --snapshot' "
              "(needs The Odds API) before matches kick off.")
        return
    if not LEDGER.exists():
        print("No ledger yet.")
        return
    ledger = pd.read_csv(LEDGER)
    settled = ledger[ledger["status"].isin(["won", "lost"])].copy()
    if settled.empty:
        print("No settled bets yet.")
        return
    settled["closing"] = [closing_odds(hist, r.match_date, r.home, r.away, r.side)
                          for r in settled.itertuples(index=False)]
    have = settled.dropna(subset=["closing"]).copy()
    if have.empty:
        print("No settled bets have a closing-odds snapshot yet.")
        return
    have["clv"] = have["odds"] / have["closing"] - 1.0
    # CLV-expected P&L: treat closing odds as fair (implied prob = 1/closing)
    have["exp_pnl"] = have["stake"] * (have["odds"] / have["closing"] - 1.0)

    pd.set_option("display.width", 160)
    show = have[["match_date", "bet", "odds", "closing", "clv", "status",
                 "stake", "pnl", "exp_pnl"]].copy()
    show["clv"] = (show["clv"] * 100).map("{:+.1f}%".format)
    show["exp_pnl"] = show["exp_pnl"].map("{:+.2f}".format)
    print(f"CLV report — {len(have)} settled bet(s) with closing snapshots:\n")
    print(show.to_string(index=False))
    print(f"\n  Rolling mean CLV : {have['clv'].mean() * 100:+.2f}%")
    print(f"  Positive-CLV rate: {(have['clv'] > 0).mean():.0%}")
    print(f"  Win rate         : {(have['status'] == 'won').mean():.0%}")
    print(f"  Actual P&L       : £{have['pnl'].sum():+.2f}")
    print(f"  CLV-expected P&L : £{have['exp_pnl'].sum():+.2f}  "
          "(if closing odds were fair)")


def main():
    ap = argparse.ArgumentParser(description="Closing line value tracking (v2 M3)")
    ap.add_argument("--snapshot", action="store_true",
                    help="record current odds for open bets (needs The Odds API)")
    ap.add_argument("--report", action="store_true",
                    help="CLV report for settled bets")
    ap.add_argument("--api-key", help="The Odds API key (the-odds-api.com)")
    args = ap.parse_args()
    if args.snapshot:
        snapshot(args.api_key)
    elif args.report:
        report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
