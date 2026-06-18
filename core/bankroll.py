#!/usr/bin/env python3
"""Bankroll tracker: records bets edge.py recommends, settles them against
real results, and compounds the bankroll so future Kelly stakes are sized
from what you actually have.

Usage:
  python3 bankroll.py             # status: bankroll, open bets, recent results
  python3 bankroll.py --settle    # settle open bets against data/results.csv
  python3 bankroll.py --reset 100 # start over with a fresh bankroll

State lives in data/bankroll.json and data/ledger.csv.

Knockout 90-minute settlement (v2 M4): bookmaker 1X2 / O-U / BTTS markets settle
on the 90-minute score, but data/results.csv records the after-extra-time score
(penalties excluded). For any match listed in data/ko_overrides.csv
(date,home,away,score90) settlement uses that 90-minute score; otherwise it falls
back to the results.csv score (exact for the group stage, and for knockouts that
either stayed level through to FT or scored no extra-time goals). The daily task
fills ko_overrides.csv from news whenever a knockout goes to extra time.
"""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
STATE = DATA / "bankroll.json"
LEDGER = DATA / "ledger.csv"
KO_OVERRIDES = DATA / "ko_overrides.csv"   # date,home,away,score90 (90-min score)
START_BANKROLL = 100.0
MIN_STAKE = 0.10   # don't bother recording stakes below 10p

COLS = ["placed_on", "match_date", "home", "away", "side", "bet", "odds",
        "stake", "status", "pnl", "bankroll_after"]


def _load_state():
    if STATE.exists():
        try:
            d = json.loads(STATE.read_text())
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {}


def current_bankroll():
    return _load_state().get("bankroll", START_BANKROLL)


def current_peak():
    """Running-peak bankroll for the drawdown brake (M7). Backward-compatible:
    if data/bankroll.json predates peak tracking, infer it from the ledger's
    bankroll_after history and the current/starting bankroll."""
    d = _load_state()
    if "peak" in d:
        return d["peak"]
    peak = max(START_BANKROLL, d.get("bankroll", START_BANKROLL))
    if LEDGER.exists():
        ba = pd.to_numeric(_load_ledger().get("bankroll_after"), errors="coerce")
        ba = ba.dropna()
        if len(ba):
            peak = max(peak, float(ba.max()))
    return round(peak, 2)


def _save_bankroll(amount, peak=None):
    d = _load_state()
    d["bankroll"] = round(amount, 2)
    if peak is None:                                 # never let the peak fall
        peak = max(amount, d.get("peak", current_peak()))
    d["peak"] = round(peak, 2)
    STATE.write_text(json.dumps(d))


def _load_ledger():
    if LEDGER.exists():
        return pd.read_csv(LEDGER)
    return pd.DataFrame(columns=COLS)


def place_bets(candidates):
    """candidates: DataFrame with match_date, home, away, side, bet, odds,
    kelly_stake (fraction). Returns the newly placed rows."""
    ledger = _load_ledger()
    bankroll = current_bankroll()
    # never let total open exposure exceed the bankroll: stakes are sized
    # off the full bankroll but capped by what isn't already committed
    available = bankroll - ledger.loc[ledger["status"] == "open", "stake"].sum()
    existing = set(zip(ledger["home"], ledger["away"], ledger["side"]))
    new_rows = []
    for r in candidates.itertuples(index=False):
        if (r.home, r.away, r.side) in existing:
            continue
        stake = min(round(r.kelly_stake * bankroll, 2), round(max(available, 0), 2))
        if stake < MIN_STAKE:
            continue
        available -= stake
        new_rows.append({"placed_on": str(date.today()),
                         "match_date": r.match_date, "home": r.home,
                         "away": r.away, "side": r.side, "bet": r.bet,
                         "odds": r.odds, "stake": stake, "status": "open",
                         "pnl": 0.0, "bankroll_after": ""})
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    return pd.DataFrame(new_rows, columns=COLS)


def _load_ko_overrides():
    """90-minute knockout scores keyed by (date, home, away).

    File schema: date,home,away,score90  where score90 is e.g. "1-1" (the score
    at the end of 90 minutes, before any extra time). Team names must match the
    dataset / ledger spelling. Missing or malformed file => no overrides."""
    if not KO_OVERRIDES.exists():
        return {}
    try:
        df = pd.read_csv(KO_OVERRIDES, dtype=str).fillna("")
    except Exception:
        return {}
    out = {}
    for r in df.itertuples(index=False):
        s = str(getattr(r, "score90", "")).strip()
        if "-" not in s:
            continue
        try:
            hs, as_ = (int(x) for x in s.split("-", 1))
        except ValueError:
            continue
        out[(str(r.date).strip(), str(r.home).strip(),
             str(r.away).strip())] = (hs, as_)
    return out


def grade(side, hs, as_):
    """Did a bet on `side` win, given the full-time score?
    Returns None for sides that can't be auto-graded (e.g. manually recorded
    outrights/specials) — those stay open for manual settlement."""
    return {"home": hs > as_, "draw": hs == as_, "away": hs < as_,
            "over25": hs + as_ >= 3, "under25": hs + as_ <= 2,
            "btts_yes": hs >= 1 and as_ >= 1,
            "btts_no": hs == 0 or as_ == 0}.get(side)


def settle(verbose=True):
    """Match open bets against played results; update pnl and bankroll."""
    ledger = _load_ledger()
    open_mask = ledger["status"] == "open"
    if not open_mask.any():
        if verbose:
            print(f"No open bets. Bankroll: £{current_bankroll():.2f}")
        return
    results = pd.read_csv(DATA / "results.csv")
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
    played = results.dropna(subset=["home_score", "away_score"])
    lookup = {(r.date, r.home_team, r.away_team): (r.home_score, r.away_score)
              for r in played.itertuples(index=False)}
    ko90 = _load_ko_overrides()   # 90-min knockout scores take priority

    bankroll = current_bankroll()
    peak = current_peak()
    settled = 0
    for i in ledger.index[open_mask]:
        key = (str(ledger.at[i, "match_date"]), ledger.at[i, "home"],
               ledger.at[i, "away"])
        if key in ko90:
            hs, as_ = ko90[key]      # settlement-correct 90-minute score
            src = " [90']"
        elif key in lookup:
            hs, as_ = lookup[key]    # results.csv score (FT, incl. extra time)
            src = ""
        else:
            continue
        won = grade(ledger.at[i, "side"], hs, as_)
        if won is None:
            continue  # not auto-gradable (manual outright/special)
        pnl = round(ledger.at[i, "stake"] * (ledger.at[i, "odds"] - 1), 2) \
            if won else -ledger.at[i, "stake"]
        bankroll = round(bankroll + pnl, 2)
        peak = max(peak, bankroll)
        ledger.at[i, "status"] = "won" if won else "lost"
        ledger.at[i, "pnl"] = pnl
        ledger.at[i, "bankroll_after"] = bankroll
        settled += 1
        if verbose:
            print(f"  {'WON ' if won else 'LOST'} {ledger.at[i, 'bet']} "
                  f"({int(hs)}-{int(as_)}){src}  stake £{ledger.at[i, 'stake']:.2f}"
                  f"  pnl £{pnl:+.2f}  bankroll £{bankroll:.2f}")
    ledger.to_csv(LEDGER, index=False)
    _save_bankroll(bankroll, peak)
    if verbose:
        still_open = (ledger["status"] == "open").sum()
        print(f"Settled {settled} bet(s); {still_open} still open. "
              f"Bankroll: £{bankroll:.2f}")


def status():
    ledger = _load_ledger()
    bankroll = current_bankroll()
    print(f"Bankroll: £{bankroll:.2f} (started £{START_BANKROLL:.2f})")
    if ledger.empty:
        print("No bets recorded yet.")
        return
    open_bets = ledger[ledger["status"] == "open"]
    closed = ledger[ledger["status"] != "open"]
    if not closed.empty:
        wins = (closed["status"] == "won").sum()
        print(f"Settled: {len(closed)} bets, {wins} won, "
              f"net £{closed['pnl'].sum():+.2f}")
        print(closed.tail(5)[["match_date", "bet", "odds", "stake",
                              "status", "pnl"]].to_string(index=False))
    if not open_bets.empty:
        print(f"\nOpen ({len(open_bets)} bets, £{open_bets['stake'].sum():.2f} "
              "at risk):")
        print(open_bets[["match_date", "bet", "odds", "stake"]]
              .to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--reset", type=float, metavar="AMOUNT")
    args = ap.parse_args()
    if args.reset is not None:
        _save_bankroll(args.reset, peak=args.reset)
        if LEDGER.exists():
            LEDGER.rename(LEDGER.with_suffix(".csv.bak"))
        print(f"Bankroll reset to £{args.reset:.2f} "
              "(old ledger kept as ledger.csv.bak)")
    elif args.settle:
        settle()
    else:
        status()


if __name__ == "__main__":
    main()
