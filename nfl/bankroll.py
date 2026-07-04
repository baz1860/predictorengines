#!/usr/bin/env python3
"""Bankroll tracking for the NFL edge finder (cfb/bankroll.py conventions).

Stakes are sized from a live bankroll (data/bankroll.json, starts £100).
edge.py logs recommendations to data/ledger.csv; settle them once results
land in data/games.csv (run fetch_data.py first).

  python3 -m nfl.bankroll            # status: bankroll, open bets, P&L
  python3 -m nfl.bankroll --settle   # settle open bets against games.csv
  python3 -m nfl.bankroll --reset 100
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER_CSV = os.path.join(HERE, "data", "ledger.csv")
BANKROLL_JSON = os.path.join(HERE, "data", "bankroll.json")
GAMES_CSV = os.path.join(HERE, "data", "games.csv")


def load_bankroll() -> float:
    if os.path.exists(BANKROLL_JSON):
        with open(BANKROLL_JSON) as f:
            return json.load(f)["bankroll"]
    return 100.0


def save_bankroll(v: float) -> None:
    with open(BANKROLL_JSON, "w") as f:
        json.dump({"bankroll": round(v, 2)}, f)


def settle_bet(bet, games: pd.DataFrame) -> float | None:
    """Return pnl, or None if the game hasn't been played/recorded yet.
    `line` in the ledger is in odds.csv's TRADITIONAL convention (negative =
    home favorite) — matches how edge.py wrote it."""
    g = games[(games["home"] == bet["home"]) & (games["away"] == bet["away"])
             & (games["date"].astype(str) >= str(bet["date"]))]
    if g.empty:
        return None
    g = g.iloc[0]
    margin = g["home_score"] - g["away_score"]
    total = g["home_score"] + g["away_score"]
    line = float(bet["line"]) if pd.notna(bet["line"]) and bet["line"] != "" else None
    m, s = bet["market"], bet["side"]
    if m == "ml":
        won = (margin > 0) if s == "home" else (margin < 0)
        push = margin == 0
    elif m == "spread":
        # traditional convention: home covers -L if margin > -L; away covers
        # +L if margin < L (L is the away-side traditional number, the
        # negative mirror of the home one).
        if line is None:
            return None
        adj = (margin + line) if s == "home" else (-margin + line)
        won, push = adj > 0, adj == 0
    elif m == "total":
        if line is None:
            return None
        won = (total > line) if s == "over" else (total < line)
        push = total == line
    else:
        return None
    stake = float(bet["stake"])
    if push:
        return 0.0
    return stake * (float(bet["odds"]) - 1.0) if won else -stake


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--reset", type=float, default=None)
    args = ap.parse_args()

    if args.reset is not None:
        save_bankroll(args.reset)
        print(f"bankroll reset to £{args.reset:.2f}")
        return 0

    bankroll = load_bankroll()
    if not os.path.exists(LEDGER_CSV):
        print(f"bankroll £{bankroll:.2f} | no bets logged yet")
        return 0
    led = pd.read_csv(LEDGER_CSV)

    if args.settle:
        games = pd.read_csv(GAMES_CSV)
        n = 0
        for i, bet in led[led["status"] == "open"].iterrows():
            pnl = settle_bet(bet, games)
            if pnl is None:
                continue
            led.loc[i, "status"] = "won" if pnl > 0 else ("push" if pnl == 0 else "lost")
            led.loc[i, "pnl"] = round(pnl, 2)
            bankroll += pnl
            n += 1
        led.to_csv(LEDGER_CSV, index=False)
        save_bankroll(bankroll)
        print(f"settled {n} bet(s)")

    closed = led[led["status"] != "open"]
    open_ = led[led["status"] == "open"]
    pnl = pd.to_numeric(closed["pnl"], errors="coerce").sum()
    print(f"bankroll £{bankroll:.2f} | open bets: {len(open_)} (£{open_['stake'].sum():.2f} staked) | "
         f"settled: {len(closed)}, P&L £{pnl:+.2f}")
    if len(open_):
        print(open_[["date", "home", "away", "market", "side", "line", "odds", "stake"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
