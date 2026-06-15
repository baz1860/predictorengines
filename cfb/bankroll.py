#!/usr/bin/env python3
"""Bankroll tracking for the CFB edge finder.

Stakes are sized from a live bankroll (data/bankroll.json, starts £100).
edge.py logs recommendations to data/ledger.csv; settle them once results land
in data/games.csv (run fetch_data.py first).

  python3 bankroll.py            # status: bankroll, open bets, P&L
  python3 bankroll.py --settle   # settle open bets against games.csv
  python3 bankroll.py --reset 100
"""
import argparse
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER_CSV = os.path.join(HERE, "data", "ledger.csv")
BANKROLL_JSON = os.path.join(HERE, "data", "bankroll.json")
GAMES_CSV = os.path.join(HERE, "data", "games.csv")


def load_bankroll():
    if os.path.exists(BANKROLL_JSON):
        with open(BANKROLL_JSON) as f:
            return json.load(f)["bankroll"]
    return 100.0


def save_bankroll(v):
    with open(BANKROLL_JSON, "w") as f:
        json.dump({"bankroll": round(v, 2)}, f)


def settle_bet(bet, games):
    """Return pnl or None if game not found yet."""
    g = games[(games["home_team"] == bet["home"]) & (games["away_team"] == bet["away"])
              & (games["date"].astype(str) >= str(bet["date"]))]
    if g.empty:
        return None
    g = g.iloc[0]
    margin = g["home_points"] - g["away_points"]
    total = g["home_points"] + g["away_points"]
    line = float(bet["line"]) if pd.notna(bet["line"]) else None
    m, s = bet["market"], bet["side"]
    if m == "ml":
        won = (margin > 0) if s == "home" else (margin < 0)
        push = margin == 0
    elif m == "spread":
        adj = margin + line if s == "home" else -margin + line
        won, push = adj > 0, adj == 0
    elif m == "total":
        won = (total > line) if s == "over" else (total < line)
        push = total == line
    else:
        return None
    stake = float(bet["stake"])
    if push:
        return 0.0
    return stake * (float(bet["odds"]) - 1.0) if won else -stake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--reset", type=float, default=None)
    args = ap.parse_args()

    if args.reset is not None:
        save_bankroll(args.reset)
        print(f"bankroll reset to £{args.reset:.2f}")
        return

    bankroll = load_bankroll()
    if not os.path.exists(LEDGER_CSV):
        print(f"bankroll £{bankroll:.2f} | no bets logged yet")
        return
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


if __name__ == "__main__":
    main()
