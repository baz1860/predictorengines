#!/usr/bin/env python3
"""Edge finder for CFB moneyline, spread, and totals markets.

Fill odds.csv with decimal odds from your bookmaker (both sides of a market
when possible — enables proper vig removal), then run. For each quote the
bookmaker's overround is removed, the implied probability is compared to the
blended model's, and edge, EV per unit, and a quarter-Kelly stake are reported.

Usage:
  python3 edge.py --template     # write odds.csv (upcoming fixtures if known)
  python3 edge.py                # edge report -> edge_report.csv, auto-log bets
  python3 edge.py --no-bet       # report only, don't touch the ledger
  python3 edge.py --bankroll 250 # override bankroll for stake sizing
"""
import argparse
import csv
import json
import math
import os
from datetime import date

import pandas as pd

import elo as E
import power as P
from predictor import blend_predict

HERE = os.path.dirname(os.path.abspath(__file__))
ODDS_CSV = os.path.join(HERE, "odds.csv")
REPORT_CSV = os.path.join(HERE, "edge_report.csv")
LEDGER_CSV = os.path.join(HERE, "data", "ledger.csv")
BANKROLL_JSON = os.path.join(HERE, "data", "bankroll.json")
UPCOMING_CSV = os.path.join(HERE, "data", "upcoming.csv")

MIN_EDGE = 0.03
KELLY_FRACTION = 0.25
DEFAULT_OVERROUND = 1.045  # assumed when only one side of a market is quoted

HEADER = ["date", "home", "away", "neutral", "market", "side", "line", "odds"]


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def get_bankroll():
    if os.path.exists(BANKROLL_JSON):
        with open(BANKROLL_JSON) as f:
            return json.load(f)["bankroll"]
    return 100.0


def write_template():
    rows = []
    if os.path.exists(UPCOMING_CSV):
        up = pd.read_csv(UPCOMING_CSV, parse_dates=["date"])
        up = up[(up["home_div"] == "fbs") & (up["away_div"] == "fbs")]
        up = up[up["date"] <= up["date"].min() + pd.Timedelta(days=7)]
        for r in up.itertuples():
            base = [str(r.date.date()), r.home_team, r.away_team, int(bool(r.neutral))]
            rows += [base + ["ml", "home", "", ""], base + ["ml", "away", "", ""],
                     base + ["spread", "home", "", ""], base + ["spread", "away", "", ""],
                     base + ["total", "over", "", ""], base + ["total", "under", "", ""]]
    if not rows:  # no upcoming schedule yet — show format
        base = [str(date.today()), "Ohio State", "Michigan", 0]
        rows = [base + ["ml", "home", "", 1.45], base + ["ml", "away", "", 2.90],
                base + ["spread", "home", -6.5, 1.91], base + ["spread", "away", 6.5, 1.91],
                base + ["total", "over", 48.5, 1.91], base + ["total", "under", 48.5, 1.91]]
        print("note: no upcoming fixtures in data/upcoming.csv (run fetch_data.py in season) — "
              "wrote sample rows; edit teams/odds by hand")
    with open(ODDS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"wrote {ODDS_CSV} ({len(rows)} rows) — fill in lines and decimal odds, blanks are skipped")


def model_prob(pred, pparams, market, side, line):
    m, t = pred["margin"], pred["total"]
    s_m, s_t = pparams["sigma"], pparams["sigma_total"]
    if market == "ml":
        return pred["p1"] if side == "home" else 1.0 - pred["p1"]
    if market == "spread":
        if side == "home":   # home line L (e.g. -6.5): covers if margin + L > 0
            return 1.0 - phi((-line - m) / s_m)
        return phi((line - m) / s_m)  # away +L: covers if margin < L
    if market == "total":
        if side == "over":
            return 1.0 - phi((line - t) / s_t)
        return phi((line - t) / s_t)
    raise ValueError(market)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--no-bet", action="store_true")
    ap.add_argument("--bankroll", type=float, default=None)
    args = ap.parse_args()

    if args.template:
        write_template()
        return

    if not os.path.exists(ODDS_CSV):
        raise SystemExit("no odds.csv — run `python3 edge.py --template` first")
    odds = pd.read_csv(ODDS_CSV)
    odds = odds[odds["odds"].notna() & (odds["odds"] != "")]
    if odds.empty:
        raise SystemExit("odds.csv has no filled-in odds")
    odds["odds"] = odds["odds"].astype(float)
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")

    eparams = E.build()
    pparams = P.load_params()
    bankroll = args.bankroll if args.bankroll is not None else get_bankroll()

    # pair key for vig removal: both sides of the same market share a key
    def key(r):
        line_key = "" if r["market"] == "ml" else round(abs(r["line"]), 1)
        return (r["home"], r["away"], r["market"], line_key)

    odds["_k"] = odds.apply(key, axis=1)
    inv_sum = odds.groupby("_k")["odds"].apply(lambda s: (1.0 / s).sum())

    report = []
    for r in odds.itertuples():
        pred = blend_predict(eparams, pparams, r.home, r.away, neutral=bool(r.neutral))
        line = None if pd.isna(r.line) else float(r.line)
        if r.market != "ml" and line is None:
            continue
        p_model = model_prob(pred, pparams, r.market, r.side, line)
        n_sides = (odds["_k"] == r._k).sum() if hasattr(r, "_k") else 1
        over = inv_sum[r._k] if n_sides == 2 else DEFAULT_OVERROUND
        p_imp = (1.0 / r.odds) / over
        edge = p_model - p_imp
        ev = p_model * r.odds - 1.0
        kelly = max(0.0, (p_model * r.odds - 1.0) / (r.odds - 1.0))
        stake = round(KELLY_FRACTION * kelly * bankroll, 2)
        report.append({
            "date": r.date, "home": r.home, "away": r.away, "market": r.market,
            "side": r.side, "line": line, "odds": r.odds, "p_model": round(p_model, 4),
            "p_implied": round(p_imp, 4), "edge": round(edge, 4),
            "ev_per_unit": round(ev, 4), "stake": stake,
        })

    rep = pd.DataFrame(report).sort_values("edge", ascending=False)
    rep.to_csv(REPORT_CSV, index=False)
    with pd.option_context("display.width", 200):
        print(rep.to_string(index=False))
    print(f"\nbankroll £{bankroll:.2f} | quarter-Kelly | edges under ~3% are model noise")
    print(f"report -> {REPORT_CSV}")

    if not args.no_bet:
        bets = rep[rep["edge"] >= MIN_EDGE]
        bets = bets.loc[bets.groupby(["home", "away", "market"])["edge"].idxmax()]
        if bets.empty:
            print("no bets logged (no edge >= 3%)")
            return
        os.makedirs(os.path.dirname(LEDGER_CSV), exist_ok=True)
        new = not os.path.exists(LEDGER_CSV)
        with open(LEDGER_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["placed", "date", "home", "away", "market", "side", "line",
                            "odds", "stake", "p_model", "edge", "status", "pnl"])
            for b in bets.itertuples():
                w.writerow([str(date.today()), b.date, b.home, b.away, b.market, b.side,
                            b.line, b.odds, b.stake, b.p_model, b.edge, "open", ""])
        print(f"logged {len(bets)} bet(s) to ledger (use --no-bet to skip)")


if __name__ == "__main__":
    main()
