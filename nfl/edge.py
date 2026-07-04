#!/usr/bin/env python3
"""Edge finder for NFL moneyline, spread, and total markets.

All probabilities are priced through the empirical margin/total PMF
(margin_dist.py) — never a normal approximation. Kelly staking is done on
the win/push/loss TRINOMIAL (not the binary win/loss formula), since pushes
at 3 and 7 are common enough to matter for stake sizing.

Sign convention note: odds.csv uses the TRADITIONAL bookmaker spread
convention (negative = home favorite, e.g. "-3.5"), matching what a
sportsbook screen shows and what cfb/odds.csv uses — NOT margin_dist.py's
internal `spread_line` convention (positive = home favorite), which is a
data-source artifact of nflverse. Conversion happens once, at the CSV
boundary (see `_internal_spread_line`).

Fill odds.csv with decimal odds (both sides of a market when possible — enables
proper de-vig), then run.

Usage:
  python3 -m nfl.edge --template     # write odds.csv from upcoming games
  python3 -m nfl.edge                # edge report -> edge_report.csv, auto-log bets
  python3 -m nfl.edge --no-bet       # report only, don't touch the ledger
  python3 -m nfl.edge --bankroll 250 # override bankroll for stake sizing
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import date

import pandas as pd

from . import margin_dist as MD
from .predictor import Models, blend_predict

HERE = os.path.dirname(os.path.abspath(__file__))
ODDS_CSV = os.path.join(HERE, "odds.csv")
REPORT_CSV = os.path.join(HERE, "edge_report.csv")
LEDGER_CSV = os.path.join(HERE, "data", "ledger.csv")
BANKROLL_JSON = os.path.join(HERE, "data", "bankroll.json")
UPCOMING_CSV = os.path.join(HERE, "data", "upcoming.csv")

MIN_EDGE = 0.03
KELLY_FRACTION = 0.25
DEFAULT_OVERROUND = 1.045

HEADER = ["season", "week", "date", "home", "away", "neutral", "market", "side", "line", "odds"]


def american_to_decimal(a) -> float | None:
    if a is None or pd.isna(a):
        return None
    a = float(a)
    if abs(a) < 100.0:
        return None
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def get_bankroll() -> float:
    if os.path.exists(BANKROLL_JSON):
        with open(BANKROLL_JSON) as f:
            return json.load(f)["bankroll"]
    return 100.0


def write_template() -> None:
    rows = []
    if os.path.exists(UPCOMING_CSV):
        up = pd.read_csv(UPCOMING_CSV, parse_dates=["date"])
        if not up.empty:
            up = up[up["date"] <= up["date"].min() + pd.Timedelta(days=7)]
            for r in up.itertuples():
                base = [r.season, r.week, str(r.date.date()), r.home, r.away, int(bool(r.neutral))]
                ml_h = american_to_decimal(r.home_moneyline)
                ml_a = american_to_decimal(r.away_moneyline)
                sp_h = american_to_decimal(r.home_spread_odds)
                sp_a = american_to_decimal(r.away_spread_odds)
                tot_o = american_to_decimal(r.over_odds)
                tot_u = american_to_decimal(r.under_odds)
                # spread_line (nflverse) is positive when home is favored; the
                # CSV uses the traditional convention (negative = home fav).
                sline = None if pd.isna(r.spread_line) else -float(r.spread_line)
                tline = None if pd.isna(r.total_line) else float(r.total_line)
                rows += [
                    base + ["ml", "home", "", ml_h or ""], base + ["ml", "away", "", ml_a or ""],
                    base + ["spread", "home", sline if sline is not None else "", sp_h or ""],
                    base + ["spread", "away", (-sline if sline is not None else ""), sp_a or ""],
                    base + ["total", "over", tline if tline is not None else "", tot_o or ""],
                    base + ["total", "under", tline if tline is not None else "", tot_u or ""],
                ]
    if not rows:
        base = [date.today().year, 1, str(date.today()), "Kansas City Chiefs", "Buffalo Bills", 0]
        rows = [base + ["ml", "home", "", 1.45], base + ["ml", "away", "", 2.90],
                base + ["spread", "home", -2.5, 1.91], base + ["spread", "away", 2.5, 1.91],
                base + ["total", "over", 47.5, 1.91], base + ["total", "under", 47.5, 1.91]]
        print("note: no upcoming fixtures in data/upcoming.csv (run fetch_data.py in season) — "
             "wrote sample rows; edit teams/odds by hand")
    with open(ODDS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"wrote {ODDS_CSV} ({len(rows)} rows) — fill in blank lines/odds, blanks are skipped")


def _internal_spread_line(side: str, traditional_line: float) -> float:
    """odds.csv traditional convention (negative = home favorite) ->
    margin_dist.py's internal convention (positive = home favorite, matching
    nflverse spread_line). Home-side traditional line negates directly;
    away-side traditional line (the mirror number, e.g. +3.5 when home is
    -3.5) already equals the internal home-perspective spread_line."""
    return -traditional_line if side == "home" else traditional_line


def model_probs(pred: dict, pmf: dict, market: str, side: str, line: float | None) -> dict:
    """Returns {p_win, p_push} for the given market/side/line (line in the
    odds.csv traditional convention for spreads)."""
    if market == "ml":
        ml = MD.moneyline_probs(pmf, pred["margin"])
        return {"p_win": ml["home_win"] if side == "home" else ml["away_win"], "p_push": ml["tie"]}
    if market == "spread":
        internal_line = _internal_spread_line(side, line)
        c = MD.cover_probs(pmf, pred["margin"], internal_line)
        return ({"p_win": c["home_cover"], "p_push": c["push"]} if side == "home"
               else {"p_win": c["away_cover"], "p_push": c["push"]})
    if market == "total":
        t = MD.total_probs(pmf, pred["total"], line)
        return {"p_win": t["over"] if side == "over" else t["under"], "p_push": t["push"]}
    raise ValueError(market)


def kelly_trinomial(p_win: float, p_push: float, odds_decimal: float) -> float:
    """Kelly fraction on the win/push/loss trinomial (push returns stake,
    contributes neither gain nor loss). b = net decimal odds (odds - 1)."""
    b = odds_decimal - 1.0
    p_loss = max(0.0, 1.0 - p_win - p_push)
    denom = b * (1.0 - p_push)
    if denom <= 0:
        return 0.0
    f = (p_win * b - p_loss) / denom
    return max(0.0, f)


def half_point_report(pred: dict, pmf: dict, line: float) -> list[dict]:
    """Fair (de-vig) price of the adjacent half-point lines around `line`,
    home side — the practical payoff of the key-number PMF work."""
    out = []
    for l in (line - 0.5, line, line + 0.5):
        c = MD.cover_probs(pmf, pred["margin"], l)
        fair_odds = (1.0 / c["home_cover"]) if c["home_cover"] > 0 else None
        out.append({"line": l, "p_home_cover": round(c["home_cover"], 4),
                    "push": round(c["push"], 4),
                    "fair_decimal_odds": round(fair_odds, 3) if fair_odds else None})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--no-bet", action="store_true")
    ap.add_argument("--bankroll", type=float, default=None)
    ap.add_argument("--half-points", action="store_true", help="also print the half-point value report")
    ap.add_argument("--min-edge", type=float, default=None,
                    help=f"minimum edge to log, in percentage points (default {MIN_EDGE * 100:g})")
    args = ap.parse_args()
    min_edge = args.min_edge / 100.0 if args.min_edge is not None else MIN_EDGE

    if args.template:
        write_template()
        return 0

    if not os.path.exists(ODDS_CSV):
        raise SystemExit("no odds.csv — run `python3 -m nfl.edge --template` first")
    odds = pd.read_csv(ODDS_CSV)
    odds = odds[odds["odds"].notna() & (odds["odds"] != "")]
    if odds.empty:
        raise SystemExit("odds.csv has no filled-in odds")
    odds["odds"] = odds["odds"].astype(float)
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")

    models = Models.load()
    bankroll = args.bankroll if args.bankroll is not None else get_bankroll()

    def key(r):
        line_key = "" if r["market"] == "ml" else round(abs(r["line"]), 1)
        return (r["home"], r["away"], r["market"], line_key)

    odds["pairkey"] = odds.apply(key, axis=1)
    inv_sum = odds.groupby("pairkey")["odds"].apply(lambda s: (1.0 / s).sum())
    sides_per_key = odds.groupby("pairkey")["odds"].size()

    report = []
    half_points = []
    for r in odds.itertuples():
        try:
            pred = blend_predict(models, r.home, r.away, neutral=bool(r.neutral),
                                 season=int(r.season), week=int(r.week))
        except Exception:
            continue
        line = None if pd.isna(r.line) else float(r.line)
        if r.market != "ml" and line is None:
            continue
        probs = model_probs(pred, models.pmf, r.market, r.side, line)
        p_model = probs["p_win"]
        n_sides = int(sides_per_key[r.pairkey])
        over = float(inv_sum[r.pairkey]) if n_sides == 2 else DEFAULT_OVERROUND
        p_imp = (1.0 / r.odds) / over
        edge = p_model - p_imp
        p_loss = max(0.0, 1.0 - p_model - probs["p_push"])
        ev = p_model * (r.odds - 1.0) - p_loss
        kelly = kelly_trinomial(p_model, probs["p_push"], r.odds)
        stake = round(KELLY_FRACTION * kelly * bankroll, 2)
        line_str = "" if line is None else f"{line:+g}"
        report.append({
            "season": r.season, "week": r.week, "date": r.date, "home": r.home, "away": r.away,
            "market": r.market, "side": r.side, "line": line, "odds": r.odds,
            "p_model": round(p_model, 4), "p_push": round(probs["p_push"], 4),
            "p_implied": round(p_imp, 4), "edge": round(edge, 4), "ev_per_unit": round(ev, 4),
            "kelly_frac": round(KELLY_FRACTION * kelly, 4), "stake": stake,
        })
        if args.half_points and r.market == "spread" and line is not None:
            hp_line = _internal_spread_line(r.side, line)
            for hp in half_point_report(pred, models.pmf, hp_line):
                half_points.append({"home": r.home, "away": r.away, **hp})

    rep = pd.DataFrame(report).sort_values("edge", ascending=False)
    rep.to_csv(REPORT_CSV, index=False)
    with pd.option_context("display.width", 200):
        print(rep.to_string(index=False))
    print(f"\nbankroll £{bankroll:.2f} | quarter-Kelly (trinomial) | edges under ~3% are model noise")
    print(f"report -> {REPORT_CSV}")

    if half_points:
        print("\nHalf-point value report (home side):")
        hp_df = pd.DataFrame(half_points).drop_duplicates()
        print(hp_df.to_string(index=False))

    if not args.no_bet:
        bets = rep[rep["edge"] >= min_edge]
        if not bets.empty:
            bets = bets.loc[bets.groupby(["home", "away", "market"])["edge"].idxmax()]
        if bets.empty:
            print("no bets logged (no edge >= 3%)")
            return 0
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
