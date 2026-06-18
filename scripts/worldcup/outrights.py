#!/usr/bin/env python3
"""Compare simulated tournament probabilities against bookmaker outright odds.

Uses tournament_odds.csv (written by simulate.py / update.sh) for the model
side. Bookmaker odds go in outright_odds.csv:

  python3 outrights.py --template   # write outright_odds.csv (48 teams)
  <fill in decimal odds for "to win" and/or "to reach the final">
  python3 outrights.py              # edge report -> outright_edge.csv

Notes on outright markets:
- Overrounds are far larger than 1X2 (often 15-40%), and the
  favourite-longshot bias means longshot prices are systematically poor even
  after vig removal. Edges are flagged STRONG only at >= 5 percentage points
  (vs 3% for match markets), and nothing here is auto-recorded in the ledger
  — outrights tie up the stake for weeks, which distorts Kelly sizing on the
  daily match bets. Record one manually if you must.
- Devig assumes you've filled in odds for ALL teams with a realistic chance;
  a partially filled column understates the overround and inflates edges.
  The report prints the implied total so you can sanity-check coverage.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
SIM_CSV = HERE / "tournament_odds.csv"
ODDS_CSV = HERE / "outright_odds.csv"
REPORT = HERE / "outright_edge.csv"
STRONG_EDGE = 0.05   # outright margins are huge; below this is noise

# market: (column in tournament_odds.csv, odds column, winners per tournament)
MARKETS = {
    "champion": ("champion", "odds_champion", 1.0),
    "reach_final": ("reach_final", "odds_reach_final", 2.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true",
                    help="write outright_odds.csv and exit")
    args = ap.parse_args()

    if not SIM_CSV.exists():
        sys.exit("tournament_odds.csv not found — run update.sh first.")
    sims = pd.read_csv(SIM_CSV)

    if args.template or not ODDS_CSV.exists():
        tmpl = sims[["team"]].copy()
        tmpl["odds_champion"] = ""
        tmpl["odds_reach_final"] = ""
        tmpl.to_csv(ODDS_CSV, index=False)
        print(f"Wrote {len(tmpl)} teams to {ODDS_CSV.name}. Fill in decimal "
              "odds (any subset of teams/markets), then run: "
              "python3 outrights.py")
        return

    odds = pd.read_csv(ODDS_CSV)
    merged = sims.merge(odds, on="team", how="left")

    rows = []
    for market, (sim_col, odds_col, n_winners) in MARKETS.items():
        if odds_col not in merged.columns:
            continue
        m = merged[pd.to_numeric(merged[odds_col], errors="coerce") > 1.0].copy()
        if m.empty:
            continue
        o = m[odds_col].astype(float)
        implied_raw = 1.0 / o
        total = implied_raw.sum()
        overround = total - n_winners
        if overround <= 0:
            print(f"[{market}] implied probabilities sum to {total:.2f} "
                  f"(< {n_winners:g}): odds look incomplete, devig skipped — "
                  "edges will be overstated.")
            implied = implied_raw
        else:
            implied = implied_raw * (n_winners / total)
        for team, p_model, p_book, oi in zip(m["team"], m[sim_col].astype(float),
                                             implied, o):
            edge = p_model - p_book
            rows.append({"market": market, "team": team, "odds": oi,
                         "p_book": round(p_book, 4),
                         "p_model": round(p_model, 4),
                         "edge": round(edge, 4),
                         "ev_per_unit": round(p_model * oi - 1.0, 3),
                         "strong": "STRONG" if edge >= STRONG_EDGE else ""})
        print(f"[{market}] {len(m)} teams priced, implied total "
              f"{total:.2f} (overround {overround:+.1%} vs {n_winners:g})")

    if not rows:
        sys.exit(f"No usable odds in {ODDS_CSV.name}. Fill in decimal odds "
                 "(> 1.0) or run with --template to regenerate it.")

    df = pd.DataFrame(rows).sort_values("ev_per_unit", ascending=False)
    df.to_csv(REPORT, index=False)
    pd.set_option("display.width", 160)
    print(f"\nTop edges (STRONG = edge >= {STRONG_EDGE:.0%}; outright margins "
          "make smaller edges unreliable):\n")
    print(df.head(15).to_string(index=False))
    print(f"\nFull report -> {REPORT.name}")
    print("Not auto-recorded in the ledger: outrights lock up bankroll for "
          "weeks and skew daily Kelly sizing. Record manually if convinced.")


if __name__ == "__main__":
    main()
