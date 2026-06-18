#!/usr/bin/env python3
"""Replay the live betting strategy over World Cup 2022.

Exactly the rules the 2026 system uses: blend model (fitted only on
pre-tournament data), proportional de-vig, only bets with edge >= 3%,
best-EV outcome per match, quarter-Kelly stakes, compounding bankroll.
Bets settle on the 90-minute result (knockout draws at 90' = draw, as a
bookmaker's 1X2 market settles), day by day in tournament order.

Odds: data/wc2022_odds.csv — OddsPortal market-average closing odds.
"""
from fractions import Fraction

import numpy as np
import pandas as pd

from engines.worldcup.predictor import load_matches, compute_elo, fit_goal_model, expected_goals, HOME_ADV, DC_RHO
from engines.worldcup.dixoncoles import fit_dc, outcome_probs
from engines.worldcup.edge import devig, kelly, KELLY_FRACTION

# Minimum 1X2 edge to back an outcome in this historical replay. Defined locally
# because edge.py's live recorder was refactored to a confidence rule
# (BET_CONF_MIN) and no longer exports an edge threshold; 3% matches the project's
# documented "edge >= 3%" convention.
BET_EDGE_MIN = 0.03

START_BANKROLL = 100.0
NAMES = {"USA": "United States"}

# --- model, leak-free as of tournament start ---
played, _ = load_matches()
ratings, played = compute_elo(played)
train = played[played["date"] < "2022-11-20"]
beta = fit_goal_model(train)
dc = fit_dc(train, anchor="2022-11-20", verbose=False)
# point-in-time Elo: each team's rating as of the tournament start
ratings_cut, _ = compute_elo(train)

def blend_probs(home, away):
    le = expected_goals(ratings_cut[home], ratings_cut[away], beta, 0.0)
    ld = dc.lambdas(home, away)               # all matches in Qatar: neutral
    pe = np.array(outcome_probs(*le, DC_RHO)[:3])
    pdc = np.array(outcome_probs(*ld, dc.rho)[:3])
    return (pe + pdc) / 2

# --- replay ---
odds = pd.read_csv("data/wc2022_odds.csv")
frac = lambda s: float(Fraction(s)) + 1.0     # fractional -> decimal odds

bankroll = START_BANKROLL
rows = []
for date, day in odds.groupby("date", sort=True):
    bets = []
    for r in day.itertuples(index=False):
        home, away = NAMES.get(r.home, r.home), NAMES.get(r.away, r.away)
        book = [frac(r.odds_home), frac(r.odds_draw), frac(r.odds_away)]
        implied, _ = devig(book)
        model = blend_probs(home, away)
        best = None
        for i, side in enumerate(("home", "draw", "away")):
            edge = model[i] - implied[i]
            ev = model[i] * book[i] - 1.0
            if edge >= BET_EDGE_MIN and (best is None or ev > best[0]):
                best = (ev, side, book[i], model[i], edge, f"{r.home} v {r.away}", r.result90)
        if best:
            bets.append(best)
    # place all of today's bets from the same bankroll, then settle
    staked = [(b, round(KELLY_FRACTION * kelly(b[3], b[2]) * bankroll, 2)) for b in bets]
    for (ev, side, o, p, edge, match, result), stake in staked:
        if stake < 0.10:
            continue
        won = side == result
        pnl = round(stake * (o - 1), 2) if won else -stake
        bankroll = round(bankroll + pnl, 2)
        rows.append({"date": date, "match": match, "bet": side, "odds": round(o, 2),
                     "p_model": round(p, 3), "edge": round(edge, 3),
                     "stake": stake, "result": "won" if won else "lost",
                     "pnl": pnl, "bankroll": bankroll})

df = pd.DataFrame(rows)
df.to_csv("wc2022_replay.csv", index=False)
pd.set_option("display.width", 150)
print(df.to_string(index=False))
won = (df.result == "won").sum()
total_staked = df.stake.sum()
print(f"\nBets placed: {len(df)} ({won} won, {len(df) - won} lost)")
print(f"Total staked: £{total_staked:.2f} | Net P&L: £{df.pnl.sum():+.2f}")
print(f"Final bankroll: £{bankroll:.2f} (from £{START_BANKROLL:.2f}, "
      f"{(bankroll / START_BANKROLL - 1) * 100:+.1f}%)")
print(f"Yield on turnover: {df.pnl.sum() / total_staked * 100:+.1f}%")
