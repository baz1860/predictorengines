#!/usr/bin/env python3
"""Counterfactual draw backtest over World Cup 2022.

Question: if we had bet every DRAW the model flags as positive-edge, would it
have made money? Uses the exact leak-free blend model from wc2022_replay.py
(fitted only on pre-2022-11-20 data), the same proportional de-vig, and the
OddsPortal market-average odds in data/wc2022_odds.csv (which DO carry a draw
price, unlike the live odds_history snapshots).

Reports realized ROI for +edge draws at flat and quarter-Kelly stakes, with
all-draws and the model's non-draw picks as reference points.

CLV is intentionally NOT reported: no opening/closing draw-odds snapshots exist
in the data (the live snapshotter never stored draw prices), so CLV cannot be
computed honestly. Stated in the printout rather than faked.
"""
from fractions import Fraction
import numpy as np
import pandas as pd

from engines.worldcup.predictor import load_matches, compute_elo, fit_goal_model, expected_goals, DC_RHO
from engines.worldcup.dixoncoles import fit_dc, outcome_probs
from engines.worldcup.edge import devig, kelly, KELLY_FRACTION

BET_EDGE_MIN = 0.03   # same 3% threshold the live system documents
NAMES = {"USA": "United States"}

# --- model, leak-free as of tournament start (mirrors wc2022_replay.py) ---
played, _ = load_matches()
ratings, played = compute_elo(played)
train = played[played["date"] < "2022-11-20"]
beta = fit_goal_model(train)
dc = fit_dc(train, anchor="2022-11-20", verbose=False)
ratings_cut, _ = compute_elo(train)


def blend_probs(home, away):
    le = expected_goals(ratings_cut[home], ratings_cut[away], beta, 0.0)
    ld = dc.lambdas(home, away)
    pe = np.array(outcome_probs(*le, DC_RHO)[:3])
    pdc = np.array(outcome_probs(*ld, dc.rho)[:3])
    return (pe + pdc) / 2   # order: home, draw, away


frac = lambda s: float(Fraction(s)) + 1.0

odds = pd.read_csv("data/wc2022_odds.csv")
rows = []
for r in odds.itertuples(index=False):
    home, away = NAMES.get(r.home, r.home), NAMES.get(r.away, r.away)
    book = [frac(r.odds_home), frac(r.odds_draw), frac(r.odds_away)]
    implied, overround = devig(book)
    model = blend_probs(home, away)
    is_draw = (r.result90 == "draw")
    for i, side in enumerate(("home", "draw", "away")):
        rows.append(dict(
            match=f"{r.home} v {r.away}", side=side,
            odds=round(book[i], 2),
            p_model=round(float(model[i]), 3),
            p_book=round(float(implied[i]), 3),
            edge=round(float(model[i] - implied[i]), 3),
            ev=round(float(model[i] * book[i] - 1.0), 3),
            won=(side == r.result90),
            overround=round(float(overround), 3),
        ))

df = pd.DataFrame(rows)
draws = df[df.side == "draw"].copy()


def summarize(g, label, stake_mode="flat"):
    if g.empty:
        print(f"  {label}: no bets")
        return
    if stake_mode == "flat":
        stake = pd.Series(1.0, index=g.index)
    else:  # quarter-Kelly on model prob at the taken odds
        stake = g.apply(lambda x: KELLY_FRACTION * kelly(x.p_model, x.odds), axis=1)
    pnl = np.where(g.won, stake * (g.odds - 1.0), -stake)
    staked = stake.sum()
    n = len(g)
    strike = g.won.mean()
    roi = pnl.sum() / staked if staked else float("nan")
    print(f"  {label}: {n} bets | strike {strike:.0%} | "
          f"avg draw odds {g.odds.mean():.2f} | staked {staked:.2f}u | "
          f"P&L {pnl.sum():+.2f}u | ROI {roi:+.1%}")


print("=== WC2022 DRAW BACKTEST (leak-free blend model) ===\n")
print(f"Matches: {len(odds)} | actual draws: {(odds.result90=='draw').sum()} "
      f"({(odds.result90=='draw').mean():.0%})\n")

pos = draws[draws.edge >= BET_EDGE_MIN]
print("DRAWS the model flags as positive-edge (edge >= 3%):")
summarize(pos, "flat 1u stake ", "flat")
summarize(pos, "quarter-Kelly ", "kelly")

print("\nReference baselines:")
summarize(draws, "ALL draws (flat)        ", "flat")
summarize(draws[draws.edge > 0], "any +edge draws (flat)  ", "flat")

# the model's actual non-draw value picks, for context
nd = df[(df.side != "draw") & (df.edge >= BET_EDGE_MIN)].copy()
# best-EV pick per match among home/away (closer to what the live system bets)
summarize(nd, "all +edge home/away (flat)", "flat")

print("\n--- the +edge draws in detail ---")
if not pos.empty:
    print(pos[["match", "odds", "p_model", "p_book", "edge", "ev", "won"]]
          .to_string(index=False))

print("\nCLV: NOT COMPUTED — no opening/closing draw-odds snapshots exist in the "
      "data.\n     The live snapshotter only stored home/away/under25, never the "
      "draw price,\n     so closing-line value on draws cannot be measured "
      "without fabricating it.")

pos.to_csv("draw_backtest_detail.csv", index=False)
