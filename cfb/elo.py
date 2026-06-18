#!/usr/bin/env python3
"""Elo ratings for FBS college football.

Margin-of-victory-scaled K, home-field advantage, between-season regression to
the mean. All non-FBS opponents are pooled into one 'FCS' pseudo-team. Spread
mapping (Elo points per point of margin) and margin sigma are fitted from data.

Usage:
  python3 elo.py "Ohio State" "Michigan"            # team 1 at home
  python3 elo.py "Ohio State" "Michigan" --neutral
  python3 elo.py --ratings                          # top 30
"""
import argparse
import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES_CSV = os.path.join(HERE, "data", "games.csv")

K = 35.0
HFA_ELO = 62.0          # home-field advantage in Elo points
START_ELO = 1500.0
NEW_TEAM_ELO = 1300.0   # FBS newcomers / transitioning programs
SEASON_REGRESS = 0.30   # fraction regressed to 1500 between seasons
FCS = "FCS"             # pseudo-team for all non-FBS opponents


def load_games(path=GAMES_CSV):
    g = pd.read_csv(path, parse_dates=["date"])
    g["home"] = g.apply(lambda r: r["home_team"] if r["home_div"] == "fbs" else FCS, axis=1)
    g["away"] = g.apply(lambda r: r["away_team"] if r["away_div"] == "fbs" else FCS, axis=1)
    return g


def win_prob(elo_diff):
    """P(team with +elo_diff wins)."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def mov_multiplier(margin, elo_diff_winner):
    return math.log(abs(margin) + 1.0) * 2.2 / (elo_diff_winner * 0.001 + 2.2)


def run_elo(games, record_pregame=False, carry=None, prior_offsets=None):
    """Run Elo through all games chronologically.

    carry: between-season carryover of (rating - 1500); default 1 - SEASON_REGRESS.
    prior_offsets: dict[(team, season)] -> Elo points added at the team's first
    game of that season (preseason talent / returning-production priors).
    Returns (ratings, history) where history is a list of pregame-rating rows
    (only if record_pregame), aligned with games' row order.
    """
    if carry is None:
        carry = 1.0 - SEASON_REGRESS
    prior_offsets = prior_offsets or {}
    ratings, last_season = {}, {}
    history = []
    for r in games.itertuples():
        h, a = r.home, r.away
        for t in (h, a):
            if t not in ratings:
                base = START_ELO if t == FCS else NEW_TEAM_ELO
                ratings[t] = base + prior_offsets.get((t, r.season), 0.0)
                last_season[t] = r.season
            elif last_season[t] != r.season:
                ratings[t] = (START_ELO + carry * (ratings[t] - START_ELO)
                              + prior_offsets.get((t, r.season), 0.0))
                last_season[t] = r.season
        hfa = 0.0 if r.neutral else HFA_ELO
        diff = ratings[h] + hfa - ratings[a]
        if record_pregame:
            history.append((ratings[h], ratings[a], diff))
        p_home = win_prob(diff)
        margin = r.home_points - r.away_points
        result = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        if margin != 0:
            elo_diff_winner = diff if margin > 0 else -diff
            mult = mov_multiplier(margin, elo_diff_winner)
        else:
            mult = 1.0
        delta = K * mult * (result - p_home)
        ratings[h] += delta
        ratings[a] -= delta
    return ratings, history


def fit_spread_map(games, history, since=2010):
    """Fit margin = slope * elo_diff via OLS; return (slope, sigma)."""
    import numpy as np

    diffs = pd.Series([h[2] for h in history], index=games.index)
    m = games["home_points"] - games["away_points"]
    mask = games["season"] >= since
    x, y = diffs[mask].values, m[mask].values
    slope = float((x * y).sum() / (x * x).sum())
    sigma = float((y - slope * x).std())
    return slope, sigma


def season_priors():
    """(carry, prior_offsets) from priors.py if CFBD prior data is present."""
    try:
        from . import priors
        feats = priors.load_features()
        if not feats:
            return None, {}
        params = priors.load_params()
        return params["carry"], priors.offsets(feats, params)
    except Exception:
        return None, {}


def build():
    games = load_games()
    carry, offs = season_priors()
    ratings, history = run_elo(games, record_pregame=True, carry=carry, prior_offsets=offs)
    slope, sigma = fit_spread_map(games, history)
    return games, ratings, slope, sigma


def predict(ratings, slope, sigma, team1, team2, neutral=False):
    for t in (team1, team2):
        if t not in ratings:
            raise SystemExit(f"Unknown team: {t!r} (FBS names as in data/games.csv, e.g. 'Ohio State')")
    diff = ratings[team1] - ratings[team2] + (0.0 if neutral else HFA_ELO)
    return {"p1": win_prob(diff), "margin": slope * diff, "sigma": sigma}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", nargs="*")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--ratings", action="store_true")
    args = ap.parse_args()

    games, ratings, slope, sigma = build()
    if args.ratings:
        fbs = {t: e for t, e in ratings.items() if t != FCS}
        for i, (t, e) in enumerate(sorted(fbs.items(), key=lambda kv: -kv[1])[:30], 1):
            print(f"{i:3d}. {t:<25s} {e:7.1f}")
        return
    if len(args.teams) != 2:
        raise SystemExit(__doc__)
    t1, t2 = args.teams
    p = predict(ratings, slope, sigma, t1, t2, args.neutral)
    venue = "neutral site" if args.neutral else f"{t1} at home"
    print(f"{t1} vs {t2} ({venue})")
    print(f"  P({t1} win) = {p['p1']:.1%}   P({t2} win) = {1 - p['p1']:.1%}")
    print(f"  Predicted margin: {t1} by {p['margin']:+.1f} (sigma {p['sigma']:.1f})")


if __name__ == "__main__":
    main()
