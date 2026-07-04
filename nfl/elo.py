#!/usr/bin/env python3
"""538-style Elo ratings for the NFL.

MOV-scaled K, preseason regression to the mean, and a *fitted* (not
hardcoded) mapping from Elo-rating-diff to predicted point margin. Home-field
advantage and rest effects are estimated separately by regression rather than
baked into the Elo update itself — that keeps the Elo K-update a pure
team-strength signal (who won by how much) and lets HFA/rest/spread-map be
refit or rolled forward without re-running the whole rating history.

Usage:
  python3 -m nfl.elo --fit                        # refit spread map, save data/elo_params.json
  python3 -m nfl.elo "Kansas City Chiefs" "Buffalo Bills"
  python3 -m nfl.elo --ratings                    # current top 32
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES_CSV = os.path.join(HERE, "data", "games.csv")
PARAMS_JSON = os.path.join(HERE, "data", "elo_params.json")

K = 20.0
START_ELO = 1505.0
SEASON_REGRESS = 1.0 / 3.0    # fraction regressed to START_ELO between seasons
FIT_SINCE, FIT_UNTIL = 2003, 2014   # structural fit window (era decision)
HFA_WINDOW_SEASONS = 3              # rolling window for the *current* HFA estimate
BYE_REST_DAYS = 12                  # rest >= this counts as "off a bye"
SHORT_REST_DAYS = 5                 # rest <= this counts as a short week
INIT_BYE_PTS = 1.0
INIT_SHORT_PTS = -0.7
INIT_HFA_PTS = 2.0


def load_games(path: str = GAMES_CSV) -> pd.DataFrame:
    g = pd.read_csv(path, parse_dates=["date"])
    g = g.sort_values("date").reset_index(drop=True)
    return g


def win_prob(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def mov_multiplier(margin: float, elo_diff_winner: float) -> float:
    return math.log(abs(margin) + 1.0) * 2.2 / (elo_diff_winner * 0.001 + 2.2)


def run_elo(games: pd.DataFrame, record_pregame: bool = False):
    """Pure team-strength Elo (no HFA/rest baked into the update). Returns
    (ratings, history) where history[i] = pregame (home_elo, away_elo, diff)
    aligned with games' row order, only if record_pregame."""
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    history = []
    for r in games.itertuples():
        h, a = r.home, r.away
        for t in (h, a):
            if t not in ratings:
                ratings[t] = START_ELO
                last_season[t] = r.season
            elif last_season[t] != r.season:
                ratings[t] = START_ELO + (1.0 - SEASON_REGRESS) * (ratings[t] - START_ELO)
                last_season[t] = r.season
        diff = ratings[h] - ratings[a]
        if record_pregame:
            history.append((ratings[h], ratings[a], diff))
        margin = r.home_score - r.away_score
        result = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        p_home = win_prob(diff)
        if margin != 0:
            elo_diff_winner = diff if margin > 0 else -diff
            mult = mov_multiplier(margin, elo_diff_winner)
        else:
            mult = 1.0
        delta = K * mult * (result - p_home)
        ratings[h] += delta
        ratings[a] -= delta
    return ratings, history


def _is_bye(rest) -> float:
    return 1.0 if pd.notna(rest) and rest >= BYE_REST_DAYS else 0.0


def _is_short(rest) -> float:
    return 1.0 if pd.notna(rest) and rest <= SHORT_REST_DAYS else 0.0


def fit_spread_map(games: pd.DataFrame, history: list, since: int = FIT_SINCE,
                   until: int = FIT_UNTIL) -> dict:
    """OLS: margin = slope*elo_diff + hfa*(1-neutral) + bye_pts*bye_diff +
    short_pts*short_diff, fit once on the structural window. Returns slope,
    hfa_pts, bye_pts, short_pts, sigma."""
    diffs = np.array([h[2] for h in history])
    mask = (games["season"] >= since).values & (games["season"] <= until).values
    g = games[mask]
    x_elo = diffs[mask]
    x_hfa = (1.0 - g["neutral"].values.astype(float))
    x_bye = np.array([_is_bye(hr) - _is_bye(ar) for hr, ar in zip(g["home_rest"], g["away_rest"])])
    x_short = np.array([_is_short(hr) - _is_short(ar) for hr, ar in zip(g["home_rest"], g["away_rest"])])
    y = (g["home_score"] - g["away_score"]).values.astype(float)

    X = np.column_stack([x_elo, x_hfa, x_bye, x_short])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    slope, hfa_pts, bye_pts, short_pts = [float(c) for c in coef]
    pred = X @ coef
    sigma = float(np.sqrt(np.mean((y - pred) ** 2)))
    return {"slope": slope, "hfa_pts": hfa_pts, "bye_pts": bye_pts,
            "short_pts": short_pts, "sigma": sigma}


def rolling_hfa(games: pd.DataFrame, history: list, params: dict,
                asof_season: int, window: int = HFA_WINDOW_SEASONS) -> float:
    """Trailing (walk-forward safe) HFA-in-points over the `window` seasons
    strictly before `asof_season`, from residuals of the structural fit
    (slope/bye/short held fixed). Falls back to the structural hfa_pts when
    there isn't enough trailing data (e.g. earliest seasons)."""
    diffs = np.array([h[2] for h in history])
    mask = (games["season"] >= asof_season - window).values & (games["season"] < asof_season).values \
        & (games["neutral"].values == 0)
    if mask.sum() < 100:
        return params["hfa_pts"]
    g = games[mask]
    x_elo = diffs[mask]
    x_bye = np.array([_is_bye(hr) - _is_bye(ar) for hr, ar in zip(g["home_rest"], g["away_rest"])])
    x_short = np.array([_is_short(hr) - _is_short(ar) for hr, ar in zip(g["home_rest"], g["away_rest"])])
    y = (g["home_score"] - g["away_score"]).values.astype(float)
    resid = y - params["slope"] * x_elo - params["bye_pts"] * x_bye - params["short_pts"] * x_short
    return float(np.mean(resid))


def build(games: pd.DataFrame | None = None) -> dict:
    """Full build: pure Elo run + structural fit + per-season rolling HFA
    table. Returns a dict bundling everything predict() needs."""
    games = games if games is not None else load_games()
    ratings, history = run_elo(games, record_pregame=True)
    sm = fit_spread_map(games, history)
    seasons = sorted(games["season"].unique())
    hfa_by_season = {int(s): rolling_hfa(games, history, sm, int(s)) for s in seasons}
    # next (future) season uses the trailing window ending at the last complete season + 1
    next_season = int(seasons[-1]) + 1
    hfa_by_season[next_season] = rolling_hfa(games, history, sm, next_season)
    return {"ratings": ratings, "spread_map": sm, "hfa_by_season": hfa_by_season,
            "current_season": next_season}


def predict(built: dict, team1: str, team2: str, rest1: float = 7.0, rest2: float = 7.0,
           neutral: bool = False, season: int | None = None) -> dict:
    ratings = built["ratings"]
    for t in (team1, team2):
        if t not in ratings:
            raise SystemExit(f"Unknown team: {t!r}")
    sm = built["spread_map"]
    season = season or built["current_season"]
    hfa = 0.0 if neutral else built["hfa_by_season"].get(season, sm["hfa_pts"])
    diff = ratings[team1] - ratings[team2]
    bye_diff = _is_bye(rest1) - _is_bye(rest2)
    short_diff = _is_short(rest1) - _is_short(rest2)
    margin = sm["slope"] * diff + hfa + sm["bye_pts"] * bye_diff + sm["short_pts"] * short_diff
    # win prob from the same margin, in elo-equivalent units (margin / slope)
    effective_diff = margin / sm["slope"] if sm["slope"] else diff
    p1 = win_prob(effective_diff)
    return {"margin": margin, "sigma": sm["sigma"], "p1": p1, "elo_diff": diff}


def load_params(path: str = PARAMS_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def save_params(built: dict, path: str = PARAMS_JSON) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"ratings": built["ratings"], "spread_map": built["spread_map"],
                   "hfa_by_season": built["hfa_by_season"],
                   "current_season": built["current_season"]}, f, indent=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", nargs="*")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--ratings", action="store_true")
    args = ap.parse_args()

    if args.fit:
        built = build()
        save_params(built)
        sm = built["spread_map"]
        cur_hfa = built["hfa_by_season"][built["current_season"]]
        print(f"fitted {len(built['ratings'])} teams. slope={sm['slope']:.4f} "
              f"({1.0 / sm['slope']:.1f} elo/pt), sigma={sm['sigma']:.2f}, "
              f"structural hfa={sm['hfa_pts']:.2f}pt, current rolling hfa={cur_hfa:.2f}pt, "
              f"bye={sm['bye_pts']:+.2f}pt, short_week={sm['short_pts']:+.2f}pt")
        return 0

    built = load_params()
    if args.ratings:
        for i, (t, e) in enumerate(sorted(built["ratings"].items(), key=lambda kv: -kv[1])[:32], 1):
            print(f"{i:3d}. {t:<25s} {e:7.1f}")
        return 0
    if len(args.teams) != 2:
        raise SystemExit(__doc__)
    t1, t2 = args.teams
    p = predict(built, t1, t2, neutral=args.neutral)
    venue = "neutral site" if args.neutral else f"{t1} at home"
    print(f"{t1} vs {t2} ({venue})")
    print(f"  P({t1} win) = {p['p1']:.1%}   P({t2} win) = {1 - p['p1']:.1%}")
    print(f"  Predicted margin: {t1} by {p['margin']:+.1f} (sigma {p['sigma']:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
