#!/usr/bin/env python3
"""Elo ratings for Division-I college football (FBS + FCS).

Margin-of-victory-scaled K, home-field advantage, between-season regression to
the mean. Two ledgers:

1. **Champion (FBS)** — unchanged from the original model: FBS teams rated
   game by game with every non-FBS opponent pooled into one 'FCS' pseudo-team.
   All FBS-vs-FBS predictions and backtests come from this ledger only, so
   adding FCS data cannot move them.
2. **FCS ledger** — individual ratings for FCS teams (games.csv carries full
   FCS schedules), updated against *frozen* champion ratings when they play
   FBS sides, two-sided against each other, and against a separate sub-FCS
   pool for D2/D3/unknown opponents. Never feeds back into the champion
   ledger. Anchored at FCS_ANCHOR (start + between-season regression target).

predict() sees the merged ratings (champion wins any name collision), so
FBS-vs-FCS games are priced against the actual opponent. Spread mapping (Elo
points per point of margin) and margin sigma are fitted from data.

Usage:
  python3 elo.py "Ohio State" "Michigan"            # team 1 at home
  python3 elo.py "Ohio State" "Michigan" --neutral
  python3 elo.py --ratings                          # top 30
"""
import argparse
import hashlib
import json
import math
import os
from datetime import date

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES_CSV = os.path.join(HERE, "data", "games.csv")

K = 35.0
HFA_ELO = 62.0          # home-field advantage in Elo points
START_ELO = 1500.0
NEW_TEAM_ELO = 1300.0   # FBS newcomers / transitioning programs
SEASON_REGRESS = 0.30   # fraction regressed to 1500 between seasons
FCS = "FCS"             # champion-ledger pseudo-team for all non-FBS opponents
FCS_ANCHOR = 850.0      # FCS-ledger start + season anchor (the pooled
                        # pseudo-team self-calibrates to ~838)
SUB_FCS = "Sub-FCS"     # FCS-ledger pool for D2/D3/unknown-division opponents


def season_for_date(value=None):
    """CFB season label for a calendar date (January postseason belongs prior year)."""
    d = pd.Timestamp(value if value is not None else date.today())
    return int(d.year - 1 if d.month == 1 else d.year)


def load_games(path=GAMES_CSV):
    g = pd.read_csv(path, parse_dates=["date"])
    g["home"] = g["home_team"].where(g["home_div"] == "fbs", FCS)
    g["away"] = g["away_team"].where(g["away_div"] == "fbs", FCS)
    return g


def win_prob(elo_diff):
    """P(team with +elo_diff wins)."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def mov_multiplier(margin, elo_diff_winner):
    return math.log(abs(margin) + 1.0) * 2.2 / (elo_diff_winner * 0.001 + 2.2)


def run_elo(games, record_pregame=False, carry=None, prior_offsets=None,
            return_state=False):
    """Run Elo through all games chronologically.

    carry: between-season carryover of (rating - 1500); default 1 - SEASON_REGRESS.
    prior_offsets: dict[(team, season)] -> Elo points added at the team's first
    game of that season (preseason talent / returning-production priors).
    Returns (ratings, history) where history is a list of pregame-rating rows
    (only if record_pregame), aligned with games' row order. With
    ``return_state=True`` a third value preserves the separate champion/FCS
    ledgers and their last-seen seasons so a production snapshot can advance
    them into an upcoming season without losing the different rating anchors.
    """
    if carry is None:
        carry = 1.0 - SEASON_REGRESS
    prior_offsets = prior_offsets or {}
    ratings, last_season = {}, {}          # champion ledger (FBS + pooled FCS)
    fcs_ratings, fcs_last = {}, {}         # FCS ledger (individual + Sub-FCS pool)
    history = []

    def _fcs_entity(team, div):
        """FCS-ledger name for a non-FBS side (individual FCS, pooled below)."""
        return team if div == "fcs" else SUB_FCS

    def _fcs_roll(t, season):
        if t not in fcs_ratings:
            fcs_ratings[t] = FCS_ANCHOR
            fcs_last[t] = season
        elif fcs_last[t] != season:
            fcs_ratings[t] = FCS_ANCHOR + carry * (fcs_ratings[t] - FCS_ANCHOR)
            fcs_last[t] = season

    def _delta(diff, margin):
        p = win_prob(diff)
        result = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        if margin != 0:
            mult = mov_multiplier(margin, diff if margin > 0 else -diff)
        else:
            mult = 1.0
        return K * mult * (result - p)

    for r in games.itertuples():
        h, a = r.home, r.away
        home_fbs, away_fbs = r.home_div == "fbs", r.away_div == "fbs"
        margin = r.home_points - r.away_points
        hfa = 0.0 if r.neutral else HFA_ELO

        # ── champion ledger: exactly the original model; rows with no FBS side
        # never touch it (they would be pool-vs-pool self-games)
        pre_h = pre_a = None  # champion pregame ratings, for the FCS ledger
        if home_fbs or away_fbs:
            for t in (h, a):
                if t not in ratings:
                    base = START_ELO if t == FCS else NEW_TEAM_ELO
                    ratings[t] = base + prior_offsets.get((t, r.season), 0.0)
                    last_season[t] = r.season
                elif last_season[t] != r.season:
                    ratings[t] = (START_ELO + carry * (ratings[t] - START_ELO)
                                  + prior_offsets.get((t, r.season), 0.0))
                    last_season[t] = r.season
            pre_h, pre_a = ratings[h], ratings[a]
            diff = pre_h + hfa - pre_a
            rec = [pre_h, pre_a, diff]
            delta = _delta(diff, margin)
            ratings[h] += delta
            ratings[a] -= delta

        # ── FCS ledger: one-sided vs frozen champion ratings, two-sided otherwise
        if not (home_fbs and away_fbs):
            eh = h if home_fbs else _fcs_entity(r.home_team, r.home_div)
            ea = a if away_fbs else _fcs_entity(r.away_team, r.away_div)
            for e, fbs in ((eh, home_fbs), (ea, away_fbs)):
                if not fbs:
                    _fcs_roll(e, r.season)
            rh = pre_h if home_fbs else fcs_ratings[eh]
            ra = pre_a if away_fbs else fcs_ratings[ea]
            fdiff = rh + hfa - ra
            if home_fbs or away_fbs:
                rec.append(fdiff)  # 4th element: individual-opponent diff
            else:
                rec = [rh, ra, fdiff]
            delta = _delta(fdiff, margin)
            if not home_fbs:
                fcs_ratings[eh] += delta
            if not away_fbs:
                fcs_ratings[ea] -= delta

        if record_pregame:
            history.append(tuple(rec))

    merged = {**{t: e for t, e in fcs_ratings.items() if t != SUB_FCS}, **ratings}
    if return_state:
        state = {
            "champion_ratings": dict(ratings),
            "champion_last_season": dict(last_season),
            "fcs_ratings": dict(fcs_ratings),
            "fcs_last_season": dict(fcs_last),
        }
        return merged, history, state
    return merged, history


def fit_spread_map(games, history, since=2010):
    """Fit margin = slope * elo_diff via OLS; return (slope, sigma).

    Fitted on games with an FBS side (the champion ledger's rows, with pooled
    diffs — identical to the original model); FCS-vs-FCS rows are excluded so
    their margin distribution can't distort the Elo→points mapping."""
    import numpy as np

    diffs = pd.Series([h[2] for h in history], index=games.index)
    m = games["home_points"] - games["away_points"]
    mask = ((games["season"] >= since)
            & ((games["home_div"] == "fbs") | (games["away_div"] == "fbs")))
    x, y = diffs[mask].values, m[mask].values
    slope = float((x * y).sum() / (x * x).sum())
    sigma = float((y - slope * x).std())
    return slope, sigma


def fit_cross_slope(games, history, since=2010):
    """Elo→points slope for FBS-vs-FCS games (margin = slope_x * individual
    diff). Cross-division Elo gaps are huge and real margins compress (starters
    sit in blowouts), so the champion slope overshoots; a separately fitted
    proportional map beats it held-out (2019+ margin MAE 17.8 → 13.4, also
    beating the old pooled pseudo-team's 14.8)."""
    import numpy as np

    x, y = [], []
    for r, h in zip(games.itertuples(), history):
        if (r.home_div == "fbs") != (r.away_div == "fbs") \
                and len(h) == 4 and r.season >= since:
            x.append(h[3])
            y.append(r.home_points - r.away_points)
    x, y = np.array(x), np.array(y)
    return float((x * y).sum() / (x * x).sum()) if len(x) else None


# set by build(); consumers fall back to a fixed ratio of the champion slope
_CROSS = {"slope": None}
CROSS_SLOPE_RATIO = 0.76  # fitted cross-slope / champion slope, 2010-2018


def cross_slope(champion_slope):
    return _CROSS["slope"] if _CROSS["slope"] else CROSS_SLOPE_RATIO * champion_slope


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


def _schedule_divisions(season):
    """Canonical team classifications from the target-season CFBD schedule."""
    path = os.path.join(HERE, "data", f"schedule_{int(season)}.json")
    if not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for g in data:
        for side in ("home", "away"):
            team = g.get(f"{side}Team")
            div = str(g.get(f"{side}Classification") or "").lower()
            if team and div:
                out[team] = div
    return out


def _snapshot_hash(target_season):
    """Short content fingerprint for the files that define preseason Elo state."""
    h = hashlib.sha256()
    paths = [GAMES_CSV, os.path.join(HERE, "data", "prior_params.json"),
             os.path.join(HERE, "data", f"schedule_{int(target_season)}.json"),
             os.path.join(HERE, "data", "cfbd", f"talent_{int(target_season)}.json"),
             os.path.join(HERE, "data", "cfbd", f"returning_{int(target_season)}.json")]
    for path in paths:
        h.update(os.path.basename(path).encode())
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:16]


def advance_state(state, target_season, carry, prior_offsets,
                  team_divisions=None):
    """Advance separate Elo ledgers into ``target_season`` exactly once.

    Champion/FBS ratings regress around 1500. Individual FCS ratings regress
    around the 850 FCS anchor. A target-season FBS team previously carried only
    by the FCS ledger transitions at the stronger of its regressed FCS rating and
    the standard 1300 new-FBS rating; this makes the transition policy explicit
    and prevents the old win-totals path from dragging every FCS team toward 1500.
    """
    team_divisions = {str(k): str(v).lower() for k, v in (team_divisions or {}).items()}
    champion = dict(state["champion_ratings"])
    champion_last = dict(state["champion_last_season"])
    fcs = dict(state["fcs_ratings"])
    fcs_last = dict(state["fcs_last_season"])
    rolled_fbs = rolled_fcs = 0

    for t in list(champion):
        if champion_last.get(t) != target_season:
            champion[t] = (START_ELO + carry * (champion[t] - START_ELO)
                           + (0.0 if t == FCS else prior_offsets.get((t, target_season), 0.0)))
            champion_last[t] = target_season
            rolled_fbs += int(t != FCS)

    for t in list(fcs):
        if fcs_last.get(t) != target_season:
            fcs[t] = FCS_ANCHOR + carry * (fcs[t] - FCS_ANCHOR)
            fcs_last[t] = target_season
            rolled_fcs += int(t != SUB_FCS)

    transitioned, new_fbs, new_fcs = [], [], []
    for team, div in team_divisions.items():
        if div == "fbs" and team not in champion:
            if team in fcs:
                champion[team] = (max(NEW_TEAM_ELO, fcs[team])
                                  + prior_offsets.get((team, target_season), 0.0))
                transitioned.append(team)
            else:
                champion[team] = (NEW_TEAM_ELO
                                  + prior_offsets.get((team, target_season), 0.0))
                new_fbs.append(team)
            champion_last[team] = target_season
        elif div == "fcs" and team not in fcs:
            fcs[team] = FCS_ANCHOR
            fcs_last[team] = target_season
            new_fcs.append(team)

    merged = {**{t: e for t, e in fcs.items() if t != SUB_FCS}, **champion}
    # A declared target-season FCS team must use its individual ledger even if it
    # previously appeared in the champion ledger.
    for team, div in team_divisions.items():
        if div == "fcs" and team in fcs:
            merged[team] = fcs[team]
    advanced = {
        "champion_ratings": champion,
        "champion_last_season": champion_last,
        "fcs_ratings": fcs,
        "fcs_last_season": fcs_last,
    }
    audit = {"rolled_fbs": rolled_fbs, "rolled_fcs": rolled_fcs,
             "transitioned_fbs": sorted(transitioned), "new_fbs": sorted(new_fbs),
             "new_fcs": sorted(new_fcs)}
    return merged, advanced, audit


def build_as_of(target_season, as_of=None, team_divisions=None):
    """Build a production Elo snapshot for a declared season and decision date.

    Unlike :func:`build`, this advances teams that have not yet played in the
    target season and reports whether the preseason prior coverage is sufficient
    for betting. The return value is ``(eparams, metadata)`` where ``eparams`` is
    the existing ``(games, ratings, slope, sigma)`` tuple accepted by predictors.
    """
    target_season = int(target_season)
    as_of = pd.Timestamp(as_of if as_of is not None else date.today()).tz_localize(None)
    games = load_games()
    games = games[games["date"] < as_of].copy()
    if games.empty:
        raise ValueError(f"no completed games before {as_of.date()}")
    carry, offs = season_priors()
    if carry is None:
        carry = 1.0 - SEASON_REGRESS
    ratings, history, state = run_elo(
        games, record_pregame=True, carry=carry, prior_offsets=offs,
        return_state=True)
    slope, sigma = fit_spread_map(games, history)
    _CROSS["slope"] = fit_cross_slope(games, history)

    divisions = dict(_schedule_divisions(target_season))
    divisions.update(team_divisions or {})
    source_max_season = int(games["season"].max())
    completed_target = int((games["season"] == target_season).sum())
    ratings, state, audit = advance_state(
        state, target_season, carry, offs, divisions)

    try:
        from . import priors
        features = priors.load_features()
    except Exception:
        features = {}
    target_fbs = {t for t, div in divisions.items() if str(div).lower() == "fbs"}
    complete_prior = {t for t in target_fbs
                      if "talent_z" in features.get((t, target_season), {})
                      and "ret_c" in features.get((t, target_season), {})}
    coverage = len(complete_prior) / len(target_fbs) if target_fbs else 0.0
    if completed_target:
        prior_mode = "in_season"
    elif coverage >= 0.80:
        prior_mode = "full_prior"
    elif coverage > 0:
        prior_mode = "partial_prior"
    else:
        prior_mode = "regression_only"
    betting_eligible = prior_mode in ("full_prior", "in_season")
    meta = {
        "model_season": target_season,
        "decision_date": str(as_of.date()),
        "games_through": str(games["date"].max().date()),
        "source_max_season": source_max_season,
        "completed_target_games": completed_target,
        "prior_mode": prior_mode,
        "prior_complete_teams": len(complete_prior),
        "prior_target_teams": len(target_fbs),
        "prior_coverage": round(coverage, 4),
        "betting_eligible": betting_eligible,
        "snapshot_hash": _snapshot_hash(target_season),
        **audit,
    }
    return (games, ratings, slope, sigma), meta


def build():
    games = load_games()
    carry, offs = season_priors()
    ratings, history = run_elo(games, record_pregame=True, carry=carry, prior_offsets=offs)
    slope, sigma = fit_spread_map(games, history)
    _CROSS["slope"] = fit_cross_slope(games, history)
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
