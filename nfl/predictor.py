#!/usr/bin/env python3
"""Blended NFL match predictor: Elo + power ratings + EPA (candidate) + QB
starter adjustment, priced through the empirical margin/total PMF (never a
normal approximation — see margin_dist.py).

Usage:
  python3 -m nfl.predictor "Kansas City Chiefs" "Buffalo Bills"
  python3 -m nfl.predictor "Eagles"... --neutral --model elo|power|epa|blend
  python3 -m nfl.predictor --home-qb "Josh Allen" --away-qb "Patrick Mahomes" ...
"""
from __future__ import annotations

import argparse
import json
import os

from . import elo as E
from . import epa as X
from . import margin_dist as MD
from . import power as P
from . import qb as QB
from .team_names import normalize as normalize_team

HERE = os.path.dirname(os.path.abspath(__file__))
BLEND_WEIGHT_JSON = os.path.join(HERE, "data", "blend_weight.json")

DEFAULT_WEIGHTS = {"elo": 0.50, "power": 0.50, "epa": 0.00}


def load_blend_weights(path: str = BLEND_WEIGHT_JSON) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                w = json.load(f)
            return _normalise_weights(w)
        except Exception:
            pass
    return dict(DEFAULT_WEIGHTS)


def _normalise_weights(weights: dict) -> dict:
    vals = {k: max(0.0, float(weights.get(k, 0.0))) for k in ("elo", "power", "epa")}
    s = sum(vals.values())
    if s <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / s for k, v in vals.items()}


class Models:
    """Bundle of everything blend_predict() needs, built once and reused
    across many predictions (CLI, edge.py, validate.py)."""

    def __init__(self, elo_built, power_params, epa_params, qb_params):
        self.elo_built = elo_built
        self.power_params = power_params
        self.epa_params = epa_params
        self.qb_params = qb_params
        self.qb_index = QB.QBIndex(qb_params) if qb_params else None
        self.pmf = MD.load()

    @classmethod
    def load(cls) -> "Models":
        return cls(E.load_params(), P.load_params(), X.load_params(), QB.load_params())


def blend_predict(models: Models, team1: str, team2: str, *, neutral: bool = False,
                  model: str = "blend", weights: dict | None = None,
                  season: int | None = None, week: int | None = None,
                  rest1: float = 7.0, rest2: float = 7.0,
                  qb1: str | None = None, qb2: str | None = None) -> dict:
    """Predict team1 (home unless neutral) vs team2. `model` selects a single
    rating source ("elo"/"power"/"epa") or the tuned blend ("blend",
    default). QB adjustment applies on top of any model when qb1/qb2 (actual
    starters) are given and differ from the rolling "expected starter"."""
    team1, team2 = normalize_team(team1), normalize_team(team2)
    w = weights or load_blend_weights()

    ep = E.predict(models.elo_built, team1, team2, rest1=rest1, rest2=rest2,
                  neutral=neutral, season=season)
    pp = P.predict(models.power_params, team1, team2, neutral=neutral)
    xp = None
    if team1 in models.epa_params["teams"] and team2 in models.epa_params["teams"]:
        xp = X.predict(models.epa_params, team1, team2, neutral=neutral)

    if model == "elo":
        margin, total = ep["margin"], pp["total"]  # elo has no total; borrow power's
    elif model == "power":
        margin, total = pp["margin"], pp["total"]
    elif model == "epa":
        if xp is None:
            raise ValueError(f"no EPA rating for {team1!r} or {team2!r}")
        margin, total = xp["margin"], xp["total"]
    elif model == "blend":
        we, wp, wx = w["elo"], w["power"], w["epa"]
        if xp is None:
            # redistribute epa's weight proportionally across elo/power
            rest = we + wp
            we, wp, wx = (we + wx * we / rest, wp + wx * wp / rest, 0.0) if rest > 0 else (0.5, 0.5, 0.0)
        margin = we * ep["margin"] + wp * pp["margin"] + (wx * xp["margin"] if xp else 0.0)
        tot_w = wp + wx
        total = (wp * pp["total"] + (wx * xp["total"] if xp else 0.0)) / tot_w if tot_w > 0 else pp["total"]
    else:
        raise ValueError(f"unknown model {model!r}")

    qb_delta = 0.0
    if models.qb_index is not None and season is not None and week is not None:
        qb_delta = (models.qb_index.qb_delta_points(team1, season, week, override_qb_name=qb1)
                   - models.qb_index.qb_delta_points(team2, season, week, override_qb_name=qb2))
        margin += qb_delta

    ml = MD.moneyline_probs(models.pmf, margin)
    return {
        "team1": team1, "team2": team2, "margin": margin, "total": total,
        "p1": ml["home_win"], "p2": ml["away_win"], "p_tie": ml["tie"],
        "qb_delta": qb_delta, "components": {
            "elo": {"margin": ep["margin"]}, "power": {"margin": pp["margin"], "total": pp["total"]},
            "epa": ({"margin": xp["margin"], "total": xp["total"]} if xp else None),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", nargs="*")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--model", default="blend", choices=["blend", "elo", "power", "epa"])
    ap.add_argument("--home-qb", default=None)
    ap.add_argument("--away-qb", default=None)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()
    if len(args.teams) != 2:
        raise SystemExit(__doc__)

    models = Models.load()
    t1, t2 = args.teams
    out = blend_predict(models, t1, t2, neutral=args.neutral, model=args.model,
                        season=args.season, week=args.week, qb1=args.home_qb, qb2=args.away_qb)
    venue = "neutral site" if args.neutral else f"{out['team1']} at home"
    line = -out["margin"]  # spread_line convention: positive = home favored, so the
                          # "line" a book would post on team1 is +margin (see margin_dist.py)
    print(f"{out['team1']} vs {out['team2']} ({venue}, model={args.model})")
    print(f"  Predicted margin: {out['team1']} {out['margin']:+.1f}, total {out['total']:.1f}")
    print(f"  P({out['team1']} win) = {out['p1']:.1%}   P({out['team2']} win) = {out['p2']:.1%}"
         f"   P(tie) = {out['p_tie']:.2%}")
    if out["qb_delta"]:
        print(f"  QB adjustment: {out['qb_delta']:+.1f} pts (home vs away)")

    # key-number table around the predicted spread
    print("\n  Key-number cover probabilities (home side):")
    for L in (-7.0, -3.0, 0.0, 3.0, 7.0):
        c = MD.cover_probs(models.pmf, out["margin"], L)
        print(f"    line {L:+.1f}: home cover {c['home_cover']:.1%}  push {c['push']:.1%}  "
             f"away cover {c['away_cover']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
