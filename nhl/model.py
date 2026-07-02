"""NHL match model.

Offline first: the model reads a local team-stat baseline and turns attack,
defence, shot, special-teams, and points-share signals into expected goals.
The pricing layer then uses independent Poisson score distributions plus an
overtime split to support NHL moneyline, puck-line, and totals markets.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
TEAM_STATS_CSV = DATA_DIR / "team_stats.csv"

MAX_GOALS = 14
HOME_GOAL_ADV = 1.045
AWAY_GOAL_ADV = 0.985
HOME_OT_EDGE = 0.07
EPS = 1e-9

TEAM_ALIASES = {
    "utah hockey club": "Utah Mammoth",
}


@dataclass(frozen=True)
class TeamRating:
    team: str
    attack: float
    defence_allowed: float
    form: float
    point_pct: float
    gf_pg: float
    ga_pg: float


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), lo), hi)


def _logistic(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _fold_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    ascii_text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(ascii_text.lower().replace(".", "").split())


@lru_cache(maxsize=1)
def load_team_stats() -> pd.DataFrame:
    if not TEAM_STATS_CSV.exists():
        raise FileNotFoundError(f"Missing NHL team stats file: {TEAM_STATS_CSV}")
    df = pd.read_csv(TEAM_STATS_CSV)
    required = {
        "team", "games", "goals_for", "goals_against", "shots_for",
        "shots_against", "power_play_pct", "penalty_kill_pct", "save_pct",
        "point_pct",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"nhl/data/team_stats.csv missing columns: {missing}")
    if df["team"].duplicated().any():
        dupes = sorted(df.loc[df["team"].duplicated(), "team"].astype(str).unique())
        raise ValueError(f"Duplicate NHL team rows: {dupes}")
    numeric = sorted(required - {"team"})
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[df[numeric].isna().any(axis=1)]
    if not bad.empty:
        raise ValueError(f"NHL team stats contain non-numeric values for: {bad['team'].tolist()}")
    if (df["games"] <= 0).any():
        raise ValueError("NHL team stats require games > 0")
    return df.copy()


def team_names() -> list[str]:
    return sorted(load_team_stats()["team"].astype(str).tolist())


@lru_cache(maxsize=1)
def _ratings() -> tuple[dict[str, TeamRating], dict[str, float]]:
    df = load_team_stats()
    games = df["games"].sum()
    league_gf_pg = float(df["goals_for"].sum() / games)
    league_ga_pg = float(df["goals_against"].sum() / games)
    league_sf_pg = float(df["shots_for"].sum() / games)
    league_sa_pg = float(df["shots_against"].sum() / games)
    league_pp = float(df["power_play_pct"].mean())
    league_pk = float(df["penalty_kill_pct"].mean())
    league_pt = float(df["point_pct"].mean())

    ratings: dict[str, TeamRating] = {}
    for r in df.itertuples(index=False):
        gf_pg = float(r.goals_for) / float(r.games)
        ga_pg = float(r.goals_against) / float(r.games)
        sf_pg = float(r.shots_for) / float(r.games)
        sa_pg = float(r.shots_against) / float(r.games)

        attack = (
            0.72 * (gf_pg / league_gf_pg)
            + 0.18 * (sf_pg / league_sf_pg)
            + 0.10 * (float(r.power_play_pct) / league_pp)
        )
        defence_allowed = (
            0.70 * (ga_pg / league_ga_pg)
            + 0.20 * (sa_pg / league_sa_pg)
            + 0.10 * ((1.0 - float(r.penalty_kill_pct)) / (1.0 - league_pk))
        )
        form = (float(r.point_pct) - league_pt) / 0.100
        ratings[str(r.team)] = TeamRating(
            team=str(r.team),
            attack=_clip(attack, 0.75, 1.28),
            defence_allowed=_clip(defence_allowed, 0.75, 1.28),
            form=_clip(form, -2.0, 2.0),
            point_pct=_clip(float(r.point_pct), 0.250, 0.800),
            gf_pg=gf_pg,
            ga_pg=ga_pg,
        )
    meta = {
        "league_goals_pg": (league_gf_pg + league_ga_pg) / 2.0,
        "league_point_pct": league_pt,
    }
    return ratings, meta


def _require_team(team: str) -> TeamRating:
    ratings, _meta = _ratings()
    if team not in ratings:
        folded = _fold_name(team)
        alias = TEAM_ALIASES.get(folded)
        if alias and alias in ratings:
            return ratings[alias]
        folded_map = {_fold_name(name): name for name in ratings}
        if folded in folded_map:
            return ratings[folded_map[folded]]
        raise ValueError(f"Unknown NHL team: {team!r}")
    return ratings[team]


def _power_lambdas(home: TeamRating, away: TeamRating, neutral: bool) -> tuple[float, float]:
    _ratings_map, meta = _ratings()
    base = meta["league_goals_pg"]
    h_adv = 1.0 if neutral else HOME_GOAL_ADV
    a_adv = 1.0 if neutral else AWAY_GOAL_ADV
    lam_h = base * home.attack * away.defence_allowed * h_adv
    lam_a = base * away.attack * home.defence_allowed * a_adv
    return _clip(lam_h, 1.35, 4.60), _clip(lam_a, 1.35, 4.60)


def _form_lambdas(home: TeamRating, away: TeamRating, neutral: bool) -> tuple[float, float]:
    power_h, power_a = _power_lambdas(home, away, neutral)
    total = power_h + power_a
    home_edge = 0.0 if neutral else 0.18
    margin = 0.44 * (home.form - away.form) + home_edge
    lam_h = (total + margin) / 2.0
    lam_a = (total - margin) / 2.0
    return _clip(lam_h, 1.25, 4.75), _clip(lam_a, 1.25, 4.75)


def expected_goals(home_team: str, away_team: str, *, neutral: bool = False,
                   model: str = "blend") -> tuple[float, float]:
    """Expected regulation goals for `home_team` and `away_team`."""
    home = _require_team(home_team)
    away = _require_team(away_team)
    model = str(model or "blend").lower()
    if model == "power":
        return _power_lambdas(home, away, neutral)
    if model == "form":
        return _form_lambdas(home, away, neutral)
    if model != "blend":
        raise ValueError(f"Unknown NHL model: {model!r}")
    ph, pa = _power_lambdas(home, away, neutral)
    fh, fa = _form_lambdas(home, away, neutral)
    return 0.65 * ph + 0.35 * fh, 0.65 * pa + 0.35 * fa


def _poisson(mu: float) -> list[float]:
    vals = [math.exp(-mu)]
    for k in range(1, MAX_GOALS + 1):
        vals.append(vals[-1] * mu / k)
    s = sum(vals)
    return [v / s for v in vals]


def _score_probs(lambda_home: float, lambda_away: float) -> list[tuple[int, int, float]]:
    hp = _poisson(lambda_home)
    ap = _poisson(lambda_away)
    return [(h, a, hp[h] * ap[a]) for h in range(len(hp)) for a in range(len(ap))]


def _ot_home_prob(lambda_home: float, lambda_away: float, home: TeamRating,
                  away: TeamRating, neutral: bool) -> float:
    home_edge = 0.0 if neutral else HOME_OT_EDGE
    strength = 0.48 * (lambda_home - lambda_away) + 1.25 * (home.point_pct - away.point_pct)
    return _clip(_logistic(strength + home_edge), 0.38, 0.62)


def _outcome_core(lambda_home: float, lambda_away: float, ot_home: float) -> dict[str, float]:
    home_reg = away_reg = tie_reg = 0.0
    home_by_2 = away_by_2 = 0.0
    over_55 = 0.0
    for h, a, p in _score_probs(lambda_home, lambda_away):
        if h > a:
            home_reg += p
            if h - a >= 2:
                home_by_2 += p
        elif a > h:
            away_reg += p
            if a - h >= 2:
                away_by_2 += p
        else:
            tie_reg += p
        if h + a > 5.5:
            over_55 += p
    p_home = home_reg + tie_reg * ot_home
    return {
        "p_home": p_home,
        "p_away": 1.0 - p_home,
        "p_home_reg": home_reg,
        "p_away_reg": away_reg,
        "p_reg_tie": tie_reg,
        "p_ot_home": ot_home,
        "p_home_minus_1_5": home_by_2,
        "p_away_minus_1_5": away_by_2,
        "p_over_5_5": over_55,
        "p_under_5_5": 1.0 - over_55,
    }


def predict_match(home_team: str, away_team: str, *, neutral: bool = False,
                  model: str = "blend") -> dict[str, Any]:
    if home_team == away_team:
        raise ValueError("Pick two different NHL teams.")
    home = _require_team(home_team)
    away = _require_team(away_team)
    lam_h, lam_a = expected_goals(home_team, away_team, neutral=neutral, model=model)
    ot_home = _ot_home_prob(lam_h, lam_a, home, away, neutral)
    probs = _outcome_core(lam_h, lam_a, ot_home)
    total = lam_h + lam_a
    margin = lam_h - lam_a
    return {
        "home": home_team,
        "away": away_team,
        "model": str(model or "blend").lower(),
        "neutral": bool(neutral),
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "total": total,
        "margin": margin,
        **probs,
    }


def market_probs(pred: dict[str, Any], market: str, side: str,
                 line: float | None = None) -> tuple[float, float]:
    """Return (win probability, push probability) for a quoted market."""
    market = normalize_market(market)
    side = str(side or "").strip().lower()
    lh = float(pred["lambda_home"])
    la = float(pred["lambda_away"])
    ot_home = float(pred["p_ot_home"])

    if market == "ml":
        if side == "home":
            return float(pred["p_home"]), 0.0
        if side == "away":
            return float(pred["p_away"]), 0.0
        raise ValueError(f"Unknown NHL moneyline side: {side!r}")

    if line is None:
        raise ValueError(f"{market} market needs a line")
    line = float(line)
    p_win = p_push = 0.0
    if market == "total":
        if side not in {"over", "under"}:
            raise ValueError(f"Unknown NHL total side: {side!r}")
        for h, a, p in _score_probs(lh, la):
            total = h + a
            if abs(total - line) < EPS:
                p_push += p
            elif (total > line and side == "over") or (total < line and side == "under"):
                p_win += p
        return p_win, p_push

    if market == "spread":
        if side not in {"home", "away"}:
            raise ValueError(f"Unknown NHL puck-line side: {side!r}")
        for h, a, p in _score_probs(lh, la):
            finals = [(h, a, 1.0)]
            if h == a:
                finals = [(h + 1, a, ot_home), (h, a + 1, 1.0 - ot_home)]
            for fh, fa, w in finals:
                margin = fh - fa
                adjusted = margin + line if side == "home" else -margin + line
                if abs(adjusted) < EPS:
                    p_push += p * w
                elif adjusted > 0:
                    p_win += p * w
        return p_win, p_push

    raise ValueError(f"Unknown NHL market: {market!r}")


def market_prob(pred: dict[str, Any], market: str, side: str,
                line: float | None = None) -> float:
    return market_probs(pred, market, side, line)[0]


def normalize_market(market: str) -> str:
    raw = str(market or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "moneyline": "ml",
        "money_line": "ml",
        "puckline": "spread",
        "puck_line": "spread",
        "run_line": "spread",
        "handicap": "spread",
        "spread": "spread",
        "total": "total",
        "totals": "total",
        "over_under": "total",
        "ou": "total",
        "ml": "ml",
    }
    out = aliases.get(raw, raw)
    if out not in {"ml", "spread", "total"}:
        raise ValueError(f"Unknown NHL market: {market!r}")
    return out
