"""League-position motivation features (P4.3).

As of each league match date, for fixtures where BOTH sides have played
>= 8 rounds: pos_diff, ppg_diff, and fight/dead flags (title race, europe
race, relegation battle, or mathematically dead) computed only when rounds
remaining R <= 8, else 0. `pos_diff` is collinear with team strength (the
context GLM already has a strength-derived offset) so only `ppg_diff` is
fed in as a raw diff; `fight_diff`/`dead_diff` are the flag differences.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .competitions import get as comp_get
from . import standings as ST

MIN_ROUNDS_PLAYED = 8
FIGHT_MARGIN_PTS = 3
ROUNDS_REMAINING_THRESHOLD = 8


def _boundaries(rows: list[tuple[str, dict]], comp) -> list[float]:
    """Points totals of the relegation cutoff, 1st place, and the last euro
    spot, from a table already sorted best-to-worst (list of (team, stats))."""
    n_teams_seen = len(rows)
    out: list[float] = []
    releg_idx = comp.teams_n - comp.releg_spots  # 1-indexed last "safe" position
    if comp.releg_spots > 0 and 1 <= releg_idx <= n_teams_seen:
        out.append(float(rows[releg_idx - 1][1]["points"]))
    if n_teams_seen >= 1:
        out.append(float(rows[0][1]["points"]))
    if comp.euro_spots > 0 and comp.euro_spots <= n_teams_seen:
        out.append(float(rows[comp.euro_spots - 1][1]["points"]))
    return out


def _fight(points: float, boundaries: list[float]) -> bool:
    return any(abs(points - b) <= FIGHT_MARGIN_PTS for b in boundaries)


def _dead(points: float, fight: bool, boundaries: list[float], rounds_left: int) -> bool:
    if fight or not boundaries:
        return False
    return all(abs(points - b) > 3 * rounds_left for b in boundaries)


def training_features(played_all: pd.DataFrame) -> pd.DataFrame:
    """pos_diff/ppg_diff/fight_diff/dead_diff per league fixture, computed
    incrementally per (competition, season) in chronological order — a
    match's own result never counts toward its own pre-match table."""
    played_all = played_all.sort_values("date").reset_index(drop=True)
    n = len(played_all)
    pos_diff = np.full(n, np.nan)
    ppg_diff = np.full(n, np.nan)
    fight_diff = np.zeros(n)
    dead_diff = np.zeros(n)

    tables: dict[tuple, dict[str, dict]] = {}

    for i, r in enumerate(played_all.itertuples(index=False)):
        if r.type == "league":
            comp = comp_get(r.competition)
            if comp is not None and comp.teams_n > 0:
                key = (r.competition, r.season)
                table = tables.setdefault(key, {})

                def _row(team: str) -> dict:
                    return table.setdefault(team, {"played": 0, "points": 0, "gf": 0, "ga": 0})

                h, a = _row(r.home), _row(r.away)
                if h["played"] >= MIN_ROUNDS_PLAYED and a["played"] >= MIN_ROUNDS_PLAYED:
                    rows = sorted(table.items(), key=lambda kv: (
                        -kv[1]["points"], -(kv[1]["gf"] - kv[1]["ga"]), -kv[1]["gf"]))
                    pos = {t: idx + 1 for idx, (t, _) in enumerate(rows)}
                    ppg = {t: (v["points"] / v["played"] if v["played"] else 0.0)
                           for t, v in table.items()}
                    rounds_left = max(0, 2 * (comp.teams_n - 1) - max(h["played"], a["played"]))
                    boundaries = _boundaries(rows, comp)

                    pos_diff[i] = pos.get(r.home, 0) - pos.get(r.away, 0)
                    ppg_diff[i] = ppg.get(r.home, 0.0) - ppg.get(r.away, 0.0)
                    if rounds_left <= ROUNDS_REMAINING_THRESHOLD:
                        fh = _fight(h["points"], boundaries)
                        fa = _fight(a["points"], boundaries)
                        dh = _dead(h["points"], fh, boundaries, rounds_left)
                        da = _dead(a["points"], fa, boundaries, rounds_left)
                        fight_diff[i] = float(fh) - float(fa)
                        dead_diff[i] = float(dh) - float(da)

                # update AFTER computing this row's features (point-in-time)
                hg, ag = int(r.home_goals), int(r.away_goals)
                h["played"] += 1; a["played"] += 1
                h["gf"] += hg; h["ga"] += ag
                a["gf"] += ag; a["ga"] += hg
                if hg > ag:
                    h["points"] += 3
                elif hg < ag:
                    a["points"] += 3
                else:
                    h["points"] += 1; a["points"] += 1

    out = played_all[["fixture_id"]].copy()
    out["pos_diff"] = pos_diff
    out["ppg_diff"] = ppg_diff
    out["fight_diff"] = fight_diff
    out["dead_diff"] = dead_diff
    return out


def live_features(home: str, away: str, competition: str, season: int,
                  match_date: str, fixtures: pd.DataFrame | None = None) -> dict:
    """Point-in-time motivation features for a concrete upcoming league
    fixture, using standings.table_asof (a single query — fine for one-off
    live use; training uses the faster incremental pass above)."""
    zero = {"ppg_diff": 0.0, "fight_diff": 0.0, "dead_diff": 0.0}
    comp = comp_get(competition)
    if comp is None or comp.teams_n <= 0:
        return zero
    table = ST.table_asof(competition, season, match_date, fixtures)
    if table.empty:
        return zero
    hrow = table[table["team"] == home]
    arow = table[table["team"] == away]
    if hrow.empty or arow.empty:
        return zero
    played_h, played_a = int(hrow.iloc[0]["played"]), int(arow.iloc[0]["played"])
    if played_h < MIN_ROUNDS_PLAYED or played_a < MIN_ROUNDS_PLAYED:
        return zero

    rows = list(table.sort_values("position").itertuples(index=False))
    rows_as_dicts = [(r.team, {"points": r.points, "gf": r.gf, "ga": r.ga}) for r in rows]
    boundaries = _boundaries(rows_as_dicts, comp)
    rounds_left = max(0, 2 * (comp.teams_n - 1) - max(played_h, played_a))

    ppg_h = hrow.iloc[0]["points"] / played_h
    ppg_a = arow.iloc[0]["points"] / played_a
    fight_diff = dead_diff = 0.0
    if rounds_left <= ROUNDS_REMAINING_THRESHOLD:
        fh = _fight(float(hrow.iloc[0]["points"]), boundaries)
        fa = _fight(float(arow.iloc[0]["points"]), boundaries)
        dh = _dead(float(hrow.iloc[0]["points"]), fh, boundaries, rounds_left)
        da = _dead(float(arow.iloc[0]["points"]), fa, boundaries, rounds_left)
        fight_diff = float(fh) - float(fa)
        dead_diff = float(dh) - float(da)

    return {"ppg_diff": float(ppg_h - ppg_a), "fight_diff": fight_diff, "dead_diff": dead_diff}
