#!/usr/bin/env python3
"""Rolling QB value adjustment for the NFL engine.

QB value = shrunk rolling EPA/dropback, decayed by *cumulative dropbacks*
(half-life ~= 250 dropbacks, roughly a season) rather than calendar time —
a QB's value should update on snaps played, not days elapsed. Low-sample
QBs (< 50 career dropbacks) are pinned to replacement level; between 50 and
~250 they're blended smoothly toward their own rolling average.

Application: each team carries a rolling "expected starter" (whoever has
logged the most recent snaps). When the actual/announced starter differs,
the team's predicted margin is adjusted by qb_pts(actual) - qb_pts(expected).
Backtests read actual starters from games.csv `home_qb_name`/`away_qb_name`
(leak-free: rolling value computed from strictly prior weeks). Live picks
read data/qb_overrides.csv (season, week, team, qb_name) for game-day news.

Usage:
  python3 -m nfl.qb --fit                          # refit, save data/qb_values.json
  python3 -m nfl.qb --ratings                       # current top/bottom starters
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES_CSV = os.path.join(HERE, "data", "games.csv")
QB_WEEK_CSV = os.path.join(HERE, "data", "qb_week.csv")
OVERRIDES_CSV = os.path.join(HERE, "data", "qb_overrides.csv")
PARAMS_JSON = os.path.join(HERE, "data", "qb_values.json")

HALF_LIFE_DROPBACKS = 250.0
MIN_DROPBACKS_FOR_OWN_VALUE = 50   # below this: pinned to replacement level
FULL_CREDIT_DROPBACKS = 250.0      # blend fully to own rolling value by here
REPLACEMENT_PTS_BELOW_AVG = 4.0    # replacement-level QB ~= -4 pts vs average starter
DEFAULT_SCALAR = 36.0              # qb_pts = (value - league_avg) * SCALAR (plan default)
FIT_SINCE, FIT_UNTIL = 2003, 2014


def _rolling_table(qb_week: pd.DataFrame) -> pd.DataFrame:
    """For every QB-week row, compute the LEAK-FREE rolling value (from
    strictly prior weeks of that player's career) and running dropback count
    *before* this game. Processes players independently, in chronological
    order."""
    qb_week = qb_week.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    out_value = np.zeros(len(qb_week))
    out_dropbacks_before = np.zeros(len(qb_week))
    sum_w = {}
    sum_wx = {}
    for idx, r in qb_week.groupby("player_id").indices.items():
        sw, swx = 0.0, 0.0
        for i in r:  # r is already chronological within player due to sort above
            row = qb_week.iloc[i]
            out_value[i] = (swx / sw) if sw > 0 else np.nan
            out_dropbacks_before[i] = sw
            d = float(row["dropbacks"])
            if d <= 0:
                continue
            rate = row["passing_epa"] / d
            decay = 0.5 ** (d / HALF_LIFE_DROPBACKS)
            sw = sw * decay + d
            swx = swx * decay + d * rate
    qb_week = qb_week.copy()
    qb_week["rolling_value_before"] = out_value
    qb_week["dropbacks_before"] = out_dropbacks_before
    return qb_week


def _league_avg(qb_week: pd.DataFrame, since: int = FIT_SINCE, until: int = FIT_UNTIL) -> float:
    w = qb_week[(qb_week["season"] >= since) & (qb_week["season"] <= until)]
    return float(np.average(w["passing_epa"] / w["dropbacks"].replace(0, np.nan),
                            weights=w["dropbacks"]))


def _shrunk_value(rolling_value: float, dropbacks_before: float, league_avg: float,
                  replacement: float) -> float:
    if pd.isna(rolling_value) or dropbacks_before < MIN_DROPBACKS_FOR_OWN_VALUE:
        return replacement
    t = min(1.0, dropbacks_before / FULL_CREDIT_DROPBACKS)
    return t * rolling_value + (1.0 - t) * replacement


def fit() -> dict:
    qb_week = pd.read_csv(QB_WEEK_CSV)
    league_avg = _league_avg(qb_week)
    replacement = league_avg - REPLACEMENT_PTS_BELOW_AVG / DEFAULT_SCALAR

    tbl = _rolling_table(qb_week)
    tbl["shrunk_value"] = [
        _shrunk_value(v, d, league_avg, replacement)
        for v, d in zip(tbl["rolling_value_before"], tbl["dropbacks_before"])
    ]
    # a team can have >1 QB row in a week (mop-up relief duty); the starter is
    # whoever took the most dropbacks that week. Keep one row per
    # (season, week, team) for team-level lookups.
    tbl = (tbl.sort_values("dropbacks", ascending=False)
              .drop_duplicates(subset=["season", "week", "team"], keep="first")
              .sort_values(["season", "week", "team"]).reset_index(drop=True))

    # fit the points-per-epa/dropback scalar on the structural window: OLS of
    # actual margin on (home_value - away_value) using each side's shrunk
    # rolling value as of that game. A simple direct regression (no team-
    # strength control) — a first-order calibration, bounded to a sane range.
    games = pd.read_csv(GAMES_CSV, parse_dates=["date"])
    win = games[(games["season"] >= FIT_SINCE) & (games["season"] <= FIT_UNTIL)]
    key = tbl.set_index(["season", "week", "team"])["shrunk_value"]

    def lookup(season, week, team):
        try:
            return key.loc[(season, week, team)]
        except KeyError:
            return np.nan

    home_val = np.array([lookup(r.season, r.week, r.home) for r in win.itertuples()])
    away_val = np.array([lookup(r.season, r.week, r.away) for r in win.itertuples()])
    margin = (win["home_score"] - win["away_score"]).values.astype(float)
    mask = ~np.isnan(home_val) & ~np.isnan(away_val)
    diff = (home_val[mask] - away_val[mask])
    scalar = float(np.sum(diff * margin[mask]) / np.sum(diff * diff)) if np.sum(diff * diff) > 0 else DEFAULT_SCALAR
    scalar = float(np.clip(scalar, 20.0, 45.0))

    # current per-team "expected starter" = most recent rolling value on file
    latest = tbl.sort_values(["season", "week"]).groupby("team").tail(1)
    current_starters = {
        row["team"]: {"qb_name": row["qb_name"], "value": float(row["shrunk_value"])}
        for _, row in latest.iterrows()
    }

    return {
        "league_avg_epa_pd": league_avg, "replacement_epa_pd": replacement,
        "scalar": scalar, "current_season": int(games["season"].max()),
        "current_starters": current_starters,
        "table": tbl[["season", "week", "team", "qb_name", "shrunk_value", "dropbacks_before"]]
                 .to_dict(orient="records"),
    }


def qb_pts(value: float, params: dict) -> float:
    return (value - params["league_avg_epa_pd"]) * params["scalar"]


def load_params(path: str = PARAMS_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


class QBIndex:
    """Prebuilt lookup structure over params["table"] so walk-forward
    backtests (thousands of games) don't linear-scan the whole table per
    lookup. Build once per params dict with `QBIndex(params)`."""

    def __init__(self, params: dict):
        self.params = params
        by_team: dict[str, list[tuple]] = {}
        by_team_qb: dict[tuple[str, str], list[tuple]] = {}
        exact: dict[tuple[int, int, str], float] = {}
        for r in params["table"]:
            key = (r["season"], r["week"])
            by_team.setdefault(r["team"], []).append((key, r["shrunk_value"]))
            if r["qb_name"]:
                by_team_qb.setdefault((r["team"], r["qb_name"]), []).append((key, r["shrunk_value"]))
            exact[(r["season"], r["week"], r["team"])] = r["shrunk_value"]
        for lst in by_team.values():
            lst.sort(key=lambda t: t[0])
        for lst in by_team_qb.values():
            lst.sort(key=lambda t: t[0])
        self.by_team = by_team
        self.by_team_qb = by_team_qb
        self.exact = exact

    @staticmethod
    def _last_at_or_before(lst: list[tuple], key: tuple, inclusive: bool) -> float | None:
        import bisect
        keys = [k for k, _ in lst]
        i = bisect.bisect_right(keys, key) if inclusive else bisect.bisect_left(keys, key)
        return lst[i - 1][1] if i > 0 else None

    def expected_starter_value(self, team: str, season: int, week: int) -> float:
        row = self.exact.get((season, week, team))
        if row is not None:
            return row
        v = self._last_at_or_before(self.by_team.get(team, []), (season, week), inclusive=False)
        if v is not None:
            return v
        return self.params["current_starters"].get(team, {}).get("value", self.params["replacement_epa_pd"])

    def actual_starter_value(self, team: str, season: int, week: int, qb_name: str | None) -> float:
        if not qb_name:
            return self.expected_starter_value(team, season, week)
        v = self._last_at_or_before(self.by_team_qb.get((team, qb_name), []), (season, week), inclusive=True)
        return v if v is not None else self.params["replacement_epa_pd"]

    def qb_delta_points(self, team: str, season: int, week: int,
                        override_qb_name: str | None = None,
                        actual_qb_name: str | None = None) -> float:
        name = override_qb_name or actual_qb_name
        expected_v = self.expected_starter_value(team, season, week)
        actual_v = self.actual_starter_value(team, season, week, name) if name else expected_v
        return qb_pts(actual_v, self.params) - qb_pts(expected_v, self.params)


def load_overrides(path: str = OVERRIDES_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["season", "week", "team", "qb_name"])
    return pd.read_csv(path)


def qb_delta_points(params: dict, team: str, season: int, week: int,
                    override_qb_name: str | None = None,
                    actual_qb_name: str | None = None,
                    index: "QBIndex | None" = None) -> float:
    """qb_pts(actual) - qb_pts(expected) in points, added to `team`'s predicted
    margin. `override_qb_name` (live, from qb_overrides.csv) takes priority
    over `actual_qb_name` (historical games.csv starter, for backtests). Pass
    a prebuilt `index` (QBIndex(params)) when calling this in a loop."""
    idx = index or QBIndex(params)
    return idx.qb_delta_points(team, season, week, override_qb_name, actual_qb_name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--ratings", action="store_true")
    args = ap.parse_args()

    if args.fit:
        params = fit()
        os.makedirs(os.path.dirname(PARAMS_JSON), exist_ok=True)
        with open(PARAMS_JSON, "w") as f:
            json.dump(params, f)
        print(f"league avg {params['league_avg_epa_pd']:.4f} epa/db, "
              f"replacement {params['replacement_epa_pd']:.4f} epa/db, scalar {params['scalar']:.1f} pts")
        return 0

    params = load_params()
    if args.ratings:
        rows = [(t, v["qb_name"], qb_pts(v["value"], params))
                for t, v in params["current_starters"].items()]
        for t, name, pts in sorted(rows, key=lambda r: -r[2]):
            print(f"{t:<25s} {name or '?':<20s} {pts:>+5.1f} pts vs avg")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
