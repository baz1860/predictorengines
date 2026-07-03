"""Point-in-time feature store for the Club Soccer engine.

Mirrors `wc_v4/feature_store.py`'s discipline: every row must be buildable
from data strictly dated before the match, so a backtest is honest.

How leakage is prevented:
  * Strength (Elo + ensemble p_model/lam): `model.fit` is refit at each
    *calendar-month* boundary using only matches strictly before that month
    (same discipline as `validate.py`'s walk-forward), cached so the build
    stays fast. `elo_h`/`elo_a` come from that same point-in-time fit.
  * Schedule/fatigue: rest days and 7/14/30-day match counts are computed
    across ALL competitions per club (a Wednesday Champions League match
    counts toward Saturday's league-match rest/congestion) from each team's
    STRICTLY EARLIER matches only.
  * Market: `market_history.csv` (see fetch_fdcouk.py) holds pre-match/
    closing odds only — these are OUTCOME/teacher columns, never features.
  * Result attaches only as a label, never as a feature.

Two entry points:
  * `build_training_matrix(since, until, competitions)` — historical matches
    with features AND outcome labels, every row point-in-time.
  * `build_asof(asof, fixtures)` — feature rows for fixtures kicking off
    on/after `asof`, using only data dated < `asof`. The live-prediction path.
"""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import model as M
from . import schema
from .player_features import PlayerFeatureStore

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
MARKET_HISTORY = DATA / "market_history.csv"

MIN_TRAIN = 200          # matches required before trusting a month-boundary fit
SOURCE_LABEL = "fixtures.csv+market_history.csv+model.fit(point-in-time)"

_ZERO_XI_LOAD = {"xi_load_7d": 0.0, "xi_load_14d": 0.0, "xi_load_30d": 0.0}


class _MinutesCache:
    """xi_load_7d/14d/30d per team, cached per as-of date (many rows on the
    same match date share it). asof is always the day BEFORE the match, so
    "apps dated < match date" (point-in-time — a match's own player minutes
    must never feature in its own prediction)."""

    def __init__(self, store: PlayerFeatureStore) -> None:
        self._store = store
        self._minutes_cache: dict[str, pd.DataFrame] = {}

    def _minutes_asof(self, match_date: str) -> pd.DataFrame:
        asof = str(pd.Timestamp(match_date).date() - pd.Timedelta(days=1))
        df = self._minutes_cache.get(asof)
        if df is None:
            from .minutes import build_player_minutes
            df = build_player_minutes(self._store, asof)
            self._minutes_cache[asof] = df
        return df

    def xi_loads(self, team: str, match_date: str) -> dict[str, float]:
        from .minutes import xi_loads
        df = self._minutes_asof(match_date)
        if df.empty:
            return dict(_ZERO_XI_LOAD)
        return xi_loads(df, team)

    def has_data(self, team: str, match_date: str) -> bool:
        """Whether the minutes-load table had any squad entries for `team`
        as of `match_date` — used to gauge xi_load coverage for the P3
        context GLM (a 0.0 xi_load is ambiguous between "no data" and
        "genuinely idle squad" without this)."""
        df = self._minutes_asof(match_date)
        return not df.empty and bool((df["team"] == team).any())


# ── provenance ────────────────────────────────────────────────────────────────
def _fetched_at() -> str:
    mtimes = [p.stat().st_mtime for p in (FIXTURES, MARKET_HISTORY) if p.exists()]
    ts = max(mtimes) if mtimes else datetime.now(timezone.utc).timestamp()
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _stamp(df: pd.DataFrame, asof: str) -> pd.DataFrame:
    df = df.copy()
    df["asof"] = asof
    df["source"] = SOURCE_LABEL
    df["fetched_at"] = _fetched_at()
    df["schema_version"] = schema.SCHEMA_VERSION
    return df


# ── schedule / fatigue (point-in-time, ALL competitions per club) ─────────────
def _schedule_features(played: pd.DataFrame) -> pd.DataFrame:
    """rest_days_{h,a} and matches_{7,14,30}d_{h,a} from each team's EARLIER
    matches only, across every competition it plays in (league + cup + Europe).

    Per-team state is updated *after* emitting a row, so a match never sees
    itself or any later match — point-in-time by construction.
    """
    played = played.sort_values("date").reset_index(drop=True)
    n = len(played)
    last_date: dict[str, pd.Timestamp] = {}
    recent: dict[str, deque] = defaultdict(deque)
    rest_h = np.full(n, np.nan); rest_a = np.full(n, np.nan)
    m7h = np.zeros(n, dtype=int); m7a = np.zeros(n, dtype=int)
    m14h = np.zeros(n, dtype=int); m14a = np.zeros(n, dtype=int)
    m30h = np.zeros(n, dtype=int); m30a = np.zeros(n, dtype=int)
    win30 = pd.Timedelta(days=30)

    for i, r in enumerate(played.itertuples(index=False)):
        d = r.date
        for team, rest_arr, c7, c14, c30 in (
            (r.home, rest_h, m7h, m14h, m30h),
            (r.away, rest_a, m7a, m14a, m30a),
        ):
            if team in last_date:
                rest_arr[i] = float((d - last_date[team]).days)
            dq = recent[team]
            while dq and (d - dq[0]) > win30:
                dq.popleft()
            c30[i] = len(dq)
            c14[i] = sum(1 for x in dq if (d - x).days <= 14)
            c7[i] = sum(1 for x in dq if (d - x).days <= 7)
        # update state AFTER emitting (so this match is invisible to itself)
        for team in (r.home, r.away):
            last_date[team] = d
            recent[team].append(d)

    out = played[["fixture_id", "date", "home", "away"]].copy()
    out["rest_days_h"] = rest_h; out["rest_days_a"] = rest_a
    out["matches_7d_h"] = m7h; out["matches_7d_a"] = m7a
    out["matches_14d_h"] = m14h; out["matches_14d_a"] = m14a
    out["matches_30d_h"] = m30h; out["matches_30d_a"] = m30a
    return out


def _schedule_state_asof(before: pd.DataFrame, asof_ts: pd.Timestamp):
    """Per-team (last_date, recent-30d-dates) as of `asof_ts`, from `before`
    (all competitions, already filtered to dates < asof_ts)."""
    last_date: dict[str, pd.Timestamp] = {}
    recent: dict[str, list] = defaultdict(list)
    win30 = pd.Timedelta(days=30)
    for r in before.sort_values("date").itertuples(index=False):
        for team in (r.home, r.away):
            last_date[team] = r.date
            recent[team] = [d for d in recent[team] if (r.date - d) <= win30] + [r.date]

    def _rest_cong(team: str) -> tuple[float, int, int, int]:
        if team not in last_date:
            return np.nan, 0, 0, 0
        rest = float((asof_ts - last_date[team]).days)
        dates = [d for d in recent.get(team, []) if (asof_ts - d) <= win30]
        c30 = len(dates)
        c14 = sum(1 for d in dates if (asof_ts - d).days <= 14)
        c7 = sum(1 for d in dates if (asof_ts - d).days <= 7)
        return rest, c7, c14, c30

    return _rest_cong


# ── strength (walk-forward, month-cached model.fit) ───────────────────────────
class _MonthlyParams:
    """model.fit refit at month boundaries on matches strictly before the
    month. Keeps the walk-forward build leak-free without refitting per match."""

    def __init__(self, all_played: pd.DataFrame) -> None:
        self._played = all_played
        self._cache: dict[str, dict] = {}

    def for_date(self, d: pd.Timestamp) -> dict:
        key = f"{d.year:04d}-{d.month:02d}"
        if key not in self._cache:
            cutoff = pd.Timestamp(year=d.year, month=d.month, day=1)
            train = self._played[self._played["date"] < cutoff]
            if len(train) < MIN_TRAIN:  # not enough history — widen to all-before
                train = self._played[self._played["date"] < d]
            self._cache[key] = M.fit(train)
        return self._cache[key]


# ── market history (closing/pre-match odds; OUTCOME columns only) ─────────────
def _market_frame() -> pd.DataFrame | None:
    if not MARKET_HISTORY.exists():
        return None
    h = pd.read_csv(MARKET_HISTORY)
    if h.empty:
        return None
    h["match_date"] = pd.to_datetime(h["match_date"]).dt.strftime("%Y-%m-%d")
    return h


def _devig3(oh: float, od: float, oa: float) -> tuple[float, float, float]:
    inv = np.array([1.0 / oh, 1.0 / od, 1.0 / oa])
    p = inv / inv.sum()
    return float(p[0]), float(p[1]), float(p[2])


def _attach_market(row: dict, mrow: pd.Series | None) -> None:
    """Fold fd.co.uk closing/pre-match odds into a feature row as OUTCOME
    columns (Pinnacle closing when available, else Bet365 pre-match)."""
    for k in ("p_close_h", "p_close_d", "p_close_a",
              "odds_close_h", "odds_close_d", "odds_close_a",
              "odds_close_over25", "odds_close_under25"):
        row[k] = np.nan
    if mrow is None:
        return
    oh = mrow.get("psc_h"); od = mrow.get("psc_d"); oa = mrow.get("psc_a")
    if pd.isna(oh) or pd.isna(od) or pd.isna(oa):
        oh, od, oa = mrow.get("b365_h"), mrow.get("b365_d"), mrow.get("b365_a")
    if pd.notna(oh) and pd.notna(od) and pd.notna(oa) and float(oh) > 1 and float(od) > 1 and float(oa) > 1:
        row["odds_close_h"], row["odds_close_d"], row["odds_close_a"] = float(oh), float(od), float(oa)
        ph, pdd, pa = _devig3(float(oh), float(od), float(oa))
        row["p_close_h"], row["p_close_d"], row["p_close_a"] = ph, pdd, pa
    ov, un = mrow.get("b365_over25"), mrow.get("b365_under25")
    if pd.notna(ov) and float(ov) > 1:
        row["odds_close_over25"] = float(ov)
    if pd.notna(un) and float(un) > 1:
        row["odds_close_under25"] = float(un)


# ── column ordering ───────────────────────────────────────────────────────────
def _ordered_columns(include_outcomes: bool) -> list[str]:
    cols = schema.PROVENANCE_COLUMNS + schema.ID_COLUMNS + schema.FEATURE_COLUMNS
    if include_outcomes:
        cols = cols + schema.OUTCOME_COLUMNS
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _frame_from_rows(rows: list[dict], include_outcomes: bool) -> pd.DataFrame:
    cols = _ordered_columns(include_outcomes)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


# ── public builders ───────────────────────────────────────────────────────────
def build_training_matrix(since: str = "2022-08-01", until: str | None = None,
                          competitions: list[str] | None = None) -> pd.DataFrame:
    """Historical matches with point-in-time features AND outcome labels."""
    fx = M.load_fixtures()
    played_all = M.played(fx).sort_values("date").reset_index(drop=True)
    sched = _schedule_features(played_all)
    played_all = played_all.merge(
        sched.drop(columns=["date", "home", "away"]), on="fixture_id", how="left")

    window = played_all
    if since:
        window = window[window["date"] >= pd.Timestamp(since)]
    if until:
        window = window[window["date"] < pd.Timestamp(until)]
    if competitions:
        window = window[window["competition"].isin(competitions)]
    window = window.sort_values("date").reset_index(drop=True)

    # The month cache must see ALL history before each event, so build it from
    # the full played frame, not the filtered window.
    mparams = _MonthlyParams(played_all)
    hist = _market_frame()
    mcache = _MinutesCache(PlayerFeatureStore().load())

    rows: list[dict] = []
    skipped = 0
    for r in window.itertuples(index=False):
        params = mparams.for_date(r.date)
        if r.home not in set(params["teams"]) or r.away not in set(params["teams"]):
            skipped += 1
            continue
        try:
            pred = M.predict(r.home, r.away, r.competition, "ensemble",
                             bool(r.neutral), params=params)
        except ValueError:
            skipped += 1
            continue
        md = pd.Timestamp(r.date).strftime("%Y-%m-%d")
        eid = f"{r.fixture_id}"
        result = "H" if r.home_goals > r.away_goals else (
            "D" if r.home_goals == r.away_goals else "A")
        elo = params["elo"]
        row: dict[str, Any] = {
            "event_id": eid, "match_date": md, "home": r.home, "away": r.away,
            "competition": r.competition, "season": r.season, "neutral": bool(r.neutral),
            "elo_h": float(elo.get(r.home, M.BASE_ELO)),
            "elo_a": float(elo.get(r.away, M.BASE_ELO)),
            "elo_diff": float(elo.get(r.home, M.BASE_ELO) - elo.get(r.away, M.BASE_ELO)),
            "lam_h": float(pred["xg_home"]), "lam_a": float(pred["xg_away"]),
            "p_model_h": pred["probs"]["home"], "p_model_d": pred["probs"]["draw"],
            "p_model_a": pred["probs"]["away"],
            "rest_days_h": r.rest_days_h, "rest_days_a": r.rest_days_a,
            "matches_7d_h": int(r.matches_7d_h), "matches_7d_a": int(r.matches_7d_a),
            "matches_14d_h": int(r.matches_14d_h), "matches_14d_a": int(r.matches_14d_a),
            "matches_30d_h": int(r.matches_30d_h), "matches_30d_a": int(r.matches_30d_a),
            "home_goals": float(r.home_goals), "away_goals": float(r.away_goals),
            "result": result,
        }
        xi_h, xi_a = mcache.xi_loads(r.home, md), mcache.xi_loads(r.away, md)
        row["xi_load_7d_h"], row["xi_load_7d_a"] = xi_h["xi_load_7d"], xi_a["xi_load_7d"]
        row["xi_load_14d_h"], row["xi_load_14d_a"] = xi_h["xi_load_14d"], xi_a["xi_load_14d"]
        row["xi_load_30d_h"], row["xi_load_30d_a"] = xi_h["xi_load_30d"], xi_a["xi_load_30d"]
        if hist is not None:
            mrows = hist[(hist["match_date"] == md) & (hist["home"] == r.home)
                        & (hist["away"] == r.away)]
            _attach_market(row, mrows.iloc[0] if not mrows.empty else None)
        else:
            _attach_market(row, None)
        rows.append({**row, "asof": md})

    df = _frame_from_rows(rows, include_outcomes=True)
    df["source"] = SOURCE_LABEL
    df["fetched_at"] = _fetched_at()
    df["schema_version"] = schema.SCHEMA_VERSION
    # belt-and-braces: the legal feature set must contain no teacher column
    schema.feature_columns(df.columns)
    if skipped:
        print(f"  feature_store: skipped {skipped} rows (team unseen in its "
              f"training window)")
    return df


def build_asof(asof: str, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature rows for fixtures kicking off on/after `asof`, using ONLY data
    dated strictly before `asof`. The live-prediction path."""
    asof_ts = pd.Timestamp(asof)
    fx = M.load_fixtures()
    played_all = M.played(fx)
    before = played_all[played_all["date"] < asof_ts].sort_values("date").reset_index(drop=True)
    if before.empty:
        raise ValueError(f"No played matches before {asof} to build features from.")
    params = M.fit(before)
    rest_cong = _schedule_state_asof(before, asof_ts)

    if fixtures is None:
        fixtures = M.upcoming(fx)
        fixtures = fixtures[fixtures["date"] >= asof_ts]

    from .minutes import build_player_minutes, xi_loads as _xi_loads
    minutes_df = build_player_minutes(PlayerFeatureStore().load(), asof)

    rows: list[dict] = []
    teams = set(params["teams"])
    for r in fixtures.itertuples(index=False):
        home, away = r.home, r.away
        if home not in teams or away not in teams:
            continue  # unrated team — fail closed, matches build_training_matrix
        neutral = bool(getattr(r, "neutral", 0))
        pred = M.predict(home, away, r.competition, "ensemble", neutral, params=params)
        md = pd.Timestamp(r.date).strftime("%Y-%m-%d")
        eid = f"{getattr(r, 'fixture_id', '')}"
        rest_h, m7h, m14h, m30h = rest_cong(home)
        rest_a, m7a, m14a, m30a = rest_cong(away)
        elo = params["elo"]
        row: dict[str, Any] = {
            "event_id": eid, "match_date": md, "home": home, "away": away,
            "competition": r.competition, "season": getattr(r, "season", None),
            "neutral": neutral,
            "elo_h": float(elo.get(home, M.BASE_ELO)),
            "elo_a": float(elo.get(away, M.BASE_ELO)),
            "elo_diff": float(elo.get(home, M.BASE_ELO) - elo.get(away, M.BASE_ELO)),
            "lam_h": float(pred["xg_home"]), "lam_a": float(pred["xg_away"]),
            "p_model_h": pred["probs"]["home"], "p_model_d": pred["probs"]["draw"],
            "p_model_a": pred["probs"]["away"],
            "rest_days_h": rest_h, "rest_days_a": rest_a,
            "matches_7d_h": m7h, "matches_7d_a": m7a,
            "matches_14d_h": m14h, "matches_14d_a": m14a,
            "matches_30d_h": m30h, "matches_30d_a": m30a,
        }
        xi_h = _xi_loads(minutes_df, home) if not minutes_df.empty else dict(_ZERO_XI_LOAD)
        xi_a = _xi_loads(minutes_df, away) if not minutes_df.empty else dict(_ZERO_XI_LOAD)
        row["xi_load_7d_h"], row["xi_load_7d_a"] = xi_h["xi_load_7d"], xi_a["xi_load_7d"]
        row["xi_load_14d_h"], row["xi_load_14d_a"] = xi_h["xi_load_14d"], xi_a["xi_load_14d"]
        row["xi_load_30d_h"], row["xi_load_30d_a"] = xi_h["xi_load_30d"], xi_a["xi_load_30d"]
        rows.append(row)

    df = _frame_from_rows(rows, include_outcomes=False)
    return _stamp(df, asof)


if __name__ == "__main__":  # pragma: no cover — manual smoke
    import argparse
    ap = argparse.ArgumentParser(description="Club Soccer point-in-time feature store")
    ap.add_argument("--asof", help="build live feature rows as of this date")
    ap.add_argument("--since", default="2022-08-01", help="training-matrix start date")
    args = ap.parse_args()
    if args.asof:
        out = build_asof(args.asof)
        print(f"as-of {args.asof}: {len(out)} fixtures")
    else:
        out = build_training_matrix(since=args.since)
        print(f"training matrix since {args.since}: {len(out)} matches, "
              f"{out['result'].notna().sum()} labelled")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.head(8).to_string(index=False))
