"""M1 — Point-in-time feature store for the World Cup engine.

Goal (V4_PLAN.md M1): "create the substrate for honest bookmaker-grade
modelling" — per-event feature rows that can be rebuilt for any historical date
without ever reading a row dated on/after that date.

How leakage is prevented (guardrail #2):
  * Strength: `predictor.compute_elo` records each match's PRE-match Elo
    (`elo_h`/`elo_a`); scoring a match with those columns never uses its result.
  * Goal model: `fit_goal_model` is refit at each *calendar-month* boundary using
    only matches strictly before that month (same discipline as `validate.py`),
    cached so the walk-forward build stays fast.
  * Schedule/fatigue: rest days and congestion for a match are computed only from
    that team's EARLIER matches.
  * Market: only the *opening* and *current* odds are features; the *closing*
    line is an OUTCOME/teacher column (see `schema.OUTCOME_COLUMNS`).
  * Result/score attach only as labels, never as features.

Two entry points:
  * `build_training_matrix(since, until)` — historical events with features AND
    outcome labels, every row point-in-time. This is what the validation harness
    and any future model trains on.
  * `build_asof(asof)` — feature rows for fixtures kicking off on/after `asof`,
    using only data dated < `asof`. This is what a live prediction would consume.

Every row carries the M1 acceptance provenance: asof, event_id, source,
fetched_at, schema_version.
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

import predictor as P  # noqa: E402
from dixoncoles import outcome_probs  # noqa: E402  (Dixon-Coles 1X2 from lambdas)
from app.engines.contracts import fixture_key  # noqa: E402  (canonical event_id)

from . import schema  # noqa: E402

DATA = ROOT / "data"
RESULTS_CSV = DATA / "results.csv"
ODDS_HISTORY = DATA / "odds_history.csv"

CONGESTION_WINDOW_DAYS = 14
SOURCE_LABEL = "results.csv+odds_history.csv+elo(point-in-time)"


# ── provenance ────────────────────────────────────────────────────────────────
def _fetched_at() -> str:
    """Most recent mtime across the M1 inputs, as ISO-8601 UTC (provenance)."""
    mtimes = []
    for p in (RESULTS_CSV, ODDS_HISTORY):
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    ts = max(mtimes) if mtimes else datetime.now(timezone.utc).timestamp()
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _stamp(df: pd.DataFrame, asof: str) -> pd.DataFrame:
    """Attach the five M1 provenance columns to every row."""
    df = df.copy()
    df["asof"] = asof
    df["source"] = SOURCE_LABEL
    df["fetched_at"] = _fetched_at()
    df["schema_version"] = schema.SCHEMA_VERSION
    return df


# ── schedule / fatigue (point-in-time) ────────────────────────────────────────
def _schedule_features(played: pd.DataFrame) -> pd.DataFrame:
    """rest_days_{h,a} and congestion_{h,a} from each team's EARLIER matches only.

    Walks the (date-sorted) match list once, keeping, per team, the date of its
    previous match and a window of recent match dates. Because we update the
    per-team state *after* emitting a row, a match never sees itself or any later
    match — point-in-time by construction.
    """
    played = played.sort_values("date").reset_index(drop=True)
    last_date: dict[str, pd.Timestamp] = {}
    recent: dict[str, deque] = defaultdict(deque)
    rest_h = np.full(len(played), np.nan)
    rest_a = np.full(len(played), np.nan)
    cong_h = np.zeros(len(played), dtype=int)
    cong_a = np.zeros(len(played), dtype=int)
    win = pd.Timedelta(days=CONGESTION_WINDOW_DAYS)

    for i, r in enumerate(played.itertuples(index=False)):
        d = r.date
        for team, rest_arr, cong_arr in ((r.home_team, rest_h, cong_h),
                                          (r.away_team, rest_a, cong_a)):
            if team in last_date:
                rest_arr[i] = (d - last_date[team]).days
            dq = recent[team]
            while dq and (d - dq[0]) > win:
                dq.popleft()
            cong_arr[i] = len(dq)
        # update state AFTER emitting (so this match is invisible to itself)
        for team in (r.home_team, r.away_team):
            last_date[team] = d
            recent[team].append(d)

    out = played[["date", "home_team", "away_team"]].copy()
    out["rest_days_h"] = rest_h
    out["rest_days_a"] = rest_a
    out["congestion_h"] = cong_h
    out["congestion_a"] = cong_a
    return out


# ── goal model (walk-forward, month-cached) ───────────────────────────────────
class _MonthlyBeta:
    """fit_goal_model refit at month boundaries on matches strictly before the
    month. Keeps the walk-forward build leak-free without refitting per match."""

    def __init__(self, played_with_elo: pd.DataFrame) -> None:
        self._played = played_with_elo
        self._cache: dict[str, np.ndarray] = {}

    def for_date(self, d: pd.Timestamp) -> np.ndarray:
        key = f"{d.year:04d}-{d.month:02d}"
        if key not in self._cache:
            cutoff = pd.Timestamp(year=d.year, month=d.month, day=1)
            train = self._played[self._played["date"] < cutoff]
            if len(train) < 500:  # not enough history yet — widen to all-before
                train = self._played[self._played["date"] < d]
            self._cache[key] = P.fit_goal_model(train)
        return self._cache[key]


def _probs_from_elo(elo_h: float, elo_a: float, beta: np.ndarray,
                    neutral: bool) -> tuple[float, float, float, float, float]:
    adv = 0.0 if neutral else P.HOME_ADV
    lam_h, lam_a = P.expected_goals(elo_h, elo_a, beta, adv)
    p_h, p_d, p_a = outcome_probs(lam_h, lam_a, P.DC_RHO)[:3]
    return lam_h, lam_a, float(p_h), float(p_d), float(p_a)


# ── odds history (open / current / close) ─────────────────────────────────────
_SIDE_COLS = {"h": "home", "d": "draw", "a": "away"}


def _odds_frame() -> pd.DataFrame | None:
    if not ODDS_HISTORY.exists():
        return None
    h = pd.read_csv(ODDS_HISTORY)
    if h.empty:
        return None
    h["snapshot_time"] = pd.to_datetime(h["snapshot_time"], utc=True,
                                        errors="coerce")
    h["match_date"] = h["match_date"].astype(str)
    return h.dropna(subset=["snapshot_time"])


def _odds_for_event(hist: pd.DataFrame, match_date: str, home: str, away: str,
                    asof_ts: pd.Timestamp | None,
                    kickoff_ts: pd.Timestamp | None) -> dict[str, float]:
    """Opening / current / closing decimal odds per side for one event.

    opening  = earliest snapshot for the event;
    current  = latest snapshot at or before `asof_ts` (all snapshots if None);
    closing  = latest snapshot at or before `kickoff_ts` (the teacher line).
    """
    ev = hist[(hist["match_date"] == match_date) & (hist["home"] == home)
              & (hist["away"] == away)]
    out: dict[str, float] = {}
    if ev.empty:
        return out
    for sk, sidename in _SIDE_COLS.items():
        s = ev[ev["side"] == sidename].sort_values("snapshot_time")
        if s.empty:
            continue
        out[f"odds_open_{sk}"] = float(s.iloc[0]["odds"])
        cur = s if asof_ts is None else s[s["snapshot_time"] <= asof_ts]
        if not cur.empty:
            out[f"odds_curr_{sk}"] = float(cur.iloc[-1]["odds"])
        clo = s if kickoff_ts is None else s[s["snapshot_time"] <= kickoff_ts]
        if not clo.empty:
            out[f"odds_close_{sk}"] = float(clo.iloc[-1]["odds"])
    return out


def _devig3(oh: float, od: float, oa: float) -> tuple[float, float, float]:
    inv = np.array([1.0 / oh, 1.0 / od, 1.0 / oa])
    p = inv / inv.sum()
    return float(p[0]), float(p[1]), float(p[2])


def _attach_market(row: dict, odds: dict) -> None:
    """Fold opening/current/closing odds into a feature row.

    Current odds -> de-vigged p_market (a FEATURE). Closing odds -> p_close (an
    OUTCOME/teacher column). Movement = current minus opening implied prob.
    """
    for k in ("odds_open_h", "odds_open_d", "odds_open_a",
              "odds_curr_h", "odds_curr_d", "odds_curr_a",
              "odds_close_h", "odds_close_d", "odds_close_a"):
        row[k] = odds.get(k, np.nan)

    if all(np.isfinite(row.get(f"odds_curr_{s}", np.nan)) for s in "hda"):
        ph, pd_, pa = _devig3(row["odds_curr_h"], row["odds_curr_d"],
                              row["odds_curr_a"])
        row["p_market_h"], row["p_market_d"], row["p_market_a"] = ph, pd_, pa
        row["book_dispersion"] = 0.0  # single-book history: dispersion report-only
    if all(np.isfinite(row.get(f"odds_open_{s}", np.nan)) for s in "hda") and \
       all(np.isfinite(row.get(f"odds_curr_{s}", np.nan)) for s in "hda"):
        oh, od, oa = _devig3(row["odds_open_h"], row["odds_open_d"],
                             row["odds_open_a"])
        row["move_open_curr_h"] = row["p_market_h"] - oh
        row["move_open_curr_d"] = row["p_market_d"] - od
        row["move_open_curr_a"] = row["p_market_a"] - oa
    if all(np.isfinite(row.get(f"odds_close_{s}", np.nan)) for s in "hda"):
        ph, pd_, pa = _devig3(row["odds_close_h"], row["odds_close_d"],
                              row["odds_close_a"])
        row["p_close_h"], row["p_close_d"], row["p_close_a"] = ph, pd_, pa


# ── column ordering ───────────────────────────────────────────────────────────
def _ordered_columns(include_outcomes: bool) -> list[str]:
    cols = (schema.PROVENANCE_COLUMNS + schema.ID_COLUMNS
            + schema.FEATURE_COLUMNS)
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
def build_training_matrix(since: str | None = "2018-01-01",
                          until: str | None = None,
                          tournaments: tuple[str, ...] | None = None
                          ) -> pd.DataFrame:
    """Historical events with point-in-time features AND outcome labels.

    Every row is leak-free: pre-match Elo, a month-boundary goal model, earlier-
    only schedule features, and opening/current odds. The result and the closing
    line attach as OUTCOME columns. `asof` is set per row to the match date — i.e.
    "what was knowable the moment before kickoff".
    """
    played, _ = P.load_matches()
    _, played = P.compute_elo(played)  # adds point-in-time elo_h/elo_a
    sched = _schedule_features(played)
    played = played.merge(sched, on=["date", "home_team", "away_team"],
                          how="left")
    if since:
        played = played[played["date"] >= pd.Timestamp(since)]
    if until:
        played = played[played["date"] < pd.Timestamp(until)]
    if tournaments:
        played = played[played["tournament"].isin(tournaments)]
    played = played.sort_values("date").reset_index(drop=True)

    # The month cache must see ALL history before each event, so build it from
    # the full elo-tagged frame, not the filtered window.
    _, full = P.compute_elo(P.load_matches()[0])
    mbeta = _MonthlyBeta(full)
    hist = _odds_frame()

    rows: list[dict] = []
    for r in played.itertuples(index=False):
        beta = mbeta.for_date(r.date)
        lam_h, lam_a, p_h, p_d, p_a = _probs_from_elo(
            r.elo_h, r.elo_a, beta, bool(r.neutral))
        md = pd.Timestamp(r.date).strftime("%Y-%m-%d")
        eid = fixture_key(md, r.home_team, r.away_team, r.tournament)
        result = "H" if r.home_score > r.away_score else (
            "D" if r.home_score == r.away_score else "A")
        row: dict[str, Any] = {
            "event_id": eid, "match_date": md,
            "home": r.home_team, "away": r.away_team,
            "competition": r.tournament, "neutral": bool(r.neutral),
            "elo_h": float(r.elo_h), "elo_a": float(r.elo_a),
            "elo_diff": float(r.elo_h - r.elo_a),
            "lam_h": float(lam_h), "lam_a": float(lam_a),
            "p_model_h": p_h, "p_model_d": p_d, "p_model_a": p_a,
            "rest_days_h": r.rest_days_h, "rest_days_a": r.rest_days_a,
            "congestion_h": int(r.congestion_h), "congestion_a": int(r.congestion_a),
            "home_score": int(r.home_score), "away_score": int(r.away_score),
            "result": result,
        }
        if hist is not None:
            kickoff = pd.Timestamp(r.date).tz_localize("UTC") + pd.Timedelta(hours=12)
            _attach_market(row, _odds_for_event(hist, md, r.home_team,
                                                r.away_team, kickoff, kickoff))
        rows.append({**row, "asof": md})

    df = _frame_from_rows(rows, include_outcomes=True)
    df["source"] = SOURCE_LABEL
    df["fetched_at"] = _fetched_at()
    df["schema_version"] = schema.SCHEMA_VERSION
    # belt-and-braces: the legal feature set must contain no teacher column
    schema.feature_columns(df.columns)
    return df


def build_asof(asof: str, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature rows for fixtures kicking off on/after `asof`, using ONLY data
    dated strictly before `asof`. This is the live-prediction path.

    Ratings and the goal model are computed from matches before `asof`; the
    fixtures default to the unplayed World Cup schedule in `results.csv`.
    """
    asof_ts = pd.Timestamp(asof)
    all_played, upcoming = P.load_matches()
    before = all_played[all_played["date"] < asof_ts]
    ratings, before_elo = P.compute_elo(before)   # ratings dict AS OF asof
    beta = P.fit_goal_model(before_elo)
    sched = _schedule_features(before_elo)

    # last-known schedule state per team, strictly before asof
    last_date: dict[str, pd.Timestamp] = {}
    recent: dict[str, list] = defaultdict(list)
    for r in before.sort_values("date").itertuples(index=False):
        last_date[r.home_team] = r.date
        last_date[r.away_team] = r.date
        recent[r.home_team].append(r.date)
        recent[r.away_team].append(r.date)

    def _rest_cong(team: str) -> tuple[float, int]:
        rest = (asof_ts - last_date[team]).days if team in last_date else np.nan
        win = pd.Timedelta(days=CONGESTION_WINDOW_DAYS)
        cong = sum(1 for d in recent.get(team, []) if (asof_ts - d) <= win)
        return rest, cong

    if fixtures is None:
        fixtures = upcoming[upcoming["date"] >= asof_ts].copy()
    hist = _odds_frame()

    rows: list[dict] = []
    for r in fixtures.itertuples(index=False):
        home, away = r.home_team, r.away_team
        if home not in ratings or away not in ratings:
            continue  # unrated team — fall through (more complex models fail closed)
        neutral = bool(getattr(r, "neutral", True))
        lam_h, lam_a, p_h, p_d, p_a = _probs_from_elo(
            ratings[home], ratings[away], beta, neutral)
        md = pd.Timestamp(r.date).strftime("%Y-%m-%d")
        comp = getattr(r, "tournament", "")
        eid = fixture_key(md, home, away, comp)
        rest_h, cong_h = _rest_cong(home)
        rest_a, cong_a = _rest_cong(away)
        row: dict[str, Any] = {
            "event_id": eid, "match_date": md, "home": home, "away": away,
            "competition": comp, "neutral": neutral,
            "elo_h": float(ratings[home]), "elo_a": float(ratings[away]),
            "elo_diff": float(ratings[home] - ratings[away]),
            "lam_h": float(lam_h), "lam_a": float(lam_a),
            "p_model_h": p_h, "p_model_d": p_d, "p_model_a": p_a,
            "rest_days_h": rest_h, "rest_days_a": rest_a,
            "congestion_h": cong_h, "congestion_a": cong_a,
        }
        if hist is not None:
            kickoff = pd.Timestamp(r.date).tz_localize("UTC") + pd.Timedelta(hours=12)
            _attach_market(row, _odds_for_event(hist, md, home, away,
                                                asof_ts.tz_localize("UTC"), kickoff))
        rows.append(row)

    df = _frame_from_rows(rows, include_outcomes=False)
    return _stamp(df, asof)


if __name__ == "__main__":  # pragma: no cover — manual smoke
    import argparse
    ap = argparse.ArgumentParser(description="V4 M1 point-in-time feature store")
    ap.add_argument("--asof", help="build live feature rows as of this date")
    ap.add_argument("--since", default="2022-01-01",
                    help="training-matrix start date")
    args = ap.parse_args()
    if args.asof:
        out = build_asof(args.asof)
        print(f"as-of {args.asof}: {len(out)} fixtures")
    else:
        out = build_training_matrix(since=args.since)
        print(f"training matrix since {args.since}: {len(out)} events, "
              f"{out['result'].notna().sum()} labelled")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.head(8).to_string(index=False))
