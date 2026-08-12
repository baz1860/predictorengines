#!/usr/bin/env python3
"""CFB walk-forward validation gate (V3 M3).

Consolidates the useful pieces of predictor.py --backtest, blend_eval.py,
ats_backtest.py and totals_backtest.py into ONE leakage-free walk-forward pass
with a stored baseline and a regression gate, matching the other engines.

Walk-forward discipline (no fitting on future games):
  * Elo is updated game-by-game; its spread slope is fitted only on seasons
    strictly before `--since`.
  * Power ratings are refit before each week with `asof = first kickoff of the
    week`, so a week is scored by a model that never saw that week.

Metrics stored in data/validation_baseline.json:
  * ml_brier    – 50/50 blend moneyline Brier
  * margin_mae  – blend margin MAE
  * total_mae   – power total MAE
  * ats_roi     – ROI per disagreement threshold vs closing spreads
  * totals_roi  – ROI per threshold vs closing totals

Gate fails (exit 1) if Brier or either MAE regresses past tolerance. ROI is
recorded for visibility but not gated (too noisy to gate on). Baseline is only
ever loosened with an explicit --update-baseline.

Usage:
  python3 validate.py [--since 2023] [--gate] [--quiet] [--update-baseline]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date

import numpy as np
import pandas as pd

from . import elo as E
from . import epa as X
from . import power as P
from . import dataset_fingerprint as DF
from .predictor import load_blend_weight, DEFAULT_W_ELO, _BLEND_WEIGHT_FILE
from .ats_backtest import SPREADS_CSV, settle as ats_settle
from .totals_backtest import TOTALS_CSV, settle as totals_settle

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "data", "validation_baseline.json")
NESTED_ARTIFACT = os.path.join(HERE, "data", "nested_validation_2025.json")

THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 4.0)
BRIER_TOL = 0.005       # blend moneyline Brier may not regress beyond this
MAE_TOL = 0.50          # margin/total MAE may not regress beyond this (points)


def walk_forward(games: pd.DataFrame, since: int, quiet: bool = False,
                 w_elo: float | None = None,
                 include_epa: bool = False) -> pd.DataFrame:
    """Per-game blended predictions for seasons >= since, games-indexed.

    `w_elo` is the weight on Elo in the win-prob/margin blend; None loads the
    stored weight (default 0.5). Raw `p_elo`/`p_pow`/`m_elo`/`m_pow` are always
    kept so the tuner can rescore any weight without re-running the walk.

    With `include_epa=True`, the same walk also fits EPA/PPA ratings before each
    week and appends `p_epa`/`m_epa`/`t_epa`. Default validation keeps this off so
    the champion path and historical baseline are unchanged."""
    carry, offs = E.season_priors()
    _, history = E.run_elo(games, record_pregame=True, carry=carry, prior_offsets=offs)
    diffs = np.array([h[2] for h in history])
    # Same masked fit as production elo.fit_spread_map: champion-ledger rows
    # only. FCS-vs-FCS rows carry FCS-ledger diffs and must not enter the fit.
    slope = E.fit_slope(games, history, (games["season"] < since).values)

    epa_data = X.load_ppa() if include_epa else None
    ev = games[(games["season"] >= since)
               & (games["home_div"] == "fbs") & (games["away_div"] == "fbs")]
    rows, idx = [], []
    for (season, week, _stype), wk in ev.groupby(["season", "week", "season_type"],
                                                 sort=False):
        asof = wk["date"].min()
        try:
            pparams = P.fit(games, asof=asof)
            xparams = X.fit(asof=asof, data=epa_data) if include_epa else None
        except ValueError:
            continue
        if not quiet:
            print(f"  fit week {season} w{week} ({len(wk)} games)", file=sys.stderr)
        for r in wk.itertuples():
            if r.home not in pparams["teams"] or r.away not in pparams["teams"]:
                continue
            if include_epa and (r.home not in xparams["teams"] or r.away not in xparams["teams"]):
                continue
            d = diffs[r.Index]
            pp = P.predict(pparams, r.home, r.away, neutral=bool(r.neutral))
            row = {
                "season": int(r.season), "week": int(r.week),
                "home_team": r.home_team, "away_team": r.away_team,
                "p_elo": E.win_prob(d), "p_pow": pp["p1"],
                "m_elo": slope * d, "m_pow": pp["margin"], "t_pow": pp["total"],
                "sigma_margin": float(pparams["sigma"]),
                "sigma_total": float(pparams["sigma_total"]),
                "margin": r.home_points - r.away_points,
                "total": r.home_points + r.away_points,
            }
            if include_epa:
                xp = X.predict(xparams, r.home, r.away, neutral=bool(r.neutral))
                row.update({"p_epa": xp["p1"], "m_epa": xp["margin"], "t_epa": xp["total"]})
            rows.append(row)
            idx.append(r.Index)
    df = pd.DataFrame(rows, index=idx)
    if df.empty:
        return df
    w = load_blend_weight() if w_elo is None else float(w_elo)
    df["p_blend"] = w * df["p_elo"] + (1.0 - w) * df["p_pow"]
    df["m_blend"] = w * df["m_elo"] + (1.0 - w) * df["m_pow"]
    return df


def _roi_by_threshold(df: pd.DataFrame, lines_csv: str, kind: str,
                      margin_col: str = "m_blend",
                      total_col: str = "t_pow") -> tuple[dict, dict]:
    """ROI + bet count per threshold for ATS (kind='ats') or totals ('total')."""
    if not os.path.exists(lines_csv):
        return {}, {}
    lines = pd.read_csv(lines_csv)
    g = df.merge(lines, on=["season", "week", "home_team", "away_team"], how="inner")
    if g.empty:
        return {}, {}
    if kind == "ats":
        g["edge_pts"] = g[margin_col] + g["home_line"]    # >0: model likes home
        settle = ats_settle
    else:
        g["edge_pts"] = g[total_col] - g["total_line"]     # >0: model says over
        settle = totals_settle
    roi, n = {}, {}
    for thr in THRESHOLDS:
        b = g[g["edge_pts"].abs() >= thr]
        if len(b) < 20:
            continue
        w, l, p, pnl = settle(b)
        roi[f"{thr:.1f}"] = round(float(pnl.mean()), 4)
        n[f"{thr:.1f}"] = int(len(b))
    return roi, n


def _norm_stack(weights: dict[str, float]) -> dict[str, float]:
    vals = {k: max(0.0, float(weights.get(k, 0.0)))
            for k in ("elo", "power", "epa")}
    s = sum(vals.values())
    if s <= 0:
        return {"elo": 0.5, "power": 0.5, "epa": 0.0}
    return {k: v / s for k, v in vals.items()}


def _stack_predictions(df: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return win prob, margin, total for an Elo/power/EPA stack.

    Totals come from power/EPA only. If a candidate is pure Elo, totals remain the
    champion power total because Elo has no total model.
    """
    w = _norm_stack(weights)
    p = w["elo"] * df["p_elo"] + w["power"] * df["p_pow"] + w["epa"] * df["p_epa"]
    m = w["elo"] * df["m_elo"] + w["power"] * df["m_pow"] + w["epa"] * df["m_epa"]
    total_w = w["power"] + w["epa"]
    if total_w <= 0:
        t = df["t_pow"]
    else:
        t = (w["power"] * df["t_pow"] + w["epa"] * df["t_epa"]) / total_w
    return p, m, t


def score_stack(df: pd.DataFrame, weights: dict[str, float]) -> dict:
    """Score a model-stack candidate on the walk-forward frame."""
    p, m, t = _stack_predictions(df, weights)
    res = (df["margin"] > 0).astype(float)
    w = _norm_stack(weights)
    return {
        "weights": {k: round(v, 3) for k, v in w.items()},
        "ml_brier": round(float(((p - res) ** 2).mean()), 5),
        "ml_acc": round(float(((p > 0.5) == (res > 0.5)).mean()), 4),
        "margin_mae": round(float((m - df["margin"]).abs().mean()), 3),
        "total_mae": round(float((t - df["total"]).abs().mean()), 3),
    }


def choose_stack_weights(df: pd.DataFrame, grid_step: float = 0.05) -> dict:
    """Constrained three-model search over Elo, points-power and EPA/PPA.

    A challenger can win only if it improves moneyline Brier without worsening
    either margin MAE or total MAE versus the current Elo/power champion.
    """
    w_elo = load_blend_weight()
    champion = {"elo": w_elo, "power": 1.0 - w_elo, "epa": 0.0}
    base = score_stack(df, champion)
    rows = []
    vals = [round(x, 2) for x in np.arange(0.0, 1.0 + 1e-9, grid_step)]
    for we in vals:
        for wp in vals:
            if we + wp > 1.0 + 1e-9:
                continue
            wx = round(1.0 - we - wp, 2)
            s = score_stack(df, {"elo": we, "power": wp, "epa": wx})
            s["eligible"] = (
                s["margin_mae"] <= base["margin_mae"] + 1e-9
                and s["total_mae"] <= base["total_mae"] + 1e-9
            )
            rows.append(s)
    feasible = [r for r in rows if r["eligible"]] or rows
    best = min(feasible, key=lambda r: (
        r["ml_brier"], r["margin_mae"], r["total_mae"], r["weights"]["epa"]))
    return {"champion": base, "chosen": best, "grid": rows}


def challenger_ablation(since: int, quiet: bool = False) -> dict:
    """Run the CFB model-signal ablation including EPA/PPA challenger columns."""
    games = E.load_games()
    df = walk_forward(games, since, quiet=quiet, include_epa=True)
    if df.empty:
        raise SystemExit("No FBS-vs-FBS EPA challenger games in the validation window.")
    w_elo = load_blend_weight()
    candidates = [
        ("elo", {"elo": 1.0}),
        ("power", {"power": 1.0}),
        ("epa", {"epa": 1.0}),
        ("champion_elo_power", {"elo": w_elo, "power": 1.0 - w_elo}),
        ("elo_epa_50_50", {"elo": 0.5, "epa": 0.5}),
        ("power_epa_50_50", {"power": 0.5, "epa": 0.5}),
        ("equal_thirds", {"elo": 1 / 3, "power": 1 / 3, "epa": 1 / 3}),
    ]
    table = []
    for name, weights in candidates:
        row = score_stack(df, weights)
        row["candidate"] = name
        table.append(row)
    search = choose_stack_weights(df)
    chosen = dict(search["chosen"])
    chosen["candidate"] = "grid_best_constrained"
    table.append(chosen)
    promotes = (
        chosen["ml_brier"] < search["champion"]["ml_brier"]
        and chosen["margin_mae"] <= search["champion"]["margin_mae"]
        and chosen["total_mae"] <= search["champion"]["total_mae"]
        and chosen["weights"].get("epa", 0.0) > 0
    )
    return {
        "window": f"{since}-{int(df['season'].max())}",
        "n_games": int(len(df)),
        "table": table,
        "champion": search["champion"],
        "chosen": chosen,
        "promote_epa": bool(promotes),
    }


PPA_SPLIT_FIELDS = {
    "pass": "passing",
    "rush": "rushing",
    "first": "firstDown",
    "second": "secondDown",
    "third": "thirdDown",
}


def split_ppa_walk_forward(games: pd.DataFrame, since: int,
                           quiet: bool = False) -> pd.DataFrame:
    carry, offs = E.season_priors()
    _, history = E.run_elo(games, record_pregame=True, carry=carry, prior_offsets=offs)
    diffs = np.array([h[2] for h in history])
    slope = E.fit_slope(games, history, (games["season"] < since).values)
    epa_data = X.load_ppa()
    w_elo = load_blend_weight()
    ev = games[(games["season"] >= since)
               & (games["home_div"] == "fbs") & (games["away_div"] == "fbs")]
    rows, idx = [], []
    for (season, week, _stype), wk in ev.groupby(["season", "week", "season_type"],
                                                 sort=False):
        asof = wk["date"].min()
        try:
            pparams = P.fit(games, asof=asof)
            xparams = {name: X.fit(asof=asof, data=epa_data, field=field)
                       for name, field in PPA_SPLIT_FIELDS.items()}
        except ValueError:
            continue
        if not quiet:
            print(f"  fit split-PPA week {season} w{week} ({len(wk)} games)", file=sys.stderr)
        for r in wk.itertuples():
            if r.home not in pparams["teams"] or r.away not in pparams["teams"]:
                continue
            if any(r.home not in xp["teams"] or r.away not in xp["teams"]
                   for xp in xparams.values()):
                continue
            d = diffs[r.Index]
            pp = P.predict(pparams, r.home, r.away, neutral=bool(r.neutral))
            row = {
                "season": int(r.season), "week": int(r.week),
                "home_team": r.home_team, "away_team": r.away_team,
                "p_champ": w_elo * E.win_prob(d) + (1.0 - w_elo) * pp["p1"],
                "m_champ": w_elo * slope * d + (1.0 - w_elo) * pp["margin"],
                "t_champ": pp["total"],
                "margin": r.home_points - r.away_points,
                "total": r.home_points + r.away_points,
            }
            for name, xp in xparams.items():
                pred = X.predict(xp, r.home, r.away, neutral=bool(r.neutral))
                row[f"p_{name}"] = pred["p1"]
                row[f"m_{name}"] = pred["margin"]
                row[f"t_{name}"] = pred["total"]
            row["p_early"] = 0.5 * (row["p_first"] + row["p_second"])
            row["m_early"] = 0.5 * (row["m_first"] + row["m_second"])
            row["t_early"] = 0.5 * (row["t_first"] + row["t_second"])
            rows.append(row)
            idx.append(r.Index)
    return pd.DataFrame(rows, index=idx)


def _norm_named(weights: dict[str, float], names: tuple[str, ...]) -> dict[str, float]:
    vals = {k: max(0.0, float(weights.get(k, 0.0))) for k in names}
    s = sum(vals.values())
    if s <= 0:
        return {"champ": 1.0, **{k: 0.0 for k in names if k != "champ"}}
    return {k: v / s for k, v in vals.items()}


def _score_named_stack(df: pd.DataFrame, weights: dict[str, float],
                       names: tuple[str, ...]) -> dict:
    w = _norm_named(weights, names)
    p = sum(w[n] * df[f"p_{n}"] for n in names)
    m = sum(w[n] * df[f"m_{n}"] for n in names)
    total_w = sum(w[n] for n in names if f"t_{n}" in df.columns)
    t = sum(w[n] * df[f"t_{n}"] for n in names if f"t_{n}" in df.columns) / max(total_w, 1e-12)
    res = (df["margin"] > 0).astype(float)
    return {
        "weights": {k: round(v, 3) for k, v in w.items()},
        "ml_brier": round(float(((p - res) ** 2).mean()), 5),
        "ml_acc": round(float(((p > 0.5) == (res > 0.5)).mean()), 4),
        "margin_mae": round(float((m - df["margin"]).abs().mean()), 3),
        "total_mae": round(float((t - df["total"]).abs().mean()), 3),
    }


def _simplex_weights(names: tuple[str, ...], step: float = 0.1):
    units = int(round(1.0 / step))
    vals = [0] * len(names)
    def rec(i, remaining):
        if i == len(names) - 1:
            vals[i] = remaining
            yield {names[j]: vals[j] * step for j in range(len(names))}
            return
        for v in range(remaining + 1):
            vals[i] = v
            yield from rec(i + 1, remaining - v)
    yield from rec(0, units)


def ppa_split_ablation(since: int, quiet: bool = False) -> dict:
    games = E.load_games()
    df = split_ppa_walk_forward(games, since, quiet=quiet)
    if df.empty:
        raise SystemExit("No FBS-vs-FBS split-PPA challenger games.")
    names = ("champ", "pass", "rush", "early", "third")
    champion = _score_named_stack(df, {"champ": 1.0}, names)
    candidates = [
        ("champion", {"champ": 1.0}),
        ("pass_rush_10", {"champ": 0.8, "pass": 0.1, "rush": 0.1}),
        ("down_split_10", {"champ": 0.8, "early": 0.1, "third": 0.1}),
        ("all_splits_20", {"champ": 0.8, "pass": 0.05, "rush": 0.05,
                           "early": 0.05, "third": 0.05}),
    ]
    table = []
    for label, weights in candidates:
        row = _score_named_stack(df, weights, names)
        row["candidate"] = label
        table.append(row)
    grid = []
    for weights in _simplex_weights(names, step=0.1):
        row = _score_named_stack(df, weights, names)
        row["eligible"] = (row["margin_mae"] <= champion["margin_mae"] + 1e-9
                           and row["total_mae"] <= champion["total_mae"] + 1e-9)
        grid.append(row)
    feasible = [r for r in grid if r["eligible"]] or grid
    chosen = min(feasible, key=lambda r: (r["ml_brier"], r["margin_mae"], r["total_mae"]))
    chosen = dict(chosen)
    chosen["candidate"] = "grid_best_constrained"
    table.append(chosen)
    promotes = (
        chosen["ml_brier"] <= champion["ml_brier"] - 0.0002
        and chosen["margin_mae"] <= champion["margin_mae"]
        and chosen["total_mae"] <= champion["total_mae"]
        and chosen["weights"].get("champ", 0.0) < 1.0
    )
    return {"window": f"{since}-{int(df['season'].max())}",
            "n_games": int(len(df)), "table": table,
            "champion": champion, "chosen": chosen,
            "promote_ppa_splits": bool(promotes)}


def evaluate(since: int, quiet: bool = False) -> dict:
    games = E.load_games()
    df = walk_forward(games, since, quiet=quiet)
    if df.empty:
        raise SystemExit("No FBS-vs-FBS games in the validation window.")
    res = (df["margin"] > 0).astype(float)
    ats_roi, ats_n = _roi_by_threshold(df, SPREADS_CSV, "ats")
    tot_roi, tot_n = _roi_by_threshold(df, TOTALS_CSV, "total")
    return {
        "window": f"{since}-{int(df['season'].max())}",
        "n_games": int(len(df)),
        "ml_brier": round(float(((df["p_blend"] - res) ** 2).mean()), 4),
        "ml_acc": round(float(((df["p_blend"] > 0.5) == (res > 0.5)).mean()), 4),
        "margin_mae": round(float((df["m_blend"] - df["margin"]).abs().mean()), 3),
        "total_mae": round(float((df["t_pow"] - df["total"]).abs().mean()), 3),
        "ats_roi": ats_roi, "ats_n": ats_n,
        "totals_roi": tot_roi, "totals_n": tot_n,
        "data_fingerprint": DF.compact_snapshot(),
    }


def choose_weight(df: pd.DataFrame, grid=None) -> dict:
    """Pick the elo blend weight that minimises moneyline Brier *without* letting
    margin MAE regress past the current 0.5-blend margin MAE (conservative: a
    weight change must not trade accuracy on the market CFB actually bets, ATS).

    Pure function of the walk-forward frame — unit-testable and re-runnable for
    any weight without repeating the walk. Returns the table + chosen weight."""
    if grid is None:
        grid = [round(x, 2) for x in np.arange(0.0, 1.001, 0.05)]
    res = (df["margin"] > 0).astype(float).values
    pe, pp = df["p_elo"].values, df["p_pow"].values
    me, mp = df["m_elo"].values, df["m_pow"].values
    y = df["margin"].values
    base_mae = float(np.abs(0.5 * (me + mp) - y).mean())   # current 50/50 margin MAE
    table = []
    for w in grid:
        p = np.clip(w * pe + (1.0 - w) * pp, 1e-6, 1 - 1e-6)
        brier = float(((p - res) ** 2).mean())
        mae = float(np.abs(w * me + (1.0 - w) * mp - y).mean())
        table.append({"w_elo": w, "ml_brier": round(brier, 5),
                      "margin_mae": round(mae, 3),
                      "ok": mae <= base_mae + 1e-9})
    feasible = [r for r in table if r["ok"]] or table
    best = min(feasible, key=lambda r: r["ml_brier"])
    base = next(r for r in table if abs(r["w_elo"] - 0.5) < 1e-9)
    return {"table": table, "chosen": best["w_elo"],
            "baseline_w": 0.5, "baseline_brier": base["ml_brier"],
            "baseline_margin_mae": base["margin_mae"],
            "chosen_brier": best["ml_brier"],
            "chosen_margin_mae": best["margin_mae"]}


def _metrics_at_weight(df: pd.DataFrame, weight: float) -> dict:
    p = weight * df["p_elo"] + (1.0 - weight) * df["p_pow"]
    margin = weight * df["m_elo"] + (1.0 - weight) * df["m_pow"]
    result = (df["margin"] > 0).astype(float)
    return {
        "n_games": int(len(df)),
        "ml_brier": round(float(((p - result) ** 2).mean()), 5),
        "ml_acc": round(float(((p > 0.5) == (result > 0.5)).mean()), 4),
        "margin_mae": round(float((margin - df["margin"]).abs().mean()), 3),
        "total_mae": round(float((df["t_pow"] - df["total"]).abs().mean()), 3),
    }


def _week_block_ci(df: pd.DataFrame, weight: float, *, draws: int = 2000,
                   seed: int = 20260801) -> dict:
    """Deterministic week-block bootstrap intervals for held-out metrics."""
    weeks = [group.index.to_numpy() for _, group in df.groupby("week", sort=True)]
    if not weeks:
        return {}
    rng = np.random.default_rng(seed)
    values = {"ml_brier": [], "margin_mae": [], "total_mae": []}
    for _ in range(draws):
        sampled = rng.integers(0, len(weeks), len(weeks))
        idx = np.concatenate([weeks[i] for i in sampled])
        score = _metrics_at_weight(df.loc[idx], weight)
        for key in values:
            values[key].append(score[key])
    return {key: [round(float(np.quantile(samples, 0.025)), 5),
                  round(float(np.quantile(samples, 0.975)), 5)]
            for key, samples in values.items()}


def nested_holdout_from_frame(df: pd.DataFrame, selection_until: int,
                              holdout_season: int) -> dict:
    """Choose blend weight on earlier seasons and score one untouched season."""
    selection = df[df["season"] <= int(selection_until)]
    holdout = df[df["season"] == int(holdout_season)]
    if selection.empty or holdout.empty:
        raise ValueError("nested CFB split has an empty selection or holdout frame")
    chosen = choose_weight(selection)
    locked = float(chosen["chosen"])
    return {
        "selection_window": f"{int(selection['season'].min())}-{selection_until}",
        "selection_games": int(len(selection)),
        "holdout_season": int(holdout_season),
        "holdout_games": int(len(holdout)),
        "locked_w_elo": locked,
        "selection": _metrics_at_weight(selection, locked),
        "holdout": _metrics_at_weight(holdout, locked),
        "holdout_ci_95": _week_block_ci(holdout, locked),
        "season_metrics": {
            str(int(season)): _metrics_at_weight(group, locked)
            for season, group in df.groupby("season", sort=True)
        },
        "selection_search": chosen,
    }


def _market_holdout(df: pd.DataFrame, weight: float, lines_csv: str,
                    kind: str, threshold: float = 3.0) -> dict:
    lines = pd.read_csv(lines_csv)
    merged = df.merge(lines, on=["season", "week", "home_team", "away_team"],
                      how="inner")
    if kind == "ats":
        merged["edge_pts"] = (weight * merged["m_elo"]
                              + (1.0 - weight) * merged["m_pow"]
                              + merged["home_line"])
        settle = ats_settle
    else:
        merged["edge_pts"] = merged["t_pow"] - merged["total_line"]
        settle = totals_settle
    bets = merged[merged["edge_pts"].abs() >= float(threshold)].copy()
    if bets.empty:
        return {"threshold": threshold, "n": 0}
    won, lost, push, pnl = settle(bets)
    bets["pnl"] = pnl
    weekly = [group["pnl"].to_numpy() for _, group in bets.groupby("week", sort=True)]
    rng = np.random.default_rng(20260801)
    rois = []
    for _ in range(2000):
        sampled = rng.integers(0, len(weekly), len(weekly))
        rois.append(float(np.concatenate([weekly[i] for i in sampled]).mean()))
    return {
        "benchmark": "closing consensus",
        "threshold": threshold,
        "n": int(len(bets)), "won": won, "lost": lost, "push": push,
        "roi": round(float(np.mean(pnl)), 4),
        "roi_ci_95": [round(float(np.quantile(rois, 0.025)), 4),
                      round(float(np.quantile(rois, 0.975)), 4)],
    }


def nested_holdout(selection_since: int = 2023, selection_until: int = 2024,
                   holdout_season: int = 2025, quiet: bool = True) -> dict:
    games = E.load_games()
    frame = walk_forward(games, selection_since, quiet=quiet, w_elo=0.5)
    frame = frame[frame["season"] <= holdout_season]
    report = nested_holdout_from_frame(frame, selection_until, holdout_season)
    holdout = frame[frame["season"] == holdout_season]
    locked = report["locked_w_elo"]
    report["markets"] = {
        "ats": _market_holdout(holdout, locked, SPREADS_CSV, "ats"),
        "total": _market_holdout(holdout, locked, TOTALS_CSV, "total"),
    }
    report["data_fingerprint"] = DF.compact_snapshot()
    report["method"] = (
        "blend weight selected only on selection seasons; 2025 held untouched "
        "until final scoring; 95% intervals use deterministic week-block bootstrap"
    )
    return report


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.",
                               dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def freeze_nested_holdout(report: dict) -> None:
    _atomic_json(NESTED_ARTIFACT, report)
    _atomic_json(_BLEND_WEIGHT_FILE, {
        "w_elo": report["locked_w_elo"],
        "method": "nested_season_holdout",
        "selection_window": report["selection_window"],
        "selection_games": report["selection_games"],
        "holdout_season": report["holdout_season"],
        "holdout_games": report["holdout_games"],
        "holdout_ml_brier": report["holdout"]["ml_brier"],
        "frozen_at": str(date.today()),
        "data_line_sha256": report["data_fingerprint"]["line_sha256"],
    })


def _print_nested(report: dict) -> None:
    print(f"CFB nested validation · selection {report['selection_window']} "
          f"({report['selection_games']} games) · holdout {report['holdout_season']} "
          f"({report['holdout_games']} games)")
    print(f"  locked w_elo {report['locked_w_elo']:.2f}")
    for label in ("selection", "holdout"):
        score = report[label]
        print(f"  {label:<9} Brier {score['ml_brier']:.5f} · margin MAE "
              f"{score['margin_mae']:.3f} · total MAE {score['total_mae']:.3f}")
    for market, result in report["markets"].items():
        print(f"  {market:<9} closing benchmark ≥{result['threshold']:.1f}: "
              f"{result['won']}-{result['lost']}-{result['push']} · "
              f"ROI {result['roi']:+.1%} · 95% CI "
              f"[{result['roi_ci_95'][0]:+.1%}, {result['roi_ci_95'][1]:+.1%}]")


def tune_blend(since: int, write: bool = False, quiet: bool = True) -> dict:
    games = E.load_games()
    df = walk_forward(games, since, quiet=quiet, w_elo=0.5)  # raw cols are weight-free
    if df.empty:
        raise SystemExit("No FBS-vs-FBS games in the validation window.")
    out = choose_weight(df)
    print(f"CFB blend-weight tuning · {since}-{int(df['season'].max())} · "
          f"{len(df)} games  (w_elo = weight on Elo)")
    print(f"\n{'w_elo':>6}{'ml_brier':>11}{'margin_mae':>12}  feasible")
    for r in out["table"]:
        star = "  <-- chosen" if abs(r["w_elo"] - out["chosen"]) < 1e-9 else ""
        print(f"{r['w_elo']:>6.2f}{r['ml_brier']:>11.5f}{r['margin_mae']:>12.3f}"
              f"  {'y' if r['ok'] else 'n'}{star}")
    db = out["chosen_brier"] - out["baseline_brier"]
    print(f"\n  default w=0.50 → Brier {out['baseline_brier']:.5f}, "
          f"margin MAE {out['baseline_margin_mae']:.3f}")
    print(f"  chosen  w={out['chosen']:.2f} → Brier {out['chosen_brier']:.5f} "
          f"({db:+.5f}), margin MAE {out['chosen_margin_mae']:.3f}")
    if write:
        os.makedirs(os.path.dirname(_BLEND_WEIGHT_FILE), exist_ok=True)
        json.dump({"w_elo": out["chosen"], "since": since,
                   "baseline_brier": out["baseline_brier"],
                   "chosen_brier": out["chosen_brier"]},
                  open(_BLEND_WEIGHT_FILE, "w"), indent=2)
        print(f"\n[blend] wrote {_BLEND_WEIGHT_FILE} (w_elo={out['chosen']:.2f}). "
              "Re-run `validate.py --gate --update-baseline` to rebaseline.")
    else:
        print(f"\n  (dry run — add --write to opt into w_elo={out['chosen']:.2f}; "
              "default stays 0.50)")
    return out


def _load_baseline() -> dict | None:
    if os.path.exists(BASELINE):
        try:
            return json.loads(open(BASELINE).read())
        except Exception:
            return None
    return None


def _save_baseline(metrics: dict) -> None:
    os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".validation_baseline.",
                               dir=os.path.dirname(BASELINE))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")
        DF.write_manifest()
        os.replace(tmp, BASELINE)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _print_metrics(m: dict) -> None:
    print(f"CFB validation · {m['window']} · {m['n_games']} games")
    print(f"  ml_brier   {m['ml_brier']:.4f}   (acc {m['ml_acc']:.1%})")
    print(f"  margin_mae {m['margin_mae']:.2f}")
    print(f"  total_mae  {m['total_mae']:.2f}")
    if m["ats_roi"]:
        print("  ATS ROI:    " + "  ".join(f"≥{k}:{v:+.1%}" for k, v in m["ats_roi"].items()))
    if m["totals_roi"]:
        print("  Totals ROI: " + "  ".join(f"≥{k}:{v:+.1%}" for k, v in m["totals_roi"].items()))


def _print_ablation(a: dict) -> None:
    print(f"CFB EPA challenger ablation · {a['window']} · {a['n_games']} games")
    print(f"{'candidate':<22s}{'weights e/p/x':>17s}{'acc':>8s}{'Brier':>9s}"
          f"{'m MAE':>9s}{'t MAE':>9s}  gate")
    for r in a["table"]:
        w = r["weights"]
        gate_status = ""
        if r["candidate"] == "grid_best_constrained":
            gate_status = "PROMOTE" if a["promote_epa"] else "reject"
        print(f"{r['candidate']:<22s}"
              f"{w.get('elo', 0):>5.2f}/{w.get('power', 0):>4.2f}/{w.get('epa', 0):>4.2f}"
              f"{r['ml_acc']:>8.1%}{r['ml_brier']:>9.5f}"
              f"{r['margin_mae']:>9.2f}{r['total_mae']:>9.2f}  {gate_status}")
    champ = a["champion"]
    chosen = a["chosen"]
    print(f"\n  champion Brier {champ['ml_brier']:.5f}, margin MAE {champ['margin_mae']:.2f}, "
          f"total MAE {champ['total_mae']:.2f}")
    print(f"  chosen   Brier {chosen['ml_brier']:.5f}, margin MAE {chosen['margin_mae']:.2f}, "
          f"total MAE {chosen['total_mae']:.2f}")
    if not a["promote_epa"]:
        print("  verdict: EPA/PPA remains an explicit challenger; default CFB model is unchanged.")


def _print_ppa_split_ablation(a: dict) -> None:
    print(f"CFB split-PPA challenger ablation · {a['window']} · {a['n_games']} games")
    print(f"{'candidate':<22s}{'champ/pass/rush/early/3d':>27s}{'acc':>8s}"
          f"{'Brier':>9s}{'m MAE':>9s}{'t MAE':>9s}  gate")
    for r in a["table"]:
        w = r["weights"]
        gate_status = ""
        if r["candidate"] == "grid_best_constrained":
            gate_status = "PROMOTE" if a["promote_ppa_splits"] else "reject"
        print(f"{r['candidate']:<22s}"
              f"{w.get('champ', 0):>5.2f}/{w.get('pass', 0):>4.2f}/"
              f"{w.get('rush', 0):>4.2f}/{w.get('early', 0):>5.2f}/"
              f"{w.get('third', 0):>4.2f}"
              f"{r['ml_acc']:>8.1%}{r['ml_brier']:>9.5f}"
              f"{r['margin_mae']:>9.2f}{r['total_mae']:>9.2f}  {gate_status}")
    champ = a["champion"]
    chosen = a["chosen"]
    print(f"\n  champion Brier {champ['ml_brier']:.5f}, margin MAE {champ['margin_mae']:.2f}, "
          f"total MAE {champ['total_mae']:.2f}")
    print(f"  chosen   Brier {chosen['ml_brier']:.5f}, margin MAE {chosen['margin_mae']:.2f}, "
          f"total MAE {chosen['total_mae']:.2f}")
    if not a["promote_ppa_splits"]:
        print("  verdict: split PPA remains a rejected challenger; default CFB model is unchanged.")


def gate(metrics: dict) -> int:
    """Compare to baseline. Returns process exit code (0 pass, 1 fail)."""
    base = _load_baseline()
    if base is None:
        print("[gate] FAIL no reviewed baseline found. Run --update-baseline explicitly.")
        return 1
    current_fp = (metrics.get("data_fingerprint") or {}).get("line_sha256")
    baseline_fp = (base.get("data_fingerprint") or {}).get("line_sha256")
    fingerprint_ok = bool(current_fp and baseline_fp and current_fp == baseline_fp)
    if not fingerprint_ok:
        print("\n[gate] FAIL validation dataset fingerprint differs from the reviewed "
              "baseline (or the baseline predates fingerprinting).")
        print(f"  current:  {current_fp or 'missing'}")
        print(f"  baseline: {baseline_fp or 'missing'}")
        print("  Review the dataset manifest, then run --update-baseline explicitly.")
    checks = [
        ("ml_brier", metrics["ml_brier"], base.get("ml_brier"), BRIER_TOL, "higher"),
        ("margin_mae", metrics["margin_mae"], base.get("margin_mae"), MAE_TOL, "higher"),
        ("total_mae", metrics["total_mae"], base.get("total_mae"), MAE_TOL, "higher"),
    ]
    print(f"\n{'metric':<12s}{'current':>10s}{'baseline':>10s}{'limit':>10s}  status")
    failed = not fingerprint_ok
    for name, cur, b, tol, _dir in checks:
        if b is None:
            print(f"{name:<12s}{cur:>10.4f}{'—':>10s}{'—':>10s}  (no baseline)")
            continue
        limit = b + tol
        ok = cur <= limit
        failed = failed or not ok
        print(f"{name:<12s}{cur:>10.4f}{b:>10.4f}{limit:>10.4f}  {'PASS' if ok else 'FAIL'}")
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2023,
                    help="first validation season (Elo slope fit on seasons before this)")
    ap.add_argument("--gate", action="store_true",
                    help="exit non-zero if Brier/MAE regressed past tolerance")
    ap.add_argument("--quiet", action="store_true", help="suppress per-week progress")
    ap.add_argument("--update-baseline", action="store_true",
                    help="overwrite the stored baseline with this run")
    ap.add_argument("--tune-blend", action="store_true",
                    help="show elo/power blend-weight before/after table (M6)")
    ap.add_argument("--ablation", action="store_true",
                    help="run model-signal ablation incl. EPA/PPA challenger")
    ap.add_argument("--ppa-splits", action="store_true",
                    help="with --ablation, test pass/rush/down-split PPA challengers")
    ap.add_argument("--write", action="store_true",
                    help="with --tune-blend, opt into the chosen weight")
    ap.add_argument("--nested-holdout", action="store_true",
                    help="select on 2023-24 and report untouched 2025 holdout")
    ap.add_argument("--selection-since", type=int, default=2023)
    ap.add_argument("--selection-until", type=int, default=2024)
    ap.add_argument("--holdout-season", type=int, default=2025)
    args = ap.parse_args()

    if args.tune_blend:
        tune_blend(args.since, write=args.write, quiet=True)
        return
    if args.nested_holdout:
        report = nested_holdout(args.selection_since, args.selection_until,
                                args.holdout_season, quiet=args.quiet)
        _print_nested(report)
        if args.write:
            freeze_nested_holdout(report)
            print(f"\n[nested] froze artifact {NESTED_ARTIFACT} and runtime weight "
                  f"w_elo={report['locked_w_elo']:.2f}")
        return
    if args.ablation:
        if args.ppa_splits:
            _print_ppa_split_ablation(ppa_split_ablation(args.since, quiet=args.quiet))
            return
        _print_ablation(challenger_ablation(args.since, quiet=args.quiet))
        return

    metrics = evaluate(args.since, quiet=args.quiet)
    _print_metrics(metrics)

    if args.update_baseline:
        _save_baseline(metrics)
        print("\n[baseline] updated.")
        return
    if args.gate:
        sys.exit(gate(metrics))


if __name__ == "__main__":
    main()
