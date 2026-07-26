"""
golf/validate.py  –  Walk-forward backtest + regression gate (the yardstick).

For each completed tournament (after a minimum history), refit the model on
rounds STRICTLY before that event, simulate the field, and score the predicted
win / top-5 / top-10 / top-20 / make-cut probabilities against what actually
happened. No look-ahead. Mirrors club_soccer/validate.py + root validate.py.

Metrics per market: Brier, log-loss, and a reliability table; plus a skill
score vs the base-rate baseline (1 − Brier/Brier_base). Win is scored both as
per-player favorite calibration and as event-level surprise −log p(winner).

Outputs:
  data/validation_predictions.csv   (feeds calibrate.py)
  data/validation_baseline.json     (Brier baseline for --gate)

Usage:
  python -m golf.validate [--since 2023-06-01] [--sims 20000] [--gate]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import model
from . import simulate as gsim

DATA_DIR = Path(__file__).parent / "data"
PRED_CSV = DATA_DIR / "validation_predictions.csv"
BASELINE_JSON = DATA_DIR / "validation_baseline.json"
WEATHER_CONFIG_JSON = DATA_DIR / "weather_config.json"
SHAPE_TUNING_JSON = DATA_DIR / "shape_tuning.json"

MARKETS = ["win", "top5", "top10", "top20", "cut"]
TOPN = {"top5": 5, "top10": 10, "top20": 20}
GATE_TOL = 0.004          # allowed Brier regression on the headline metric
MIN_TRAIN_ROUNDS = 4000   # don't evaluate until the model has enough history
EPS = 1e-12
VALIDATION_SCHEMA_VERSION = 3


def _write_json(path: Path, payload: dict) -> None:
    from .io_utils import atomic_write_text
    atomic_write_text(path, json.dumps(payload, indent=1))


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p: np.ndarray, y: np.ndarray, bins=10) -> list[tuple]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    idx = np.digitize(p, edges[1:-1])
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append((round(float(p[m].mean()), 3), round(float(y[m].mean()), 3), int(m.sum())))
    return out


def _actuals(event: pd.DataFrame) -> dict[str, dict]:
    """Per-player actual outcomes for one tournament.

    Placement dead heats come from final totals. Winner is the official ESPN
    finish=1 result so a playoff loser is not mislabeled as half a winner.
    """
    g = event.groupby("player")
    total = g["score_to_par"].sum()
    made = g["made_cut"].max()
    nrounds = g["round"].count()
    # rank only players who completed the tournament; missed-cut → no top-N
    # Only complete finishers are eligible for placement labels. A WD/DQ with
    # three rounds must not be ranked against four-round totals.
    max_rounds = int(nrounds.max()) if len(nrounds) else 4
    finishers = total[(made == 1) & (nrounds == max_rounds)]
    rank = finishers.rank(method="min")
    score_groups = finishers.groupby(finishers).groups
    official_finish = g["finish"].min()

    def credit(player: str, top_n: int) -> float:
        if player not in rank.index:
            return 0.0
        tied = score_groups[finishers.loc[player]]
        start = int(rank.loc[player])
        slots = max(0, min(len(tied), top_n - start + 1))
        return slots / len(tied)
    out = {}
    for player in total.index:
        mc = int(made.loc[player])
        r = int(rank.loc[player]) if (mc == 1 and player in rank.index) else 999
        out[player] = {
            "made_cut": mc,
            "win": float(official_finish.loc[player] == 1),
            "top5": credit(player, 5),
            "top10": credit(player, 10),
            "top20": credit(player, 20),
            "finish": r,
        }
    return out


# ─────────────────────────────────────────────
# Walk-forward loop
# ─────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, since: str, sims: int,
                 seed: int = 0, verbose: bool = True,
                 config: dict | None = None,
                 sim_config: dict | None = None,
                 feature_flags: dict | None = None) -> pd.DataFrame:
    """Walk-forward predictions. `config` tunes the model fit; `sim_config`
    (optional) overrides the one joint scoring shape passed to the simulator,
    e.g. ``{"round_corr": .3, "tail_df": None, "blowup_mix": .2}``."""
    sc = sim_config or {}
    # Current feature files have no historical snapshots. Honest walk-forward
    # evaluation disables them unless the caller supplies point-in-time data.
    safe_flags = {
        "public_stat": False,
        "global_priors": False,
        "weather": False,
        "exact_course": False,
    }
    safe_flags.update(feature_flags or {})
    event_columns = [
        "tournament_id", "date", "course", "is_major", "event_name", "tour"
    ]
    event_columns += [
        c for c in (
            "cut_rule", "total_rounds", "no_cut", "course_par", "course_yards"
        )
        if c in df.columns
    ]
    events = (df[event_columns]
              .drop_duplicates("tournament_id")
              .sort_values("date"))
    for column in ("par3_holes", "par4_holes", "par5_holes"):
        if column in df.columns:
            events[column] = events["tournament_id"].map(
                df.groupby("tournament_id")[column].max()
            )
    rng = np.random.default_rng(seed)
    rows = []
    since_ts = pd.Timestamp(since)

    for ev in events.itertuples():
        start = pd.Timestamp(ev.date)
        if start < since_ts:
            continue
        prior = df[df["date"] < start]
        if len(prior) < MIN_TRAIN_ROUNDS:
            continue
        event_rounds = df[df["tournament_id"] == ev.tournament_id]
        total_rounds = int(getattr(ev, "total_rounds", 4) or 4)
        if total_rounds != 4 or int(event_rounds["round"].max()) != total_rounds:
            continue  # the production simulator is a 72-hole model
        field_rows = (
            event_rounds[["player", "dg_id"]]
            .drop_duplicates("player")
            .sort_values("player")
        )
        field = [
            model.Player(name=str(row.player), dg_id=str(row.dg_id))
            for row in field_rows.itertuples()
        ]
        if len(field) < 30:
            continue
        try:
            params = model.fit(df, asof=start, config=config,
                               include_public_stats=False)
        except ValueError:
            continue
        rated = model.predict_field(field, params, course=str(ev.course),
                                    course_par=int(getattr(ev, "course_par", 0) or 0),
                                    course_yards=int(getattr(ev, "course_yards", 0) or 0),
                                    par3_holes=int(getattr(ev, "par3_holes", 0) or 0),
                                    par4_holes=int(getattr(ev, "par4_holes", 0) or 0),
                                    par5_holes=int(getattr(ev, "par5_holes", 0) or 0),
                                    is_major=bool(ev.is_major),
                                    feature_flags=safe_flags)
        event_no_cut = bool(int(getattr(ev, "no_cut", 0) or 0))
        cut_rule = int(getattr(ev, "cut_rule", 65) or 65)
        res = gsim.simulate_tournament(
            rated, n_sims=sims, cut_rule=cut_rule, no_cut=event_no_cut, rng=rng,
            round_corr=sc.get("round_corr"),
            tail_df=sc.get("tail_df", gsim._USE_CONFIG),
            blowup_mix=sc.get("blowup_mix"))
        actual = _actuals(event_rounds)
        for p in rated:
            a = actual.get(p.name)
            if a is None:
                continue
            r = res[p.name]
            rows.append({
                "tournament_id": ev.tournament_id, "date": str(start.date()),
                "point_in_time_safe": 1,
                "validation_schema_version": VALIDATION_SCHEMA_VERSION,
                "cut_rule": cut_rule, "no_cut": int(event_no_cut),
                "is_major": int(bool(ev.is_major)), "player": p.name,
                "p_win": r["win"], "p_top5": r["top5"], "p_top10": r["top10"],
                "p_top20": r["top20"], "p_cut": r["made_cut"],
                "y_win": a["win"], "y_top5": a["top5"], "y_top10": a["top10"],
                "y_top20": a["top20"], "y_cut": a["made_cut"],
            })
        if verbose:
            print(f"  {str(start.date())}  {ev.tournament_id}  "
                  f"{len(field):>3} players  (train={len(prior):,})")
    return pd.DataFrame(rows)


def _prepare_walk_forward_cases(
    df: pd.DataFrame,
    since: str,
    config: dict | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Fit each historical event once for simulation/profile tuning.

    The returned cases contain only point-in-time model parameters and event
    facts. Candidate scoring shapes can therefore reuse the expensive fits
    without changing the information available to any prediction.
    """
    event_columns = [
        "tournament_id", "date", "course", "is_major", "event_name", "tour"
    ]
    event_columns += [
        c for c in (
            "cut_rule", "total_rounds", "no_cut", "course_par", "course_yards"
        )
        if c in df.columns
    ]
    events = (
        df[event_columns]
        .drop_duplicates("tournament_id")
        .sort_values("date")
    )
    for column in ("par3_holes", "par4_holes", "par5_holes"):
        if column in df.columns:
            events[column] = events["tournament_id"].map(
                df.groupby("tournament_id")[column].max()
            )
    since_ts = pd.Timestamp(since)
    cases: list[dict] = []
    for ev in events.itertuples():
        start = pd.Timestamp(ev.date)
        if start < since_ts:
            continue
        prior_count = int((df["date"] < start).sum())
        if prior_count < MIN_TRAIN_ROUNDS:
            continue
        event_rounds = df[df["tournament_id"] == ev.tournament_id]
        total_rounds = int(getattr(ev, "total_rounds", 4) or 4)
        if total_rounds != 4 or int(event_rounds["round"].max()) != total_rounds:
            continue
        field_rows = (
            event_rounds[["player", "dg_id"]]
            .drop_duplicates("player")
            .sort_values("player")
        )
        field = [
            model.Player(name=str(row.player), dg_id=str(row.dg_id))
            for row in field_rows.itertuples()
        ]
        if len(field) < 30:
            continue
        try:
            params = model.fit(
                df, asof=start, config=config, include_public_stats=False
            )
        except ValueError:
            continue
        cases.append({
            "tournament_id": ev.tournament_id,
            "date": start,
            "course": str(ev.course),
            "is_major": bool(ev.is_major),
            "cut_rule": int(getattr(ev, "cut_rule", 65) or 65),
            "no_cut": bool(int(getattr(ev, "no_cut", 0) or 0)),
            "course_par": int(getattr(ev, "course_par", 0) or 0),
            "course_yards": int(getattr(ev, "course_yards", 0) or 0),
            "par3_holes": int(getattr(ev, "par3_holes", 0) or 0),
            "par4_holes": int(getattr(ev, "par4_holes", 0) or 0),
            "par5_holes": int(getattr(ev, "par5_holes", 0) or 0),
            "field": field,
            "params": params,
            "actual": _actuals(event_rounds),
        })
        if verbose:
            print(
                f"  fit {str(start.date())}  {ev.tournament_id}  "
                f"{len(field):>3} players  (train={prior_count:,})"
            )
    return cases


def _simulate_cases(
    cases: list[dict],
    sims: int,
    seed: int,
    sim_config: dict,
    course_profile_weight: float,
) -> pd.DataFrame:
    """Simulate cached point-in-time cases under one candidate configuration."""
    rng = np.random.default_rng(seed)
    rows = []
    safe_flags = {
        "public_stat": False,
        "global_priors": False,
        "weather": False,
        "exact_course": False,
        "course_profile": course_profile_weight != 0.0,
    }
    for case in cases:
        params = dict(case["params"])
        params["course_profile_weight"] = float(course_profile_weight)
        rated = model.predict_field(
            case["field"],
            params,
            course=case["course"],
            course_par=case["course_par"],
            course_yards=case["course_yards"],
            par3_holes=case["par3_holes"],
            par4_holes=case["par4_holes"],
            par5_holes=case["par5_holes"],
            is_major=case["is_major"],
            feature_flags=safe_flags,
        )
        res = gsim.simulate_tournament(
            rated,
            n_sims=sims,
            cut_rule=case["cut_rule"],
            no_cut=case["no_cut"],
            rng=rng,
            round_corr=float(sim_config["round_corr"]),
            tail_df=sim_config.get("tail_df"),
            blowup_mix=float(sim_config["blowup_mix"]),
        )
        for player in rated:
            actual = case["actual"].get(player.name)
            if actual is None:
                continue
            result = res[player.name]
            rows.append({
                "tournament_id": case["tournament_id"],
                "date": str(case["date"].date()),
                "point_in_time_safe": 1,
                "validation_schema_version": VALIDATION_SCHEMA_VERSION,
                "cut_rule": case["cut_rule"],
                "no_cut": int(case["no_cut"]),
                "is_major": int(case["is_major"]),
                "player": player.name,
                "p_win": result["win"],
                "p_top5": result["top5"],
                "p_top10": result["top10"],
                "p_top20": result["top20"],
                "p_cut": result["made_cut"],
                "y_win": actual["win"],
                "y_top5": actual["top5"],
                "y_top10": actual["top10"],
                "y_top20": actual["top20"],
                "y_cut": actual["made_cut"],
            })
    return pd.DataFrame(rows)


def summarize(pred: pd.DataFrame) -> dict:
    report = {}
    field_size = pred.groupby("tournament_id")["player"].transform("count").values.astype(float)
    if "no_cut" in pred:
        no_cut_event = pred["no_cut"].astype(bool).values
    else:
        no_cut_event = pred.groupby("tournament_id")["y_cut"].transform("min").values == 1
    if "cut_rule" in pred:
        cut_rule = pred["cut_rule"].values.astype(float)
    else:
        cut_rule = np.full(len(pred), 65.0)
    naive_by_market = {
        "win": 1.0 / field_size,
        "top5": np.minimum(1.0, 5.0 / field_size),
        "top10": np.minimum(1.0, 10.0 / field_size),
        "top20": np.minimum(1.0, 20.0 / field_size),
        "cut": np.where(no_cut_event, 1.0, np.minimum(1.0, cut_rule / field_size)),
    }
    for mkt in MARKETS:
        col = "cut" if mkt == "cut" else mkt
        eligible = ~no_cut_event if mkt == "cut" else np.ones(len(pred), dtype=bool)
        p = pred.loc[eligible, f"p_{col}"].values
        y = pred.loc[eligible, f"y_{col}"].values.astype(float)
        naive = naive_by_market[mkt][eligible]
        base = float(y.mean())
        b = brier(p, y)
        b_base = brier(naive, y)
        report[mkt] = {
            "n": int(len(y)), "base_rate": round(base, 4),
            "brier": round(b, 5), "brier_base": round(b_base, 5),
            "skill": round(1 - b / b_base, 4) if b_base > 0 else 0.0,
            "logloss": round(logloss(p, y), 5),
            "reliability": reliability(p, y),
        }
    # event-level win surprise: −log p(actual winner)
    surprises, base_surprises = [], []
    for _tid, g in pred.groupby("tournament_id"):
        winners = g[g["y_win"] == 1]
        if winners.empty:
            continue
        pw = float(np.clip(winners["p_win"].mean(), EPS, 1))
        surprises.append(-np.log(pw))
        base_surprises.append(-np.log(1.0 / len(g)))
    report["win_event"] = {
        "events": len(surprises),
        "mean_surprise": round(float(np.mean(surprises)), 4) if surprises else None,
        "uniform_surprise": round(float(np.mean(base_surprises)), 4) if base_surprises else None,
    }
    # headline gate metric: mean skill across the lower-variance markets
    report["headline_brier"] = round(
        float(np.mean([report[m]["brier"] for m in ("top10", "top20", "cut")])), 5)
    return report


def _candidate_configs(base: dict) -> list[dict]:
    grid = {
        "form_weight": [0.0, 0.4, 0.7, 1.0],
        "form_halflife_days": [14, 21, 35],
        "skill_halflife_days": [270, 365, 540],
        "sigma_shrink_rounds": [15, 25, 40],
    }
    out = [dict(base)]
    for key, vals in grid.items():
        for val in vals:
            cfg = dict(base)
            cfg[key] = float(val)
            out.append(cfg)
    seen, uniq = set(), []
    for cfg in out:
        sig = tuple((k, float(cfg[k])) for k in sorted(model.DEFAULT_MODEL_CONFIG))
        if sig not in seen:
            seen.add(sig)
            uniq.append(cfg)
    return uniq


def _rep_for_dates(pred: pd.DataFrame, before: str | None = None,
                   after: str | None = None) -> dict:
    sub = pred
    if before is not None:
        sub = sub[pd.to_datetime(sub["date"]) < pd.Timestamp(before)]
    if after is not None:
        sub = sub[pd.to_datetime(sub["date"]) >= pd.Timestamp(after)]
    if sub.empty:
        return {}
    return summarize(sub)


def _config_label(cfg: dict) -> str:
    base = model.DEFAULT_MODEL_CONFIG
    diffs = [f"{k}={cfg[k]:g}" for k in sorted(cfg) if float(cfg[k]) != float(base[k])]
    return ", ".join(diffs) if diffs else "current"


def tune_config(since: str, sims: int, seed: int = 0, write: bool = False,
                split: str = "2025-01-01", holdout: str = "2026-01-01") -> dict:
    df = model.load_rounds_df()
    base_cfg = model.load_model_config()
    search_sims = max(300, min(750, sims // 8))
    candidates = _candidate_configs(base_cfg)
    screened = []
    print(f"Golf config tuning · {len(candidates)} candidates · "
          f"screen {search_sims} sims, confirm {sims} sims")
    for i, cfg in enumerate(candidates, 1):
        pred = walk_forward(df, since=since, sims=search_sims, seed=seed,
                            verbose=False, config=cfg)
        if pred.empty:
            continue
        train_rep = _rep_for_dates(pred, before=split) or summarize(pred)
        test_rep = _rep_for_dates(pred, after=split, before=holdout)
        holdout_rep = _rep_for_dates(pred, after=holdout)
        if not test_rep or not holdout_rep:
            raise SystemExit("Config tuning requires non-empty selection and final holdout windows.")
        row = {"config": cfg, "label": _config_label(cfg),
               "train_headline": train_rep["headline_brier"],
               "train_top10": train_rep["top10"]["brier"],
               "train_top20": train_rep["top20"]["brier"],
               "train_cut": train_rep["cut"]["brier"],
               "test_headline": test_rep["headline_brier"],
               "test_top10": test_rep["top10"]["brier"],
               "test_top20": test_rep["top20"]["brier"],
               "test_cut": test_rep["cut"]["brier"],
               "holdout_headline": holdout_rep["headline_brier"]}
        screened.append(row)
        print(f"  [{i:>2}/{len(candidates)}] {row['label']:<42s} "
              f"train {row['train_headline']:.5f} test {row['test_headline']:.5f}")
    if not screened:
        raise SystemExit("No golf config candidates produced validation predictions.")
    current = next((r for r in screened if r["label"] == "current"), screened[0])
    feasible = [r for r in screened
                if r["train_top10"] <= current["train_top10"] + 0.002
                and r["train_top20"] <= current["train_top20"] + 0.002
                and r["train_cut"] <= current["train_cut"] + 0.002]
    best_screen = min(feasible or screened, key=lambda r: r["test_headline"])

    print("\nFinal confirmation:")
    pred_cur = walk_forward(df, since=since, sims=sims, seed=seed,
                            verbose=False, config=base_cfg)
    rep_cur = summarize(pred_cur)
    val_cur = _rep_for_dates(pred_cur, after=holdout)
    pred_best = walk_forward(df, since=since, sims=sims, seed=seed,
                             verbose=False, config=best_screen["config"])
    rep_best = summarize(pred_best)
    val_best = _rep_for_dates(pred_best, after=holdout)
    if not val_cur or not val_best:
        raise SystemExit("Config promotion requires an untouched final holdout window.")
    deltas = {m: val_best[m]["brier"] - val_cur[m]["brier"]
              for m in ("top10", "top20", "cut")}
    promote = (
        val_best["headline_brier"] <= val_cur["headline_brier"] - 0.001
        and all(v <= 0.002 for v in deltas.values())
        and best_screen["label"] != "current"
    )
    print(f"  selected on {split}..{holdout}: {best_screen['label']}")
    print(f"  validation current {val_cur['headline_brier']:.5f} config {base_cfg}")
    print(f"  validation chosen  {val_best['headline_brier']:.5f} config {best_screen['config']}")
    print("  validation market deltas: "
          + " ".join(f"{k} {v:+.5f}" for k, v in deltas.items()))
    print(f"  full-window current {rep_cur['headline_brier']:.5f}")
    print(f"  full-window chosen  {rep_best['headline_brier']:.5f}")
    print(f"  verdict: {'PROMOTE' if promote else 'reject'}")
    out = {"current": rep_cur, "chosen": rep_best,
           "validation_current": val_cur, "validation_chosen": val_best,
           "chosen_config": best_screen["config"], "promote": bool(promote),
           "screened": screened}
    if write:
        if not promote:
            print("  not writing model_config.json because the promotion gate failed")
        else:
            model.save_model_config(best_screen["config"], metrics={
                "previous_validation_headline_brier": val_cur["headline_brier"],
                "chosen_validation_headline_brier": val_best["headline_brier"],
                "previous_full_headline_brier": rep_cur["headline_brier"],
                "chosen_full_headline_brier": rep_best["headline_brier"],
                "sims": sims,
                "since": since,
                "selection_split": split,
                "holdout": holdout,
            })
            _write_json(BASELINE_JSON,
                {"headline_brier": rep_best["headline_brier"], "gate_tol": GATE_TOL,
                 "asof": pred_best["date"].max(),
                 "schema_version": VALIDATION_SCHEMA_VERSION,
                 "point_in_time_safe": True})
            print(f"  wrote {model.MODEL_CONFIG_JSON}")
            print(f"  baseline updated -> {BASELINE_JSON}")
    return out


def _shape_candidates(
    current_sim: dict,
    current_course_weight: float,
) -> list[dict]:
    """Conservative joint grid for scoring shape and general course profile."""
    correlations = [0.0, 0.10, 0.20, 0.30]
    blowup_mixes = [0.0, 0.10, 0.20]
    course_weights = sorted({0.0, 0.25, float(current_course_weight)})
    candidates = [{
        "round_corr": rc,
        "tail_df": None,
        "blowup_mix": mix,
        "course_profile_weight": weight,
    } for rc in correlations for mix in blowup_mixes for weight in course_weights]
    current = {
        "round_corr": float(current_sim["round_corr"]),
        "tail_df": current_sim.get("tail_df"),
        "blowup_mix": float(current_sim["blowup_mix"]),
        "course_profile_weight": float(current_course_weight),
    }
    if current not in candidates:
        candidates.insert(0, current)
    return candidates


def _shape_label(candidate: dict) -> str:
    return (
        f"corr={candidate['round_corr']:.2f},"
        f" blowup={candidate['blowup_mix']:.2f},"
        f" course={candidate['course_profile_weight']:.2f}"
    )


def _shape_result(candidate: dict, report: dict) -> dict:
    return {
        "candidate": candidate,
        "label": _shape_label(candidate),
        "headline": report["headline_brier"],
        "top10": report["top10"]["brier"],
        "top20": report["top20"]["brier"],
        "cut": report["cut"]["brier"],
        "winner_surprise": report["win_event"]["mean_surprise"],
        "events": report["win_event"]["events"],
    }


def _shape_metrics(report: dict) -> dict:
    """Compact promotion metrics suitable for the persisted tuning audit."""
    return {
        "headline": report["headline_brier"],
        "win": report["win"]["brier"],
        "top5": report["top5"]["brier"],
        "top10": report["top10"]["brier"],
        "top20": report["top20"]["brier"],
        "cut": report["cut"]["brier"],
        "winner_surprise": report["win_event"]["mean_surprise"],
        "events": report["win_event"]["events"],
    }


def _shape_market_deltas(chosen: dict, current: dict) -> dict:
    return {
        market: chosen[market]["brier"] - current[market]["brier"]
        for market in ("top10", "top20", "cut")
    }


def tune_shape(
    since: str,
    sims: int,
    seed: int = 0,
    write: bool = False,
    split: str = "2025-07-01",
    holdout: str = "2026-01-01",
) -> dict:
    """Tune joint round shape and course-profile weight without holdout peeking.

    All candidates are screened before ``holdout``. Only the single challenger
    selected there is evaluated on the final holdout, alongside the frozen
    current configuration.
    """
    since_ts = pd.Timestamp(since)
    split_ts = pd.Timestamp(split)
    holdout_ts = pd.Timestamp(holdout)
    if not since_ts < split_ts < holdout_ts:
        raise SystemExit("Shape tuning requires since < split < holdout.")
    df = model.load_rounds_df()
    base_model_config = model.load_model_config()
    rc, tail_df, blowup_mix = gsim.load_sim_config()
    current_sim = {
        "round_corr": rc, "tail_df": tail_df, "blowup_mix": blowup_mix
    }
    current_course_weight = float(base_model_config["course_profile_weight"])
    current_candidate = {
        **current_sim,
        "course_profile_weight": current_course_weight,
    }
    candidates = _shape_candidates(current_sim, current_course_weight)
    screen_sims = max(300, min(600, sims // 10))
    confirm_sims = max(1500, min(3000, sims // 3))

    print(
        f"Golf shape tuning · {len(candidates)} candidates · "
        f"{screen_sims} screen / {confirm_sims} confirm / {sims} holdout sims"
    )
    print(
        f"  selection windows: {since}..{split} stability, "
        f"{split}..{holdout} choice; final holdout {holdout}+"
    )
    print("Preparing point-in-time event fits (one fit per event)…")
    cases = _prepare_walk_forward_cases(
        df, since=since, config=base_model_config, verbose=True
    )
    development = [case for case in cases if case["date"] < split_ts]
    selection = [
        case for case in cases if split_ts <= case["date"] < holdout_ts
    ]
    final_holdout = [case for case in cases if case["date"] >= holdout_ts]
    if not development or not selection or not final_holdout:
        raise SystemExit(
            "Shape tuning requires non-empty stability, selection, and holdout windows."
        )
    print(
        f"Prepared {len(cases)} events: {len(development)} stability, "
        f"{len(selection)} selection, {len(final_holdout)} sealed holdout."
    )

    screened = []
    current_screen = None
    print("\nPre-holdout screen:")
    pre_holdout = development + selection
    for index, candidate in enumerate(candidates, 1):
        pred = _simulate_cases(
            pre_holdout,
            sims=screen_sims,
            seed=seed,
            sim_config=candidate,
            course_profile_weight=candidate["course_profile_weight"],
        )
        dev_report = _rep_for_dates(pred, before=split)
        selection_report = _rep_for_dates(pred, after=split, before=holdout)
        row = {
            **_shape_result(candidate, selection_report),
            "stability_headline": dev_report["headline_brier"],
            "stability_top10": dev_report["top10"]["brier"],
            "stability_top20": dev_report["top20"]["brier"],
            "stability_cut": dev_report["cut"]["brier"],
        }
        screened.append(row)
        if candidate == current_candidate:
            current_screen = row
        print(
            f"  [{index:>2}/{len(candidates)}] {row['label']:<43} "
            f"stability {row['stability_headline']:.5f} "
            f"selection {row['headline']:.5f}"
        )
    if current_screen is None:
        raise RuntimeError("Current production shape is missing from candidate grid.")
    feasible = [
        row for row in screened
        if row["stability_top10"] <= current_screen["stability_top10"] + 0.0015
        and row["stability_top20"] <= current_screen["stability_top20"] + 0.0015
        and row["stability_cut"] <= current_screen["stability_cut"] + 0.0015
    ]
    shortlist = sorted(
        feasible or screened, key=lambda row: row["headline"]
    )[:4]
    if not any(row["candidate"] == current_candidate for row in shortlist):
        shortlist.append(current_screen)

    print("\nPre-holdout confirmation:")
    confirmed = []
    for row in shortlist:
        candidate = row["candidate"]
        pred = _simulate_cases(
            pre_holdout,
            sims=confirm_sims,
            seed=seed,
            sim_config=candidate,
            course_profile_weight=candidate["course_profile_weight"],
        )
        dev_report = _rep_for_dates(pred, before=split)
        selection_report = _rep_for_dates(pred, after=split, before=holdout)
        result = {
            **_shape_result(candidate, selection_report),
            "stability": dev_report,
            "selection": selection_report,
        }
        confirmed.append(result)
        print(
            f"  {result['label']:<43} stability "
            f"{dev_report['headline_brier']:.5f} selection "
            f"{selection_report['headline_brier']:.5f} winner "
            f"{selection_report['win_event']['mean_surprise']:.4f}"
        )
    current_confirm = next(
        row for row in confirmed if row["candidate"] == current_candidate
    )
    eligible = [
        row for row in confirmed
        if row["candidate"] != current_candidate
        and row["stability"]["headline_brier"]
        <= current_confirm["stability"]["headline_brier"] + 0.0005
        and all(
            row["selection"][market]["brier"]
            <= current_confirm["selection"][market]["brier"] + 0.001
            for market in ("top10", "top20", "cut")
        )
        and row["selection"]["win_event"]["mean_surprise"]
        <= current_confirm["selection"]["win_event"]["mean_surprise"] + 0.05
    ]
    challenger = min(
        eligible, key=lambda row: row["selection"]["headline_brier"]
    ) if eligible else None
    selection_pass = bool(
        challenger
        and challenger["selection"]["headline_brier"]
        <= current_confirm["selection"]["headline_brier"] - 0.00015
    )

    holdout_current = None
    holdout_chosen = None
    promote = False
    if selection_pass:
        print(
            "\nOpening final holdout for one challenger: "
            f"{challenger['label']}"
        )
        holdout_current_pred = _simulate_cases(
            final_holdout,
            sims=sims,
            seed=seed,
            sim_config=current_candidate,
            course_profile_weight=current_course_weight,
        )
        holdout_chosen_pred = _simulate_cases(
            final_holdout,
            sims=sims,
            seed=seed,
            sim_config=challenger["candidate"],
            course_profile_weight=challenger["candidate"]["course_profile_weight"],
        )
        holdout_current = summarize(holdout_current_pred)
        holdout_chosen = summarize(holdout_chosen_pred)
        market_deltas = _shape_market_deltas(holdout_chosen, holdout_current)
        winner_delta = (
            holdout_chosen["win_event"]["mean_surprise"]
            - holdout_current["win_event"]["mean_surprise"]
        )
        promote = bool(
            holdout_chosen["headline_brier"]
            <= holdout_current["headline_brier"] - 0.00005
            and all(delta <= 0.001 for delta in market_deltas.values())
            and winner_delta <= 0.05
        )
        print(
            f"  current    headline {holdout_current['headline_brier']:.5f} "
            f"winner {holdout_current['win_event']['mean_surprise']:.4f}"
        )
        print(
            f"  challenger headline {holdout_chosen['headline_brier']:.5f} "
            f"winner {holdout_chosen['win_event']['mean_surprise']:.4f}"
        )
        print(
            "  market deltas: "
            + " ".join(f"{key} {value:+.5f}" for key, value in market_deltas.items())
        )
    else:
        print("\nNo challenger cleared the pre-holdout gate; holdout remains sealed.")

    chosen_candidate = (
        challenger["candidate"] if challenger is not None else current_candidate
    )
    out = {
        "schema_version": 1,
        "since": since,
        "selection_split": split,
        "holdout": holdout,
        "screen_sims": screen_sims,
        "confirm_sims": confirm_sims,
        "holdout_sims": sims,
        "current_candidate": current_candidate,
        "chosen_candidate": chosen_candidate,
        "selection_pass": selection_pass,
        "promote": promote,
        "screened": screened,
        "confirmed": [{
            "candidate": row["candidate"],
            "label": row["label"],
            "stability": _shape_metrics(row["stability"]),
            "selection": _shape_metrics(row["selection"]),
        } for row in confirmed],
        "holdout_current": holdout_current,
        "holdout_chosen": holdout_chosen,
    }
    print(f"\nVerdict: {'PROMOTE' if promote else 'reject'}")
    if write:
        _write_json(SHAPE_TUNING_JSON, out)
        print(f"  wrote audit report {SHAPE_TUNING_JSON}")
        if promote:
            sim_payload = {
                "round_corr": chosen_candidate["round_corr"],
                "tail_df": chosen_candidate["tail_df"],
                "blowup_mix": chosen_candidate["blowup_mix"],
                "source": "golf.validate --tune-shape",
                "metrics": {
                    "selection_split": split,
                    "holdout": holdout,
                    "holdout_sims": sims,
                    "current_headline_brier": holdout_current["headline_brier"],
                    "chosen_headline_brier": holdout_chosen["headline_brier"],
                    "current_winner_surprise":
                        holdout_current["win_event"]["mean_surprise"],
                    "chosen_winner_surprise":
                        holdout_chosen["win_event"]["mean_surprise"],
                },
            }
            _write_json(gsim.SIM_CONFIG_JSON, sim_payload)
            updated_model_config = {
                **base_model_config,
                "course_profile_weight":
                    chosen_candidate["course_profile_weight"],
            }
            model.save_model_config(updated_model_config, metrics={
                "source": "golf.validate --tune-shape",
                "selection_split": split,
                "holdout": holdout,
                "holdout_sims": sims,
                "current_headline_brier": holdout_current["headline_brier"],
                "chosen_headline_brier": holdout_chosen["headline_brier"],
            })
            final_predictions = _simulate_cases(
                cases,
                sims=sims,
                seed=seed,
                sim_config=chosen_candidate,
                course_profile_weight=chosen_candidate["course_profile_weight"],
            )
            from .io_utils import atomic_write_csv
            atomic_write_csv(
                PRED_CSV,
                list(final_predictions.columns),
                final_predictions.to_dict(orient="records"),
            )
            final_report = summarize(final_predictions)
            _write_json(BASELINE_JSON, {
                "headline_brier": final_report["headline_brier"],
                "gate_tol": GATE_TOL,
                "asof": final_predictions["date"].max(),
                "schema_version": VALIDATION_SCHEMA_VERSION,
                "point_in_time_safe": True,
                "reason": "honest joint scoring-shape and course-profile retune",
            })
            out["full_promoted"] = final_report
            _write_json(SHAPE_TUNING_JSON, out)
            print(f"  wrote {gsim.SIM_CONFIG_JSON}")
            print(f"  wrote {model.MODEL_CONFIG_JSON}")
            print(f"  refreshed {PRED_CSV} and {BASELINE_JSON}")
        else:
            _write_json(gsim.SIM_CONFIG_JSON, {
                "round_corr": current_candidate["round_corr"],
                "tail_df": current_candidate["tail_df"],
                "blowup_mix": current_candidate["blowup_mix"],
                "source": "golf.validate --tune-shape; retained control",
                "metrics": {
                    "status": "retained",
                    "selection_split": split,
                    "holdout": holdout,
                    "holdout_opened": bool(selection_pass),
                    "screen_sims": screen_sims,
                    "confirm_sims": confirm_sims,
                    "reason": (
                        "No challenger cleared the pre-holdout improvement "
                        "and market/winner guardrails."
                    ),
                },
            })
            print(f"  recorded retained control in {gsim.SIM_CONFIG_JSON}")
    return out


def tune_weather_coefficients(write: bool = False) -> dict:
    """Estimate conservative weather score coefficients from enriched rounds.

    Requires rounds.csv columns: wind_speed, wind_gust, precipitation. Tee times
    are ideal but not required for this first coefficient pass because the model
    applies coefficients to early/late wave differences at refresh time.
    """
    df = model.load_rounds_df()
    required = ["wind_speed", "wind_gust", "precipitation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            "Weather tuning needs historical weather-enriched rounds.csv columns: "
            + ", ".join(missing)
            + ". Run/implement historical Open-Meteo enrichment before promotion."
        )
    wx = df.dropna(subset=["score_to_par", *required]).copy()
    if len(wx) < 1000:
        raise SystemExit(f"Weather tuning needs >=1000 enriched rounds, found {len(wx)}.")
    wx["gust_delta"] = (wx["wind_gust"].astype(float) - wx["wind_speed"].astype(float)).clip(lower=0)
    # Remove tournament-round setup difficulty first; fit weather to residual setup.
    tr = wx["tournament_id"].astype(str) + "|" + wx["round"].astype(str)
    y = wx["score_to_par"].astype(float) - wx.groupby(tr)["score_to_par"].transform("mean")
    X = wx[["wind_speed", "gust_delta", "precipitation"]].astype(float).values
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
    coefs = {
        "intercept": round(float(beta[0]), 6),
        "wind_speed": round(float(beta[1]), 6),
        "gust_delta": round(float(beta[2]), 6),
        "precipitation": round(float(beta[3]), 6),
        "n_rounds": int(len(wx)),
        "source": "golf.validate --tune-weather",
    }
    print("Weather coefficient fit:")
    for k, v in coefs.items():
        print(f"  {k}: {v}")
    if write:
        from .io_utils import atomic_write_text
        atomic_write_text(WEATHER_CONFIG_JSON, json.dumps(coefs, indent=2) + "\n")
        print(f"  wrote {WEATHER_CONFIG_JSON}")
    return coefs


def ablation_report(since: str, sims: int, seed: int = 0,
                    features: list[str] | None = None) -> dict:
    features = features or ["weather", "global_priors"]
    df = model.load_rounds_df()
    base = walk_forward(df, since=since, sims=sims, seed=seed, verbose=False)
    if base.empty:
        raise SystemExit("No base predictions for ablation.")
    out = {"base": summarize(base), "ablations": {}}
    print("Feature ablation report")
    print(f"  base headline {out['base']['headline_brier']:.5f}")
    for feat in features:
        pred = walk_forward(
            df, since=since, sims=sims, seed=seed, verbose=False,
            feature_flags={feat: False})
        rep = summarize(pred)
        out["ablations"][feat] = rep
        delta = rep["headline_brier"] - out["base"]["headline_brier"]
        print(f"  without {feat:<13} headline {rep['headline_brier']:.5f} Δ {delta:+.5f}")
    return out


def print_report(rep: dict) -> None:
    print(f"\n{'Market':<8}{'N':>7}{'base':>8}{'Brier':>9}{'vs base':>9}"
          f"{'skill':>8}{'logloss':>9}")
    print("-" * 58)
    for mkt in MARKETS:
        r = rep[mkt]
        print(f"{mkt:<8}{r['n']:>7}{r['base_rate']:>8.3f}{r['brier']:>9.4f}"
              f"{r['brier_base']:>9.4f}{r['skill']:>8.1%}{r['logloss']:>9.4f}")
    we = rep["win_event"]
    if we["mean_surprise"] is not None:
        print(f"\nWinner surprise −log p: model {we['mean_surprise']:.3f}  vs "
              f"uniform {we['uniform_surprise']:.3f}  over {we['events']} events "
              f"(lower = better)")
    print(f"\nHeadline Brier (top10+top20+cut): {rep['headline_brier']:.5f}")
    # show reliability for make-cut (the cleanest signal)
    print("\nMake-cut reliability  (pred → actual, n):")
    for pp, yy, nn in rep["cut"]["reliability"]:
        bar = "█" * int(yy * 30)
        print(f"  {pp:>5.2f} → {yy:>5.2f}  {bar}  ({nn})")


def main():
    ap = argparse.ArgumentParser(description="Walk-forward golf backtest + gate")
    ap.add_argument("--since", default="2023-06-01",
                    help="Evaluate events on/after this date (default %(default)s)")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", action="store_true",
                    help="Exit non-zero if headline Brier regresses vs baseline")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tune-config", action="store_true",
                    help="screen/tune free-data fit hyperparameters")
    ap.add_argument("--tune-shape", action="store_true",
                    help="retune round correlation, scoring shape, and "
                         "general course-profile weight")
    ap.add_argument("--selection-split", default="2025-07-01",
                    help="start of the shape-tuning confirmation window")
    ap.add_argument("--holdout", default="2026-01-01",
                    help="start of the sealed shape-tuning holdout")
    ap.add_argument("--tune-weather", action="store_true",
                    help="fit weather coefficients from weather-enriched historical rounds")
    ap.add_argument("--ablate", nargs="*", default=None,
                    help="feature ablation report, e.g. --ablate weather global_priors")
    ap.add_argument("--write", action="store_true",
                    help="with --tune-config, write promoted config")
    ap.add_argument("--rebaseline", action="store_true",
                    help="explicitly replace the frozen validation baseline, "
                         "including after a reviewed data-integrity correction")
    args = ap.parse_args()

    if args.tune_config:
        tune_config(args.since, sims=args.sims, seed=args.seed, write=args.write)
        return

    if args.tune_shape:
        tune_shape(
            args.since,
            sims=args.sims,
            seed=args.seed,
            write=args.write,
            split=args.selection_split,
            holdout=args.holdout,
        )
        return

    if args.tune_weather:
        tune_weather_coefficients(write=args.write)
        return

    if args.ablate is not None:
        ablation_report(args.since, sims=args.sims, seed=args.seed,
                        features=args.ablate or None)
        return

    df = model.load_rounds_df()
    print(f"Walk-forward from {args.since}  ({args.sims:,} sims/event)…")
    pred = walk_forward(df, since=args.since, sims=args.sims, seed=args.seed,
                        verbose=not args.quiet)
    if pred.empty:
        print("No evaluable events — seed more history.")
        sys.exit(1)
    pred.to_csv(PRED_CSV, index=False)
    print(f"\n{len(pred):,} player-event predictions → {PRED_CSV}")

    rep = summarize(pred)
    print_report(rep)

    head = rep["headline_brier"]
    if args.rebaseline:
        _write_json(BASELINE_JSON,
            {"headline_brier": head, "gate_tol": GATE_TOL,
             "asof": pred["date"].max(),
             "schema_version": VALIDATION_SCHEMA_VERSION,
             "point_in_time_safe": True,
             "reason": (
                 "repaired cut-rule leakage; official playoff winners; "
                 "market-specific no-cut eligibility; stable source identities"
             )})
        print(f"\nBaseline explicitly replaced → {BASELINE_JSON}")
        return
    if BASELINE_JSON.exists():
        baseline = json.loads(BASELINE_JSON.read_text())
        safe_baseline = (
                         baseline.get("schema_version") == VALIDATION_SCHEMA_VERSION
                         and baseline.get("point_in_time_safe") is True)
        if not safe_baseline:
            if args.write:
                _write_json(BASELINE_JSON,
                    {"headline_brier": head, "gate_tol": GATE_TOL,
                     "asof": pred["date"].max(),
                     "schema_version": VALIDATION_SCHEMA_VERSION,
                     "point_in_time_safe": True})
                print("Legacy/leaky baseline replaced by explicit honest baseline.")
                return
            if args.gate:
                print("GATE FAIL: baseline predates point-in-time leakage controls; "
                      "re-run validation with --write after reviewing results.")
                sys.exit(2)
            print("Ignoring legacy baseline that predates point-in-time leakage controls.")
            return
        prev = baseline.get("headline_brier", head)
        delta = head - prev
        print(f"\nBaseline headline Brier {prev:.5f}  →  now {head:.5f}  "
              f"(Δ {delta:+.5f}, tol {GATE_TOL})")
        if args.gate and delta > GATE_TOL:
            print("GATE FAIL: model regressed beyond tolerance.")
            sys.exit(2)
        if args.write and delta < -GATE_TOL:  # explicit promotion only
            _write_json(BASELINE_JSON,
                {"headline_brier": head, "gate_tol": GATE_TOL,
                 "asof": pred["date"].max(),
                 "schema_version": VALIDATION_SCHEMA_VERSION,
                 "point_in_time_safe": True})
            print("Improved — baseline updated.")
    elif args.write:
        _write_json(BASELINE_JSON,
            {"headline_brier": head, "gate_tol": GATE_TOL,
             "asof": pred["date"].max(),
             "schema_version": VALIDATION_SCHEMA_VERSION,
             "point_in_time_safe": True})
        print(f"\nBaseline written → {BASELINE_JSON}")
    elif args.gate:
        print("GATE FAIL: no frozen baseline; create one explicitly with --write.")
        sys.exit(2)


if __name__ == "__main__":
    main()
