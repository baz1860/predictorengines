"""Chronological walk-forward validation for the horse-racing win model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from . import model as M
from .features import build_feature_frame
from .schema import (DATA_DIR, DataError, latest_odds_snapshot, load_bundle,
                     race_cutoff, race_row, runner_snapshot)

REPORT_JSON = DATA_DIR / "validation_report.json"
BASELINE_JSON = DATA_DIR / "validation_baseline.json"


def _provider_manifest(data_dir: str | Path | None) -> dict | None:
    root = Path(data_dir) if data_dir else DATA_DIR
    path = root / "provider_manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DataError(f"provider manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        return None
    market_path = root / "betfair_manifest.json"
    if market_path.exists():
        try:
            market = json.loads(market_path.read_text())
        except (OSError, ValueError) as exc:
            raise DataError(f"Betfair manifest is unreadable: {exc}") from exc
        if isinstance(market, dict):
            payload = payload | {"market_data": market}
    return payload


def _metrics(frame: pd.DataFrame, column: str) -> dict:
    losses, briers, winners = [], [], []
    for _rid, g in frame.groupby("race_id", sort=False):
        winner = g[g["won"] == 1]
        if len(winner) != 1:
            continue
        p = g[column].astype(float).to_numpy()
        y = g["won"].astype(float).to_numpy()
        losses.append(-np.log(max(float(winner.iloc[0][column]), 1e-12)))
        briers.append(float(np.sum((p - y) ** 2)))
        winners.append(float(winner.iloc[0][column]))
    return {"races": len(losses), "log_loss": float(np.mean(losses)),
            "race_brier": float(np.mean(briers)),
            "winner_probability_mean": float(np.mean(winners))}


def _paired_logloss(frame: pd.DataFrame, a: str, b: str, seed: int = 17) -> dict:
    """Per-race log-loss difference with a day-clustered bootstrap CI.

    Races on the same meeting day share going, weather, market conditions and
    often stables, so the bootstrap resamples whole days rather than races
    (plan section 8).
    """
    diffs, days = [], []
    for _rid, group in frame.groupby("race_id", sort=False):
        winner = group[group["won"] == 1]
        if len(winner) != 1 or winner[[a, b]].isna().any(axis=None):
            continue
        diffs.append(-np.log(max(float(winner.iloc[0][a]), 1e-12))
                     + np.log(max(float(winner.iloc[0][b]), 1e-12)))
        stamp = winner.iloc[0].get("cutoff")
        days.append(str(pd.Timestamp(stamp).date()) if pd.notna(stamp) else "unknown")
    values = np.asarray(diffs, dtype=float)
    if len(values) == 0:
        return {"races": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    clusters: dict[str, list[int]] = {}
    for i, day in enumerate(days):
        clusters.setdefault(day, []).append(i)
    cluster_idx = [np.asarray(idx, dtype=int) for idx in clusters.values()]
    n_clusters = len(cluster_idx)
    boot = np.empty(2000, dtype=float)
    for b_i in range(2000):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        sample = np.concatenate([cluster_idx[c] for c in chosen])
        boot[b_i] = float(values[sample].mean())
    return {"races": int(len(values)), "days": n_clusters,
            "mean": float(values.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "bootstrap": "day_clustered"}


_FIELD_BANDS = ((0, 7, "small_le7"), (8, 11, "medium_8_11"), (12, 99, "large_ge12"))
_DIST_GROUPS = ((0, 1300, "sprint_le1300"), (1301, 1900, "mile_1301_1900"),
                (1901, 2500, "middle_1901_2500"), (2501, 99999, "staying_gt2500"))


def _band(value: float, bands) -> str:
    for low, high, name in bands:
        if low <= value <= high:
            return name
    return "unknown"


def _slice_reports(frame: pd.DataFrame, min_races: int = 20) -> dict:
    """Per-slice model vs uniform metrics (plan section 8 slice reporting)."""
    work = frame.copy()
    sizes = work.groupby("race_id")["runner_id"].transform("size")
    work["_field_band"] = sizes.map(lambda n: _band(float(n), _FIELD_BANDS))
    if "distance_band" in work:
        work["_dist_group"] = pd.to_numeric(work["distance_band"], errors="coerce") \
            .fillna(0).map(lambda d: _band(float(d), _DIST_GROUPS))
    work["_month"] = work["cutoff"].map(
        lambda t: str(pd.Timestamp(t).strftime("%Y-%m")) if pd.notna(t) else "unknown")
    dimensions = {"surface": "surface", "going": "going_group",
                  "handicap": "handicap_flag", "field_size": "_field_band",
                  "distance": "_dist_group", "month": "_month",
                  "course": "course_id"}
    report: dict = {}
    for label, column in dimensions.items():
        if column not in work:
            continue
        section = {}
        for value, group in work.groupby(work[column].astype(str), sort=True):
            races = group["race_id"].nunique()
            if races < min_races:
                continue
            entry = {"model": _metrics(group, "p_model"),
                     "uniform": _metrics(group, "p_uniform")}
            section[str(value)] = entry
        if section:
            report[label] = section
    return report


def _attach_market_probabilities(frame: pd.DataFrame, bundle) -> pd.DataFrame:
    out = frame.copy()
    out["p_market"] = np.nan
    for rid, group in out.groupby("race_id", sort=False):
        race = race_row(bundle, str(rid))
        cutoff = race_cutoff(race)
        runners = runner_snapshot(bundle, str(rid), cutoff)
        odds = latest_odds_snapshot(bundle, str(rid), cutoff)
        if odds.empty or set(odds["runner_id"].astype(str)) != set(runners["runner_id"].astype(str)):
            continue
        span = (odds["captured_at"].max() - odds["captured_at"].min()).total_seconds()
        age = (cutoff - odds["captured_at"].max()).total_seconds()
        if span > 300 or age < 0 or age > 1800:
            continue
        inv = 1.0 / pd.to_numeric(odds["decimal_odds"], errors="coerce")
        if inv.isna().any() or float(inv.sum()) <= 0:
            continue
        mapping = dict(zip(odds["runner_id"].astype(str), inv / float(inv.sum())))
        out.loc[group.index, "p_market"] = group["runner_id"].astype(str).map(mapping)
    return out


def _attach_starting_price_probabilities(frame: pd.DataFrame, data_dir) -> pd.DataFrame:
    """Attach a post-race market benchmark without treating it as cutoff data."""
    out = frame.copy()
    out["p_starting_price"] = np.nan
    path = Path(data_dir) / "starting_prices.csv" if data_dir else DATA_DIR / "starting_prices.csv"
    if not path.exists():
        return out
    prices = pd.read_csv(path, dtype={"race_id": str, "runner_id": str})
    prices["decimal_odds"] = pd.to_numeric(prices["decimal_odds"], errors="coerce")
    prices = prices[np.isfinite(prices["decimal_odds"]) & (prices["decimal_odds"] > 1)]
    for rid, group in out.groupby("race_id", sort=False):
        board = prices[prices["race_id"].astype(str) == str(rid)]
        if set(board["runner_id"].astype(str)) != set(group["runner_id"].astype(str)):
            continue
        inv = 1.0 / board["decimal_odds"].astype(float)
        if float(inv.sum()) <= 0:
            continue
        mapping = dict(zip(board["runner_id"].astype(str), inv / float(inv.sum())))
        out.loc[group.index, "p_starting_price"] = group["runner_id"].astype(str).map(mapping)
    return out


def _subset_profile(frame: pd.DataFrame, bundle) -> dict:
    if frame.empty:
        return {"races": 0}
    race_ids = set(frame["race_id"].astype(str))
    metadata = bundle.races[bundle.races["race_id"].astype(str).isin(race_ids)].copy()
    sizes = frame.groupby("race_id").size()
    profile = {
        "races": int(len(race_ids)),
        "model": _metrics(frame, "p_model"),
        "field_size_mean": float(sizes.mean()),
        "jurisdiction_counts": {
            str(key): int(value) for key, value in
            metadata["jurisdiction"].astype(str).value_counts().sort_index().items()
        },
        "surface_counts": {
            str(key): int(value) for key, value in
            metadata["surface"].astype(str).value_counts().sort_index().items()
        },
    }
    complete_sp = frame[frame.groupby("race_id")["p_starting_price"].transform(
        lambda values: values.notna().all())]
    if not complete_sp.empty:
        profile["starting_price"] = _metrics(complete_sp, "p_starting_price")
    return profile


def walk_forward(data_dir: str | Path | None = None, min_train: int = 30,
                 test_size: int = 15, l2: float = 1.0,
                 half_life_scale: float = 1.0, features: list[str] | None = None,
                 lockbox_frac: float = 0.0,
                 frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Expanding chronological walk-forward evaluation.

    ``min_train`` doubles as the warm-up segment: those races initialize
    state and coefficients but are never scored. ``lockbox_frac`` withholds
    the final fraction of races entirely (plan section 8); they are neither
    fitted nor scored, and remain untouched until a deliberate final read.
    ``features`` restricts the model to a feature subset (ablation support);
    ``frame`` allows the caller to reuse a prebuilt feature frame.
    """
    bundle = load_bundle(data_dir)
    provider_manifest = _provider_manifest(data_dir)
    if frame is None:
        frame = M._valid_labelled(build_feature_frame(
            bundle, half_life_scale=half_life_scale))
    race_order = (frame[["race_id", "scheduled_off_utc"]].drop_duplicates()
                  .sort_values("scheduled_off_utc").reset_index(drop=True))
    lockbox_races = 0
    if lockbox_frac > 0:
        lockbox_races = int(len(race_order) * float(lockbox_frac))
        if lockbox_races:
            locked = set(race_order.iloc[-lockbox_races:]["race_id"].astype(str))
            race_order = race_order.iloc[:-lockbox_races].reset_index(drop=True)
            frame = frame[~frame["race_id"].astype(str).isin(locked)].copy()
    if len(race_order) < min_train + 5:
        raise DataError(f"need at least {min_train + 5} valid races for walk-forward; "
                        f"found {len(race_order)}")
    predictions = []
    folds = []
    for start in range(min_train, len(race_order), test_size):
        test_meta = race_order.iloc[start:start + test_size]
        if test_meta.empty:
            continue
        train_ids = set(race_order.iloc[:start]["race_id"].astype(str))
        test_ids = set(test_meta["race_id"].astype(str))
        train = frame[frame["race_id"].astype(str).isin(train_ids)]
        test = frame[frame["race_id"].astype(str).isin(test_ids)].copy()

        inner_order = race_order.iloc[:start]
        inner_split = max(10, int(len(inner_order) * 0.8))
        fit_ids = set(inner_order.iloc[:inner_split]["race_id"].astype(str))
        cal_ids = set(inner_order.iloc[inner_split:]["race_id"].astype(str))
        fit_frame = train[train["race_id"].astype(str).isin(fit_ids)]
        cal_frame = train[train["race_id"].astype(str).isin(cal_ids)]
        beta0, means0, scales0, _ = M._fit_coefficients(fit_frame, l2=l2,
                                                        features=features)
        temperature = M._temperature(cal_frame, beta0, means0, scales0,
                                     features=features) \
            if len(cal_ids) >= 5 else 1.0
        beta, means, scales, _ = M._fit_coefficients(train, l2=l2, features=features)
        test["p_model"] = M._probabilities(test, beta, means, scales, temperature,
                                           features=features)
        # Transparent race-relative official-rating baseline.
        test["p_uniform"] = test.groupby("race_id")["runner_id"].transform(
            lambda s: 1.0 / len(s))
        official = pd.to_numeric(test["official_rating_rel"], errors="coerce").fillna(0.0)
        test["p_official"] = 0.0
        for _rid, idx in test.groupby("race_id", sort=False).groups.items():
            z = official.loc[idx].to_numpy()
            test.loc[idx, "p_official"] = np.exp(z - logsumexp(z))
        predictions.append(test)
        folds.append({"train_races": len(train_ids), "test_races": len(test_ids),
                      "test_start": str(test_meta.iloc[0]["scheduled_off_utc"]),
                      "test_end": str(test_meta.iloc[-1]["scheduled_off_utc"]),
                      "temperature": float(temperature)})
    out = _attach_market_probabilities(pd.concat(predictions, ignore_index=True), bundle)
    out = _attach_starting_price_probabilities(out, data_dir)
    complete_market_races = [rid for rid, group in out.groupby("race_id", sort=False)
                             if group["p_market"].notna().all()]
    market = out[out["race_id"].isin(complete_market_races)]
    unmatched = out[~out["race_id"].isin(complete_market_races)]
    metrics = {name: _metrics(out, name)
               for name in ("p_model", "p_uniform", "p_official")}
    if not market.empty:
        metrics["p_market"] = _metrics(market, "p_market")
        metrics["p_model_market_subset"] = _metrics(market, "p_model")
        market_with_sp = market[market.groupby("race_id")["p_starting_price"].transform(
            lambda values: values.notna().all())]
        if not market_with_sp.empty:
            metrics["p_starting_price_market_subset"] = _metrics(
                market_with_sp, "p_starting_price")
    sp_races = [rid for rid, group in out.groupby("race_id", sort=False)
                if group["p_starting_price"].notna().all()]
    starting = out[out["race_id"].isin(sp_races)]
    if not starting.empty:
        metrics["p_starting_price"] = _metrics(starting, "p_starting_price")
        metrics["p_model_starting_price_subset"] = _metrics(starting, "p_model")
    report = {"model": M.MODEL_NAME, "folds": folds, "metrics": metrics,
              "evaluation": {
                  "scheme": "expanding_chronological_walk_forward",
                  "warmup_races": int(min_train),
                  "test_block_races": int(test_size),
                  "lockbox_races_withheld": int(lockbox_races),
                  "half_life_scale": float(half_life_scale),
                  "features": list(features) if features is not None else "all",
                  "primary_metric": "log_loss",
                  "secondary_metrics": ["race_brier", "calibration"],
                  "diagnostic_only": ["winner_accuracy"],
              },
              "slices": _slice_reports(out),
              "data_provenance": provider_manifest or {
                  "provider": "canonical_manual", "validation_grade": "unknown"},
              "paired_logloss": {
                  "model_minus_uniform": _paired_logloss(out, "p_model", "p_uniform"),
                  "model_minus_official": _paired_logloss(out, "p_model", "p_official"),
                  "model_minus_market": _paired_logloss(market, "p_model", "p_market"),
                  "market_minus_starting_price": _paired_logloss(
                      market, "p_market", "p_starting_price"),
                  "model_minus_starting_price": _paired_logloss(
                      starting, "p_model", "p_starting_price"),
              },
              "market_selection": {
                  "matched": _subset_profile(market, bundle),
                  "unmatched": _subset_profile(unmatched, bundle),
                  "scope_note": (
                      "Market comparisons apply only to held-out races with an exact identity "
                      "join and a complete, open, fresh cutoff board. Matched/unmatched "
                      "differences quantify selection bias; they are not causal adjustments."),
              }}
    return out, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Horse-racing walk-forward validation")
    ap.add_argument("--data-dir")
    ap.add_argument("--min-train", type=int, default=30)
    ap.add_argument("--test-size", type=int, default=15)
    ap.add_argument("--half-life-scale", type=float, default=1.0)
    ap.add_argument("--lockbox-frac", type=float, default=0.0,
                    help="withhold the final fraction of races entirely")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args(argv)
    manifest = _provider_manifest(args.data_dir)
    grade = (manifest or {}).get("validation_grade", "unknown")
    if manifest is not None and (args.gate or args.write_baseline) \
            and grade != "point_in_time":
        action = "baseline REFUSED" if args.write_baseline else "gate UNPROVEN"
        print(f"{action}: validation_grade={grade}; point_in_time required")
        return 2
    try:
        _pred, report = walk_forward(args.data_dir, args.min_train, args.test_size,
                                     half_life_scale=args.half_life_scale,
                                     lockbox_frac=args.lockbox_frac)
    except (DataError, ValueError) as exc:
        print(f"horse_racing validation unavailable: {exc}")
        return 2
    report_path = (Path(args.data_dir) / "validation_report.json"
                   if args.data_dir else REPORT_JSON)
    baseline_path = (Path(args.data_dir) / "validation_baseline.json"
                     if args.data_dir else BASELINE_JSON)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for name, metrics in report["metrics"].items():
        print(f"{name:<12s} races={metrics['races']:4d} "
              f"logloss={metrics['log_loss']:.5f} brier={metrics['race_brier']:.5f}")
    if args.write_baseline:
        baseline_path.write_text(json.dumps(report["metrics"]["p_model"], indent=2) + "\n")
        print(f"wrote {baseline_path}")
    if args.gate:
        if not baseline_path.exists():
            print("gate UNPROVEN: no committed validation_baseline.json")
            return 2
        baseline = json.loads(baseline_path.read_text())
        current = report["metrics"]["p_model"]
        if current["log_loss"] > float(baseline["log_loss"]) + 0.005:
            print("gate FAIL: log-loss regression")
            return 1
        print("gate PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
