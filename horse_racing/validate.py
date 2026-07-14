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
    diffs = []
    for _rid, group in frame.groupby("race_id", sort=False):
        winner = group[group["won"] == 1]
        if len(winner) != 1 or winner[[a, b]].isna().any(axis=None):
            continue
        diffs.append(-np.log(max(float(winner.iloc[0][a]), 1e-12))
                     + np.log(max(float(winner.iloc[0][b]), 1e-12)))
    values = np.asarray(diffs, dtype=float)
    if len(values) == 0:
        return {"races": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    return {"races": int(len(values)), "mean": float(values.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}


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


def walk_forward(data_dir: str | Path | None = None, min_train: int = 30,
                 test_size: int = 15, l2: float = 1.0) -> tuple[pd.DataFrame, dict]:
    bundle = load_bundle(data_dir)
    frame = M._valid_labelled(build_feature_frame(bundle))
    race_order = (frame[["race_id", "scheduled_off_utc"]].drop_duplicates()
                  .sort_values("scheduled_off_utc").reset_index(drop=True))
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
        beta0, means0, scales0, _ = M._fit_coefficients(fit_frame, l2=l2)
        temperature = M._temperature(cal_frame, beta0, means0, scales0) \
            if len(cal_ids) >= 5 else 1.0
        beta, means, scales, _ = M._fit_coefficients(train, l2=l2)
        test["p_model"] = M._probabilities(test, beta, means, scales, temperature)
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
    complete_market_races = [rid for rid, group in out.groupby("race_id", sort=False)
                             if group["p_market"].notna().all()]
    market = out[out["race_id"].isin(complete_market_races)]
    metrics = {name: _metrics(out, name)
               for name in ("p_model", "p_uniform", "p_official")}
    if not market.empty:
        metrics["p_market"] = _metrics(market, "p_market")
        metrics["p_model_market_subset"] = _metrics(market, "p_model")
    report = {"model": M.MODEL_NAME, "folds": folds, "metrics": metrics,
              "paired_logloss": {
                  "model_minus_uniform": _paired_logloss(out, "p_model", "p_uniform"),
                  "model_minus_official": _paired_logloss(out, "p_model", "p_official"),
                  "model_minus_market": _paired_logloss(market, "p_model", "p_market"),
              }}
    return out, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Horse-racing walk-forward validation")
    ap.add_argument("--data-dir")
    ap.add_argument("--min-train", type=int, default=30)
    ap.add_argument("--test-size", type=int, default=15)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args(argv)
    try:
        _pred, report = walk_forward(args.data_dir, args.min_train, args.test_size)
    except (DataError, ValueError) as exc:
        print(f"horse_racing validation unavailable: {exc}")
        return 2
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    for name, metrics in report["metrics"].items():
        print(f"{name:<12s} races={metrics['races']:4d} "
              f"logloss={metrics['log_loss']:.5f} brier={metrics['race_brier']:.5f}")
    if args.write_baseline:
        BASELINE_JSON.write_text(json.dumps(report["metrics"]["p_model"], indent=2) + "\n")
        print(f"wrote {BASELINE_JSON}")
    if args.gate:
        if not BASELINE_JSON.exists():
            print("gate UNPROVEN: no committed validation_baseline.json")
            return 2
        baseline = json.loads(BASELINE_JSON.read_text())
        current = report["metrics"]["p_model"]
        if current["log_loss"] > float(baseline["log_loss"]) + 0.005:
            print("gate FAIL: log-loss regression")
            return 1
        print("gate PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
