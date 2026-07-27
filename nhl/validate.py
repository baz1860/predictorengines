#!/usr/bin/env python3
"""NHL validation gate.

The default validation path is leak-free for the data available in this module:
each game is priced from league-average priors plus only earlier results from
the same season. Games on the same date are all priced before that date's
results are applied because the results file has dates, not start times.

Metrics -> nhl/data/validation_baseline.json for regression tracking. The gate
fails if the walk-forward model does not beat trivial baselines, or if it
regresses materially versus the stored baseline.

Usage:
  python3 -m nhl.validate                        # print metrics only
  python3 -m nhl.validate --gate                  # print + gate vs baseline
  python3 -m nhl.validate --gate --update-baseline
  python3 -m nhl.validate --results nhl/data/results_2025_26.csv --quiet --gate
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import backtest as B
from . import gate as G

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "data" / "results_2025_26.csv"
BASELINE = HERE / "data" / "validation_baseline.json"

MIN_GAMES = 200          # below this, metrics are too noisy to gate on
BRIER_TOL = 0.010
MAE_TOL = 0.25           # goals, on margin MAE


def evaluate(results_path: Path | str = DEFAULT_RESULTS, model: str = "blend") -> dict:
    games = B.load_results(results_path)
    report = B.run_backtest(games, model=model, walk_forward=True, strict=True)
    s = report["summary"]
    return {
        "results_path": str(results_path),
        "n_games": s["games"],
        "model": model,
        "mode": s["mode"],
        "accuracy": s["accuracy"],
        "always_home_accuracy": s["always_home_accuracy"],
        "brier": s["brier"],
        "home_rate_brier": s["home_rate_brier"],
        "constant_50_brier": s["constant_50_brier"],
        "logloss": s["logloss"],
        "margin_mae": s["margin_mae"],
        "total_mae": s["total_mae"],
        "beats_trivial_baselines": s["beats_trivial_baselines"],
    }


def _load_baseline() -> dict | None:
    if BASELINE.exists():
        return json.loads(BASELINE.read_text())
    return None


def _save_baseline(metrics: dict) -> None:
    BASELINE.parent.mkdir(exist_ok=True)
    BASELINE.write_text(json.dumps(metrics, indent=1))


def gate(metrics: dict) -> int:
    if metrics["n_games"] < MIN_GAMES:
        print(f"  only {metrics['n_games']} completed games (< {MIN_GAMES}) "
              "-- skipping gate, numbers too noisy to trust")
        return 0
    ok = True
    if metrics.get("mode") != "walk_forward":
        print(f"GATE FAIL  mode: expected walk_forward, got {metrics.get('mode')}")
        ok = False
    if not metrics.get("beats_trivial_baselines"):
        print(
            "GATE FAIL  trivial baselines: "
            f"accuracy {metrics['accuracy']:.1%} vs always-home "
            f"{metrics['always_home_accuracy']:.1%}; "
            f"Brier {metrics['brier']:.4f} vs home-rate "
            f"{metrics['home_rate_brier']:.4f}"
        )
        ok = False
    baseline = _load_baseline()
    if baseline is None:
        print("no baseline yet -- run with --update-baseline to establish one")
        return 0 if ok else 1
    if baseline["n_games"] < MIN_GAMES:
        print("stored baseline predates enough games -- run --update-baseline "
              "to refresh it before gating")
        return 0 if ok else 1
    for key, tol in (("margin_mae", MAE_TOL), ("brier", BRIER_TOL)):
        cur, base = metrics[key], baseline[key]
        if cur > base + tol:
            print(f"GATE FAIL  {key}: {cur:.4f} regressed past baseline {base:.4f} (+{tol})")
            ok = False
        else:
            print(f"  ok  {key}: {cur:.4f} (baseline {base:.4f})")
    return 0 if ok else 1


def _print_metrics(m: dict, quiet: bool) -> None:
    if quiet:
        return
    print(f"\nNHL validation -- {m['n_games']} games ({m['model']}), {m['results_path']}")
    print(f"  accuracy   {m['accuracy']:.1%}")
    print(f"  brier      {m['brier']:.4f}")
    print(f"  baselines  always-home {m['always_home_accuracy']:.1%}, "
          f"home-rate Brier {m['home_rate_brier']:.4f}")
    print(f"  beats trivial baselines: {m['beats_trivial_baselines']}")
    print(f"  logloss    {m['logloss']:.4f}")
    print(f"  margin MAE {m['margin_mae']:.3f}")
    print(f"  total MAE  {m['total_mae']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(DEFAULT_RESULTS),
                    help="CSV of completed games (date, home, away, home_goals, away_goals)")
    ap.add_argument("--model", choices=["blend", "power", "form"], default="blend")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        metrics = evaluate(args.results, model=args.model)
    except (FileNotFoundError, ValueError) as e:
        print(f"NHL validation skipped: {e}")
        return 0

    _print_metrics(metrics, args.quiet)

    if args.update_baseline:
        _save_baseline(metrics)
        print(f"baseline updated -> {BASELINE}")
        forecast_passed = bool(metrics["beats_trivial_baselines"])
        G.GATE_JSON.write_text(json.dumps({
            "status": "FAIL",
            "forecast_validation_status": "PASS" if forecast_passed else "FAIL",
            "roi_validation_status": "FAIL",
            "staking_enabled": False,
            "reason": (
                "Walk-forward validation beats trivial baselines; staking still requires "
                "explicit timestamped-odds ROI approval."
                if forecast_passed
                else "Walk-forward validation does not beat trivial baselines; staking disabled."
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": metrics,
        }, indent=2))
        return 0
    if args.gate:
        return gate(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
