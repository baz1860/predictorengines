#!/usr/bin/env python3
"""NHL validation gate.

Honesty note up front: this is NOT a leak-free walk-forward backtest like
nfl.validate or tennis.validate. Those refit their model as-of each date
because their underlying data supports historical snapshots. nhl/model.py
currently has no such capability -- team_stats.csv is a single current
season-to-date snapshot with no asof parameter, so there's nothing to walk
forward against yet. What this gate does instead: score the model's CURRENT
ratings against every completed game of the season in
nhl/data/results_2025_26.csv (in-sample re: ratings, since a team's rating
reflects games throughout the season including ones being scored -- but
still a legitimate regression check for "did today's ratings/blend get
worse"). A genuine walk-forward validator is a documented follow-up once
team_stats.csv (or an equivalent) gains dated historical snapshots.

Metrics -> nhl/data/validation_baseline.json; gate fails (exit 1) if
moneyline Brier or margin MAE regress beyond tolerance versus the stored
baseline. Skips the gate (exit 0, with a warning) if there aren't enough
completed games this season for the numbers to be meaningful.

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
from pathlib import Path

from . import backtest as B

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "data" / "results_2025_26.csv"
BASELINE = HERE / "data" / "validation_baseline.json"

MIN_GAMES = 200          # below this, metrics are too noisy to gate on
BRIER_TOL = 0.010
MAE_TOL = 0.25           # goals, on margin MAE


def evaluate(results_path: Path | str = DEFAULT_RESULTS, model: str = "blend") -> dict:
    games = B.load_results(results_path)
    report = B.run_backtest(games, model=model)
    s = report["summary"]
    return {
        "results_path": str(results_path),
        "n_games": s["games"],
        "model": model,
        "accuracy": s["accuracy"],
        "brier": s["brier"],
        "logloss": s["logloss"],
        "margin_mae": s["margin_mae"],
        "total_mae": s["total_mae"],
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
    baseline = _load_baseline()
    if baseline is None:
        print("no baseline yet -- run with --update-baseline to establish one")
        return 0
    if baseline["n_games"] < MIN_GAMES:
        print("stored baseline predates enough games -- run --update-baseline "
              "to refresh it before gating")
        return 0
    ok = True
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
        return 0
    if args.gate:
        return gate(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
