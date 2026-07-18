"""Feature-family experiment ladder with promotion gates (plan sections 3, 8, 9).

Runs the recommended experiment order as controlled, chronological ablations:

    rung 0   core            (V1-parity baseline, frozen reference)
    rung 1   + form_multi    (multi-horizon state features)
    rung 2   + class_struct  (class moves, rating/weight changes)
    rung 3   + suitability   (going/course-distance/continuous distance)
    rung 4   + draw_hier     (hierarchical draw modeling)
    rung 5   + weight_rating (weight and rating interactions)

then a decay half-life grid on the winning feature set. Every rung is scored
with the shared walk-forward scheme, and promotion is decided by explicit
gates BEFORE looking at the numbers:

- paired day-clustered log-loss delta vs the previous accepted rung must have
  mean <= -margin and a 95% CI upper bound < 0;
- no reported slice with >= ``slice_min_races`` races may regress by more
  than ``slice_tolerance`` log loss;
- race Brier (calibration proxy) must not worsen by more than
  ``brier_tolerance``.

Gate thresholds are CLI-configurable and should be recalculated once the
larger backfill is in place (plan section 9).

Usage:
    python3 -m horse_racing.experiments --data-dir horse_racing/data/rpscrape \
        --min-train 500 --test-size 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import model as M
from .features import ALL_FAMILIES, build_feature_frame, feature_names
from .registry import verify_registry
from .schema import DataError, load_bundle
from .validate import _metrics, _paired_logloss, _slice_reports, walk_forward

LADDER: list[tuple[str, tuple[str, ...]]] = [
    ("core", ("core",)),
    ("+form_multi", ("core", "form_multi")),
    ("+class_struct", ("core", "form_multi", "class_struct")),
    ("+suitability", ("core", "form_multi", "class_struct", "suitability")),
    ("+draw_hier", ("core", "form_multi", "class_struct", "suitability",
                    "draw_hier")),
    ("+weight_rating", ALL_FAMILIES),
]
DEFAULT_HL_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)


def _slice_regressions(prev_slices: dict, curr_slices: dict,
                       tolerance: float, min_races: int) -> list[dict]:
    regressions = []
    for dimension, prev_section in prev_slices.items():
        curr_section = curr_slices.get(dimension, {})
        for value, prev_entry in prev_section.items():
            curr_entry = curr_section.get(value)
            if not curr_entry:
                continue
            prev_m, curr_m = prev_entry["model"], curr_entry["model"]
            if min(prev_m["races"], curr_m["races"]) < min_races:
                continue
            if prev_m["log_loss"] is None or curr_m["log_loss"] is None:
                continue
            delta = curr_m["log_loss"] - prev_m["log_loss"]
            if delta > tolerance:
                regressions.append({"dimension": dimension, "slice": value,
                                    "races": curr_m["races"],
                                    "log_loss_delta": round(float(delta), 5)})
    return regressions


def _evaluate_gates(delta: dict, prev_report: dict, curr_report: dict,
                    margin: float, slice_tolerance: float,
                    brier_tolerance: float, slice_min_races: int) -> dict:
    reasons = []
    mean = delta.get("mean")
    upper = (delta.get("ci95") or [None, None])[1]
    if mean is None:
        reasons.append("no comparable races")
    else:
        if mean > -margin:
            reasons.append(f"mean delta {mean:+.5f} above -{margin}")
        if upper is None or upper >= 0:
            reasons.append(f"ci95 upper {upper} not below 0")
    prev_brier = prev_report["metrics"]["p_model"]["race_brier"]
    curr_brier = curr_report["metrics"]["p_model"]["race_brier"]
    brier_delta = curr_brier - prev_brier
    if brier_delta > brier_tolerance:
        reasons.append(f"race brier worsened by {brier_delta:+.5f} "
                       f"(tolerance {brier_tolerance})")
    regressions = _slice_regressions(prev_report.get("slices", {}),
                                     curr_report.get("slices", {}),
                                     slice_tolerance, slice_min_races)
    if regressions:
        reasons.append(f"{len(regressions)} slice regression(s) beyond "
                       f"{slice_tolerance}")
    return {"promoted": not reasons, "reasons": reasons,
            "paired_delta": delta, "race_brier_delta": float(brier_delta),
            "slice_regressions": regressions}


def _paired_between(curr: pd.DataFrame, prev: pd.DataFrame) -> dict:
    merged = curr.merge(
        prev[["race_id", "runner_id", "p_model"]].rename(columns={"p_model": "p_prev"}),
        on=["race_id", "runner_id"], how="inner")
    return _paired_logloss(merged, "p_model", "p_prev")


def run_ladder(data_dir: str | Path, min_train: int = 30, test_size: int = 15,
               l2: float = 1.0, hl_grid: tuple[float, ...] = DEFAULT_HL_GRID,
               gate_margin: float = 0.002, slice_tolerance: float = 0.05,
               brier_tolerance: float = 0.005, slice_min_races: int = 50,
               lockbox_frac: float = 0.0, skip_hl_search: bool = False) -> dict:
    verify_registry()
    bundle = load_bundle(data_dir)
    print(f"building full feature frame (scale=1.0) over "
          f"{len(bundle.races)} races ...", flush=True)
    base_frame = M._valid_labelled(build_feature_frame(bundle))

    rungs = []
    accepted_preds, accepted_report, accepted_name = None, None, None
    for name, families in LADDER:
        features = feature_names(families)
        preds, report = walk_forward(data_dir, min_train, test_size, l2=l2,
                                     features=features, lockbox_frac=lockbox_frac,
                                     frame=base_frame.copy())
        entry = {"rung": name, "families": list(families),
                 "n_features": len(features),
                 "model": report["metrics"]["p_model"],
                 "uniform": report["metrics"]["p_uniform"]}
        if accepted_preds is None:
            entry["gates"] = {"promoted": True,
                              "reasons": ["frozen baseline (plan step 1)"]}
            accepted_preds, accepted_report, accepted_name = preds, report, name
        else:
            delta = _paired_between(preds, accepted_preds)
            entry["gates"] = _evaluate_gates(delta, accepted_report, report,
                                             gate_margin, slice_tolerance,
                                             brier_tolerance, slice_min_races)
            entry["compared_to"] = accepted_name
            if entry["gates"]["promoted"]:
                accepted_preds, accepted_report, accepted_name = preds, report, name
        print(f"  {name:<16s} logloss={entry['model']['log_loss']:.5f} "
              f"races={entry['model']['races']} "
              f"promoted={entry['gates']['promoted']}", flush=True)
        rungs.append(entry)

    hl_results = []
    if not skip_hl_search:
        best_families = tuple(
            next(entry["families"] for entry in reversed(rungs)
                 if entry["gates"]["promoted"]))
        features = feature_names(best_families)
        for scale in hl_grid:
            if float(scale) == 1.0:
                frame = base_frame.copy()
            else:
                print(f"building feature frame at half-life scale {scale} ...",
                      flush=True)
                frame = M._valid_labelled(build_feature_frame(
                    bundle, half_life_scale=float(scale)))
            preds, report = walk_forward(data_dir, min_train, test_size, l2=l2,
                                         features=features,
                                         half_life_scale=float(scale),
                                         lockbox_frac=lockbox_frac, frame=frame)
            delta = _paired_between(preds, accepted_preds)
            hl_results.append({"half_life_scale": float(scale),
                               "model": report["metrics"]["p_model"],
                               "paired_delta_vs_accepted": delta})
            print(f"  hl_scale={scale:<5} "
                  f"logloss={report['metrics']['p_model']['log_loss']:.5f}",
                  flush=True)

    result = {
        "experiment": "feature_family_ladder_v2",
        "config": {"min_train": min_train, "test_size": test_size, "l2": l2,
                   "gate_margin": gate_margin, "slice_tolerance": slice_tolerance,
                   "brier_tolerance": brier_tolerance,
                   "slice_min_races": slice_min_races,
                   "lockbox_frac": lockbox_frac,
                   "note": "gate thresholds provisional until the full backfill "
                           "is in place (plan section 9)"},
        "rungs": rungs,
        "accepted": accepted_name,
        "accepted_metrics": accepted_report["metrics"]["p_model"],
        "reference_metrics": {
            key: accepted_report["metrics"][key]
            for key in ("p_uniform", "p_official", "p_market", "p_starting_price")
            if key in accepted_report["metrics"]},
        "accepted_slices": accepted_report.get("slices", {}),
        "half_life_search": hl_results,
        "data_provenance": accepted_report.get("data_provenance"),
    }
    return result


def _markdown(result: dict) -> str:
    lines = ["# Horse Racing V2 — Feature Ladder Report", ""]
    cfg = result["config"]
    lines += [f"Walk-forward: warm-up {cfg['min_train']} races, "
              f"blocks of {cfg['test_size']}, L2={cfg['l2']}, "
              f"lockbox={cfg['lockbox_frac']}.",
              f"Gates: paired Δlogloss mean ≤ -{cfg['gate_margin']} with CI95 "
              f"upper < 0; slice tolerance {cfg['slice_tolerance']}; "
              f"Brier tolerance {cfg['brier_tolerance']}.", "",
              "| Rung | Features | Log loss | Δ vs accepted | Promoted |",
              "|------|----------|----------|---------------|----------|"]
    for entry in result["rungs"]:
        delta = entry.get("gates", {}).get("paired_delta", {})
        mean = delta.get("mean")
        delta_txt = f"{mean:+.5f}" if isinstance(mean, float) else "—"
        lines.append(f"| {entry['rung']} | {entry['n_features']} | "
                     f"{entry['model']['log_loss']:.5f} | {delta_txt} | "
                     f"{'✅' if entry['gates']['promoted'] else '❌'} |")
    lines += ["", f"**Accepted configuration:** `{result['accepted']}`", ""]
    for entry in result["rungs"]:
        reasons = entry["gates"].get("reasons")
        if reasons and not entry["gates"]["promoted"]:
            lines.append(f"- {entry['rung']} rejected: {'; '.join(reasons)}")
    if result["half_life_search"]:
        lines += ["", "## Half-life scale search", "",
                  "| Scale | Log loss | Δ vs accepted |", "|-------|----------|----|"]
        for entry in result["half_life_search"]:
            mean = entry["paired_delta_vs_accepted"].get("mean")
            delta_txt = f"{mean:+.5f}" if isinstance(mean, float) else "—"
            lines.append(f"| {entry['half_life_scale']} | "
                         f"{entry['model']['log_loss']:.5f} | {delta_txt} |")
    ref = result.get("reference_metrics", {})
    if ref:
        lines += ["", "## Reference benchmarks", ""]
        for key, metrics in ref.items():
            lines.append(f"- `{key}`: log loss {metrics['log_loss']:.5f} "
                         f"over {metrics['races']} races")
    lines += ["", "_Slices, provenance and full gate detail: "
              "experiments_report.json_", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Horse-racing V2 experiment ladder")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--min-train", type=int, default=30)
    ap.add_argument("--test-size", type=int, default=15)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--gate-margin", type=float, default=0.002)
    ap.add_argument("--slice-tolerance", type=float, default=0.05)
    ap.add_argument("--brier-tolerance", type=float, default=0.005)
    ap.add_argument("--slice-min-races", type=int, default=50)
    ap.add_argument("--lockbox-frac", type=float, default=0.0)
    ap.add_argument("--hl-grid", default=",".join(str(s) for s in DEFAULT_HL_GRID))
    ap.add_argument("--skip-hl-search", action="store_true")
    args = ap.parse_args(argv)
    grid = tuple(float(part) for part in args.hl_grid.split(",") if part.strip())
    try:
        result = run_ladder(args.data_dir, args.min_train, args.test_size,
                            l2=args.l2, hl_grid=grid,
                            gate_margin=args.gate_margin,
                            slice_tolerance=args.slice_tolerance,
                            brier_tolerance=args.brier_tolerance,
                            slice_min_races=args.slice_min_races,
                            lockbox_frac=args.lockbox_frac,
                            skip_hl_search=args.skip_hl_search)
    except (DataError, ValueError) as exc:
        print(f"experiment ladder unavailable: {exc}")
        return 2
    root = Path(args.data_dir)
    (root / "experiments_report.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n")
    (root / "experiments_report.md").write_text(_markdown(result))
    print(f"\naccepted: {result['accepted']}")
    print(f"wrote {root / 'experiments_report.json'}")
    print(f"wrote {root / 'experiments_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
