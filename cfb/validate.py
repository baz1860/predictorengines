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

import numpy as np
import pandas as pd

import elo as E
import power as P
from ats_backtest import SPREADS_CSV, settle as ats_settle
from totals_backtest import TOTALS_CSV, settle as totals_settle

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "data", "validation_baseline.json")

THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 4.0)
BRIER_TOL = 0.005       # blend moneyline Brier may not regress beyond this
MAE_TOL = 0.50          # margin/total MAE may not regress beyond this (points)


def walk_forward(games: pd.DataFrame, since: int, quiet: bool = False) -> pd.DataFrame:
    """Per-game blended predictions for seasons >= since, games-indexed."""
    carry, offs = E.season_priors()
    _, history = E.run_elo(games, record_pregame=True, carry=carry, prior_offsets=offs)
    diffs = np.array([h[2] for h in history])
    pre = (games["season"] < since).values
    m_all = (games["home_points"] - games["away_points"]).values
    slope = float((diffs[pre] * m_all[pre]).sum() / (diffs[pre] ** 2).sum())

    ev = games[(games["season"] >= since)
               & (games["home"] != E.FCS) & (games["away"] != E.FCS)]
    rows, idx = [], []
    for (season, week, _stype), wk in ev.groupby(["season", "week", "season_type"],
                                                 sort=False):
        try:
            pparams = P.fit(games, asof=wk["date"].min())
        except ValueError:
            continue
        if not quiet:
            print(f"  fit week {season} w{week} ({len(wk)} games)", file=sys.stderr)
        for r in wk.itertuples():
            if r.home not in pparams["teams"] or r.away not in pparams["teams"]:
                continue
            d = diffs[r.Index]
            pp = P.predict(pparams, r.home, r.away, neutral=bool(r.neutral))
            rows.append({
                "season": int(r.season), "week": int(r.week),
                "home_team": r.home_team, "away_team": r.away_team,
                "p_elo": E.win_prob(d), "p_pow": pp["p1"],
                "m_elo": slope * d, "m_pow": pp["margin"], "t_pow": pp["total"],
                "margin": r.home_points - r.away_points,
                "total": r.home_points + r.away_points,
            })
            idx.append(r.Index)
    df = pd.DataFrame(rows, index=idx)
    if df.empty:
        return df
    df["p_blend"] = 0.5 * (df["p_elo"] + df["p_pow"])
    df["m_blend"] = 0.5 * (df["m_elo"] + df["m_pow"])
    return df


def _roi_by_threshold(df: pd.DataFrame, lines_csv: str, kind: str) -> tuple[dict, dict]:
    """ROI + bet count per threshold for ATS (kind='ats') or totals ('total')."""
    if not os.path.exists(lines_csv):
        return {}, {}
    lines = pd.read_csv(lines_csv)
    g = df.merge(lines, on=["season", "week", "home_team", "away_team"], how="inner")
    if g.empty:
        return {}, {}
    if kind == "ats":
        g["edge_pts"] = g["m_blend"] + g["home_line"]    # >0: model likes home
        settle = ats_settle
    else:
        g["edge_pts"] = g["t_pow"] - g["total_line"]     # >0: model says over
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
    }


def _load_baseline() -> dict | None:
    if os.path.exists(BASELINE):
        try:
            return json.loads(open(BASELINE).read())
        except Exception:
            return None
    return None


def _save_baseline(metrics: dict) -> None:
    os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
    with open(BASELINE, "w") as f:
        json.dump(metrics, f, indent=2)


def _print_metrics(m: dict) -> None:
    print(f"CFB validation · {m['window']} · {m['n_games']} games")
    print(f"  ml_brier   {m['ml_brier']:.4f}   (acc {m['ml_acc']:.1%})")
    print(f"  margin_mae {m['margin_mae']:.2f}")
    print(f"  total_mae  {m['total_mae']:.2f}")
    if m["ats_roi"]:
        print("  ATS ROI:    " + "  ".join(f"≥{k}:{v:+.1%}" for k, v in m["ats_roi"].items()))
    if m["totals_roi"]:
        print("  Totals ROI: " + "  ".join(f"≥{k}:{v:+.1%}" for k, v in m["totals_roi"].items()))


def gate(metrics: dict) -> int:
    """Compare to baseline. Returns process exit code (0 pass, 1 fail)."""
    base = _load_baseline()
    if base is None:
        _save_baseline(metrics)
        print("[gate] no baseline found — stored this run as baseline. PASS")
        return 0
    checks = [
        ("ml_brier", metrics["ml_brier"], base.get("ml_brier"), BRIER_TOL, "higher"),
        ("margin_mae", metrics["margin_mae"], base.get("margin_mae"), MAE_TOL, "higher"),
        ("total_mae", metrics["total_mae"], base.get("total_mae"), MAE_TOL, "higher"),
    ]
    print(f"\n{'metric':<12s}{'current':>10s}{'baseline':>10s}{'limit':>10s}  status")
    failed = False
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
    args = ap.parse_args()

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
