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
from predictor import load_blend_weight, DEFAULT_W_ELO, _BLEND_WEIGHT_FILE
from ats_backtest import SPREADS_CSV, settle as ats_settle
from totals_backtest import TOTALS_CSV, settle as totals_settle

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "data", "validation_baseline.json")

THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 4.0)
BRIER_TOL = 0.005       # blend moneyline Brier may not regress beyond this
MAE_TOL = 0.50          # margin/total MAE may not regress beyond this (points)


def walk_forward(games: pd.DataFrame, since: int, quiet: bool = False,
                 w_elo: float | None = None) -> pd.DataFrame:
    """Per-game blended predictions for seasons >= since, games-indexed.

    `w_elo` is the weight on Elo in the win-prob/margin blend; None loads the
    stored weight (default 0.5). Raw `p_elo`/`p_pow`/`m_elo`/`m_pow` are always
    kept so the tuner can rescore any weight without re-running the walk."""
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
    w = load_blend_weight() if w_elo is None else float(w_elo)
    df["p_blend"] = w * df["p_elo"] + (1.0 - w) * df["p_pow"]
    df["m_blend"] = w * df["m_elo"] + (1.0 - w) * df["m_pow"]
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
    ap.add_argument("--tune-blend", action="store_true",
                    help="show elo/power blend-weight before/after table (M6)")
    ap.add_argument("--write", action="store_true",
                    help="with --tune-blend, opt into the chosen weight")
    args = ap.parse_args()

    if args.tune_blend:
        tune_blend(args.since, write=args.write, quiet=True)
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
