#!/usr/bin/env python3
"""Walk-forward validation for Club Soccer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import model as M

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BASELINE = DATA / "validation_baseline.json"
CALIB_FILE = DATA / "calibration.json"
CALIB_SPLIT = "2025-12-01"   # held-out boundary for the calibration acceptance test
GATE_TOL = 0.01
ENSEMBLE_SPLITS = ("2025-01-01", "2025-07-01", "2025-12-01")
ENSEMBLE_REGRESS_TOL = 0.0015


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "accuracy": 0.0, "brier": 0.0, "log_loss": 0.0}
    correct = 0
    brier = 0.0
    log_loss = 0.0
    for r in rows:
        probs = np.array([r["p_home"], r["p_draw"], r["p_away"]])
        actual = int(r["actual"])
        correct += int(probs.argmax() == actual)
        one = np.eye(3)[actual]
        brier += float(np.sum((probs - one) ** 2))
        log_loss += float(-np.log(max(1e-12, probs[actual])))
    n = len(rows)
    return {"n": n, "accuracy": correct / n, "brier": brier / n,
            "log_loss": log_loss / n}


def _metrics_arr(P: np.ndarray, A: np.ndarray) -> tuple[float, float, float]:
    n = len(A)
    acc = float((P.argmax(1) == A).mean())
    onehot = np.eye(3)[A]
    brier = float(((P - onehot) ** 2).sum(1).mean())
    ll = float((-np.log(np.clip(P[np.arange(n), A], 1e-12, 1.0))).mean())
    return acc, brier, ll


def walk_forward(min_train: int = 200, verbose: bool = False) -> tuple[list[dict], dict]:
    """Monthly-refit walk-forward: refit once per calendar month on all prior
    matches, then predict that month. O(months) fits, not O(matches) — required
    once fixtures.csv holds real (thousands of rows) data rather than the seed.
    """
    df = M.played(M.load_fixtures()).sort_values("date").reset_index(drop=True)
    df["_ym"] = df["date"].dt.to_period("M")
    months = sorted(df["_ym"].unique())
    rows: list[dict] = []
    skipped = 0
    for k, ym in enumerate(months, 1):
        test = df[df["_ym"] == ym]
        train = df[df["date"] < test["date"].min()]
        if len(train) < min_train:
            continue
        try:
            params = M.fit(train)
        except Exception:
            continue
        seen = set(params["teams"])
        kept = 0
        for r in test.itertuples(index=False):
            if r.home not in seen or r.away not in seen:
                skipped += 1
                continue
            try:
                pred = M.predict(r.home, r.away, r.competition, "ensemble",
                                 bool(r.neutral), params)
            except Exception:
                skipped += 1
                continue
            actual = 0 if r.home_goals > r.away_goals else (
                1 if r.home_goals == r.away_goals else 2)
            p = pred["probs"]
            rows.append({"date": str(r.date.date()), "home": r.home,
                         "away": r.away, "actual": actual,
                         "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"]})
            kept += 1
        if verbose:
            print(f"  [{k:>2}/{len(months)}] {ym}  tested {kept}")
    if verbose and skipped:
        print(f"  ({skipped} matches skipped — team unseen in its training window)")
    return rows, metrics(rows)


def component_walk_forward(min_train: int = 200, verbose: bool = False) -> pd.DataFrame:
    """Walk-forward component probabilities for ensemble tuning.

    No model is fit on or after the tested month. The returned frame is pure
    predictions + labels, so weight searches can be repeated without refitting.
    """
    df = M.played(M.load_fixtures()).sort_values("date").reset_index(drop=True)
    df["_ym"] = df["date"].dt.to_period("M")
    rows: list[dict] = []
    skipped = 0
    months = sorted(df["_ym"].unique())
    for k, ym in enumerate(months, 1):
        test = df[df["_ym"] == ym]
        train = df[df["date"] < test["date"].min()]
        if len(train) < min_train:
            continue
        try:
            params = M.fit(train)
        except Exception:
            continue
        seen = set(params["teams"])
        kept = 0
        for r in test.itertuples(index=False):
            if r.home not in seen or r.away not in seen:
                skipped += 1
                continue
            try:
                parts = M.component_matrices(params, r.home, r.away,
                                             r.competition, bool(r.neutral))
            except Exception:
                skipped += 1
                continue
            actual = 0 if r.home_goals > r.away_goals else (
                1 if r.home_goals == r.away_goals else 2)
            row = {"date": str(r.date.date()), "home": r.home, "away": r.away,
                   "actual": actual}
            for name, mat in parts.items():
                p = M.probs_from_matrix(mat)
                row[f"{name}_home"] = p["home"]
                row[f"{name}_draw"] = p["draw"]
                row[f"{name}_away"] = p["away"]
            rows.append(row)
            kept += 1
        if verbose:
            print(f"  [{k:>2}/{len(months)}] {ym}  components tested {kept}")
    if verbose and skipped:
        print(f"  ({skipped} component rows skipped)")
    return pd.DataFrame(rows)


def _component_array(df: pd.DataFrame, component: str) -> np.ndarray:
    return df[[f"{component}_home", f"{component}_draw", f"{component}_away"]].to_numpy(float)


def ensemble_probs(df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    w = M._normalise_weights(weights)
    P = np.zeros((len(df), 3), dtype=float)
    for comp, wt in w.items():
        if wt:
            P += wt * _component_array(df, comp)
    s = P.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return P / s


def score_ensemble(df: pd.DataFrame, weights: dict[str, float]) -> dict:
    A = df["actual"].to_numpy(int)
    acc, brier, ll = _metrics_arr(ensemble_probs(df, weights), A)
    return {"weights": {k: round(v, 3) for k, v in M._normalise_weights(weights).items()},
            "accuracy": round(acc, 5), "brier": round(brier, 6),
            "log_loss": round(ll, 6), "n": int(len(df))}


def choose_ensemble_weights(df: pd.DataFrame, grid_step: float = 0.05) -> dict:
    vals = [round(x, 2) for x in np.arange(0.0, 1.0 + 1e-9, grid_step)]
    best = None
    for wg in vals:
        for we in vals:
            if wg + we > 1.0 + 1e-9:
                continue
            for wx in vals:
                if wg + we + wx > 1.0 + 1e-9:
                    continue
                for wxf in vals:
                    wxp = round(1.0 - wg - we - wx - wxf, 2)
                    if wxp < -1e-9:
                        continue
                    weights = {"goals": wg, "elo": we, "xg": wx,
                               "xgf": wxf, "xpress": wxp}
                    row = score_ensemble(df, weights)
                    if best is None or (row["brier"], row["log_loss"]) < (best["brier"], best["log_loss"]):
                        best = row
    return best


def tune_ensemble(write: bool = False, verbose: bool = True) -> dict:
    df = component_walk_forward(verbose=verbose)
    if df.empty:
        raise SystemExit("No component walk-forward rows to tune.")
    current_w = M.load_ensemble_weights()
    current = score_ensemble(df, current_w)
    chosen = choose_ensemble_weights(df)
    split_results = []
    split_wins = 0
    max_regress = 0.0
    for split in ENSEMBLE_SPLITS:
        tr = df[pd.to_datetime(df["date"]) < pd.Timestamp(split)]
        te = df[pd.to_datetime(df["date"]) >= pd.Timestamp(split)]
        if len(tr) < 1000 or len(te) < 100:
            continue
        split_choice = choose_ensemble_weights(tr)
        cur_te = score_ensemble(te, current_w)
        ch_te = score_ensemble(te, split_choice["weights"])
        delta = ch_te["brier"] - cur_te["brier"]
        split_wins += int(delta < 0)
        max_regress = max(max_regress, delta)
        split_results.append({"split": split, "train_n": int(len(tr)), "test_n": int(len(te)),
                              "weights": split_choice["weights"],
                              "current_brier": cur_te["brier"],
                              "chosen_brier": ch_te["brier"],
                              "delta_brier": round(delta, 6),
                              "current_log_loss": cur_te["log_loss"],
                              "chosen_log_loss": ch_te["log_loss"]})
    promotes = (
        chosen["brier"] < current["brier"]
        and chosen["log_loss"] <= current["log_loss"]
        and split_wins >= 2
        and max_regress <= ENSEMBLE_REGRESS_TOL
    )
    out = {"current": current, "chosen": chosen, "splits": split_results,
           "promote": bool(promotes), "split_wins": split_wins,
           "max_split_regression": round(max_regress, 6)}
    print(f"\nClub ensemble tuning · {len(df)} predictions")
    print(f"  current Brier {current['brier']:.6f} log-loss {current['log_loss']:.6f} weights {current['weights']}")
    print(f"  chosen  Brier {chosen['brier']:.6f} log-loss {chosen['log_loss']:.6f} weights {chosen['weights']}")
    for r in split_results:
        print(f"  split {r['split']} n={r['test_n']}: ΔBrier {r['delta_brier']:+.6f} weights {r['weights']}")
    print(f"  verdict: {'PROMOTE' if promotes else 'reject'}")
    if write:
        if not promotes:
            print("  not writing ensemble_weights.json because the promotion gate failed")
        else:
            payload = {"weights": chosen["weights"], "source": "club_soccer/validate.py --tune-ensemble",
                       "metrics": {"previous_brier": current["brier"],
                                   "chosen_brier": chosen["brier"],
                                   "previous_log_loss": current["log_loss"],
                                   "chosen_log_loss": chosen["log_loss"]},
                       "splits": split_results}
            M.ENSEMBLE_WEIGHTS.write_text(json.dumps(payload, indent=2))
            BASELINE.write_text(json.dumps({"brier": chosen["brier"], "gate_tol": GATE_TOL}, indent=2))
            print(f"  wrote {M.ENSEMBLE_WEIGHTS}")
            print(f"  baseline updated -> {BASELINE}")
    return out


# ── Probability calibration: isotonic regression per outcome ─────────────────
def _pav(x, y):
    """Pool-adjacent-violators isotonic fit (no sklearn dependency).
    Returns (sorted x, monotone-nondecreasing fitted y)."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order].astype(float)
    w = np.ones_like(ys)
    vals, wts = list(ys), list(w)
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1]:
            new_w = wts[i] + wts[i + 1]
            new_v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / new_w
            vals[i:i + 2] = [new_v]
            wts[i:i + 2] = [new_w]
            if i > 0:
                i -= 1
        else:
            i += 1
    yhat = np.empty_like(ys)
    pos = 0
    for v, wt in zip(vals, wts):
        cnt = int(round(wt))
        yhat[pos:pos + cnt] = v
        pos += cnt
    return xs, yhat


def _knots(xs, yhat, max_knots=300):
    """Compact piecewise-linear knots from the isotonic step fit."""
    ux, uy = [], []
    for x, y in zip(xs, yhat):
        if ux and x == ux[-1]:
            uy[-1] = y
        else:
            ux.append(float(x))
            uy.append(float(y))
    if len(ux) > max_knots:
        idx = np.linspace(0, len(ux) - 1, max_knots).round().astype(int)
        ux = [ux[i] for i in idx]
        uy = [uy[i] for i in idx]
    return ux, uy


def _isotonic(x, y):
    """Isotonic fit via sklearn when installed, else the dependency-free PAV
    above — both compute the same pool-adjacent-violators solution."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return _pav(x, y)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ir = IsotonicRegression(out_of_bounds="clip")
    yhat = ir.fit_transform(xs, y[order])
    return xs, np.asarray(yhat, dtype=float)


def fit_calibration(P, A):
    """Per-outcome (H/D/A) isotonic maps from predictions P[n,3], labels A[n]."""
    maps = {}
    for k, side in enumerate(("home", "draw", "away")):
        xs, yhat = _isotonic(P[:, k], (A == k).astype(float))
        kx, ky = _knots(xs, yhat)
        maps[side] = {"x": kx, "y": ky}
    return maps


def apply_maps(P, maps):
    """Apply calibration maps to P[n,3], renormalised to sum 1."""
    out = np.empty_like(P, dtype=float)
    for k, side in enumerate(("home", "draw", "away")):
        m = maps[side]
        out[:, k] = np.interp(P[:, k], m["x"], m["y"])
    s = out.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return out / s


def _arrays_from_rows(rows):
    P = np.array([[r["p_home"], r["p_draw"], r["p_away"]] for r in rows], dtype=float)
    A = np.array([int(r["actual"]) for r in rows], dtype=int)
    dates = np.array([np.datetime64(r["date"]) for r in rows])
    return P, A, dates


def cmd_calibrate(verbose=True):
    rows, _ = walk_forward(verbose=verbose)
    if not rows:
        sys.exit("No walk-forward predictions to calibrate. Seed real fixtures first.")
    P, A, dates = _arrays_from_rows(rows)
    split = np.datetime64(CALIB_SPLIT)
    tr, te = dates < split, dates >= split
    print(f"\nCalibration (isotonic per outcome) on {len(A)} walk-forward predictions")
    if tr.sum() == 0 or te.sum() == 0:
        print(f"  Not enough data to split at {CALIB_SPLIT}; fitting on all and saving.")
    else:
        maps_tr = fit_calibration(P[tr], A[tr])
        P_cal = apply_maps(P[te], maps_tr)
        ar, br, lr = _metrics_arr(P[te], A[te])
        ac, bc, lc = _metrics_arr(P_cal, A[te])
        print(f"  Held-out test (fit < {CALIB_SPLIT}, test >=), n={int(te.sum())}:")
        print(f"    {'':12}{'accuracy':>10}{'Brier':>10}{'log-loss':>11}")
        print(f"    {'raw':12}{ar:>9.1%}{br:>10.4f}{lr:>11.4f}")
        print(f"    {'calibrated':12}{ac:>9.1%}{bc:>10.4f}{lc:>11.4f}")
        print(f"    Brier {'improved' if bc <= br else 'WORSE'} ({bc - br:+.4f}); "
              f"log-loss {'improved' if lc <= lr else 'WORSE'} ({lc - lr:+.4f})")
    maps_all = fit_calibration(P, A)     # production map: fit on all data
    CALIB_FILE.write_text(json.dumps(maps_all))
    print(f"\nSaved production calibration (fit on all {len(A)}) -> {CALIB_FILE.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="fit isotonic 1X2 calibration, report held-out improvement, "
                         "and write data/calibration.json")
    ap.add_argument("--tune-ensemble", action="store_true",
                    help="tune goals/elo/xg/xgf/xpress ensemble weights")
    ap.add_argument("--write", action="store_true",
                    help="with --tune-ensemble, write promoted weights and baseline")
    args = ap.parse_args()
    if args.tune_ensemble:
        tune_ensemble(write=args.write)
        return
    if args.calibrate:
        cmd_calibrate()
        return
    rows, m = walk_forward(verbose=True)
    print(f"Walk-forward Club Soccer validation (n={m['n']})")
    print(f"accuracy {m['accuracy']:.1%}  Brier {m['brier']:.4f}  log-loss {m['log_loss']:.4f}")
    DATA.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA / "validation_predictions.csv", index=False)
    if args.update_baseline or not BASELINE.exists():
        BASELINE.write_text(json.dumps({"brier": m["brier"], "gate_tol": GATE_TOL}, indent=2))
        print(f"Baseline written -> {BASELINE}")
    if args.gate:
        base = json.loads(BASELINE.read_text())
        limit = float(base["brier"]) + float(base.get("gate_tol", GATE_TOL))
        ok = m["brier"] <= limit
        print(f"[gate] Brier {m['brier']:.4f} vs baseline {base['brier']:.4f} "
              f"(limit {limit:.4f}) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
