"""
golf/calibrate.py  –  Per-market probability calibration.

The independent-round Monte Carlo is systematically miscalibrated (the
make-cut backtest shows pred 0.35 → actual ~0.50). This fits an isotonic map
per market (win, top5, top10, top20, cut) from validate.py's walk-forward
predictions and applies them as a pure consumer in the edge/sim path.

Same isotonic machinery and JSON shape as the root calibrate.py / validate.py,
plus a nesting guard so a calibrated row keeps win ≤ top5 ≤ top10 ≤ top20 ≤ cut.

  python -m golf.calibrate --fit         # refit from data/validation_predictions.csv
  from calibrate import apply_row    # consumer
  probs = apply_row({"win":..,"top5":..,"top10":..,"top20":..,"cut":..})
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
PRED_CSV = DATA_DIR / "validation_predictions.csv"
CALIB_FILE = DATA_DIR / "calibration.json"

MARKETS = ["win", "top5", "top10", "top20", "cut"]   # nested, widest last


# ── isotonic helpers (mirror of root validate.py) ──
def _pav(x, y):
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order].astype(float)
    blocks = []
    for v in ys:
        blocks.append([v, 1.0, v])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2] + 1e-15:
            s2, c2, _ = blocks.pop()
            s1, c1, _ = blocks.pop()
            s, c = s1 + s2, c1 + c2
            blocks.append([s, c, s / c])
    yhat = np.empty(len(ys))
    i = 0
    for s, c, m in blocks:
        c = int(round(c))
        yhat[i:i + c] = m
        i += c
    return xs, yhat


def _knots(xs, yhat, max_knots=300):
    ux, uy = [], []
    for x, y in zip(xs, yhat):
        if ux and x == ux[-1]:
            uy[-1] = float(y)
        else:
            ux.append(float(x))
            uy.append(float(y))
    if len(ux) > max_knots:
        idx = sorted(set(np.linspace(0, len(ux) - 1, max_knots)
                         .round().astype(int).tolist()))
        ux, uy = [ux[i] for i in idx], [uy[i] for i in idx]
    return ux, uy


def _isotonic(x, y):
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return _pav(x, y)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    yhat = ir.fit_transform(xs, y[order])
    return xs, np.asarray(yhat, dtype=float)


# ── fit ──
def fit_maps(pred_df) -> dict:
    """Per-market isotonic maps from walk-forward predictions."""
    maps = {}
    for mkt in MARKETS:
        x = pred_df[f"p_{mkt}"].values.astype(float)
        y = pred_df[f"y_{mkt}"].values.astype(float)
        xs, yhat = _isotonic(x, y)
        kx, ky = _knots(xs, yhat)
        maps[mkt] = {"x": kx, "y": ky}
    return maps


def fit_from_csv(path: Path | None = None) -> dict:
    import pandas as pd
    path = path or PRED_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"No {path}. Run: python -m golf.validate  first.")
    maps = fit_maps(pd.read_csv(path))
    CALIB_FILE.write_text(json.dumps(maps, indent=1))
    return maps


# ── apply (consumer) ──
def load_maps():
    if CALIB_FILE.exists():
        return json.loads(CALIB_FILE.read_text())
    return None


def apply_one(market: str, p: float, maps=None) -> float:
    if maps is None:
        maps = load_maps()
    if maps is None or market not in maps:
        return p
    m = maps[market]
    return float(np.interp(p, m["x"], m["y"]))


def apply_row(probs: dict, maps=None) -> dict:
    """Calibrate a player's market probabilities and enforce nesting
    win ≤ top5 ≤ top10 ≤ top20 ≤ cut (each finish band contains the tighter)."""
    if maps is None:
        maps = load_maps()
    if maps is None:
        return dict(probs)
    out = {m: apply_one(m, float(probs.get(m, 0.0)), maps) for m in MARKETS
           if m in probs}
    # enforce monotone nesting from widest (cut) inward
    prev = 1.0
    for m in reversed(MARKETS):
        if m in out:
            out[m] = min(out[m], prev)
            prev = out[m]
    return {**probs, **out}


def _report(pred_df, maps, folds: int = 5, seed: int = 0) -> None:
    """Diagnostic: out-of-sample Brier improvement from calibration.

    The production maps are (correctly) fit on all available data, but scoring
    them on that same data overstates the gain. Here we report an honest, grouped
    K-fold estimate: hold out whole tournaments, fit the isotonic map on the
    rest, and score the held-out fold. This is what the calibration is expected
    to deliver on unseen events.
    """
    from .validate import brier  # reuse metric

    # group by tournament so a fold never splits one event across train/test
    if "tournament_id" in pred_df.columns:
        groups = pred_df["tournament_id"].values
    else:
        groups = np.arange(len(pred_df))
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % folds for i, g in enumerate(uniq)}
    test_fold = np.array([fold_of[g] for g in groups])

    print(f"\n{'Market':<8}{'Brier raw':>11}{'Brier cal':>11}{'Δ':>9}"
          f"   (in-sample → {folds}-fold OOS)")
    print("-" * 56)
    for mkt in MARKETS:
        p = pred_df[f"p_{mkt}"].values.astype(float)
        y = pred_df[f"y_{mkt}"].values.astype(float)
        # in-sample (uses production maps)
        pc_in = np.interp(p, maps[mkt]["x"], maps[mkt]["y"])
        # out-of-sample: fit on the other folds, predict the held-out fold
        pc_oos = p.copy()
        for k in range(folds):
            tr, te = test_fold != k, test_fold == k
            if te.sum() == 0 or tr.sum() == 0:
                continue
            xs, yhat = _isotonic(p[tr], y[tr])
            pc_oos[te] = np.interp(p[te], xs, yhat)
        br = brier(p, y)
        bc_in, bc_oos = brier(pc_in, y), brier(pc_oos, y)
        print(f"{mkt:<8}{br:>11.5f}{bc_in:>11.5f}{bc_in - br:>+9.5f}"
              f"   (OOS {bc_oos:.5f}, Δ {bc_oos - br:+.5f})")


if __name__ == "__main__":
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser(description="Fit/inspect golf calibration")
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()

    if args.fit:
        maps = fit_from_csv()
        print(f"Fitted calibration for {list(maps)} → {CALIB_FILE}")
        _report(pd.read_csv(PRED_CSV), maps)
    else:
        m = load_maps()
        if not m:
            print("No calibration yet. Fit it: python -m golf.calibrate --fit")
        else:
            print(f"Calibration loaded; markets: {list(m)}")
            for mkt in MARKETS:
                print(f"  {mkt}: {len(m[mkt]['x'])} knots")
