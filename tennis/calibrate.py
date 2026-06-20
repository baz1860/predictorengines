"""tennis/calibrate.py — per-market probability calibration.

Fits an isotonic map per market from validate.py's walk-forward predictions and
applies them as a pure consumer in the predict/edge path. Same isotonic
machinery and JSON shape as golf/calibrate.py.

Markets fall in two groups:
  * match markets — match_winner, set_hcp, first_set — each calibrated
    independently (binary, scored straight off completed matches);
  * outright markets — win, final, sf, qf — calibrated with a nesting guard so a
    calibrated player keeps win ≤ final ≤ sf ≤ qf.

Only the markets actually present as p_<m>/y_<m> columns in the predictions CSV
are fitted, so this works before the outright (draw-sim) backtest is wired.

  python -m tennis.calibrate --fit          # refit from validation_predictions.csv
  from tennis import calibrate as C
  p_cal = C.apply_one("match_winner", p)     # consumer
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
PRED_CSV = DATA_DIR / "validation_predictions.csv"
CALIB_FILE = DATA_DIR / "calibration.json"

MATCH_MARKETS = ["match_winner", "set_hcp", "first_set"]
OUTRIGHT_MARKETS = ["win", "final", "sf", "qf"]   # nested, widest (qf) last
ALL_MARKETS = MATCH_MARKETS + OUTRIGHT_MARKETS


# ── isotonic helpers (mirror of golf/calibrate.py) ──
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
    """Per-market isotonic maps for whichever markets are present."""
    maps = {}
    for mkt in ALL_MARKETS:
        pcol, ycol = f"p_{mkt}", f"y_{mkt}"
        if pcol not in pred_df.columns or ycol not in pred_df.columns:
            continue
        sub = pred_df[[pcol, ycol]].dropna()
        if len(sub) < 50:
            continue
        x = sub[pcol].to_numpy(dtype=float)
        y = sub[ycol].to_numpy(dtype=float)
        xs, yhat = _isotonic(x, y)
        kx, ky = _knots(xs, yhat)
        maps[mkt] = {"x": kx, "y": ky}
    return maps


def fit_from_csv(path: Path | None = None) -> dict:
    import pandas as pd
    path = path or PRED_CSV
    if not path.exists():
        raise FileNotFoundError(f"No {path}. Run: python -m tennis.validate first.")
    maps = fit_maps(pd.read_csv(path))
    if not maps:
        raise ValueError("No calibratable markets in predictions CSV.")
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


def apply_match(probs: dict, maps=None) -> dict:
    """Calibrate independent match-market probabilities (no nesting)."""
    if maps is None:
        maps = load_maps()
    if maps is None:
        return dict(probs)
    out = dict(probs)
    for m in MATCH_MARKETS:
        if m in probs:
            out[m] = apply_one(m, float(probs[m]), maps)
    return out


def apply_outright(probs: dict, maps=None) -> dict:
    """Calibrate outright probabilities and enforce win ≤ final ≤ sf ≤ qf."""
    if maps is None:
        maps = load_maps()
    if maps is None:
        return dict(probs)
    out = {m: apply_one(m, float(probs.get(m, 0.0)), maps)
           for m in OUTRIGHT_MARKETS if m in probs}
    prev = 1.0
    for m in reversed(OUTRIGHT_MARKETS):   # widest (qf) inward
        if m in out:
            out[m] = min(out[m], prev)
            prev = out[m]
    return {**probs, **out}


def _report(pred_df, maps, folds: int = 5, seed: int = 0) -> None:
    """Honest out-of-sample Brier improvement: grouped K-fold over events so the
    map is never scored on rows it was fitted on."""
    from .validate import brier

    rng = np.random.default_rng(seed)
    has_group = "tourney_id" in pred_df.columns

    print(f"\n{'Market':<14}{'Brier raw':>11}{'Brier cal':>11}{'Δ':>9}"
          f"   (in-sample → {folds}-fold OOS)")
    print("-" * 60)
    for mkt in ALL_MARKETS:
        if mkt not in maps:
            continue
        cols = [f"p_{mkt}", f"y_{mkt}"] + (["tourney_id"] if has_group else [])
        sub = pred_df[cols].dropna(subset=[f"p_{mkt}", f"y_{mkt}"])
        if len(sub) < folds:
            continue
        p = sub[f"p_{mkt}"].to_numpy(dtype=float)
        y = sub[f"y_{mkt}"].to_numpy(dtype=float)
        groups = sub["tourney_id"].to_numpy() if has_group else np.arange(len(sub))
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        fold_of = {g: i % folds for i, g in enumerate(uniq)}
        test_fold = np.array([fold_of[g] for g in groups])
        pc_in = np.interp(p, maps[mkt]["x"], maps[mkt]["y"])
        pc_oos = p.copy()
        for k in range(folds):
            tr, te = test_fold != k, test_fold == k
            if te.sum() == 0 or tr.sum() == 0:
                continue
            xs, yhat = _isotonic(p[tr], y[tr])
            pc_oos[te] = np.interp(p[te], xs, yhat)
        br, bc_in, bc_oos = brier(p, y), brier(pc_in, y), brier(pc_oos, y)
        print(f"{mkt:<14}{br:>11.5f}{bc_in:>11.5f}{bc_in - br:>+9.5f}"
              f"   (OOS {bc_oos:.5f}, Δ {bc_oos - br:+.5f})")


if __name__ == "__main__":
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser(description="Fit/inspect tennis calibration")
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()

    if args.fit:
        maps = fit_from_csv()
        print(f"Fitted calibration for {list(maps)} → {CALIB_FILE}")
        _report(pd.read_csv(PRED_CSV), maps)
    else:
        m = load_maps()
        if not m:
            print("No calibration yet. Fit it: python -m tennis.calibrate --fit")
        else:
            print(f"Calibration loaded; markets: {list(m)}")
            for mkt, mp in m.items():
                print(f"  {mkt}: {len(mp['x'])} knots")
