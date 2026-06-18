#!/usr/bin/env python3
"""validate.py — walk-forward validation harness with regression gates (v2, M1).

This is the trustworthy yardstick the rest of v2 is measured against. It evaluates
the three match models (elo / dc / blend) out-of-sample, walking forward in time so
no future information leaks into any prediction:

  * Elo is already point-in-time: compute_elo records each match's PRE-match ratings
    (elo_h / elo_a), so scoring a match with those columns never uses its own result.
  * The Elo->goals map (fit_goal_model) and the Dixon-Coles attack/defence model are
    refit at each calendar-month boundary using only matches strictly before that
    month. DC fits are cached in data/validation_cache/ (slow first run, fast after).

Metrics per model: 3-way accuracy, Brier score, log-loss, and a 10-bin reliability
table (predicted vs observed frequency, pooled one-vs-rest over H/D/A).

Usage:
  python3 validate.py                 # full walk-forward table + reliability; writes
                                       #   data/validation_baseline.json on first run
  python3 validate.py --since 2024-01-01   # restrict the test window
  python3 validate.py --reliability   # also print per-model reliability tables
  python3 validate.py --gate          # CI gate: exit non-zero if blend Brier has
                                       #   regressed > 0.002 vs the stored baseline
  python3 validate.py --update-baseline    # overwrite the stored baseline on purpose

Notes:
  * Randomness is seeded (numpy default_rng(42)); the harness is otherwise
    deterministic given data/results.csv.
  * Caches never delete (the mounted data/ folder forbids unlink from the sandbox);
    stale month caches are simply ignored as new ones are written.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engines.worldcup.predictor import (load_matches, compute_elo, fit_goal_model,
                       expected_goals, HOME_ADV, DC_RHO)
from engines.worldcup.dixoncoles import fit_dc, DCModel, outcome_probs

HERE = Path(__file__).parent
CACHE = HERE / "data" / "validation_cache"
BASELINE = HERE / "data" / "validation_baseline.json"
CALIB_FILE = HERE / "data" / "calibration.json"
CALIB_SPLIT = "2025-12-01"   # held-out boundary for the calibration acceptance test

START = "2022-01-01"     # walk-forward test window start
GATE_TOL = 0.002         # blend Brier may not regress by more than this
SEED = 42
MODELS = ("elo", "dc", "blend")
MODEL_LABELS = {"elo": "Elo+Poisson", "dc": "Dixon-Coles", "blend": "50/50 blend"}


def _fit_month(played, cutoff, verbose=False):
    """DC model fit on matches strictly before `cutoff` (first day of a test month).

    Cached per month in data/validation_cache/. The anchor is the day before the
    cutoff so no match dated on the cutoff itself can leak into training; the decay
    weighting being one day early is immaterial."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"dc_{cutoff:%Y-%m}.json"
    if path.exists():
        return DCModel.load(path)
    anchor = cutoff - pd.Timedelta(days=1)
    dc = fit_dc(played, anchor=anchor, verbose=False)
    dc.save(path)
    if verbose:
        print(f"  fit DC for {cutoff:%Y-%m} ({len(dc.att)} teams) -> {path.name}")
    return dc


def walk_forward(since=START, verbose=True):
    """Return per-match predictions for every model plus the actual outcome.

    Output: dict model -> (probs ndarray [n,3]), actuals ndarray [n] in {0,1,2},
    and a parallel `dates` ndarray for sub-window slicing."""
    np.random.default_rng(SEED)  # seed all randomness (harness is deterministic)
    played, _ = load_matches()
    ratings, played = compute_elo(played)          # adds pre-match elo_h / elo_a
    since_ts = pd.Timestamp(since)
    test = played[played["date"] >= since_ts].copy()
    test["ym"] = test["date"].dt.to_period("M")
    months = sorted(test["ym"].unique())

    probs = {m: [] for m in MODELS}
    actuals, dates = [], []
    skipped = 0
    for k, ym in enumerate(months, 1):
        cutoff = ym.to_timestamp()                 # first day of the test month
        train = played[played["date"] < cutoff]
        beta = fit_goal_model(train)               # point-in-time Elo->goals map
        dc = _fit_month(played, cutoff, verbose=False)
        chunk = test[test["ym"] == ym]
        if verbose:
            print(f"  [{k:2d}/{len(months)}] {ym}  test={len(chunk):3d}", flush=True)
        for r in chunk.itertuples(index=False):
            if r.home_team not in dc.att or r.away_team not in dc.att:
                skipped += 1
                continue
            h = 0.0 if r.neutral else 1.0
            a = 0 if r.home_score > r.away_score else (
                1 if r.home_score == r.away_score else 2)
            le1, le2 = expected_goals(r.elo_h, r.elo_a, beta, h * HOME_ADV)
            ld1, ld2 = dc.lambdas(r.home_team, r.away_team, h1=h)
            pe = np.array(outcome_probs(le1, le2, DC_RHO)[:3])
            pdc = np.array(outcome_probs(ld1, ld2, dc.rho)[:3])
            probs["elo"].append(pe)
            probs["dc"].append(pdc)
            probs["blend"].append((pe + pdc) / 2)
            actuals.append(a)
            dates.append(r.date)
    if verbose and skipped:
        print(f"  ({skipped} matches skipped — team unseen in its training window)")
    out = {m: np.array(probs[m]) for m in MODELS}
    return out, np.array(actuals), pd.to_datetime(pd.Series(dates))


def metrics(P, A):
    """3-way accuracy, Brier, log-loss for predictions P[n,3] vs actuals A[n]."""
    onehot = np.eye(3)[A]
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
    acc = float(np.mean(P.argmax(1) == A))
    p_act = np.clip(P[np.arange(len(A)), A], 1e-15, 1.0)
    logloss = float(-np.mean(np.log(p_act)))
    return acc, brier, logloss


def reliability(P, A, bins=10):
    """Pooled one-vs-rest reliability over H/D/A. Returns rows of
    (lo, hi, count, mean_predicted, observed_frequency)."""
    p = P.ravel()
    occ = np.eye(3)[A].ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for k in range(bins):
        lo, hi = edges[k], edges[k + 1]
        m = (p >= lo) & (p <= hi) if k == bins - 1 else (p >= lo) & (p < hi)
        n = int(m.sum())
        if n == 0:
            rows.append((lo, hi, 0, float("nan"), float("nan")))
        else:
            rows.append((lo, hi, n, float(p[m].mean()), float(occ[m].mean())))
    return rows


# ── Probability calibration (M2): isotonic regression per outcome ────────────
def _pav(x, y):
    """Pool-adjacent-violators isotonic fit (no sklearn dependency).
    Returns (sorted x, monotone-nondecreasing fitted y)."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order].astype(float)
    blocks = []                       # each: [sum, count, mean]
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
    """Compact piecewise-linear knots from the isotonic step fit."""
    ux, uy = [], []
    for x, y in zip(xs, yhat):
        if ux and x == ux[-1]:
            uy[-1] = float(y)         # collapse duplicate x (stay monotone)
        else:
            ux.append(float(x))
            uy.append(float(y))
    if len(ux) > max_knots:
        idx = sorted(set(np.linspace(0, len(ux) - 1, max_knots)
                         .round().astype(int).tolist()))
        ux, uy = [ux[i] for i in idx], [uy[i] for i in idx]
    return ux, uy


def _isotonic(x, y):
    """Isotonic fit returning (sorted x, fitted y). Uses sklearn's
    IsotonicRegression when installed (as on Barrie's machine), else the
    dependency-free PAV above — both compute the same pool-adjacent-violators
    solution, so the daily pipeline never hard-depends on sklearn."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return _pav(x, y)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    yhat = ir.fit_transform(xs, y[order])
    return xs, np.asarray(yhat, dtype=float)


def fit_calibration(probs, A):
    """Per-outcome (H/D/A) isotonic maps from blend predictions probs[n,3]."""
    maps = {}
    for k, side in enumerate(("home", "draw", "away")):
        xs, yhat = _isotonic(probs[:, k], (A == k).astype(float))
        kx, ky = _knots(xs, yhat)
        maps[side] = {"x": kx, "y": ky}
    return maps


def apply_maps(probs, maps):
    """Apply calibration maps to probs[n,3], renormalised to sum 1."""
    out = np.empty_like(probs, dtype=float)
    for k, side in enumerate(("home", "draw", "away")):
        m = maps[side]
        out[:, k] = np.interp(probs[:, k], m["x"], m["y"])
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def _print_reliability_one(P, A, label):
    print(f"\n  Reliability — {label} (pooled H/D/A, |pred-obs| lower is better)")
    print(f"    {'bin':<12}{'n':>7}{'pred':>9}{'obs':>9}{'gap':>9}")
    for lo, hi, n, pred, obs in reliability(P, A):
        if n == 0:
            print(f"    [{lo:.1f},{hi:.1f}){'':>7}{'—':>9}")
        else:
            print(f"    [{lo:.1f},{hi:.1f}){n:>7d}{pred:>9.3f}{obs:>9.3f}"
                  f"{pred - obs:>+9.3f}")


def cmd_calibrate(since, quiet):
    out, A, dates = walk_forward(since=since, verbose=not quiet)
    P = out["blend"]
    split = pd.Timestamp(CALIB_SPLIT)
    tr = (dates < split).to_numpy()
    te = ~tr
    print(f"\nCalibration (isotonic per outcome) — blend predictions since {since}")
    if te.sum() == 0 or tr.sum() == 0:
        print(f"  Not enough data to split at {CALIB_SPLIT}; fitting on all and saving.")
    else:
        maps_tr = fit_calibration(P[tr], A[tr])
        P_cal = apply_maps(P[te], maps_tr)
        ar, brr, llr = metrics(P[te], A[te])
        ac, brc, llc = metrics(P_cal, A[te])
        print(f"  Held-out test (fit < {CALIB_SPLIT}, test >=), n={int(te.sum())}:")
        print(f"    {'':10}{'accuracy':>10}{'Brier':>10}{'log-loss':>11}")
        print(f"    {'raw blend':10}{ar:>9.1%}{brr:>10.4f}{llr:>11.4f}")
        print(f"    {'calibrated':10}{ac:>9.1%}{brc:>10.4f}{llc:>11.4f}")
        print(f"    Brier {'improved' if brc <= brr else 'WORSE'} "
              f"({brc - brr:+.4f}); log-loss "
              f"{'improved' if llc <= llr else 'WORSE'} ({llc - llr:+.4f})")
        _print_reliability_one(P[te], A[te], "RAW blend (held-out)")
        _print_reliability_one(P_cal, A[te], "CALIBRATED blend (held-out)")

    maps_all = fit_calibration(P, A)        # production map: fit on all data
    CALIB_FILE.write_text(json.dumps(maps_all))
    print(f"\nSaved production calibration (fit on all {len(A)} predictions) -> "
          f"{CALIB_FILE.name}")


def print_metrics_table(out, A, title):
    print(f"\n{title}  (n={len(A)})\n")
    print(f"{'model':<14}{'accuracy':>10}{'Brier':>10}{'log-loss':>11}")
    res = {}
    for m in MODELS:
        acc, br, ll = metrics(out[m], A)
        res[m] = (acc, br, ll)
        print(f"{MODEL_LABELS[m]:<14}{acc:>9.1%}{br:>10.4f}{ll:>11.4f}")
    return res


def print_reliability(out, A):
    for m in MODELS:
        print(f"\nReliability — {MODEL_LABELS[m]} (pooled H/D/A, lower |pred-obs| better)")
        print(f"  {'bin':<12}{'n':>7}{'pred':>9}{'obs':>9}{'gap':>9}")
        for lo, hi, n, pred, obs in reliability(out[m], A):
            if n == 0:
                print(f"  [{lo:.1f},{hi:.1f}){'':>7}{'—':>9}")
                continue
            print(f"  [{lo:.1f},{hi:.1f}){n:>7d}{pred:>9.3f}{obs:>9.3f}{pred - obs:>+9.3f}")


def load_baseline():
    return json.loads(BASELINE.read_text()) if BASELINE.exists() else None


def write_baseline(res, n, since):
    payload = {
        "since": since,
        "n": int(n),
        "blend_brier": res["blend"][1],
        "blend_logloss": res["blend"][2],
        "blend_accuracy": res["blend"][0],
        "all": {m: {"accuracy": res[m][0], "brier": res[m][1],
                    "logloss": res[m][2]} for m in MODELS},
        "gate_tol": GATE_TOL,
    }
    BASELINE.write_text(json.dumps(payload, indent=2))
    return payload


def main():
    ap = argparse.ArgumentParser(description="Walk-forward validation harness (v2 M1)")
    ap.add_argument("--since", default=START, help=f"test window start (default {START})")
    ap.add_argument("--reliability", action="store_true",
                    help="also print per-model reliability tables")
    ap.add_argument("--gate", action="store_true",
                    help="exit non-zero if blend Brier regressed > %.3f vs baseline"
                         % GATE_TOL)
    ap.add_argument("--update-baseline", action="store_true",
                    help="overwrite the stored baseline with this run")
    ap.add_argument("--quiet", action="store_true", help="suppress progress lines")
    ap.add_argument("--calibrate", action="store_true",
                    help="fit isotonic calibration on walk-forward blend predictions, "
                         "report held-out improvement, and write data/calibration.json")
    args = ap.parse_args()

    if args.calibrate:
        cmd_calibrate(args.since, args.quiet)
        return

    out, A, dates = walk_forward(since=args.since, verbose=not args.quiet)
    if len(A) == 0:
        sys.exit("No test matches in the requested window.")

    res = print_metrics_table(out, A, f"Walk-forward, matches since {args.since}")

    # Reference subset: matches since 2024-01-01 line up with the v1 backtest number.
    if pd.Timestamp(args.since) < pd.Timestamp("2024-01-01"):
        mask = (dates >= pd.Timestamp("2024-01-01")).to_numpy()
        if mask.any():
            sub = {m: out[m][mask] for m in MODELS}
            print_metrics_table(sub, A[mask],
                                "Sub-window since 2024-01-01 (v1 reference: "
                                "blend 60.4% / 0.5038)")

    if args.reliability or not args.gate:
        print_reliability(out, A)

    base = load_baseline()
    if args.gate:
        if base is None:
            write_baseline(res, len(A), args.since)
            print(f"\n[gate] no baseline found — stored this run as baseline. PASS")
        else:
            cur = res["blend"][1]
            limit = base["blend_brier"] + base.get("gate_tol", GATE_TOL)
            status = "PASS" if cur <= limit else "FAIL"
            print(f"\n[gate] blend Brier {cur:.4f} vs baseline {base['blend_brier']:.4f} "
                  f"(limit {limit:.4f}) -> {status}")
            if status == "FAIL":
                sys.exit(1)
    elif args.update_baseline or base is None:
        p = write_baseline(res, len(A), args.since)
        action = "updated" if base is not None else "wrote"
        print(f"\nBaseline {action} -> {BASELINE.name} "
              f"(blend Brier {p['blend_brier']:.4f})")
    else:
        print(f"\nBaseline unchanged ({BASELINE.name}, blend Brier "
              f"{base['blend_brier']:.4f}). Use --update-baseline to overwrite.")


if __name__ == "__main__":
    main()
