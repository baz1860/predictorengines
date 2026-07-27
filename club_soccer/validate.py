#!/usr/bin/env python3
"""Walk-forward validation for Club Soccer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import model as M
from . import calibrate as CAL
from . import walkforward_cache as WFC

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
# ── baseline ownership ────────────────────────────────────────────────────
# Two files, two owners. Validation is DESCRIPTIVE: it reports where the model
# currently stands and may freely rewrite validation_latest.json. It must never
# move the promotion gate — a reseed that silently re-baselines can hide a
# regression by redefining "normal". promotion_baseline.json is the gate
# reference and is owned exclusively by the nested-holdout promoter (build 3).
BASELINE = DATA / "validation_baseline.json"          # legacy, read-only fallback
PROMOTION_BASELINE = DATA / "promotion_baseline.json"  # promoter-owned gate
LATEST = DATA / "validation_latest.json"               # validation-owned, descriptive
GATE_STATE = DATA / "validation_gate_state.json"       # derived exact-input cache
CALIB_FILE = DATA / "calibration.json"
OPPONENT_XG_EVIDENCE = DATA / "opponent_adjusted_xg_evidence.json"
CALIB_SPLIT = "2025-12-01"   # held-out boundary for the calibration acceptance test
CALIBRATION_SPLITS = ("2025-01-01", "2025-07-01", "2025-12-01")
GATE_TOLERANCES = {
    "brier": 0.0010,
    "log_loss": 0.0015,
    "brier_ou25": 0.0010,
    "brier_btts": 0.0010,
}
ENSEMBLE_SPLITS = ("2025-01-01", "2025-07-01", "2025-12-01")
ENSEMBLE_REGRESS_TOL = 0.0015

CLUBELO_CACHE = DATA / "clubelo_cache"
CLUBELO_URL = "http://api.clubelo.com/{date}"
CLUBELO_HOME_ADV = 65.0          # ClubElo's own published home-advantage constant
CLUBELO_WINDOW_DAYS = 60         # a single Elo snapshot is only meaningful near its date


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "accuracy": 0.0, "brier": 0.0, "log_loss": 0.0,
                "brier_ou25": 0.0, "brier_btts": 0.0}
    correct = 0
    brier = 0.0
    log_loss = 0.0
    brier_ou25 = 0.0
    brier_btts = 0.0
    for r in rows:
        probs = np.array([r["p_home"], r["p_draw"], r["p_away"]])
        actual = int(r["actual"])
        correct += int(probs.argmax() == actual)
        one = np.eye(3)[actual]
        brier += float(np.sum((probs - one) ** 2))
        log_loss += float(-np.log(max(1e-12, probs[actual])))
        if "p_over25" in r and "total_goals" in r:
            y_over = 1.0 if float(r["total_goals"]) > 2.5 else 0.0
            brier_ou25 += (float(r["p_over25"]) - y_over) ** 2
        if "p_btts" in r and "btts_actual" in r:
            brier_btts += (float(r["p_btts"]) - float(r["btts_actual"])) ** 2
    n = len(rows)
    return {"n": n, "accuracy": correct / n, "brier": brier / n,
            "log_loss": log_loss / n,
            "brier_ou25": brier_ou25 / n, "brier_btts": brier_btts / n}


def metrics_by_group(rows: list[dict], key: str = "competition") -> dict[str, dict]:
    """Return the same scoring metrics split by a prediction-row field."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        value = row.get(key)
        try:
            missing = value is None or bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = value is None
        group = "missing" if missing or not str(value).strip() else str(value)
        groups.setdefault(group, []).append(row)
    return {group: metrics(group_rows) for group, group_rows in sorted(groups.items())}


def _metrics_arr(P: np.ndarray, A: np.ndarray) -> tuple[float, float, float]:
    n = len(A)
    acc = float((P.argmax(1) == A).mean())
    onehot = np.eye(3)[A]
    brier = float(((P - onehot) ** 2).sum(1).mean())
    ll = float((-np.log(np.clip(P[np.arange(n), A], 1e-12, 1.0))).mean())
    return acc, brier, ll


def walk_forward(min_train: int = 200, verbose: bool = False,
                 opponent_adjusted_xg: bool | None = None,
                 fixtures: pd.DataFrame | None = None,
                 test_from: str | None = None,
                 test_to: str | None = None,
                 use_cache: bool = True,
                 league_seed: bool | None = None) -> tuple[list[dict], dict]:
    """Monthly-refit walk-forward: refit once per calendar month on all prior
    matches, then predict that month. O(months) fits, not O(matches) — required
    once fixtures.csv holds real (thousands of rows) data rather than the seed.

    Folds are cached (walkforward_cache.py). A fold trains only on data before
    its test month, so on a typical day almost every fold's inputs are
    unchanged and its result is reloaded rather than recomputed — the metric is
    identical, not approximated. Pass use_cache=False to force a full recompute.
    """
    df = M.played(M.load_fixtures() if fixtures is None else fixtures).sort_values("date").reset_index(drop=True)
    df["_ym"] = df["date"].dt.to_period("M")
    months = sorted(df["_ym"].unique())
    first_test = pd.Timestamp(test_from).to_period("M") if test_from else None
    last_test = pd.Timestamp(test_to).to_period("M") if test_to else None
    rows: list[dict] = []
    skipped = 0
    # EVERY fit option must appear here. An option absent from the cache key
    # means the cache serves results produced under different settings — a
    # silent wrong answer, not a slow one.
    # Defaults to the PRODUCTION setting (model.LEAGUE_SEED_DEFAULT), so the
    # gate measures the model that actually runs. Resolved to a concrete bool
    # before it reaches the cache key — caching under `None` would collide two
    # different models under one entry the moment the production default moved.
    league_seed = M.LEAGUE_SEED_DEFAULT if league_seed is None else bool(league_seed)
    opponent_adjusted_xg = (
        M.OPPONENT_ADJUSTED_XG_DEFAULT
        if opponent_adjusted_xg is None else bool(opponent_adjusted_xg)
    )
    cache_opts = {"min_train": min_train,
                  "opponent_adjusted_xg": opponent_adjusted_xg,
                  "league_seed": league_seed}
    row_hash = WFC.row_hashes(df) if use_cache else None
    hits = misses = 0
    seen_keys: set[tuple[str, str]] = set()
    for k, ym in enumerate(months, 1):
        if first_test is not None and ym < first_test:
            continue
        if last_test is not None and ym >= last_test:
            continue
        test = df[df["_ym"] == ym]
        train = df[df["date"] < test["date"].min()]
        if len(train) < min_train:
            continue

        cache_key = None
        if use_cache:
            cache_key = WFC.fold_key(str(ym), row_hash[train.index.to_numpy()],
                                     row_hash[test.index.to_numpy()], cache_opts)
            seen_keys.add((str(ym), cache_key))
            cached = WFC.load(str(ym), cache_key)
            if cached is not None:
                rows.extend(cached)
                hits += 1
                if verbose:
                    print(f"  [{k:>2}/{len(months)}] {ym}  cached ({len(cached)})")
                continue
        misses += 1

        try:
            # Seed from coefficients that existed at this fold's cutoff, not
            # today's — the walk-forward must not leak future UEFA rankings.
            cutoff = str(test["date"].min())[:10]
            params = M.fit(train, opponent_adjusted_xg=opponent_adjusted_xg,
                           league_seed=league_seed, coef_as_of=cutoff)
        except Exception:
            continue
        fold_rows: list[dict] = []
        seen = set(params["teams"])
        kept = 0
        for r in test.itertuples(index=False):
            if r.home not in seen or r.away not in seen:
                skipped += 1
                continue
            try:
                # Validate the same date-aware path used by the card and edge
                # layers.
                pred = M.predict_match(
                    r.home, r.away, r.competition, str(r.date.date()), "ensemble",
                    bool(r.neutral), params=params,
                    fixture_id=getattr(r, "fixture_id", None),
                    ensemble_weights=dict(M.DEFAULT_ENSEMBLE_W),
                )
            except Exception:
                skipped += 1
                continue
            actual = 0 if r.home_goals > r.away_goals else (
                1 if r.home_goals == r.away_goals else 2)
            p = pred["probs"]
            total_goals = float(r.home_goals) + float(r.away_goals)
            btts_actual = 1.0 if (r.home_goals > 0 and r.away_goals > 0) else 0.0
            fold_rows.append({"date": str(r.date.date()), "home": r.home,
                              "away": r.away, "competition": r.competition,
                              "type": r.type,
                              "fixture_id": getattr(r, "fixture_id", None),
                              "xg_source": getattr(r, "xg_source", ""),
                              "actual": actual,
                              "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"],
                              "p_over25": p["over25"], "p_btts": p["btts_yes"],
                              "total_goals": total_goals, "btts_actual": btts_actual})
            kept += 1
        rows.extend(fold_rows)
        if use_cache and cache_key is not None:
            WFC.store(str(ym), cache_key, fold_rows)
        if verbose:
            print(f"  [{k:>2}/{len(months)}] {ym}  tested {kept}")
    if verbose and skipped:
        print(f"  ({skipped} matches skipped — team unseen in its training window)")
    if use_cache:
        WFC.prune(seen_keys)
        if verbose:
            print(f"  (folds: {hits} cached, {misses} recomputed)")
    return rows, metrics(rows)


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


# ── ClubElo benchmark (optional, report-only — never a model input) ──────────
def _fetch_clubelo(date: str) -> pd.DataFrame | None:
    """One cached snapshot of api.clubelo.com/{date} (all clubs, that date's
    ratings). Degrades to None on any network problem — this is a sanity-check
    comparison, not something a pipeline should ever fail on."""
    CLUBELO_CACHE.mkdir(parents=True, exist_ok=True)
    path = CLUBELO_CACHE / f"{date}.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    req = urllib.request.Request(CLUBELO_URL.format(date=date),
                                 headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  clubelo: fetch failed for {date} ({exc}) — benchmark skipped")
        return None
    path.write_text(raw, encoding="utf-8")
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def benchmark_clubelo(verbose: bool = True) -> dict | None:
    """Report-only: convert ClubElo ratings to a 1X2 probability via the same
    shape as model._lambdas_elo, and compare Brier to ours on walk-forward
    rows near the snapshot date (a single snapshot is only meaningful close
    to its own date — matches within CLUBELO_WINDOW_DAYS of the most recent
    played match). Never fed into model.fit/predict. Skips cleanly offline
    or if no team names match ClubElo's naming."""
    from .names import FDCOUK_ALIASES
    rows, _ = walk_forward(verbose=False)
    if not rows:
        print("clubelo benchmark: no walk-forward rows — skipped")
        return None
    df = pd.DataFrame(rows)
    anchor = str(df["date"].max())
    elo_df = _fetch_clubelo(anchor)
    if elo_df is None or elo_df.empty or "Club" not in elo_df.columns:
        print("clubelo benchmark: unavailable — skipped")
        return None
    elo_map = {FDCOUK_ALIASES.get(r.Club, r.Club): float(r.Elo)
              for r in elo_df.itertuples(index=False)}

    window = df[pd.to_datetime(df["date"]) >= pd.Timestamp(anchor) - pd.Timedelta(days=CLUBELO_WINDOW_DAYS)]
    matched = 0
    brier_sum = 0.0
    for r in window.itertuples(index=False):
        eh, ea = elo_map.get(r.home), elo_map.get(r.away)
        if eh is None or ea is None:
            continue
        diff = (eh + CLUBELO_HOME_ADV - ea) / 400.0
        total = 2.55 + 0.20 * abs(diff)
        share = 1.0 / (1.0 + math.exp(-1.2 * diff))
        lam_h, lam_a = max(0.15, total * share), max(0.15, total * (1 - share))
        mat = M.score_matrix(lam_h, lam_a, M.DC_RHO)
        p = M.probs_from_matrix(mat)
        probs = np.array([p["home"], p["draw"], p["away"]])
        one = np.eye(3)[int(r.actual)]
        brier_sum += float(np.sum((probs - one) ** 2))
        matched += 1
    if matched == 0:
        print("clubelo benchmark: no team names matched ClubElo's naming — skipped")
        return None
    clubelo_brier = brier_sum / matched
    our_brier = metrics(window.to_dict("records"))["brier"]
    out = {"anchor_date": anchor, "window_days": CLUBELO_WINDOW_DAYS,
           "n_matched": matched, "n_window": int(len(window)),
           "our_brier": round(our_brier, 4), "clubelo_brier": round(clubelo_brier, 4)}
    if verbose:
        print(f"\nClubElo benchmark (report-only, never a model input)")
        print(f"  snapshot {anchor}, matches in the trailing {CLUBELO_WINDOW_DAYS}d, "
              f"n={matched}/{len(window)} name-matched")
        print(f"  our Brier {our_brier:.4f}  vs  ClubElo-implied Brier {clubelo_brier:.4f}")
    return out


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
    print(f"\nCalibration (temperature scaling) on {len(A)} walk-forward predictions")
    split_results = []
    for split_name in CALIBRATION_SPLITS:
        split = np.datetime64(split_name)
        tr, te = dates < split, dates >= split
        if tr.sum() < 1000 or te.sum() < 100:
            continue
        temperature = CAL.fit_temperature(P[tr], A[tr])
        raw = _metrics_arr(P[te], A[te])
        calibrated = _metrics_arr(CAL.temperature_probs(P[te], temperature), A[te])
        split_results.append({
            "split": split_name, "train_n": int(tr.sum()), "test_n": int(te.sum()),
            "temperature": round(float(temperature), 3),
            "raw": {"accuracy": raw[0], "brier": raw[1], "log_loss": raw[2]},
            "calibrated": {"accuracy": calibrated[0], "brier": calibrated[1],
                           "log_loss": calibrated[2]},
            "delta_brier": calibrated[1] - raw[1],
            "delta_log_loss": calibrated[2] - raw[2],
        })
        if verbose:
            print(f"  split {split_name}: T={temperature:.3f}, n={int(te.sum())}, "
                  f"ΔBrier {calibrated[1] - raw[1]:+.6f}, "
                  f"Δlog-loss {calibrated[2] - raw[2]:+.6f}")

    promotes = bool(split_results) and all(
        row["calibrated"]["brier"] < row["raw"]["brier"]
        and row["calibrated"]["log_loss"] < row["raw"]["log_loss"]
        for row in split_results
    )
    temperature_all = CAL.fit_temperature(P, A)
    raw_all = _metrics_arr(P, A)
    cal_all = _metrics_arr(CAL.temperature_probs(P, temperature_all), A)
    print(f"  all-data temperature: {temperature_all:.3f}; "
          f"Brier {raw_all[1]:.6f} -> {cal_all[1]:.6f}; "
          f"log-loss {raw_all[2]:.6f} -> {cal_all[2]:.6f}")
    print(f"  gate: {'PROMOTE' if promotes else 'keep inactive'} "
          f"(both Brier and log-loss must improve on every split)")
    payload = {
        "active": promotes,
        "method": "temperature",
        "temperature": round(float(temperature_all), 3),
        "heldout": {
            "splits": split_results,
            "promote": promotes,
            "all_data": {
                "n": int(len(A)),
                "raw": {"accuracy": raw_all[0], "brier": raw_all[1], "log_loss": raw_all[2]},
                "calibrated": {"accuracy": cal_all[0], "brier": cal_all[1],
                               "log_loss": cal_all[2]},
            },
        },
        "note": "Temperature scaling is active only when both Brier and log-loss "
                "improve on every fixed held-out split.",
    }
    CALIB_FILE.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved temperature calibration (active={promotes}, T={temperature_all:.3f}) "
          f"-> {CALIB_FILE.name}")


def load_promotion_baseline() -> dict | None:
    """The promotion gate reference, or None if none has been established.

    Read-only from validation's perspective. The legacy expanding-window
    baseline is deliberately not a fallback: comparing different samples is
    not a regression test."""
    if not PROMOTION_BASELINE.exists():
        return None
    try:
        data = json.loads(PROMOTION_BASELINE.read_text())
    except (ValueError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _evaluation_hash(rows: list[dict]) -> str:
    """Identity/outcome hash for the exact fixed evaluation population."""
    h = hashlib.sha256()
    for row in rows:
        payload = [
            row.get("date"), row.get("competition"), row.get("fixture_id"),
            row.get("home"), row.get("away"), row.get("actual"),
            row.get("total_goals"), row.get("btts_actual"),
        ]
        h.update(json.dumps(payload, separators=(",", ":"), default=str).encode())
        h.update(b"\n")
    return h.hexdigest()[:20]


def gate_failures(rows: list[dict], measured: dict, baseline: dict) -> list[str]:
    """Validate a model on the baseline's identical fixed folds and metrics."""
    failures: list[str] = []
    for key in ("test_from", "test_to", "evaluation_hash", "n"):
        if key not in baseline:
            failures.append(f"promotion baseline lacks {key}")
    if failures:
        return failures
    if int(measured.get("n", -1)) != int(baseline["n"]):
        failures.append(
            f"evaluation row count {measured.get('n')} != baseline {baseline['n']}"
        )
    current_hash = _evaluation_hash(rows)
    if current_hash != baseline["evaluation_hash"]:
        failures.append(
            f"evaluation population hash {current_hash} != "
            f"baseline {baseline['evaluation_hash']}"
        )
    tolerances = baseline.get("tolerances", GATE_TOLERANCES)
    if not isinstance(tolerances, dict):
        failures.append("promotion baseline tolerances must be an object")
        return failures
    for metric in GATE_TOLERANCES:
        try:
            value = float(measured[metric])
            reference = float(baseline[metric])
            tolerance = float(tolerances.get(metric, GATE_TOLERANCES[metric]))
        except (KeyError, TypeError, ValueError):
            failures.append(f"{metric}: malformed current/baseline value")
            continue
        if not (math.isfinite(value) and math.isfinite(reference)
                and math.isfinite(tolerance) and tolerance >= 0):
            failures.append(f"{metric}: non-finite value or tolerance")
        elif value > reference + tolerance:
            failures.append(
                f"{metric} {value:.6f} > baseline {reference:.6f} "
                f"+ tolerance {tolerance:.6f}"
            )
    return failures


def gate_input_fingerprint(baseline: dict) -> str:
    """Fingerprint exactly the fixed-window rows and behaviour used by --gate.

    Results after the pinned test window cannot affect any gate fold and
    therefore do not invalidate this cache. Historical corrections, identity
    changes, model/validation code, coefficients and baseline edits all do.
    """
    frame = M.played(M.load_fixtures()).sort_values("date").reset_index(drop=True)
    cutoff = pd.Timestamp(str(baseline["test_to"])).to_period("M").start_time
    relevant = frame[frame["date"] < cutoff].reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(WFC.code_fingerprint().encode())
    digest.update(
        json.dumps(baseline, sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(WFC.row_hashes(relevant).tobytes())
    return digest.hexdigest()[:32]


def _load_gate_state(fingerprint: str) -> dict | None:
    if not GATE_STATE.exists():
        return None
    try:
        state = json.loads(GATE_STATE.read_text())
    except (OSError, ValueError):
        return None
    if (
        isinstance(state, dict)
        and state.get("fingerprint") == fingerprint
        and isinstance(state.get("passed"), bool)
        and isinstance(state.get("metrics"), dict)
    ):
        return state
    return None


def _write_gate_state(fingerprint: str, passed: bool, measured: dict,
                      failures: list[str]) -> None:
    payload = {
        "fingerprint": fingerprint,
        "passed": bool(passed),
        "metrics": measured,
        "failures": failures,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    GATE_STATE.parent.mkdir(exist_ok=True)
    tmp = GATE_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(GATE_STATE)


def _write_latest(measured: dict, reused: bool = False) -> None:
    LATEST.write_text(json.dumps(
        {"brier": measured["brier"], "log_loss": measured["log_loss"],
         "n": measured["n"], "brier_ou25": measured["brier_ou25"],
         "brier_btts": measured["brier_btts"],
         "generated_at_utc": datetime.now(timezone.utc).isoformat(),
         "source": "deduplicated canonical match identities",
         "reused_exact_inputs": reused,
         "note": "Descriptive validation output. This file does NOT move the "
                 "promotion gate; promotion_baseline.json is promoter-owned."},
        indent=2))


def opponent_xg_ab(test_from: str, test_to: str,
                    write: bool = False, verbose: bool = True) -> dict:
    """Reproducible fixed-window A/B for opponent-adjusted xG.

    Both arms use identical fixture identities and fold cutoffs. The artifact
    records code, data and evaluation-population hashes so its promotion claim
    can be reproduced rather than surviving as unauditable prose.
    """
    arms: dict[str, tuple[list[dict], dict]] = {}
    for label, active in (("incumbent", False),
                          ("opponent_adjusted_xg", True)):
        if verbose:
            print(f"[opponent-xg A/B] {label}")
        arms[label] = walk_forward(
            verbose=verbose, opponent_adjusted_xg=active,
            test_from=test_from, test_to=test_to,
        )
    incumbent_rows, incumbent = arms["incumbent"]
    adjusted_rows, adjusted = arms["opponent_adjusted_xg"]
    incumbent_hash = _evaluation_hash(incumbent_rows)
    adjusted_hash = _evaluation_hash(adjusted_rows)
    if len(incumbent_rows) != len(adjusted_rows) or incumbent_hash != adjusted_hash:
        raise RuntimeError(
            "opponent-xG A/B arms produced different evaluation populations"
        )
    metric_names = ("brier", "log_loss", "brier_ou25", "brier_btts")
    deltas = {
        metric: float(adjusted[metric]) - float(incumbent[metric])
        for metric in metric_names
    }
    promoted = all(delta < 0 for delta in deltas.values())
    fixture_frame = M.played(M.load_fixtures()).sort_values(
        "date"
    ).reset_index(drop=True)
    data_hash = hashlib.sha256(
        WFC.row_hashes(fixture_frame).tobytes()
    ).hexdigest()[:20]
    payload = {
        "status": "promoted" if promoted else "retired",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "monthly walk-forward with identical fixed folds",
        "test_from": test_from,
        "test_to": test_to,
        "n": len(adjusted_rows),
        "evaluation_hash": adjusted_hash,
        "code_hash": WFC.code_fingerprint(),
        "fixture_data_hash": data_hash,
        "incumbent": {metric: incumbent[metric] for metric in metric_names},
        "opponent_adjusted_xg": {
            metric: adjusted[metric] for metric in metric_names
        },
        "deltas": deltas,
        "decision": "promote" if promoted else "retire",
        "command": (
            "python3 -m club_soccer.validate --opponent-xg-ab "
            f"--test-from {test_from} --test-to {test_to} --write-evidence"
        ),
    }
    if write:
        OPPONENT_XG_EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
        if verbose:
            print(f"Evidence -> {OPPONENT_XG_EVIDENCE.name}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--if-needed", action="store_true",
                    help="with --gate, reuse the exact prior result when all "
                         "fixed-window inputs are unchanged")
    ap.add_argument("--opponent-xg-ab", action="store_true",
                    help="run a fixed-window opponent-adjusted-xG A/B")
    ap.add_argument("--test-from", default="2024-07-01")
    ap.add_argument("--test-to", default="2026-07-01")
    ap.add_argument("--write-evidence", action="store_true",
                    help="write the opponent-xG A/B evidence artifact")
    ap.add_argument("--calibrate", action="store_true",
                    help="fit isotonic 1X2 calibration, report held-out improvement, "
                         "and write data/calibration.json")
    ap.add_argument("--benchmark-clubelo", action="store_true",
                    help="report-only: compare ClubElo-implied 1X2 Brier to ours "
                         "near the most recent walk-forward date; never a model input")
    args = ap.parse_args()
    if args.opponent_xg_ab:
        result = opponent_xg_ab(
            args.test_from, args.test_to, write=args.write_evidence
        )
        print(json.dumps(result, indent=2))
        return
    if args.calibrate:
        cmd_calibrate()
        return
    if args.benchmark_clubelo:
        benchmark_clubelo()
        return
    if args.if_needed and not args.gate:
        sys.exit("--if-needed is valid only with --gate")
    base = load_promotion_baseline()
    if args.gate:
        if base is None:
            sys.exit(
                f"--gate needs {PROMOTION_BASELINE.name}, owned by the "
                "model promoter and established deliberately in a commit."
            )
        test_from = base.get("test_from")
        test_to = base.get("test_to")
        if not test_from or not test_to:
            sys.exit(
                f"{PROMOTION_BASELINE.name} must pin test_from and test_to; "
                "an expanding-window baseline is not comparable."
            )
    else:
        test_from = test_to = None

    gate_fingerprint = None
    if args.gate:
        gate_fingerprint = gate_input_fingerprint(base)
        if args.if_needed:
            state = _load_gate_state(gate_fingerprint)
            if state is not None:
                _write_latest(state["metrics"], reused=True)
                result = "PASS" if state["passed"] else "FAIL"
                print(
                    f"[gate] fixed-window inputs unchanged -> cached {result}"
                )
                for failure in state.get("failures", []):
                    print(f"  - {failure}")
                if not state["passed"]:
                    sys.exit(1)
                return
    rows, m = walk_forward(
        verbose=True, test_from=test_from, test_to=test_to
    )
    print(f"Walk-forward Club Soccer validation (n={m['n']})")
    print(f"accuracy {m['accuracy']:.1%}  Brier {m['brier']:.4f}  log-loss {m['log_loss']:.4f}")
    print(f"OU2.5 Brier {m['brier_ou25']:.4f}  BTTS Brier {m['brier_btts']:.4f}")
    DATA.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA / "validation_predictions.csv", index=False)
    # Descriptive output: always written, owned by validation, never a gate.
    _write_latest(m)
    print(f"Validation metrics -> {LATEST.name} "
          "(descriptive; the promotion gate is unchanged)")

    if base and "brier_ou25" in base:
        print(f"  Δ vs promotion baseline: "
              f"OU2.5 {m['brier_ou25'] - base['brier_ou25']:+.4f}  "
              f"BTTS {m['brier_btts'] - base.get('brier_btts', 0):+.4f}")
    if args.gate:
        failures = gate_failures(rows, m, base)
        _write_gate_state(
            gate_fingerprint, passed=not failures, measured=m, failures=failures
        )
        print(f"[gate] fixed window {base['test_from']}..{base['test_to']} "
              f"(n={m['n']}) -> {'PASS' if not failures else 'FAIL'}")
        for failure in failures:
            print(f"  - {failure}")
        if failures:
            sys.exit(1)


if __name__ == "__main__":
    main()
