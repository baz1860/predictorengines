#!/usr/bin/env python3
"""E0 — bootstrap spread probe (report-only diagnostic, promotes nothing).

Asks one question before any modelling work is done: **does the engine's
uncertainty about `p_model` actually vary from match to match?**

The staking layer currently sizes every bet with a flat quarter-Kelly
(`KELLY_FRACTION = 0.25` in `edge.py`). That constant is a uniform hedge
against estimation error. Replacing it with a per-match, uncertainty-aware
stake is only worth building if the underlying uncertainty is genuinely
uneven — and `experiments.json` already records `variance_inflation` as
"rejected by A/B", i.e. uniformly widening the predictive distribution did not
work. If posterior spread turns out to be near-constant here, this proposal
collapses into that retired one and must be abandoned on the same evidence.

See `plans/club_soccer_uncertainty_experiment.md` §5 for the gate this feeds.

Method
------
Random-weight (Bayesian) bootstrap. Rather than resampling matches with
replacement, each resample draws a positive weight per training match from
Exponential(1), normalised to mean 1, and passes it to `model.fit(row_weights=)`.

Why weights and not row resampling: a multinomial resample can drop every match
a thin-data club played, which removes that club from `params["teams"]` and
makes it unpredictable. Thin-data clubs are precisely the population this probe
exists to measure, so dropping them would bias the answer toward the
well-measured clubs and understate dispersion — the exact direction that would
produce a false negative. Exponential weights are proportional to a
Dirichlet(1,...,1) draw, so this is the standard Bayesian bootstrap; every club
survives every resample with its match count intact and only its influence
varying. Normalising to mean 1 preserves the effective sample size, which
matters because `XG_RATING_PRIOR` is measured against accumulated weight.

Point-in-time (engine plan §0.5) is inherited from the fold structure: each
test month is priced only from matches strictly before it, refit per resample.

Read-outs (thresholds in `plans/club_soccer_uncertainty_experiment.md` §5)
-------------------------------------------------------------------------
1. **Dispersion.** p90/p10 ratio of per-match posterior SD. Below
   `MIN_DISPERSION_RATIO` the uncertainty is effectively constant and H2 dies.
2. **Signal.** Spearman correlation between posterior SD and *excess* Brier
   error. See `_excess_error` for why the excess, and not the raw error, is
   the quantity that decides.
3. **Thin-club concentration.** Posterior SD for matches involving
   low-history clubs vs well-measured ones. Informational: confirms the
   effect sits where theory says it should. Not a kill criterion.

Usage
-----
    python3 -m club_soccer.bootstrap_probe --write-evidence
    python3 -m club_soccer.bootstrap_probe --n-boot 40 --months 6   # quick look
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import model as M
from . import walkforward_cache as WFC

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EVIDENCE = DATA / "bootstrap_spread_evidence.json"
# Per-month checkpoints. A full run is thousands of refits and can take the
# better part of an hour on a machine that is also being used for other things,
# so finished months are kept and a re-run resumes rather than restarts. Keyed
# on the code fingerprint and the fold's own training rows, so a cached month
# can never be served for a different model or a changed fixture history.
CACHE_DIR = DATA / ".bootstrap_cache"

# The window pinned in promotion_baseline.json, so the probe describes the
# same population every other club-soccer measurement is taken on.
DEFAULT_TEST_FROM = "2024-07-01"
DEFAULT_TEST_TO = "2026-07-01"
# 120 resamples puts the read-out 1 noise floor at 1.18 (see
# _dispersion_noise_floor) against a 1.5 threshold — comfortable margin — while
# read-out 2's power comes from the ~27k matches in the window, not from the
# resample count. Going to 200 buys a 1.14 floor for two-thirds more compute.
DEFAULT_N_BOOT = 120
DEFAULT_SEED = 20260815
MIN_TRAIN = 200

# Decision thresholds. Fixed here, before the numbers are seen, so the probe
# cannot be graded on a curve after the fact.
MIN_DISPERSION_RATIO = 1.5   # read-out 1: p90/p10 of per-match posterior SD
MIN_ERROR_RHO = 0.05         # read-out 2: Spearman(SD, excess error)
MAX_ERROR_P = 0.01           # read-out 2: and significantly so

_OUTCOMES = ("home", "draw", "away")

# Worker-process state. Each worker loads the fixture frame once in the
# initializer rather than receiving it per task.
_DF: pd.DataFrame | None = None


def load_frame() -> pd.DataFrame:
    """The deterministic played-match frame every arm of the probe shares."""
    df = M.played(M.load_fixtures()).sort_values("date").reset_index(drop=True)
    df["_ym"] = df["date"].dt.to_period("M")
    return df


def _init_worker() -> None:
    global _DF
    _DF = load_frame()


def _fold(df: pd.DataFrame, ym: pd.Period) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = df[df["_ym"] == ym]
    train = df[df["date"] < test["date"].min()]
    return train, test


def _boot_weights(n: int, seed_parts: list[int]) -> np.ndarray:
    """Mean-1 Exponential(1) weights, reproducible from the seed parts alone.

    Seeding from (master, month, replicate) rather than from a stream means a
    replicate's weights do not depend on how tasks were scheduled, so the run
    reproduces under any worker count — including 1.
    """
    rng = np.random.default_rng(seed_parts)
    w = rng.exponential(1.0, size=n)
    mean = w.mean()
    if mean <= 0:                      # unreachable for n>=1; guard the divide
        return np.ones(n)
    return w / mean


def _predict_fold(params: dict, test: pd.DataFrame) -> dict[int, tuple]:
    """Price a test month off `params`, via the same path the card uses."""
    seen = set(params["teams"])
    out: dict[int, tuple] = {}
    for r in test.itertuples():
        if r.home not in seen or r.away not in seen:
            continue
        try:
            pred = M.predict_match(
                r.home, r.away, r.competition, str(r.date.date()), "ensemble",
                bool(r.neutral), params=params,
                fixture_id=getattr(r, "fixture_id", None),
                ensemble_weights=dict(M.DEFAULT_ENSEMBLE_W),
            )
        except Exception:
            continue
        p = pred["probs"]
        out[int(r.Index)] = (p["home"], p["draw"], p["away"])
    return out


def _fold_cache_key(train: pd.DataFrame, test: pd.DataFrame,
                    n_boot: int, seed: int) -> str:
    h = hashlib.sha256()
    h.update(WFC.code_fingerprint().encode())
    h.update(f"{n_boot}|{seed}|".encode())
    for frame in (train, test):
        h.update(WFC.row_hashes(frame.drop(columns=["_ym"])).tobytes())
    return h.hexdigest()[:20]


def _cache_load(ym: str, key: str) -> list[dict] | None:
    path = CACHE_DIR / f"{ym}-{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _cache_store(ym: str, key: str, rows: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_DIR / f".{ym}-{key}.tmp"
    tmp.write_text(json.dumps(rows))
    tmp.replace(CACHE_DIR / f"{ym}-{key}.json")


def _boot_task(args: tuple) -> dict[int, tuple]:
    """One resample of one month. Runs in a worker process."""
    ym_str, month_ord, replicate, master_seed = args
    assert _DF is not None, "worker not initialised"
    train, test = _fold(_DF, pd.Period(ym_str, "M"))
    weights = _boot_weights(len(train), [master_seed, month_ord, replicate])
    params = M.fit(
        train,
        coef_as_of=str(test["date"].min())[:10],
        row_weights=pd.Series(weights, index=train.index),
    )
    return _predict_fold(params, test)


def _excess_error(probs: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Realised Brier minus the Brier the prediction itself implies.

    Correlating posterior SD against raw Brier error would be confounded:
    a match priced near (0.33, 0.33, 0.33) has a high expected Brier score no
    matter how certain the model is of those numbers, and posterior SD is also
    largest for mid-range probabilities. The two would correlate through
    predictive entropy alone, with parameter uncertainty contributing nothing.

    Under its own prediction p, a match's expected Brier is sum_k p_k(1 - p_k).
    Subtracting it leaves the error the point prediction did NOT already
    account for. If posterior SD predicts *that*, the model is genuinely more
    wrong when it is more uncertain, which is the claim H2 rests on.
    """
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual)), actual] = 1.0
    realised = ((probs - onehot) ** 2).sum(axis=1)
    implied = (probs * (1.0 - probs)).sum(axis=1)
    return realised - implied


def _team_history(train: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for col in ("home", "away"):
        for team, n in train[col].value_counts().items():
            counts[team] = counts.get(team, 0) + int(n)
    return counts


def run_probe(test_from: str = DEFAULT_TEST_FROM,
              test_to: str = DEFAULT_TEST_TO,
              n_boot: int = DEFAULT_N_BOOT,
              seed: int = DEFAULT_SEED,
              months: int | None = None,
              workers: int | None = None,
              verbose: bool = True) -> dict:
    """Run the probe and return the evidence payload (does not write it)."""
    started = time.monotonic()
    df = load_frame()
    all_months = sorted(df["_ym"].unique())
    first = pd.Timestamp(test_from).to_period("M")
    last = pd.Timestamp(test_to).to_period("M")
    fold_months = [ym for ym in all_months if first <= ym < last]
    if months is not None:
        # Even spread across the window, not the first N — a contiguous head
        # would describe one season rather than the pinned window.
        idx = np.linspace(0, len(fold_months) - 1, num=min(months, len(fold_months)))
        fold_months = [fold_months[int(round(i))] for i in idx]
    if not fold_months:
        raise ValueError(f"no test months in [{test_from}, {test_to})")

    # Leave the machine some headroom. This runs for tens of minutes on a
    # personal Mac that is usually doing other things, and saturating every
    # core makes the box unusable while buying little — the fits are already
    # CPU-bound and contend with each other.
    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 2))
    if verbose:
        print(f"[bootstrap probe] {len(fold_months)} months x {n_boot} resamples "
              f"on {workers} workers", flush=True)

    sd_rows: list[dict] = []
    dropped_months: list[str] = []
    total_fits = 0

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker) as pool:
        for k, ym in enumerate(fold_months, 1):
            train, test = _fold(df, ym)
            if len(train) < MIN_TRAIN:
                dropped_months.append(str(ym))
                continue

            cache_key = _fold_cache_key(train, test, n_boot, seed)
            cached = _cache_load(str(ym), cache_key)
            if cached is not None:
                sd_rows.extend(cached)
                if verbose:
                    print(f"  [{k:>2}/{len(fold_months)}] {ym}  "
                          f"{len(cached)} matches (cached)", flush=True)
                continue

            started_month = time.monotonic()
            # Baseline = the production fit. Supplies the point prediction the
            # excess-error metric is measured against.
            base_params = M.fit(train, coef_as_of=str(test["date"].min())[:10])
            base = _predict_fold(base_params, test)
            history = _team_history(train)
            if not base:
                dropped_months.append(str(ym))
                continue

            tasks = [(str(ym), ym.ordinal, b, seed) for b in range(n_boot)]
            draws: dict[int, list[tuple]] = {i: [] for i in base}
            for result in pool.map(_boot_task, tasks, chunksize=1):
                for i, p in result.items():
                    if i in draws:
                        draws[i].append(p)
            total_fits += n_boot + 1

            month_rows: list[dict] = []
            kept = 0
            for i, row_draws in draws.items():
                # A match must be priced by essentially every resample for its
                # spread to mean anything. Weighted bootstrap keeps all clubs,
                # so a shortfall signals a fit failure worth excluding.
                if len(row_draws) < 0.8 * n_boot:
                    continue
                arr = np.asarray(row_draws, dtype=float)
                sds = arr.std(axis=0, ddof=1)
                r = df.loc[i]
                month_rows.append({
                    "index": int(i),
                    "month": str(ym),
                    "competition": r["competition"],
                    "sd_mean": float(sds.mean()),
                    "sd_home": float(sds[0]),
                    "sd_draw": float(sds[1]),
                    "sd_away": float(sds[2]),
                    "p_home": base[i][0],
                    "p_draw": base[i][1],
                    "p_away": base[i][2],
                    "actual": (0 if r["home_goals"] > r["away_goals"]
                               else (1 if r["home_goals"] == r["away_goals"] else 2)),
                    "min_history": int(min(history.get(r["home"], 0),
                                           history.get(r["away"], 0))),
                })
                kept += 1
            sd_rows.extend(month_rows)
            _cache_store(str(ym), cache_key, month_rows)
            if verbose:
                print(f"  [{k:>2}/{len(fold_months)}] {ym}  {kept} matches "
                      f"({time.monotonic() - started_month:.0f}s)", flush=True)

    if not sd_rows:
        raise RuntimeError("probe produced no measurable matches")

    payload = _readouts(sd_rows, n_boot=n_boot)
    payload.update({
        "experiment": "bootstrap_spread_probe",
        "purpose": ("E0 — does per-match model uncertainty vary enough to be "
                    "worth staking against? Report-only; promotes nothing."),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": ("random-weight (Bayesian) bootstrap, mean-1 Exponential(1) "
                   "row weights, monthly refit, point-in-time folds"),
        "test_from": test_from,
        "test_to": test_to,
        "window_note": (
            "full pinned window (matches promotion_baseline.json)"
            if (test_from == DEFAULT_TEST_FROM and test_to == DEFAULT_TEST_TO
                and months is None)
            else "PARTIAL — narrower than the pinned promotion window "
                 f"({DEFAULT_TEST_FROM} -> {DEFAULT_TEST_TO}); the verdict "
                 "below describes only the months actually measured"
        ),
        "months_tested": len(fold_months) - len(dropped_months),
        "months_dropped": dropped_months,
        "n_boot": n_boot,
        "seed": seed,
        "n": len(sd_rows),
        "total_fits": total_fits,
        "elapsed_s": round(time.monotonic() - started, 1),
        "code_hash": WFC.code_fingerprint(),
        "fixture_data_hash": _data_hash(df),
        "thresholds": {
            "min_dispersion_ratio": MIN_DISPERSION_RATIO,
            "min_error_rho": MIN_ERROR_RHO,
            "max_error_p": MAX_ERROR_P,
        },
        "command": ("python3 -m club_soccer.bootstrap_probe "
                    f"--test-from {test_from} --test-to {test_to} "
                    f"--n-boot {n_boot} --seed {seed} --write-evidence"),
    })
    return payload


def _data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        WFC.row_hashes(df.drop(columns=["_ym"])).tobytes()
    ).hexdigest()[:20]


def _dispersion_noise_floor(n_boot: int) -> float:
    """The p90/p10 ratio read-out 1 would show if every match had the SAME
    true spread.

    An SD estimated from `n_boot` resamples is itself noisy, and that noise
    alone spreads the measured values out — so a ratio above 1.0 proves
    nothing on its own. With sd_hat/sd_true distributed as
    sqrt(chi2_{B-1}/(B-1)), the floor is the ratio of that distribution's own
    90th and 10th percentiles. Read-out 1 is only meaningful to the extent it
    clears this.
    """
    from scipy.stats import chi2

    dof = max(1, n_boot - 1)
    return float(np.sqrt(chi2.ppf(0.90, dof) / chi2.ppf(0.10, dof)))


def _readouts(sd_rows: list[dict], n_boot: int = DEFAULT_N_BOOT) -> dict:
    """Turn per-match spreads into the three decisions."""
    from scipy.stats import spearmanr

    sd = np.array([r["sd_mean"] for r in sd_rows], dtype=float)
    probs = np.array([[r["p_home"], r["p_draw"], r["p_away"]] for r in sd_rows],
                     dtype=float)
    actual = np.array([r["actual"] for r in sd_rows], dtype=int)
    history = np.array([r["min_history"] for r in sd_rows], dtype=float)

    p10, p50, p90 = (float(np.percentile(sd, q)) for q in (10, 50, 90))
    dispersion = p90 / p10 if p10 > 0 else float("inf")

    excess = _excess_error(probs, actual)
    raw = ((probs - np.eye(3)[actual]) ** 2).sum(axis=1)
    if np.ptp(sd) == 0.0:
        # Perfectly constant spread: Spearman is undefined and returns NaN,
        # which is not valid JSON for anything reading the evidence file
        # outside Python. Report the honest answer — no rank association —
        # and let the thresholds fail it, which is what read-out 1 already
        # concluded for this case anyway.
        rho, pval, raw_rho = 0.0, 1.0, 0.0
    else:
        rho, pval = spearmanr(sd, excess)
        # Reported for contrast: the confounded version this deliberately avoids.
        raw_rho, _ = spearmanr(sd, raw)

    cut = float(np.median(history))
    thin, thick = sd[history <= cut], sd[history > cut]
    thin_mean = float(thin.mean()) if len(thin) else float("nan")
    thick_mean = float(thick.mean()) if len(thick) else float("nan")

    noise_floor = _dispersion_noise_floor(n_boot)
    dispersion_pass = bool(dispersion >= MIN_DISPERSION_RATIO
                           and dispersion > noise_floor)
    signal_pass = bool(rho > MIN_ERROR_RHO and pval < MAX_ERROR_P)
    verdict = "proceed" if (dispersion_pass and signal_pass) else "stop"

    return {
        "readout_1_dispersion": {
            "sd_p10": p10, "sd_p50": p50, "sd_p90": p90,
            "ratio_p90_p10": float(dispersion),
            "threshold": MIN_DISPERSION_RATIO,
            "noise_floor_ratio": noise_floor,
            "noise_floor_note": ("the ratio this read-out would show if every "
                                 "match had identical true spread, given the "
                                 "resample count — a pass must clear it"),
            "pass": dispersion_pass,
        },
        "readout_2_signal": {
            "spearman_rho_excess_error": float(rho),
            "p_value": float(pval),
            "spearman_rho_raw_error_confounded": float(raw_rho),
            "threshold_rho": MIN_ERROR_RHO,
            "threshold_p": MAX_ERROR_P,
            "pass": signal_pass,
            "note": ("decided on excess error (realised Brier minus the Brier "
                     "the prediction implies); the raw correlation is reported "
                     "only to show the confounding it removes"),
        },
        "readout_3_thin_clubs": {
            "history_split_matches": cut,
            "sd_mean_thin": thin_mean,
            "sd_mean_thick": thick_mean,
            "ratio_thin_thick": (thin_mean / thick_mean
                                 if thick_mean and np.isfinite(thick_mean) and thick_mean > 0
                                 else float("nan")),
            "note": "informational — confirms where the effect sits, not a kill criterion",
        },
        "verdict": verdict,
        "decision": (
            "E0 passes: per-match uncertainty varies and predicts excess error. "
            "E2 (posterior-variance Kelly) is worth building."
            if verdict == "proceed" else
            "E0 fails: uncertainty is effectively uniform or does not predict "
            "excess error. H2 is dead — retire, and leave quarter-Kelly alone."
        ),
    }


def _print_report(payload: dict) -> None:
    d = payload["readout_1_dispersion"]
    s = payload["readout_2_signal"]
    t = payload["readout_3_thin_clubs"]
    print(f"\n== E0 bootstrap spread probe ==")
    print(f"   {payload['n']:,} matches, {payload['n_boot']} resamples x "
          f"{payload['months_tested']} months "
          f"({payload['total_fits']:,} fits, {payload['elapsed_s']:.0f}s)")
    print(f"\n  1. dispersion   p10={d['sd_p10']:.5f}  p50={d['sd_p50']:.5f}  "
          f"p90={d['sd_p90']:.5f}")
    print(f"     ratio p90/p10 = {d['ratio_p90_p10']:.2f} "
          f"(need >= {d['threshold']}, noise floor "
          f"{d['noise_floor_ratio']:.2f})  [{'PASS' if d['pass'] else 'FAIL'}]")
    print(f"\n  2. signal       spearman(SD, excess error) = "
          f"{s['spearman_rho_excess_error']:+.4f}  p={s['p_value']:.2e}")
    print(f"     (raw, confounded: {s['spearman_rho_raw_error_confounded']:+.4f})")
    print(f"     need rho > {s['threshold_rho']} and p < {s['threshold_p']}  "
          f"[{'PASS' if s['pass'] else 'FAIL'}]")
    print(f"\n  3. thin clubs   SD thin={t['sd_mean_thin']:.5f}  "
          f"thick={t['sd_mean_thick']:.5f}  ratio={t['ratio_thin_thick']:.2f}")
    print(f"     (informational)")
    print(f"\n  VERDICT: {payload['verdict'].upper()}")
    print(f"  {payload['decision']}\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="E0 bootstrap spread probe (report-only; promotes nothing)")
    ap.add_argument("--test-from", default=DEFAULT_TEST_FROM)
    ap.add_argument("--test-to", default=DEFAULT_TEST_TO)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--months", type=int, default=None,
                    help="sample this many months evenly across the window "
                         "(default: every month)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--write-evidence", action="store_true",
                    help=f"write {EVIDENCE.name}")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    payload = run_probe(test_from=args.test_from, test_to=args.test_to,
                        n_boot=args.n_boot, seed=args.seed, months=args.months,
                        workers=args.workers, verbose=not args.quiet)
    _print_report(payload)
    if args.write_evidence:
        EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Evidence -> {EVIDENCE.name}")


if __name__ == "__main__":
    main()
