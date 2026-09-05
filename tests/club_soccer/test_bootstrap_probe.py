#!/usr/bin/env python3
"""Tests for the E0 bootstrap spread probe and the fit hook it rides on.

Two things need guarding. First, `model.fit(row_weights=...)` touches the
production fitter, so its unused path must be provably identical to what ran
before — a diagnostic that perturbs live prices is worse than no diagnostic.
Second, the probe's read-outs decide whether weeks of modelling work happen,
so the metrics behind that decision are tested against cases with known
answers rather than trusted because they look plausible.

Offline: every test builds its own synthetic league, nothing reads fixtures.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from club_soccer import bootstrap_probe as BP
from club_soccer import model as M


def _league(n_rounds: int = 14, seed: int = 7) -> pd.DataFrame:
    """A deterministic synthetic league that `model.fit` will accept."""
    rng = np.random.default_rng(seed)
    teams = [f"Club {c}" for c in "ABCDEF"]
    rows, fid = [], 0
    d = pd.Timestamp("2025-01-04")
    for _ in range(n_rounds):
        for h, a in zip(teams[::2], teams[1::2]):
            fid += 1
            rows.append(dict(fixture_id=f"syn{fid}", date=d, season=2024,
                             competition="Synthetic League", type="league",
                             neutral=0, home=h, away=a, status="FIN",
                             home_goals=int(rng.poisson(1.4)),
                             away_goals=int(rng.poisson(1.1)),
                             home_sot=np.nan, away_sot=np.nan, xg_source=""))
        teams = teams[1:] + teams[:1]
        d += pd.Timedelta(days=16)
    return pd.DataFrame(rows)


def _key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


# ── the fit hook must be invisible when unused ──────────────────────────────

def test_row_weights_none_matches_unit_weights_exactly():
    """The OFF state is not "close enough" — it is the same fit.

    Both places the multiplier lands multiply by 1.0, which is exact in
    IEEE-754, so this is an equality assertion and not a tolerance.
    """
    fx = _league()
    assert _key(M.fit(fx)) == _key(M.fit(fx, row_weights=np.ones(len(fx))))


def test_row_weights_perturb_every_weighted_component():
    """Goals, xG and Elo must all move, or the probe measures a fraction of
    the ensemble and understates the spread.

    Elo is the one that can silently fail: it is a sequential loop that never
    sees the time-decay weights, so it needs the multiplier applied to its
    update size. If that wiring breaks, this test is what catches it.
    """
    fx = _league()
    base = M.fit(fx)
    rng = np.random.default_rng(0)
    w = rng.exponential(1.0, len(fx))
    boot = M.fit(fx, row_weights=w / w.mean())

    assert boot["elo"] != base["elo"], "Elo did not respond to row_weights"
    assert boot["attack"] != base["attack"], "goals model did not respond"


def test_row_weights_realign_to_the_frame_fit_actually_uses():
    """Weights are indexed to the frame as passed, not to fit's internal view.

    `fit` re-filters to played matches and re-sorts by date. Pairing weights
    positionally after that would silently attach each weight to the wrong
    match — a bug that produces plausible numbers and no error.

    Dates are made unique here so `sort_values` has exactly one valid answer.
    With tied dates the two frames sort into different orders, and the
    weighted sums then accumulate in different orders — which moves
    `global_hfa` by ~4e-16 and would fail an equality assertion for reasons
    that have nothing to do with realignment.
    """
    fx = _league()
    fx["date"] = pd.date_range("2025-01-04", periods=len(fx), freq="D")
    fx.loc[fx.index[:4], "status"] = "SCHEDULED"     # dropped by played()
    shuffled = fx.sample(frac=1.0, random_state=3)   # and not date-sorted

    rng = np.random.default_rng(11)
    weights = pd.Series(rng.exponential(1.0, len(shuffled)),
                        index=shuffled.index)

    from_unsorted = M.fit(shuffled, row_weights=weights)
    ordered = M.played(shuffled).sort_values("date")
    from_ordered = M.fit(ordered, row_weights=weights.reindex(ordered.index))

    assert _key(from_unsorted) == _key(from_ordered)


def test_row_weights_reject_malformed_input():
    fx = _league()
    with pytest.raises(ValueError, match="entries for"):
        M.fit(fx, row_weights=np.ones(len(fx) - 1))
    with pytest.raises(ValueError, match="non-negative"):
        bad = np.ones(len(fx)); bad[0] = -1.0
        M.fit(fx, row_weights=bad)
    with pytest.raises(ValueError, match="finite"):
        bad = np.ones(len(fx)); bad[0] = np.inf
        M.fit(fx, row_weights=bad)


def test_no_club_disappears_under_extreme_weights():
    """The reason this is a weighted bootstrap and not a row resample.

    Thin-data clubs are the population the probe exists to measure. A
    multinomial resample can drop every match one of them played, which
    removes it from `params["teams"]` and makes it unpredictable — biasing
    the result toward well-measured clubs and understating dispersion. Down-
    weighting cannot do that.
    """
    fx = _league()
    weights = np.where(fx["home"].eq("Club A") | fx["away"].eq("Club A"),
                       1e-9, 1.0)
    assert set(M.fit(fx, row_weights=weights)["teams"]) == set(M.fit(fx)["teams"])


# ── the probe's own machinery ───────────────────────────────────────────────

def test_boot_weights_are_reproducible_and_mean_one():
    """Seeded from (master, month, replicate), so the run reproduces under any
    worker count — results must not depend on task scheduling."""
    a = BP._boot_weights(500, [7, 2024, 3])
    b = BP._boot_weights(500, [7, 2024, 3])
    c = BP._boot_weights(500, [7, 2024, 4])
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # Mean 1 preserves effective sample size, which XG_RATING_PRIOR is
    # measured against.
    assert a.mean() == pytest.approx(1.0)
    assert (a >= 0).all()


def test_excess_error_removes_the_entropy_confound():
    """Read-out 2 correlates SD against EXCESS error for a reason.

    A near-uniform prediction carries a high expected Brier score no matter
    how certain the model is of it, and posterior SD is also largest for
    mid-range probabilities — so raw error and SD correlate through predictive
    entropy alone. Subtracting the implied Brier leaves only the error the
    point prediction did not already account for.

    Both arms below are perfectly self-consistent: each outcome happens
    exactly as often as the prediction claims. A confounded metric ranks the
    high-entropy arm as 3.5x worse; the excess scores both at zero, which is
    the correct answer for two equally honest predictions.
    """
    sharp = np.tile([0.90, 0.05, 0.05], (20, 1))
    sharp_actual = np.array([0] * 18 + [1, 2])            # 90% home, as priced

    flat = np.tile([0.40, 0.30, 0.30], (20, 1))
    flat_actual = np.array([0] * 8 + [1] * 6 + [2] * 6)   # 40/30/30, as priced

    assert BP._excess_error(sharp, sharp_actual).mean() == pytest.approx(0.0, abs=1e-12)
    assert BP._excess_error(flat, flat_actual).mean() == pytest.approx(0.0, abs=1e-12)

    # The confounded metric these replace would have ranked them far apart.
    raw_sharp = ((sharp - np.eye(3)[sharp_actual]) ** 2).sum(axis=1).mean()
    raw_flat = ((flat - np.eye(3)[flat_actual]) ** 2).sum(axis=1).mean()
    assert raw_flat > 3 * raw_sharp


def test_excess_error_is_zero_for_a_self_consistent_prediction():
    """A prediction that is right exactly as often as it claims has no excess."""
    p = np.tile([0.5, 0.25, 0.25], (4, 1))
    actual = np.array([0, 0, 1, 2])          # 2 home, 1 draw, 1 away — as priced
    assert BP._excess_error(p, actual).mean() == pytest.approx(0.0, abs=0.02)


def _rows(sds, probs, actuals, history):
    return [{"index": i, "month": "2025-01", "competition": "L",
             "sd_mean": s, "sd_home": s, "sd_draw": s, "sd_away": s,
             "p_home": p[0], "p_draw": p[1], "p_away": p[2],
             "actual": a, "min_history": h}
            for i, (s, p, a, h) in enumerate(zip(sds, probs, actuals, history))]


def test_readouts_stop_when_uncertainty_is_uniform():
    """Constant spread is the `variance_inflation` case that was already
    rejected by A/B — the probe must return STOP, not a marginal pass."""
    n = 400
    rng = np.random.default_rng(1)
    sds = np.full(n, 0.03)
    probs = np.tile([0.45, 0.28, 0.27], (n, 1))
    actual = rng.integers(0, 3, n)
    out = BP._readouts(_rows(sds, probs, actual, rng.integers(5, 90, n)))

    assert out["readout_1_dispersion"]["ratio_p90_p10"] == pytest.approx(1.0)
    assert out["readout_1_dispersion"]["pass"] is False
    assert out["verdict"] == "stop"


def test_readouts_proceed_when_spread_tracks_excess_error():
    """The converse: wide, uneven spread that genuinely predicts extra error."""
    n = 3000
    rng = np.random.default_rng(2)
    sds = rng.uniform(0.01, 0.09, n)
    probs = np.tile([0.45, 0.28, 0.27], (n, 1))
    # High-SD matches lose against the priced favourite more often, which is
    # exactly "the model is more wrong where it is less sure".
    upset = rng.random(n) < (sds / 0.09) * 0.6
    actual = np.where(upset, 2, 0)
    out = BP._readouts(_rows(sds, probs, actual, rng.integers(5, 90, n)))

    assert out["readout_1_dispersion"]["pass"] is True
    assert out["readout_2_signal"]["pass"] is True
    assert out["verdict"] == "proceed"


def test_readouts_report_the_confounded_correlation_alongside():
    """Both numbers are published so a reader can see what the excess removes,
    but only the excess one is allowed to decide."""
    n = 500
    rng = np.random.default_rng(4)
    rows = _rows(rng.uniform(0.01, 0.09, n),
                 np.tile([0.45, 0.28, 0.27], (n, 1)),
                 rng.integers(0, 3, n), rng.integers(5, 90, n))
    signal = BP._readouts(rows)["readout_2_signal"]
    assert "spearman_rho_raw_error_confounded" in signal
    assert signal["pass"] == bool(
        signal["spearman_rho_excess_error"] > BP.MIN_ERROR_RHO
        and signal["p_value"] < BP.MAX_ERROR_P
    )


def test_probe_promotes_nothing():
    """E0 is report-only: the evidence file is the only thing it may write.

    Checks calls rather than mentions — the module names
    `promotion_baseline.json` in prose, explaining that it borrows that file's
    window, and a substring match would flag it for documenting itself.
    """
    src = (BP.HERE / "bootstrap_probe.py").read_text()
    for forbidden in ("save_params(", "write_evidence_baseline",
                      "DEFAULT_ENSEMBLE_W =", "PROMOTION_BASELINE"):
        assert forbidden not in src, f"probe must not touch {forbidden}"

    writes = [line.strip() for line in src.splitlines()
              if ".write_text(" in line or ".to_csv(" in line]
    assert sorted(writes) == sorted([
        "EVIDENCE.write_text(json.dumps(payload, indent=2) + \"\\n\")",
        "tmp.write_text(json.dumps(rows))",          # per-month resume cache
    ]), f"probe writes somewhere unexpected: {writes}"

    assert BP.EVIDENCE.name == "bootstrap_spread_evidence.json"
    # The resume cache is scratch, and must stay out of the way of the data
    # files the engine actually reads.
    assert BP.CACHE_DIR.name.startswith(".")
    assert BP.CACHE_DIR.parent == BP.DATA
