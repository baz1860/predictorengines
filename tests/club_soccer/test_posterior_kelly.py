#!/usr/bin/env python3
"""Tests for E2 posterior-variance-aware staking.

This one touches the code that decides how much real money goes on a bet, so
the OFF path is tested as an exact identity rather than an approximation, and
the shrink rule is tested for the properties that make it safe (never
increases a stake, never leaves [0, 1]) rather than for the numbers it happens
to produce today.

The A/B's own honesty machinery is tested too: the matched-stake
normalisation, and the verdict refusing to call a result on thin evidence.

Offline: no ledger reads, no fits.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from club_soccer import edge as E
from club_soccer import posterior_kelly_ab as PK


# ── the OFF path is an identity, not an approximation ───────────────────────

def test_default_is_off():
    assert E.POSTERIOR_KELLY_DEFAULT is False


@pytest.mark.parametrize("p_model,p_book,odds", [
    (0.55, 0.50, 2.10), (0.33, 0.30, 3.40), (0.72, 0.68, 1.45),
    (0.10, 0.12, 9.00), (0.90, 0.88, 1.08),
])
def test_off_path_reproduces_flat_quarter_kelly_exactly(p_model, p_book, odds):
    """With the flag off the arithmetic must be the pre-E2 arithmetic.

    Equality, not tolerance — the whole point of gating is that a card built
    today is stake-for-stake identical to one built before this existed, and
    the posterior arguments are ignored entirely.
    """
    expected = E.KELLY_FRACTION * E.kelly(p_model, odds)
    assert E.posterior_kelly_stake(p_model, p_book, odds, 0.13, 0.09,
                                   active=False) == expected
    # And with no posterior available at all, whatever the flag says.
    assert E.posterior_kelly_stake(p_model, p_book, odds, None, None,
                                   active=True) == expected


# ── the shrink rule ─────────────────────────────────────────────────────────

def test_shrink_is_one_at_the_reference():
    assert E.posterior_shrink(0.10, 0.10) == pytest.approx(1.0)


def test_shrink_falls_as_uncertainty_rises():
    values = [E.posterior_shrink(sd, 0.10)
              for sd in (0.10, 0.12, 0.15, 0.20, 0.40)]
    assert values == sorted(values, reverse=True)
    assert values[-1] < values[0]


def test_shrink_never_exceeds_one():
    """A confident model has not earned MORE than its measured edge.

    Kelly at the posterior mean is already the log-growth optimum — expected
    log growth is linear in p, so parameter uncertainty alone does not justify
    betting bigger. Only selection bias, which grows with variance, justifies
    betting smaller. So the multiplier is capped and the rule is one-sided.
    """
    for sd in (1e-9, 0.001, 0.01, 0.05):
        assert E.posterior_shrink(sd, 0.10) <= 1.0


def test_shrink_is_a_no_op_on_missing_or_degenerate_input():
    for sd, ref in ((None, 0.1), (0.1, None), (None, None),
                    (0.1, 0.0), (0.1, -1.0), (float("nan"), 0.1),
                    (0.1, float("inf")), ("x", 0.1)):
        assert E.posterior_shrink(sd, ref) == 1.0


def test_more_uncertainty_never_increases_a_stake():
    """The property that makes this safe to switch on at all."""
    base = E.posterior_kelly_stake(0.55, 0.50, 2.10, 0.10, 0.10, active=True)
    for sd in (0.11, 0.15, 0.25, 0.60):
        worse = E.posterior_kelly_stake(0.55, 0.50, 2.10, sd, 0.10, active=True)
        assert worse <= base


def test_stake_stays_in_range_at_degenerate_inputs():
    for p_model, p_book, odds in ((0.999, 0.5, 1.01), (0.001, 0.5, 50.0),
                                  (1.0, 0.0, 1.0001), (0.0, 1.0, 20.0)):
        for active in (False, True):
            f = E.posterior_kelly_stake(p_model, p_book, odds, 0.2, 0.1,
                                        active=active)
            assert 0.0 <= f <= 1.0
            assert math.isfinite(f)


def test_shrinking_moves_the_edge_toward_the_book_not_toward_zero():
    """The rule corrects selection bias, so it pulls p_model toward the price
    the bet was actually taken at — not toward 0.5, and not toward no bet."""
    high_sd = E.posterior_kelly_stake(0.55, 0.50, 2.10, 0.40, 0.10, active=True)
    at_book = E.KELLY_FRACTION * E.kelly(0.50, 2.10)
    unshrunk = E.KELLY_FRACTION * E.kelly(0.55, 2.10)
    assert at_book <= high_sd <= unshrunk


# ── the log-growth metric ───────────────────────────────────────────────────

def test_log_growth_is_maximised_at_the_kelly_fraction():
    """Sanity-check the objective against the thing it is meant to be.

    If the closing price equals the model, full Kelly is the optimum by
    construction — so the metric peaks there and not somewhere else.
    """
    p, odds = 0.55, 2.10
    star = E.kelly(p, odds)
    best = PK._log_growth(star, odds, p)
    for f in (star * 0.5, star * 0.8, star * 1.25, star * 2.0):
        assert PK._log_growth(f, odds, p) <= best + 1e-12


def test_log_growth_is_zero_for_a_zero_stake():
    assert PK._log_growth(0.0, 2.0, 0.5) == 0.0


def test_log_growth_penalises_a_bet_the_close_disagrees_with():
    """Staking into a price the market says is wrong must score negative."""
    assert PK._log_growth(0.05, 2.10, 0.30) < 0.0


# ── the A/B's own honesty machinery ─────────────────────────────────────────

def _arms(n_days: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for _ in range(8):
            odds = float(rng.uniform(1.6, 4.0))
            rows.append({
                "day": f"2026-08-{d + 1:02d}",
                "odds": odds,
                "p_close": float(rng.uniform(0.25, 0.6)),
                "stake_incumbent": float(rng.uniform(0.005, 0.04)),
                "won": float(rng.integers(0, 2)),
            })
    frame = pd.DataFrame(rows)
    frame["stake_candidate"] = frame["stake_incumbent"] * rng.uniform(
        0.5, 1.0, len(frame))
    return frame


def test_matched_stake_equalises_total_money_at_risk():
    """The primary metric must not reward simply betting less.

    The shrink is one-sided, and the decision-time book runs a negative flat
    ROI, so an unmatched comparison would favour the candidate for a reason
    unrelated to uncertainty. This is the correction that removes it.
    """
    frame = _arms()
    scale = frame["stake_incumbent"].sum() / frame["stake_candidate"].sum()
    frame["stake_candidate_matched"] = frame["stake_candidate"] * scale
    assert frame["stake_candidate_matched"].sum() == pytest.approx(
        frame["stake_incumbent"].sum())
    assert scale > 1.0, "fixture should have the candidate staking less"


def test_block_bootstrap_resamples_days_not_bets():
    """Bets on one day share a model fit and often a fixture, so the block is
    the day. Counting 1,244 correlated decisions as 1,244 samples would
    overstate the evidence by an order of magnitude."""
    frame = _arms(n_days=12)
    for arm in ("incumbent", "candidate"):
        frame[f"growth_{arm}"] = [
            PK._log_growth(f, o, p) for f, o, p in
            zip(frame[f"stake_{arm}"], frame["odds"], frame["p_close"])
        ]
    out = PK._block_bootstrap(frame, "growth_candidate")
    assert out["blocks"] == 12
    assert out["ci_low"] < out["observed"] < out["ci_high"]


def test_verdict_refuses_to_decide_on_too_few_blocks():
    """A refinement to how a strategy stakes cannot be held to a weaker
    standard than the strategy itself, and `evidence_gate.MIN_INDEPENDENT_BLOCKS`
    is 8."""
    from club_soccer import evidence_gate as EG

    thin = {"primary_matched_stake": {"blocks": 3, "delta": 0.5,
                                      "excludes_zero": True}}
    verdict = PK._verdict(thin)
    assert verdict["decision"] == "undecidable"
    assert str(EG.MIN_INDEPENDENT_BLOCKS) in verdict["reason"]


def test_verdict_is_undecidable_when_the_interval_spans_zero():
    spanning = {"primary_matched_stake": {"blocks": 20, "delta": 0.02,
                                          "excludes_zero": False}}
    assert PK._verdict(spanning)["decision"] == "undecidable"


def test_verdict_can_still_promote_and_retire():
    good = {"primary_matched_stake": {"blocks": 20, "delta": 0.02,
                                      "excludes_zero": True}}
    bad = {"primary_matched_stake": {"blocks": 20, "delta": -0.02,
                                     "excludes_zero": True}}
    assert PK._verdict(good)["decision"] == "promote"
    assert PK._verdict(bad)["decision"] == "retire"
