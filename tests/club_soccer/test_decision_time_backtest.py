#!/usr/bin/env python3
"""Tests for the decision-time backtest engine (Phase A)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from club_soccer import decision_time_backtest as B
from club_soccer import evidence_gate as G
from club_soccer import market_settlement as MS


# ── de-vig ────────────────────────────────────────────────────────────────

def test_devig_removes_the_overround():
    probs = MS.devig({"home": 2.0, "draw": 4.0, "away": 4.0})
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["home"] > probs["draw"]


# ── settlement ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("market,side,hg,ag,won", [
    ("1x2", "home", 2, 0, True),
    ("1x2", "home", 0, 2, False),
    ("1x2", "draw", 1, 1, True),
    ("1x2", "away", 0, 2, True),
    ("total25", "over", 2, 1, True),
    ("total25", "over", 1, 1, False),
    ("total25", "under", 1, 1, True),
])
def test_settlement(market, side, hg, ag, won):
    assert MS.side_won(market, side, hg, ag) is won


def test_clv_is_positive_when_you_beat_the_close():
    # executed at 2.20, close de-vigged prob 0.50 -> log(2.20*0.50) = log(1.10) > 0
    clv = MS.clv("1x2", "home", 2.20, {"home": 0.50})
    assert clv > 0


def test_clv_computes_for_totals_when_a_close_is_supplied():
    """A closing-totals feed now exists (Pinnacle PC>2.5/PC<2.5) for the
    European leagues, so totals CLV is real where the close is present."""
    clv = MS.clv("total25", "over", 2.10, {"over": 0.52, "under": 0.48})
    assert clv is not None and clv > 0      # 2.10 * 0.52 > 1


def test_clv_is_none_for_totals_without_a_close():
    """The non-UEFA leagues have no closing-totals feed, so their totals CLV
    stays None — that market can never open the gate for them."""
    assert MS.clv("total25", "over", 2.0, None) is None


def test_closing_probs_returns_both_markets():
    one, tot = MS.closing_probs()
    # 1X2 close is broad; totals close exists for the European leagues.
    assert isinstance(one, dict) and isinstance(tot, dict)


# ── metrics ───────────────────────────────────────────────────────────────

def _bets(rows):
    return pd.DataFrame(rows)


def test_threshold_metrics_filter_by_edge():
    bets = _bets([
        {"market": "1x2", "side": "home", "odds": 2.0, "p_model": 0.55,
         "p_book": 0.50, "edge": 0.05, "won": 1, "clv": 0.02, "date": "2026-07-10"},
        {"market": "1x2", "side": "home", "odds": 2.0, "p_model": 0.51,
         "p_book": 0.50, "edge": 0.01, "won": 0, "clv": -0.01, "date": "2026-07-10"},
    ])
    out = B._threshold_metrics(bets)
    # only the +0.05 edge bet clears the 2% threshold
    assert out["1x2"]["2%"]["n_bets"] == 1
    assert out["1x2"]["4%"]["n_bets"] == 1
    assert out["1x2"]["6%"]["n_bets"] == 0


def test_threshold_metrics_emit_the_clv_scored_count():
    """n_clv (bets actually CLV-scored) must be reported separately from n_bets,
    so the gate's Wilson bound runs on the real closing sample (blocker 4)."""
    bets = _bets([
        {"market": "1x2", "side": "home", "odds": 2.0, "p_model": 0.55,
         "p_book": 0.50, "edge": 0.05, "won": 1, "clv": 0.02, "date": "2026-07-10"},
        {"market": "1x2", "side": "away", "odds": 2.0, "p_model": 0.55,
         "p_book": 0.50, "edge": 0.05, "won": 0, "clv": float("nan"),
         "date": "2026-07-10"},
    ])
    out = B._threshold_metrics(bets)
    assert out["1x2"]["2%"]["n_bets"] == 2
    assert out["1x2"]["2%"]["n_clv"] == 1     # only one bet had a real CLV


def test_log_losses_use_the_frozen_closing_prob_not_a_match_key():
    """Blocker 2: market log-loss must come from the settled p_close, on exactly
    the fixtures where the model also has a prob. The old code looked the close
    up by provider fixture id against a date|home|away map, so it never matched
    and 1X2 could never open."""
    import math
    bets = _bets([
        {"key": "111", "market": "1x2", "side": "home", "p_model": 0.55,
         "p_close": 0.50, "won": 1},
        {"key": "111", "market": "1x2", "side": "away", "p_model": 0.20,
         "p_close": 0.25, "won": 0},
        {"key": "222", "market": "1x2", "side": "home", "p_model": 0.60,
         "p_close": 0.0, "won": 1},          # no closing ref -> excluded from BOTH
    ])
    ml, mk = B._log_losses(bets)
    assert ml == pytest.approx(-math.log(0.55), abs=1e-4)
    assert mk == pytest.approx(-math.log(0.50), abs=1e-4)


def test_confidence_table_reports_hit_rate_by_bucket():
    bets = _bets([
        {"market": "1x2", "p_model": 0.57, "odds": 1.8, "won": 1, "edge": 0.05},
        {"market": "1x2", "p_model": 0.58, "odds": 1.9, "won": 0, "edge": -0.02},
        {"market": "1x2", "p_model": 0.62, "odds": 1.6, "won": 1, "edge": 0.03},
    ])
    table = {r["bucket"]: r for r in B._confidence_table(bets)}
    assert table["55%-60%"]["n"] == 2
    assert table["55%-60%"]["hit_rate"] == 0.5
    assert table["60%-65%"]["hit_rate"] == 1.0


def test_value_subset_excludes_negative_edge():
    bets = _bets([
        {"market": "1x2", "p_model": 0.57, "odds": 1.8, "won": 1, "edge": 0.05},
        {"market": "1x2", "p_model": 0.58, "odds": 1.9, "won": 0, "edge": -0.02},
    ])
    table = {r["bucket"]: r for r in B._confidence_table(bets, value_only=True)}
    assert table["55%-60%"]["n"] == 1     # the -edge bet is dropped


def test_empty_input_produces_a_structurally_valid_artifact(tmp_path, monkeypatch):
    """With no bets the artifact must still be the right shape — the gate should
    reject on n_bets, never crash on a malformed file."""
    monkeypatch.setattr(B, "SNAPSHOTS", tmp_path / "none.csv")
    monkeypatch.setattr(B, "ARTIFACT", tmp_path / "bt.json")
    monkeypatch.setattr(B, "LEDGER", tmp_path / "ledger.csv")
    art = B.run(verbose=False)
    assert art["backtest_version"] == "decision_time_v2"
    assert set(art["simulated_betting"]) == {"1x2", "total_over_under_2_5"}
    for m in art["simulated_betting"].values():
        assert set(m) == {"2%", "4%", "6%"}


# ── gate contract ─────────────────────────────────────────────────────────

def test_artifact_declares_the_methodology_the_gate_requires(tmp_path, monkeypatch):
    # Build on a throwaway path so a clean tree cannot be made to write the
    # production artifact as a side effect of this test (finding 15).
    monkeypatch.setattr(B, "ARTIFACT", tmp_path / "bt.json")
    monkeypatch.setattr(B, "LEDGER", tmp_path / "ledger.csv")
    art = B.run(verbose=False)
    assert art["backtest_version"] == G.REQUIRED_VERSION
    for field, want in G.REQUIRED_METHODOLOGY.items():
        assert art[field] == want
    assert G.MIN_DECISION_LEAD_MINUTES <= art["decision_lead_minutes"] <= G.MAX_DECISION_LEAD_MINUTES


def test_gate_stays_closed_on_insufficient_volume(tmp_path, monkeypatch):
    """The whole point: real but thin evidence must NOT open staking.

    Runs against a temp artifact so the shared production file is not mutated
    for the next suite (test hygiene — an earlier version polluted it)."""
    monkeypatch.setattr(B, "ARTIFACT", tmp_path / "bt.json")
    monkeypatch.setattr(B, "LEDGER", tmp_path / "ledger.csv")
    monkeypatch.setattr(G, "BACKTEST_JSON", tmp_path / "bt.json")
    B.run(verbose=False)
    # An empty/thin ledger means no market has evidence, so under per-market
    # gating the gate is closed with no market open (and, correctly, no
    # "failing criteria" noise from inactive markets).
    ok, _reasons = G.staking_allowed()
    assert ok is False
    assert not any(G.market_staking_allowed().values())


def test_decision_time_owns_the_gate_file():
    assert B.ARTIFACT.name == "backtest_market.json"


def test_wired_into_the_daily_pipeline():
    """The backtest must run in the daily flow so its artifact stays fresh for
    the gate. It lives in the pipeline body (_run_steps), which run() wraps."""
    import inspect
    from club_soccer import season
    src = inspect.getsource(season)
    assert "decision_time_backtest" in src
    assert "Decision-time backtest" in src
