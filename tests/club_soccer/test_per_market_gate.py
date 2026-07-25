#!/usr/bin/env python3
"""Per-market staking gate (adversarial finding 5).

A market that cannot be CLV-scored (or fails) must not veto a market that has
earned staking. The gate opens per market; the stake-zeroing honours it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from club_soccer import edge as E
from club_soccer import evidence_gate as G


def _pass(n=5000):
    return {"n_bets": n, "n_clv": n, "flat_roi": 0.05, "kelly_roi": 0.04,
            "flat_roi_lb95": 0.02, "kelly_roi_lb95": 0.02,
            "clv_mean": 0.03, "clv_frac_positive": 0.60}


def _fail(n=5000):
    return {"n_bets": n, "n_clv": n, "flat_roi": -0.05, "kelly_roi": -0.04,
            "flat_roi_lb95": -0.1, "kelly_roi_lb95": -0.1,
            "clv_mean": -0.01, "clv_frac_positive": 0.45}


def _empty():
    return {"n_bets": 0, "n_clv": 0, "flat_roi": None, "kelly_roi": None,
            "clv_mean": None, "clv_frac_positive": None}


def _artifact(one_x_two, totals):
    return {
        "backtest_version": "decision_time_v2",
        "selection_method": "latest_quote_at_or_before_decision_time",
        "execution_method": "same_decision_time_quote",
        "clv_reference": "pinnacle_closing_devigged",
        "decision_lead_minutes": 90,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulated_betting": {
            "1x2": {t: dict(one_x_two) for t in ("2%", "4%", "6%")},
            "total_over_under_2_5": {t: dict(totals) for t in ("2%", "4%", "6%")},
        },
        "simulated_betting_by_league": {
            "1x2": {
                "Test League": {
                    t: dict(one_x_two) for t in ("2%", "4%", "6%")
                }
            },
            "total_over_under_2_5": {
                "Test League": {
                    t: dict(totals) for t in ("2%", "4%", "6%")
                }
            },
        },
        "model_log_loss_1x2": 0.98,
        "market_log_loss_1x2_devigged_pinnacle_closing": 1.00,
    }


@pytest.fixture()
def gate(tmp_path, monkeypatch):
    path = tmp_path / "bt.json"
    monkeypatch.setattr(G, "BACKTEST_JSON", path)
    def _write(one, tot):
        path.write_text(json.dumps(_artifact(one, tot)))
    return _write


@pytest.fixture()
def gate_path(tmp_path, monkeypatch):
    path = tmp_path / "bt.json"
    monkeypatch.setattr(G, "BACKTEST_JSON", path)
    return path


def _clvless(n=400):
    """A league/market with bets but NO closing feed — the non-UEFA totals case."""
    return {"n_bets": n, "n_clv": 0, "flat_roi": 0.05, "kelly_roi": 0.04,
            "flat_roi_lb95": 0.02, "kelly_roi_lb95": 0.02,
            "clv_mean": None, "clv_frac_positive": None}


def test_passing_1x2_opens_even_with_inactive_totals(gate):
    gate(_pass(), _empty())
    m = G.market_staking_allowed()
    assert m["1x2"] is True
    assert m["total_over_under_2_5"] is False


def test_failing_totals_does_not_block_passing_1x2(gate):
    gate(_pass(), _fail())
    m = G.market_staking_allowed()
    assert m["1x2"] is True, "a failing OU2.5 must not veto a passing 1X2"


def test_passing_totals_opens_independently(gate):
    gate(_fail(), _pass())
    m = G.market_staking_allowed()
    assert m["total_over_under_2_5"] is True
    assert m["1x2"] is False


def test_inactive_market_is_not_a_veto_reason(gate):
    """An all-zero market's threshold complaints must not appear as reasons the
    whole gate is closed."""
    gate(_pass(), _empty())
    v = G.evaluate()
    assert v["allowed"] is True
    assert not any("OU2.5" in r for r in v["reasons"])


def test_global_staking_allowed_is_any_market_open(gate):
    gate(_pass(), _empty())
    ok, _ = G.staking_allowed()
    assert ok is True


# ── stake-zeroing honours per-market ──────────────────────────────────────

def test_stake_zeroing_is_per_market(gate, monkeypatch):
    gate(_pass(), _empty())
    monkeypatch.setattr("club_soccer.evidence_gate.BACKTEST_JSON", G.BACKTEST_JSON)
    rows = [
        {"market": "1x2", "competition": "Test League",
         "kelly_stake": 0.02, "stake_gbp": 2.0},
        {"market": "total", "competition": "Test League",
         "kelly_stake": 0.02, "stake_gbp": 2.0},
        {"market": "btts", "competition": "Test League",
         "kelly_stake": 0.02, "stake_gbp": 2.0},
    ]
    E.apply_evidence_gate(rows)
    by = {r["market"]: r for r in rows}
    assert by["1x2"]["stake_gbp"] == 2.0, "open 1X2 stake must stand"
    assert by["total"]["stake_gbp"] == 0.0, "inactive OU2.5 must be zeroed"
    assert by["btts"]["stake_gbp"] == 0.0, "BTTS is never stakeable"


def test_btts_is_never_stakeable(gate):
    """BTTS has no CLV reference and no gate key — it is display-only."""
    assert "btts" not in E._GATE_MARKET


# ── CLV count vs bet count (adversarial blocker 4) ─────────────────────────

def test_wilson_runs_on_the_clv_count_not_the_bet_count(gate):
    """1000 bets but only 10 CLV-scored is 10 samples of closing evidence. The
    gate must not treat clv_frac_positive as if it had 1000 samples behind it."""
    thin = {"n_bets": 1000, "n_clv": 10, "flat_roi": 0.05, "kelly_roi": 0.04,
            "flat_roi_lb95": 0.02, "kelly_roi_lb95": 0.02,
            "clv_mean": 0.03, "clv_frac_positive": 0.60}
    gate(thin, _empty())
    assert G.market_staking_allowed()["1x2"] is False


def test_poor_clv_coverage_cannot_open_a_market(gate):
    """Even with enough CLV samples, a market where most bets have no closing
    price is not backed by closing evidence and must stay closed."""
    poor = {"n_bets": 1000, "n_clv": 300, "flat_roi": 0.05, "kelly_roi": 0.04,
            "flat_roi_lb95": 0.02, "kelly_roi_lb95": 0.02,
            "clv_mean": 0.03, "clv_frac_positive": 0.60}
    gate(poor, _empty())     # 300/1000 = 30% coverage, below MIN_CLV_COVERAGE
    assert G.market_staking_allowed()["1x2"] is False


# ── per-league gate (adversarial blocker 4) ────────────────────────────────

def _with_leagues(one_x_two_market, leagues_1x2):
    art = _artifact(one_x_two_market, _empty())
    art["simulated_betting_by_league"] = {
        "1x2": {comp: {t: dict(row) for t in ("2%", "4%", "6%")}
                for comp, row in leagues_1x2.items()}}
    return art


def test_open_market_still_blocks_a_clvless_league(gate_path):
    """The core of blocker 4: an EU-driven 1X2/totals pass must not unlock a
    league that has no closing evidence of its own."""
    gate_path.write_text(json.dumps(_with_leagues(
        _pass(), {"Premier League": _pass(400), "Brazil Serie A": _clvless(400)})))
    lg = G.market_league_staking_allowed()
    assert lg[("1x2", "Premier League")] is True
    assert lg[("1x2", "Brazil Serie A")] is False


def test_stake_zeroing_honours_the_per_league_gate(gate_path):
    gate_path.write_text(json.dumps(_with_leagues(
        _pass(), {"Premier League": _pass(400), "Brazil Serie A": _clvless(400)})))
    rows = [
        {"market": "1x2", "competition": "Premier League",
         "kelly_stake": 0.02, "stake_gbp": 2.0},
        {"market": "1x2", "competition": "Brazil Serie A",
         "kelly_stake": 0.02, "stake_gbp": 2.0},
    ]
    E.apply_evidence_gate(rows)
    by = {r["competition"]: r for r in rows}
    assert by["Premier League"]["stake_gbp"] == 2.0, "proven league stakes"
    assert by["Brazil Serie A"]["stake_gbp"] == 0.0, "CLV-less league is zeroed"


def test_no_by_league_section_fails_closed(gate_path):
    """decision_time_v2 promises per-league evidence; omission cannot authorize
    every competition through a pooled market pass."""
    art = _artifact(_pass(), _empty())
    art.pop("simulated_betting_by_league")
    gate_path.write_text(json.dumps(art))
    assert G.market_staking_allowed()["1x2"] is False
    assert G.market_league_staking_allowed() == {}
    rows = [{"market": "1x2", "competition": "Anything",
             "kelly_stake": 0.02, "stake_gbp": 2.0}]
    E.apply_evidence_gate(rows)
    assert rows[0]["stake_gbp"] == 0.0


def test_closed_market_with_nonzero_stake_raises(gate, monkeypatch):
    """The money-safety invariant must fire for a per-market violation too."""
    gate(_empty(), _empty())          # nothing open
    monkeypatch.setattr("club_soccer.evidence_gate.BACKTEST_JSON", G.BACKTEST_JSON)

    # A row that refuses to be zeroed (simulate a downstream bug) must crash.
    class Sticky(dict):
        def __setitem__(self, k, v):
            if k in ("kelly_stake", "stake_gbp"):
                return           # ignore zeroing
            super().__setitem__(k, v)

    row = Sticky(market="1x2", kelly_stake=0.02, stake_gbp=2.0)
    with pytest.raises(RuntimeError, match="nonzero stake"):
        E.apply_evidence_gate([row])
