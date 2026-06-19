#!/usr/bin/env python3
"""Tests for the V5 adaptive intelligence layer."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance import drift, live, portfolio, registry, research, review, scenario, store


def _backup(path: Path):
    return path.read_bytes() if path.exists() else None


def _restore(path: Path, data):
    if data is None:
        if path.exists():
            path.unlink()
    else:
        path.write_bytes(data)


def test_model_registry_promotion_gate_rejects_failed_challenger():
    old = _backup(store.REGISTRY)
    try:
        champ = registry.register_model(
            "worldcup", "1x2", metrics={"logloss": 1.0},
            feature_schema=["elo_diff"], role="champion")
        challenger = registry.register_model(
            "worldcup", "1x2", metrics={"logloss": 1.1},
            feature_schema=["elo_diff", "market_move"])
        rejected = registry.promote(
            "worldcup", "1x2", challenger["version"],
            {"passed": False, "reason": "heldout_regression"})
        assert rejected["status"] == "rejected"
        assert registry.champion("worldcup", "1x2")["version"] == champ["version"]
    finally:
        _restore(store.REGISTRY, old)


def test_feature_snapshot_and_recommendation_are_auditable():
    backups = {p: _backup(p) for p in (store.REGISTRY, store.FEATURE_SNAPSHOTS,
                                       store.RECOMMENDATIONS)}
    try:
        model = registry.register_model("worldcup", "1x2", role="champion")
        feat = registry.feature_snapshot(
            "worldcup", "2026-06-11|brazil|morocco", "2026-06-11",
            {"elo_diff": 25.0, "p_model_h": 0.42},
            schema_version=1, source="test")
        rec = registry.record_recommendation({
            "engine": "worldcup", "sport": "soccer",
            "event_id": "2026-06-11|brazil|morocco",
            "match_date": "2026-06-11", "home": "Brazil", "away": "Morocco",
            "market": "1x2", "side": "home", "odds": 2.5,
            "p_model": 0.42, "p_market": 0.40, "edge": 0.02,
            "stake_gbp": 5.0, "feature_version": feat["feature_version"],
        })
        assert rec["model_version"] == model["version"]
        assert rec["feature_version"] == feat["feature_version"]
        df = registry.recommendations("worldcup")
        assert rec["recommendation_id"] in set(df["recommendation_id"])
    finally:
        for p, data in backups.items():
            _restore(p, data)


def test_drift_report_handles_thin_samples_without_false_promotion():
    old = _backup(store.RECOMMENDATIONS)
    try:
        store.append_csv(store.RECOMMENDATIONS, [{
            "recommendation_id": "r1", "created_at": store.now_iso(),
            "engine": "worldcup", "sport": "soccer", "event_id": "e",
            "match_date": "2026-06-11", "home": "A", "away": "B",
            "market": "1x2", "side": "home", "line": "", "bet": "A",
            "odds": 2.0, "p_model": 0.55, "p_market": 0.50,
            "edge": 0.05, "ev_per_unit": 0.10, "stake_gbp": 2.0,
            "model_version": "m", "feature_version": "f", "source": "test",
            "status": "recommended", "reason_codes": "[]",
        }], registry.RECOMMENDATION_COLS)
        rep = drift.recommendation_drift("worldcup", min_n=5)
        assert rep["confidence"] == "low"
        assert "thin_recommendation_sample" in rep["alerts"]
    finally:
        _restore(store.RECOMMENDATIONS, old)
        if store.DRIFT_REPORT.exists():
            store.DRIFT_REPORT.unlink()


def test_portfolio_optimizer_respects_hard_caps():
    rows = [
        {"engine": "worldcup", "event_id": "e1", "market": "1x2",
         "home": "Brazil", "away": "Morocco", "stake_gbp": 100.0,
         "ev_per_unit": 0.20},
        {"engine": "worldcup", "event_id": "e1", "market": "btts",
         "home": "Brazil", "away": "Morocco", "stake_gbp": 100.0,
         "ev_per_unit": 0.10},
    ]
    res = portfolio.optimize(rows, bankroll=1000.0, caps={"per_event": 0.03})
    assert sum(r["allocated_stake_gbp"] for r in res["rows"]) <= 30.0
    assert any(r["risk_limited"] for r in res["rows"])


def test_scenario_lab_is_synthetic_and_deterministic():
    res = scenario.worldcup_line_lab("Brazil", "Morocco", "2026-06-11",
                                    home_elo_delta=-50)
    assert res["status"] == "synthetic"
    assert res["delta"]["home"] < 0
    assert "production feature snapshots are untouched" in res["note"]


def test_live_model_passes_on_missing_timestamp():
    res = live.soccer_live_1x2("Brazil", "Morocco", "2026-06-11",
                               minute=20, home_score=0, away_score=0)
    assert res["status"] == "pass"
    assert res["reason"] == "missing_live_state_timestamp"


def test_human_review_is_analytics_only():
    backups = {p: _backup(p) for p in (store.RECOMMENDATIONS, store.REVIEWS)}
    try:
        rec = registry.record_recommendation({
            "engine": "worldcup", "event_id": "e2", "market": "1x2",
            "side": "home", "edge": 0.04, "odds": 2.1,
        })
        review.add_review(rec["recommendation_id"], "rejected", tags=["team_news"])
        a = review.analytics()
        assert a["states"]["rejected"] == 1
        assert a["training_use"] == "excluded_by_default"
        backlog = research.generate_backlog()
        assert any(i["source"] == "human_review" for i in backlog["items"])
    finally:
        for p, data in backups.items():
            _restore(p, data)
        if store.RESEARCH_BACKLOG.exists():
            store.RESEARCH_BACKLOG.unlink()


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} V5 tests passed.")


if __name__ == "__main__":
    _run_all()
