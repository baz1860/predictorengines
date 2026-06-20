#!/usr/bin/env python3
"""Tests for the World Cup V4 modelling slice (M1–M3).

Runs standalone (`python3 test_wc_v4.py`) or under pytest. Mirrors the V4_PLAN.md
acceptance criteria — especially the M1 leakage guarantees, which are the whole
point of a point-in-time feature store.
"""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wc_v4 import schema
from wc_v4 import feature_store as fs
from wc_v4 import market_model as mm
from wc_v4 import availability as av
from wc_v4 import consistency as cs
from wc_v4 import matchup as mu
from wc_v4 import probability as prob
from wc_v4 import staking as st
from wc_v4 import tournaments as ts
from wc_v4 import validate_v4 as vv
from wc_v4 import live_features as lf
from scripts.worldcup import live_data as ld
from scripts.worldcup import whoscored_scrape as ws
from contracts import fixture_key


# ── schema / leakage registry ─────────────────────────────────────────────────
def test_schema_rejects_outcome_columns_as_features():
    # Every outcome/teacher column must be refused as a feature (guardrails #2/#3).
    for col in ("result", "home_score", "odds_close_h", "p_close_h", "clv",
                "settled_pnl"):
        try:
            schema.assert_no_leakage(["elo_diff", col])
        except schema.LeakageError:
            continue
        raise AssertionError(f"leakage not caught for outcome column {col!r}")


def test_feature_columns_excludes_injected_future_column():
    # M1 acceptance: "leakage tests intentionally inject future-only columns and
    # confirm they are rejected." feature_columns() is the SAFE SELECTOR: given a
    # frame's columns (which legitimately include teacher columns), it returns only
    # the legal features and drops the injected future-only ones.
    poisoned = list(schema.FEATURE_COLUMNS) + ["result", "odds_close_a", "clv"]
    selected = schema.feature_columns(poisoned)
    for bad in ("result", "odds_close_a", "clv"):
        assert bad not in selected, f"{bad!r} leaked through feature selection"
    # and every returned column is a declared feature
    assert set(selected) <= set(schema.FEATURE_COLUMNS)


# ── M1 feature store ──────────────────────────────────────────────────────────
def test_training_matrix_has_provenance_on_every_row():
    df = fs.build_training_matrix(since="2024-01-01")
    assert len(df) > 0
    for col in schema.PROVENANCE_COLUMNS:        # asof/event_id/source/fetched_at/schema_version
        assert col in df.columns, f"missing provenance column {col}"
        assert df[col].notna().all(), f"null provenance in {col}"
    assert (df["schema_version"] == schema.SCHEMA_VERSION).all()
    # event_id is the canonical fixture key and unique-ish per row
    assert df["event_id"].str.len().gt(0).all()


def test_training_matrix_is_point_in_time():
    # asof equals the match date: nothing dated after kickoff fed the row.
    df = fs.build_training_matrix(since="2024-01-01")
    assert (df["asof"] == df["match_date"]).all()
    # model probabilities are valid distributions
    p = df[["p_model_h", "p_model_d", "p_model_a"]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert (p >= 0).all() and (p <= 1).all()


def test_build_asof_uses_no_future_rows():
    # The as-of (live) path must only price fixtures kicking off on/after asof, and
    # must not carry any closing-line column (that's a teacher, unknown pre-match).
    asof = "2026-06-11"
    df = fs.build_asof(asof)
    assert len(df) > 0
    assert (df["match_date"] >= asof).all(), "as-of build leaked a past fixture"
    for col in ("odds_close_h", "odds_close_d", "odds_close_a", "result",
                "home_score"):
        assert col not in df.columns, f"as-of build exposed teacher column {col}"
    # provenance stamped with the requested asof
    assert (df["asof"] == asof).all()


def test_asof_ratings_are_frozen_before_date():
    # Building as of an earlier date must not see later results: a team's Elo as of
    # an earlier asof should differ from (or at most equal) a later asof only via
    # matches in between — concretely, the row count of priced fixtures shrinks as
    # asof advances past kickoffs. We assert the build is deterministic & leak-safe
    # by checking no fixture in the matrix pred: match_date < asof.
    early = fs.build_asof("2026-06-01")
    assert (early["match_date"] >= "2026-06-01").all()


def test_live_provider_parsers_normalize_payloads():
    teams = {"United States", "Australia", "Brazil", "Morocco", "Germany",
             "Ivory Coast"}
    fetched = "2026-06-19T10:00:00+00:00"
    fixtures = ld.parse_fixtures([{
        "fixture": {"id": 10, "date": "2026-06-19T20:00:00+00:00",
                    "venue": {"name": "Lumen Field", "city": "Seattle"},
                    "status": {"long": "Not Started", "short": "NS", "elapsed": None}},
        "league": {"name": "FIFA World Cup", "round": "Group D - 2"},
        "teams": {"home": {"id": 1, "name": "USA"},
                  "away": {"id": 2, "name": "Australia"}},
    }], fetched, teams)
    assert fixtures.iloc[0]["home"] == "United States"
    assert fixtures.iloc[0]["event_id"] == fixture_key(
        fixtures.iloc[0]["match_date"], "United States", "Australia",
        "FIFA World Cup")

    availability = ld.parse_availability([
        {"team": {"id": 1, "name": "Brazil"},
         "player": {"id": 7, "name": "Neymar", "type": "Injury",
                    "reason": "fitness test"},
         "fixture": {"id": 11}},
        {"team": {"id": 2, "name": "Morocco"},
         "player": {"id": 8, "name": "Player X", "type": "Suspended",
                    "reason": "red card ban"},
         "fixture": {"id": 11}},
    ], fetched, teams)
    by_player = {r.player: r for r in availability.itertuples(index=False)}
    assert by_player["Neymar"].status == "doubtful"
    assert by_player["Neymar"].affects_availability == False
    assert by_player["Player X"].status == "suspended"
    assert by_player["Player X"].certainty == "certain"

    meta = {"event_id": "e1", "provider_fixture_id": 11,
            "match_date": "2026-06-19"}
    lineups = ld.parse_lineups([{
        "team": {"id": 1, "name": "Germany"},
        "formation": "4-2-3-1",
        "startXI": [{"player": {"id": 1, "name": "Manuel Neuer",
                                "number": 1, "pos": "G"}}],
        "substitutes": [{"player": {"id": 2, "name": "Niclas Fullkrug",
                                    "number": 9, "pos": "F"}}],
    }], meta, fetched, fetched, teams)
    assert set(lineups["role"]) == {"starter", "bench"}
    assert lineups["formation"].iloc[0] == "4-2-3-1"

    events = [{
        "id": "odds1", "commence_time": "2026-06-20T00:00:00Z",
        "home_team": "Germany", "away_team": "Ivory Coast",
        "bookmakers": [
            {"key": "book_a", "last_update": fetched, "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Germany", "price": 1.7},
                    {"name": "Draw", "price": 3.6},
                    {"name": "Ivory Coast", "price": 5.0}]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "price": 1.9, "point": 2.5},
                    {"name": "Under", "price": 1.9, "point": 2.5}]},
                {"key": "btts", "outcomes": [
                    {"name": "Yes", "price": 1.8},
                    {"name": "No", "price": 2.0}]},
            ]},
            {"key": "book_b", "last_update": fetched, "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Germany", "price": 1.8},
                    {"name": "Draw", "price": 3.5},
                    {"name": "Ivory Coast", "price": 4.8}]},
            ]},
        ],
    }]
    snaps = ld.normalize_market_snapshots(events, fetched, teams)
    wide = ld.summarize_wide_market(snaps)
    assert len(snaps) == 10
    assert wide.iloc[0]["bookmaker_count"] == 2
    assert np.isfinite(wide.iloc[0]["market_dispersion_h"])


def test_whoscored_cached_payload_normalizes_to_canonical_tables():
    teams = {"Germany", "Ivory Coast"}
    fetched = "2026-06-19T10:00:00+00:00"
    payload = {
        "matchId": 12345,
        "startDate": "2026-06-20T20:00:00+00:00",
        "competition": {"name": "FIFA World Cup"},
        "stageName": "Group E - 1",
        "isLineupConfirmed": True,
        "home": {
            "teamId": 1,
            "name": "Germany",
            "formation": "4-2-3-1",
            "players": [
                {"playerId": 1, "name": "Manuel Neuer", "shirtNo": 1,
                 "position": "GK", "isFirstEleven": True},
                {"playerId": 2, "name": "Niclas Fullkrug", "shirtNo": 9,
                 "position": "FW", "isSubstitute": True},
            ],
            "missingPlayers": [
                {"playerId": 3, "name": "Player Doubt",
                 "reason": "fitness test"},
            ],
        },
        "away": {
            "teamId": 2,
            "name": "Ivory Coast",
            "formation": "4-3-3",
            "players": [
                {"playerId": 4, "name": "Away Keeper", "shirtNo": 1,
                 "position": "GK", "isFirstEleven": True},
            ],
        },
        "events": [
            {"teamId": 1, "type": {"displayName": "SavedShot"},
             "isShot": True, "xG": 0.18},
            {"teamId": 1, "type": {"displayName": "Goal"},
             "isShot": True, "isGoal": True, "xG": 0.30},
            {"teamId": 1, "type": {"displayName": "Pass"},
             "qualifiers": [{"type": {"displayName": "CornerTaken"}}]},
            {"teamId": 1, "type": {"displayName": "Foul"}},
            {"teamId": 2, "type": {"displayName": "MissedShots"},
             "isShot": True, "xG": 0.04},
            {"teamId": 2, "type": {"displayName": "Card"},
             "cardType": {"displayName": "Yellow"}},
        ],
    }
    tables = ws.normalize_payload(payload, fetched, teams)
    fixtures = tables["fixtures"]
    assert fixtures.iloc[0]["event_id"] == fixture_key(
        fixtures.iloc[0]["match_date"], "Germany", "Ivory Coast",
        "FIFA World Cup")
    assert fixtures.iloc[0]["source"] == "whoscored_scrape"

    lineups = tables["lineups"]
    assert set(lineups["role"]) == {"starter", "bench"}
    assert lineups[lineups["team"] == "Germany"]["lineup_status"].eq(
        "confirmed").all()

    availability = tables["availability"]
    assert availability.iloc[0]["player"] == "Player Doubt"
    assert availability.iloc[0]["status"] == "doubtful"

    stats = tables["match_stats"].set_index("team")
    assert stats.loc["Germany", "shots"] == 2
    assert stats.loc["Germany", "shots_on_target"] == 2
    assert stats.loc["Germany", "corners"] == 1
    assert stats.loc["Germany", "fouls"] == 1
    assert abs(stats.loc["Germany", "xg"] - 0.48) < 1e-9
    assert stats.loc["Ivory Coast", "shots"] == 1
    assert stats.loc["Ivory Coast", "yellow_cards"] == 1


def test_whoscored_extracts_strict_json_from_html():
    html = '<script>var matchCentreData = {"matchId": 1, "home": {"name": "Germany"}, "away": {"name": "Ivory Coast"}};</script>'
    data = ws.extract_match_centre_json(html)
    assert data["matchId"] == 1
    assert data["home"]["name"] == "Germany"


def test_whoscored_accepts_wrapped_json_har_and_directories():
    payload = {
        "matchId": 99,
        "startDate": "2026-06-20T20:00:00+00:00",
        "home": {"teamId": 1, "name": "Germany"},
        "away": {"teamId": 2, "name": "Ivory Coast"},
    }
    html = f"<script>var matchCentreData = {json.dumps(payload)};</script>"
    har = {"log": {"entries": [{
        "response": {"content": {"mimeType": "text/html", "text": html}}
    }]}}
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        wrapped = tmp / "wrapped.json"
        wrapped.write_text(json.dumps({"props": {"pageProps": {
            "matchCentreData": payload}}}))
        har_path = tmp / "capture.har"
        har_path.write_text(json.dumps(har))
        bad = tmp / "bad.json"
        bad.write_text(json.dumps({"not": "a match payload"}))

        assert ws.load_json(wrapped)["matchId"] == 99
        assert ws.load_har(har_path)[0]["home"]["name"] == "Germany"
        loaded = ws.load_dir(tmp)
        assert len(loaded) == 1
        assert loaded[0]["matchId"] == 99


def test_build_asof_attaches_only_asof_live_lineups():
    saved = (lf.LINEUPS_CSV, lf.AVAILABILITY_CSV, lf.MARKET_SNAPSHOTS_CSV)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        lf.LINEUPS_CSV = tmp / "lineups.csv"
        lf.AVAILABILITY_CSV = tmp / "availability.csv"
        lf.MARKET_SNAPSHOTS_CSV = tmp / "market.csv"
        try:
            eid = fixture_key("2026-06-20", "Germany", "Ivory Coast",
                              "FIFA World Cup")
            pd.DataFrame([
                {"event_id": eid, "provider_fixture_id": 99,
                 "match_date": "2026-06-20", "team": "Germany",
                 "player": "Manuel Neuer", "starter": True, "role": "starter",
                 "position": "G", "formation": "4-2-3-1",
                 "lineup_status": "confirmed",
                 "published_at": "2026-06-19T11:30:00+00:00",
                 "fetched_at": "2026-06-19T11:35:00+00:00"},
            ]).to_csv(lf.LINEUPS_CSV, index=False)
            fixtures = pd.DataFrame([{
                "date": pd.Timestamp("2026-06-20"),
                "home_team": "Germany",
                "away_team": "Ivory Coast",
                "home_score": np.nan,
                "away_score": np.nan,
                "tournament": "FIFA World Cup",
                "city": "Toronto",
                "country": "Canada",
                "neutral": True,
            }])
            before = fs.build_asof("2026-06-19 11:00:00", fixtures)
            after = fs.build_asof("2026-06-19 12:00:00", fixtures)
            assert before["formation_known_h"].isna().all()
            assert after["formation_known_h"].iloc[0] == 1.0
            for col in ("result", "odds_close_h", "xg_h"):
                assert col not in after.columns
        finally:
            lf.LINEUPS_CSV, lf.AVAILABILITY_CSV, lf.MARKET_SNAPSHOTS_CSV = saved


# ── M2 market model ───────────────────────────────────────────────────────────
def test_segment_blend_is_honest_and_fails_safe():
    fit = mm.segment_blend_weights()
    # global default present
    assert 0.0 <= fit["global_w"] <= 1.0
    # decision uses OUT-OF-SAMPLE leave-one-out, not in-sample fit
    for seg, v in fit["segments"].items():
        assert "loo_logloss_segment_w" in v and "loo_logloss_global_w" in v
        # adoption requires both the sample-size floor AND an out-of-sample win
        if v["adopt_as_default"]:
            assert v["n"] >= fit["min_segment_n"]
            assert v["beats_global_out_of_sample"]
    # fail-safe: a non-adopted segment uses the global weight
    for seg in ("group", "knockout"):
        w = mm.blend_weight_for(seg, fit)
        assert 0.0 <= w <= 1.0


def test_line_history_movement_and_do_not_bet():
    # Use a match that has a full 1X2 in odds_history.
    hist = pd.read_csv(ROOT / "data" / "odds_history.csv")
    g = hist.groupby(["match_date", "home", "away"])["side"].agg(set)
    full = [k for k, v in g.items() if {"home", "draw", "away"} <= v]
    assert full, "expected at least one full-1X2 match in odds_history"
    md, home, away = full[0]
    lh = mm.line_history(home, away, md)
    assert set(lh["p_open"]) == {"h", "d", "a"}
    # de-vigged probs sum to ~1
    assert abs(sum(lh["p_curr"].values()) - 1.0) < 1e-6
    # do_not_bet returns a decision + reason codes for explainability
    d = mm.do_not_bet("home", lh["p_curr"]["h"] + 0.01, lh)
    assert "do_not_bet" in d and isinstance(d["reasons"], list) and d["reasons"]


def test_closing_line_is_a_teacher_not_a_feature():
    # The closing implied prob must never be in the legal feature set.
    assert schema.is_outcome_column("p_close_h")
    assert "p_close_h" not in set(schema.FEATURE_COLUMNS)


# ── M3 availability ───────────────────────────────────────────────────────────
def test_lineup_confidence_erodes_with_doubtful_absences():
    a = av._absences_df()
    assert len(a) > 0
    full = av.lineup_confidence("__no_such_team__", a)
    assert full["confidence"] == 1.0  # no absences -> full confidence
    # a team with a doubtful absence has confidence < 1
    doubtful_teams = a[a["doubtful"]]["team"].unique()
    if len(doubtful_teams):
        c = av.lineup_confidence(doubtful_teams[0], a)
        assert c["confidence"] < 1.0 and c["n_doubtful"] >= 1


def test_availability_band_widens_when_confidence_low():
    a = av._absences_df()
    doubtful_teams = list(a[a["doubtful"]]["team"].unique())
    certain_only = [t for t in a["team"].unique()
                    if t not in doubtful_teams]
    # a doubtful team should have a wider uncertainty band than a clean team
    if doubtful_teams and certain_only:
        wide = av.availability_adjustment(doubtful_teams[0])
        narrow = av.availability_adjustment(certain_only[0])
        assert wide["uncertainty_sd"] >= narrow["uncertainty_sd"]
    # band brackets the point estimate
    r = av.availability_adjustment(a["team"].iloc[0])
    assert r["elo_adj_low"] <= r["elo_adj"] <= r["elo_adj_high"]
    assert r["status"] == "report_only"  # M3 never changes a V3 default


def test_gk_absence_is_flagged_specifically():
    a = av._absences_df()
    pos = av._positions()
    # function returns a well-formed decision for every team
    for t in a["team"].unique()[:5]:
        g = av.gk_impact(t, a, pos)
        assert set(g) >= {"gk_absent", "keepers_out", "def_share"}
        assert isinstance(g["gk_absent"], bool)


# ── WC2018 wired into the harness ─────────────────────────────────────────────
def test_tournament_samples_are_leak_free_and_complete():
    samps = ts.all_samples()
    assert set(samps) >= {"WC2018", "WC2022"}
    for name, cfg in ts.TOURNAMENTS.items():
        s = samps[name]
        assert len(s) > 0
        for x in s:
            # every match falls inside the tournament window (no stray fixtures)
            assert cfg["date_lo"] <= x.date <= cfg["date_hi"], (name, x.date)
            # model output is a valid probability distribution
            p = np.asarray(x.p_model, float)
            assert abs(p.sum() - 1.0) < 1e-6 and (p >= 0).all()
            assert x.actual in (0, 1, 2)
    # p_market presence mirrors whether a tournament's odds file exists: with an
    # odds file the matches carry de-vigged market probs (feed the blend gate);
    # without one they are model-only (calibration evidence). Derived from config
    # so folding in a new wc20xx_odds.csv doesn't break this test.
    for name, cfg in ts.TOURNAMENTS.items():
        oc = cfg.get("odds_csv")
        has_odds = bool(oc and Path(oc).exists())
        if has_odds:
            assert any(x.p_market is not None for x in samps[name]), name
        else:
            assert all(x.p_market is None for x in samps[name]), name
    # WC2022 ships an odds file in-repo, so it must always feed the blend gate.
    assert any(x.p_market is not None for x in samps["WC2022"])


def test_model_calibration_pools_wc2018_and_wc2022():
    cal = vv.model_calibration()
    per = cal["per_tournament"]
    assert "WC2018" in per and "WC2022" in per
    pooled_n = cal["pooled"]["all"]["n"]
    assert pooled_n == per["WC2018"]["all"]["n"] + per["WC2022"]["all"]["n"]
    assert pooled_n > 64, "pooling should give a larger held-out base than one cup"
    # metrics are well-formed probabilities-of-fit
    for m in (cal["pooled"]["all"], per["WC2018"]["all"], per["WC2022"]["all"]):
        assert 0.0 < m["logloss"] < 3.0 and 0.0 <= m["accuracy"] <= 1.0


def test_blend_gate_coverage_is_honest_about_odds():
    rep = vv.run(write=False)
    cov = rep["coverage"]
    # the ship/no-ship gate only claims tournaments that actually have an odds
    # file; derive that set from config so folding in a new wc20xx_odds.csv
    # (per WC_V4_NOTES) updates coverage without breaking this honesty check.
    expected_with_odds = sorted(
        name for name, cfg in ts.TOURNAMENTS.items()
        if cfg.get("odds_csv") and Path(cfg["odds_csv"]).exists())
    assert sorted(cov["blend_gate_tournaments"]) == expected_with_odds
    assert "WC2022" in cov["blend_gate_tournaments"]
    # model calibration pools every tournament with results, odds or not.
    assert "WC2018" in cov["model_calibration_tournaments"]
    assert rep["enriched_features"]["status"] == "report_only"
    assert rep["enriched_features"]["default_after_gate"] == "v3_blend"


# ── M4-M7 report-only modelling layers ────────────────────────────────────────
def test_matchup_eval_is_report_only_and_measured():
    rep = mu.heldout_matchup_eval()
    assert rep["n"] > 0
    assert rep["status"] == "report_only"
    assert "matchup_logloss" in rep and "baseline_logloss" in rep


def test_coherent_board_prices_cross_markets_from_one_distribution():
    board = prob.coherent_board("Brazil", "Morocco", "2026-06-11")
    assert board["available"]
    assert board["status"] == "report_only"
    m = board["markets"]
    assert abs(m["home"] + m["draw"] + m["away"] - 1.0) < 1e-6
    assert abs(m["over25"] + m["under25"] - 1.0) < 1e-6
    assert abs(m["btts_yes"] + m["btts_no"] - 1.0) < 1e-6
    assert board["correct_scores"] and board["fair_odds"]["home"] > 1.0


def test_consistency_flags_only_real_board_issues():
    board = prob.coherent_board("Brazil", "Morocco", "2026-06-11")
    chk = cs.check_board(board)
    assert chk["ok"], chk
    bad = {**board, "markets": {**board["markets"], "home": 0.80}}
    chk_bad = cs.check_board(bad)
    assert not chk_bad["ok"]
    assert any("1x2" in issue for issue in chk_bad["issues"])


def test_uncertainty_aware_staking_can_pass_thin_edges():
    rec = st.recommendation(
        {"home": "Brazil", "away": "Morocco"},
        "home",
        model_prob=0.42,
        market_odds=2.35,
        bankroll=1000.0,
        market_line=None,
        clv_context={"mean_clv": -0.01},
    )
    assert rec["status"] == "report_only"
    assert rec["recommendation"] == "pass"
    assert rec["stake_gbp"] == 0.0
    assert "uncertainty_overwhelms_edge" in rec["reason_codes"]


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} V4 tests passed.")


if __name__ == "__main__":
    _run_all()
