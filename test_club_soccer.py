#!/usr/bin/env python3
"""Regression tests for the Club Soccer engine.

Run: python3 test_club_soccer.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# club_soccer is a real package (Phase 4) — import its modules package-qualified.
from club_soccer import competitions as C
from club_soccer import edge as E
from club_soccer import engine as ENGINE
from club_soccer import model as M
from api_keys import get_key
from app.engines.club_soccer import ClubSoccerAdapter

_fails: list[str] = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def test_registry():
    print("1. competition registry")
    expected = {
        "Premier League", "Championship", "League One", "League Two",
        "Scottish Premiership", "Scottish Championship", "Scottish League One",
        "Scottish League Two", "Bundesliga", "Serie A", "Ligue 1", "La Liga",
        "Champions League", "Europa League", "Conference League", "UEFA Super Cup",
        "FA Cup", "EFL Cup", "Scottish Cup", "Scottish League Cup",
        "DFB-Pokal", "Coppa Italia", "Coupe de France", "Copa del Rey",
    }
    names = set(C.names())
    check("contains every requested competition", expected <= names)
    check("public rows include API-Football IDs", all(r["api_football_id"] for r in C.public_rows()))


def test_api_key_lookup():
    print("2. API key lookup")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "api_keys.json"
        p.write_text(json.dumps({"api-football": "file-key", "api_football": "alias-key"}))
        old = os.environ.get("API_FOOTBALL_KEY")
        os.environ["API_FOOTBALL_KEY"] = "env-key"
        try:
            check("environment wins", get_key("api-football", env="API_FOOTBALL_KEY", path=p) == "env-key")
        finally:
            if old is None:
                os.environ.pop("API_FOOTBALL_KEY", None)
            else:
                os.environ["API_FOOTBALL_KEY"] = old
        check("file lookup works", get_key("api-football", path=p) in {"file-key", "alias-key"})
        check("alias lookup works", get_key("football", path=p) in {"file-key", "alias-key"})


def test_model_math():
    print("3. model probabilities")
    params = M.fit()
    pred = M.predict("Arsenal", "Chelsea", "Premier League", params=params)
    p = pred["probs"]
    check("1X2 probabilities sum to one", abs(p["home"] + p["draw"] + p["away"] - 1.0) < 0.002)
    check("totals probabilities sum to one", abs(p["over25"] + p["under25"] - 1.0) < 0.002)
    check("BTTS probabilities sum to one", abs(p["btts_yes"] + p["btts_no"] - 1.0) < 0.002)
    check("score matrix normalizes", abs(float(pred["matrix"].sum()) - 1.0) < 1e-9)
    parts = M.component_matrices(params, "Arsenal", "Chelsea", "Premier League", False)
    check("shot-pressure component is available", "xpress" in parts)
    check("component matrices normalize",
          all(abs(float(mx.sum()) - 1.0) < 1e-9 for mx in parts.values()))
    xp = M.predict("Arsenal", "Chelsea", "Premier League", model="xpress", params=params)
    px = xp["probs"]
    check("xpress probabilities sum to one",
          abs(px["home"] + px["draw"] + px["away"] - 1.0) < 0.002)
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "ensemble_weights.json"
        f.write_text(json.dumps({"weights": {"goals": 2, "elo": 1, "xpress": 1}}))
        w = M.load_ensemble_weights(f)
        check("ensemble weights normalize from artifact", abs(sum(w.values()) - 1.0) < 1e-12)
        check("missing ensemble components default to zero", w["xg"] == 0.0 and w["xgf"] == 0.0)
        check("xpress artifact weight loaded", abs(w["xpress"] - 0.25) < 1e-12)
    try:
        M.predict("Not A Club", "Chelsea", "Premier League", params=params)
        check("unknown team raises", False)
    except ValueError as e:
        check("unknown team raises", "Unknown team" in str(e))


def test_edge_and_settlement():
    print("4. edge and settlement")
    check("1X2 de-vig sums to one", abs(float(E.devig([2.2, 3.3, 3.1]).sum()) - 1.0) < 1e-12)
    odds = E.load_odds()
    rows = E.rows_from_odds(odds, bankroll=100)
    markets = {r["market"] for r in rows}
    check("edge covers 1X2 totals BTTS", {"1x2", "total", "btts"} <= markets)
    check("API odds mapper handles home team names",
          E._map_api_bet("Match Winner", "Arsenal", "2.20", "Arsenal", "Chelsea") == ("1x2", "home", "", 2.2))
    check("API odds mapper handles over 2.5",
          E._map_api_bet("Goals Over/Under", "Over 2.5", "1.90") == ("total", "over", 2.5, 1.9))
    check("API odds mapper handles BTTS",
          E._map_api_bet("Both Teams Score", "Yes", "1.80") == ("btts", "yes", "", 1.8))
    try:
        json.dumps({"rows": rows}, allow_nan=False)
        check("edge rows are strict JSON safe", True)
    except ValueError:
        check("edge rows are strict JSON safe", False)
    check("home win grades won", E.grade("home", "1x2", "", 2, 1) == "won")
    check("draw grades won", E.grade("draw", "1x2", "", 1, 1) == "won")
    check("away win grades won", E.grade("away", "1x2", "", 0, 2) == "won")
    check("over grades won", E.grade("over", "total", 2.5, 2, 1) == "won")
    check("under grades won", E.grade("under", "total", 2.5, 1, 0) == "won")
    check("BTTS grades won", E.grade("yes", "btts", "", 2, 1) == "won")
    check("BTTS no grades won", E.grade("no", "btts", "", 2, 0) == "won")


def test_runner_and_adapter():
    print("5. in-process engine and adapter")
    adapter = ClubSoccerAdapter()
    schema = adapter.predict_schema()
    edge_schema = adapter.edge_schema()
    check("schema exposes filters", bool(schema.get("filters")))
    check("schema exposes date range filters", {"date_from", "date_to"} <= {f["id"] for f in schema.get("filters", [])})
    check("edge schema exposes The Odds API fallback",
          "the-odds-api" in {s["id"] for s in edge_schema.get("odds_sources", [])})
    pred = adapter.predict({"team1": "Arsenal", "team2": "Chelsea", "competition": "Premier League"})
    check("adapter predict returns outcomes", len(pred.get("outcomes", [])) == 3)
    edge = adapter.edge({"odds_source": "manual", "model": "ensemble"})
    check("adapter edge returns columns", bool(edge.get("columns")))
    filtered = adapter.edge({"odds_source": "manual", "model": "ensemble", "date_from": "2026-06-21"})
    check("adapter edge honors date_from", all(r["date"] >= "2026-06-21" for r in filtered.get("rows", [])))
    # In-process engine command path (Phase 4 — replaces the old subprocess runner).
    from app.engines._inproc import run_inprocess
    data = run_inprocess(ENGINE.COMMANDS, "schema")
    check("in-process schema returns names", "names" in data and "error" not in data)
    try:
        run_inprocess(ENGINE.COMMANDS, "rm -rf /")
        check("unknown command rejected", False)
    except ValueError:
        check("unknown command rejected", True)


def test_health_and_repair():
    print("6. data health + repair guard")
    from club_soccer import health as H
    from club_soccer import fetch as F

    report = H.run_checks()
    check("future_ft_rows is 0 on shipped data", report["future_ft_rows"] == 0)
    check("duplicate_fixture_ids is 0 on shipped data", report["duplicate_fixture_ids"] == 0)
    check("health report ok flag set", report["ok"] is True)

    future_date = "2099-01-01"
    finished_event = {
        "id": 1, "home_team": "A", "away_team": "B", "status": "finished",
        "event_date": f"{future_date}T00:00:00Z",
    }
    row = F._bsd_to_fixture_row(finished_event, "Premier League", 39, "England", "league")
    check("_bsd_to_fixture_row drops future-dated finished event", row is None)

    upcoming_event = dict(finished_event, status="notstarted")
    row2 = F._bsd_to_fixture_row(upcoming_event, "Premier League", 39, "England", "league")
    check("_bsd_to_fixture_row keeps future-dated unplayed event", row2 is not None
          and row2["date"] == future_date)

    # Regression: comp_from_bsd_league must not cross-match unrelated leagues
    # that merely contain a competition name as a substring (e.g. "USL
    # Championship" vs "Championship", "CAF Champions League" vs "Champions
    # League") — this silently blended other continents' results in.
    check("USL Championship does not match England Championship",
          C.comp_from_bsd_league("USL Championship") is None)
    check("CAF Champions League does not match UEFA Champions League",
          C.comp_from_bsd_league("CAF Champions League") is None)
    check("Brasileirão Serie A does not match Italy Serie A",
          C.comp_from_bsd_league("Brasileirão Serie A") is None)
    check("exact league name still matches",
          C.comp_from_bsd_league("Championship") is not None
          and C.comp_from_bsd_league("Championship").name == "Championship")


def test_match_identity_reconciliation():
    print("6b. canonical match identity reconciliation")
    from club_soccer.identities import (
        conflicting_score_identity_count,
        dedupe_fixtures,
        duplicate_identity_count,
    )

    rows = pd.DataFrame([
        {"fixture_id": "fd-1", "date": "2026-08-15", "competition": "Premier League",
         "home": "Arsenal", "away": "Chelsea", "home_goals": 2, "away_goals": 1,
         "home_xg": np.nan, "away_xg": np.nan, "xg_source": "", "source": "football-data"},
        {"fixture_id": "bsd-1", "date": "2026-08-15", "competition": "Premier League",
         "home": " Arsenal ", "away": "Chelsea", "home_goals": 2, "away_goals": 1,
         "home_xg": 1.9, "away_xg": 0.8, "xg_source": "bsd", "source": "bsd"},
    ])
    check("duplicate identity is detected", duplicate_identity_count(rows) == 1)
    check("duplicate scores are not conflicting", conflicting_score_identity_count(rows) == 0)
    clean = dedupe_fixtures(rows)
    check("provider duplicates collapse to one match", len(clean) == 1)
    check("rich BSD row survives reconciliation",
          clean.iloc[0]["fixture_id"] == "bsd-1"
          and float(clean.iloc[0]["home_xg"]) == 1.9)


def test_feature_store_pit():
    print("7. feature store point-in-time")
    from club_soccer import feature_store as FS
    from club_soccer import schema as S

    played = pd.DataFrame([
        {"fixture_id": 1, "date": pd.Timestamp("2026-01-01"), "home": "A", "away": "X"},
        {"fixture_id": 2, "date": pd.Timestamp("2026-01-05"), "home": "A", "away": "Y"},
        {"fixture_id": 3, "date": pd.Timestamp("2026-01-08"), "home": "A", "away": "Z"},
    ])
    sched = FS._schedule_features(played)
    row8 = sched[sched["fixture_id"] == 3].iloc[0]
    check("d8 row rest_days_h counts from d5 only (3 days)", row8["rest_days_h"] == 3.0)
    check("d8 row matches_30d_h sees d1 and d5 only (2 prior)", row8["matches_30d_h"] == 2)
    check("d8 row matches_7d_h sees d1 (exactly 7d ago, inclusive) and d5",
          row8["matches_7d_h"] == 2)
    row1 = sched[sched["fixture_id"] == 1].iloc[0]
    check("d1 row has no prior match (rest_days_h is NaN)", pd.isna(row1["rest_days_h"]))

    poisoned = list(S.FEATURE_COLUMNS) + ["home_goals", "result"]
    safe = S.feature_columns(poisoned)
    check("OUTCOME columns excluded from schema.feature_columns()",
          "home_goals" not in safe and "result" not in safe)
    try:
        S.assert_no_leakage(["elo_h", "p_close_h"])
        check("assert_no_leakage rejects an outcome column", False)
    except S.LeakageError:
        check("assert_no_leakage rejects an outcome column", True)


def test_fdcouk_alias_coverage():
    print("8. fd.co.uk odds alias coverage")
    from club_soccer import fetch_fdcouk as FD
    if not FD.CACHE.exists() or not any(FD.CACHE.glob("*.csv")):
        print("  SKIP  no fdcouk_cache present (offline / not yet fetched)")
        return
    df = FD.build(refresh_current_only=False, verbose=False)  # cache-only, no network
    fx = M.load_fixtures()
    played = fx.dropna(subset=["home_goals", "away_goals"]).copy()
    played["date"] = played["date"].dt.strftime("%Y-%m-%d")
    ok = True
    for comp, grp in df.groupby("competition"):
        p = played[played["competition"] == comp]
        if p.empty:
            continue
        # BSD now contains additional matches beyond the football-data.co.uk
        # market universe. Test that every market row still joins to our
        # canonical fixture table; do not fail merely because BSD has a
        # legitimate fixture for which fd.co.uk has no odds row.
        merged = grp.merge(p, left_on=["match_date", "home", "away"],
                           right_on=["date", "home", "away"],
                         how="left", indicator=True)
        cov = float((merged["_merge"] == "both").mean())
        if cov < 0.95:
            ok = False
            print(f"  {comp}: coverage {cov:.1%} < 95%")
    check("fd.co.uk join coverage >= 95% per competition", ok)


def test_minutes_windows():
    print("9. minutes-load windows")
    from club_soccer.minutes import player_minutes_row

    apps = [
        {"date": "2026-01-01", "team": "A", "mins": 90, "xg": 0.1, "xa": 0.0},
        {"date": "2026-01-05", "team": "A", "mins": 90, "xg": 0.2, "xa": 0.1},
        {"date": "2026-01-10", "team": "A", "mins": 60, "xg": 0.0, "xa": 0.0},
    ]
    row = player_minutes_row(apps, "2026-01-10")
    # window is (D-7, D]: D-7 = 2026-01-03, so 01-01 (exactly D-9) falls outside
    check("mins_7d excludes the 01-01 app (outside D-7..D)", row["mins_7d"] == 150.0)
    check("mins_14d includes all three apps", row["mins_14d"] == 240.0)
    check("mins_30d includes all three apps", row["mins_30d"] == 240.0)
    check("mins_season sums apps since 2025-07-01", row["mins_season"] == 240.0)
    check("starts_season counts 3 apps", row["starts_season"] == 3)

    empty_row = player_minutes_row([], "2026-01-10")
    check("player_minutes_row handles no apps", empty_row["mins_7d"] == 0.0)


def test_transfer_reattribution():
    print("10. transfer re-attribution")
    from club_soccer.player_features import PlayerFeatureStore
    from club_soccer import club_squads as CS

    store = PlayerFeatureStore()
    store._data = {
        "v": 2,
        "j smith": {
            "name": "J. Smith", "pos": "FW",
            "apps": [
                {"date": "2026-01-01", "team": "Club A", "mins": 90, "xg": 0.3, "xa": 0.0,
                 "side_confident": True},
                {"date": "2026-02-01", "team": "Club B", "mins": 90, "xg": 0.2, "xa": 0.1,
                 "side_confident": True},
            ],
        },
    }
    store._loaded = True
    squads = CS.squad_asof(store, "2026-02-15", apply_manual=False)
    row = squads[squads["player"] == "J. Smith"]
    check("player with A-then-B apps lands in Club B's squad",
          not row.empty and row.iloc[0]["team"] == "Club B")

    transfers = CS.detect_transfers(store)
    hit = transfers[transfers["player"] == "J. Smith"]
    check("transfer detected from Club A to Club B",
          not hit.empty and hit.iloc[0]["from_team"] == "Club A"
          and hit.iloc[0]["to_team"] == "Club B")


def test_player_quality_pit():
    print("10b. point-in-time player quality")
    from club_soccer import player_quality as PQ

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "player_stats_cache.json"
        apps = [
            {"date": "2026-01-01", "team": "Club A", "mins": 450,
             "xg": 2.25, "metrics": {"total_pass": 200, "accurate_pass": 160}},
            {"date": "2026-04-01", "team": "Club A", "mins": 450,
             "xg": 9.0, "metrics": {"total_pass": 200, "accurate_pass": 200}},
        ]
        path.write_text(json.dumps({"v": 3, "id:1": {
            "name": "Test Forward", "player_id": 1, "pos": "FW", "apps": apps}}))
        store = PQ.PlayerQualityStore(path).load()
        before = store.team_quality("Club A", "2026-03-01")
        after = store.team_quality("Club A", "2026-06-01")
        check("quality snapshot excludes future appearances",
              before["n_players"] == 1 and after["n_players"] == 1
              and after["attack_xg90"] > before["attack_xg90"])
        check("quality match features require sufficient XI coverage",
              not store.match_features("Club A", "Club B", "2026-06-01")["usable"])

    p = PQ.quality_probs(np.array([0.45, 0.25, 0.30]), 0.5, 0.03,
                         {"attack_xg90": 0.2, "pass_pct": 1.0})
    check("quality probability correction stays normalized", abs(float(p.sum()) - 1.0) < 1e-12)


def test_context_apply():
    print("11. context GLM application")
    lam_h, lam_a = 1.4, 1.1
    mat = M.score_matrix(lam_h, lam_a)
    coef = {"rest_diff": 0.02}
    rd = 5.0
    mult_h = np.exp(coef["rest_diff"] * rd)
    mult_a = np.exp(coef["rest_diff"] * (-rd))
    context_adj = {"home": {"mult": mult_h}, "away": {"mult": mult_a}}
    adjusted = M.apply_context_adj(mat, context_adj)
    xg_h0 = sum(i * mat[i, :].sum() for i in range(mat.shape[0]))
    xg_a0 = sum(j * mat[:, j].sum() for j in range(mat.shape[1]))
    xg_h1 = sum(i * adjusted[i, :].sum() for i in range(adjusted.shape[0]))
    xg_a1 = sum(j * adjusted[:, j].sum() for j in range(adjusted.shape[1]))
    # score_matrix's Dixon-Coles low-score correction perturbs the marginal
    # mean slightly away from the raw lambda it's built from (same as
    # apply_player_adj), so this is approximate, not exact.
    check("context_adj raises home lambda by ~exp(+0.02*rd)",
          abs(xg_h1 - xg_h0 * mult_h) < 1e-3)
    check("context_adj lowers away lambda by ~exp(-0.02*rd)",
          abs(xg_a1 - xg_a0 * mult_a) < 1e-3)
    check("opposite-direction shift (home up, away down)", mult_h > 1.0 and mult_a < 1.0)

    empty = M.apply_context_adj(mat, {})
    check("apply_context_adj no-op on empty dict", np.allclose(empty, mat))

    from club_soccer import context as CTX
    hangover = CTX._euro_hangover_flags(pd.DataFrame([
        {"fixture_id": 1, "date": pd.Timestamp("2026-01-01"), "home": "A", "away": "X",
         "competition": "Champions League", "type": "europe"},
        {"fixture_id": 2, "date": pd.Timestamp("2026-01-03"), "home": "A", "away": "Y",
         "competition": "Premier League", "type": "league"},
    ]))
    check("euro_hangover_h fires 2 days after a Europe match",
          hangover[hangover["fixture_id"] == 2].iloc[0]["euro_hangover_h"] == 1)

    tier = CTX._cup_tier_gap(pd.DataFrame([
        {"fixture_id": 10, "date": pd.Timestamp("2026-01-01"), "home": "A", "away": "X",
         "competition": "Premier League", "type": "league"},
        {"fixture_id": 11, "date": pd.Timestamp("2026-01-01"), "home": "Y", "away": "X",
         "competition": "League Two", "type": "league"},
        {"fixture_id": 12, "date": pd.Timestamp("2026-01-08"), "home": "A", "away": "Y",
         "competition": "FA Cup", "type": "cup"},
    ]))
    check("tier_gap = away tier(4) - home tier(1) = 3 for a top-flight-vs-League-Two cup tie",
          tier[tier["fixture_id"] == 12].iloc[0]["tier_gap"] == 3.0)


def test_standings_asof():
    print("12. standings point-in-time")
    from club_soccer import standings as ST

    fx = pd.DataFrame([
        {"fixture_id": 1, "date": pd.Timestamp("2026-01-01"), "season": 2025,
         "competition": "Test League", "type": "league",
         "home": "A", "away": "B", "home_goals": 2, "away_goals": 0},
        {"fixture_id": 2, "date": pd.Timestamp("2026-01-02"), "season": 2025,
         "competition": "Test League", "type": "league",
         "home": "C", "away": "D", "home_goals": 1, "away_goals": 1},
        {"fixture_id": 3, "date": pd.Timestamp("2026-01-08"), "season": 2025,
         "competition": "Test League", "type": "league",
         "home": "A", "away": "C", "home_goals": 3, "away_goals": 0},
    ])
    table = ST.table_asof("Test League", 2025, "2026-01-08", fixtures=fx)
    check("table has 4 teams", len(table) == 4)
    a = table[table["team"] == "A"].iloc[0]
    check("A has 3 points, 1 played (match 3 excluded, asof date not < asof)",
          a["points"] == 3 and a["played"] == 1)
    c = table[table["team"] == "C"].iloc[0]
    check("C has 1 point, 1 played (draw with D only)", c["points"] == 1 and c["played"] == 1)
    check("A tops the table (3 points beats 1 point)", table.iloc[0]["team"] == "A")

    table_later = ST.table_asof("Test League", 2025, "2026-01-09", fixtures=fx)
    a2 = table_later[table_later["team"] == "A"].iloc[0]
    check("match 3 counted once asof moves past it", a2["played"] == 2 and a2["points"] == 6)


def test_weather_features():
    print("13. weather feature formulas")
    from club_soccer import weather as W

    calm = W.features(temp_c=15.0, precip_mm=0.0, wind_kmh=10.0)
    check("calm/mild weather has zero shift on every term",
          all(v == 0.0 for v in calm.values()))

    windy = W.features(temp_c=15.0, precip_mm=0.0, wind_kmh=35.0)
    check("wind_high = max(0, 35-25)/10 = 1.0", windy["wind_high"] == 1.0)

    wet = W.features(temp_c=15.0, precip_mm=20.0, wind_kmh=0.0)
    check("precip caps at min(20,10)/5 = 2.0", wet["precip"] == 2.0)

    cold = W.features(temp_c=-10.0, precip_mm=0.0, wind_kmh=0.0)
    check("temp_cold = max(0, 0-(-10))/5 = 2.0", cold["temp_cold"] == 2.0)

    hot = W.features(temp_c=38.0, precip_mm=0.0, wind_kmh=0.0)
    check("temp_hot = max(0, 38-28)/5 = 2.0", hot["temp_hot"] == 2.0)

    missing = W.missing_venues()
    check("missing_venues() returns a list without crashing", isinstance(missing, list))


def test_snapshot_odds_dedupe():
    print("14. snapshot_odds dedupe + market parsing")
    import tempfile
    from club_soccer import snapshot_odds as SO

    orig_path = SO.ODDS_HISTORY_CSV
    with tempfile.TemporaryDirectory() as tmp:
        SO.ODDS_HISTORY_CSV = Path(tmp) / "odds_history_club.csv"
        try:
            rows1 = [{"snapshot_time": "2026-07-02T09:00:00+00:00", "match_date": "2026-07-10",
                     "competition": "Premier League", "home": "Arsenal", "away": "Chelsea",
                     "market": "1x2", "side": "home", "odds_median": 2.1, "n_books": 10, "disp": 0.01}]
            rows2 = [
                {"snapshot_time": "2026-07-02T10:00:00+00:00", "match_date": "2026-07-10",
                 "competition": "Premier League", "home": "Arsenal", "away": "Chelsea",
                 "market": "1x2", "side": "home", "odds_median": 2.05, "n_books": 11, "disp": 0.011},
                {"snapshot_time": "2026-07-02T16:00:00+00:00", "match_date": "2026-07-10",
                 "competition": "Premier League", "home": "Arsenal", "away": "Chelsea",
                 "market": "1x2", "side": "home", "odds_median": 1.95, "n_books": 12, "disp": 0.012},
            ]
            SO.append_snapshots(rows1)
            out = SO.append_snapshots(rows2)
            # regression: mixed "T"-separator (fresh) vs space-separator
            # (post-CSV-round-trip str(Timestamp)) snapshot_time formats
            # must not silently NaT-and-drop rows during the dedupe pass.
            check("snapshot within the 6h dedupe window is dropped, "
                  "the one outside it is kept (2 rows survive, not 1 or 3)",
                  len(out) == 2)
            check("earliest and latest snapshot both survive",
                  set(out["snapshot_time"]) == {"2026-07-02T09:00:00+00:00", "2026-07-02T16:00:00+00:00"})
        finally:
            SO.ODDS_HISTORY_CSV = orig_path

    m1x2 = SO._market_rows({"HOME": {"bookmakers": {"a": {"decimal_odds": 2.0}, "b": {"decimal_odds": 2.2},
                                                     "c": {"decimal_odds": 1.9}}},
                            "DRAW": {"bookmakers": {"a": {"decimal_odds": 3.4}, "b": {"decimal_odds": 3.3},
                                                     "c": {"decimal_odds": 3.5}}},
                            "AWAY": {"bookmakers": {"a": {"decimal_odds": 3.6}, "b": {"decimal_odds": 3.5},
                                                     "c": {"decimal_odds": 3.7}}}},
                           {"home": "HOME", "draw": "DRAW", "away": "AWAY"})
    check("_market_rows computes a median odds and n_books=3 per side",
          m1x2["home"]["n_books"] == 3 and abs(m1x2["home"]["odds_median"] - 2.0) < 1e-9)
    check("_market_rows computes a dispersion value when >=3 books quote all sides",
          m1x2["home"]["disp"] is not None and m1x2["home"]["disp"] >= 0)


def test_do_not_bet():
    print("15. do_not_bet suppression rules")
    import tempfile
    from club_soccer import market_model as MM
    from club_soccer import snapshot_odds as SO

    orig_path = SO.ODDS_HISTORY_CSV
    with tempfile.TemporaryDirectory() as tmp:
        SO.ODDS_HISTORY_CSV = Path(tmp) / "odds_history_club.csv"
        try:
            rows = [
                # (a) steam: home odds shortened 2.50 -> 2.10, p moves
                # 0.400 -> 0.476, a +0.076 move >= the 0.03 threshold.
                {"snapshot_time": "2026-07-01T09:00:00+00:00", "match_date": "2026-07-10",
                 "competition": "Premier League", "home": "Arsenal", "away": "Chelsea",
                 "market": "1x2", "side": "home", "odds_median": 2.50, "n_books": 10, "disp": 0.02},
                {"snapshot_time": "2026-07-02T09:00:00+00:00", "match_date": "2026-07-10",
                 "competition": "Premier League", "home": "Arsenal", "away": "Chelsea",
                 "market": "1x2", "side": "home", "odds_median": 2.10, "n_books": 10, "disp": 0.02},
                # (b) books unanimous (thin disp), single snapshot -> no steam.
                {"snapshot_time": "2026-07-01T09:00:00+00:00", "match_date": "2026-07-11",
                 "competition": "Premier League", "home": "Liverpool", "away": "Everton",
                 "market": "1x2", "side": "away", "odds_median": 3.00, "n_books": 12, "disp": 0.002},
                # clean row: neither rule should fire.
                {"snapshot_time": "2026-07-01T09:00:00+00:00", "match_date": "2026-07-12",
                 "competition": "Premier League", "home": "Fulham", "away": "Brentford",
                 "market": "1x2", "side": "home", "odds_median": 2.20, "n_books": 10, "disp": 0.02},
            ]
            SO.append_snapshots(rows)

            steam_row = {"home": "Arsenal", "away": "Chelsea", "date": "2026-07-10",
                        "market": "1x2", "side": "home", "edge": 0.10}
            d1 = MM.do_not_bet(steam_row)
            check("steam rule (a) fires when the market moved >=3pts toward our side",
                  d1["suppress"] and "market_moved" in (d1["reason"] or ""))

            thin_row = {"home": "Liverpool", "away": "Everton", "date": "2026-07-11",
                       "market": "1x2", "side": "away", "edge": 0.02}
            d2 = MM.do_not_bet(thin_row)
            check("thin-edge rule (b) fires when books are unanimous and edge < 4%",
                  d2["suppress"] and "books_unanimous" in (d2["reason"] or ""))

            clean_row = {"home": "Fulham", "away": "Brentford", "date": "2026-07-12",
                        "market": "1x2", "side": "home", "edge": 0.10}
            d3 = MM.do_not_bet(clean_row)
            check("no rule fires on a clean row (no steam, thick disp)", not d3["suppress"])
        finally:
            SO.ODDS_HISTORY_CSV = orig_path


def test_card_written():
    print("16. season.py --no-network writes card.md")
    from club_soccer import season as S

    if S.CARD.exists():
        S.CARD.unlink()
    proc = subprocess.run(
        [sys.executable, "-m", "club_soccer.season", "--no-network", "--fast"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    check("season.py --no-network --fast exits 0", proc.returncode == 0)
    check("card.md was written", S.CARD.exists())
    if S.CARD.exists():
        text = S.CARD.read_text()
        check("card.md has the freshness header", text.startswith("# Club Soccer —"))
        check("card.md reports upcoming fixture count", "Upcoming fixtures:" in text)


def test_bsd_detail_and_player_contracts():
    print("17. BSD detail, cup metadata, and player-stat contracts")
    import json as _json
    from bsd_client import (fixture_detail_fields, normalized_match_statistics,
                            _STATUS_ALIASES)
    from club_soccer import fetch as F
    from club_soccer import seed_real as SR
    from club_soccer.player_features import _players_from_event, _extract_player_entry

    check("upcoming status alias maps to BSD notstarted",
          _STATUS_ALIASES["upcoming"] == "notstarted")
    detail = _json.loads((ROOT / "club_soccer/data/bsd_cache/event_205762.json").read_text())
    stats = normalized_match_statistics(detail)
    check("BSD live_stats maps shots, SoT, corners, and xG",
          stats["home"]["shots"] == 16.0 and stats["home"]["sot"] == 8.0
          and stats["home"]["corners"] == 4.0 and stats["home"]["xg"] == 4.88)
    extracted = SR._extract_stats(detail)
    check("seed_real uses the same normalized statistic contract",
          extracted["home"]["shots"] == 16 and extracted["home"]["xg"] == 4.88)

    shootout = _json.loads((ROOT / "club_soccer/data/bsd_cache/event_207339.json").read_text())
    fields = fixture_detail_fields(shootout)
    check("cup shootout metadata records scope and winner",
          fields["result_scope"] == "penalties" and fields["shootout_home"] == 4.0
          and fields["shootout_away"] == 5.0 and fields["shootout_winner"] == "away")
    check("BSD xG carries explicit source provenance",
          fixture_detail_fields(detail).get("xg_source") == "bsd")
    lineup_rows = _players_from_event(detail)
    positive = [r for r, _ in lineup_rows if r["mins"] > 0]
    unused = [r for r, _ in lineup_rows if r["mins"] == 0]
    ids = {r["player_id"] for r, _ in lineup_rows if r["player_id"] is not None}
    check("current BSD lineups separate appearances from unused substitutes",
          len(positive) >= 20 and len(unused) >= 1 and len(ids) >= 20)
    canonical = _extract_player_entry({
        "event": {"id": 1},
        "player": {"id": 99, "name": "Test Striker", "position": "FW"},
        "minutes_played": 72, "expected_goals": 0.42,
        "expected_assists": 0.11, "goal_assist": 1,
    })
    check("canonical player-stat row keeps provider ID/minutes/xG/xA",
          canonical["player_id"] == 99 and canonical["mins"] == 72.0
          and canonical["xg"] == 0.42 and canonical["xa"] == 0.11)

    old = pd.DataFrame([{"fixture_id": 1, "home_shots": 10, "home_xg": 1.2}])
    incoming = pd.DataFrame([{"fixture_id": 1, "home_shots": "", "home_xg": 2.0},
                             {"fixture_id": 2, "home_shots": 4, "home_xg": 0.5}])
    merged = F._merge_fixture_rows(old, incoming)
    row = merged[merged["fixture_id"].astype(str) == "1"].iloc[0]
    check("fixture merge preserves rich observations while filling new fields",
          float(row["home_shots"]) == 10.0 and float(row["home_xg"]) == 2.0
          and len(merged) == 2)


def test_bsd_enrichment_flattening():
    print("17b. BSD v2 enrichment flattening")
    from club_soccer.bsd_enrichment import flatten_record

    record = {
        "event": {"id": 379, "event_date": "2026-05-24T15:00:00Z",
                  "season_id": 1, "league_id": 1, "league_name": "Premier League",
                  "home_team": "Home FC", "away_team": "Away FC",
                  "home_team_id": 10, "away_team_id": 20,
                  "home_score": 2, "away_score": 1},
        "stats": {"stats": {"home": {"expected_goals": 1.7, "total_shots": 12,
                                         "pass_accuracy_pct": 82},
                               "away": {"expected_goals": 0.8, "total_shots": 8}},
                  "shotmap": [{"home": True, "xg": 0.4, "xgot": 0.2,
                               "type": "goal", "sit": "assisted", "pos": {"x": 12}},
                              {"home": False, "xg": 0.1, "type": "miss",
                               "sit": "set-piece", "pos": {"x": 30}}]},
        "lineups": {"lineup_status": "confirmed",
                     "lineups": {"home": {"starters": [{"id": 1}] * 11,
                                            "substitutes": [{"id": 2}] * 5},
                                  "away": {"starters": [{"id": 3}] * 10,
                                            "substitutes": []}}},
        "incidents": {"incidents": [{"type": "goal"}, {"type": "yellow_card"},
                                      {"type": "substitution"}]},
        "player_stats": {"player_stats": [
            {"team_id": 10, "minutes_played": 90, "expected_goals": 0.4,
             "goals": 1, "total_pass": 20, "accurate_pass": 16, "rating": 7.2},
            {"team_id": 20, "minutes_played": 90, "expected_goals": 0.1,
             "goals": 0, "total_pass": 10, "accurate_pass": 7, "rating": 6.5},
        ]},
    }
    row = flatten_record(record)
    check("v2 flatten keeps team xG and shotmap totals",
          row["home_bsd_xg"] == 1.7 and row["home_shotmap_shots"] == 1
          and row["away_shotmap_xg"] == 0.1)
    check("v2 flatten keeps confirmed XI and player aggregates",
          row["lineup_status"] == "confirmed" and row["home_lineup_starters"] == 11
          and row["home_player_goals"] == 1.0 and row["away_player_count"] == 1)
    check("v2 flatten counts incidents", row["incident_goals"] == 1
          and row["incident_yellow_cards"] == 1 and row["incident_substitutions"] == 1)


def test_bsd_enrichment_candidate_join():
    print("17c. BSD xG candidate join is isolated and fill-only")
    from club_soccer.bsd_enrichment import candidate_fixtures

    base = pd.DataFrame({
        "fixture_id": [1, 2], "home_xg": [np.nan, 1.0],
        "away_xg": [np.nan, 0.8], "xg_source": ["", "existing"],
    })
    enriched = pd.DataFrame({
        "fixture_id": [1, 2], "fixture_joined": [True, True],
        "home_shotmap_xg": [1.4, 1.8], "away_shotmap_xg": [0.7, 0.9],
    })
    candidate, report = candidate_fixtures(base, enriched)
    check("candidate fills only missing xG pairs",
          report == {"eligible": 2, "filled": 1, "skipped_existing": 1}
          and float(candidate.loc[0, "home_xg"]) == 1.4
          and float(candidate.loc[1, "home_xg"]) == 1.0
          and candidate.loc[0, "xg_source"] == "bzzoiro_v2")


def test_real_xg_and_date_aware_prediction():
    print("18. real xG fit and date-aware prediction path")
    fx = M.load_fixtures()
    sample = M.played(fx).head(300).copy()
    sample["home_xg"] = sample["home_goals"].astype(float) + 0.2
    sample["away_xg"] = sample["away_goals"].astype(float) + 0.2
    params = M.fit(sample)
    check("fit records complete real-xG coverage", params.get("real_xg_coverage") == 1.0)
    teams = params["teams"]
    if len(teams) >= 2:
        comp = str(sample.iloc[0]["competition"])
        pred = M.predict_match(teams[0], teams[1], comp, "2026-07-20", params=params)
        check("predict_match returns a normalized date-aware forecast",
              abs(sum(pred["probs"][k] for k in ("home", "draw", "away")) - 1.0) < 0.002)
    else:
        check("predict_match returns a normalized date-aware forecast", False)


def test_gated_production_layers():
    print("19. gated calibration, market blend, and immutable adjustments")
    from club_soccer import calibrate as CAL
    from app import market_blend as MB

    stored_calibration = CAL.load_maps()
    check("temperature calibration is explicitly gated",
          isinstance(stored_calibration, dict)
          and stored_calibration.get("method") == "temperature"
          and 0.5 < float(stored_calibration.get("temperature", 0.0)) < 1.5
          and (CAL.load_active_maps() is not None) == bool(
              json.loads(CAL.CALIB_FILE.read_text()).get("active", False)))
    check("market blend remains off until an explicit promotion",
          not MB.is_default_on("club_soccer"))

    params = M.load_params()
    adj = {"home": {"attack_mult": 0.5, "defense_mult": 1.5},
           "away": {"attack_mult": 1.1, "defense_mult": 0.9}}
    original = json.loads(json.dumps(adj))
    M.predict("Arsenal", "Chelsea", "Premier League", params=params, player_adj=adj)
    check("prediction does not mutate caller-owned player adjustments", adj == original)
    base = M.predict("Arsenal", "Chelsea", "Premier League", params=params)
    inactive = M.predict("Arsenal", "Chelsea", "Premier League", params=params,
                         quality_adj={"active": False, "shift": 0.2})
    check("inactive player quality is an exact no-op",
          np.allclose([base["probs"][k] for k in ("home", "draw", "away")],
                      [inactive["probs"][k] for k in ("home", "draw", "away")]))
    active = M.predict("Arsenal", "Chelsea", "Premier League", params=params,
                       quality_adj={"active": True, "shift": 0.1, "coverage": 1.0})
    check("active quality adjustment remains normalized",
          abs(sum(active["probs"][k] for k in ("home", "draw", "away")) - 1.0) < 0.002)

    from club_soccer import schema as S
    check("fixture schema includes xG provenance", "xg_source" in S.FIXTURE_COLUMNS)


if __name__ == "__main__":
    test_registry()
    test_api_key_lookup()
    test_model_math()
    test_edge_and_settlement()
    test_runner_and_adapter()
    test_health_and_repair()
    test_match_identity_reconciliation()
    test_feature_store_pit()
    test_fdcouk_alias_coverage()
    test_minutes_windows()
    test_transfer_reattribution()
    test_player_quality_pit()
    test_context_apply()
    test_standings_asof()
    test_weather_features()
    test_snapshot_odds_dedupe()
    test_do_not_bet()
    test_card_written()
    test_bsd_detail_and_player_contracts()
    test_bsd_enrichment_flattening()
    test_bsd_enrichment_candidate_join()
    test_real_xg_and_date_aware_prediction()
    test_gated_production_layers()
    print()
    if _fails:
        print(f"{len(_fails)} FAILURE(S): " + ", ".join(_fails))
        raise SystemExit(1)
    print("All Club Soccer tests passed.")
