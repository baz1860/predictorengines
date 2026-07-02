#!/usr/bin/env python3
"""Offline tests for the free-source golf provider stack."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from golf import model, store
from golf import refresh as golf_refresh
from golf.providers import ROUNDS_CSV
from golf.providers.odds_manual import ManualOddsProvider, parse_skybet_threeball_text
from golf.providers.pgatour_stats import parse_stat_page
from golf.providers.weather import OpenMeteoProvider
from golf.round_pricer import price_round_3balls
from golf.tee_times import parse_tee_sheet_text
from golf import simulate as golf_sim

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_provider_paths():
    check("legacy provider still points at golf/data/rounds.csv",
          ROUNDS_CSV.as_posix().endswith("golf/data/rounds.csv"),
          str(ROUNDS_CSV))


def test_manual_threeball_parser():
    raw = """
    3 Ball Round 1 - Smith / Jones / Brown
    Alex Smith
    2.50
    Ben Jones
    3.20
    Cam Brown
    4.00
    """
    groups = parse_skybet_threeball_text(raw)
    check("parses one 3-ball group", len(groups) == 1, str(groups))
    check("parses three players and odds", groups[0]["players"][2] == ("Cam Brown", 4.0),
          str(groups))
    quotes = ManualOddsProvider().parse_threeball_text(raw, event_id="E1", round_no=2)
    check("normalizes parsed quotes", len(quotes) == 3 and quotes[0].round_no == 2,
          str(quotes))


def test_pgatour_stats_text_parser():
    html = """
    <table>
      <tr><th>Rank</th><th>Player</th><th>Avg</th></tr>
      <tr><td>1</td><td>Scottie Scheffler</td><td>2.162</td></tr>
      <tr><td>2</td><td>Ludvig Åberg</td><td>1.715</td></tr>
    </table>
    """
    rows = parse_stat_page(html, stat_id="02675", stat_name="sg_total", season=2026)
    check("parses PGA stat text rows", len(rows) == 2, str(rows))
    check("keeps accented names", rows[1].player_name == "Ludvig Åberg", str(rows[1]))
    check("parses stat value", abs(rows[0].value - 2.162) < 1e-9, str(rows[0]))


def test_store_round_import_and_field_export():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rounds = root / "rounds.csv"
        with rounds.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "tournament_id", "date", "tour", "is_major", "course", "round",
                "player", "dg_id", "score_to_par", "field_size", "made_cut", "finish",
            ])
            w.writeheader()
            w.writerow({"tournament_id": "T1", "date": "2026-01-01", "tour": "pga",
                        "is_major": 0, "course": "Test Course", "round": 1,
                        "player": "Alex Smith", "dg_id": "", "score_to_par": -2,
                        "field_size": 3, "made_cut": 1, "finish": 1})
        db = root / "golf.db"
        n = store.import_rounds_csv(rounds, db_path=db)
        with store.connect(db) as con:
            store.upsert_field(con, "T1", [{
                "name": "Alex Smith",
                "status": "active",
                "tee_time_r1": "09:10",
                "start_hole_r1": "1",
                "world_rank": 42,
            }])
        out = store.export_field_csv("T1", path=root / "field.csv", db_path=db)
        check("imports rounds into store", n == 1 and db.exists(), str(n))
        check("exports field csv from store", out.exists() and "Alex Smith" in out.read_text(),
              out.read_text())
        exported = list(csv.DictReader(out.open()))
        check("exports tee metadata", exported[0]["tee_time_r1"] == "09:10",
              str(exported[0]))
        check("exports world rank", exported[0]["world_rank"] == "42", str(exported[0]))


def test_public_stat_priors_and_round_pricer():
    with tempfile.TemporaryDirectory() as td:
        stats = Path(td) / "pgatour_stats.csv"
        stats.write_text(
            "season,stat_id,stat_name,player_name,rank,value,raw_json,source\n"
            "2026,02675,sg_total,Free Source Player,1,1.25,{},pgatour\n"
        )
        priors = model.load_public_stat_priors(stats)
        check("loads public SG prior", priors["Free Source Player"]["sg_total"] == 1.25,
              str(priors))

    params = {
        "sigma_field": 2.8,
        "default_skill": -0.4,
        "public_stat_priors": {"Free Source Player": {"sg_total": 1.25}},
        "players": {
            "Opponent One": {"skill": 0.0, "form": 0.0, "sigma": 2.8, "n_rounds": 120},
            "Opponent Two": {"skill": -0.2, "form": 0.0, "sigma": 2.8, "n_rounds": 120},
        },
        "form_weight": 0.7,
        "public_stat_blend": 0.15,
    }
    quotes = ManualOddsProvider().parse_threeball_text(
        """
        3 Ball Round 1 - Free / One / Two
        Free Source Player
        2.80
        Opponent One
        3.00
        Opponent Two
        3.50
        """,
        event_id="E1",
        round_no=1,
    )
    rows = price_round_3balls(quotes, params, sims=5000, seed=1)
    check("round pricer returns three sides", len(rows) == 3, str(rows))
    check("round pricer includes dead-heat equivalent probability",
          "p_dead_heat_equiv" in rows[0], str(rows[0]))


def test_model_feature_adjustments():
    params = {
        "sigma_field": 2.8,
        "default_skill": -0.4,
        "players": {
            "Approach Fit": {"skill": 0.5, "form": 0.0, "sigma": 2.8, "n_rounds": 120},
            "Putt Fit": {"skill": 0.5, "form": 0.0, "sigma": 2.8, "n_rounds": 120},
        },
        "public_stat_priors": {
            "Approach Fit": {"sg_total": 0.5, "stats": {"sg_app": 2.0, "sg_putt": -0.5}},
            "Putt Fit": {"sg_total": 0.5, "stats": {"sg_app": -0.5, "sg_putt": 2.0}},
        },
        "public_stat_blend": 0.0,
        "form_weight": 0.7,
    }
    rated = model.predict_field(
        ["Approach Fit", "Putt Fit"],
        params,
        course="Augusta National Golf Club",
        weather_features={},
    )
    by_name = {p.name: p for p in rated}
    check("course archetype adjustment is applied",
          by_name["Approach Fit"].course_arch_adj > by_name["Putt Fit"].course_arch_adj,
          str({k: v.course_arch_adj for k, v in by_name.items()}))

    early = model.Player(name="Early Player", tee_time_r1="08:00", owgr=20)
    late = model.Player(name="Late Player", tee_time_r1="14:00", owgr=20)
    weather = {"rounds": {"1": {"wave_penalty": {
        "split_hour": 12, "early_penalty": -0.10, "late_penalty": 0.10}}}}
    rated_weather = model.predict_field([early, late], {
        "sigma_field": 2.8,
        "default_skill": -0.4,
        "players": {},
        "public_stat_priors": {},
    }, weather_features=weather)
    by_name = {p.name: p for p in rated_weather}
    check("weather wave rewards easier tee side",
          by_name["Early Player"].weather_wave_adj > by_name["Late Player"].weather_wave_adj,
          str({k: v.weather_wave_adj for k, v in by_name.items()}))
    check("world-rank prior is used for unknown global players",
          by_name["Early Player"].global_prior_adj > 0.0,
          str(by_name["Early Player"].global_prior_adj))


def test_weather_resolution_and_tee_overrides():
    provider = OpenMeteoProvider()
    loc, matched = provider.resolve_location(
        course_name="Travelers Championship",
        event_name="Travelers Championship",
    )
    check("resolves event alias to course location",
          loc is not None and loc.course_name == "TPC River Highlands",
          f"{loc} matched={matched}")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tee_times.csv"
        old = golf_refresh.TEE_TIMES_CSV
        path.write_text(
            "event_id,event,name,round,tee_time,start_hole\n"
            "E1,Test Event,Alex Smith,1,08:10,1\n"
            "E1,Test Event,Alex Smith,2,13:20,10\n"
        )
        golf_refresh.TEE_TIMES_CSV = path
        try:
            overrides = golf_refresh._load_tee_time_overrides("E1", "Test Event")
            rows = [{"name": "Alex Smith", "status": "active", "source_player_id": "1"}]
            patched = golf_refresh._apply_tee_time_overrides(rows, overrides)
        finally:
            golf_refresh.TEE_TIMES_CSV = old
        check("loads manual tee overrides",
              overrides["alex smith"]["tee_time_r1"] == "08:10",
              str(overrides))
        check("applies manual tee overrides before field export",
              patched[0]["tee_time_r2"] == "13:20" and patched[0]["start_hole_r2"] == "10",
              str(patched))


def test_tee_sheet_parser_and_weather_shift():
    raw = """
    Round 1
    8:05 AM Tee 1 Scottie Scheffler / Rory McIlroy / Xander Schauffele
    13:20 10 Tommy Fleetwood, Justin Rose
    """
    rows = parse_tee_sheet_text(raw, event_id="E1", event="Test", default_round=1)
    check("parses pasted tee sheet groups", len(rows) == 5, str(rows))
    check("parses tee time and start hole",
          rows[0]["tee_time"] == "8:05 AM" and rows[0]["start_hole"] == "1",
          str(rows[0]))

    early = model.Player(name="Early", tee_time_r1="08:00")
    late = model.Player(name="Late", tee_time_r1="14:00")
    weather = {"rounds": {"1": {"wave_penalty": {
        "split_hour": 12, "early_penalty": -0.2, "late_penalty": 0.2}}}}
    rated = model.predict_field([early, late], {
        "sigma_field": 2.8,
        "default_skill": 0.0,
        "players": {},
        "public_stat_priors": {},
    }, weather_features=weather)
    shifts = golf_sim._weather_score_shifts(rated)
    check("simulator receives round-specific weather score shifts",
          shifts is not None and shifts.shape == (2, 4) and abs(shifts[:, 0]).sum() > 0,
          str(shifts))


def test_global_player_priors_loader():
    with tempfile.TemporaryDirectory() as td:
        priors = Path(td) / "global_player_priors.csv"
        priors.write_text(
            "name,sg_total,sigma,source,notes\n"
            "Global Star,1.4,2.7,manual,test\n"
        )
        loaded = model.load_global_player_priors(priors)
        check("loads global player prior", loaded["global star"]["sg_total"] == 1.4,
              str(loaded))
        old = model.GLOBAL_PRIORS_CSV
        model.GLOBAL_PRIORS_CSV = priors
        try:
            rated = model.predict_field(["Global Star"], {
                "sigma_field": 2.8,
                "default_skill": -0.5,
                "players": {},
                "public_stat_priors": {},
            }, weather_features={})
        finally:
            model.GLOBAL_PRIORS_CSV = old
        check("global prior feeds unknown-player rating",
              abs(rated[0].sg_baseline - 1.4) < 1e-9 or rated[0].global_prior_adj > 0,
              str(rated[0]))


def main():
    print("Golf free-source tests")
    test_provider_paths()
    test_manual_threeball_parser()
    test_pgatour_stats_text_parser()
    test_store_round_import_and_field_export()
    test_public_stat_priors_and_round_pricer()
    test_model_feature_adjustments()
    test_weather_resolution_and_tee_overrides()
    test_tee_sheet_parser_and_weather_shift()
    test_global_player_priors_loader()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
