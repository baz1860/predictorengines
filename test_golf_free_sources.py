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
from golf import simulate as golf_sim
from golf import edge as golf_edge
from golf import portfolio as golf_portfolio

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if not cond:
        FAIL += 1
        raise AssertionError(f"{name}: {detail}")
    PASS += 1
    print(f"  PASS  {name}")


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


def test_manual_threeball_parser_rejects_shifted_or_incomplete_groups():
    issues = []
    raw = """
    3 Ball Round 1 - Rai / Morikawa / Day
    Aaron Rai
    Collin Morikawa
    2.38
    Jason Day
    3.50
    """
    assert parse_skybet_threeball_text(raw, issues=issues) == []
    assert issues and "missing odds" in issues[0]


def test_market_blend_preserves_nested_probabilities(monkeypatch):
    class Rated:
        name = "Player A"
    results = {"__cut_binds__": True, "Player A": {
        "win": .100, "top5": .101, "top10": .2, "top20": .3, "made_cut": .8}}
    odds = {"player a": {"name": "Player A", "odds_win": 5.0, "odds_top5": 9.0}}
    monkeypatch.setattr(golf_edge.market, "blend_weights",
                        lambda: {"win": .6, "top5": .45})
    monkeypatch.setattr(golf_edge.market, "devig_outright",
                        lambda *a, **k: {"Player A": .2})
    monkeypatch.setattr(golf_edge.market, "devig_line", lambda *a, **k: .11)
    rows = golf_edge.price_all([Rated()], results, odds, {}, {}, bankroll=100,
                               calibrated=False, blended=True, min_edge=-100.0)
    probs = {r["side"]: r["p_model"] for r in rows}
    assert probs["win"] <= probs["top5"]


def test_portfolio_caps_mutually_exclusive_outrights():
    rows = [{"player": f"P{i}", "side": "win", "p_model": .1,
             "stake_gbp": 5.0} for i in range(8)]
    staked = golf_portfolio.apply_portfolio(rows, bankroll=100.0)
    assert sum(r["stake_gbp"] for r in staked) <= 10.0


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


def test_store_is_live_only_and_exports_field():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db = root / "golf.db"
        store.init_db(db)
        with store.connect(db) as con:
            tables = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            store.upsert_events(con, [{
                "event_id": "T1",
                "name": "Test Event",
                "course_name": "Test Course",
            }])
            store.upsert_field(con, "T1", [{
                "name": "Alex Smith",
                "status": "active",
                "tee_time_r1": "09:10",
                "start_hole_r1": "1",
                "world_rank": 42,
            }])
        out = store.export_field_csv("T1", path=root / "field.csv", db_path=db)
        check("SQLite excludes historical rounds", "rounds" not in tables, str(tables))
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


def test_weather_resolution():
    provider = OpenMeteoProvider()
    loc, matched = provider.resolve_location(
        course_name="Travelers Championship",
        event_name="Travelers Championship",
    )
    check("resolves event alias to course location",
          loc is not None and loc.course_name == "TPC River Highlands",
          f"{loc} matched={matched}")


def test_espn_tee_times_drive_weather_shift():
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


def _bovada_coupon(markets):
    """Minimal Bovada coupon payload: one event under one group."""
    return [{
        "events": [{
            "id": "e1",
            "link": "/golf/pga-tour/genesis-scottish-open-2026",
            "displayGroups": [{"markets": markets}],
        }],
    }]


def _bovada_market(desc, names_odds):
    return {"description": desc,
            "outcomes": [{"description": n, "price": {"decimal": str(o)}}
                         for n, o in names_odds]}


def test_bovada_live_coupon():
    from golf.providers.bovada import (BovadaGolfProvider, COUPON_URL,
                                       LIVE_COUPON_URL)
    check("live coupon URL asks for in-play markets only",
          "liveOnly=true" in LIVE_COUPON_URL and "preMatchOnly" not in LIVE_COUPON_URL,
          LIVE_COUPON_URL)
    check("pre-match coupon URL unchanged",
          "preMatchOnly=true" in COUPON_URL, COUPON_URL)

    # Separate cache files per feed, so a cached pre-match board can never be
    # served when the live board is requested (and vice versa).
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        provider = BovadaGolfProvider(cache_dir=cache_dir, ttl_seconds=3600)
        (cache_dir / "coupon_golf.json").write_text('[{"pre": true}]')
        (cache_dir / "coupon_golf_live.json").write_text('[{"live": true}]')
        pre = provider.fetch_coupon(use_cache=True)
        live = provider.fetch_coupon(use_cache=True, live=True)
        check("pre-match and live coupons cached separately",
              pre == [{"pre": True}] and live == [{"live": True}],
              f"pre={pre} live={live}")

    # live_event_quotes keeps only tournament-level markets: in-play round
    # 2/3-ball prices reflect holes already played and must not reach the
    # pre-round group pricer.
    provider = BovadaGolfProvider()
    coupon = _bovada_coupon([
        _bovada_market("winner", [("Rory McIlroy", 4.5), ("Jon Rahm", 6.0)]),
        _bovada_market("tournament match-ups", [("A One", 1.8), ("B Two", 2.0)]),
        _bovada_market("3rd round 2-balls", [("C Three", 1.9), ("D Four", 1.9)]),
        _bovada_market("3rd round match-ups", [("E Five", 1.8), ("F Six", 2.0)]),
    ])
    quotes = provider.live_event_quotes(coupon, "Genesis Scottish Open", event_id="401")
    markets = sorted({q.market for q in quotes})
    check("live quotes keep win + tournament matchup only",
          markets == ["tournament_matchup", "win"], str(markets))
    check("live quotes drop in-play round groups",
          not any(q.market in ("2ball", "3ball", "round_matchup") for q in quotes),
          str([q.market for q in quotes]))
    check("live quote count", len(quotes) == 4, str(len(quotes)))


def test_bovada_live_merge_prefers_live_price():
    from golf.providers.bovada import BovadaGolfProvider, _dedupe
    provider = BovadaGolfProvider()
    pre = provider.event_quotes(
        _bovada_coupon([_bovada_market("winner", [("Rory McIlroy", 8.0)])]),
        "Genesis Scottish Open", event_id="401")
    live = provider.live_event_quotes(
        _bovada_coupon([_bovada_market("winner", [("Rory McIlroy", 4.5)])]),
        "Genesis Scottish Open", event_id="401")
    merged = _dedupe(live + pre)  # refresh puts live first: fresher price wins
    check("live price wins the merge dedupe",
          len(merged) == 1 and merged[0].decimal_odds == 4.5,
          str([(q.player_name, q.decimal_odds) for q in merged]))


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
    test_bovada_live_coupon()
    test_bovada_live_merge_prefers_live_price()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
