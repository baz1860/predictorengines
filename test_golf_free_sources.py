#!/usr/bin/env python3
"""Offline tests for the free-source golf provider stack."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from golf import model, store
from golf.providers import ROUNDS_CSV
from golf.providers.odds_manual import ManualOddsProvider, parse_skybet_threeball_text
from golf.providers.pgatour_stats import parse_stat_page
from golf.round_pricer import price_round_3balls

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
            store.upsert_field(con, "T1", [{"name": "Alex Smith", "status": "active"}])
        out = store.export_field_csv("T1", path=root / "field.csv", db_path=db)
        check("imports rounds into store", n == 1 and db.exists(), str(n))
        check("exports field csv from store", out.exists() and "Alex Smith" in out.read_text(),
              out.read_text())


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


def main():
    print("Golf free-source tests")
    test_provider_paths()
    test_manual_threeball_parser()
    test_pgatour_stats_text_parser()
    test_store_round_import_and_field_export()
    test_public_stat_priors_and_round_pricer()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
