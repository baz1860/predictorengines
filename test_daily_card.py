#!/usr/bin/env python3
"""Combined daily-card tests.

Covers:
  * market-key normalization across the engines' spellings;
  * human bet labels (tennis "A to beat B", golf matchups/outrights, soccer
    passthrough);
  * per-market narrative prose renders with the row's numbers and no template
    artifacts, and rotates phrasing across a run of similar bets;
  * render() produces the full document from a synthetic collect() payload
    (glance table, per-engine sections, quiet section, notes) with no network;
  * the World Cup availability fix: a successful live-data check that finds
    zero unavailable players still refreshes player_availability.csv's mtime
    (creating the header if the file is missing), so provenance freshness
    reflects "last checked" rather than flagging a daily update as stale.

Run: python3 test_daily_card.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DC = _load(ROOT / "scripts" / "daily_card.py", "daily_card_test")


def _row(**kw) -> dict:
    base = {"match_date": "", "home": "", "away": "", "market": "", "side": "",
            "line": "", "bet": "", "odds": 2.0, "p_model": 0.55, "p_book": 0.5,
            "edge": 0.05, "ev_per_unit": 0.1, "kelly_frac": 0.01,
            "stake_gbp": 1.0, "recommended": True}
    base.update(kw)
    return base


def test_market_key():
    check("golf outright", DC._market_key(_row(market="win_outright", side="win")) == "win")
    check("golf matchup via market",
          DC._market_key(_row(market="matchup_vs_keita_nakajima",
                              side="matchup:Lee Hodges|Keita Nakajima")) == "matchup")
    check("tennis match_winner", DC._market_key(_row(market="match_winner")) == "match")
    check("soccer total passthrough", DC._market_key(_row(market="total")) == "total")
    check("golf top10 spellings", DC._market_key(_row(market="top_10")) == "top10")
    check("cut", DC._market_key(_row(market="make_cut")) == "cut")


def test_bet_labels():
    r = _row(market="match_winner", home="Ann Li", away="Iga Swiatek")
    check("tennis label", DC._bet_label(r) == "Ann Li to beat Iga Swiatek",
          DC._bet_label(r))
    r = _row(market="matchup_vs_x", side="matchup:Lee Hodges|Keita Nakajima",
             home="Lee Hodges", bet="Matchup vs Keita Nakajima — Lee Hodges")
    check("golf matchup label", DC._bet_label(r) == "Lee Hodges to beat Keita Nakajima",
          DC._bet_label(r))
    r = _row(market="win_outright", side="win", home="Lucas Glover")
    check("golf outright label", DC._bet_label(r) == "Lucas Glover to win outright",
          DC._bet_label(r))
    r = _row(market="total", side="over", home="Brazil", away="Norway",
             bet="Over 2.5 goals")
    lab = DC._bet_label(r)
    check("soccer total keeps bet text + fixture",
          "Over 2.5 goals" in lab and "Brazil v Norway" in lab, lab)


def test_prose():
    today = date(2026, 7, 5)
    rows = {
        "total": _row(market="total", side="over", line="2.5", home="Brazil",
                      away="Norway", bet="Over 2.5 goals", odds=1.73,
                      p_model=0.564, p_book=0.546, edge=0.018,
                      match_date="2026-07-05"),
        "1x2": _row(market="1x2", side="home", home="Brazil", away="Norway",
                    bet="Brazil win", odds=1.83, p_model=0.532, p_book=0.525,
                    edge=0.007),
        "match": _row(market="match_winner", home="Marcos Giron",
                      away="Alexander Zverev", odds=9.55, p_model=0.20,
                      p_book=0.10, edge=0.102),
        "matchup": _row(market="matchup_vs_keita_nakajima",
                        side="matchup:Lee Hodges|Keita Nakajima",
                        home="Lee Hodges", odds=1.87, p_model=0.88, p_book=0.5,
                        edge=0.384),
        "win": _row(market="win_outright", side="win", home="Lucas Glover",
                    odds=126.0, p_model=0.021, p_book=0.005, edge=0.016),
        "spread": _row(market="spread", side="home", line="-3.5",
                       home="Eagles", away="Cowboys", odds=1.91,
                       p_model=0.56, p_book=0.52, edge=0.04),
    }
    for kind, r in rows.items():
        s = DC._bet_prose(r, "soccer" if kind in ("total", "1x2") else "", today)
        check(f"{kind} prose is substantial", len(s) > 80, s[:60])
        check(f"{kind} prose has no template artifacts",
              "{" not in s and "}" not in s and "None" not in s, s)
    # numbers actually appear
    s = DC._bet_prose(rows["match"], "tennis", today)
    check("match prose mentions both players",
          "Marcos Giron" in s and "Alexander Zverev" in s, s)
    check("match prose flags the underdog case", "underdog" in s, s)
    # rotation: a run of similar bets doesn't repeat the same paragraph
    texts = {DC._bet_prose(rows["matchup"], "", today, i) for i in range(3)}
    check("matchup prose rotates", len(texts) == 3, str(len(texts)))


def test_render():
    sec_bets = {
        "id": "tennis", "name": "Tennis (ATP + WTA)", "sport": "tennis",
        "n_priced": 12, "note": "", "reason": "", "gate": "PASS",
        "freshness": [],
        "bets": [_row(market="match_winner", home="Ann Li", away="Iga Swiatek",
                      odds=3.1, p_model=0.4, p_book=0.32, edge=0.08,
                      stake_gbp=2.5)],
    }
    sec_quiet = {"id": "nhl", "name": "NHL", "sport": "hockey", "bets": [],
                 "n_priced": 0, "note": "", "gate": "ok", "freshness": [],
                 "reason": "odds.csv has no filled-in rows."}
    card = {"date": date(2026, 7, 5), "sections": [sec_bets, sec_quiet],
            "bankroll": {"bankroll": 120.64, "net_pnl": 20.64, "open": 3},
            "generated": "2026-07-05 09:00"}
    doc = DC.render(card)
    check("title has the day", "Sunday 5 July 2026" in doc, doc[:80])
    check("glance table present", "| Sport | When | Bet |" in doc)
    check("engine section present", "## Tennis (ATP + WTA)" in doc)
    check("bet header present", "### Ann Li to beat Iga Swiatek" in doc)
    check("quiet section explains NHL",
          "NHL" in doc and "no odds filled in" in doc, doc)
    check("PASS gate not flagged as warning", "validation gate" not in doc)
    check("bankroll in notes", "£120.64" in doc)
    # no-bets day reads as an editorial note, not an error
    card["sections"] = [sec_quiet]
    doc2 = DC.render(card)
    check("empty day has a lead", "quiet day" in doc2.lower(), doc2[:200])


def test_availability_touch():
    LD = _load(ROOT / "scripts" / "worldcup" / "live_data.py", "live_data_test")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "data").mkdir()
        LD.ROOT = tmp
        LD.RAW_DIR = tmp / "raw"
        LD.FIXTURES_CSV = tmp / "fixtures_live.csv"
        LD.AVAILABILITY_CSV = tmp / "player_availability.csv"
        LD._fetch_bsd_events = lambda api_key, mode: []  # BSD answers, empty

        # missing file: a zero-row check creates it with the schema header
        LD.fetch_bsd("morning", "fake-key")
        check("zero-row check creates the CSV", LD.AVAILABILITY_CSV.exists())
        if LD.AVAILABILITY_CSV.exists():
            head = LD.AVAILABILITY_CSV.read_text().strip()
            check("created CSV carries the schema header",
                  head == ",".join(LD.AVAILABILITY_COLUMNS), head)

        # stale file: a zero-row check refreshes the mtime (the reported bug)
        old = time.time() - 15 * 86400
        os.utime(LD.AVAILABILITY_CSV, (old, old))
        LD.fetch_bsd("prekickoff", "fake-key")
        age_days = (time.time() - LD.AVAILABILITY_CSV.stat().st_mtime) / 86400
        check("zero-row check refreshes mtime (no more '14d old' warning)",
              age_days < 0.01, f"age {age_days:.2f}d")


def main():
    print("daily-card tests")
    test_market_key()
    test_bet_labels()
    test_prose()
    test_render()
    test_availability_touch()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
