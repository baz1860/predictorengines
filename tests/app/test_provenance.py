#!/usr/bin/env python3
"""M7 data-provenance & refresh-hygiene tests.

Covers the V3 M7 acceptance:
  * each engine reports freshness from file mtimes with NO network call;
  * stale / missing inputs are flagged; fresh ones are not;
  * manifests record source / fetched_at / row counts / schema version;
  * manual-odds mistakes are reported with row number, column and expected value;
  * a valid odds file yields no errors; freshness_warnings never raises.

Runs on a temp ROOT and temp odds files; real engine data is never touched.

Run: python3 test_provenance.py
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import provenance as PV

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")
        assert cond, detail or name


def test_freshness_and_manifest():
    saved_root = PV.ROOT
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        PV.ROOT = tmp
        try:
            # build a fake cfb tree: fresh games, stale model, missing odds
            (tmp / "cfb" / "data").mkdir(parents=True)
            games = tmp / "cfb" / "data" / "games.csv"
            games.write_text("h,a\n1,2\n3,4\n5,6\n")          # 3 data rows
            model = tmp / "cfb" / "data" / "power_params.json"
            model.write_text("{}")
            old = time.time() - 60 * 86400                     # 60 days old
            os.utime(model, (old, old))
            # cfb/odds.csv intentionally absent

            fr = {f["key"]: f for f in PV.freshness("cfb")}
            check("fresh games → ok", fr["games"]["status"] == "ok", str(fr["games"]))
            check("60d model → stale", fr["model"]["status"] == "stale", str(fr["model"]))
            check("absent odds → missing", fr["odds"]["status"] == "missing", str(fr["odds"]))

            warns = PV.freshness_warnings("cfb")
            check("warnings mention stale + missing only", len(warns) == 2, str(warns))

            path = PV.write_manifest("cfb")
            check("manifest written", path.exists())
            m = PV.read_manifest("cfb")
            check("manifest schema_version", m["schema_version"] == PV.SCHEMA_VERSION)
            check("manifest row count", m["inputs"]["games"]["rows"] == 3,
                  str(m["inputs"]["games"]))
            check("manifest records source", bool(m["inputs"]["games"]["source"]))
            check("manifest fetched_at set for existing file",
                  m["inputs"]["games"]["fetched_at"] is not None)
            check("manifest marks missing odds", m["inputs"]["odds"]["exists"] is False)
        finally:
            PV.ROOT = saved_root


def test_freshness_warnings_never_raises():
    # an unknown engine simply has no inputs → empty list, no exception
    check("unknown engine → []", PV.freshness_warnings("not_an_engine") == [])


def test_odds_validation_cfb():
    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "good.csv"
        good.write_text(
            "date,home,away,neutral,market,side,line,odds\n"
            "2026-06-14,Ohio State,Michigan,0,ml,home,,1.45\n"
            "2026-06-14,Ohio State,Michigan,0,spread,home,-6.5,1.91\n")
        check("valid cfb odds → no errors", PV.validate_odds_file("cfb", good) == [])

        bad = Path(d) / "bad.csv"
        bad.write_text(
            "date,home,away,neutral,market,side,line,odds\n"
            "2026-06-14,A,B,2,moneyline,home,,1.45\n"   # bad neutral, bad market
            "2026-06-14,A,B,0,ml,over,,1.45\n"          # bad side for ml
            "2026-06-14,A,B,0,total,over,,1.91\n"       # total missing line
            "2026-06-14,A,B,0,ml,home,,0.5\n")          # odds <= 1.0
        errs = PV.validate_odds_file("cfb", bad)
        cols = {(e["row"], e["column"]) for e in errs}
        check("flags bad neutral with row+col", (1, "neutral") in cols, str(cols))
        check("flags bad market", (1, "market") in cols, str(cols))
        check("flags bad side for ml", (2, "side") in cols, str(cols))
        check("flags missing line for total", (3, "line") in cols, str(cols))
        check("flags odds <= 1.0", (4, "odds") in cols, str(cols))
        check("error message names expected value",
              any("expected" in e["message"] for e in errs))


def test_odds_validation_wide():
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "wc.csv"
        bad.write_text(
            "date,home,away,odds_home,odds_draw,odds_away,odds_over25,odds_under25,odds_btts_yes,odds_btts_no\n"
            "2026-06-13,Qatar,Switzerland,abc,,,,,,\n")    # odds_home not numeric
        errs = PV.validate_odds_file("worldcup", bad)
        check("wide format flags bad odds cell",
              any(e["column"] == "odds_home" and e["row"] == 1 for e in errs), str(errs))


def main():
    print("M7 provenance tests")
    test_freshness_and_manifest()
    test_freshness_warnings_never_raises()
    test_odds_validation_cfb()
    test_odds_validation_wide()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
