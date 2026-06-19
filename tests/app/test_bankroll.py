#!/usr/bin/env python3
"""M4 shared bankroll, portfolio & settlement tests.

Covers the V3 M4 acceptance criteria:
  * old (V2) suite ledger loads unchanged;
  * recording the same open bet twice stays deduped;
  * suite caps clamp single-event exposure;
  * golf settlement is event-safe — a stale outright stays open when the latest
    completed event is not its event;
  * settle dry-run reports what would settle without writing the ledger.

Runs entirely on temp files; the real ledger/state are never touched.

Run: python3 test_bankroll.py
"""
from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.bankroll_store as B
from app.engines import golf as G

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


def _point_store_at(tmp: Path, bankroll=100.0):
    B.DATA = tmp
    B.LEDGER = tmp / "suite_ledger.csv"
    B.STATE = tmp / "suite_bankroll.json"
    B._save_state(bankroll, peak=bankroll, start=bankroll)


# ── old ledger loads unchanged ────────────────────────────────────────────────
def test_legacy_ledger_loads():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _point_store_at(tmp)
        legacy_cols = B._CORE_COLS
        legacy = pd.DataFrame([{
            "placed_on": "2026-01-01", "engine": "worldcup", "sport": "soccer",
            "match_date": "2026-06-12", "home": "Spain", "away": "Brazil",
            "side": "home", "bet": "Spain win", "odds": 2.1, "stake": 5.0,
            "status": "open", "pnl": 0.0, "bankroll_after": ""}])[legacy_cols]
        legacy.to_csv(tmp / "suite_ledger.csv", index=False)
        df = B.load_ledger()
        check("legacy ledger loads without error", len(df) == 1)
        check("V3 columns backfilled", all(c in df.columns for c in B._V3_COLS))
        check("legacy data intact", df.iloc[0]["home"] == "Spain")
        check("backfilled event_id is empty", df.iloc[0]["event_id"] == "")


# ── dedupe ────────────────────────────────────────────────────────────────────
def test_dedupe():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _point_store_at(tmp)
        cand = pd.DataFrame([{
            "match_date": "2026-09-01", "home": "Ohio State", "away": "Michigan",
            "side": "home", "bet": "ML home", "odds": 1.8, "stake": 5.0,
            "event_id": "ev1", "market": "ml", "line": "", "source": "manual",
            "model": "blend"}])
        first = B.place_bets("cfb", "cfb", cand)
        second = B.place_bets("cfb", "cfb", cand)
        check("first placement recorded", len(first) == 1)
        check("duplicate placement deduped", len(second) == 0)


# ── suite single-event cap ────────────────────────────────────────────────────
def test_single_event_cap():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _point_store_at(tmp, bankroll=100.0)
        # three legs on ONE event, £10 each (£30) — single-event cap is 15%
        cand = pd.DataFrame([
            {"match_date": "2026-09-01", "home": "A", "away": "B", "side": s,
             "bet": f"bet {s}", "odds": 2.0, "stake": 10.0, "event_id": "bigEv",
             "market": "1x2", "line": "", "source": "manual", "model": "blend"}
            for s in ("home", "draw", "away")])
        placed = B.place_bets("worldcup", "soccer", cand)
        total = placed["stake"].astype(float).sum()
        check("single-event exposure capped at ~15% of bankroll",
              total <= 15.0 + 0.05, f"total={total}")
        check("some legs still recorded", len(placed) >= 1)


# ── golf event-safe settlement ────────────────────────────────────────────────
def test_golf_event_safe_settlement():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rounds = tmp / "rounds.csv"
        # One completed event (id 1) ending 2026-01-04. P1 wins it.
        rows = []
        for rnd, dt in enumerate(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], 1):
            rows.append({"tournament_id": 1, "date": dt, "round": rnd,
                         "player": "P1", "score_to_par": -5, "made_cut": 1})
            rows.append({"tournament_id": 1, "date": dt, "round": rnd,
                         "player": "P2", "score_to_par": -1, "made_cut": 1})
        pd.DataFrame(rows).to_csv(rounds, index=False)
        G.ROUNDS_CSV = rounds

        open_bets = pd.DataFrame([
            # placed for THIS event (ref date before/at event) → settles
            {"placed_on": "2026-01-01", "match_date": "2026-01-01",
             "home": "P1", "side": "win"},
            # stale future outright: ref date AFTER the only completed event → open
            {"placed_on": "2026-06-01", "match_date": "2026-06-01",
             "home": "P1", "side": "win"},
        ])
        graded = G.GolfAdapter().grade_open_bets(open_bets)
        check("in-event golf bet settles", 0 in graded and graded[0][0] == "won",
              str(graded.get(0)))
        check("stale future golf outright stays open", 1 not in graded,
              str(graded.get(1)))


# ── settle dry-run ────────────────────────────────────────────────────────────
class _StubAdapter:
    def grade_open_bets(self, rows):
        return {i: ("won", "1-0") for i in rows.index}


class _StubRegistry:
    def get(self, eid):
        if eid == "cfb":
            return _StubAdapter()
        raise KeyError(eid)


def test_settle_dry_run():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _point_store_at(tmp, bankroll=100.0)
        ledger = B.load_ledger()
        ledger.loc[0] = {**{c: "" for c in B.COLS},
                         "placed_on": str(date.today()), "engine": "cfb",
                         "sport": "cfb", "home": "A", "away": "B", "side": "home",
                         "bet": "ML home", "odds": 2.0, "stake": 10.0,
                         "status": "open", "pnl": 0.0, "event_id": "ev1"}
        B._save_ledger(ledger)

        reg = _StubRegistry()
        preview = B.settle(reg, dry_run=True)
        check("dry-run reports a settlement", preview["settled"] == 1)
        check("dry-run previews pnl", preview["preview"][0]["pnl"] == 10.0)
        # ledger untouched on disk
        after = B.load_ledger()
        check("dry-run leaves bet open", (after["status"] == "open").all())
        check("dry-run leaves bankroll unchanged", B.current_bankroll() == 100.0)

        # real settle writes
        real = B.settle(reg, dry_run=False)
        check("real settle commits", real["settled"] == 1)
        check("real settle updates bankroll", B.current_bankroll() == 110.0,
              str(B.current_bankroll()))
        settled = B.load_ledger()
        check("real settle closes the bet", (settled["status"] == "won").all())


def main():
    for fn in [test_legacy_ledger_loads, test_dedupe, test_single_event_cap,
               test_golf_event_safe_settlement, test_settle_dry_run]:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
