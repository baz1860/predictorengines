#!/usr/bin/env python3
"""Suite-level Closing Line Value (CLV) tracking for every priced engine (V3 M5).

CLV is the most reliable signal that a betting operation is genuinely +EV: it
measures whether the odds you took beat the market's closing odds.

    CLV% (per bet) = bet_odds / closing_odds - 1

The V2 `clv.py` only knew about the World Cup ledger. This module works off the
shared `data/suite_ledger.csv` (M4) and snapshots **every** engine's open bets,
matching a closing-odds proxy by `(engine, event_id, market, side)` — the same
identity the ledger and settlement use.

    python3 clv_suite.py --snapshot            # snapshot current odds for all
                                               #   open bets (offline: reads each
                                               #   engine's odds file)
    python3 clv_suite.py --snapshot --engine cfb
    python3 clv_suite.py --report              # per-settled-bet CLV + summary
    python3 clv_suite.py --report --write-closing
                                               #   also backfill ledger.closing_odds

Snapshots are taken from each engine's *manual odds file* (the same file the app
Edge view reads — refresh it via the engine's "API" odds source to capture live
lines, then snapshot). This keeps CLV fully offline-safe: no network, no crash
when a file or the network is absent — just a clear "no data" report.

History: data/clv_history.csv
    snapshot_time, engine, event_id, market, side, home, away, match_date, odds
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
HISTORY = DATA / "clv_history.csv"
HISTORY_COLS = ["snapshot_time", "engine", "event_id", "market", "side",
                "home", "away", "match_date", "odds"]

# Per-engine manual odds files (offline snapshot source).
WC_ODDS = HERE / "odds.csv"
CLUB_ODDS = HERE / "club_soccer" / "data" / "odds.csv"
CFB_ODDS = HERE / "cfb" / "odds.csv"
GOLF_ODDS = HERE / "golf" / "data" / "odds.csv"
TENNIS_ODDS = HERE / "tennis" / "data" / "odds.csv"

# Wide-format side -> column maps.
_WC_SIDE_COL = {"home": "odds_home", "draw": "odds_draw", "away": "odds_away",
                "over25": "odds_over25", "under25": "odds_under25",
                "btts_yes": "odds_btts_yes", "btts_no": "odds_btts_no"}
_GOLF_SIDE_COL = {"win": "odds_win", "top5": "odds_top5", "top10": "odds_top10",
                  "top20": "odds_top20", "cut": "odds_cut"}


# ── identity ──────────────────────────────────────────────────────────────────
def _event_id(row) -> str:
    """Stable event id for a ledger row — mirrors bankroll_store._event_id_for."""
    eid = str(row.get("event_id", "") or "").strip()
    if eid:
        return eid
    from contracts import fixture_key
    return fixture_key(row.get("match_date", ""), row.get("home", ""),
                       row.get("away", ""))


def _norm_line(v) -> str:
    try:
        if v == "" or pd.isna(v):
            return ""
        return f"{abs(float(v)):.1f}"
    except (TypeError, ValueError):
        return str(v)


def _f(v):
    try:
        f = float(v)
        return f if f > 1.0 else None
    except (TypeError, ValueError):
        return None


# ── providers: ledger-row -> current odds (offline, from odds files) ──────────
def _provider_wide(path: Path, side_col: dict, by_player: bool = False):
    """Build a lookup for a wide-format odds file (WC by home/away, golf by name)."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    def lookup(row) -> float | None:
        col = side_col.get(str(row.get("side", "")))
        if not col or col not in df.columns:
            return None
        if by_player:
            m = df[df["name"].astype(str) == str(row.get("home", ""))]
        else:
            m = df[(df["home"].astype(str) == str(row.get("home", "")))
                   & (df["away"].astype(str) == str(row.get("away", "")))]
        if m.empty:
            return None
        return _f(m.iloc[0].get(col))
    return lookup


def _provider_long(path: Path):
    """Build a lookup for a long-format odds file (club soccer / CFB)."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "odds" not in df.columns:
        return None
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df = df.dropna(subset=["odds"])

    def lookup(row) -> float | None:
        m = df[(df["home"].astype(str) == str(row.get("home", "")))
               & (df["away"].astype(str) == str(row.get("away", "")))
               & (df["side"].astype(str) == str(row.get("side", "")))]
        bet_market = str(row.get("market", "") or "")
        if bet_market and "market" in df.columns:
            mm = m[m["market"].astype(str) == bet_market]
            if not mm.empty:
                m = mm
        bet_line = _norm_line(row.get("line", ""))
        if bet_line and "line" in m.columns:
            ml = m[m["line"].map(_norm_line) == bet_line]
            if not ml.empty:
                m = ml
        if m.empty:
            return None
        return _f(m.iloc[0]["odds"])
    return lookup


def _provider_tennis(path: Path):
    """Tennis odds lookup: odds.csv columns are player_a, player_b, odds_a, odds_b.
    A bet row has home=player, away=opponent, side='win'. We find the matching row
    regardless of which direction the players appear in odds.csv."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "odds_a" not in df.columns:
        return None

    def lookup(row) -> float | None:
        home = str(row.get("home", "")).strip()
        away = str(row.get("away", "")).strip()
        if not home or not away:
            return None
        # Match found when players appear in either order in odds.csv.
        m_ab = df[(df["player_a"].astype(str).str.strip() == home)
                  & (df["player_b"].astype(str).str.strip() == away)]
        if not m_ab.empty:
            return _f(m_ab.iloc[0].get("odds_a"))   # home player = player_a
        m_ba = df[(df["player_a"].astype(str).str.strip() == away)
                  & (df["player_b"].astype(str).str.strip() == home)]
        if not m_ba.empty:
            return _f(m_ba.iloc[0].get("odds_b"))   # home player = player_b
        return None
    return lookup


def _providers():
    """{engine: lookup(row)->odds|None}. Missing files simply yield no matches."""
    return {
        "worldcup": _provider_wide(WC_ODDS, _WC_SIDE_COL),
        "club_soccer": _provider_long(CLUB_ODDS),
        "cfb": _provider_long(CFB_ODDS),
        "golf": _provider_wide(GOLF_ODDS, _GOLF_SIDE_COL, by_player=True),
        "tennis": _provider_tennis(TENNIS_ODDS),
    }


# ── history ───────────────────────────────────────────────────────────────────
def _load_history() -> pd.DataFrame | None:
    if not HISTORY.exists():
        return None
    try:
        df = pd.read_csv(HISTORY)
    except Exception:
        return None
    return df if not df.empty else None


def _load_ledger() -> pd.DataFrame:
    from app import bankroll_store
    return bankroll_store.load_ledger()


# ── snapshot ──────────────────────────────────────────────────────────────────
def snapshot(engine: str | None = None) -> int:
    ledger = _load_ledger()
    if ledger.empty:
        print("No suite ledger yet — nothing to snapshot.")
        return 0
    open_bets = ledger[ledger["status"] == "open"].copy()
    if engine:
        open_bets = open_bets[open_bets["engine"] == engine]
    if open_bets.empty:
        print("No open bets to snapshot." + (f" (engine={engine})" if engine else ""))
        return 0

    providers = _providers()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_rows, no_provider = [], set()
    for _, r in open_bets.iterrows():
        eng = str(r.get("engine", ""))
        lookup = providers.get(eng)
        if lookup is None:
            no_provider.add(eng)
            continue
        odds = lookup(r)
        if odds is None:
            continue
        new_rows.append({
            "snapshot_time": now, "engine": eng, "event_id": _event_id(r),
            "market": str(r.get("market", "") or ""), "side": str(r.get("side", "")),
            "home": str(r.get("home", "")), "away": str(r.get("away", "")),
            "match_date": str(r.get("match_date", "")), "odds": round(odds, 3)})

    if no_provider:
        print(f"No current odds file for: {', '.join(sorted(no_provider))} "
              "(refresh that engine's odds, then snapshot).")
    if not new_rows:
        print("No open bets matched current odds (names/markets); nothing recorded.")
        return 0
    hist = _load_history()
    out = (pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)
           if hist is not None else pd.DataFrame(new_rows, columns=HISTORY_COLS))
    DATA.mkdir(exist_ok=True)
    out.to_csv(HISTORY, index=False)
    print(f"Recorded {len(new_rows)} odds snapshot(s) at {now} -> "
          f"{HISTORY.name} ({len(out)} rows total).")
    return len(new_rows)


# ── closing proxy + report ────────────────────────────────────────────────────
def closing_odds(hist: pd.DataFrame, engine: str, event_id: str, market: str,
                 side: str, match_date) -> float | None:
    """Latest snapshot at/before kick-off for this exact outcome, or None."""
    if hist is None:
        return None
    m = hist[(hist["engine"].astype(str) == str(engine))
             & (hist["event_id"].astype(str) == str(event_id))
             & (hist["market"].astype(str) == str(market))
             & (hist["side"].astype(str) == str(side))].copy()
    if m.empty:
        return None
    cutoff = pd.Timestamp(str(match_date)) + pd.Timedelta(days=1)
    m["t"] = pd.to_datetime(m["snapshot_time"], errors="coerce", utc=True)
    m = m[m["t"] <= cutoff.tz_localize("UTC")]
    if m.empty:
        return None
    return _f(m.sort_values("t").iloc[-1]["odds"])


def _clv_table(ledger: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    settled = ledger[ledger["status"].isin(["won", "lost", "push"])].copy()
    if settled.empty:
        return settled
    closings = []
    for _, r in settled.iterrows():
        closings.append(closing_odds(hist, r.get("engine", ""), _event_id(r),
                                     str(r.get("market", "") or ""),
                                     str(r.get("side", "")), r.get("match_date", "")))
    settled["closing_proxy"] = closings
    have = settled.dropna(subset=["closing_proxy"]).copy()
    if have.empty:
        return have
    have["odds"] = pd.to_numeric(have["odds"], errors="coerce")
    have["clv"] = have["odds"] / have["closing_proxy"] - 1.0
    return have


def report(write_closing: bool = False) -> None:
    hist = _load_history()
    if hist is None:
        print("No CLV snapshots yet. Run 'python3 clv_suite.py --snapshot' "
              "before events start (refresh odds first to capture live lines).")
        return
    ledger = _load_ledger()
    if ledger.empty:
        print("No suite ledger yet.")
        return
    have = _clv_table(ledger, hist)
    if have.empty:
        print("No settled bets have a matching closing-odds snapshot yet.")
        return

    show = have[["engine", "match_date", "bet", "odds", "closing_proxy",
                 "clv", "status", "pnl"]].copy()
    show = show.rename(columns={"closing_proxy": "closing"})
    show["clv"] = (show["clv"] * 100).map("{:+.1f}%".format)
    pd.set_option("display.width", 170)
    print(f"Suite CLV report — {len(have)} settled bet(s) with closing snapshots:\n")
    print(show.to_string(index=False))
    print(f"\n  Rolling mean CLV : {have['clv'].mean() * 100:+.2f}%")
    print(f"  Positive-CLV rate: {(have['clv'] > 0).mean():.0%}")
    print(f"  Win rate         : {(have['status'] == 'won').mean():.0%}")
    for eng, grp in have.groupby("engine"):
        print(f"    {eng:<12} n={len(grp):<3} mean CLV {grp['clv'].mean() * 100:+.2f}%")

    if write_closing:
        _backfill_closing(ledger, have)


def _backfill_closing(ledger: pd.DataFrame, have: pd.DataFrame) -> None:
    """Write the closing proxy into ledger.closing_odds (backed up first)."""
    from app import bankroll_store
    led = ledger.copy()
    led["closing_odds"] = led["closing_odds"].astype("object")
    for idx, r in have.iterrows():
        led.at[idx, "closing_odds"] = round(float(r["closing_proxy"]), 3)
    if bankroll_store.LEDGER.exists():
        shutil.copy(bankroll_store.LEDGER,
                    bankroll_store.LEDGER.with_suffix(".csv.bak.clv"))
    led.to_csv(bankroll_store.LEDGER, index=False)
    print(f"\nBackfilled closing_odds for {len(have)} settled bet(s) "
          f"-> {bankroll_store.LEDGER.name} (backup: .csv.bak.clv).")


def compute_rolling_clv(ledger: pd.DataFrame) -> list[dict]:
    """Return [{i, v}] rolling-mean CLV% series for dashboard rendering.

    Reads the current clv_history.csv; returns [] when no snapshots exist yet.
    Used by app/dashboard_data.py — no arguments beyond the ledger DataFrame
    (which the caller already holds) so the dashboard never imports bankroll_store
    through this path."""
    hist = _load_history()
    if hist is None:
        return []
    table = _clv_table(ledger, hist)
    if table.empty or "clv" not in table.columns:
        return []
    c = table["clv"].dropna()
    if c.empty:
        return []
    roll = c.cumsum() / range(1, len(c) + 1)
    return [{"i": i, "v": round(float(v) * 100, 3)} for i, v in enumerate(roll.tolist())]


def main():
    ap = argparse.ArgumentParser(description="Suite-level CLV tracking (V3 M5)")
    ap.add_argument("--snapshot", action="store_true",
                    help="record current odds for open bets (offline, from odds files)")
    ap.add_argument("--engine", help="limit snapshot to one engine slug")
    ap.add_argument("--report", action="store_true", help="CLV report for settled bets")
    ap.add_argument("--write-closing", action="store_true",
                    help="with --report, backfill ledger.closing_odds (backed up first)")
    args = ap.parse_args()
    if args.snapshot:
        snapshot(args.engine)
    elif args.report:
        report(write_closing=args.write_closing)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
