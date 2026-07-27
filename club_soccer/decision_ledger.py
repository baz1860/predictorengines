#!/usr/bin/env python3
"""Append-only decision ledger — the trustworthy basis for staking evidence.

Why this replaces the reconstruction backtest
---------------------------------------------
`decision_time_backtest` (v1) rebuilt each past decision from TODAY's state:
today's alias map to resolve identities, a median across an unknown set of
bookmakers as the "price". Both are hindsight. The adversarial review showed it
concretely — sample inclusion and ROI changed with the alias-map version (11
fixtures vs 5), and the median is a price no bookmaker actually offered.

You cannot fix reconstruction; you have to stop reconstructing. So the decision
that matters — which club, which league, which executable price, which model
probability — is FROZEN at the moment it is made, in an append-only ledger, and
never rewritten:

    decision_ledger.csv     one immutable row per (fixture, market, side)
    settlement_ledger.csv   appended later, keyed on provider_fixture_id

The backtest then reads only these frozen records. Deleting the alias map, or
refitting the model, cannot change a single historical number — which is the
whole point, and is asserted by a test.

The hard constraint (unchanged)
-------------------------------
A decision can only be recorded while the fixture is still upcoming, so the
ledger accumulates FORWARD from the day it ships. It starts empty; the gate
stays closed until enough settled decisions exist. There is no honest shortcut
— seeding it from the old median snapshots would reintroduce the non-executable
price this module exists to remove.

CLI:
  python3 -m club_soccer.decision_ledger --record     # append today's decisions
  python3 -m club_soccer.decision_ledger --settle     # append results/CLV
  python3 -m club_soccer.decision_ledger --status
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNTIME = Path(os.environ.get("CLUB_SOCCER_RUNTIME_DIR", str(DATA)))
DECISIONS = RUNTIME / "decision_ledger.csv"
SETTLEMENTS = RUNTIME / "settlement_ledger.csv"

# Decision window: a snapshot must sit at least this far before kick-off (the
# gate's floor) and no earlier than the far edge, so the recorded lead is a
# realistic pre-kickoff decision rather than a days-early quote.
#
# This window is designed for FREQUENT capture (see deploy/*.capture.plist,
# every 15 min), not a single daily run. A once-a-day 07:30 job with the old
# 60–180 window only ever saw the ~2% of fixtures that happened to kick off
# 60–180 min later, so "a season to 1,000 bets" was unreachable. Running every
# 15 minutes and recording the first sighting inside [60, 120] captures every
# fixture that has odds, once, at a realistic 60–120 min pre-kickoff instant —
# the 60-min span tolerates a missed run or two without dropping a fixture,
# while still freezing a decision close to the gate's 60-min floor.
MIN_LEAD_MIN = 60
MAX_LEAD_MIN = 120

DECISION_FIELDS = [
    "decision_id", "decision_ts", "provider_fixture_id", "kickoff_utc",
    "competition", "raw_home", "raw_away", "club_home", "club_away",
    "resolver_version", "market", "side", "book", "odds_executed",
    "p_book_devig", "p_consensus_devig", "p_model", "edge", "decision_lead_min",
    "lineup_confidence", "strategy_eligible",
    "train_cutoff", "model_hash", "code_hash", "prior_hash",
]
SETTLEMENT_FIELDS = [
    "provider_fixture_id", "settled_ts", "home_goals", "away_goals",
    "market", "side", "won", "pinnacle_close_devig", "clv",
]


# ── frozen provenance ─────────────────────────────────────────────────────

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""


def resolver_version() -> str:
    """Hash of the identity artifacts used to resolve a name RIGHT NOW, so a
    later alias-map change is detectable and never silently rewrites history."""
    h = hashlib.sha256()
    for name in ("club_alias_map.json", "club_registry.json"):
        h.update(_sha(DATA / name).encode())
    return h.hexdigest()[:16]


def _code_hash() -> str:
    """Stable pricing-strategy version.

    Learned parameters deliberately do not belong here: a normal refit changes
    model_params.json and must not erase all accumulated forward evidence.
    Every code path that can change a priced or selected row does belong here.
    """
    h = hashlib.sha256()
    for name in (
        "model.py", "competitions.py", "club_identity.py", "edge.py",
        "decision_ledger.py", "evidence_gate.py", "player_features.py",
        "availability.py", "calibrate.py", "market_model.py",
    ):
        p = HERE / name
        h.update(p.read_bytes() if p.exists() else b"")
    shared_blend = HERE.parent / "app" / "market_blend.py"
    h.update(shared_blend.read_bytes() if shared_blend.exists() else b"")
    return h.hexdigest()[:16]


# ── executable quote selection ────────────────────────────────────────────

def select_executable_quote(side_odds: dict[str, dict[str, float]],
                            our_side: str) -> tuple[str, float, dict] | None:
    """Pick ONE real, executable quote — never a synthetic median.

    `side_odds` = {side: {book: decimal_odds}}. Among books that quote the
    COMPLETE market (so the de-vig is honest), choose the one offering the best
    price for `our_side`. Returns (book, odds, that_book's_devig_probs) or None.
    """
    sides = list(side_odds)
    books = set().union(*(set(d) for d in side_odds.values())) if side_odds else set()
    best = None
    for book in books:
        prices = {}
        ok = True
        for s in sides:
            o = side_odds[s].get(book)
            if not o or float(o) <= 1.0:
                ok = False
                break
            prices[s] = float(o)
        if not ok:
            continue                      # book must quote the whole market
        inv = {s: 1.0 / p for s, p in prices.items()}
        tot = sum(inv.values())
        devig = {s: inv[s] / tot for s in sides}
        odds = prices[our_side]
        if best is None or odds > best[1]:
            best = (book, odds, devig)
    return best


def market_consensus_devig(side_odds: dict[str, dict[str, float]],
                           sides: list[str] | None = None) -> dict[str, float] | None:
    """Selection-INDEPENDENT benchmark probability per side.

    The executable quote is the book offering our side its BEST price — and the
    best price for a side is, by construction, that book's LOWEST implied
    probability for it. Benchmarking the model against that same de-vig therefore
    understates the market and inflates the recorded edge by the selection
    procedure itself (worse the more books quote — a best-of-N extremum). So the
    edge is instead measured against the mean de-vig across EVERY book that
    quotes the complete market, chosen before looking at who is cheapest. Returns
    {side: mean_devig} or None if fewer than the full set of sides can be priced.
    """
    sides = sides or list(side_odds)
    books = set().union(*(set(d) for d in side_odds.values())) if side_odds else set()
    per_side: dict[str, list[float]] = {s: [] for s in sides}
    for book in books:
        prices = {}
        ok = True
        for s in sides:
            o = side_odds[s].get(book)
            if not o or float(o) <= 1.0:
                ok = False
                break
            prices[s] = float(o)
        if not ok:
            continue                      # book must quote the whole market
        inv = {s: 1.0 / prices[s] for s in sides}
        tot = sum(inv.values())
        for s in sides:
            per_side[s].append(inv[s] / tot)
    if not all(per_side[s] for s in sides):
        return None
    return {s: sum(v) / len(v) for s, v in per_side.items()}


# ── recording ─────────────────────────────────────────────────────────────

def _append(path: Path, fields: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _existing_decision_ids() -> set[str]:
    if not DECISIONS.exists():
        return set()
    with DECISIONS.open(encoding="utf-8") as fh:
        return {row["decision_id"] for row in csv.DictReader(fh)}


def _decision_id(fixture_id, market: str, side: str) -> str:
    return hashlib.sha256(f"{fixture_id}|{market}|{side}".encode()).hexdigest()[:20]


def record(api_key: str | None = None, verbose: bool = True,
           events: list[dict] | None = None,
           odds_cache: dict[str, dict] | None = None) -> int:
    """Append immutable decision rows for fixtures now in the decision window.

    Idempotent per (fixture, market, side): a decision is recorded ONCE, the
    first time the fixture enters the window. Re-runs never rewrite it — that is
    what makes the record trustworthy.
    """
    from api_keys import get_key
    from bsd_client import get_all_events
    from . import model as M
    from .club_identity import canonical_name
    from .competitions import comp_from_bsd_league
    from .fetch import bsd_league_name
    from .snapshot_odds import odds_comparison

    key = api_key or get_key("bsd", env="BSD_API_KEY")
    if not key:
        if verbose:
            print("  decision_ledger: no BSD key — nothing recorded")
        return 0

    now = datetime.now(timezone.utc)
    params = M.load_params()
    rv, ch, mh = resolver_version(), _code_hash(), _sha(DATA / "model_params.json")
    ph = _sha(DATA / "uefa_coefficients_history.json")
    train_cutoff = str(now.date())
    seen = _existing_decision_ids()

    # Match the production probability pipeline. Evidence from a bare model
    # must not authorize availability-adjusted, calibrated or market-blended
    # prices that were never recorded.
    from .calibrate import apply as apply_calibration, load_active_maps
    from . import market_model as MM
    from app.market_blend import blend_two, is_default_on, weight_for

    calib_maps = load_active_maps()
    dnb_active = MM.history_age_days() >= MM.WARMUP_DAYS
    blend_active = is_default_on("club_soccer")
    blend_w = weight_for("club_soccer") if blend_active else None
    try:
        from .availability import match_availability, match_confidence
        from .player_features import PlayerFeatureStore

        player_store = PlayerFeatureStore().load()
        if not player_store._player_records():
            player_store.refresh_from_cache()
    except Exception:
        player_store = None

    if events is None:
        try:
            events = get_all_events(
                key, status="notstarted", date_from=str(now.date()),
                date_to=str(now.date())
            )
        except Exception as exc:
            if verbose:
                print(f"  decision_ledger: BSD fetch failed ({exc})")
            return 0
    else:
        from .schema import normalize_status
        events = [
            ev for ev in events
            if normalize_status(ev.get("status")) == "NOT"
            and str(ev.get("event_date") or ev.get("date") or "")[:10]
            == str(now.date())
        ]

    out: list[dict] = []
    for ev in events:
        comp = comp_from_bsd_league(bsd_league_name(ev))
        if comp is None:
            continue
        fid = ev.get("id")
        ko = ev.get("event_date") or ev.get("date")
        if fid is None or not ko:
            continue
        try:
            kickoff = datetime.fromisoformat(str(ko).replace("Z", "+00:00"))
            lead = (kickoff - now).total_seconds() / 60.0
        except (ValueError, TypeError):
            continue
        if not (MIN_LEAD_MIN <= lead <= MAX_LEAD_MIN):
            continue                      # outside the decision window

        raw_home = str(ev.get("home_team") or "")
        raw_away = str(ev.get("away_team") or "")
        club_home = canonical_name(raw_home, country=comp.country)
        club_away = canonical_name(raw_away, country=comp.country)
        if club_home not in params["teams"] or club_away not in params["teams"]:
            continue
        player_adj = None
        lineup_confidence = 1.0
        if player_store is not None:
            try:
                player_adj = player_store.adjustments_for_match(ev)
                lineup_confidence = match_confidence(
                    match_availability(player_store, ev)
                )
            except Exception:
                player_adj = None
                lineup_confidence = 1.0
        try:
            pred = M.predict_match(club_home, club_away, comp.name,
                                   str(kickoff.date()), "ensemble",
                                   params=params, player_adj=player_adj)
        except Exception:
            continue
        p = pred["probs"]
        if calib_maps is not None:
            p["home"], p["draw"], p["away"] = apply_calibration(
                p["home"], p["draw"], p["away"], calib_maps
            )
        model_p = {"1x2": {"home": p["home"], "draw": p["draw"], "away": p["away"]},
                   "total25": {"over": p["over25"], "under": 1.0 - p["over25"]}}

        cache_key = str(fid)
        cmp = odds_cache.get(cache_key) if odds_cache is not None else None
        if cmp is None:
            try:
                cmp = odds_comparison(key, fid)
            except Exception:
                continue
            if odds_cache is not None:
                odds_cache[cache_key] = cmp
        markets = cmp.get("markets") or {}

        for market, bsd_map in (("1x2", {"home": "HOME", "draw": "DRAW", "away": "AWAY"}),
                                ("total25", {"over": "over@2.5", "under": "under@2.5"})):
            entry = markets.get("1x2") if market == "1x2" else markets.get("over_under_25")
            entry = entry or {}
            side_odds: dict[str, dict[str, float]] = {}
            for our_side, bsd_key in bsd_map.items():
                books = (entry.get(bsd_key) or {}).get("bookmakers") or {}
                side_odds[our_side] = {
                    b: float(v["decimal_odds"]) for b, v in books.items()
                    if v.get("decimal_odds") and float(v["decimal_odds"]) > 1.0}
            for our_side in bsd_map:
                did = _decision_id(fid, market, our_side)
                if did in seen:
                    continue
                quote = select_executable_quote(side_odds, our_side)
                if quote is None:
                    continue
                book, odds, devig = quote
                # Keep consensus for diagnostics, but select on the SAME
                # executing-book de-vig used by edge.rows_from_odds. A gate
                # trained on a different edge definition authorizes a different
                # betting strategy.
                consensus = market_consensus_devig(side_odds, list(side_odds))
                p_consensus = (
                    float(consensus[our_side])
                    if consensus else float(devig[our_side])
                )
                p_bench = float(devig[our_side])
                pm = float(model_p[market][our_side])
                if blend_active and blend_w is not None:
                    pm = blend_two(pm, p_bench, blend_w)
                edge = pm - p_bench
                strategy_eligible = True
                if dnb_active:
                    decision = MM.do_not_bet({
                        "home": club_home, "away": club_away,
                        "date": str(kickoff.date()), "market": market,
                        "side": our_side, "edge": edge,
                    }, asof=now.isoformat())
                    strategy_eligible = not bool(decision["suppress"])
                out.append({
                    "decision_id": did, "decision_ts": now.isoformat(),
                    "provider_fixture_id": fid, "kickoff_utc": kickoff.isoformat(),
                    "competition": comp.name, "raw_home": raw_home, "raw_away": raw_away,
                    "club_home": club_home, "club_away": club_away,
                    "resolver_version": rv, "market": market, "side": our_side,
                    "book": book, "odds_executed": round(odds, 4),
                    "p_book_devig": round(float(devig[our_side]), 5),
                    "p_consensus_devig": round(p_consensus, 5),
                    "p_model": round(pm, 5), "edge": round(edge, 5),
                    "decision_lead_min": round(lead, 1),
                    "lineup_confidence": round(float(lineup_confidence), 4),
                    "strategy_eligible": int(strategy_eligible),
                    "train_cutoff": train_cutoff, "model_hash": mh,
                    "code_hash": ch, "prior_hash": ph,
                })
                seen.add(did)

    _append(DECISIONS, DECISION_FIELDS, out)
    if verbose:
        print(f"  decision_ledger: recorded {len(out)} new decision(s)")
    return len(out)


# ── settlement ─────────────────────────────────────────────────────────────

def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_decisions() -> list[dict]:
    return _load(DECISIONS)


def load_settlements() -> list[dict]:
    return _load(SETTLEMENTS)


def _match_result(d: dict,
                  results: dict[tuple[str, str], list],
                  tol_days: int = 1):
    """Find the finished fixture for a decision by canonical match identity.

    The decision's stored `provider_fixture_id` is a BSD event id, but
    `fixtures.csv.fixture_id` is whichever provider row survived deduplication
    (identities.dedupe_fixtures keeps the RICHEST row, often a non-BSD one), so
    the two ids disagree for the majority of matches and a fixture-id join
    silently drops most settlements. Identity — canonical (home, away) plus the
    kickoff date (±`tol_days` to absorb UTC/local date skew) — is the stable key
    the rest of the module already settles CLV on. Returns (home_goals,
    away_goals, closing_key) or None.
    """
    from .club_identity import canonical_name
    cands = results.get((canonical_name(str(d.get("club_home") or "")),
                         canonical_name(str(d.get("club_away") or ""))))
    if not cands:
        return None
    try:
        kd = date.fromisoformat(str(d.get("kickoff_utc") or "")[:10])
    except ValueError:
        # No usable kickoff date: only settle if the pair is unambiguous.
        return cands[0][1:] if len(cands) == 1 else None
    best = None
    for fd, hg, ag, mkey in cands:
        diff = abs((fd - kd).days)
        if diff <= tol_days and (best is None or diff < best[0]):
            best = (diff, hg, ag, mkey)
    return best[1:] if best else None


def settle(verbose: bool = True) -> int:
    """Append settlement rows for decisions whose fixtures have finished.

    The settlement ROW is keyed on provider_fixture_id + market + side (so the
    decision↔settlement join never changes), but the RESULT is looked up by
    canonical match identity, not by fixtures.csv's fixture_id — that id is a
    post-dedup provider id that disagrees with the recorded BSD event id for
    most matches. CLV is scored against the de-vigged Pinnacle close (1X2 and
    OU2.5 where a closing feed exists).
    """
    import pandas as pd

    from . import model as M
    from .club_identity import canonical_name
    from .market_settlement import closing_probs, match_key, side_won
    from .schema import OFFICIAL_RESULT_STATUSES, normalize_status

    decisions = load_decisions()
    if not decisions:
        return 0
    already = {(s["provider_fixture_id"], s["market"], s["side"])
               for s in load_settlements()}

    # Settle ONLY on a standing official result (FT/FIN/AET/PEN/AWD). Using the
    # training filter (M.played) was wrong twice over: it admits a live in-play
    # row whose current score is not final, and it excludes AWD, which is a
    # legal awarded result that must settle. Terminal-but-non-sporting AWD
    # settles here even though it never trains the goals model.
    fx = M.load_fixtures()
    if {"home_goals", "away_goals"}.issubset(fx.columns):
        fx = fx.dropna(subset=["home_goals", "away_goals"])
    if "status" in fx.columns:
        fx = fx[fx["status"].map(normalize_status).isin(OFFICIAL_RESULT_STATUSES)]
    results: dict[tuple[str, str], list] = defaultdict(list)
    if {"home", "away", "date"}.issubset(fx.columns):
        for r in fx.itertuples(index=False):
            try:
                hg, ag = float(r.home_goals), float(r.away_goals)
                fd = date.fromisoformat(str(r.date)[:10])
            except (TypeError, ValueError):
                continue
            results[(canonical_name(str(r.home)), canonical_name(str(r.away)))].append(
                (fd, hg, ag, match_key(r.date, r.home, r.away)))
    close_1x2, close_tot = closing_probs()

    out = []
    now = datetime.now(timezone.utc).isoformat()
    for d in decisions:
        fid = str(d["provider_fixture_id"])
        keyt = (fid, d["market"], d["side"])
        if keyt in already:
            continue
        match = _match_result(d, results)
        if match is None:
            continue
        hg, ag, mkey = match
        won = side_won(d["market"], d["side"], hg, ag)
        cmap = close_1x2 if d["market"] == "1x2" else close_tot
        pc = (cmap.get(mkey) or {}).get(d["side"])
        clv = None
        odds = float(d["odds_executed"])
        if pc and pc > 0 and odds > 1:
            clv = round(math.log(odds * pc), 5)
        out.append({
            "provider_fixture_id": fid, "settled_ts": now,
            "home_goals": hg, "away_goals": ag, "market": d["market"],
            "side": d["side"], "won": int(won),
            "pinnacle_close_devig": round(pc, 5) if pc else "",
            "clv": clv if clv is not None else "",
        })
    _append(SETTLEMENTS, SETTLEMENT_FIELDS, out)
    if verbose:
        print(f"  decision_ledger: settled {len(out)} decision(s)")
    return len(out)


def settled_bets() -> list[dict]:
    """Decisions joined to their settlements — the frozen basis for metrics."""
    settle_map = {(s["provider_fixture_id"], s["market"], s["side"]): s
                  for s in load_settlements()}
    out = []
    for d in load_decisions():
        s = settle_map.get((d["provider_fixture_id"], d["market"], d["side"]))
        if s is None:
            continue
        row = dict(d)
        row["won"] = int(s["won"])
        row["clv"] = float(s["clv"]) if s.get("clv") not in ("", None) else None
        # Closing prob of the settled side, frozen at settlement — the log-loss
        # benchmark, joined here so the backtest never re-derives a match key.
        row["p_close"] = (float(s["pinnacle_close_devig"])
                          if s.get("pinnacle_close_devig") not in ("", None) else None)
        for f in ("odds_executed", "p_model", "p_book_devig", "edge",
                  "decision_lead_min"):
            row[f] = float(d[f])
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.record:
        record()
    if args.settle:
        settle()
    if args.status or not (args.record or args.settle):
        d, s = load_decisions(), load_settlements()
        joined = settled_bets()
        print(f"decisions recorded : {len(d)}")
        print(f"settlements        : {len(s)}")
        print(f"settled bets       : {len(joined)}  (toward the 1000-bet gate)")
        if d:
            leads = [float(x["decision_lead_min"]) for x in d]
            print(f"decision leads     : {min(leads):.0f}-{max(leads):.0f} min "
                  f"(window {MIN_LEAD_MIN}-{MAX_LEAD_MIN})")


if __name__ == "__main__":
    main()
