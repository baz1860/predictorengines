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

    decision_ledger.csv          one immutable row per (fixture, market, side)
    decision_strategy_ledger.csv explicit compatibility metadata
    closing_market_ledger_v2.csv raw complete closing markets per bookmaker
    settlement_ledger.csv        appended result + legacy diagnostic CLV
    settlement_clv_v2.csv        power fair CLV + same-book raw price CLV

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
import fcntl
import hashlib
import json
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
DECISION_STRATEGIES = RUNTIME / "decision_strategy_ledger.csv"
IDENTITY_EXCLUSIONS = RUNTIME / "identity_exclusions.csv"

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
DECISION_STRATEGY_FIELDS = [
    "decision_id", "strategy_version", "strategy_manifest_hash",
    "identity_version", "pricing_code_hash", "recorded_at_utc",
]
IDENTITY_EXCLUSION_FIELDS = [
    "decision_id", "provider_fixture_id", "reviewed_at_utc", "action", "reason",
    "reviewed_identity_version",
]

# ── closing snapshot (self-sourced CLV reference) ──────────────────────────
# fd.co.uk only carries closing odds for the top European leagues, so every
# rest-of-world fixture we capture (K League, MLS, Brazil, …) settled with NO
# CLV and could never open the gate. We already pull multi-book odds from BSD at
# decision time, and BSD's odds_comparison is best-populated close to kickoff —
# so a second snapshot in the final minutes before kickoff gives a de-vigged
# consensus CLOSE for EVERY league BSD prices, at zero extra dependency.
CLOSING = RUNTIME / "closing_ledger.csv"
RAW_CLOSING = RUNTIME / "closing_market_ledger_v2.csv"
SETTLEMENT_CLV_V2 = RUNTIME / "settlement_clv_v2.csv"
# As near kickoff as a 15-min capture cadence allows: the first sighting inside
# this window is kept (idempotent), i.e. a quote from the fixture's final ~1–20
# minutes. Wide enough (19 min) to survive a skipped run.
CLOSE_MIN_LEAD = 1
CLOSE_MAX_LEAD = 20
CLOSING_FIELDS = [
    "provider_fixture_id", "close_ts", "market", "side",
    "p_close_devig", "close_lead_min",
]
RAW_CLOSING_FIELDS = [
    "close_id", "provider_fixture_id", "close_ts", "market", "source",
    "book", "odds_json", "overround", "proportional_probs_json",
    "power_probs_json", "power_k", "close_lead_min", "schema_version",
]
SETTLEMENT_CLV_V2_FIELDS = [
    "provider_fixture_id", "market", "side", "settled_ts", "close_source",
    "close_book", "close_odds", "raw_price_clv", "fair_close_probability",
    "fair_clv", "devig_method", "power_k_mean", "n_complete_books",
    "schema_version",
]
CLV_SCHEMA_VERSION = "raw_complete_market_v2"
CLV_DEVIG_METHOD = "power_consensus_v1"


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
    """Byte-level pricing-code provenance (not a compatibility boundary).

    Learned parameters deliberately do not belong here: a normal refit changes
    model_params.json and must not erase all accumulated forward evidence.
    Every code path that can change a priced or selected row belongs here for
    auditability. Compatibility is the explicit strategy contract; harmless
    byte changes therefore do not reset accumulated evidence.
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
    with path.open("a+", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        first = fh.readline().strip()
        if first and next(csv.reader([first])) != fields:
            raise ValueError(
                f"refusing to append {path.name}: header does not match the "
                "declared append-only schema"
            )
        fh.seek(0, os.SEEK_END)
        w = csv.DictWriter(fh, fieldnames=fields)
        if fh.tell() == 0:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
        fh.flush()
        os.fsync(fh.fileno())


def _append_unique(path: Path, fields: list[str], rows: list[dict],
                   key_fields: tuple[str, ...]) -> int:
    """Append rows atomically, de-duplicating immutable composite keys."""
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        reader = csv.DictReader(fh)
        if reader.fieldnames and reader.fieldnames != fields:
            raise ValueError(
                f"refusing to append {path.name}: header does not match the "
                "declared append-only schema"
            )
        seen = {tuple(str(r.get(k, "")) for k in key_fields) for r in reader}
        fresh = []
        for row in rows:
            key = tuple(str(row.get(k, "")) for k in key_fields)
            if key in seen:
                continue
            fresh.append(row)
            seen.add(key)
        fh.seek(0, os.SEEK_END)
        writer = csv.DictWriter(fh, fieldnames=fields)
        if fh.tell() == 0:
            writer.writeheader()
        for row in fresh:
            writer.writerow({k: row.get(k, "") for k in fields})
        fh.flush()
        os.fsync(fh.fileno())
        return len(fresh)


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
    from .runtime_safety import assert_writer_host
    assert_writer_host()
    from api_keys import get_key
    from bsd_client import get_all_events
    from . import model as M
    from .club_identity import canonical_name
    from .competitions import comp_from_bsd_league
    from .fetch import bsd_league_name
    from .snapshot_odds import odds_comparison
    from .strategy_contract import STRATEGY_VERSION, manifest_hash

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
    _append_unique(
        DECISION_STRATEGIES, DECISION_STRATEGY_FIELDS,
        [{
            "decision_id": row["decision_id"],
            "strategy_version": STRATEGY_VERSION,
            "strategy_manifest_hash": manifest_hash(),
            "identity_version": row["resolver_version"],
            "pricing_code_hash": row["code_hash"],
            "recorded_at_utc": row["decision_ts"],
        } for row in out],
        ("decision_id",),
    )
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


def load_closing() -> list[dict]:
    return _load(CLOSING)


def load_raw_closing() -> list[dict]:
    return _load(RAW_CLOSING)


def load_decision_strategies() -> list[dict]:
    return _load(DECISION_STRATEGIES)


def load_identity_exclusions() -> list[dict]:
    return _load(IDENTITY_EXCLUSIONS)


def review_identity(*, decision_id: str = "", provider_fixture_id: str = "",
                    action: str, reason: str) -> None:
    """Append an affected-row exclusion or a later explicit reinstatement."""
    from .runtime_safety import assert_writer_host
    assert_writer_host()
    if not decision_id and not provider_fixture_id:
        raise ValueError("decision_id or provider_fixture_id is required")
    if action not in {"exclude", "reinstate"}:
        raise ValueError("action must be 'exclude' or 'reinstate'")
    if not str(reason).strip():
        raise ValueError("an auditable reason is required")
    _append(IDENTITY_EXCLUSIONS, IDENTITY_EXCLUSION_FIELDS, [{
        "decision_id": decision_id,
        "provider_fixture_id": provider_fixture_id,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "reason": str(reason).strip(),
        "reviewed_identity_version": resolver_version(),
    }])


def load_settlement_clv_v2() -> list[dict]:
    return _load(SETTLEMENT_CLV_V2)


def _existing_closing_keys() -> set[tuple[str, str, str]]:
    return {(r["provider_fixture_id"], r["market"], r["side"])
            for r in load_closing()}


def _complete_book_markets(side_odds: dict[str, dict[str, float]]) -> list[dict]:
    """Auditable raw complete markets plus proportional and power diagnostics."""
    from .market_settlement import devig, power_devig

    sides = list(side_odds)
    books = set().union(*(set(v) for v in side_odds.values())) if side_odds else set()
    out = []
    for book in sorted(books):
        prices = {side: float(side_odds[side].get(book) or 0) for side in sides}
        if any(value <= 1 for value in prices.values()):
            continue
        prop = devig(prices)
        power, exponent = power_devig(prices)
        if not power or exponent is None:
            continue
        out.append({
            "book": book,
            "odds": prices,
            "overround": sum(1.0 / value for value in prices.values()) - 1.0,
            "proportional": prop,
            "power": power,
            "power_k": exponent,
        })
    return out


def capture_closing(api_key: str | None = None, verbose: bool = True) -> int:
    """Snapshot complete raw near-kickoff markets for decided fixtures.

    Runs in the same frequent capture pass as record()/settle(). For every
    fixture that (a) has a recorded decision and (b) is now inside the
    [CLOSE_MIN_LEAD, CLOSE_MAX_LEAD] pre-kickoff window, it de-vigs the
    complete markets are retained per book and the power and proportional
    de-vigs are recorded beside the raw odds. The legacy consensus ledger is
    still written for backward-compatible diagnostics; it cannot feed v3 CLV.

    This is the self-sourced CLV reference that lets EVERY BSD-priced league —
    not just the European ones fd.co.uk covers — earn closing-line evidence.
    """
    from .runtime_safety import assert_writer_host
    assert_writer_host()
    from api_keys import get_key
    from bsd_client import get_all_events

    from .snapshot_odds import odds_comparison

    key = api_key or get_key("bsd", env="BSD_API_KEY")
    if not key:
        if verbose:
            print("  decision_ledger: no BSD key — no closing snapshot")
        return 0
    want_fids = {str(d["provider_fixture_id"]) for d in load_decisions()}
    if not want_fids:
        return 0

    now = datetime.now(timezone.utc)
    try:
        events = get_all_events(key, status="notstarted",
                                date_from=str(now.date()), date_to=str(now.date()))
    except Exception as exc:
        if verbose:
            print(f"  decision_ledger: BSD fetch failed ({exc})")
        return 0

    seen = _existing_closing_keys()
    bsd_markets = {"1x2": {"home": "HOME", "draw": "DRAW", "away": "AWAY"},
                   "total25": {"over": "over@2.5", "under": "under@2.5"}}
    out: list[dict] = []
    raw_out: list[dict] = []
    for ev in events:
        fid = ev.get("id")
        if fid is None or str(fid) not in want_fids:
            continue
        ko = ev.get("event_date") or ev.get("date")
        try:
            kickoff = datetime.fromisoformat(str(ko).replace("Z", "+00:00"))
            lead = (kickoff - now).total_seconds() / 60.0
        except (ValueError, TypeError):
            continue
        if not (CLOSE_MIN_LEAD <= lead <= CLOSE_MAX_LEAD):
            continue
        try:
            cmp = odds_comparison(key, fid)
        except Exception:
            continue
        markets = (cmp.get("markets") or {}) if isinstance(cmp, dict) else {}
        for market, sides in bsd_markets.items():
            entry = markets.get("1x2") if market == "1x2" else markets.get("over_under_25")
            entry = entry or {}
            side_odds: dict[str, dict[str, float]] = {}
            for our_side, bsd_key in sides.items():
                books = (entry.get(bsd_key) or {}).get("bookmakers") or {}
                side_odds[our_side] = {
                    b: float(v["decimal_odds"]) for b, v in books.items()
                    if v.get("decimal_odds") and float(v["decimal_odds"]) > 1.0}
            complete = _complete_book_markets(side_odds)
            for item in complete:
                close_id = hashlib.sha256(
                    f"{fid}|{market}|{item['book']}".encode()
                ).hexdigest()[:24]
                raw_out.append({
                    "close_id": close_id,
                    "provider_fixture_id": str(fid),
                    "close_ts": now.isoformat(),
                    "market": market,
                    "source": "bsd_odds_comparison",
                    "book": item["book"],
                    "odds_json": json.dumps(item["odds"], sort_keys=True,
                                             separators=(",", ":")),
                    "overround": round(float(item["overround"]), 8),
                    "proportional_probs_json": json.dumps(
                        item["proportional"], sort_keys=True, separators=(",", ":")
                    ),
                    "power_probs_json": json.dumps(
                        item["power"], sort_keys=True, separators=(",", ":")
                    ),
                    "power_k": round(float(item["power_k"]), 8),
                    "close_lead_min": round(lead, 1),
                    "schema_version": CLV_SCHEMA_VERSION,
                })
            consensus = market_consensus_devig(side_odds, list(sides))
            if not consensus:
                continue
            for our_side in sides:
                keyc = (str(fid), market, our_side)
                if keyc in seen:
                    continue
                out.append({
                    "provider_fixture_id": str(fid), "close_ts": now.isoformat(),
                    "market": market, "side": our_side,
                    "p_close_devig": round(float(consensus[our_side]), 5),
                    "close_lead_min": round(lead, 1),
                })
                seen.add(keyc)
    _append(CLOSING, CLOSING_FIELDS, out)
    raw_n = _append_unique(
        RAW_CLOSING, RAW_CLOSING_FIELDS, raw_out, ("close_id",)
    )
    if verbose:
        print(f"  decision_ledger: captured {len(out)} closing probability row(s), "
              f"{raw_n} raw complete market(s)")
    return len(out)


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


def _raw_close_summary(fid: str, market: str, side: str, decision_book: str,
                       rows: list[dict]) -> dict | None:
    """Derive auditable same-book price CLV inputs and power-consensus fair p."""
    matches = [r for r in rows
               if str(r.get("provider_fixture_id")) == str(fid)
               and r.get("market") == market]
    probabilities, exponents = [], []
    same_book_odds = None
    for row in matches:
        try:
            probs = json.loads(row["power_probs_json"])
            odds = json.loads(row["odds_json"])
            probability = float(probs[side])
            exponent = float(row["power_k"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not (0 < probability < 1 and math.isfinite(exponent)):
            continue
        probabilities.append(probability)
        exponents.append(exponent)
        if str(row.get("book")) == str(decision_book):
            try:
                value = float(odds[side])
                same_book_odds = value if value > 1 else None
            except (KeyError, TypeError, ValueError):
                pass
    if not probabilities:
        return None
    return {
        "fair_probability": sum(probabilities) / len(probabilities),
        "power_k_mean": sum(exponents) / len(exponents),
        "n_complete_books": len(probabilities),
        "same_book_odds": same_book_odds,
        "close_book": decision_book if same_book_odds else "",
        "source": "bsd_complete_market",
    }


def settle(verbose: bool = True) -> int:
    """Append settlement rows for decisions whose fixtures have finished.

    The settlement ROW is keyed on provider_fixture_id + market + side (so the
    decision↔settlement join never changes), but the RESULT is looked up by
    canonical match identity, not by fixtures.csv's fixture_id — that id is a
    post-dedup provider id that disagrees with the recorded BSD event id for
    most matches. v3 CLV is scored from the frozen raw complete-market capture:
    a power-de-vigged consensus plus same-executing-book raw price movement.
    Legacy proportional/Pinnacle-derived CLV remains diagnostic only.
    """
    from .runtime_safety import assert_writer_host
    assert_writer_host()
    import pandas as pd

    from . import model as M
    from .club_identity import canonical_name
    from .market_settlement import closing_probs, match_key, raw_price_clv, side_won
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
    # BSD self-captured close takes precedence — it covers every league we bet.
    # fd.co.uk's closing map is the fallback for the European leagues where a
    # BSD close was not captured (e.g. odds not populated near kickoff).
    bsd_close = {(c["provider_fixture_id"], c["market"], c["side"]):
                 float(c["p_close_devig"])
                 for c in load_closing()
                 if c.get("p_close_devig") not in ("", None)}
    raw_closing = load_raw_closing()

    out = []
    clv_v2_out = []
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
        raw_summary = _raw_close_summary(
            fid, d["market"], d["side"], str(d.get("book") or ""), raw_closing
        )
        pc = raw_summary["fair_probability"] if raw_summary else None
        if not (pc and pc > 0):
            pc = bsd_close.get((fid, d["market"], d["side"]))
        if not (pc and pc > 0):
            cmap = close_1x2 if d["market"] == "1x2" else close_tot
            pc = (cmap.get(mkey) or {}).get(d["side"])
        clv = None
        odds = float(d["odds_executed"])
        if pc and pc > 0 and odds > 1:
            clv = round(math.log(odds * pc), 5)
        if raw_summary:
            close_odds = raw_summary["same_book_odds"]
            clv_v2_out.append({
                "provider_fixture_id": fid,
                "market": d["market"],
                "side": d["side"],
                "settled_ts": now,
                "close_source": raw_summary["source"],
                "close_book": raw_summary["close_book"],
                "close_odds": round(close_odds, 5) if close_odds else "",
                "raw_price_clv": (
                    round(raw_price_clv(odds, close_odds), 5)
                    if close_odds else ""
                ),
                "fair_close_probability": round(float(pc), 8),
                "fair_clv": clv if clv is not None else "",
                "devig_method": CLV_DEVIG_METHOD,
                "power_k_mean": round(raw_summary["power_k_mean"], 8),
                "n_complete_books": raw_summary["n_complete_books"],
                "schema_version": CLV_SCHEMA_VERSION,
            })
        out.append({
            "provider_fixture_id": fid, "settled_ts": now,
            "home_goals": hg, "away_goals": ag, "market": d["market"],
            "side": d["side"], "won": int(won),
            "pinnacle_close_devig": round(pc, 5) if pc else "",
            "clv": clv if clv is not None else "",
        })
    _append_unique(
        SETTLEMENT_CLV_V2, SETTLEMENT_CLV_V2_FIELDS, clv_v2_out,
        ("provider_fixture_id", "market", "side"),
    )
    _append(SETTLEMENTS, SETTLEMENT_FIELDS, out)
    if verbose:
        print(f"  decision_ledger: settled {len(out)} decision(s)")
    return len(out)


def settled_bets() -> list[dict]:
    """Decisions joined to their settlements — the frozen basis for metrics."""
    settle_map = {(s["provider_fixture_id"], s["market"], s["side"]): s
                  for s in load_settlements()}
    clv_v2_map = {(s["provider_fixture_id"], s["market"], s["side"]): s
                  for s in load_settlement_clv_v2()}
    strategy_map = {s["decision_id"]: s for s in load_decision_strategies()}
    exclusions_by_decision, exclusions_by_fixture = {}, {}
    for exclusion_row in load_identity_exclusions():
        if exclusion_row.get("decision_id"):
            exclusions_by_decision[str(exclusion_row["decision_id"])] = exclusion_row
        if exclusion_row.get("provider_fixture_id"):
            exclusions_by_fixture[str(exclusion_row["provider_fixture_id"])] = exclusion_row
    from .strategy_contract import (
        STRATEGY_VERSION, manifest_hash, version_for_legacy_code_hash,
    )

    out = []
    for d in load_decisions():
        s = settle_map.get((d["provider_fixture_id"], d["market"], d["side"]))
        if s is None:
            continue
        row = dict(d)
        row["won"] = int(s["won"])
        legacy_clv = float(s["clv"]) if s.get("clv") not in ("", None) else None
        v2 = clv_v2_map.get((d["provider_fixture_id"], d["market"], d["side"]))
        row["legacy_clv"] = legacy_clv
        row["clv"] = (float(v2["fair_clv"])
                      if v2 and v2.get("fair_clv") not in ("", None) else None)
        row["raw_price_clv"] = (
            float(v2["raw_price_clv"])
            if v2 and v2.get("raw_price_clv") not in ("", None) else None
        )
        row["clv_method"] = v2.get("devig_method", "") if v2 else ""
        row["clv_schema_version"] = v2.get("schema_version", "") if v2 else ""
        row["close_source"] = v2.get("close_source", "") if v2 else ""
        row["close_odds"] = (
            float(v2["close_odds"])
            if v2 and v2.get("close_odds") not in ("", None) else None
        )
        # Closing prob of the settled side, frozen at settlement — the log-loss
        # benchmark, joined here so the backtest never re-derives a match key.
        row["legacy_p_close"] = (float(s["pinnacle_close_devig"])
                                 if s.get("pinnacle_close_devig") not in ("", None)
                                 else None)
        row["p_close"] = (
            float(v2["fair_close_probability"])
            if v2 and v2.get("fair_close_probability") not in ("", None) else None
        )
        meta = strategy_map.get(str(d.get("decision_id") or ""))
        row["strategy_version"] = (
            str(meta.get("strategy_version")) if meta
            else version_for_legacy_code_hash(d.get("code_hash"))
        )
        row["strategy_manifest_hash"] = (
            str(meta.get("strategy_manifest_hash") or "") if meta
            else (manifest_hash()
                  if row["strategy_version"] == STRATEGY_VERSION else "")
        )
        exclusion = (exclusions_by_decision.get(str(d.get("decision_id") or ""))
                     or exclusions_by_fixture.get(str(d.get("provider_fixture_id") or "")))
        row["identity_excluded"] = bool(
            exclusion and exclusion.get("action", "exclude") == "exclude"
        )
        row["identity_exclusion_reason"] = (
            str(exclusion.get("reason") or "") if row["identity_excluded"] else ""
        )
        for f in ("odds_executed", "p_model", "p_book_devig", "edge",
                  "decision_lead_min"):
            row[f] = float(d[f])
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--close", action="store_true",
                    help="snapshot raw complete near-kickoff BSD markets")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--status", action="store_true")
    review = ap.add_mutually_exclusive_group()
    review.add_argument("--exclude-decision", metavar="DECISION_ID")
    review.add_argument("--exclude-fixture", metavar="PROVIDER_FIXTURE_ID")
    review.add_argument("--reinstate-decision", metavar="DECISION_ID")
    review.add_argument("--reinstate-fixture", metavar="PROVIDER_FIXTURE_ID")
    ap.add_argument("--reason", help="required audit reason for identity review")
    args = ap.parse_args()
    if args.record:
        record()
    if args.close:
        capture_closing()
    if args.settle:
        settle()
    review_value = (args.exclude_decision or args.exclude_fixture
                    or args.reinstate_decision or args.reinstate_fixture)
    if review_value:
        if not args.reason:
            ap.error("--reason is required for identity exclusion/reinstatement")
        review_identity(
            decision_id=(args.exclude_decision or args.reinstate_decision or ""),
            provider_fixture_id=(args.exclude_fixture or args.reinstate_fixture or ""),
            action=("exclude" if args.exclude_decision or args.exclude_fixture
                    else "reinstate"),
            reason=args.reason,
        )
    if args.status or not (args.record or args.close or args.settle or review_value):
        d, s = load_decisions(), load_settlements()
        joined = settled_bets()
        clv_scored = sum(1 for b in joined if b.get("clv") is not None)
        raw_scored = sum(1 for b in joined if b.get("raw_price_clv") is not None)
        legacy_scored = sum(1 for b in joined if b.get("legacy_clv") is not None)
        print(f"decisions recorded : {len(d)}")
        print(f"legacy close rows  : {len(load_closing())}")
        print(f"raw close markets  : {len(load_raw_closing())}")
        print(f"settlements        : {len(s)}")
        print(f"settled bets       : {len(joined)}  (toward the 1000-bet gate)")
        print(f"  v3 fair CLV      : {clv_scored}")
        print(f"  v3 raw price CLV : {raw_scored}")
        print(f"  legacy CLV only  : {legacy_scored}")
        if d:
            leads = [float(x["decision_lead_min"]) for x in d]
            print(f"decision leads     : {min(leads):.0f}-{max(leads):.0f} min "
                  f"(window {MIN_LEAD_MIN}-{MAX_LEAD_MIN})")


if __name__ == "__main__":
    main()
