#!/usr/bin/env python3
"""Club soccer edge finder: 1X2, over/under 2.5, and BTTS."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import model as M
from api_keys import get_key

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ODDS_CSV = DATA / "odds.csv"
REPORT = DATA / "edge_report.csv"
CACHE = DATA / "bsd_cache"
KELLY_FRACTION = 0.25
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
ODDS_API_SPORTS = {
    "Premier League": "soccer_epl",
    "Championship": "soccer_efl_champ",
    "Bundesliga": "soccer_germany_bundesliga",
    "Serie A": "soccer_italy_serie_a",
    "Ligue 1": "soccer_france_ligue_one",
    "La Liga": "soccer_spain_la_liga",
    "Champions League": "soccer_uefa_champs_league",
    "Europa League": "soccer_uefa_europa_league",
    "Conference League": "soccer_uefa_europa_conference_league",
}

MARKETS = {
    "1x2": [("home", None), ("draw", None), ("away", None)],
    "total": [("over", 2.5), ("under", 2.5)],
    "btts": [("yes", None), ("no", None)],
}


# Manual odds.csv is never staked past this age — a stale quote on a settled
# match must never reach a staking table (season.py card OR app recording).
MANUAL_ODDS_MAX_AGE_DAYS = 2.0
# Live quotes carry a per-quote `quoted_at_utc` timestamp and expire fast: a
# cached/replayed live frame must not price a card hours later.
LIVE_QUOTE_MAX_AGE_HOURS = 6.0
# A quote whose timestamp is only the FETCH time (quote_time_source ==
# "fetch_time_only", e.g. BSD, which exposes no provider update time) proves
# nothing about how stale the bookmaker's own price is. It is stakeable only
# as an immediate live observation — fetched moments ago — never as a
# several-hour-old cached frame. This 2-minute lifetime is the DEFAULT for any
# live quote: only an allowlisted, trustworthy provider-timestamp source may
# claim the longer LIVE_QUOTE_MAX_AGE_HOURS window (fail-closed on missing,
# misspelled or unknown provenance).
FETCH_TIME_ONLY_MAX_AGE_MINUTES = 2.0
# quote_time_source values whose timestamp reflects the provider's own last
# price update (not merely our fetch time) and therefore earn the 6h window.
PROVIDER_TIMESTAMP_SOURCES = frozenset({"provider_last_update"})


def validate_quotes(odds: pd.DataFrame, source: str = "manual",
                    now: datetime | None = None,
                    max_manual_age_days: float = MANUAL_ODDS_MAX_AGE_DAYS,
                    ) -> tuple[pd.DataFrame, list[str]]:
    """The one quote-validation gate for every pricing entry point
    (season.py card, engine.cmd_edge, the app adapter's record path).

    Returns (filtered_quotes, issues). Rules:
    - quotes on past or undated fixtures are dropped — a settled or in-play
      match must never be priced as a bet;
    - when a `quoted_at_utc` column is present (stamped by the live
      fetchers), each quote is age-checked individually: manual quotes
      expire after `max_manual_age_days`, live quotes after
      LIVE_QUOTE_MAX_AGE_HOURS; un-timestamped rows in a timestamped frame
      are dropped;
    - for source="manual" without timestamps, the whole file is rejected
      when odds.csv's mtime is older than `max_manual_age_days` (weak
      proxy — trivially defeated by `touch` — which is why timestamped
      quotes take precedence whenever available)."""
    issues: list[str] = []
    if odds is None or odds.empty:
        return odds, issues
    now = now or datetime.now(timezone.utc)
    now_ts = pd.Timestamp(now)
    if "quoted_at_utc" in odds.columns:
        q = pd.to_datetime(odds["quoted_at_utc"], utc=True, errors="coerce",
                           format="mixed")
        if source == "manual":
            max_age = pd.Series(pd.Timedelta(days=max_manual_age_days),
                                index=odds.index)
        else:
            # Live quotes default to the strict 2-minute fetch-time lifetime.
            # Missing/misspelled/unknown provenance therefore fails closed;
            # only an allowlisted provider-timestamp source earns the 6h window.
            max_age = pd.Series(
                pd.Timedelta(minutes=FETCH_TIME_ONLY_MAX_AGE_MINUTES),
                index=odds.index)
            if "quote_time_source" in odds.columns:
                trusted = odds["quote_time_source"].astype(str).isin(
                    PROVIDER_TIMESTAMP_SOURCES)
                max_age[trusted] = pd.Timedelta(hours=LIVE_QUOTE_MAX_AGE_HOURS)
        age = now_ts - q
        # A future-dated timestamp is corrupt provenance, not freshness:
        # its negative age must fail, with only a small clock-skew tolerance.
        skew = pd.Timedelta(minutes=5)
        fresh = q.notna() & (age >= -skew) & (age <= max_age)
        n_stale = int((~fresh).sum())
        if n_stale:
            issues.append(f"dropped {n_stale} quote(s) with stale, missing, or "
                          "future-dated quoted_at_utc (live quotes without an "
                          "allowlisted provider timestamp expire after "
                          f"{FETCH_TIME_ONLY_MAX_AGE_MINUTES:g}m)")
        odds = odds[fresh]
        if odds.empty:
            return odds.copy(), issues
    elif source == "manual":
        try:
            age_days = (now.timestamp() - ODDS_CSV.stat().st_mtime) / 86400.0
        except FileNotFoundError:
            age_days = None
        if age_days is not None and age_days > max_manual_age_days:
            issues.append(f"manual odds.csv is {age_days:.1f} days old (limit "
                          f"{max_manual_age_days:g}d) — stale quotes are never staked")
            return odds.iloc[0:0].copy(), issues
    # format="mixed": a frame mixing "2026-07-19 12:00:00" and "2026-07-20"
    # must not silently NaT-out (and thereby drop) the second style.
    d = pd.to_datetime(odds["date"], errors="coerce", format="mixed")
    today_ts = pd.Timestamp(now.date())
    n_past = int((d.isna() | (d < today_ts)).sum())
    if n_past:
        issues.append(f"dropped {n_past} quote(s) on past or undated fixtures")
    odds = odds[d.notna() & (d >= today_ts)].copy()
    # kickoff_utc closes the date-only granularity hole: a 12:00 UTC kickoff
    # must not price at 23:00 the same day. LIVE quotes are fetched from the
    # provider event that carries kickoff, so a live row without a parseable
    # future kickoff is malformed and non-stakeable. Manual rows without a
    # kickoff keep the (age-gated, opt-in) date-only rule above.
    if not odds.empty:
        if "kickoff_utc" in odds.columns:
            k = pd.to_datetime(odds["kickoff_utc"], utc=True, errors="coerce",
                               format="mixed")
        else:
            # tz-aware NaT series: a naive one cannot be compared to now.
            k = pd.Series(pd.NaT, index=odds.index).dt.tz_localize("UTC")
        if source == "live":
            bad = k.isna() | (k <= now_ts)
        else:
            bad = k.notna() & (k <= now_ts)
        n_ko = int(bad.sum())
        if n_ko:
            issues.append(f"dropped {n_ko} quote(s) already kicked off or "
                          "missing kickoff_utc"
                          + (" (required for live quotes)" if source == "live" else ""))
        odds = odds[~bad].copy()
    return odds, issues


def devig(odds: list[float]) -> np.ndarray:
    inv = np.array([1.0 / float(o) for o in odds])
    return inv / inv.sum()


def kelly(p: float, odds: float) -> float:
    b = odds - 1.0
    return max(0.0, (p * b - (1.0 - p)) / b)


def side_prob(pred: dict, market: str, side: str) -> float:
    p = pred["probs"]
    if market == "1x2":
        return float(p[side])
    if market == "total":
        return float(p["over25" if side == "over" else "under25"])
    if market == "btts":
        return float(p["btts_yes" if side == "yes" else "btts_no"])
    raise ValueError(f"Unknown market: {market}")


def load_odds(path: Path = ODDS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Use --template or API odds.")
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df["line"] = pd.to_numeric(df.get("line", np.nan), errors="coerce")
    return df.dropna(subset=["odds"]).copy()


def write_template(path: Path = ODDS_CSV) -> None:
    fixtures = M.upcoming(M.load_fixtures())
    rows = []
    for r in fixtures.itertuples(index=False):
        for market, sides in MARKETS.items():
            for side, line in sides:
                rows.append({"date": r.date.date(), "competition": r.competition,
                             "home": r.home, "away": r.away, "market": market,
                             "side": side, "line": "" if line is None else line,
                             "odds": ""})
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def rows_from_odds(odds: pd.DataFrame, model_name: str = "ensemble",
                   bankroll: float = 100.0, calib_maps=None,
                   player_adj_map: dict | None = None,
                   apply_do_not_bet: bool = False,
                   market_blend: bool | None = None) -> list[dict]:
    """Compute edge rows from priced odds.

    Odds may carry a ``bookmaker`` column (absent → treated as one book,
    "manual"). Each book is de-vigged independently and only if it quotes the
    market's complete outcome set; the output has exactly one row per outcome
    (best executable price, edge vs that book's own de-vigged probability,
    cross-book mean reported as p_consensus), so an outcome is never staked
    more than once per match.

    player_adj_map, if provided, maps (home_lower, away_lower, comp) →
    player_adj dict from PlayerFeatureStore, e.g.:
        {("arsenal", "chelsea", "Premier League"):
            {"home": {"attack_mult": 0.88, ...}, "away": {...}}}

    apply_do_not_bet: when True, run market_model.do_not_bet on each row
    (P6.2). A suppressed row keeps its edge/EV for visibility but its stake
    is zeroed and `suppressed_reason` is filled — excluded from the card and
    from staking, never from the report.

    market_blend: explicit True/False from an interactive caller. None uses
    the validated production default. Blending is owned here so it can happen
    exactly once, before the final evidence gate.
    """
    out = []
    valid_models = {"ensemble", "goals", "elo", "xg"}
    if model_name not in valid_models:
        raise ValueError(
            f"Unknown model {model_name!r}; choose one of {sorted(valid_models)}"
        )
    params = M.load_params()
    odds = odds.copy()
    if "bookmaker" not in odds.columns:
        odds["bookmaker"] = "manual"
    odds["bookmaker"] = (odds["bookmaker"].fillna("manual").astype(str)
                         .str.strip().str.lower().replace("", "manual"))
    # Normalize text and the line up front: mixed-case sides must not silently
    # change grouping, and a blank-string line must never reach float().
    odds["market"] = odds["market"].astype(str).str.strip().str.lower()
    odds["side"] = odds["side"].astype(str).str.strip().str.lower()
    if "line" in odds.columns:
        line_num = pd.to_numeric(odds["line"], errors="coerce")
    else:
        line_num = pd.Series(np.nan, index=odds.index)
    odds["_line_key"] = line_num.map(lambda v: "" if pd.isna(v) else f"{v:g}")
    skipped = {"unknown_market_or_line": 0, "duplicate_side_book": 0,
               "incomplete_book": 0, "unknown_team": 0}
    # One fixture is normally represented by three market groups (1X2,
    # total, BTTS). Prediction is fixture-level, so price it once and reuse it
    # for every market.
    prediction_cache: dict[tuple[str, str, str, str], dict] = {}
    for (date, comp, home, away, market, line_key), grp in odds.groupby(
            ["date", "competition", "home", "away", "market", "_line_key"],
            dropna=False):
        # The line is part of the market identity: `over 2.5` and `under 3.5`
        # are different markets and must never be de-vigged as a pair. Only
        # the exact lines MARKETS supports are priced.
        expected_sides = {s for s, l in MARKETS.get(str(market), [])
                          if line_key == ("" if l is None else f"{l:g}")}
        if not expected_sides:
            skipped["unknown_market_or_line"] += len(grp)
            continue
        priced = grp[np.isfinite(grp["odds"]) & (grp["odds"] > 1.0)]
        # De-vig each bookmaker separately. Normalizing several books' prices
        # as one market deflates every implied probability and manufactures
        # fake edges. A book only counts if it quotes the complete outcome
        # set exactly once per side (a duplicate side means a stale update or
        # corrupt join — reject the book rather than guess which price is
        # real); the card then carries ONE row per outcome — best executable
        # price across books, edge measured against that book's own de-vigged
        # probability — so the same outcome is never staked twice.
        book_probs: dict[str, list[float]] = {}
        best_quote: dict[str, tuple[float, pd.Series, float]] = {}
        for _, bg in priced.groupby("bookmaker"):
            sides = bg["side"].tolist()
            if len(sides) != len(set(sides)):
                skipped["duplicate_side_book"] += 1
                continue
            if set(sides) != expected_sides:
                skipped["incomplete_book"] += 1
                continue
            implied = devig(bg["odds"].tolist())
            for (_, r), p_bk in zip(bg.iterrows(), implied):
                side = str(r["side"])
                book_probs.setdefault(side, []).append(float(p_bk))
                o = float(r["odds"])
                if side not in best_quote or o > best_quote[side][0]:
                    best_quote[side] = (o, r, float(p_bk))
        if not best_quote:
            continue

        # Player availability adjustment for this match
        p_adj = None
        p_adj_meta: dict = {}
        lineup_confidence = 1.0
        if player_adj_map is not None:
            p_adj = player_adj_map.get((str(home).lower(), str(away).lower(), str(comp)))
            if p_adj:
                h_a = p_adj.get("home", {})
                a_a = p_adj.get("away", {})
                lineup_confidence = float(p_adj.get("lineup_confidence", 1.0))
                p_adj_meta = {
                    "player_adj_home": round(float(h_a.get("attack_mult", 1.0)), 4),
                    "player_adj_away": round(float(a_a.get("attack_mult", 1.0)), 4),
                    "def_adj_home":    round(float(h_a.get("defense_mult", 1.0)), 4),
                    "def_adj_away":    round(float(a_a.get("defense_mult", 1.0)), 4),
                    "n_missing_home":  int(h_a.get("n_missing", 0)),
                    "n_missing_away":  int(a_a.get("n_missing", 0)),
                    "lineup_confidence": round(lineup_confidence, 3),
                }

        fixture_key = (str(date), str(comp), str(home), str(away))
        pred = prediction_cache.get(fixture_key)
        if pred is None:
            try:
                pred = M.predict_match(home, away, comp, str(date), model_name,
                                       params=params, player_adj=p_adj)
            except ValueError as exc:
                if str(exc).startswith("Unknown team:"):
                    skipped["unknown_team"] += 1
                    continue
                raise
            if calib_maps is not None:
                from .calibrate import apply as _apply_calib
                ph, pdr, pa = _apply_calib(
                    pred["probs"]["home"], pred["probs"]["draw"],
                    pred["probs"]["away"], calib_maps
                )
                pred["probs"]["home"], pred["probs"]["draw"], pred["probs"]["away"] = \
                    ph, pdr, pa
            prediction_cache[fixture_key] = pred
        # P0 evidence coverage. Report-only by design: pricing continues
        # unchanged so the build can be measured against live behaviour, but
        # every suggestion carries the tier of its least-evidenced side so a
        # Sturm-Graz-shaped price is never presented as if it were a Premier
        # League one. Stake suppression is deliberately NOT applied here.
        cov = pred.get("coverage") or {}
        cov_meta = {
            "evidence_tier": str(cov.get("tier", "unknown")),
            "evidence_ok": bool(cov.get("reliable", False)),
            "evidence_note": " | ".join(cov.get("notes") or []),
            "n_matches_home": int((cov.get("home") or {}).get("n", 0)),
            "n_matches_away": int((cov.get("away") or {}).get("n", 0)),
        }
        for side in sorted(best_quote):
            o, r, p_book = best_quote[side]
            # Edge is measured against the EXECUTING book's own de-vigged
            # probability. Measuring against a cross-book mean while taking
            # the best price manufactures EV out of pure line-shopping: with
            # books at fair 49% and 44%, a model equal to the 46.5% mean
            # shows +2.4% "EV" at the better price. The cross-book mean is
            # kept as p_consensus for reporting only.
            p_consensus = float(np.mean(book_probs[side]))
            p_model = side_prob(pred, str(r["market"]), side)
            ev = p_model * o - 1.0
            # Haircut the stake (not the EV/edge display) by lineup-read
            # confidence — a shaky lineup should bet smaller, not look less +EV.
            kfrac = KELLY_FRACTION * kelly(p_model, o) * lineup_confidence
            # line_key is the already-normalized group line ("" = no line).
            line = "" if line_key == "" else float(line_key)
            label = bet_label(home, away, str(r["market"]), side, line)
            row = {"date": str(date), "competition": comp,
                   "match": f"{home} v {away}", "home": home, "away": away,
                   "model": model_name,
                   "market": str(r["market"]), "side": side, "line": line,
                   "bet": label, "odds": round(o, 3),
                   "p_model": round(float(p_model), 3),
                   "p_book": round(float(p_book), 3),
                   "p_consensus": round(p_consensus, 3),
                   "edge": round(float(p_model - p_book), 3),
                   "ev_per_unit": round(float(ev), 3),
                   "kelly_stake": round(float(kfrac), 4),
                   "stake_gbp": round(float(kfrac) * bankroll, 2)}
            row.update(cov_meta)
            if p_adj_meta:
                row.update(p_adj_meta)
            if apply_do_not_bet:
                from . import market_model as MM
                dnb_market = {"1x2": "1x2", "total": "total25"}.get(str(r["market"]))
                if dnb_market:
                    decision = MM.do_not_bet({**row, "market": dnb_market})
                    if decision["suppress"]:
                        row["suppressed_reason"] = decision["reason"]
                        row["kelly_stake"] = 0.0
                        row["stake_gbp"] = 0.0
            out.append(row)
    dropped = {k: v for k, v in skipped.items() if v}
    if dropped:
        print(f"  rows_from_odds: rejected odds groups {dropped}")
    out.sort(key=lambda x: -x["ev_per_unit"])
    # A fitted weight alone never changes production pricing. The shared
    # registry must validate the default, or an interactive caller must opt in.
    # This is a required in-repo dependency: if an enabled blend is malformed,
    # fail visibly instead of silently pricing a different model.
    from app.market_blend import apply_blend_to_rows, is_default_on
    blend_enabled = (is_default_on("club_soccer")
                     if market_blend is None else bool(market_blend))
    if blend_enabled:
        apply_blend_to_rows(out, "club_soccer", bankroll, KELLY_FRACTION,
                            kelly_key="kelly_stake")
        out.sort(key=lambda x: -x["ev_per_unit"])
        if not is_default_on("club_soccer"):
            for row in out:
                row["strategy_variant"] = "experimental market blend"
    if model_name != "ensemble":
        for row in out:
            row["strategy_variant"] = f"ungated model {model_name}"
    # Evidence gate runs LAST — after every probability/stake transformation.
    # The market blend above rewrites kelly_stake/stake_gbp, so zeroing before
    # it would let the blend silently reopen stakes on gated rows. Any later
    # consumer that mutates stakes must call apply_evidence_gate again.
    apply_evidence_gate(out)
    return out


# Edge-row market string -> evidence-gate market key. BTTS is deliberately
# absent: it has no CLV reference and is never staked (display-only).
_GATE_MARKET = {"1x2": "1x2", "total": "total_over_under_2_5"}


def apply_evidence_gate(rows: list[dict]) -> bool:
    """Zero a stake unless ITS market's evidence gate is open. Per-market: a
    proven 1X2 book can stake while an unproven (or CLV-less) OU2.5 stays
    zeroed, and BTTS — which has no CLV reference — is never stakeable.

    Idempotent and fail-CLOSED: call it as the FINAL step after any code that
    rewrites kelly_stake/stake_gbp. Returns True if ANY market is open. Ends
    with a hard invariant: a closed market and a nonzero stake never coexist."""
    try:
        from .evidence_gate import (market_league_staking_allowed,
                                     market_staking_allowed)
        market_open = market_staking_allowed()
        league_open = market_league_staking_allowed()
    except Exception as exc:
        market_open, league_open = {}, {}
        print(f"  evidence gate import failed ({exc}) — all stakes zeroed")

    def _stands(row) -> bool:
        """A stake may stand only if its MARKET is open and — when the artifact
        carries per-league evidence — its LEAGUE is open too. With no per-league
        data the market-level gate decides (backward-compatible)."""
        if row.get("strategy_variant"):
            return False
        gkey = _GATE_MARKET.get(str(row.get("market")))
        if gkey is None or not market_open.get(gkey):
            return False
        return bool(league_open.get((gkey, str(row.get("competition"))), False))

    zeroed = 0
    for row in rows:
        if _stands(row):
            continue                       # market (and league) open — stake stands
        # market/league closed (or ungateable, e.g. BTTS): no stake may stand
        if float(row.get("kelly_stake") or 0) or float(row.get("stake_gbp") or 0):
            row["kelly_stake"] = 0.0
            row["stake_gbp"] = 0.0
            zeroed += 1
        row.setdefault("suppressed_reason", "evidence-gate: no demonstrated edge")
    any_open = any(market_open.values())
    if zeroed:
        print(f"  evidence gate: {zeroed} stake(s) zeroed "
              f"(open markets: {[k for k, v in market_open.items() if v] or 'none'})")
    # Unconditional runtime invariant — a closed market with a nonzero stake is
    # a money-losing state and must crash in every interpreter mode.
    for row in rows:
        if not _stands(row) and (
                float(row.get("kelly_stake") or 0) or float(row.get("stake_gbp") or 0)):
            raise RuntimeError("closed evidence-gate market with nonzero stake")
    return any_open


def bet_label(home: str, away: str, market: str, side: str, line) -> str:
    if market == "1x2":
        return {"home": f"{home} win", "draw": "Draw", "away": f"{away} win"}[side]
    if market == "total":
        try:
            ln = f"{float(line):g}"
        except (TypeError, ValueError):
            ln = "?"           # never crash pricing over a display label
        return f"{'Over' if side == 'over' else 'Under'} {ln} goals"
    if market == "btts":
        return "Both teams to score" if side == "yes" else "BTTS no"
    return f"{market} {side}"


def _decimal(v: object) -> float | None:
    """Parse a decimal odds value; return None if invalid."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if f > 1.0 else None
    except (TypeError, ValueError):
        return None


def _map_api_bet(bet_name: str, value_name: str, odd,
                 home: str = "", away: str = "") -> tuple[str, str, float | str, float] | None:
    """Map a bookmaker-API bet/value pair to our (market, side, line, odds) shape."""
    bet = bet_name.strip().lower()
    value = value_name.strip().lower()
    decimal = _decimal(odd)
    if decimal is None:
        return None
    if bet in {"match winner", "fulltime result", "1x2", "winner"}:
        if value in {"home", "1", home.strip().lower()}:
            return ("1x2", "home", "", decimal)
        if value in {"draw", "x"}:
            return ("1x2", "draw", "", decimal)
        if value in {"away", "2", away.strip().lower()}:
            return ("1x2", "away", "", decimal)
    if "over/under" in bet or bet in {"goals over/under", "total goals"}:
        parts = value.replace("goals", "").split()
        if len(parts) >= 2 and parts[0] in {"over", "under"}:
            try:
                line = float(parts[1])
            except ValueError:
                return None
            if abs(line - 2.5) < 1e-9:
                return ("total", parts[0], line, decimal)
    if bet in {"both teams score", "both teams to score", "btts"}:
        if value in {"yes", "no"}:
            return ("btts", value, "", decimal)
    return None


def fetch_bsd_odds(api_key: str | None = None) -> pd.DataFrame:
    """Fetch BSD odds for upcoming local fixtures.

    BSD embeds 1X2 odds directly in each event response
    (``odds_home``, ``odds_draw``, ``odds_away``).  Over/under 2.5 and
    BTTS odds are read from common BSD field names if present
    (``odds_over25``, ``odds_under25``, ``odds_btts_yes``, ``odds_btts_no``).

    BSD key: data/api_keys.json -> "bsd", or env BSD_API_KEY.
    Register free at https://sports.bzzoiro.com/register/
    """
    from bsd_client import get_all_events, league_name as bsd_league_name, \
        event_date_utc
    from .competitions import comp_from_bsd_league

    key = api_key or get_key("bsd", env="BSD_API_KEY")
    if not key:
        raise ValueError(
            "No BSD key. Register at https://sports.bzzoiro.com/register/ "
            "and add 'bsd' to data/api_keys.json, or set BSD_API_KEY."
        )

    # Match BSD events to our upcoming fixtures by (home, away, competition)
    fixtures = M.upcoming(M.load_fixtures())
    if fixtures.empty:
        raise ValueError("No upcoming fixtures found in fixtures.csv.")

    # Build lookup: (home_lower, away_lower, comp) -> fixture row
    fixture_lookup: dict[tuple[str, str, str], object] = {}
    for fx in fixtures.itertuples(index=False):
        key_t = (str(fx.home).lower(), str(fx.away).lower(), str(fx.competition))
        fixture_lookup[key_t] = fx

    # BSD's canonical enum is notstarted. bsd_client also accepts the human
    # alias, but keep the wire value explicit here because a silent empty
    # response is worse than a visible fetch failure.
    events = get_all_events(key, status="notstarted")
    CACHE.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ev in events:
        comp = comp_from_bsd_league(bsd_league_name(ev))
        if comp is None:
            continue
        home_raw = str(ev.get("home_team") or "")
        away_raw = str(ev.get("away_team") or "")
        lookup_key = (home_raw.lower(), away_raw.lower(), comp.name)
        fixture = fixture_lookup.get(lookup_key)
        if fixture is None:
            # Try partial match (BSD names may differ slightly)
            for (fh, fa, fc), fx in fixture_lookup.items():
                if fc == comp.name and (fh in home_raw.lower() or home_raw.lower() in fh):
                    if fa in away_raw.lower() or away_raw.lower() in fa:
                        fixture = fx
                        break
        if fixture is None:
            continue

        date = getattr(fixture, "date", None)
        if hasattr(date, "date"):
            date = date.date()

        # 1X2 odds (always embedded in BSD events)
        odds_h = _decimal(ev.get("odds_home"))
        odds_d = _decimal(ev.get("odds_draw"))
        odds_a = _decimal(ev.get("odds_away"))

        # Over/under 2.5 and BTTS (BSD may provide these as top-level fields)
        odds_over = _decimal(ev.get("odds_over_25") or ev.get("odds_over25")
                             or ev.get("odds_over_2_5"))
        odds_under = _decimal(ev.get("odds_under_25") or ev.get("odds_under25")
                              or ev.get("odds_under_2_5"))
        odds_btts_y = _decimal(ev.get("odds_btts_yes") or ev.get("odds_btts"))
        odds_btts_n = _decimal(ev.get("odds_btts_no"))

        # Also try nested odds dict if present
        nested = ev.get("odds") or {}
        if isinstance(nested, dict):
            if odds_h is None:
                odds_h = _decimal(nested.get("home") or nested.get("1"))
            if odds_d is None:
                odds_d = _decimal(nested.get("draw") or nested.get("x"))
            if odds_a is None:
                odds_a = _decimal(nested.get("away") or nested.get("2"))
            if odds_over is None:
                odds_over = _decimal(nested.get("over25") or nested.get("over_2_5"))
            if odds_under is None:
                odds_under = _decimal(nested.get("under25") or nested.get("under_2_5"))
            if odds_btts_y is None:
                odds_btts_y = _decimal(nested.get("btts_yes"))
            if odds_btts_n is None:
                odds_btts_n = _decimal(nested.get("btts_no"))

        # Kickoff comes from the LIVE event itself — the legacy fixture row
        # predates the kickoff_utc column and would leave this blank.
        kickoff = str(event_date_utc(ev) or "") \
            or str(getattr(fixture, "kickoff_utc", "") or "")
        base = {
            "date": date,
            "kickoff_utc": kickoff,
            "competition": comp.name,
            "home": getattr(fixture, "home", home_raw),
            "away": getattr(fixture, "away", away_raw),
            "bookmaker": "bsd",
            # BSD exposes no per-quote update time, so this is FETCH time —
            # a stale provider feed would still look fresh. The provenance
            # label lets a stronger future gate refuse fetch-time-only quotes.
            "quoted_at_utc": datetime.now(timezone.utc).isoformat(),
            "quote_time_source": "fetch_time_only",
        }
        if odds_h is not None:
            rows.append({**base, "market": "1x2", "side": "home", "line": "", "odds": odds_h})
        if odds_d is not None:
            rows.append({**base, "market": "1x2", "side": "draw", "line": "", "odds": odds_d})
        if odds_a is not None:
            rows.append({**base, "market": "1x2", "side": "away", "line": "", "odds": odds_a})
        if odds_over is not None:
            rows.append({**base, "market": "total", "side": "over", "line": 2.5, "odds": odds_over})
        if odds_under is not None:
            rows.append({**base, "market": "total", "side": "under", "line": 2.5, "odds": odds_under})
        if odds_btts_y is not None:
            rows.append({**base, "market": "btts", "side": "yes", "line": "", "odds": odds_btts_y})
        if odds_btts_n is not None:
            rows.append({**base, "market": "btts", "side": "no", "line": "", "odds": odds_btts_n})

    if not rows:
        raise ValueError(
            "No BSD odds matched upcoming fixtures. "
            "Check your BSD key, or use manual club_soccer/data/odds.csv."
        )
    return pd.DataFrame(rows)


def fetch_the_odds_api(api_key: str | None = None) -> pd.DataFrame:
    """Best-effort The Odds API fallback for competitions it publishes."""
    key = api_key or get_key("the-odds-api", env="THE_ODDS_API_KEY")
    if not key:
        raise ValueError("No The Odds API key. Add data/api_keys.json key 'the-odds-api'.")
    fixtures = M.upcoming(M.load_fixtures())
    rows = []
    for comp, sport in ODDS_API_SPORTS.items():
        wanted = fixtures[fixtures["competition"] == comp]
        if wanted.empty:
            continue
        query = urllib.parse.urlencode({
            "apiKey": key, "regions": "uk,eu,us",
            "markets": "h2h,totals,btts", "oddsFormat": "decimal",
        })
        url = f"{ODDS_API_URL.format(sport=sport)}?{query}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for event in payload:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            match = wanted[(wanted["home"] == home) & (wanted["away"] == away)]
            if match.empty:
                continue
            fixture = match.iloc[0]
            for book in event.get("bookmakers", []) or []:
                for market in book.get("markets", []) or []:
                    key_name = market.get("key", "")
                    # The Odds API publishes the bookmaker's own update time —
                    # use it (market-level, falling back to book-level) so a
                    # stale provider quote can't masquerade as fresh.
                    provider_ts = market.get("last_update") or book.get("last_update")
                    if provider_ts:
                        quoted_at, ts_source = str(provider_ts), "provider_last_update"
                    else:
                        quoted_at = datetime.now(timezone.utc).isoformat()
                        ts_source = "fetch_time_only"
                    for outcome in market.get("outcomes", []) or []:
                        name = str(outcome.get("name", ""))
                        odds = outcome.get("price")
                        if key_name == "h2h":
                            mapped = _map_api_bet("Match Winner", name, odds, home, away)
                        elif key_name == "totals":
                            mapped = _map_api_bet("Goals Over/Under",
                                                  f"{name} {outcome.get('point', '')}", odds)
                        elif key_name == "btts":
                            mapped = _map_api_bet("Both Teams Score", name, odds)
                        else:
                            mapped = None
                        if not mapped:
                            continue
                        market_name, side, line, decimal = mapped
                        rows.append({"date": fixture["date"].date(),
                                     "kickoff_utc": str(event.get("commence_time") or ""),
                                     "competition": comp, "home": home, "away": away,
                                     "market": market_name, "side": side,
                                     "line": line, "odds": decimal,
                                     "bookmaker": book.get("title", ""),
                                     "quoted_at_utc": quoted_at,
                                     "quote_time_source": ts_source})
    if not rows:
        raise ValueError("No The Odds API odds matched upcoming local fixtures; use BSD odds (--bsd-odds) or manual odds.csv.")
    return pd.DataFrame(rows)


def fetch_player_adjustments(api_key: str | None = None) -> dict:
    """Build player availability adjustments for all upcoming BSD matches.

    Fetches upcoming BSD events (which embed ``unavailable_players``),
    runs each through PlayerFeatureStore, and returns a dict:

        { (home_lower, away_lower, competition_name): player_adj_dict }

    where player_adj_dict = {"home": {"attack_mult": float, ...},
                              "away": {"attack_mult": float, ...}}

    Also returns market dispersion info if BSD provides multi-bookmaker odds.
    Silently returns an empty dict if BSD key is missing or the feature store
    has no data yet — the edge report will just lack player columns in that case.
    """
    try:
        from bsd_client import get_all_events, league_name as bsd_league_name
        from .competitions import comp_from_bsd_league
        from .player_features import PlayerFeatureStore, market_dispersion
        from .availability import match_availability, match_confidence
    except ImportError as exc:
        print(f"  player_adj: import failed ({exc}), skipping.")
        return {}

    key = api_key or get_key("bsd", env="BSD_API_KEY")
    if not key:
        return {}

    store = PlayerFeatureStore()
    store.load()

    # Also try to rebuild from already-cached event files (no extra API calls)
    if not store._player_records():
        n = store.refresh_from_cache()
        if n:
            print(f"  player_adj: built player stats from {n} cached events.")

    try:
        # BSD's status enum is notstarted|inprogress|finished|postponed|
        # cancelled — "upcoming" isn't a real value and silently matches 0
        # rows. BSD also defaults to a ~7-day forward window with no
        # date_from/date_to, which is exactly the near-term horizon we want
        # for pricing, so it's left unset here (unlike fetch.py's --current).
        events = get_all_events(key, status="notstarted")
    except Exception as exc:
        print(f"  player_adj: BSD fetch failed ({exc}), skipping.")
        return {}

    adj_map: dict = {}
    n_adj = 0
    for ev in events:
        comp = comp_from_bsd_league(bsd_league_name(ev))
        if comp is None:
            continue
        home = str(ev.get("home_team") or "")
        away = str(ev.get("away_team") or "")
        if not home or not away:
            continue

        adj = store.adjustments_for_match(ev)
        disp = market_dispersion(ev)
        # Embed market dispersion into adj for downstream use
        adj["market_dispersion"] = disp
        # Uncertainty band + lineup confidence (report-only; edge.py haircuts
        # the Kelly stake by lineup_confidence, never the point estimate).
        report = match_availability(store, ev)
        adj["availability_report"] = report
        adj["lineup_confidence"] = match_confidence(report)

        map_key = (home.lower(), away.lower(), comp.name)
        adj_map[map_key] = adj
        if adj["home"]["n_missing"] > 0 or adj["away"]["n_missing"] > 0:
            n_adj += 1

    if adj_map:
        print(f"  player_adj: {len(adj_map)} upcoming matches; "
              f"{n_adj} with listed absentees.")
    return adj_map


def late_lineup_card(api_key: str | None = None, window_minutes: int = 90) -> list[dict]:
    """Report-only "late card": for BSD events within `window_minutes` of
    kickoff where a confirmed starting XI is available, compare the base
    (pre-match) prediction to a lineup-adjusted one. Printed only — never
    written to edge_report.csv, never auto-bet (P2.5)."""
    from bsd_client import get_all_events, league_name as bsd_league_name, event_date_utc
    from .competitions import comp_from_bsd_league
    from .player_features import PlayerFeatureStore

    key = api_key or get_key("bsd", env="BSD_API_KEY")
    if not key:
        print("  lineups: no BSD key — skipped")
        return []
    store = PlayerFeatureStore().load()
    if not store._player_records():
        store.refresh_from_cache()

    now = datetime.now(timezone.utc)
    try:
        events = get_all_events(key, status="notstarted",
                                date_from=str(now.date()),
                                date_to=str(now.date() + pd.Timedelta(days=1)))
    except Exception as exc:
        print(f"  lineups: BSD fetch failed ({exc}) — skipped")
        return []

    params = M.load_params()
    rows: list[dict] = []
    for ev in events:
        comp = comp_from_bsd_league(bsd_league_name(ev))
        if comp is None:
            continue
        try:
            kickoff = pd.Timestamp(event_date_utc(ev))
            if kickoff.tzinfo is None:
                kickoff = kickoff.tz_localize("UTC")
        except Exception:
            continue
        mins_to_ko = (kickoff - pd.Timestamp(now)).total_seconds() / 60.0
        if not (0.0 <= mins_to_ko <= window_minutes):
            continue

        lu = store.adjustments_from_lineups(ev)
        if not lu or not lu.get("home") or not lu.get("away"):
            continue  # lineups not confirmed yet — nothing to report

        home = str(ev.get("home_team") or "")
        away = str(ev.get("away_team") or "")
        try:
            base = M.predict(home, away, comp.name, "ensemble", params=params)
        except ValueError:
            continue
        h_ratio = float(np.clip(lu["home"]["xi_ratio"], 0.80, 1.25))
        a_ratio = float(np.clip(lu["away"]["xi_ratio"], 0.80, 1.25))
        player_adj = {"home": {"attack_mult": h_ratio, "defense_mult": 1.0},
                      "away": {"attack_mult": a_ratio, "defense_mult": 1.0}}
        adjusted = M.predict(home, away, comp.name, "ensemble", params=params,
                             player_adj=player_adj)
        rows.append({
            "kickoff_utc": str(kickoff), "mins_to_kickoff": round(mins_to_ko, 1),
            "competition": comp.name, "home": home, "away": away,
            "base_p_home": base["probs"]["home"], "base_p_draw": base["probs"]["draw"],
            "base_p_away": base["probs"]["away"],
            "lineup_p_home": adjusted["probs"]["home"], "lineup_p_draw": adjusted["probs"]["draw"],
            "lineup_p_away": adjusted["probs"]["away"],
            "home_xi_ratio": lu["home"]["xi_ratio"], "away_xi_ratio": lu["away"]["xi_ratio"],
            "home_n_starters": lu["home"]["n_starters"], "away_n_starters": lu["away"]["n_starters"],
        })

    if rows:
        print(f"\nLate lineup card ({len(rows)} match(es) with confirmed XI, report-only):")
        for r in rows:
            print(f"  {r['home']} v {r['away']} ({r['competition']}, "
                  f"kickoff in {r['mins_to_kickoff']:.0f}min)")
            print(f"    base:    H {r['base_p_home']:.1%}  D {r['base_p_draw']:.1%}  "
                  f"A {r['base_p_away']:.1%}")
            print(f"    lineup:  H {r['lineup_p_home']:.1%}  D {r['lineup_p_draw']:.1%}  "
                  f"A {r['lineup_p_away']:.1%}  "
                  f"(xi_ratio home={r['home_xi_ratio']:.3f} away={r['away_xi_ratio']:.3f})")
    else:
        print("  lineups: no confirmed starting XIs within the window")
    return rows


def grade(side: str, market: str, line, home_goals: float, away_goals: float) -> str:
    hg, ag = int(home_goals), int(away_goals)
    if market == "1x2":
        actual = "home" if hg > ag else ("draw" if hg == ag else "away")
        return "won" if side == actual else "lost"
    if market == "total":
        total = hg + ag
        line = float(line)
        if total == line:
            return "push"
        return "won" if ((side == "over" and total > line) or
                         (side == "under" and total < line)) else "lost"
    if market == "btts":
        yes = hg > 0 and ag > 0
        return "won" if ((side == "yes" and yes) or (side == "no" and not yes)) else "lost"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--bsd-odds", action="store_true",
                    help="fetch live odds from BSD (free; recommended)")
    ap.add_argument("--the-odds-api", action="store_true",
                    help="fetch odds from The Odds API (paid)")
    ap.add_argument("--api-key",
                    help="API key override (BSD key for --bsd-odds, "
                         "Odds API key for --the-odds-api)")
    ap.add_argument("--model", choices=["ensemble", "goals", "elo", "xg"], default="ensemble")
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--calibrated", action="store_true",
                    help="apply fitted 1X2 calibration (needs validate.py --calibrate)")
    ap.add_argument("--player-adj", "--availability", dest="player_adj", action="store_true",
                    help="adjust predictions for player availability (injuries/suspensions) "
                         "from BSD unavailable_players data, and haircut the Kelly stake by "
                         "lineup-read confidence (a shaky/doubtful absence read bets smaller). "
                         "Also adds market-dispersion columns. Requires a BSD key.")
    ap.add_argument("--lineups", action="store_true",
                    help="print a report-only 'late card' comparing base vs "
                         "confirmed-lineup predictions for matches kicking off "
                         "within ~90 minutes. Never auto-bets. Requires a BSD key.")
    args = ap.parse_args()
    if args.lineups:
        late_lineup_card(args.api_key)
        return
    if args.template:
        write_template()
        print(f"Wrote {ODDS_CSV}")
        return
    calib_maps = None
    if args.calibrated:
        from .calibrate import load_maps
        calib_maps = load_maps()
        if calib_maps is None:
            sys.exit("--calibrated needs data/calibration.json. "
                     "Fit it first: python3 validate.py --calibrate")

    # Player availability adjustments (optional; needs BSD key)
    player_adj_map: dict | None = None
    if args.player_adj:
        print("Fetching player availability adjustments from BSD...")
        player_adj_map = fetch_player_adjustments(args.api_key)
        if not player_adj_map:
            print("  (no adjustments computed — check BSD key or player cache)")

    # do-not-bet suppression (P6.2) — on by default once enough snapshot
    # history exists to trust the movement signal.
    from . import market_model as MM
    history_days = MM.history_age_days()
    apply_dnb = history_days >= MM.WARMUP_DAYS
    if apply_dnb:
        print(f"  market-model: do-not-bet active ({history_days:.1f}d of snapshot history)")
    else:
        print(f"  market-model warming up: {history_days:.1f}d "
              f"(needs {MM.WARMUP_DAYS}d before do-not-bet activates)")

    try:
        if args.bsd_odds:
            odds = fetch_bsd_odds(args.api_key)
        elif args.the_odds_api:
            odds = fetch_the_odds_api(args.api_key)
        else:
            odds = load_odds()
        rows = rows_from_odds(odds, args.model, args.bankroll, calib_maps, player_adj_map,
                              apply_do_not_bet=apply_dnb)
    except Exception as e:
        sys.exit(str(e))
    DATA.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT, index=False)
    if rows:
        # Show player adjustment columns if present
        show_cols = ["date", "match", "market", "side", "odds", "p_model",
                     "p_book", "edge", "ev_per_unit", "kelly_stake", "stake_gbp"]
        if player_adj_map:
            show_cols += ["n_missing_home", "n_missing_away",
                          "player_adj_home", "player_adj_away"]
        df_out = pd.DataFrame(rows)
        visible = [c for c in show_cols if c in df_out.columns]
        print(df_out[visible].head(30).to_string(index=False))
    else:
        print("No priced edges found.")
    print(f"Saved -> {REPORT}")


if __name__ == "__main__":
    main()
