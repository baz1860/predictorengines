#!/usr/bin/env python3
"""Compare model probabilities against bookmaker odds to find value bets.

Where the edge comes from: a bookmaker's decimal odds imply probabilities
(1/odds), but they sum to more than 1 — the overround (vig). After removing
the vig, any outcome where YOUR probability exceeds the market's implied
probability has positive expected value:

    EV per unit staked = model_prob * odds - 1

Workflow:
  python edge.py --template      # write odds.csv with upcoming fixtures
  <fill in decimal odds from your bookmaker>
  python edge.py                 # report edges from odds.csv

  python edge.py --api-key KEY   # or pull live odds from the-odds-api.com
                                 # (free key: https://the-odds-api.com)

Outputs edge_report.csv sorted by expected value, with quarter-Kelly
suggested stakes (fraction of bankroll).

This is an analytical tool, not betting advice. Bookmaker closing lines are
sharp; treat small edges (<3%) as noise, and never bet more than you can
afford to lose.
"""
import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from api_keys import get_key
from contracts import fixture_key
from .predictor import load_matches, score_matrix
from .dixoncoles import build_sources
from .names import canonical_team

HERE = Path(__file__).resolve().parents[2]
ODDS_CSV = HERE / "odds.csv"
# Snapshot of the odds actually used on the last successful run (live API or
# filled manual CSV). Written every run so freshness checks can measure when the
# odds were really refreshed — odds.csv stays the manual-fallback input and is
# never overwritten (writing to it would make the next run prefer it over a live
# fetch). See app/provenance.py, which points the worldcup "odds" input here.
ODDS_LIVE_CSV = HERE / "data" / "odds_live.csv"
REPORT = HERE / "edge_report.csv"
KELLY_FRACTION = 0.25   # quarter Kelly: tempers overconfident models
MIN_EDGE = 0.0          # show everything; flag strong edges in the report
# ── Portfolio staking discipline (M7), all fractions of bankroll ─────────────
SINGLE_MATCH_CAP = 0.10     # max stake on any one match/outcome
DAILY_EXPOSURE_CAP = 0.25   # max total stake across all new same-day bets
CORR_CAP_MULT = 1.5         # correlated-group total <= 1.5x the single-match cap
DRAWDOWN_TRIGGER = 0.70     # halve Kelly while bankroll < 70% of its running peak
DRAWDOWN_BRAKE = 0.5
DEFAULT_API_KEY = get_key("the-odds-api", env="THE_ODDS_API_KEY")
# WC 2026 is played across US/Canada/Mexico.  The Odds API returns UTC times;
# convert to US Pacific (UTC-7 in summer) before extracting the date so that
# late kick-offs land on the correct local date.  PDT is the westernmost WC
# venue: a 9 pm PT start = 4 am UTC next day, and 4 am - 7 h = 9 pm PT same
# day.  No match starts after ~10 pm local time, so UTC-7 is always safe.
_TZ_PDT = timezone(timedelta(hours=-7))
BET_CONF_MIN = 0.40   # model must assign >= 40% probability to auto-record a bet
RECORD_MIN_EDGE = 0.0 # only auto-record per-market picks with edge strictly above this
                      # (model prob > bookmaker implied prob); raise for stronger value only
                      # (ensures we're backing genuine predictions, not noise)
MAX_ELO_GAP  = 350    # informational only — shown in report, not used as bet filter
WORLD_CUP_SPORT_KEY = "soccer_fifa_world_cup"

def canon(name, ratings, exit_on_error=True):
    name = canonical_team(str(name).strip())
    if name not in ratings:
        if not exit_on_error:
            raise ValueError(
                f"Unknown team {name!r} — add an alias in engines/worldcup/names.py")
        sys.exit(
            f"Unknown team {name!r} — add an alias in engines/worldcup/names.py")
    return name


# market -> [(side, odds-column)]; sides are graded by bankroll.py at settlement
MARKETS = {
    "1x2":  [("home", "odds_home"), ("draw", "odds_draw"), ("away", "odds_away")],
    "ou25": [("over25", "odds_over25"), ("under25", "odds_under25")],
    "btts": [("btts_yes", "odds_btts_yes"), ("btts_no", "odds_btts_no")],
}
ODDS_COLS = [col for sides in MARKETS.values() for _, col in sides]


def bet_label(side, home, away):
    return {"home": f"{home} win", "draw": "Draw", "away": f"{away} win",
            "over25": "Over 2.5 goals", "under25": "Under 2.5 goals",
            "btts_yes": "Both teams to score", "btts_no": "BTTS no"}[side]


def market_probs(home, away, sources, neutral_lookup, ctx=None, totals_lam_mult=1.0):
    """All market probabilities from the blended scoreline matrix.

    The scoreline matrix is the single-match (90-minute) goal distribution, so the
    1X2 probabilities here are the correct settlement basis for knockout matches
    too: a bookmaker's 1X2 / O-U / BTTS markets settle on the 90-minute result, so
    the draw is kept as a draw (extra-time / penalties are a separate progression
    question handled only in simulate.py, never in edge pricing).

    `totals_lam_mult` is the gated scoring-level calibration (see
    totals_calibration_check.py): the goal model runs ~9% low on total goals at
    World Cups, so the TOTALS and BTTS distribution is built from lambdas scaled by
    this factor. 1X2 stays on the unscaled matrix so the shipped, separately-fitted
    result prices are not disturbed. Default 1.0 is a no-op."""
    h1 = 0.0 if neutral_lookup.get((home, away), True) else 1.0
    Ms, Ms_tot = [], []
    for fn, rho in sources:
        l1, l2 = fn(home, away, h1, 0.0)
        if ctx is not None:
            l1, l2 = l1 * ctx[0], l2 * ctx[1]   # M6 rest/altitude correction
        Ms.append(score_matrix(l1, l2, rho))
        Ms_tot.append(Ms[-1] if totals_lam_mult == 1.0
                      else score_matrix(l1 * totals_lam_mult,
                                        l2 * totals_lam_mult, rho))
    M = np.mean(Ms, axis=0)
    Mt = M if totals_lam_mult == 1.0 else np.mean(Ms_tot, axis=0)
    n = M.shape[0]
    total = np.add.outer(np.arange(n), np.arange(n))
    p_under = Mt[total <= 2].sum()
    p_btts = Mt[1:, 1:].sum()
    return {"home": np.tril(M, -1).sum(), "draw": np.trace(M),
            "away": np.triu(M, 1).sum(),
            "over25": 1.0 - p_under, "under25": p_under,
            "btts_yes": p_btts, "btts_no": 1.0 - p_btts}


def devig(odds):
    """Proportional vig removal: implied = (1/odds) / overround."""
    inv = np.array([1.0 / o for o in odds])
    return inv / inv.sum(), inv.sum() - 1.0


def kelly(p, odds):
    """Kelly fraction for a single outcome at decimal odds.

    When the model has genuine edge (f > 0), returns standard Kelly.
    When there's no edge but the model still believes it, returns a small
    confidence-proportional floor (5% of p) so a stake is always suggested.
    The quarter-Kelly multiplier applied by the caller keeps both cases modest.
    """
    b = odds - 1.0
    f = (p * b - (1.0 - p)) / b
    return max(f, p * 0.05)


def _bet_teams(home, away, side):
    """Teams a bet is exposed to (for the correlation guard). Match bets touch
    both sides; outrights (side 'outright'/'back', or away marked —OUTRIGHT—)
    touch only the named team in `home`."""
    if str(side) in ("outright", "back") or "OUTRIGHT" in str(away).upper():
        return {str(home)}
    return {str(home), str(away)}


def portfolio_size(auto, bankroll, peak, ledger=None, verbose=True):
    """Apply M7 staking discipline to the day's candidate bets and return them
    with rescaled `kelly_stake` (so place_bets records the disciplined amount).

    Order: drawdown brake -> single-match cap -> correlation guard (incl. open
    exposure) -> simultaneous-Kelly daily cap. Reports pre/post stakes."""
    df = auto.copy().reset_index(drop=True)
    if df.empty:
        return df
    pre = (df["kelly_stake"].to_numpy() * bankroll).round(2)

    in_dd = bankroll < DRAWDOWN_TRIGGER * peak
    brake = DRAWDOWN_BRAKE if in_dd else 1.0
    stake = df["kelly_stake"].to_numpy() * brake * bankroll
    stake = np.minimum(stake, SINGLE_MATCH_CAP * bankroll)        # single-match cap

    # correlation guard: per-team total (new + existing open) <= corr cap
    corr_cap = CORR_CAP_MULT * SINGLE_MATCH_CAP * bankroll
    if ledger is None:
        from core.bankroll import _load_ledger
        ledger = _load_ledger()
    team_open = {}
    if ledger is not None and not ledger.empty:
        for r in ledger[ledger["status"] == "open"].itertuples(index=False):
            for t in _bet_teams(r.home, r.away, r.side):
                team_open[t] = team_open.get(t, 0.0) + float(r.stake)
    new_alloc = {}
    ev = df["ev_per_unit"].to_numpy() if "ev_per_unit" in df else np.zeros(len(df))
    for i in np.argsort(-ev):                                     # best bets first
        teams = _bet_teams(df.at[i, "home"], df.at[i, "away"], df.at[i, "side"])
        head = min(corr_cap - team_open.get(t, 0.0) - new_alloc.get(t, 0.0)
                   for t in teams)
        s = max(0.0, min(float(stake[i]), head))
        stake[i] = s
        for t in teams:
            new_alloc[t] = new_alloc.get(t, 0.0) + s

    total, cap = stake.sum(), DAILY_EXPOSURE_CAP * bankroll       # simultaneous Kelly
    scaled = total > cap and total > 0
    if scaled:
        # floor to pennies so the rounded total never creeps back over the cap
        stake_post = np.floor(stake * (cap / total) * 100) / 100.0
    else:
        stake_post = np.round(stake, 2)

    df["stake_pre"] = pre
    df["stake_post"] = stake_post
    df["kelly_stake"] = stake_post / bankroll   # so place_bets records stake_post
    if verbose:
        note = []
        if in_dd:
            note.append(f"drawdown brake ON (bankroll £{bankroll:.2f} < 70% of "
                        f"peak £{peak:.2f}): Kelly halved")
        if scaled:
            note.append(f"daily exposure scaled to cap £{cap:.2f} "
                        f"({DAILY_EXPOSURE_CAP:.0%} of bankroll)")
        capped = (df["stake_post"] < df["stake_pre"] - 0.005).sum()
        print(f"\nPortfolio staking: {len(df)} candidate(s), "
              f"£{pre.sum():.2f} pre -> £{df['stake_post'].sum():.2f} post"
              + (f"  [{'; '.join(note)}]" if note else ""))
        if capped:
            show = df[["match_date", "bet", "stake_pre", "stake_post"]].copy()
            print(show.to_string(index=False))
    return df


def write_template(upcoming):
    wc = upcoming[upcoming["tournament"] == "FIFA World Cup"]
    rows = [{"date": r.date.date(), "home": r.home_team, "away": r.away_team,
             **{col: "" for col in ODDS_COLS}}
            for r in wc.itertuples(index=False)]
    pd.DataFrame(rows).to_csv(ODDS_CSV, index=False)
    print(f"Wrote {len(rows)} fixtures to {ODDS_CSV.name}. "
          "Fill in decimal odds (e.g. 2.50), then run: python edge.py")


def _select_worldcup_sport_key(sports):
    """Choose the match-odds sport key, not an unrelated World Cup competition."""
    keys = [s.get("key", "") for s in sports]
    if WORLD_CUP_SPORT_KEY in keys:
        return WORLD_CUP_SPORT_KEY
    for s in sports:
        key = s.get("key", "")
        title = s.get("title", "").lower()
        if (key.startswith("soccer_") and "world cup" in title
                and "winner" not in key):
            return key
    return None


def fetch_api_odds(api_key, exit_on_error=True):
    """Pull h2h odds for the World Cup from the-odds-api.com (v4)."""
    base = "https://api.the-odds-api.com/v4"
    if not api_key:
        msg = ("No Odds API key available. Pass --api-key, set THE_ODDS_API_KEY, "
               "add data/api_keys.json, or fill in odds.csv manually.")
        if not exit_on_error:
            raise ValueError(msg)
        sys.exit(msg)
    try:
        with urllib.request.urlopen(f"{base}/sports/?apiKey={api_key}") as r:
            sports = json.load(r)
    except Exception as e:
        msg = (f"Could not reach The Odds API ({e}). If you're running in a "
               "restricted environment, fill in odds.csv manually instead.")
        if not exit_on_error:
            raise ValueError(msg)
        sys.exit(msg)
    sport_key = _select_worldcup_sport_key(sports)
    if not sport_key:
        msg = "No World Cup match odds found on The Odds API right now."
        if not exit_on_error:
            raise ValueError(msg)
        sys.exit(msg)
    # Request markets individually and merge — avoids a single bad market
    # (e.g. btts not on this plan) silently dropping all the others.
    events_by_id = {}
    for mkt in ("h2h", "totals", "btts"):
        mkt_url = (f"{base}/sports/{sport_key}/odds/?apiKey={api_key}"
                   f"&regions=eu&markets={mkt}&oddsFormat=decimal")
        try:
            with urllib.request.urlopen(mkt_url) as r:
                for ev in json.load(r):
                    eid = ev["id"]
                    if eid not in events_by_id:
                        events_by_id[eid] = ev
                    else:
                        # merge bookmakers from this market into existing event
                        existing_bks = {b["key"]: b
                                        for b in events_by_id[eid]["bookmakers"]}
                        for bk in ev["bookmakers"]:
                            if bk["key"] in existing_bks:
                                existing_bks[bk["key"]]["markets"].extend(
                                    bk["markets"])
                            else:
                                existing_bks[bk["key"]] = bk
                        events_by_id[eid]["bookmakers"] = list(
                            existing_bks.values())
        except Exception:
            pass   # market not supported on this plan — skip silently
    events = list(events_by_id.values())
    try:
        from scripts.worldcup.live_data import (
            MARKET_SNAPSHOTS_CSV,
            _append_snapshot_csv,
            normalize_market_snapshots,
            summarize_wide_market,
            utc_now,
        )
        fetched = utc_now()
        snapshots = normalize_market_snapshots(events, fetched)
        _append_snapshot_csv(
            MARKET_SNAPSHOTS_CSV,
            snapshots,
            ["snapshot_time", "event_id", "bookmaker", "market", "side", "line"],
        )
        wide = summarize_wide_market(snapshots)
        return wide.dropna(subset=["odds_home", "odds_draw", "odds_away"])
    except Exception as e:
        msg = f"Could not normalize The Odds API World Cup response ({e})."
        if not exit_on_error:
            raise ValueError(msg)
        sys.exit(msg)


def load_manual_odds():
    """Filled-in manual odds, requiring the 1X2 columns that identify a fixture."""
    if not ODDS_CSV.exists():
        return None
    return pd.read_csv(ODDS_CSV).dropna(
        subset=["odds_home", "odds_draw", "odds_away"])


def load_edge_modifiers(calibrated=False, market_blend=False, context_enabled=False):
    """Load optional probability/lambda modifiers once, for CLI and app parity."""
    mods = {"calib_maps": None, "mkt_blend_w": None,
            "ctx_mod": None, "ctx_coef": None, "ctx_played": None,
            "ctx_home_alt": None, "ctx_venue": None, "totals_lam_mult": 1.0}
    # Totals scoring-level calibration: a gated global lambda multiplier applied
    # to the totals/BTTS distribution only (1X2 untouched). Written by
    # totals_calibration_check.py --fit once it clears the leave-one-tournament-out
    # gate; absent -> 1.0 (no-op). This is a model calibration like calibrate.py,
    # so it loads whenever present rather than behind a flag.
    _cf = HERE / "data" / "totals_calibration.json"
    if _cf.exists():
        try:
            mods["totals_lam_mult"] = float(json.loads(_cf.read_text())
                                            .get("lambda_mult", 1.0))
        except (ValueError, OSError):
            mods["totals_lam_mult"] = 1.0
    if calibrated:
        from .calibrate import load_maps
        mods["calib_maps"] = load_maps()
        if mods["calib_maps"] is None:
            raise ValueError("--calibrated needs data/calibration.json. "
                             "Fit it first: python3 validate.py --calibrate")
    if market_blend:
        from .market_blend import load_w
        mods["mkt_blend_w"] = load_w()
        if mods["mkt_blend_w"] is None:
            raise ValueError("--market-blend needs data/market_blend.json. "
                             "Fit it first: python3 market_blend.py --fit")
    if context_enabled:
        from . import context as ctx_mod
        mods["ctx_mod"] = ctx_mod
        mods["ctx_coef"] = ctx_mod.load_coef()
        if not mods["ctx_coef"]:
            raise ValueError("--context needs data/context_coef.json. "
                             "Fit it first: python3 context.py --fit")
        mods["ctx_played"], _up = load_matches()
        mods["ctx_home_alt"] = ctx_mod._team_home_alt(mods["ctx_played"])
        mods["ctx_venue"] = {(u.home_team, u.away_team): (u.date, u.city)
                             for u in _up.itertuples(index=False)}
    return mods


def edge_rows(odds, sources, ratings, neutral_lookup, modifiers=None,
              strict_names=False):
    """Compute every priced outcome. Shared by the CLI and desktop adapter."""
    modifiers = modifiers or {}
    calib_maps = modifiers.get("calib_maps")
    mkt_blend_w = modifiers.get("mkt_blend_w")
    ctx_mod = modifiers.get("ctx_mod")
    ctx_coef = modifiers.get("ctx_coef")
    ctx_played = modifiers.get("ctx_played")
    ctx_home_alt = modifiers.get("ctx_home_alt")
    ctx_venue = modifiers.get("ctx_venue") or {}
    try:
        from wc_v4 import live_features as lf
        asof_now = pd.Timestamp.now(tz="UTC")
        live_avail = lf._availability_features(asof_now)
        live_lineups = lf._lineup_features(asof_now)
    except Exception:
        live_avail, live_lineups = {}, {}
    rows = []
    for r in odds.itertuples(index=False):
        home = canon(r.home, ratings, exit_on_error=not strict_names)
        away = canon(r.away, ratings, exit_on_error=not strict_names)
        match_date = getattr(r, "date", "")
        event_id = fixture_key(match_date, home, away, "FIFA World Cup")
        confs = [live_avail.get(t, {}).get("lineup_conf")
                 for t in (home, away)]
        confs = [float(c) for c in confs if c is not None and np.isfinite(c)]
        availability_confidence = min(confs) if confs else np.nan
        lineup_status = ("confirmed" if ((event_id, home) in live_lineups
                                         and (event_id, away) in live_lineups)
                         else "not_confirmed")
        ctx = None
        if ctx_mod is not None:
            dc = ctx_venue.get((home, away))
            fdate = dc[0] if dc else getattr(r, "date", None)
            fcity = dc[1] if dc else None
            if fdate is not None and not pd.isna(fdate):
                rd, gh, ga = ctx_mod.fixture_features(
                    home, away, fdate, fcity, played=ctx_played,
                    home_alt=ctx_home_alt)
                ctx = ctx_mod.multipliers(rd, gh, ga, ctx_coef)
        probs = market_probs(home, away, sources, neutral_lookup, ctx=ctx,
                             totals_lam_mult=modifiers.get("totals_lam_mult", 1.0))
        if calib_maps is not None:
            from .calibrate import apply as cal_apply
            ch, cd, ca = cal_apply(probs["home"], probs["draw"], probs["away"],
                                   calib_maps)
            probs["home"], probs["draw"], probs["away"] = ch, cd, ca
        if mkt_blend_w is not None:
            try:
                b1x2 = [float(r.odds_home), float(r.odds_draw), float(r.odds_away)]
            except (AttributeError, TypeError, ValueError):
                b1x2 = None
            if b1x2 and all(np.isfinite(o) and o > 1.0 for o in b1x2):
                from .market_blend import blend as mb_blend
                imp1x2, _ = devig(b1x2)
                pb = mb_blend(np.array([probs["home"], probs["draw"],
                                        probs["away"]]), imp1x2, mkt_blend_w)
                probs["home"], probs["draw"], probs["away"] = (
                    float(pb[0]), float(pb[1]), float(pb[2]))
            # Totals and BTTS get the SAME market discipline as 1X2. These markets
            # were previously pure model output, so any model bias (e.g. the
            # scoring-level gap) passed straight into the recommendation with no
            # market check. We blend each 2-way market toward its de-vigged book
            # line with the same fitted weight w. Edges are still measured against
            # the raw de-vigged book, so this only shrinks fake edges -- it does
            # not invent any. (No historical O/U closing odds exist to fit a
            # totals-specific w, so w inherits the 1X2 fit as a deliberate prior;
            # the line is at least as efficient on totals as on 1X2.)
            from .market_blend import blend as mb_blend
            for (a_side, a_col), (b_side, b_col) in (
                    (("over25", "odds_over25"), ("under25", "odds_under25")),
                    (("btts_yes", "odds_btts_yes"), ("btts_no", "odds_btts_no"))):
                try:
                    two = [float(getattr(r, a_col)), float(getattr(r, b_col))]
                except (AttributeError, TypeError, ValueError):
                    continue
                if all(np.isfinite(o) and o > 1.0 for o in two):
                    imp2, _ = devig(two)
                    pbm = mb_blend(np.array([probs[a_side], probs[b_side]]),
                                   imp2, mkt_blend_w)
                    probs[a_side], probs[b_side] = float(pbm[0]), float(pbm[1])
        for market, sides in MARKETS.items():
            try:
                book = [float(getattr(r, col)) for _, col in sides]
            except (AttributeError, TypeError, ValueError):
                continue
            if any(not np.isfinite(o) or o <= 1.0 for o in book):
                continue
            implied, overround = devig(book)
            for (side, _), p_book, o in zip(sides, implied, book):
                p_model = probs[side]
                ev = p_model * o - 1.0
                rows.append({
                    "date": match_date, "match": f"{home} v {away}",
                    "home": home, "away": away, "side": side, "market": market,
                    "bet": bet_label(side, home, away), "odds": o,
                    "p_book": round(float(p_book), 3),
                    "p_model": round(float(p_model), 3),
                    "edge": round(float(p_model - p_book), 3),
                    "ev_per_unit": round(float(ev), 3),
                    "kelly_stake": round(KELLY_FRACTION * kelly(p_model, o), 4),
                    "overround": round(float(overround), 3),
                    "elo_gap": round(abs(ratings.get(home, 1500)
                                        - ratings.get(away, 1500))),
                    "bookmaker_count": getattr(r, "bookmaker_count", np.nan),
                    "market_dispersion_h": getattr(r, "market_dispersion_h", np.nan),
                    "market_dispersion_d": getattr(r, "market_dispersion_d", np.nan),
                    "market_dispersion_a": getattr(r, "market_dispersion_a", np.nan),
                    "lineup_status": lineup_status,
                    "availability_confidence": availability_confidence,
                })
    return rows


def top_confident_picks(df, ledger=None):
    """Model's top pick per match+market, annotated for ledger recording."""
    if df.empty:
        return df.copy()
    top_pick_idx = df.groupby(["match", "market"])["p_model"].idxmax()
    confident = (df.loc[top_pick_idx]
                   .sort_values("p_model", ascending=False)
                   .copy())
    confident = confident[confident["p_model"] >= BET_CONF_MIN].copy()
    open_keys = set()
    if ledger is not None and not ledger.empty:
        open_rows = ledger[ledger["status"] == "open"]
        open_keys = set(zip(open_rows["home"], open_rows["away"], open_rows["side"]))
    horizon = pd.Timestamp.now() + pd.Timedelta(hours=36)

    def _ledger_note(r):
        if (r["home"], r["away"], r["side"]) in open_keys:
            return "already in ledger"
        if pd.to_datetime(r["date"]) > horizon:
            return "not imminent (>36h)"
        if r["edge"] <= RECORD_MIN_EDGE:
            return "no edge - not ledgered"
        return "AUTO-LEDGER"

    confident["ledger"] = confident.apply(_ledger_note, axis=1)
    return confident


def auto_bet_candidates(confident, bankroll, portfolio=True, peak=None,
                        ledger=None, verbose=True):
    """Imminent positive-edge picks, optionally passed through portfolio sizing."""
    if confident.empty:
        return confident.rename(columns={"date": "match_date"}).copy()
    auto = confident.rename(columns={"date": "match_date"})
    horizon = pd.Timestamp.now() + pd.Timedelta(hours=36)
    auto = auto[(pd.to_datetime(auto["match_date"]) <= horizon)
                & (auto["edge"] > RECORD_MIN_EDGE)].copy()
    if portfolio and not auto.empty:
        auto = portfolio_size(auto, bankroll, peak or bankroll, ledger=ledger,
                              verbose=verbose)
    return auto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true",
                    help="write odds.csv fixture template and exit")
    ap.add_argument("--api-key", help="The Odds API key (the-odds-api.com)")
    ap.add_argument("--bankroll", type=float, default=None,
                    help="override bankroll (default: live value from "
                         "data/bankroll.json, tracked by bankroll.py)")
    ap.add_argument("--no-bet", action="store_true",
                    help="don't record recommended bets in the ledger")
    ap.add_argument("--no-portfolio", action="store_true",
                    help="disable M7 portfolio staking discipline (caps, "
                         "correlation guard, drawdown brake) — size bets the old way")
    ap.add_argument("--model", choices=["elo", "dc", "blend"], default="blend",
                    help="lambda source (default: blend, best in backtest)")
    ap.add_argument("--squad-adj", action="store_true",
                    help="apply availability adjustments from squads.py "
                         "(data/squad_ratings.csv; see also injuries.py)")
    ap.add_argument("--conf-adj", action="store_true",
                    help="apply confederation strength adjustment to Elo ratings "
                         "(calibrate fraction first: python confederation_adj.py --backtest)")
    ap.add_argument("--calibrated", action="store_true",
                    help="apply isotonic probability calibration to the model's 1X2 "
                         "(maps from data/calibration.json; fit with: "
                         "python3 validate.py --calibrate). Applied before any "
                         "--market-blend.")
    ap.add_argument("--market-blend", action="store_true",
                    help="anchor the model's 1X2, Over/Under 2.5 and BTTS probs "
                         "toward the de-vigged market in logit space (weight from "
                         "data/market_blend.json; fit with: python3 market_blend.py "
                         "--fit). Edges are still computed against the raw market.")
    ap.add_argument("--context", action="store_true",
                    help="apply rest/altitude lambda correction to each fixture "
                         "(coefficients from data/context_coef.json; fit with: "
                         "python3 context.py --fit)")
    args = ap.parse_args()

    _, upcoming = load_matches()

    # ── Build rating sources, applying optional adjustments ──────────────────
    conf_active = getattr(args, "conf_adj", False)
    squad_active = args.squad_adj

    if conf_active:
        # Start from confederation-adjusted sources
        from .confederation_adj import build_conf_adj_sources
        sources, ratings, conf_means, adjs = build_conf_adj_sources(args.model)
        from .confederation_adj import load_optimal_fraction
        frac = load_optimal_fraction()
        print(f"Confederation adjustment (fraction={frac:.2f}): " +
              ", ".join(f"{c} mean adj {v:+.0f} Elo"
                        for c, v in sorted(conf_means.items(),
                                           key=lambda kv: kv[1])))
    else:
        sources, ratings = build_sources(args.model)

    if squad_active:
        from .squads import adjusted_sources
        if conf_active:
            # Squad adj on top of conf-adj ratings: rebuild Elo source with
            # conf-adjusted ratings already applied, then layer squad deltas
            from .predictor import load_matches as _lm, compute_elo, fit_goal_model
            from .predictor import expected_goals, HOME_ADV, DC_RHO
            from .squads import load_adj as _load_adj
            squad_adj_map = _load_adj()          # team -> Elo delta from absences
            active_sq = {t: d for t, d in squad_adj_map.items() if d}
            for team, delta in active_sq.items():
                if team in ratings:
                    ratings[team] += delta
            # Rebuild Elo source with fully adjusted ratings
            _played, _ = _lm()
            _, _played = compute_elo(_played)
            _beta = fit_goal_model(_played)
            _r = dict(ratings)
            from .confederation_adj import load_params as _lp, apply_match_adj as _ama
            _ca = adjs          # confederation adjustments dict
            _thr = _lp()[1]     # threshold from conf_adj.json
            def _elo_fn(t1, t2, h1=0.0, h2=0.0,
                        __r=_r, __b=_beta, __ha=HOME_ADV,
                        __ca=_ca, __thr=_thr):
                e1, e2 = _ama(__r.get(t1, 1500.0), __r.get(t2, 1500.0),
                              __ca.get(t1, 0.0), __ca.get(t2, 0.0), __thr)
                return expected_goals(e1, e2, __b, (h1 - h2) * __ha)
            sources = [(_elo_fn, DC_RHO)] + [s for s in sources
                                              if not hasattr(s[0], '__name__')
                                              or s[0].__name__ != '_elo_fn']
            if active_sq:
                print("Squad availability adjustments (on top of conf-adj): " +
                      ", ".join(f"{t} {d:+.0f} Elo"
                                for t, d in active_sq.items()))
        else:
            sources, ratings, _adj = adjusted_sources(args.model)
            active_sq = {t: a for t, a in _adj.items() if a}
            if active_sq:
                print("Squad availability adjustments: " +
                      ", ".join(f"{t} {a:+.0f} Elo"
                                for t, a in active_sq.items()))
            else:
                print("Squad adjustment on, but no absences listed "
                      "(data/absences.csv / injuries.py).")
    neutral_lookup = {(r.home_team, r.away_team): bool(r.neutral)
                      for r in upcoming.itertuples(index=False)}

    if args.template:
        write_template(upcoming)
        return

    # priority: explicit --api-key > filled-in odds.csv > THE_ODDS_API_KEY
    csv_odds = load_manual_odds()
    api_key = (args.api_key or DEFAULT_API_KEY).strip()
    if args.api_key or (api_key and (csv_odds is None or csv_odds.empty)):
        odds = fetch_api_odds(api_key)
        print(f"Fetched median odds for {len(odds)} matches from The Odds API.")
    elif csv_odds is not None and not csv_odds.empty:
        odds = csv_odds
    else:
        sys.exit("No odds found. Run with --template to create odds.csv, "
                 "or pass --api-key.")

    # Snapshot the odds we actually used so freshness checks reflect the real
    # refresh time, whatever the source. Never blocks the report if it fails.
    try:
        ODDS_LIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
        odds.to_csv(ODDS_LIVE_CSV, index=False)
    except Exception as e:
        print(f"   (could not write {ODDS_LIVE_CSV.name}: {e})")

    try:
        modifiers = load_edge_modifiers(args.calibrated, args.market_blend,
                                        args.context)
    except ValueError as e:
        sys.exit(str(e))

    if args.calibrated:
        print("Calibration active (isotonic per-outcome on model 1X2).")
    if args.market_blend:
        mkt_blend_w = modifiers["mkt_blend_w"]
        print(f"Market blend active (w={mkt_blend_w:.3f} on model 1X2; edges still "
              "computed vs the raw de-vigged market).")
    if args.context:
        print(f"Context correction active (rest/altitude): {modifiers['ctx_coef']}")

    rows = edge_rows(odds, sources, ratings, neutral_lookup, modifiers)

    from core.bankroll import current_bankroll, current_peak, place_bets
    bankroll = args.bankroll if args.bankroll is not None else current_bankroll()
    edge_cols = ["date", "match", "home", "away", "side", "market", "bet",
                 "odds", "p_book", "p_model", "edge", "ev_per_unit",
                 "kelly_stake", "overround", "elo_gap"]
    df = pd.DataFrame(rows, columns=edge_cols if not rows else None)
    df["stake_gbp"] = (df["kelly_stake"] * bankroll).round(2)
    df.to_csv(REPORT, index=False)

    pd.set_option("display.width", 160)
    show_cols = [c for c in df.columns if c not in ("home", "away", "side",
                                                     "kelly_stake",
                                                     "ev_per_unit")]
    from core.bankroll import _load_ledger
    _led = _load_ledger()
    confident = top_confident_picks(df, ledger=_led)
    print(f"\nBankroll £{bankroll:.2f}  —  model's top prediction per market "
          f"per match (confidence ≥ {BET_CONF_MIN:.0%}, sorted by confidence):\n")
    if not confident.empty:
        print(confident[show_cols + ["ledger"]].to_string(index=False))
        print(f"\n  'ledger' column: AUTO-LEDGER = will be recorded; "
              f"'no edge' = model doesn't beat the price, shown but not bet; "
              f"others not bet for the reason given.")
    else:
        print(f"  No matches where the model's top pick reaches "
              f"{BET_CONF_MIN:.0%} confidence.")

    # ── Morning bet queue (M8): bettable candidates, sized, for review ────────
    # model's top pick per market with confidence >= BET_CONF_MIN, positive edge,
    # imminent (<=36h). Written whether or not bets are recorded, so the daily
    # summary can surface it before real money is placed.
    auto = auto_bet_candidates(confident, bankroll,
                               portfolio=not args.no_portfolio,
                               peak=current_peak(), ledger=_led)

    flags = []
    if args.calibrated:
        flags.append("calibrated")
    if args.market_blend:
        flags.append(f"market-blend(w={modifiers['mkt_blend_w']:.2f},1X2+OU+BTTS)")
    if modifiers.get("totals_lam_mult", 1.0) != 1.0:
        flags.append(f"totals-calib(lam x{modifiers['totals_lam_mult']:.2f})")
    if args.context:
        flags.append("context")
    if args.squad_adj:
        flags.append("squad-adj")
    if getattr(args, "conf_adj", False):
        flags.append("conf-adj")
    flags_str = "+".join(flags) if flags else "raw-model"
    sq_map = active_sq if "active_sq" in locals() else {}
    QUEUE = HERE / "bet_queue.csv"
    qcols = ["match_date", "match", "bet", "odds", "p_model", "p_book", "edge",
             "stake", "adjustments", "squad_adj"]
    if not auto.empty:
        q = auto.copy()
        q["stake"] = (q["stake_post"] if "stake_post" in q.columns
                      else (q["kelly_stake"] * bankroll).round(2))
        q["adjustments"] = flags_str
        q["squad_adj"] = [", ".join(f"{t} {sq_map[t]:+.0f}" for t in (h, a)
                                    if t in sq_map) or "-"
                          for h, a in zip(q["home"], q["away"])]
        q[qcols].to_csv(QUEUE, index=False)
    else:
        pd.DataFrame(columns=qcols).to_csv(QUEUE, index=False)
    print(f"\nMorning bet queue -> {QUEUE.name} "
          f"({0 if auto.empty else len(auto)} candidate(s); adjustments: {flags_str})")

    if not args.no_bet:
        placed = place_bets(auto)
        if len(placed):
            print(f"\nRecorded {len(placed)} bet(s) in the ledger "
                  f"(confidence ≥ {BET_CONF_MIN:.0%}, edge > {RECORD_MIN_EDGE:.0%}):")
            print(placed[["match_date", "bet", "odds", "stake"]]
                  .to_string(index=False))
            print("Settle after matches with: python3 bankroll.py --settle")
        else:
            print(f"\nNo new bets recorded (already in ledger, model "
                  f"confidence < {BET_CONF_MIN:.0%}, or no positive edge).")

    print(f"\nFull report (all outcomes) -> {REPORT.name}")


if __name__ == "__main__":
    main()
