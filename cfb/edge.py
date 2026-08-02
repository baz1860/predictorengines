#!/usr/bin/env python3
"""Edge finder for CFB moneyline, spread, and totals markets.

Fill odds.csv with decimal odds from your bookmaker (both sides of a market
when possible — enables proper vig removal), then run. For each quote the
bookmaker's overround is removed, the implied probability is compared to the
blended model's, and edge, EV per unit, and a quarter-Kelly stake are reported.

Usage:
  python3 edge.py --template     # write odds.csv (upcoming fixtures if known)
  python3 edge.py                # edge report -> edge_report.csv, auto-log bets
  python3 edge.py --no-bet       # report only, don't touch the ledger
  python3 edge.py --bankroll 250 # override bankroll for stake sizing
"""
import argparse
import csv
import glob
import json
import math
import os
from datetime import date

import pandas as pd

from . import elo as E
from . import identity as IDENTITY
from . import policy as POLICY
from . import power as P
from .predictor import blend_predict

HERE = os.path.dirname(os.path.abspath(__file__))
ODDS_CSV = os.path.join(HERE, "odds.csv")
REPORT_CSV = os.path.join(HERE, "edge_report.csv")
LEDGER_CSV = os.path.join(HERE, "data", "ledger.csv")
BANKROLL_JSON = os.path.join(HERE, "data", "bankroll.json")
UPCOMING_CSV = os.path.join(HERE, "data", "upcoming.csv")

MIN_EDGE = 0.03
KELLY_FRACTION = 0.25
MAX_QUOTE_AGE_HOURS = 24.0

HEADER = ["date", "home", "away", "neutral", "market", "side", "line", "odds",
          "event_id", "cfbd_game_id", "commence_time", "bookmaker", "quote_time",
          "source", "identity_version"]


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def get_bankroll():
    if os.path.exists(BANKROLL_JSON):
        with open(BANKROLL_JSON) as f:
            return json.load(f)["bankroll"]
    return 100.0


def _id_text(value):
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def fixture_registry(seasons):
    """CFBD fixture identity keyed by game ID and exact legacy date/name tuple."""
    by_id, legacy = {}, {}
    for season in sorted({int(s) for s in seasons}):
        paths = glob.glob(os.path.join(HERE, "data", f"schedule_{season}.json"))
        for path in paths:
            try:
                games = json.load(open(path))
            except (OSError, json.JSONDecodeError):
                continue
            for g in games:
                try:
                    kickoff = pd.to_datetime(g.get("startDate"), utc=True)
                    day = str(kickoff.tz_localize(None).date())
                except Exception:
                    continue
                rec = {"cfbd_game_id": _id_text(g.get("id")),
                       "date": day, "home": g.get("homeTeam"),
                       "away": g.get("awayTeam"), "kickoff": kickoff}
                if rec["cfbd_game_id"]:
                    by_id[rec["cfbd_game_id"]] = rec
                legacy[(day, rec["home"], rec["away"])] = rec
    return {"by_id": by_id, "legacy": legacy}


def prepare_odds(odds, seasons, *, now=None, max_age_hours=MAX_QUOTE_AGE_HOURS):
    """Attach fixture, paired-market, provenance, and quote-freshness gates.

    Legacy eight-column files remain readable, but are diagnostic-only because
    they lack bookmaker timestamps and stable CFBD event identity.
    """
    out = odds.copy()
    for col in HEADER:
        if col not in out:
            out[col] = ""
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["odds"] = pd.to_numeric(out["odds"], errors="coerce")
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out = out[out["odds"].notna() & (out["odds"] > 1.0)].copy()
    out["event_id"] = out["event_id"].map(_id_text)
    out["cfbd_game_id"] = out["cfbd_game_id"].map(_id_text)
    for col in ("home", "away", "market", "side", "bookmaker", "source",
                "commence_time", "quote_time", "identity_version"):
        out[col] = out[col].fillna("").astype(str).str.strip()

    registry = fixture_registry(seasons)

    def match_fixture(r):
        rec = None
        if r.cfbd_game_id:
            rec = registry["by_id"].get(r.cfbd_game_id)
        if rec is None and not r.cfbd_game_id:
            rec = registry["legacy"].get((r.date, r.home, r.away))
        if rec is None or rec["home"] != r.home or rec["away"] != r.away:
            return False
        if str(r.commence_time or "").strip():
            try:
                quoted = pd.to_datetime(r.commence_time, utc=True)
                if abs((quoted - rec["kickoff"]).total_seconds()) > 12 * 3600:
                    return False
            except Exception:
                return False
        return True

    out["fixture_matched"] = [match_fixture(r) for r in out.itertuples()]
    out["_fixture"] = out.apply(
        lambda r: (f"cfbd:{r['cfbd_game_id']}" if r["cfbd_game_id"] else
                   f"legacy:{r['date']}|{r['home']}|{r['away']}"), axis=1)
    out["_line_key"] = out.apply(
        lambda r: "" if r["market"] == "ml" else (
            round(abs(r["line"]), 3)
            if r["market"] == "spread" and pd.notna(r["line"])
            else (round(r["line"], 3) if pd.notna(r["line"]) else "")), axis=1)
    out["_pair"] = out.apply(
        lambda r: (r["_fixture"], r["bookmaker"], r["market"], r["_line_key"]), axis=1)
    stats = out.groupby("_pair", dropna=False).agg(
        n=("side", "size"), sides=("side", lambda s: frozenset(s)))
    expected = {"ml": frozenset(("home", "away")),
                "spread": frozenset(("home", "away")),
                "total": frozenset(("over", "under"))}
    complete = {k: int(v.n) == 2 and v.sides == expected.get(k[2], frozenset())
                for k, v in stats.iterrows()}
    out["pair_complete"] = out["_pair"].map(complete).fillna(False).astype(bool)
    inv = out.groupby("_pair", dropna=False)["odds"].transform(lambda s: (1.0 / s).sum())
    out["p_implied"] = (1.0 / out["odds"]) / inv
    out.loc[~out["pair_complete"], "p_implied"] = float("nan")

    quote_ts = pd.to_datetime(out["quote_time"], errors="coerce", utc=True)
    now_ts = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    out["quote_age_hours"] = (now_ts - quote_ts).dt.total_seconds() / 3600.0
    out["quote_fresh"] = out["quote_age_hours"].between(0.0, float(max_age_hours))
    out["provenance_complete"] = (
        out["cfbd_game_id"].ne("") & out["bookmaker"].ne("") & quote_ts.notna()
        & out["identity_version"].ne(""))
    out["quote_eligible"] = (out["fixture_matched"] & out["pair_complete"]
                             & out["quote_fresh"] & out["provenance_complete"])
    return out


def write_template():
    rows = []
    if os.path.exists(UPCOMING_CSV):
        up = pd.read_csv(UPCOMING_CSV, parse_dates=["date"])
        up = up[(up["home_div"] == "fbs") & (up["away_div"] == "fbs")]
        up = up[up["date"] <= up["date"].min() + pd.Timedelta(days=7)]
        for r in up.itertuples():
            base = [str(r.date.date()), r.home_team, r.away_team, int(bool(r.neutral))]
            season = int(getattr(r, "season", E.season_for_date(r.date)))
            meta = ["", getattr(r, "game_id", ""), "", "", "", "manual",
                    IDENTITY.registry_version(season)]
            rows += [base + ["ml", "home", "", ""] + meta,
                     base + ["ml", "away", "", ""] + meta,
                     base + ["spread", "home", "", ""] + meta,
                     base + ["spread", "away", "", ""] + meta,
                     base + ["total", "over", "", ""] + meta,
                     base + ["total", "under", "", ""] + meta]
    if not rows:  # no upcoming schedule yet — show format
        base = [str(date.today()), "Ohio State", "Michigan", 0]
        meta = ["", "", "", "", "", "manual",
                IDENTITY.registry_version(E.season_for_date(date.today()))]
        rows = [base + ["ml", "home", "", 1.45] + meta,
                base + ["ml", "away", "", 2.90] + meta,
                base + ["spread", "home", -6.5, 1.91] + meta,
                base + ["spread", "away", 6.5, 1.91] + meta,
                base + ["total", "over", 48.5, 1.91] + meta,
                base + ["total", "under", 48.5, 1.91] + meta]
        print("note: no upcoming fixtures in data/upcoming.csv (run fetch_data.py in season) — "
              "wrote sample rows; edit teams/odds by hand")
    with open(ODDS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"wrote {ODDS_CSV} ({len(rows)} rows) — fill in lines and decimal odds, blanks are skipped")


def model_prob(pred, pparams, market, side, line):
    m, t = pred["margin"], pred["total"]
    s_m, s_t = pparams["sigma"], pparams["sigma_total"]
    if market == "ml":
        return pred["p1"] if side == "home" else 1.0 - pred["p1"]
    if market == "spread":
        if side == "home":   # home line L (e.g. -6.5): covers if margin + L > 0
            return 1.0 - phi((-line - m) / s_m)
        return phi((line - m) / s_m)  # away +L: covers if margin < L
    if market == "total":
        if side == "over":
            return 1.0 - phi((line - t) / s_t)
        return phi((line - t) / s_t)
    raise ValueError(market)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--no-bet", action="store_true")
    ap.add_argument("--bankroll", type=float, default=None)
    args = ap.parse_args()

    if args.template:
        write_template()
        return

    if not os.path.exists(ODDS_CSV):
        raise SystemExit("no odds.csv — run `python3 edge.py --template` first")
    odds = pd.read_csv(ODDS_CSV)
    odds = odds[odds["odds"].notna() & (odds["odds"] != "")]
    if odds.empty:
        raise SystemExit("odds.csv has no filled-in odds")
    quote_dates = pd.to_datetime(odds["date"], errors="coerce").dropna()
    if quote_dates.empty:
        raise SystemExit("odds.csv has no valid fixture dates")
    seasons = {E.season_for_date(d) for d in quote_dates}
    if len(seasons) != 1:
        raise SystemExit(f"odds.csv spans ambiguous seasons: {sorted(seasons)}")
    target_season = seasons.pop()
    odds = prepare_odds(odds, {target_season})
    unmatched = int((~odds["fixture_matched"]).sum())
    incomplete = int((odds["fixture_matched"] & ~odds["pair_complete"]).sum())
    odds = odds[odds["fixture_matched"] & odds["pair_complete"]].copy()
    if odds.empty:
        raise SystemExit("odds.csv has no exactly matched, complete two-sided markets")
    eparams, state_meta = E.build_as_of(target_season, as_of=date.today())
    pparams = P.load_params()
    market_policy = POLICY.load_policy()
    bankroll = args.bankroll if args.bankroll is not None else get_bankroll()

    report = []
    for r in odds.itertuples():
        pred = blend_predict(eparams, pparams, r.home, r.away, neutral=bool(r.neutral))
        line = None if pd.isna(r.line) else float(r.line)
        if r.market != "ml" and line is None:
            continue
        p_model = model_prob(pred, pparams, r.market, r.side, line)
        p_imp = float(r.p_implied)
        edge = p_model - p_imp
        ev = p_model * r.odds - 1.0
        kelly = max(0.0, (p_model * r.odds - 1.0) / (r.odds - 1.0))
        market_status = POLICY.status(r.market, market_policy)
        row_eligible = bool(E.event_betting_eligible(state_meta, r.home, r.away)
                            and r.quote_eligible
                            and market_status == "eligible")
        stake = (round(KELLY_FRACTION * kelly * bankroll, 2)
                 if row_eligible else 0.0)
        report.append({
            "date": r.date, "home": r.home, "away": r.away, "market": r.market,
            "side": r.side, "line": line, "odds": r.odds, "p_model": round(p_model, 4),
            "p_implied": round(p_imp, 4), "edge": round(edge, 4),
            "ev_per_unit": round(ev, 4), "stake": stake,
            "event_id": r.event_id, "cfbd_game_id": r.cfbd_game_id,
            "bookmaker": r.bookmaker, "quote_time": r.quote_time,
            "identity_version": r.identity_version,
            "quote_age_hours": (round(float(r.quote_age_hours), 2)
                                if pd.notna(r.quote_age_hours) else None),
            "model_season": state_meta["model_season"],
            "prior_mode": state_meta["prior_mode"],
            "market_status": market_status,
            "betting_eligible": row_eligible,
        })

    rep = pd.DataFrame(report).sort_values("edge", ascending=False)
    rep.to_csv(REPORT_CSV, index=False)
    with pd.option_context("display.width", 200):
        print(rep.to_string(index=False))
    print(f"\nbankroll £{bankroll:.2f} | quarter-Kelly | edges under ~3% are model noise")
    print(f"model season {state_meta['model_season']} | state {state_meta['prior_mode']} | "
          f"snapshot {state_meta['snapshot_hash']}")
    if unmatched or incomplete:
        print(f"quote gate: rejected {unmatched} unmatched and {incomplete} unpaired row(s)")
    if not state_meta["betting_eligible"]:
        print("STAKING DISABLED: incomplete target-season preseason priors; diagnostic edges only")
    elif not rep["betting_eligible"].any():
        print("STAKING DISABLED: no quote/market currently passes recommendation policy")
    print(f"report -> {REPORT_CSV}")

    if not args.no_bet:
        if not state_meta["betting_eligible"]:
            print("no bets logged (model state is not betting-eligible)")
            return
        bets = rep[(rep["edge"] >= MIN_EDGE) & rep["betting_eligible"]]
        bets = bets.loc[bets.groupby(["home", "away", "market"])["edge"].idxmax()]
        if bets.empty:
            print("no bets logged (no edge >= 3%)")
            return
        os.makedirs(os.path.dirname(LEDGER_CSV), exist_ok=True)
        new = not os.path.exists(LEDGER_CSV)
        with open(LEDGER_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["placed", "date", "home", "away", "market", "side", "line",
                            "odds", "stake", "p_model", "edge", "status", "pnl"])
            for b in bets.itertuples():
                w.writerow([str(date.today()), b.date, b.home, b.away, b.market, b.side,
                            b.line, b.odds, b.stake, b.p_model, b.edge, "open", ""])
        print(f"logged {len(bets)} bet(s) to ledger (use --no-bet to skip)")


if __name__ == "__main__":
    main()
