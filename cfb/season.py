#!/usr/bin/env python3
"""cfb/season.py — the one front door for the CFB engine.

Same mental model as the tennis and golf engines: pull the upcoming week's FBS
slate, let the fitted Elo+power blend price every matchup, and write one card —
the model's moneyline pick, its spread and total, the **ATS pick against the
market line**, and any value bets with quarter-Kelly stakes. Everything else in
this package (`elo`, `power`, `predictor`, `edge`, …) is plumbing this drives;
you should not need to call those directly for a normal week.

    python3 -m cfb.season                    # price this week's card (uses cfb/odds.csv if filled)
    python3 -m cfb.season --odds-api         # pull NCAAF lines from The Odds API first
    python3 -m cfb.season --refresh          # refresh games/upcoming + refit power first
    python3 -m cfb.season --days 7           # how far ahead the "week" reaches
    python3 -m cfb.season --min-edge 0.05    # stricter value threshold

Lines come from The Odds API (`--odds-api`, key 'the-odds-api' in
data/api_keys.json or THE_ODDS_API_KEY) or manually via `python3 -m cfb.edge
--template` + editing cfb/odds.csv. Without any lines the card still gives the
model's straight-up pick, spread, and total for every game — there is just no
ATS pick (that needs a market line) and no value section.

The card is written to cfb/data/card.md and printed to stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date

import pandas as pd

from . import elo as E
from . import power as P
from .edge import (DEFAULT_OVERROUND, HEADER, KELLY_FRACTION, MIN_EDGE,
                   ODDS_CSV, get_bankroll, model_prob)
from .predictor import blend_predict

HERE = os.path.dirname(os.path.abspath(__file__))
UPCOMING_CSV = os.path.join(HERE, "data", "upcoming.csv")
CARD_MD = os.path.join(HERE, "data", "card.md")

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_ncaaf"


def _odds_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key.strip()
    try:
        import sys
        sys.path.insert(0, os.path.dirname(HERE))
        from api_keys import get_key
        return (get_key("the-odds-api", env="THE_ODDS_API_KEY") or "").strip()
    except Exception:
        return (os.environ.get("THE_ODDS_API_KEY") or "").strip()


# ── team-name matching (Odds API appends mascots: "Ohio State Buckeyes") ─────

def _fold(name: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", "and")
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def _match_team(provider_name: str, teams: list[str]) -> str | None:
    """Longest model-team name that prefixes the provider name (mascot-tolerant)."""
    p = _fold(provider_name)
    best = None
    for t in teams:
        f = _fold(t)
        if p == f or p.startswith(f + " "):
            if best is None or len(f) > len(_fold(best)):
                best = t
    return best


# ── slate ─────────────────────────────────────────────────────────────────────

def _slate_from_schedule_json(days: int) -> pd.DataFrame | None:
    """Offseason fallback: build the next week's slate from data/schedule_<yr>.json
    (CFBD /games) when upcoming.csv is empty (fetch_data mirrors lag preseason)."""
    import glob
    files = sorted(glob.glob(os.path.join(HERE, "data", "schedule_*.json")))
    if not files:
        return None
    sched = json.load(open(files[-1]))
    rows = [{"date": pd.Timestamp(g["startDate"]).tz_localize(None).normalize(),
             "neutral": bool(g.get("neutralSite")),
             "home_team": g["homeTeam"], "home_div": g.get("homeClassification"),
             "away_team": g["awayTeam"], "away_div": g.get("awayClassification")}
            for g in sched if not g.get("completed")]
    up = pd.DataFrame(rows)
    up = up[up["date"] >= pd.Timestamp(date.today())]
    return None if up.empty else up


def load_slate(days: int = 7) -> pd.DataFrame:
    up = pd.DataFrame()
    if os.path.exists(UPCOMING_CSV):
        up = pd.read_csv(UPCOMING_CSV, parse_dates=["date"])
    if up.empty:
        up = _slate_from_schedule_json(days)
        if up is None:
            raise SystemExit("no upcoming fixtures — run `python3 -m cfb.fetch_data` "
                             "(in season) or drop a CFBD schedule_<year>.json in cfb/data/")
        print(f"note: upcoming.csv empty — slate taken from schedule JSON")
    up = up[(up["home_div"] == "fbs") & (up["away_div"] == "fbs")].copy()
    if up.empty:
        raise SystemExit("no upcoming FBS-vs-FBS fixtures")
    start = up["date"].min()
    up = up[up["date"] <= start + pd.Timedelta(days=days)]
    return up.sort_values("date").reset_index(drop=True)


# ── The Odds API → odds.csv (edge.py format, both sides of each market) ──────

def fetch_odds_api(slate: pd.DataFrame, api_key: str, regions: str = "us") -> int:
    query = urllib.parse.urlencode({
        "apiKey": api_key, "regions": regions,
        "markets": "h2h,spreads,totals", "oddsFormat": "decimal",
    })
    with urllib.request.urlopen(
            f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds/?{query}", timeout=30) as r:
        events = json.load(r)

    teams = sorted(set(slate["home_team"]) | set(slate["away_team"]))
    slate_idx = {(r.home_team, r.away_team): r for r in slate.itertuples()}

    rows, unmatched = [], []
    for ev in events:
        home = _match_team(ev.get("home_team", ""), teams)
        away = _match_team(ev.get("away_team", ""), teams)
        fix = slate_idx.get((home, away)) if home and away else None
        if fix is None:
            unmatched.append(f"{ev.get('away_team')} at {ev.get('home_team')}")
            continue
        base = [str(fix.date.date()), home, away, int(bool(fix.neutral))]

        # collect (point, price) per market/outcome across books
        acc: dict[tuple, list] = defaultdict(list)
        for book in ev.get("bookmakers") or []:
            for mk in book.get("markets") or []:
                for out in mk.get("outcomes") or []:
                    try:
                        price = float(out.get("price"))
                    except (TypeError, ValueError):
                        continue
                    if price <= 1.0:
                        continue
                    point = out.get("point")
                    acc[(mk.get("key"), _fold(out.get("name", "")))].append((point, price))

        def side_row(market, side, key_name, needs_line):
            entries = acc.get((market, key_name), [])
            if not entries:
                return None
            if needs_line:
                # consensus line = modal point across books; median price at it
                pts = Counter(p for p, _ in entries if p is not None)
                if not pts:
                    return None
                line = pts.most_common(1)[0][0]
                prices = [pr for p, pr in entries if p == line]
            else:
                line, prices = "", [pr for _, pr in entries]
            return base + [market_name(market), side, line,
                           round(statistics.median(prices), 3)]

        def market_name(k):
            return {"h2h": "ml", "spreads": "spread", "totals": "total"}[k]

        for market, sides in (("h2h", [("home", _fold(ev["home_team"])),
                                       ("away", _fold(ev["away_team"]))]),
                              ("spreads", [("home", _fold(ev["home_team"])),
                                           ("away", _fold(ev["away_team"]))]),
                              ("totals", [("over", "over"), ("under", "under")])):
            for side, key_name in sides:
                row = side_row(market, side, key_name, market != "h2h")
                if row:
                    rows.append(row)

    with open(ODDS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    n_games = len({(r[1], r[2]) for r in rows})
    print(f"odds-api: {n_games} slate game(s) priced -> {ODDS_CSV}")
    if unmatched:
        print(f"  (skipped {len(unmatched)} event(s) not on this week's slate or "
              f"unmatched by name, e.g. {unmatched[0]})")
    return n_games


# ── card ──────────────────────────────────────────────────────────────────────

def load_market(slate: pd.DataFrame) -> dict:
    """odds.csv → {(home, away): {market: {side: (line, odds)}}} with vig removed
    per two-way pair (falls back to a 4.5% assumed overround one-sided)."""
    market: dict = {}
    if not os.path.exists(ODDS_CSV):
        return market
    odds = pd.read_csv(ODDS_CSV)
    odds = odds[odds["odds"].notna()]
    if odds.empty:
        return market
    odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")
    odds = odds[odds["odds"] > 1.0]
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    for r in odds.itertuples():
        game = market.setdefault((r.home, r.away), {})
        game.setdefault(r.market, {})[r.side] = (
            None if pd.isna(r.line) else float(r.line), float(r.odds))
    return market


def build_card(days: int = 7, min_edge: float = MIN_EDGE,
               bankroll: float | None = None, model: str = "blend") -> dict:
    slate = load_slate(days)
    market = load_market(slate)
    eparams = E.build()
    pparams = P.load_params()
    known = set(pparams["teams"]) & set(eparams[1])
    off_model = ~(slate["home_team"].isin(known) & slate["away_team"].isin(known))
    if off_model.any():
        print(f"note: skipped {int(off_model.sum())} game(s) with teams unknown to "
              f"the model (new FBS members / reclassified)")
        slate = slate[~off_model]
    bk = bankroll if bankroll is not None else get_bankroll()

    lines_md, value = [], []
    ats_picks = 0
    for g in slate.itertuples():
        pred = blend_predict(eparams, pparams, g.home_team, g.away_team,
                             neutral=bool(g.neutral), model=model)
        fav = g.home_team if pred["margin"] >= 0 else g.away_team
        vs = "vs" if g.neutral else "at"
        lines_md.append(f"### {g.date.date()} — {g.away_team} {vs} {g.home_team}")
        lines_md.append(
            f"- Model: **{fav}** -{abs(pred['margin']):.1f} "
            f"(home win {pred['p1']*100:.1f}%) · total {pred['total']:.1f}")

        mkt = market.get((g.home_team, g.away_team), {})
        sp = mkt.get("spread", {})
        home_line = sp.get("home", (None, None))[0]
        if home_line is None and "away" in sp and sp["away"][0] is not None:
            home_line = -sp["away"][0]
        if home_line is not None:
            p_home_cover = model_prob(pred, pparams, "spread", "home", home_line)
            if p_home_cover >= 0.5:
                pick, p_cover, line_s = g.home_team, p_home_cover, home_line
            else:
                pick, p_cover, line_s = g.away_team, 1.0 - p_home_cover, -home_line
            ats_picks += 1
            lines_md.append(
                f"- **ATS pick: {pick} {line_s:+g}** — cover {p_cover*100:.1f}% "
                f"(market: {g.home_team} {home_line:+g})")
        else:
            lines_md.append("- ATS pick: no market spread loaded")

        tot = mkt.get("total", {})
        t_line = next((v[0] for v in tot.values() if v[0] is not None), None)
        if t_line is not None:
            p_over = model_prob(pred, pparams, "total", "over", t_line)
            lean = "Over" if p_over >= 0.5 else "Under"
            lines_md.append(f"- Total lean: {lean} {t_line:g} — "
                            f"{max(p_over, 1-p_over)*100:.1f}%")

        # value: every quoted side, de-vigged in pairs
        for mkey, sides in mkt.items():
            inv = sum(1.0 / o for _, o in sides.values())
            over_r = inv if len(sides) == 2 else DEFAULT_OVERROUND
            for side, (line, o) in sides.items():
                if mkey != "ml" and line is None:
                    continue
                p_model = model_prob(pred, pparams, mkey, side, line)
                edge = p_model - (1.0 / o) / over_r
                if edge >= min_edge:
                    kelly = max(0.0, (p_model * o - 1.0) / (o - 1.0))
                    if mkey == "ml":
                        bet = f"{g.home_team if side == 'home' else g.away_team} ML"
                    elif mkey == "spread":
                        bet = f"{g.home_team if side == 'home' else g.away_team} {line:+g}"
                    else:
                        bet = f"{side.capitalize()} {line:g}"
                    value.append({
                        "date": str(g.date.date()), "game": f"{g.away_team} {vs} {g.home_team}",
                        "bet": bet,
                        "odds": o, "p_model": p_model, "edge": edge,
                        "stake": round(KELLY_FRACTION * kelly * bk, 2)})
        lines_md.append("")

    value.sort(key=lambda v: -v["edge"])
    md = [f"# CFB weekly card — {date.today()}",
          f"_{len(slate)} FBS games · model: {model} · bankroll £{bk:.2f} · "
          f"quarter-Kelly · min edge {min_edge:.0%}_", ""]
    md.append("## Value bets")
    if value:
        md.append("| Date | Game | Bet | Odds | Model % | Edge | Stake |")
        md.append("|---|---|---|---|---|---|---|")
        for v in value:
            md.append(f"| {v['date']} | {v['game']} | **{v['bet']}** | {v['odds']:.2f} "
                      f"| {v['p_model']*100:.1f}% | {v['edge']*100:+.1f}% | £{v['stake']:.2f} |")
    else:
        md.append("_None at this threshold" +
                  (" (no odds loaded — run with --odds-api or fill cfb/odds.csv)_"
                   if not market else "._"))
    md += ["", "## Matchups (straight-up + ATS)", ""] + lines_md

    text = "\n".join(md)
    os.makedirs(os.path.dirname(CARD_MD), exist_ok=True)
    with open(CARD_MD, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\ncard -> {CARD_MD}")
    return {"games": len(slate), "ats_picks": ats_picks, "value_bets": len(value),
            "card": CARD_MD}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true",
                    help="refresh games/upcoming.csv and refit power first")
    ap.add_argument("--odds-api", action="store_true",
                    help="pull NCAAF ml/spread/total lines from The Odds API")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--regions", default="us")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--model", default="blend", choices=["blend", "elo", "power"])
    ap.add_argument("--min-edge", type=float, default=MIN_EDGE)
    ap.add_argument("--bankroll", type=float, default=None)
    args = ap.parse_args()

    if args.refresh:
        from . import fetch_data
        fetch_data.main()
        params = P.fit(E.load_games())
        with open(P.PARAMS_JSON, "w") as f:
            json.dump(params, f, indent=1)
        print(f"power refit: {len(params['teams'])} teams as of {params['asof']}")

    if args.odds_api:
        key = _odds_api_key(args.api_key)
        if not key:
            raise SystemExit("No The Odds API key. Set THE_ODDS_API_KEY or add "
                             "data/api_keys.json with key 'the-odds-api'.")
        fetch_odds_api(load_slate(args.days), key, args.regions)

    build_card(days=args.days, min_edge=args.min_edge,
               bankroll=args.bankroll, model=args.model)


if __name__ == "__main__":
    main()
