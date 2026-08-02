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
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from . import elo as E
from . import dataset_fingerprint as DATASET_FP
from . import identity as IDENTITY
from . import policy as POLICY
from . import power as P
from .edge import (HEADER, KELLY_FRACTION, MIN_EDGE,
                   ODDS_CSV, get_bankroll, model_prob, prepare_odds)
from .predictor import blend_predict, load_blend_weights

HERE = os.path.dirname(os.path.abspath(__file__))
UPCOMING_CSV = os.path.join(HERE, "data", "upcoming.csv")
CARD_MD = os.path.join(HERE, "data", "card.md")

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_ncaaf"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str | None:
    source = Path(path)
    return _sha256_bytes(source.read_bytes()) if source.exists() else None


def _atomic_write(path: str | Path, payload: bytes) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def _publish_card(text: str, manifest: dict) -> tuple[Path, Path]:
    """Atomically publish a card and its input/output evidence manifest."""
    card = Path(CARD_MD)
    card_payload = (text + "\n").encode()
    manifest = {**manifest, "card_sha256": _sha256_bytes(card_payload)}
    manifest_path = card.with_name("card_manifest.json")
    _atomic_write(card, card_payload)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n").encode(),
    )
    return card, manifest_path


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


# ── reviewed team identity ──────────────────────────────────────────────────

def _match_team(provider_name: str, teams: list[str], season: int) -> str | None:
    """Resolve only canonical CFBD names or reviewed provider aliases."""
    match = IDENTITY.resolve(provider_name, season, provider="the-odds-api")
    return match["canonical"] if match and match["canonical"] in teams else None


# ── slate ─────────────────────────────────────────────────────────────────────

def _slate_from_schedule_json(days: int) -> pd.DataFrame | None:
    """Offseason fallback: build the next week's slate from data/schedule_<yr>.json
    (CFBD /games) when upcoming.csv is empty (fetch_data mirrors lag preseason)."""
    import glob
    files = sorted(glob.glob(os.path.join(HERE, "data", "schedule_*.json")))
    if not files:
        return None
    sched = json.load(open(files[-1]))
    rows = [{"game_id": g.get("id"), "season": g.get("season"),
             "week": g.get("week"), "season_type": g.get("seasonType"),
             "date": pd.Timestamp(g["startDate"]).tz_localize(None).normalize(),
             "commence_time": g.get("startDate"),
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
    # At least one FBS side: FCS teams carry their own ratings now, so
    # FBS-vs-FCS games are priceable too (unrated teams get skipped later).
    up = up[(up["home_div"] == "fbs") | (up["away_div"] == "fbs")].copy()
    if up.empty:
        raise SystemExit("no upcoming fixtures with an FBS side")
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
    seasons = {int(value) for value in slate["season"].dropna().unique()}
    if len(seasons) != 1:
        raise RuntimeError(f"odds slate spans ambiguous seasons: {sorted(seasons)}")
    season = seasons.pop()
    slate_idx = {(r.home_team, r.away_team): r for r in slate.itertuples()}

    rows, unmatched, unresolved_relevant = [], [], []
    identity_version = IDENTITY.registry_version(season)
    slate_start = pd.Timestamp(slate["date"].min()).date()
    slate_end = pd.Timestamp(slate["date"].max()).date()
    for ev in events:
        home = _match_team(ev.get("home_team", ""), teams, season)
        away = _match_team(ev.get("away_team", ""), teams, season)
        fix = slate_idx.get((home, away)) if home and away else None
        if fix is None:
            label = f"{ev.get('away_team')} at {ev.get('home_team')}"
            unmatched.append(label)
            try:
                kickoff = pd.Timestamp(ev.get("commence_time")).date()
            except (TypeError, ValueError):
                kickoff = None
            if kickoff is not None and slate_start <= kickoff <= slate_end:
                unresolved_relevant.append(label)
            continue
        base = [str(fix.date.date()), home, away, int(bool(fix.neutral))]
        event_id = str(ev.get("id") or "")
        commence_time = str(ev.get("commence_time") or "")
        for book in ev.get("bookmakers") or []:
            bookmaker = str(book.get("key") or book.get("title") or "")
            quote_time = str(book.get("last_update") or "")
            for mk in book.get("markets") or []:
                for out in mk.get("outcomes") or []:
                    try:
                        price = float(out.get("price"))
                    except (TypeError, ValueError):
                        continue
                    if price <= 1.0:
                        continue
                    market_key = mk.get("key")
                    market = {"h2h": "ml", "spreads": "spread",
                              "totals": "total"}.get(market_key)
                    if not market:
                        continue
                    name = IDENTITY.fold(out.get("name", ""))
                    if market == "total":
                        side = name if name in ("over", "under") else None
                    elif name == IDENTITY.fold(ev.get("home_team", "")):
                        side = "home"
                    elif name == IDENTITY.fold(ev.get("away_team", "")):
                        side = "away"
                    else:
                        side = None
                    if side is None:
                        continue
                    line = "" if market == "ml" else out.get("point")
                    if market != "ml" and line is None:
                        continue
                    rows.append(base + [market, side, line, round(price, 3),
                                        event_id, getattr(fix, "game_id", ""),
                                        commence_time, bookmaker,
                                        quote_time, "the-odds-api", identity_version])

    if unresolved_relevant:
        preview = "; ".join(unresolved_relevant[:3])
        raise RuntimeError(
            f"{len(unresolved_relevant)} in-window Odds API event(s) failed exact "
            f"schedule/identity matching ({preview}); last-good odds retained"
        )
    if not rows:
        raise RuntimeError("odds-api returned no usable, matched CFB quotes; last-good odds retained")
    required_width = len(HEADER)
    if any(len(r) != required_width for r in rows):
        raise RuntimeError("internal odds normalization produced malformed rows")
    fd, tmp = tempfile.mkstemp(prefix="odds.", suffix=".csv", dir=HERE)
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)
        check = pd.read_csv(tmp)
        required = {"event_id", "cfbd_game_id", "commence_time", "bookmaker",
                    "quote_time", "source", "identity_version"}
        if check.empty or not required.issubset(check.columns):
            raise RuntimeError("normalized odds snapshot failed validation")
        if check[list(required - {"source"})].replace("", pd.NA).isna().any().any():
            raise RuntimeError("normalized odds snapshot lacks provider provenance")
        os.replace(tmp, ODDS_CSV)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    n_games = len({(r[1], r[2]) for r in rows})
    n_books = len({r[11] for r in rows})
    print(f"odds-api: {n_games} slate game(s), {n_books} book(s), "
          f"{len(rows)} executable quote rows -> {ODDS_CSV}")
    if unmatched:
        print(f"  (skipped {len(unmatched)} event(s) not on this week's slate or "
              f"unmatched by name, e.g. {unmatched[0]})")
    return n_games


# ── card ──────────────────────────────────────────────────────────────────────

def load_market(slate: pd.DataFrame) -> dict:
    """Best executable quote at the modal line for each matched fixture/side."""
    market: dict = {}
    if not os.path.exists(ODDS_CSV):
        return market
    odds = pd.read_csv(ODDS_CSV)
    odds = odds[odds["odds"].notna()]
    if odds.empty:
        return market
    seasons = {int(s) for s in slate["season"].dropna().unique()}
    odds = prepare_odds(odds, seasons)
    rejected = int((~odds["fixture_matched"] | ~odds["pair_complete"]).sum())
    odds = odds[odds["fixture_matched"] & odds["pair_complete"]].copy()
    if rejected:
        print(f"note: rejected {rejected} unmatched or unpaired odds row(s)")
    if odds.empty:
        return market

    picks = []
    for (_, _, _, market_name, side), group in odds.groupby(
            ["date", "home", "away", "market", "side"], dropna=False):
        if market_name == "ml":
            at_line = group
        else:
            mode = group["line"].mode(dropna=True)
            if mode.empty:
                continue
            at_line = group[group["line"] == mode.iloc[0]]
        picks.append(at_line.sort_values("odds", ascending=False).iloc[0])
    for r in picks:
        game = market.setdefault((r["date"], r["home"], r["away"]), {})
        game.setdefault(r["market"], {})[r["side"]] = (
            None if pd.isna(r["line"]) else float(r["line"]), float(r["odds"]),
            float(r["p_implied"]), r["bookmaker"], bool(r["quote_eligible"]))
    return market


def build_card(days: int = 7, min_edge: float = MIN_EDGE,
               bankroll: float | None = None, model: str = "blend") -> dict:
    slate = load_slate(days)
    market = load_market(slate)
    seasons = sorted({int(s) for s in slate["season"].dropna().unique()})
    if len(seasons) != 1:
        raise SystemExit(f"slate spans ambiguous seasons: {seasons}")
    target_season = seasons[0]
    divisions = {r.home_team: r.home_div for r in slate.itertuples()}
    divisions.update({r.away_team: r.away_div for r in slate.itertuples()})
    eparams, state_meta = E.build_as_of(
        target_season, as_of=date.today(), team_divisions=divisions)
    pparams = P.load_params()
    market_policy = POLICY.load_policy()
    executable_quotes = sum(
        1 for game in market.values() for sides in game.values()
        for quote in sides.values() if quote[4])
    recordable_quotes = sum(
        1 for game in market.values() for market_name, sides in game.items()
        for quote in sides.values()
        if quote[4] and POLICY.recordable(market_name, market_policy))
    # Elo rates FBS and FCS teams; blend_predict substitutes the pooled FCS
    # power entity for FCS sides, so Elo membership is the requirement.
    known = set(eparams[1])
    off_model = ~(slate["home_team"].isin(known) & slate["away_team"].isin(known))
    if off_model.any():
        print(f"note: skipped {int(off_model.sum())} game(s) with teams unknown to "
              f"the model (new FBS members / reclassified)")
        slate = slate[~off_model]
    slate_eligible = any(
        E.event_betting_eligible(state_meta, row.home_team, row.away_team)
        for row in slate.itertuples())
    betting_enabled = bool(
        state_meta["betting_eligible"] and recordable_quotes and slate_eligible)
    bk = bankroll if bankroll is not None else get_bankroll()

    lines_md, value = [], []
    ats_picks = 0
    for g in slate.itertuples():
        event_eligible = E.event_betting_eligible(
            state_meta, g.home_team, g.away_team)
        pred = blend_predict(eparams, pparams, g.home_team, g.away_team,
                             neutral=bool(g.neutral), model=model)
        fav = g.home_team if pred["margin"] >= 0 else g.away_team
        vs = "vs" if g.neutral else "at"
        lines_md.append(f"### {g.date.date()} — {g.away_team} {vs} {g.home_team}")
        lines_md.append(
            f"- Model: **{fav}** -{abs(pred['margin']):.1f} "
            f"(home win {pred['p1']*100:.1f}%) · total {pred['total']:.1f}")

        mkt = market.get((str(g.date.date()), g.home_team, g.away_team), {})
        sp = mkt.get("spread", {})
        home_line = sp.get("home", (None, None, None, None, False))[0]
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

        # value: each selected executable quote already carries the de-vigged
        # implied probability from its same-book paired market.
        for mkey, sides in mkt.items():
            for side, (line, o, p_imp, bookmaker, quote_eligible) in sides.items():
                if mkey != "ml" and line is None:
                    continue
                p_model = model_prob(pred, pparams, mkey, side, line)
                edge = p_model - p_imp
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
                        "event_id": f"cfbd:{getattr(g, 'game_id', '')}",
                        "market": mkey, "side": side,
                        "bookmaker": bookmaker or "legacy", "odds": o,
                        "p_model": p_model, "edge": edge,
                        "market_status": POLICY.status(mkey, market_policy),
                        "stake": (round(KELLY_FRACTION * kelly * bk, 2)
                                  if betting_enabled and event_eligible and quote_eligible
                                  and POLICY.recordable(mkey, market_policy) else 0.0),
                        "betting_eligible": bool(
                            betting_enabled and event_eligible and quote_eligible
                            and POLICY.recordable(mkey, market_policy))})
        lines_md.append("")

    value.sort(key=lambda v: -v["edge"])
    eligible = [v for v in value if v["betting_eligible"]]
    if eligible:
        try:
            from app.bankroll_store import preview_bets
            candidates = []
            for i, v in enumerate(eligible):
                v["_candidate_id"] = str(i)
                candidates.append({"candidate_id": str(i), "engine": "cfb",
                                   "event_id": v["event_id"], "stake": v["stake"]})
            capped = {c["candidate_id"]: c for c in preview_bets(candidates, bankroll=bk)}
            for v in eligible:
                raw = v["stake"]
                v["stake"] = float(capped.get(v["_candidate_id"], {}).get("stake", 0.0))
                v["stake_capped"] = v["stake"] < raw - 1e-9
        except Exception as exc:
            print(f"note: portfolio preview failed closed ({exc})")
            for v in eligible:
                v["stake"], v["stake_capped"] = 0.0, True
    state_label = (f"season {state_meta['model_season']} · state {state_meta['prior_mode']} · "
                   f"as of {state_meta['decision_date']} · snapshot {state_meta['snapshot_hash']}")
    md = [f"# CFB weekly card — {date.today()}",
          f"_{len(slate)} FBS games · model: {model} · bankroll £{bk:.2f} · "
          f"quarter-Kelly · min edge {min_edge:.0%}_",
          f"_{state_label}_", ""]
    if not betting_enabled:
        if not state_meta["betting_eligible"]:
            reason = ("the preseason snapshot does not have adequate target-season talent "
                      "and returning-production coverage")
        elif not executable_quotes:
            reason = "there are no fresh, matched, complete bookmaker quotes"
        elif not slate_eligible:
            reason = "every slate event contains a team still behind its evidence gate"
        else:
            reason = "no CFB market is currently approved for real-money recording"
        md += [f"> **Staking disabled:** {reason}. Edges below are diagnostic only.", ""]
    md.append("## Value bets" if betting_enabled
              else "## Diagnostic edges — no staking")
    if value:
        md.append("| Date | Game | Bet | Status | Book | Odds | Model % | Edge | Stake |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for v in value:
            md.append(f"| {v['date']} | {v['game']} | **{v['bet']}** | {v['market_status']} "
                      f"| {v['bookmaker']} | {v['odds']:.2f} "
                      f"| {v['p_model']*100:.1f}% | {v['edge']*100:+.1f}% | £{v['stake']:.2f} |")
    else:
        md.append("_None at this threshold" +
                  (" (no odds loaded — run with --odds-api or fill cfb/odds.csv)_"
                   if not market else "._"))
    md += ["", "## Matchups (straight-up + ATS)", ""] + lines_md

    text = "\n".join(md)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision_date": str(state_meta["decision_date"]),
        "model": model,
        "model_state": state_meta,
        "blend_weights": load_blend_weights(),
        "market_policy": market_policy,
        "identity_version": IDENTITY.registry_version(target_season),
        "validation_data": DATASET_FP.compact_snapshot(),
        "odds_sha256": _sha256_file(ODDS_CSV),
        "configuration": {
            "days": days,
            "minimum_edge": min_edge,
            "bankroll": bk,
        },
        "result": {
            "games": int(len(slate)),
            "ats_picks": int(ats_picks),
            "value_bets": int(sum(1 for v in value if v["betting_eligible"])),
            "diagnostic_edges": int(sum(
                1 for v in value if not v["betting_eligible"])),
            "betting_eligible": bool(betting_enabled),
            "total_stake": round(sum(float(v["stake"]) for v in value), 2),
        },
    }
    card_path, manifest_path = _publish_card(text, manifest)
    print(text)
    print(f"\ncard -> {card_path}")
    print(f"manifest -> {manifest_path}")
    return {"games": len(slate), "ats_picks": ats_picks,
            "value_bets": sum(1 for v in value if v["betting_eligible"]),
            "diagnostic_edges": sum(1 for v in value if not v["betting_eligible"]),
            "betting_eligible": betting_enabled,
            "model_state": state_meta, "card": str(card_path),
            "manifest": str(manifest_path)}


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
        P.save_params(params)
        print(f"power refit: {len(params['teams'])} teams as of {params['asof']}")

    if args.odds_api:
        key = _odds_api_key(args.api_key)
        if not key:
            raise SystemExit("No The Odds API key. Set THE_ODDS_API_KEY or add "
                             "data/api_keys.json with key 'the-odds-api'.")
        fetch_odds_api(load_slate(args.days), key, args.regions)

    build_card(days=args.days, min_edge=args.min_edge,
               bankroll=args.bankroll, model=args.model)
    if args.odds_api:
        from . import live_evidence
        evidence = live_evidence.capture()
        print(f"live evidence: {evidence['quote_capture']['new_quote_rows']} new quotes, "
              f"{evidence['signal_capture']['new_paper_signals']} new paper signals")


if __name__ == "__main__":
    main()
