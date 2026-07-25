"""tennis/season.py — the front door for the tennis engine.

Same mental model as the World Cup engine: pull the week's tournament list, take
all active draws for the selected tour, let the fitted model price them, and
print the best bets — round by round (R128 → … → final). Everything else in
this package (`fetch`, `model`, `simulate`, `edge`, …) is plumbing this drives;
you should not need to call those directly for a normal week.

    python -m tennis.season --schedule              # live ATP tournaments + draws
    python -m tennis.season --schedule --tour wta
    python -m tennis.season                         # price all active ATP events
    python -m tennis.season --tour wta               # price all active WTA events
    python -m tennis.season --tour both              # price both tours
    python -m tennis.season --tour wta --event Berlin  # narrow to one event
    python -m tennis.season --no-fetch              # reprice the saved draw.csv
    python -m tennis.season --event Wimbledon --odds-api

The draws are pulled automatically from ESPN and saved to `tennis/data/draw.csv`
(so `simulate`/`edge` keep working). Book odds can be fetched from The Odds API
with `--odds-api` or entered manually in `tennis/data/odds.csv`
(`--odds-template` writes a skeleton). Any match the model rates above the market
shows an edge and a stake. Without odds the card still gives you the model's pick
and win probability for every match in every round.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from . import calibrate as C
from . import market as MK
from . import model as M
from . import portfolio as PORT
from .providers import DATA_DIR, fetch_draws
from .rounds import DRAW_COLUMNS, ROUND_LABEL, is_tbd, ordered_rounds

DRAW_CSV = DATA_DIR / "draw.csv"
ODDS_CSV = DATA_DIR / "odds.csv"
CARD_MD = DATA_DIR / "card.md"
DEFAULT_KELLY = 0.25

# ── schedule ──────────────────────────────────────────────────────────────────

def print_schedule(tour: str = "atp") -> None:
    if tour == "both":
        print_schedule("atp")
        print_schedule("wta")
        return
    draws = _all_draws(tour)
    print(f"{tour.upper()} — {len(draws)} active tournament(s)\n")
    if not draws:
        print("  (none returned — ESPN may be offline or no play this week)")
        return
    for d in draws:
        rounds = _by_round(d.matches)
        span = ", ".join(f"{r}×{len(rounds[r])}" for r in _ordered(rounds))
        print(f"  · {d.tourney_name}  [{d.surface}, best-of-{d.best_of}]  {span}")
    print(f"\nPrice one with: python -m tennis.season --tour {tour} "
          f"--event \"{draws[0].tourney_name.split()[0]}\"")


def _all_draws(tour: str) -> list:
    from .providers import _espn_draw
    return _espn_draw(tour)


# ── card ────────────────────────────────────────────────────────────────────

def build_card(
    *,
    tour: str = "atp",
    tourney: str = "",
    fetch: bool = True,
    bankroll: float = 100.0,
    peak: float | None = None,
    kelly: float = DEFAULT_KELLY,
    min_edge: float = 0.0,
    calibrated: bool = True,
    blended: bool = True,
    fetch_odds: bool = False,
    api_key: str | None = None,
    odds_regions: str = "eu",
    output: Path = CARD_MD,
) -> dict:
    """Pull the draw, price every match with the fitted model + book odds, and
    write a round-by-round best-bets card. Returns a summary dict."""
    params = M.load_params(tour)
    if not params:
        raise ValueError(f"No fitted {tour.upper()} model. Run: "
                         f"python -m tennis.model --fit --tour {tour}")
    M.assert_params_fresh(params)

    notes: list[str] = []
    draws = fetch_draws(tour, tourney) if fetch else []
    if draws:
        # A filtered refresh replaces only that event; an unfiltered refresh
        # replaces the active tour's event set while retaining the other tour.
        replace_events = {d.tourney_name for d in draws} if tourney else None
        write_draw_csv(draws, replace_events=replace_events)
        if len(draws) == 1:
            notes.append(f"draw: {draws[0].tourney_name} (ESPN) → draw.csv")
        else:
            names = ", ".join(d.tourney_name for d in draws)
            notes.append(f"draw: {len(draws)} events ({names}) (ESPN) → draw.csv")
        matches = [_draw_match_row(m, d) for d in draws for m in d.matches]
    else:
        matches = _load_draw_csv(tour, tourney)
        notes.append(f"draw: {len({m.get('tourney_name', '') for m in matches})} saved event(s)"
                     if matches
                     else "draw: none found")

    if fetch_odds:
        try:
            from . import fetch as FETCH
            odds_event = tourney
            odds_rows = FETCH.fetch_odds_api(tour=tour, event=odds_event,
                                             api_key=api_key,
                                             regions=odds_regions)
            if odds_rows:
                FETCH.write_odds_csv(odds_rows)
                notes.append(f"odds: {len(odds_rows)} h2h rows "
                             f"(The Odds API → odds.csv)")
            else:
                notes.append("odds: no h2h rows fetched")
        except Exception as exc:
            notes.append(f"odds: fetch skipped ({exc})")

    odds = _load_odds(tour)
    maps = C.load_maps() if calibrated else None
    w_mkt = MK.blend_weights().get("match_winner", 0.5)

    by_event: dict[str, dict] = {}
    n_bets = 0
    for m in matches:
        rnd, a, b = m["round"], m["player_a"], m["player_b"]
        event_name = m.get("tourney_name") or ""
        surface = (m.get("surface") or "hard").lower()
        try:
            best_of = int(float(m.get("best_of") or 3))
        except (TypeError, ValueError):
            best_of = 3
        odds_pair = odds.get((event_name.lower(), _key(a, b)))
        if odds_pair is None:
            odds_pair = odds.get(("", _key(a, b)))
        row = _price_match(a, b, surface, params, maps,
                           odds_pair, w_mkt if blended else None,
                           bankroll, kelly, state=m.get("state", ""),
                           winner=m.get("winner", ""))
        event = by_event.setdefault(event_name, {
            "surface": surface, "best_of": best_of, "by_round": {}})
        event["by_round"].setdefault(rnd or "R?", []).append(row)

    priced_rows = [
        row
        for event in by_event.values()
        for round_rows in event["by_round"].values()
        for row in round_rows
        if row.get("ev_per_unit", 0.0) > 0 and row.get("stake_gbp", 0.0) > 0
    ]
    for row in priced_rows:
        row["recommended"] = False
    staked = PORT.apply_portfolio(
        priced_rows, bankroll=bankroll, peak=peak or bankroll,
    )
    for row in staked:
        row["stake"] = row["stake_gbp"]
        row["recommended"] = (
            row["stake_gbp"] >= 0.5 and row["ev_per_unit"] > min_edge
        )
    n_bets = sum(1 for row in staked if row["recommended"])
    notes.append(f"portfolio: {PORT.summary(staked, bankroll, peak or bankroll)}")

    text = _render_card(tour, by_event, min_edge, notes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    event_names = [name for name in by_event if name]
    return {"event": event_names[0] if len(event_names) == 1 else "",
            "events": event_names, "tour": tour, "matches": len(matches),
            "bets": n_bets, "output": str(output), "notes": notes}


def _draw_match_row(m, draw) -> dict:
    return {"tour": draw.tour, "tourney_name": draw.tourney_name,
            "event_id": getattr(draw, "event_id", ""),
            "surface": draw.surface, "best_of": draw.best_of,
            "round": m.round, "player_a": m.player_a, "player_b": m.player_b,
            "state": getattr(m, "state", ""), "winner": getattr(m, "winner", ""),
            "score": getattr(m, "score", ""), "match_id": getattr(m, "match_id", "")}


def _price_match(a, b, surface, params, maps, odds_pair, w_mkt,
                 bankroll, kelly, state: str = "", winner: str = "") -> dict:
    base_row = {"round_a": a, "round_b": b, "favourite": "", "p_fav": None,
                "odds": 0.0, "p_market": 0.0, "edge": 0.0, "stake": 0.0,
                "ev_per_unit": 0.0, "stake_gbp": 0.0,
                "recommended": False, "completed": False, "status": ""}
    if state == "post" and winner:
        base_row.update(favourite=winner, completed=True, status="Complete")
        return base_row
    if is_tbd(a) or is_tbd(b):
        base_row.update(favourite="TBD", status="Pending")
        return base_row

    prediction = M.predict_match(a, b, surface, params)
    if prediction["unresolved"]:
        base_row.update(
            favourite="Unpriced",
            status=f"Unknown: {', '.join(prediction['unresolved'])}",
        )
        return base_row
    p_a = prediction["p_a"]
    if maps:
        p_a = C.apply_oriented(a, b, p_a, maps)
    fav, p_fav = (a, p_a) if p_a >= 0.5 else (b, 1 - p_a)

    opponent = b if fav == a else a
    row = {**base_row, "favourite": fav, "p_fav": p_fav,
           "player": fav, "opponent": opponent, "p_model": p_fav,
           "status": "Live" if state == "in" else ""}
    if (not odds_pair or a.strip().lower() not in odds_pair
            or b.strip().lower() not in odds_pair):
        return row
    # Price the favourite's side with the book. odds_pair maps lowered name → odds.
    oa, ob = odds_pair[a.strip().lower()], odds_pair[b.strip().lower()]
    o_fav = oa if fav == a else ob
    pm_a, pm_b = MK.devig_two_way(oa, ob)
    pm_fav = pm_a if fav == a else pm_b
    p_eff = MK.blend(p_fav, pm_fav, w_mkt) if w_mkt is not None else p_fav
    ev = p_eff * o_fav - 1.0
    row.update(odds=o_fav, p_market=pm_fav, p_blend=p_eff,
               edge=ev, ev_per_unit=ev)
    if ev > 0:
        row["stake_gbp"] = PORT.kelly_stake(p_eff, o_fav, bankroll, kelly)
    return row


def _render_card(tour, by_event, min_edge, notes) -> str:
    generated = time.strftime("%Y-%m-%d %H:%M")
    event_names = [name for name in by_event if name]
    title = event_names[0] if len(event_names) == 1 else tour.upper()
    L = [
        f"# {title} — Best Bets",
        "",
        f"_Generated {generated} · {tour.upper()} · {len(by_event)} event(s) · fitted model_",
        "",
    ]
    if not by_event:
        L += ["_No draw available. Run `--schedule` to see live tournaments, or "
              "fill in `tennis/data/draw.csv`._", ""]
        return "\n".join(L)

    for event_name, event in by_event.items():
        by_round = event["by_round"]
        if len(by_event) > 1:
            L.append(f"## {event_name or tour.upper()}")
            L.append("")
        L.append(f"_{event['surface']} · best-of-{event['best_of']}_")
        L.append("")
        for rnd in _ordered(by_round):
            rows = by_round[rnd]
            L.append(f"### {ROUND_LABEL.get(rnd, rnd)}")
            L.append("")
            L.append("| Match | Status | Model pick | P(win) | Odds | Market | Edge | Stake |")
            L.append("|---|---|---|--:|--:|--:|--:|--:|")
            rows.sort(key=lambda r: (r["completed"], -r["edge"] if r["odds"] else 1,
                                     -(r["p_fav"] or 0)))
            for r in rows:
                match = f"{r['round_a']} v {r['round_b']}"
                status = r.get("status") or "To play"
                if r["odds"]:
                    odds = f"{r['odds']:.2f}"
                    mkt = f"{r['p_market']*100:.0f}%"
                    edge = f"{r['edge']*100:+.1f}%"
                    stake = f"£{r['stake']:.2f}" if r["recommended"] else "—"
                else:
                    odds = mkt = edge = stake = "—"
                pwin = f"{r['p_fav']*100:.0f}%" if r['p_fav'] is not None else "—"
                pick = f"**{r['favourite']}**" if r["recommended"] else r["favourite"]
                L.append(f"| {match} | {status} | {pick} | {pwin} | {odds} "
                         f"| {mkt} | {edge} | {stake} |")
            L.append("")

    L.append("## Notes")
    L.append("")
    n_bets = sum(1 for event in by_event.values()
                 for rows in event["by_round"].values()
                 for r in rows if r["recommended"] and r["edge"] >= min_edge)
    L.append(f"- {n_bets} bet(s) backed (model edge over the book, staked).")
    L.append("- Bold pick = staked bet. Add prices to `tennis/data/odds.csv` to "
             "price more matches.")
    for n in notes:
        L.append(f"- {n}")
    L.append("")
    return "\n".join(L)


# ── draw + odds I/O ──────────────────────────────────────────────────────────

def write_draw_csv(draws, path: Path = DRAW_CSV,
                   replace_events: set[str] | None = None) -> Path:
    """Persist one or more fetched draws without dropping other tours/events."""
    if not isinstance(draws, (list, tuple)):
        draws = [draws]
    draws = [d for d in draws if d is not None]
    if not draws:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    target_tours = {str(d.tour).lower() for d in draws}
    target_events = {str(e).lower() for e in (replace_events or set())}
    existing: list[dict] = []
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row_tour = (row.get("tour") or "").lower()
                row_event = (row.get("tourney_name") or "").lower()
                if row_tour in target_tours and (
                        replace_events is None or row_event in target_events):
                    continue
                existing.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DRAW_COLUMNS)
        w.writeheader()
        w.writerows({c: row.get(c, "") for c in DRAW_COLUMNS} for row in existing)
        for draw in draws:
            for m in draw.matches:
                w.writerow({"tour": draw.tour, "tourney_name": draw.tourney_name,
                            "event_id": getattr(draw, "event_id", ""),
                            "surface": draw.surface, "best_of": draw.best_of,
                            "round": m.round, "player_a": m.player_a,
                            "player_b": m.player_b, "state": getattr(m, "state", ""),
                            "winner": getattr(m, "winner", ""),
                            "score": getattr(m, "score", ""),
                            "match_id": getattr(m, "match_id", "")})
    return path


def _load_draw_csv(tour: str, tourney_filter: str = "") -> list[dict]:
    if not DRAW_CSV.exists():
        return []
    matches = []
    wanted = tourney_filter.lower().strip()
    with open(DRAW_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("tour") or "").lower() not in ("", tour):
                continue
            if wanted and wanted not in (r.get("tourney_name") or "").lower():
                continue
            a, b = (r.get("player_a") or "").strip(), (r.get("player_b") or "").strip()
            if not (a and b):
                continue
            matches.append({"tour": (r.get("tour") or tour).lower(),
                            "tourney_name": r.get("tourney_name") or "",
                            "event_id": r.get("event_id") or "",
                            "surface": (r.get("surface") or "hard").lower(),
                            "best_of": r.get("best_of") or 3,
                            "round": r.get("round") or "R?", "player_a": a,
                            "player_b": b, "state": r.get("state") or "",
                            "winner": r.get("winner") or "",
                            "score": r.get("score") or "",
                            "match_id": r.get("match_id") or ""})
    return matches


def _load_odds(tour: str) -> dict:
    """Return event-aware, order-independent odds lookups.

    Keys are ``(lower_tourney_name, pair_key)``.  A blank tournament name is a
    deliberate fallback for legacy odds.csv files that predate multi-event
    support.
    """
    if not ODDS_CSV.exists():
        return {}
    out = {}
    with open(ODDS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("tour") or "").lower() not in ("", tour):
                continue
            a, b = (r.get("player_a") or "").strip(), (r.get("player_b") or "").strip()
            try:
                oa, ob = float(r["odds_a"]), float(r["odds_b"])
            except (ValueError, KeyError, TypeError):
                continue
            if a and b:
                event = (r.get("tourney_name") or "").strip().lower()
                out[(event, _key(a, b))] = {a.lower(): oa, b.lower(): ob}
    return out


def _key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a.strip().lower(), b.strip().lower())))


def _by_round(matches) -> dict:
    out: dict[str, list] = {}
    for m in matches:
        out.setdefault(m.round or "R?", []).append(m)
    return out


def _ordered(rounds) -> list[str]:
    return ordered_rounds(rounds.keys())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tennis engine — live schedule and round-by-round best-bets card")
    ap.add_argument("--schedule", action="store_true",
                    help="list live tournaments + draws for the tour and exit")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta", "both"])
    ap.add_argument("--event", default="", help="tournament name filter (e.g. Wimbledon)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="reprice the saved draw.csv instead of pulling ESPN")
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--peak", type=float, default=None,
                    help="peak bankroll for the drawdown staking brake")
    ap.add_argument("--kelly", type=float, default=DEFAULT_KELLY)
    ap.add_argument("--min-edge", type=float, default=0.0,
                    help="minimum EV fraction to back (0.02 = 2%%; default 0)")
    ap.add_argument("--odds-api", action="store_true",
                    help="fetch h2h prices from The Odds API into odds.csv before pricing")
    ap.add_argument("--api-key", default=None,
                    help="The Odds API key; defaults to THE_ODDS_API_KEY/data/api_keys.json")
    ap.add_argument("--regions", default="eu",
                    help="The Odds API regions for --odds-api (default: eu)")
    ap.add_argument("--output", default="",
                    help="card path; --tour both uses this for the index")
    args = ap.parse_args()

    if args.schedule:
        print_schedule(args.tour)
        return

    common = dict(tourney=args.event, fetch=not args.no_fetch,
                  bankroll=args.bankroll, peak=args.peak, kelly=args.kelly,
                  min_edge=args.min_edge, fetch_odds=args.odds_api,
                  api_key=args.api_key, odds_regions=args.regions)
    if args.tour == "both":
        summaries = []
        for tour in ("atp", "wta"):
            summary = build_card(tour=tour, output=DATA_DIR / f"card_{tour}.md",
                                 **common)
            summaries.append(summary)
            print(f"{tour.upper()} — {summary['matches']} match(es), "
                  f"{summary['bets']} bet(s) backed")
            for n in summary["notes"]:
                print(f"  · {n}")
            print(f"Card → {summary['output']}")
        index = Path(args.output) if args.output else CARD_MD
        index.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Tennis — Best Bets", "",
                 "_Run both tours with all active events._", ""]
        for summary in summaries:
            card_name = Path(summary["output"]).name
            lines.extend([f"## {summary['tour'].upper()}", "",
                          f"{len(summary['events'])} event(s), "
                          f"{summary['matches']} match(es), "
                          f"[{card_name}]({card_name})", ""])
        index.write_text("\n".join(lines))
        print(f"Index → {index}")
        return

    default_output = DATA_DIR / f"card_{args.tour}.md"
    summary = build_card(tour=args.tour, output=Path(args.output) if args.output else default_output,
                         **common)
    print(f"{summary['event'] or summary['tour'].upper()} — "
          f"{summary['matches']} match(es), {summary['bets']} bet(s) backed")
    for n in summary["notes"]:
        print(f"  · {n}")
    print(f"Card → {summary['output']}")


if __name__ == "__main__":
    main()
