"""tennis/season.py — the one front door for the tennis engine.

Same mental model as the World Cup engine: pull the week's tournament list, take
a draw, let the fitted model price it, and print the best bets — round by round
(R128 → … → final). Everything else in this package (`fetch`, `model`,
`simulate`, `edge`, …) is plumbing this drives; you should not need to call those
directly for a normal week.

    python -m tennis.season --schedule              # live ATP tournaments + draws
    python -m tennis.season --schedule --tour wta
    python -m tennis.season                         # price the current ATP draw
    python -m tennis.season --tour wta --event Berlin
    python -m tennis.season --no-fetch              # reprice the saved draw.csv
    python -m tennis.season --event Wimbledon --odds-api

The draw is pulled automatically from ESPN and saved to `tennis/data/draw.csv`
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
from .providers import DATA_DIR, fetch_draw

DRAW_CSV = DATA_DIR / "draw.csv"
ODDS_CSV = DATA_DIR / "odds.csv"
CARD_MD = DATA_DIR / "card.md"
DEFAULT_KELLY = 0.25

DRAW_COLUMNS = ["tour", "tourney_name", "surface", "best_of", "round",
                "player_a", "player_b", "state", "winner", "score", "match_id"]

# Bracket order, earliest round first.
_ROUND_ORDER = ["Q1", "Q2", "QF-Q", "R256", "R128", "R64", "R32", "R16",
                "R1", "R2", "R3", "R4", "QF", "SF", "F"]
_ROUND_LABEL = {"R128": "Round of 128", "R64": "Round of 64", "R32": "Round of 32",
                "R16": "Round of 16", "QF": "Quarterfinals", "SF": "Semifinals",
                "F": "Final", "Q1": "Qualifying R1", "Q2": "Qualifying R2",
                "QF-Q": "Qualifying final", "R256": "Round of 256",
                "R1": "Round 1", "R2": "Round 2", "R3": "Round 3",
                "R4": "Round 4"}


# ── schedule ──────────────────────────────────────────────────────────────────

def print_schedule(tour: str = "atp") -> None:
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

    draw = fetch_draw(tour, tourney) if fetch else None
    notes: list[str] = []
    if draw:
        write_draw_csv(draw)
        notes.append(f"draw: {draw.tourney_name} (ESPN) → draw.csv")
        tourney_name, surface, best_of = draw.tourney_name, draw.surface, draw.best_of
        matches = [_draw_match_row(m) for m in draw.matches]
    else:
        matches, surface, best_of, tourney_name = _load_draw_csv(tour)
        notes.append(f"draw: {tourney_name} (saved draw.csv)" if matches
                     else "draw: none found")

    if fetch_odds:
        try:
            from . import fetch as FETCH
            odds_event = tourney_name or tourney
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
    h2h_fn = _h2h_fn(tour)
    w_mkt = MK.blend_weights().get("match_winner", 0.5)

    by_round: dict[str, list[dict]] = {}
    n_bets = 0
    for m in matches:
        rnd, a, b = m["round"], m["player_a"], m["player_b"]
        row = _price_match(a, b, surface, params, h2h_fn, maps,
                           odds.get(_key(a, b)), w_mkt if blended else None,
                           bankroll, kelly, state=m.get("state", ""),
                           winner=m.get("winner", ""))
        if row.get("recommended") and row["edge"] >= min_edge:
            n_bets += 1
        by_round.setdefault(rnd or "R?", []).append(row)

    text = _render_card(tourney_name, tour, surface, best_of, by_round,
                        min_edge, notes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return {"event": tourney_name, "tour": tour, "matches": len(matches),
            "bets": n_bets, "output": str(output), "notes": notes}


def _draw_match_row(m) -> dict:
    return {"round": m.round, "player_a": m.player_a, "player_b": m.player_b,
            "state": getattr(m, "state", ""), "winner": getattr(m, "winner", ""),
            "score": getattr(m, "score", ""), "match_id": getattr(m, "match_id", "")}


def _is_tbd(name: str) -> bool:
    return not str(name or "").strip() or str(name).strip().upper() == "TBD"


def _price_match(a, b, surface, params, h2h_fn, maps, odds_pair, w_mkt,
                 bankroll, kelly, state: str = "", winner: str = "") -> dict:
    base_row = {"round_a": a, "round_b": b, "favourite": "", "p_fav": None,
                "odds": 0.0, "p_market": 0.0, "edge": 0.0, "stake": 0.0,
                "recommended": False, "completed": False, "status": ""}
    if state == "post" and winner:
        base_row.update(favourite=winner, completed=True, status="Complete")
        return base_row
    if _is_tbd(a) or _is_tbd(b):
        base_row.update(favourite="TBD", status="Pending")
        return base_row

    h2h = h2h_fn(a, b, surface) if h2h_fn else 0.0
    p_a = M.predict_match(a, b, surface, params, h2h_log_odds=h2h)["p_a"]
    if maps:
        p_a = C.apply_one("match_winner", p_a, maps)
    fav, p_fav = (a, p_a) if p_a >= 0.5 else (b, 1 - p_a)

    row = {**base_row, "favourite": fav, "p_fav": p_fav,
           "status": "Live" if state == "in" else ""}
    if not odds_pair:
        return row
    # Price the favourite's side with the book. odds_pair maps lowered name → odds.
    oa, ob = odds_pair[a.strip().lower()], odds_pair[b.strip().lower()]
    o_fav = oa if fav == a else ob
    pm_a, pm_b = MK.devig_two_way(oa, ob)
    pm_fav = pm_a if fav == a else pm_b
    p_eff = MK.blend(p_fav, pm_fav, w_mkt) if w_mkt is not None else p_fav
    ev = p_eff * o_fav - 1.0
    row.update(odds=o_fav, p_market=pm_fav, edge=ev * 100)
    if ev > 0:
        b_odds = o_fav - 1.0
        f = (b_odds * p_eff - (1 - p_eff)) / b_odds if b_odds > 0 else 0.0
        row["stake"] = round(bankroll * max(0.0, f) * kelly, 2)
        row["recommended"] = row["stake"] >= 0.5
    return row


def _render_card(tourney_name, tour, surface, best_of, by_round, min_edge,
                 notes) -> str:
    generated = time.strftime("%Y-%m-%d %H:%M")
    L = [
        f"# {tourney_name or tour.upper()} — Best Bets",
        "",
        f"_Generated {generated} · {tour.upper()} · {surface} · "
        f"best-of-{best_of} · fitted model_",
        "",
    ]
    if not by_round:
        L += ["_No draw available. Run `--schedule` to see live tournaments, or "
              "fill in `tennis/data/draw.csv`._", ""]
        return "\n".join(L)

    for rnd in _ordered(by_round):
        rows = by_round[rnd]
        L.append(f"## {_ROUND_LABEL.get(rnd, rnd)}")
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
                edge = f"{r['edge']:+.1f}%"
                stake = f"£{r['stake']:.2f}" if r["recommended"] else "—"
            else:
                odds = mkt = edge = stake = "—"
            pwin = f"{r['p_fav']*100:.0f}%" if r["p_fav"] is not None else "—"
            pick = f"**{r['favourite']}**" if r["recommended"] else r["favourite"]
            L.append(f"| {match} | {status} | {pick} | {pwin} | {odds} "
                     f"| {mkt} | {edge} | {stake} |")
        L.append("")

    L.append("## Notes")
    L.append("")
    n_bets = sum(1 for rows in by_round.values() for r in rows
                 if r["recommended"] and r["edge"] >= min_edge)
    L.append(f"- {n_bets} bet(s) backed (model edge over the book, staked).")
    L.append("- Bold pick = staked bet. Add prices to `tennis/data/odds.csv` to "
             "price more matches.")
    for n in notes:
        L.append(f"- {n}")
    L.append("")
    return "\n".join(L)


# ── draw + odds I/O ──────────────────────────────────────────────────────────

def write_draw_csv(draw, path: Path = DRAW_CSV) -> Path:
    """Persist a fetched TournamentDraw to draw.csv (the simulate/edge contract)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DRAW_COLUMNS)
        w.writeheader()
        for m in draw.matches:
            w.writerow({"tour": draw.tour, "tourney_name": draw.tourney_name,
                        "surface": draw.surface, "best_of": draw.best_of,
                        "round": m.round, "player_a": m.player_a,
                        "player_b": m.player_b, "state": getattr(m, "state", ""),
                        "winner": getattr(m, "winner", ""),
                        "score": getattr(m, "score", ""),
                        "match_id": getattr(m, "match_id", "")})
    return path


def _load_draw_csv(tour: str):
    if not DRAW_CSV.exists():
        return [], "hard", 3, ""
    matches, surface, best_of, name = [], "hard", 3, ""
    with open(DRAW_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("tour") or "").lower() not in ("", tour):
                continue
            a, b = (r.get("player_a") or "").strip(), (r.get("player_b") or "").strip()
            if not (a and b):
                continue
            matches.append({"round": r.get("round") or "R?", "player_a": a,
                            "player_b": b, "state": r.get("state") or "",
                            "winner": r.get("winner") or "",
                            "score": r.get("score") or "",
                            "match_id": r.get("match_id") or ""})
            surface = (r.get("surface") or surface).lower()
            name = r.get("tourney_name") or name
            try:
                best_of = int(float(r.get("best_of") or best_of))
            except ValueError:
                pass
    return matches, surface, best_of, name


def _load_odds(tour: str) -> dict:
    """{frozenset(name_a, name_b) key → {name_lower: decimal_odds}}, so a lookup
    is order-independent and each player's price is recovered by name."""
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
                out[_key(a, b)] = {a.lower(): oa, b.lower(): ob}
    return out


def _key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a.strip().lower(), b.strip().lower())))


def _h2h_fn(tour: str):
    try:
        df = M.load_matches_df()
    except FileNotFoundError:
        return None
    df = df[df["tour"].astype(str).str.lower() == tour]
    if df.empty:
        return None
    return lambda a, b, s: M.h2h_log_odds_from_df(a, b, s, df)


def _by_round(matches) -> dict:
    out: dict[str, list] = {}
    for m in matches:
        out.setdefault(m.round or "R?", []).append(m)
    return out


def _ordered(rounds) -> list[str]:
    keys = list(rounds.keys())
    return sorted(keys, key=lambda r: _ROUND_ORDER.index(r)
                  if r in _ROUND_ORDER else len(_ROUND_ORDER))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tennis engine — live schedule and round-by-round best-bets card")
    ap.add_argument("--schedule", action="store_true",
                    help="list live tournaments + draws for the tour and exit")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--event", default="", help="tournament name filter (e.g. Wimbledon)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="reprice the saved draw.csv instead of pulling ESPN")
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--kelly", type=float, default=DEFAULT_KELLY)
    ap.add_argument("--min-edge", type=float, default=0.0,
                    help="min %% edge to count a bet as backed (default 0)")
    ap.add_argument("--odds-api", action="store_true",
                    help="fetch h2h prices from The Odds API into odds.csv before pricing")
    ap.add_argument("--api-key", default=None,
                    help="The Odds API key; defaults to THE_ODDS_API_KEY/data/api_keys.json")
    ap.add_argument("--regions", default="eu",
                    help="The Odds API regions for --odds-api (default: eu)")
    args = ap.parse_args()

    if args.schedule:
        print_schedule(args.tour)
        return

    summary = build_card(tour=args.tour, tourney=args.event,
                         fetch=not args.no_fetch, bankroll=args.bankroll,
                         kelly=args.kelly, min_edge=args.min_edge,
                         fetch_odds=args.odds_api, api_key=args.api_key,
                         odds_regions=args.regions)
    print(f"{summary['event'] or summary['tour'].upper()} — "
          f"{summary['matches']} match(es), {summary['bets']} bet(s) backed")
    for n in summary["notes"]:
        print(f"  · {n}")
    print(f"Card → {summary['output']}")


if __name__ == "__main__":
    main()
