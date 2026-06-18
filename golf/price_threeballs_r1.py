"""
golf/price_threeballs_r1.py — Price single-round (Round 1) 3-ball markets.

Sky Bet's "3 Ball (Round 1)" settles on the LOWEST SCORE IN ROUND 1 only, not on
72-hole finish. The engine's simulate.py 3-ball logic settles on the tournament
total, so it is the wrong model here. This module prices the single-round market:

    round_score[player] ~ Normal(-rating, sigma)      (one round)
    P(player wins 3-ball) = P(their round score is the lowest of the three)

rating/sigma come straight from the fitted model via model.predict_field (so the
same skill/form/major-sigma logic as everywhere else; course fit applied if known).
We Monte-Carlo one shared round for the whole field, then read off each trio.

Output: data/threeballs_r1_edges.csv  +  a printed card.
Usage:  python3 price_threeballs_r1.py [--sims 200000] [--kelly 0.25]
        [--course "Shinnecock Hills"] [--major] [--min-edge 0.0] [--bankroll 100]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import model as M
import market as MK
import edge as E

DATA = Path(__file__).parent / "data"
RAW = DATA / "threeballs_r1_raw.txt"
OUT = DATA / "threeballs_r1_edges.csv"

HEADER_RE = re.compile(r"^3\s*Ball.*-\s*(.+)$", re.I)
NUM_RE = re.compile(r"^\d+(\.\d+)?$")


def parse_raw(path: Path) -> list[dict]:
    """Parse the pasted board into [{group, players:[(name,odds),...]}]."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    groups, cur, pending = [], None, []
    for ln in lines:
        h = HEADER_RE.match(ln)
        if h:
            if cur is not None:
                groups.append(cur)
            cur = {"group": h.group(1).strip(), "players": []}
            pending = []
            continue
        if cur is None:
            continue
        if NUM_RE.match(ln):                       # an odds line → pairs with last name
            if pending:
                cur["players"].append((pending.pop(0), float(ln)))
        else:                                      # a name line
            pending.append(ln)
    if cur is not None:
        groups.append(cur)
    return [g for g in groups if len(g["players"]) == 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--course", default="Shinnecock Hills")
    ap.add_argument("--major", action="store_true", default=True)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--bankroll", type=float, default=None)
    ap.add_argument("--min-edge", type=float, default=0.0, help="min EV%% to flag a bet")
    ap.add_argument("--min-rounds", type=int, default=60,
                    help="exclude players with fewer fitted rounds (thin sample, "
                         "unreliable skill estimate) from recommended bets")
    args = ap.parse_args()

    params = M.load_params()
    if not params:
        sys.exit("No model_params.json — run model.py --fit first.")
    bankroll = args.bankroll if args.bankroll is not None else E.load_bankroll()

    groups = parse_raw(RAW)
    print(f"Parsed {len(groups)} Round-1 3-balls.\n")

    # Collect every unique player and rate them (single set of ratings/sigmas)
    names = sorted({nm for g in groups for nm, _ in g["players"]})
    rated = M.predict_field(names, params, course=args.course, is_major=args.major)
    rating = {p.name: p.rating for p in rated}
    sigma = {p.name: p.sigma for p in rated}
    resolved = {nm: M.resolve_name(nm, params) for nm in names}

    # One shared simulated round for the whole set: score ~ Normal(-rating, sigma)
    rng = np.random.default_rng(7)
    order = list(names)
    mu = np.array([-rating[n] for n in order])
    sd = np.array([sigma[n] for n in order])
    draws = rng.normal(mu[:, None], sd[:, None], size=(len(order), args.sims))
    row_of = {n: i for i, n in enumerate(order)}

    rows = []
    for g in groups:
        trio = g["players"]
        idx = [row_of[nm] for nm, _ in trio]
        sub = draws[idx]                     # (3, sims) round scores
        winner = np.argmin(sub, axis=0)      # lowest score wins the round
        p_model = [float(np.mean(winner == k)) for k in range(3)]
        books = [o for _, o in trio]
        fair = MK.devig(books, method="multiplicative")  # de-vigged market prob
        for k, (nm, odds) in enumerate(trio):
            pm = p_model[k]
            ev = pm * odds - 1.0
            kf = max(0.0, E.kelly_fraction(pm, odds) * args.kelly)
            canon = resolved[nm]
            nr = params.get("players", {}).get(canon, {}).get("n_rounds", 0) if canon else 0
            thin = nr < args.min_rounds
            rows.append({
                "group": g["group"], "player": nm,
                "resolved": canon or "(default skill)",
                "n_rounds": nr,
                "odds": round(odds, 2),
                "p_model": round(pm, 4), "p_market": round(fair[k], 4),
                "ev_pct": round(ev * 100, 1),
                "kelly_stake": round(kf * bankroll, 2),
                "thin_sample": thin,
                "_ev": ev,
            })

    rows.sort(key=lambda r: -r["_ev"])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[c for c in rows[0] if c != "_ev"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "_ev"})

    flag = args.min_edge
    picks = [r for r in rows if r["ev_pct"] >= flag and r["kelly_stake"] >= 0.50
             and not r["thin_sample"]]
    print(f"{'EV%':>6}  {'odds':>5}  {'model':>6} {'mkt':>6}  {'£stk':>6}  {'n':>4}  player")
    print("-" * 70)
    for r in rows[:20]:
        if r in picks:
            star = " *"
        elif r["thin_sample"] and r["ev_pct"] >= flag and r["kelly_stake"] >= 0.50:
            star = " ~"   # would qualify but thin sample → excluded
        else:
            star = "  "
        print(f"{r['ev_pct']:>6.1f}  {r['odds']:>5.2f}  {r['p_model']*100:>5.1f}% "
              f"{r['p_market']*100:>5.1f}%  {r['kelly_stake']:>6.2f}  {r['n_rounds']:>4}{star}  {r['player']}")
    tot = sum(r["kelly_stake"] for r in picks)
    print(f"\n{len(picks)} recommended (* ) at +{flag:.0f}% EV, ≥£0.50 stake, "
          f"≥{args.min_rounds} rounds · total £{tot:.2f} of £{bankroll:.0f} bankroll")
    print("~ = positive edge but excluded (thin sample, unreliable rating)")
    print(f"Full card → {OUT}")


if __name__ == "__main__":
    main()
