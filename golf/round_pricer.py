"""Round-specific golf market pricer.

This prices single-round 3-ball markets from manual/free odds. It is separate
from the 72-hole tournament simulator because bookmaker 3-balls usually settle
on the lowest score in one round only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from . import edge as E
from . import market as MK
from . import model as M
from .providers.odds_manual import ManualOddsProvider, OddsQuote

DATA_DIR = Path(__file__).parent / "data"
OUT_CSV = DATA_DIR / "round_edges.csv"

# Round-level group markets this pricer understands (twosomes / threesomes).
ROUND_GROUP_MARKETS = ("2ball", "3ball")


def _norm_name(name: str) -> str:
    """Case/spacing-insensitive key for matching board names to the field."""
    return " ".join(str(name).casefold().split())


def field_mismatch(quotes: list[OddsQuote], field_names: list[str]) -> list[str]:
    """Return 3-ball players that are NOT in the current event field.

    A non-empty result almost always means the 3-ball board is stale — e.g.
    last week's tournament was re-priced against this week's field. Callers
    should refuse to price rather than emit a confident card for the wrong
    event. An empty field_names disables the check (nothing to compare against).
    """
    field = {_norm_name(n) for n in field_names if str(n).strip()}
    if not field:
        return []
    board = {q.player_name for q in quotes if q.market in ROUND_GROUP_MARKETS}
    return sorted(n for n in board if _norm_name(n) not in field)


def price_round_groups(
    quotes: list[OddsQuote],
    params: dict,
    course: str = "",
    is_major: bool = False,
    sims: int = 200_000,
    bankroll: float = 100.0,
    kelly: float = 0.25,
    min_rounds: int = 60,
    seed: int = 7,
) -> list[dict]:
    """Price single-round group markets (2-balls and 3-balls). The lowest-score
    Monte Carlo and dead-heat split are identical for any group size."""
    groups: dict[str, list[OddsQuote]] = {}
    for q in quotes:
        if q.market in ROUND_GROUP_MARKETS:
            groups.setdefault(q.group_id, []).append(q)

    names = sorted({q.player_name for qs in groups.values() for q in qs})
    if not names:
        return []
    rated = M.predict_field(names, params, course=course, is_major=is_major)
    rating = {p.name: p.rating for p in rated}
    sigma = {p.name: p.sigma for p in rated}
    resolved = {name: M.resolve_name(name, params) for name in names}
    n_rounds = {
        name: params.get("players", {}).get(resolved[name] or "", {}).get("n_rounds", 0)
        for name in names
    }

    rng = np.random.default_rng(seed)
    order = list(names)
    mu = np.array([-rating[n] for n in order])
    sd = np.array([sigma[n] for n in order])
    # Round scores are integer outcomes. Rounding normal draws is crude but it
    # creates realistic non-zero tie probability, which continuous draws cannot.
    draws = np.rint(rng.normal(mu[:, None], sd[:, None], size=(len(order), sims)))
    row_of = {n: i for i, n in enumerate(order)}

    rows = []
    for group_id, qs in groups.items():
        if len(qs) not in (2, 3):
            continue
        members = [q.player_name for q in qs]
        idx = [row_of[n] for n in members]
        sub = draws[idx]
        mins = sub.min(axis=0)
        best = sub == mins
        tie_count = best.sum(axis=0)
        odds = [q.decimal_odds for q in qs]
        fair = MK.devig(odds, method="multiplicative")
        for k, q in enumerate(qs):
            is_best = best[k]
            p_best = float(is_best.mean())
            # Dead-heat expected return per unit stake: if tied, stake is split
            # across tied winners and each split is paid at the quoted odds.
            returns = np.where(is_best, q.decimal_odds / tie_count, 0.0)
            expected_return = float(returns.mean())
            ev = expected_return - 1.0
            dead_heat_prob_equiv = expected_return / q.decimal_odds
            kf = max(0.0, E.kelly_fraction(dead_heat_prob_equiv, q.decimal_odds) * kelly)
            thin = int(n_rounds.get(q.player_name, 0)) < min_rounds
            rows.append({
                "round": q.round_no or "",
                "market": q.market,
                "group_id": group_id,
                "player": q.player_name,
                "resolved": resolved[q.player_name] or "(public/default skill)",
                "n_rounds": int(n_rounds.get(q.player_name, 0)),
                "book": q.book,
                "odds": round(q.decimal_odds, 3),
                "p_best": round(p_best, 4),
                "p_dead_heat_equiv": round(dead_heat_prob_equiv, 4),
                "p_market": round(fair[k], 4),
                "ev_pct": round(ev * 100, 2),
                "kelly_stake": round(kf * bankroll, 2),
                "thin_sample": thin,
                "settlement_rule": q.settlement_rule or "dead_heat",
                "_ev": ev,
            })
    rows.sort(key=lambda r: -r["_ev"])
    for r in rows:
        r.pop("_ev", None)
    return rows


# Back-compat alias: callers/tests may still import the old name.
price_round_3balls = price_round_groups


def write_round_edges(rows: list[dict], path: Path | None = None) -> Path:
    path = path or OUT_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    cols = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Price round-specific group markets (2-balls / 3-balls)")
    ap.add_argument("--round", type=int, default=1, dest="round_no")
    ap.add_argument("--event-id", default="")
    ap.add_argument("--course", default="")
    ap.add_argument("--major", action="store_true")
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--bankroll", type=float, default=None)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--min-rounds", type=int, default=60)
    ap.add_argument("--min-edge", type=float, default=0.0)
    args = ap.parse_args()

    params = M.load_params()
    if not params:
        raise SystemExit("No model_params.json - run python -m golf.model --fit first.")
    bankroll = args.bankroll if args.bankroll is not None else E.load_bankroll()
    quotes = ManualOddsProvider().load_threeballs(event_id=args.event_id, round_no=args.round_no)
    if not quotes:
        raise SystemExit("No 3-ball odds found. Add golf/data/threeballs.csv or run golf.refresh on a raw paste.")
    try:
        field_names = [p.name for p in M.load_field(players=M.load_players())]
    except FileNotFoundError:
        field_names = []
    missing = field_mismatch(quotes, field_names)
    if missing:
        raise SystemExit(
            f"Round-group board does not match the current field: {len(missing)} "
            f"player(s) are not in field.csv (stale board from another event?): "
            + ", ".join(missing)
            + "\nRe-paste this event's tee groups into golf/data/threeballs_r1_raw.txt "
              "and rerun golf.refresh."
        )
    rows = price_round_groups(
        quotes,
        params,
        course=args.course,
        is_major=args.major,
        sims=args.sims,
        bankroll=bankroll,
        kelly=args.kelly,
        min_rounds=args.min_rounds,
    )
    out = write_round_edges(rows)
    picks = [r for r in rows if r["ev_pct"] >= args.min_edge and r["kelly_stake"] >= 0.5 and not r["thin_sample"]]
    print(f"Round {args.round_no} group pricing: {len(rows)} sides, {len(picks)} recommended")
    print(f"{'EV%':>7} {'Odds':>6} {'Model':>7} {'Mkt':>7} {'Stake':>7} Player")
    print("-" * 72)
    for r in rows[:25]:
        print(f"{r['ev_pct']:>7.1f} {r['odds']:>6.2f} {r['p_dead_heat_equiv']*100:>6.1f}% "
              f"{r['p_market']*100:>6.1f}% {r['kelly_stake']:>7.2f} {r['player']}")
    print(f"Full card -> {out}")


if __name__ == "__main__":
    main()
