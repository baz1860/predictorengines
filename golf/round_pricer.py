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
from .providers.odds_manual import (ManualOddsProvider, OddsQuote,
                                    THREEBALLS_CSV, board_event, norm_event,
                                    post_cut_round, threeballs_csv_path,
                                    threeballs_raw_path)

DATA_DIR = Path(__file__).parent / "data"
OUT_CSV = DATA_DIR / "round_edges.csv"

# Round-level group markets this pricer understands (twosomes / threesomes).
ROUND_GROUP_MARKETS = ("2ball", "3ball")


def _kelly_for_returns(returns: np.ndarray) -> float:
    """Log-optimal fraction for an empirical multi-outcome return distribution."""
    profit = np.asarray(returns, dtype=float) - 1.0
    if profit.size == 0 or float(profit.mean()) <= 0.0:
        return 0.0

    def derivative(fraction: float) -> float:
        return float(np.mean(profit / (1.0 + fraction * profit)))

    lo, hi = 0.0, 0.999999
    if derivative(hi) > 0:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if derivative(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _norm_name(name: str) -> str:
    """Case/spacing-insensitive key for matching board names to the field."""
    return M._fold_name(name)


def field_mismatch(quotes: list[OddsQuote], field_names: list[str],
                   params: dict | None = None) -> list[str]:
    """Return round-group players that are NOT in the current event field.

    A non-empty result almost always means the board is stale — e.g. last
    week's tournament was re-priced against this week's field. Callers should
    refuse to price rather than emit a confident card for the wrong event. An
    empty field_names disables the check (nothing to compare against).

    When `params` is given, board names are resolved through the model's name
    map (M.resolve_name) before the membership test, so a bookmaker spelling
    that differs from field.csv (accents, case, a known alias such as
    "Samuel Stevens" → "Sam Stevens") counts as a match instead of tripping the
    guard. The same resolver prices the board, so this keeps the two in step.
    """
    field = {_norm_name(n) for n in field_names if str(n).strip()}
    if not field:
        return []
    if params is not None:
        # Add the field's own canonical forms so a board name that resolves to
        # the same player still matches even if field.csv uses a variant.
        for n in field_names:
            canon = M.resolve_name(n, params)
            if canon:
                field.add(_norm_name(canon))
    board = {q.player_name for q in quotes if q.market in ROUND_GROUP_MARKETS}

    def _in_field(name: str) -> bool:
        if _norm_name(name) in field:
            return True
        if params is not None:
            canon = M.resolve_name(name, params)
            if canon and _norm_name(canon) in field:
                return True
        return False

    return sorted(n for n in board if not _in_field(n))


def price_round_groups(
    quotes: list[OddsQuote],
    params: dict,
    course: str = "",
    is_major: bool = False,
    sims: int = 200_000,
    bankroll: float = 100.0,
    kelly: float = E.DEFAULT_KELLY,
    min_rounds: int = 60,
    seed: int = 7,
) -> list[dict]:
    """Price single-round group markets (2-balls and 3-balls). The lowest-score
    Monte Carlo and dead-heat split are identical for any group size."""
    groups: dict[str, list[OddsQuote]] = {}
    for q in quotes:
        if q.market in ROUND_GROUP_MARKETS:
            groups.setdefault(q.group_id, []).append(q)

    round_no = next((q.round_no for qs in groups.values() for q in qs if q.round_no), 1)
    # Post-cut rounds (3 and 4) are played in twosomes. A 3-ball group tagged
    # for one of these rounds is an impossible market — the group market is
    # inferred from group size, so this means a stale round-1/2 threesome board
    # is being re-priced for a later round. Refuse rather than invent a group
    # no book offers.
    if post_cut_round(round_no):
        oversized = [gid for gid, qs in groups.items() if len(qs) >= 3]
        if oversized:
            raw_name = threeballs_raw_path(round_no).name
            raise ValueError(
                f"Round {int(round_no)} is played in 2-balls after the cut, but the "
                f"round-group board has {len(oversized)} three-ball group(s) — this is "
                f"a stale round-1/2 board. Paste this round's 2-ball tee groups into "
                f"golf/data/{raw_name} and rerun refresh before pricing.")

    names = sorted({q.player_name for qs in groups.values() for q in qs})
    if not names:
        return []
    field_by_norm = {}
    try:
        for p in M.load_field(players=M.load_players()):
            field_by_norm[_norm_name(p.name)] = p
            canon = M.resolve_name(p.name, params)
            if canon:
                field_by_norm[_norm_name(canon)] = p
    except FileNotFoundError:
        pass
    field_items = []
    for name in names:
        item = field_by_norm.get(_norm_name(name))
        if item is None:
            canon = M.resolve_name(name, params)
            item = field_by_norm.get(_norm_name(canon or ""))
        if item is None:
            item = M.Player(name=name)
        else:
            item = M.Player(
                name=name,
                dg_id=item.dg_id,
                owgr=item.owgr,
                country=item.country,
                tee_time_r1=item.tee_time_r1,
                tee_time_r2=item.tee_time_r2,
                start_hole_r1=item.start_hole_r1,
                start_hole_r2=item.start_hole_r2,
            )
        field_items.append(item)
    context = M.load_field_context()
    rated = M.predict_field(
        field_items,
        params,
        course=course or context.get("course", ""),
        course_par=int(context.get("course_par") or 0),
        course_yards=int(context.get("course_yards") or 0),
        par3_holes=int(context.get("par3_holes") or 0),
        par4_holes=int(context.get("par4_holes") or 0),
        par5_holes=int(context.get("par5_holes") or 0),
        is_major=is_major,
        round_no=int(round_no or 1),
    )
    rating = {
        p.name: p.rating + float((getattr(p, "weather_round_adj", {}) or {}).get(int(round_no or 1), 0.0))
        for p in rated
    }
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
            unique_win = is_best & (tie_count == 1)
            tied_best = is_best & (tie_count > 1)
            if (q.settlement_rule or "dead_heat") == "push_tie":
                # A tied 2-ball is void: return the original stake.
                returns = np.where(unique_win, q.decimal_odds,
                                   np.where(tied_best, 1.0, 0.0))
            else:
                # Dead heat: split the stake across the tied winners.
                returns = np.where(is_best, q.decimal_odds / tie_count, 0.0)
            expected_return = float(returns.mean())
            ev = expected_return - 1.0
            dead_heat_prob_equiv = expected_return / q.decimal_odds
            kf = max(0.0, _kelly_for_returns(returns) * kelly)
            thin = int(n_rounds.get(q.player_name, 0)) < min_rounds
            # Never stake a thin-sample player: with too little (or no) history the
            # rating is a default-skill guess, so a big "edge" against the book is
            # spurious — the book is pricing information the model can't see. Keep
            # the probabilities/EV for context, but stake nothing.
            stake = 0.0 if thin else round(kf * bankroll, 2)
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
                "kelly_stake": stake,
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
        from .io_utils import atomic_write_text
        atomic_write_text(path, "")
        return path
    cols = list(rows[0].keys())
    from .io_utils import atomic_write_csv
    atomic_write_csv(path, cols, rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Price round-specific group markets (2-balls / 3-balls)")
    ap.add_argument("--round", type=int, default=1, dest="round_no")
    ap.add_argument("--event-id", default="")
    ap.add_argument("--course", default="")
    ap.add_argument("--major", action="store_true")
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--bankroll", type=float, default=None)
    ap.add_argument("--kelly", type=float, default=E.DEFAULT_KELLY)
    ap.add_argument("--min-rounds", type=int, default=60)
    ap.add_argument("--min-edge", type=float, default=0.0)
    args = ap.parse_args()

    params = M.load_params()
    if not params:
        raise SystemExit("No model_params.json - run python -m golf.model --fit first.")
    bankroll = args.bankroll if args.bankroll is not None else E.load_bankroll()
    board_csv = threeballs_csv_path(args.round_no)
    raw_name = threeballs_raw_path(args.round_no).name
    quotes = ManualOddsProvider().load_threeballs(event_id=args.event_id, round_no=args.round_no)
    if not quotes:
        raise SystemExit(
            f"No round-group odds found for round {args.round_no}. Paste this "
            f"round's tee groups into golf/data/{raw_name} and run golf.refresh.")
    try:
        field_names = [p.name for p in M.load_field(players=M.load_players())]
    except FileNotFoundError:
        field_names = []
    current_event = M.load_field_event()
    # Same event-tag guard as engine.cmd_round_3balls: name overlap alone
    # cannot catch a stale board when consecutive events share players.
    if current_event:
        tag = board_event(board_csv)
        if norm_event(tag) != norm_event(current_event):
            raise SystemExit(
                f"Round-group board is from '{tag or 'an untagged capture'}' but the "
                f"current event is '{current_event}' — stale board. Re-paste this "
                f"event's tee groups into golf/data/{raw_name} and rerun "
                "golf.refresh.")
    missing = field_mismatch(quotes, field_names, params)
    if missing:
        raise SystemExit(
            f"Round-group board does not match the current field: {len(missing)} "
            f"player(s) are not in field.csv (stale board from another event?): "
            + ", ".join(missing)
            + f"\nRe-paste this event's tee groups into golf/data/{raw_name} "
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
