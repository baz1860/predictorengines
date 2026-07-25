"""
golf/simulate_inplay.py  –  In-tournament (in-play) simulation.

Reads current scores through any number of completed rounds, then simulates
the remaining rounds to produce updated win/top-N probabilities.

Usage:
  python simulate_inplay.py --scores data/scores_r2.csv --rounds-done 2
                             [--sims 50000] [--course COURSE] [--major]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .model import (
    Player,
    compute_ratings,
    load_course_history,
    load_field,
    load_players,
    load_recent_form,
    DEFAULT_SIGMA,
)

DATA_DIR = Path(__file__).parent / "data"
TOTAL_ROUNDS = 4


def load_scores(path: Path) -> dict[str, float]:
    """
    Load current in-tournament scores.
    Returns dict: player_name_lower → cumulative score vs par.
    """
    scores: dict[str, float] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            made_cut = row.get("made_cut", "1").strip()
            score_col = row.get("score_36") or row.get("score_48") or row.get("score_72") or row.get("score", "0")
            if name and made_cut not in ("0", "false", "False", "no"):
                try:
                    scores[name.lower()] = float(score_col)
                except (ValueError, TypeError):
                    pass
    return scores


def simulate_inplay(
    players: list[Player],
    current_scores: dict[str, float],
    rounds_done: int,
    n_sims: int = 50_000,
    rng: np.random.Generator | None = None,
    matchups: list[tuple[str, str]] | None = None,
    threeballs: list[tuple[str, str, str]] | None = None,
    cut_rule: int = 65,
    no_cut: bool = False,
    total_rounds: int = TOTAL_ROUNDS,
) -> dict[str, dict]:
    """
    Simulate the remaining rounds starting from current scores.

    Players NOT in current_scores are assumed to have missed the cut.

    Args:
        players:        Rated player list (from model.compute_ratings)
        current_scores: name_lower → cumulative score through rounds_done
        rounds_done:    Rounds already completed (1, 2, or 3)
        n_sims:         Monte Carlo iterations
        matchups:       Optional tournament-long head-to-heads to settle from the
                        SAME simulated finishes (lower 72-hole total wins). Pairs
                        naming a non-survivor are skipped.
        threeballs:     Optional tournament-long 3-balls, settled likewise.

    When matchups/threeballs are given, the result carries the reserved keys
    ``__matchups__`` {(a,b): {a, b, tie}} and ``__threeballs__``
    {(a,b,c): {a, b, c, tie}}, matching simulate.simulate_tournament's contract.
    """
    from . import simulate as pre

    rng = rng or np.random.default_rng()
    rounds_left = total_rounds - rounds_done

    if rounds_left <= 0:
        raise ValueError("Tournament is already complete")

    # Filter to survivors only
    survivors = [p for p in players if p.name.lower() in current_scores]
    if not survivors:
        raise ValueError("No players matched between field and scores file. Check names.")

    n = len(survivors)
    names        = [p.name for p in survivors]
    ratings      = np.array([p.rating for p in survivors])
    sigmas       = np.array([p.sigma  for p in survivors])
    base_scores  = np.array([current_scores[p.name.lower()] for p in survivors])

    means = -ratings
    rc, tdf, blowup_mix = pre.load_sim_config()
    shifts = pre._weather_score_shifts(survivors)
    birdie_rates = np.array([
        getattr(player, "birdie_rate", 0.18) for player in survivors
    ])
    bogey_rates = np.array([
        getattr(player, "bogey_rate", 0.14) for player in survivors
    ])
    blowup_rates = np.array([
        getattr(player, "blowup_rate", 0.02) for player in survivors
    ])
    drawn = pre._draw_scores(rng, means, sigmas, n_sims, n, rc, tdf,
                             score_shifts=shifts, birdie_rates=birdie_rates,
                             bogey_rates=bogey_rates, blowup_rates=blowup_rates,
                             blowup_mix=blowup_mix)
    future_scores = drawn[:, :, :rounds_left]
    totals = base_scores[np.newaxis, :] + future_scores.sum(axis=2)

    # After R1 the 36-hole cut is still uncertain. Apply it inside every draw;
    # after R2 the snapshot already contains survivors only.
    cut_binds = (not no_cut) and rounds_done < 2 and cut_rule < n
    survived = np.ones((n_sims, n), dtype=bool)
    if cut_binds:
        r36 = base_scores[np.newaxis, :] + future_scores[:, :, 0]
        cut_line = np.partition(r36, cut_rule - 1, axis=1)[:, cut_rule - 1]
        survived = r36 <= cut_line[:, np.newaxis]
        totals = np.where(survived, totals, np.inf)
    settlement_totals = np.where(
        np.isinf(totals), 1e6 + (r36 if cut_binds else 0.0), totals)

    win_totals = totals

    # Tournament-long matchups / 3-balls off the SAME simulated finals. Only
    # survivor-vs-survivor groups are settled here; a group naming a cut player
    # is dropped (that bet is already decided and must not be sim-priced).
    idx_of = {nm.lower(): i for i, nm in enumerate(names)}
    mu_idx = [(a, b, idx_of[a.lower()], idx_of[b.lower()])
              for a, b in (matchups or [])
              if a.lower() in idx_of and b.lower() in idx_of]
    tb_idx = [(a, b, c, idx_of[a.lower()], idx_of[b.lower()], idx_of[c.lower()])
              for a, b, c in (threeballs or [])
              if a.lower() in idx_of and b.lower() in idx_of and c.lower() in idx_of]

    mres: dict = {}
    for a, b, ia, ib in mu_idx:
        ta, tb = settlement_totals[:, ia], settlement_totals[:, ib]
        a_w = int(np.count_nonzero(ta < tb))
        b_w = int(np.count_nonzero(tb < ta))
        mres[(a, b)] = {a: a_w / n_sims, b: b_w / n_sims,
                        "tie": (n_sims - a_w - b_w) / n_sims}

    tres: dict = {}
    for a, b, c, ia, ib, ic in tb_idx:
        ta, tb, tc = (settlement_totals[:, ia], settlement_totals[:, ib],
                      settlement_totals[:, ic])
        mn = np.minimum(np.minimum(ta, tb), tc)
        a_best, b_best, c_best = (ta == mn), (tb == mn), (tc == mn)
        tie_size = a_best.astype(np.int8) + b_best + c_best
        a_w = float(np.sum(a_best / tie_size))
        b_w = float(np.sum(b_best / tie_size))
        c_w = float(np.sum(c_best / tie_size))
        tres[(a, b, c)] = {a: a_w / n_sims, b: b_w / n_sims, c: c_w / n_sims,
                           "tie": float(np.mean(tie_size > 1))}

    # Rank every simulation at once. Cut players (inf) rank behind survivors.
    order = np.argsort(totals, axis=1, kind="stable")          # (n_sims, n)
    ranks = np.empty_like(order)
    rows = np.arange(n_sims)[:, np.newaxis]
    ranks[rows, order] = np.arange(1, n + 1)[np.newaxis, :]     # 1 = best score
    win_best = win_totals == np.min(win_totals, axis=1, keepdims=True)
    wins = (win_best / win_best.sum(axis=1, keepdims=True)).sum(axis=0)
    top5s   = np.count_nonzero(ranks <= 5,  axis=0)
    top10s  = np.count_nonzero(ranks <= 10, axis=0)
    top20s  = np.count_nonzero(ranks <= 20, axis=0)
    fin_sum = ranks.sum(axis=0, dtype=np.float64)
    made_cuts = survived.mean(axis=0) if cut_binds else np.ones(n)

    results = {}
    for i, name in enumerate(names):
        results[name] = {
            "win":         wins[i]   / n_sims,
            "top5":        top5s[i]  / n_sims,
            "top10":       top10s[i] / n_sims,
            "top20":       top20s[i] / n_sims,
            "made_cut":    float(made_cuts[i]),
            "current_score": base_scores[i],
            "avg_finish":  fin_sum[i] / n_sims,
            "n_sims":      n_sims,
        }

    results["__cut_binds__"] = cut_binds
    if mres:
        results["__matchups__"] = mres
    if tres:
        results["__threeballs__"] = tres

    return results


def write_predictions_inplay(
    survivors: list[Player],
    results: dict[str, dict],
    rounds_done: int,
    path: Path | None = None,
) -> Path:
    path = path or DATA_DIR / "predictions_inplay.csv"
    cols = [
        "rank", "name", "score_thru",  "rating",
        "win_pct", "top5_pct", "top10_pct", "top20_pct", "avg_finish",
    ]
    ranked = sorted(survivors, key=lambda p: results[p.name]["win"], reverse=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rank, p in enumerate(ranked, 1):
            r = results[p.name]
            score = int(r["current_score"])
            score_str = f"{score:+d}" if score != 0 else "E"
            w.writerow({
                "rank":       rank,
                "name":       p.name,
                "score_thru": score_str,
                "rating":     f"{p.rating:+.3f}",
                "win_pct":    f"{r['win']*100:.2f}",
                "top5_pct":   f"{r['top5']*100:.1f}",
                "top10_pct":  f"{r['top10']*100:.1f}",
                "top20_pct":  f"{r['top20']*100:.1f}",
                "avg_finish": f"{r['avg_finish']:.1f}",
            })
    print(f"  → {path}")
    return path


def print_inplay(
    survivors: list[Player],
    results: dict[str, dict],
    top_n: int = 25,
) -> None:
    ranked = sorted(survivors, key=lambda p: results[p.name]["win"], reverse=True)
    print(
        f"\n{'#':<4} {'Player':<26} {'Thru':>5} {'Win%':>6} {'T5%':>5} "
        f"{'T10%':>6} {'T20%':>6} {'AvgFin':>7} {'Rating':>7}"
    )
    print("-" * 82)
    for i, p in enumerate(ranked[:top_n], 1):
        r = results[p.name]
        score = int(r["current_score"])
        score_str = f"{score:+d}" if score != 0 else "E"
        print(
            f"{i:<4} {p.name:<26} {score_str:>5} {r['win']*100:>5.2f}% "
            f"{r['top5']*100:>4.1f}% {r['top10']*100:>5.1f}% "
            f"{r['top20']*100:>5.1f}% {r['avg_finish']:>7.1f} {p.rating:>+7.3f}"
        )

    total_win = sum(results[p.name]["win"] for p in survivors) * 100
    print(f"\n  ∑win% = {total_win:.1f}%  (should be ~100%)")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="In-play golf tournament simulator")
    ap.add_argument("--scores", default="data/scores_r2.csv",
                    help="CSV file with current scores (default: data/scores_r2.csv)")
    ap.add_argument("--rounds-done", type=int, default=2,
                    help="Rounds already completed (default: 2)")
    ap.add_argument("--course", default="", help="Course name for history lookup")
    ap.add_argument("--major", action="store_true")
    ap.add_argument("--sims", type=int, default=50_000)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rounds_left = TOTAL_ROUNDS - args.rounds_done
    print(f"In-play simulator  —  {args.rounds_done} rounds done, {rounds_left} to go")
    print(f"Scores file: {args.scores}")
    print(f"Sims: {args.sims:,}")

    # Load data
    all_players = load_players()
    scores_path = Path(args.scores)
    if not scores_path.is_absolute():
        scores_path = Path(__file__).parent / scores_path

    if not scores_path.exists():
        print(f"Error: scores file not found at {scores_path}")
        sys.exit(1)

    current_scores = load_scores(scores_path)
    print(f"Survivors loaded: {len(current_scores)}")

    # Build field from survivors (use players.csv ratings, fall back to generic)
    survivors = []
    unmatched = []
    for name_lower, score in current_scores.items():
        if name_lower in all_players:
            survivors.append(all_players[name_lower])
        else:
            # Create a placeholder with average rating
            p = Player(name=name_lower.title(), sigma=DEFAULT_SIGMA)
            survivors.append(p)
            unmatched.append(name_lower)

    if unmatched:
        print(f"  No rating data for {len(unmatched)} players (using 0.0): {', '.join(unmatched[:5])}{'...' if len(unmatched)>5 else ''}")

    # Compute composite ratings
    ch = load_course_history(args.course) if args.course else {}
    rf = load_recent_form()
    survivors = compute_ratings(survivors, course=args.course, is_major=args.major,
                                course_history=ch, recent_form=rf)

    # Simulate
    rng = np.random.default_rng(args.seed)
    print(f"\nRunning {args.sims:,} simulations for remaining {rounds_left} round(s)...", flush=True)
    results = simulate_inplay(survivors, current_scores, args.rounds_done,
                               n_sims=args.sims, rng=rng)

    print_inplay(survivors, results, top_n=args.top)
    write_predictions_inplay(survivors, results, args.rounds_done)


if __name__ == "__main__":
    main()
