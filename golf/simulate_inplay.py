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
) -> dict[str, dict]:
    """
    Simulate the remaining rounds starting from current scores.

    Players NOT in current_scores are assumed to have missed the cut.

    Args:
        players:        Rated player list (from model.compute_ratings)
        current_scores: name_lower → cumulative score through rounds_done
        rounds_done:    Rounds already completed (1, 2, or 3)
        n_sims:         Monte Carlo iterations
    """
    rng = rng or np.random.default_rng()
    rounds_left = TOTAL_ROUNDS - rounds_done

    if rounds_left <= 0:
        raise ValueError("Tournament is already complete (rounds_done >= 4)")

    # Filter to survivors only
    survivors = [p for p in players if p.name.lower() in current_scores]
    if not survivors:
        raise ValueError("No players matched between field and scores file. Check names.")

    n = len(survivors)
    names        = [p.name for p in survivors]
    ratings      = np.array([p.rating for p in survivors])
    sigmas       = np.array([p.sigma  for p in survivors])
    base_scores  = np.array([current_scores[p.name.lower()] for p in survivors])

    # Expected score per round = -rating
    means = -ratings  # (n,)

    # Accumulators
    wins      = np.zeros(n, dtype=np.int64)
    top5s     = np.zeros(n, dtype=np.int64)
    top10s    = np.zeros(n, dtype=np.int64)
    top20s    = np.zeros(n, dtype=np.int64)
    fin_sum   = np.zeros(n, dtype=np.float64)

    # Draw all remaining rounds at once: (n_sims, n, rounds_left)
    future_scores = rng.normal(
        loc=means[np.newaxis, :, np.newaxis],
        scale=sigmas[np.newaxis, :, np.newaxis],
        size=(n_sims, n, rounds_left),
    )

    # Total = base + sum of future rounds
    future_totals = future_scores.sum(axis=2)                  # (n_sims, n)
    totals        = base_scores[np.newaxis, :] + future_totals # (n_sims, n)

    # Rank for each sim
    for sim in range(n_sims):
        t = totals[sim]
        order = np.argsort(t)

        prev_score = None
        prev_pos   = 0
        for rank, idx in enumerate(order):
            score = t[idx]
            if score != prev_score:
                prev_pos   = rank + 1
                prev_score = score

            pos = prev_pos
            fin_sum[idx] += pos
            if pos == 1:  wins[idx]  += 1
            if pos <= 5:  top5s[idx] += 1
            if pos <= 10: top10s[idx]+= 1
            if pos <= 20: top20s[idx]+= 1

    results = {}
    for i, name in enumerate(names):
        results[name] = {
            "win":         wins[i]   / n_sims,
            "top5":        top5s[i]  / n_sims,
            "top10":       top10s[i] / n_sims,
            "top20":       top20s[i] / n_sims,
            "current_score": base_scores[i],
            "avg_finish":  fin_sum[i] / n_sims,
            "n_sims":      n_sims,
        }

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
