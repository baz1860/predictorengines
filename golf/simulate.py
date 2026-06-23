"""
golf/simulate.py  –  4-round Monte Carlo tournament simulation.

For each simulation:
  1. Draw R1 and R2 scores for all players.
  2. Apply cut: keep top 65 + ties (configurable).
  3. Draw R3 and R4 for survivors.
  4. Rank by 72-hole total; record finish position.

Output (data/predictions.csv):
  name, win%, top5%, top10%, top20%, cut%, avg_finish, rating, sigma

Usage:
  python -m golf.simulate [--course COURSE] [--major] [--sims 50000]
                     [--cut-rule 65] [--no-cut] [--top TOP_N]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .model import (
    Player,
    compute_ratings,
    load_course_history,
    load_field,
    load_players,
    load_recent_form,
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SIM_CONFIG_JSON = DATA_DIR / "sim_config.json"
# Validated scoring-shape defaults (see sim_config.json / golf README). Fall back
# to legacy independent-Normal rounds when no config is present.
_DEFAULT_ROUND_CORR = 0.0
_DEFAULT_TAIL_DF: float | None = None


def load_sim_config(path: Path | None = None) -> tuple[float, float | None, float, float | None]:
    """Champion scoring-shape params: (round_corr, tail_df, win_round_corr, win_tail_df).

    `round_corr`/`tail_df` shape the place/cut markets. `win_round_corr`/
    `win_tail_df` shape the *win* market via a separate draw (see
    simulate_tournament); when absent they mirror the place params, so the win
    market falls back to the single-regime behaviour. A lower win_round_corr
    counters the leaderboard dispersion that under-prices outright favourites.
    """
    path = path or SIM_CONFIG_JSON
    rc, tdf = _DEFAULT_ROUND_CORR, _DEFAULT_TAIL_DF
    win_rc, win_tdf = None, _USE_CONFIG  # sentinel: "mirror place params"
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            rc = float(raw.get("round_corr", rc))
            tv = raw.get("tail_df", tdf)
            tdf = float(tv) if tv is not None else None
            if "win_round_corr" in raw and raw["win_round_corr"] is not None:
                win_rc = float(raw["win_round_corr"])
            if "win_tail_df" in raw:
                wv = raw["win_tail_df"]
                win_tdf = float(wv) if wv is not None else None
        except Exception:
            pass
    if win_rc is None:
        win_rc = rc
    if win_tdf is _USE_CONFIG:
        win_tdf = tdf
    return rc, tdf, win_rc, win_tdf


_USE_CONFIG = object()   # sentinel: "fall back to load_sim_config()"


def _draw_scores(rng, means, sigmas, n_sims, n, round_corr, tail_df):
    """Draw (n_sims, n, 4) round scores under the round-correlation / t-tail
    scoring model. Pulled out so the win market can be re-drawn under its own
    regime; the rng draw order matches the historical inline code exactly."""
    rc = float(min(max(round_corr, 0.0), 0.95))
    if rc <= 0.0 and tail_df is None:
        # Legacy path: independent Gaussian rounds (bit-for-bit unchanged).
        return rng.normal(
            loc=means[np.newaxis, :, np.newaxis],
            scale=sigmas[np.newaxis, :, np.newaxis],
            size=(n_sims, n, 4),
        )
    persist_sd = sigmas * math.sqrt(rc)          # per-player week effect
    trans_sd = sigmas * math.sqrt(1.0 - rc)      # per-round component
    u = rng.standard_normal((n_sims, n)) * persist_sd[np.newaxis, :]
    if tail_df is not None and float(tail_df) > 2.0:
        df = float(tail_df)
        # variance-standardise the t so the marginal per-round spread stays
        # sigma (Var(t_df) = df/(df-2)); preserves single-round calibration.
        z = rng.standard_t(df, size=(n_sims, n, 4)) * math.sqrt((df - 2.0) / df)
    else:
        z = rng.standard_normal((n_sims, n, 4))
    return (means[np.newaxis, :, np.newaxis]
            + u[:, :, np.newaxis]
            + z * trans_sd[np.newaxis, :, np.newaxis])


def _win_frac(rng, means, sigmas, n_sims, n, round_corr, tail_df) -> np.ndarray:
    """Per-player win probability (fraction of sims with the low 72-hole total)
    under a given scoring regime. Ties count for each tied player, matching the
    dense-rank==1 convention of the full simulator. No cut is applied — the
    winner is always well inside it. Vectorised; skips the ranking loop."""
    scores = _draw_scores(rng, means, sigmas, n_sims, n, round_corr, tail_df)
    tot = scores.sum(axis=2)                              # (n_sims, n) 72-hole
    is_best = tot == tot.min(axis=1, keepdims=True)
    return is_best.sum(axis=0) / n_sims


def win_market_probs(players: list[Player], n_sims: int, round_corr: float,
                     tail_df: float | None,
                     rng: np.random.Generator | None = None) -> dict[str, float]:
    """Standalone win-market probabilities for a rated field under a scoring
    regime — used by the win-corr sweep in validate.py without paying for the
    full place/cut ranking loop."""
    rng = rng or np.random.default_rng()
    means = np.array([-p.rating for p in players])
    sigmas = np.array([p.sigma for p in players])
    frac = _win_frac(rng, means, sigmas, n_sims, len(players), round_corr, tail_df)
    return {p.name: float(frac[i]) for i, p in enumerate(players)}


def simulate_tournament(
    players: list[Player],
    n_sims: int = 50_000,
    cut_rule: int = 65,
    no_cut: bool = False,
    rng: np.random.Generator | None = None,
    matchups: list[tuple[str, str]] | None = None,
    threeballs: list[tuple[str, str, str]] | None = None,
    round_corr: float | None = None,
    tail_df=_USE_CONFIG,
    win_round_corr: float | None = None,
    win_tail_df=_USE_CONFIG,
) -> dict[str, dict]:
    """
    Monte Carlo simulation.

    Returns dict keyed by player name:
      {win, top5, top10, top20, made_cut, missed_cut, avg_finish, n_sims}
    All probabilities are fractions (0-1).

    Scoring model
    -------------
    Each player's round score is drawn around `-rating` with spread `sigma`.
    Two optional, validated knobs shape the *joint* distribution of a player's
    four rounds (a common per-round shock that hits every player equally cancels
    out of any rank-based market, so it is deliberately NOT modelled here):

      * `round_corr` ∈ [0, 1): correlation between a player's own rounds — a
        player who runs hot/cold does so for the week. We split each round into a
        per-player tournament effect u ~ N(0, sigma²·round_corr) (drawn once for
        all four rounds) and a per-round term with variance sigma²·(1−round_corr).
        The marginal per-round variance stays sigma² (so single-round / cut
        calibration is unchanged) while the 72-hole-total variance widens to
        4·sigma²·(1 + 3·round_corr). This disperses the leaderboard so longshots
        win at a realistic rate instead of favourites converging to the top as
        independent-round noise averages out. `round_corr = 0.0` reproduces the
        legacy independent-rounds behaviour exactly.
      * `tail_df`: if set, the per-round term is drawn from a Student-t with this
        many degrees of freedom (variance-standardised so the marginal spread is
        still sigma), giving the right-skew/blow-up rounds a Normal misses. None
        ⇒ Gaussian rounds (legacy behaviour).

    If `matchups` / `threeballs` are given, head-to-head and 3-ball probabilities
    are computed from the SAME simulated finishes (so they are internally
    consistent with the outright/place numbers) and returned under the reserved
    keys "__matchups__" {(a,b): {a, b, tie}} and "__threeballs__"
    {(a,b,c): {a, b, c, tie}}. Players who miss the cut are ranked behind all
    survivors, ordered by their 36-hole score (standard matchup settlement).
    """
    rng = rng or np.random.default_rng()
    # Resolve scoring-shape params: None / sentinel ⇒ validated config default;
    # explicit values (incl. round_corr=0.0, tail_df=None) force that behaviour.
    if (round_corr is None or tail_df is _USE_CONFIG
            or win_round_corr is None or win_tail_df is _USE_CONFIG):
        cfg_rc, cfg_tdf, cfg_win_rc, cfg_win_tdf = load_sim_config()
        if round_corr is None:
            round_corr = cfg_rc
        if tail_df is _USE_CONFIG:
            tail_df = cfg_tdf
        if win_round_corr is None:
            win_round_corr = cfg_win_rc
        if win_tail_df is _USE_CONFIG:
            win_tail_df = cfg_win_tdf
    # The win market gets its own draw only when its regime actually differs.
    win_regime = (win_round_corr != round_corr) or (win_tail_df != tail_df)
    n = len(players)

    # Does the 36-hole cut actually bind? If the field is no larger than the cut
    # rule (limited-field / no-cut events, or an off-week stub field), every
    # player "survives" and make-cut/top-N collapse to ~1.0. That is correct for
    # a genuine no-cut event but must NOT be priced as a betting market. We flag
    # it here so callers (edge.price_all) can suppress the degenerate markets.
    cut_binds = (not no_cut) and (cut_rule < n)

    # Player ratings and sigmas as arrays (aligned)
    names   = [p.name for p in players]
    idx_of  = {nm: i for i, nm in enumerate(names)}
    ratings = np.array([p.rating for p in players])   # expected SG vs field
    sigmas  = np.array([p.sigma  for p in players])

    # Resolve requested pairings to index tuples (skip any unknown name)
    mu_idx = [(idx_of[a], idx_of[b]) for a, b in (matchups or [])
              if a in idx_of and b in idx_of]
    tb_idx = [(idx_of[a], idx_of[b], idx_of[c]) for a, b, c in (threeballs or [])
              if a in idx_of and b in idx_of and c in idx_of]
    mu_counts = np.zeros((len(mu_idx), 3), dtype=np.int64)   # a_better, b_better, tie
    tb_counts = np.zeros((len(tb_idx), 4), dtype=np.int64)   # a_best, b_best, c_best, tie

    # Expected score per round = -rating (lower = better)
    # Scores are relative to field average (0 = average field score)
    means = -ratings  # shape (n,)

    # Accumulators
    wins      = np.zeros(n, dtype=np.int64)
    top5s     = np.zeros(n, dtype=np.int64)
    top10s    = np.zeros(n, dtype=np.int64)
    top20s    = np.zeros(n, dtype=np.int64)
    made_cuts = np.zeros(n, dtype=np.int64)
    fin_sum   = np.zeros(n, dtype=np.float64)
    fin_count = np.zeros(n, dtype=np.int64)

    # Draw all rounds in bulk for speed.  Shape: (n_sims, n_players, 4_rounds).
    # Place/cut markets use this primary draw (validated round_corr/tail_df).
    scores_all = _draw_scores(rng, means, sigmas, n_sims, n, round_corr, tail_df)

    # Market-aware win shape: redraw the field under the win regime and price the
    # win market off *its* leaderboard. Place/cut markets stay on the primary
    # draw, so they are invariant to win_round_corr. The win regime usually runs
    # a lower round_corr — less hot/cold-week dispersion — so dominant favourites
    # are not washed out (the place-tuned rc systematically under-prices them).
    # Drawn after the primary draw so primary rng consumption is unchanged.
    win_frac = None
    if win_regime:
        win_frac = _win_frac(rng, means, sigmas, n_sims, n,
                             win_round_corr, win_tail_df)

    # R1+R2 totals
    r36 = scores_all[:, :, 0] + scores_all[:, :, 1]  # (n_sims, n)

    for sim in range(n_sims):
        r36_sim = r36[sim]  # (n,)

        if no_cut:
            survivors = np.arange(n)
        else:
            # Sort by 36-hole score
            sorted_idx = np.argsort(r36_sim)
            # Top cut_rule positions + all ties at the cut line
            if cut_rule >= n:
                survivors = sorted_idx
            else:
                cut_score = r36_sim[sorted_idx[cut_rule - 1]]
                survivors = np.where(r36_sim <= cut_score)[0]

        made_cuts[survivors] += 1

        # 72-hole totals for survivors only
        r72 = np.full(n, np.inf)
        r72[survivors] = (
            r36_sim[survivors]
            + scores_all[sim, survivors, 2]
            + scores_all[sim, survivors, 3]
        )

        # Matchup / 3-ball settlement: survivors by 72-hole total, missed-cut
        # players ranked behind everyone, ordered by their 36-hole score.
        if mu_idx or tb_idx:
            rank_score = np.where(np.isinf(r72), 1e6 + r36_sim, r72)
            for k, (ia, ib) in enumerate(mu_idx):
                sa, sb = rank_score[ia], rank_score[ib]
                mu_counts[k, 0 if sa < sb else 1 if sa > sb else 2] += 1
            for k, (ia, ib, ic) in enumerate(tb_idx):
                s = (rank_score[ia], rank_score[ib], rank_score[ic])
                mn = min(s)
                if s.count(mn) > 1:
                    tb_counts[k, 3] += 1
                else:
                    tb_counts[k, s.index(mn)] += 1

        # Rank (lower = better)
        order = np.argsort(r72)
        # Assign positions handling ties (dense ranking)
        positions = np.full(n, n + 1)  # default: missed cut
        prev_score = None
        prev_pos = 0
        tied_count = 0
        for rank, idx in enumerate(order):
            if r72[idx] == np.inf:
                break
            score = r72[idx]
            if score != prev_score:
                prev_pos = rank + 1
                tied_count = 1
                prev_score = score
            else:
                tied_count += 1
            positions[idx] = prev_pos

        # Accumulate
        for i, pos in enumerate(positions):
            if pos <= n:  # survived
                fin_sum[i]   += pos
                fin_count[i] += 1
                if pos == 1:   wins[i]   += 1
                if pos <= 5:   top5s[i]  += 1
                if pos <= 10:  top10s[i] += 1
                if pos <= 20:  top20s[i] += 1

    # Build results dict
    results = {"__cut_binds__": cut_binds}
    for i, name in enumerate(names):
        top5_frac = top5s[i] / n_sims
        # Win from the win-regime draw when active; clamp to top5 so the
        # cross-draw numbers stay coherent (win ⊆ top5).
        win_val = win_frac[i] if win_frac is not None else wins[i] / n_sims
        win_val = min(win_val, top5_frac)
        results[name] = {
            "win":       win_val,
            "top5":      top5_frac,
            "top10":     top10s[i]    / n_sims,
            "top20":     top20s[i]    / n_sims,
            "made_cut":  made_cuts[i] / n_sims,
            "missed_cut":1.0 - made_cuts[i] / n_sims,
            "avg_finish":fin_sum[i] / fin_count[i] if fin_count[i] > 0 else 0.0,
            "n_sims":    n_sims,
        }

    if mu_idx:
        mres = {}
        for k, (ia, ib) in enumerate(mu_idx):
            a_w, b_w, tie = mu_counts[k]
            mres[(names[ia], names[ib])] = {
                names[ia]: a_w / n_sims, names[ib]: b_w / n_sims,
                "tie": tie / n_sims}
        results["__matchups__"] = mres
    if tb_idx:
        tres = {}
        for k, (ia, ib, ic) in enumerate(tb_idx):
            a_w, b_w, c_w, tie = tb_counts[k]
            tres[(names[ia], names[ib], names[ic])] = {
                names[ia]: a_w / n_sims, names[ib]: b_w / n_sims,
                names[ic]: c_w / n_sims, "tie": tie / n_sims}
        results["__threeballs__"] = tres

    return results


def write_predictions(
    players: list[Player],
    results: dict[str, dict],
    path: Path | None = None,
) -> Path:
    path = path or DATA_DIR / "predictions.csv"
    cols = [
        "rank", "name", "rating", "sigma", "owgr",
        "win_pct", "top5_pct", "top10_pct", "top20_pct",
        "cut_pct", "avg_finish",
    ]

    # Sort by win probability
    ranked = sorted(players, key=lambda p: results[p.name]["win"], reverse=True)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rank, p in enumerate(ranked, 1):
            r = results[p.name]
            w.writerow({
                "rank":       rank,
                "name":       p.name,
                "rating":     f"{p.rating:+.3f}",
                "sigma":      f"{p.sigma:.2f}",
                "owgr":       p.owgr,
                "win_pct":    f"{r['win']*100:.2f}",
                "top5_pct":   f"{r['top5']*100:.1f}",
                "top10_pct":  f"{r['top10']*100:.1f}",
                "top20_pct":  f"{r['top20']*100:.1f}",
                "cut_pct":    f"{r['made_cut']*100:.1f}",
                "avg_finish": f"{r['avg_finish']:.1f}",
            })

    print(f"  Predictions → {path}")
    return path


def print_predictions(
    players: list[Player],
    results: dict[str, dict],
    top_n: int = 20,
) -> None:
    ranked = sorted(players, key=lambda p: results[p.name]["win"], reverse=True)
    print(
        f"\n{'#':<4} {'Player':<28} {'Win%':>6} {'T5%':>5} {'T10%':>6} "
        f"{'T20%':>6} {'Cut%':>6} {'AvgFin':>7} {'Rating':>7}"
    )
    print("-" * 82)
    for i, p in enumerate(ranked[:top_n], 1):
        r = results[p.name]
        print(
            f"{i:<4} {p.name:<28} {r['win']*100:>5.2f}% "
            f"{r['top5']*100:>4.1f}% {r['top10']*100:>5.1f}% "
            f"{r['top20']*100:>5.1f}% {r['made_cut']*100:>5.1f}% "
            f"{r['avg_finish']:>7.1f} {p.rating:>+7.3f}"
        )

    # Sanity check: win% should sum to ~100%
    total_win = sum(results[p.name]["win"] for p in players) * 100
    print(f"\n  ∑win% = {total_win:.1f}%  (should be ~100%)")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Simulate PGA Tour tournament")
    ap.add_argument("--course", default="", help="Course name for course fit lookup")
    ap.add_argument("--major", action="store_true", help="Apply major sigma bump (+0.15)")
    ap.add_argument("--sims", type=int, default=50_000, help="Number of Monte Carlo simulations")
    ap.add_argument("--cut-rule", type=int, default=65, help="Cut to top N after 36 holes")
    ap.add_argument("--no-cut", action="store_true", help="No cut (invitational format)")
    ap.add_argument("--top", type=int, default=20, help="Rows to display")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    args = ap.parse_args()

    print(f"Golf simulator  —  {args.sims:,} simulations")
    print(f"Cut rule: top {args.cut_rule}" if not args.no_cut else "No cut")

    # Load data
    all_players = load_players()
    try:
        field = load_field(players=all_players)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    if not field:
        print("Empty field. Check data/field.csv.")
        sys.exit(1)

    print(f"Field: {len(field)} players")

    # Rate the field: fitted model (model_params.json) if available, else the
    # legacy players.csv composite.
    from . import model as M
    params = M.load_params()
    if params:
        print("Ratings: fitted model (model_params.json)")
        field = M.predict_field([p.name for p in field], params,
                                course=args.course, is_major=args.major)
    else:
        print("Ratings: legacy players.csv composite")
        field = compute_ratings(
            field, course=args.course, is_major=args.major,
            course_history=load_course_history(args.course) if args.course else {},
            recent_form=load_recent_form(),
        )

    # Simulate
    rng = np.random.default_rng(args.seed)
    print(f"\nRunning {args.sims:,} simulations...", flush=True)
    results = simulate_tournament(
        field,
        n_sims=args.sims,
        cut_rule=args.cut_rule,
        no_cut=args.no_cut,
        rng=rng,
    )

    # Output
    print_predictions(field, results, top_n=args.top)
    write_predictions(field, results)


if __name__ == "__main__":
    main()
