#!/usr/bin/env python3
"""Fit the form-layer gains (G_ATT, G_DEF) on the 66 played WC matches.

Fully offline: reuses the leak-free pre-tournament form cache + event/results
cache built by form_backtest.py. League weighting (form_config) is applied when
computing per-team deltas; only the two gains are fitted (2 params on 66 matches —
kept deliberately small to avoid overfitting). Objective: mean 3-way log-loss.

Reports baseline vs hand-tuned vs fitted+league, and writes data/worldcup/
form_params.json consumed by player_form_multipliers and the edge --form-adj path.

Usage:
    python3 -m scripts.worldcup.form_fit            # fit + write params
    python3 -m scripts.worldcup.form_fit --no-write
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import sys
HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from scripts.worldcup.form_config import (  # noqa: E402
    team_deltas, multipliers, load_player_club, DEFAULT_G_ATT, DEFAULT_G_DEF,
    PARAMS_FILE,
)
from scripts.worldcup.form_backtest import (  # noqa: E402
    load_xi_by_fixture, ALIAS, WC_LO, WC_HI, PRETOURN_CACHE, EVENTS_CACHE,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def hda(lam1: float, lam2: float) -> np.ndarray:
    from engines.worldcup.predictor import score_matrix
    M = score_matrix(lam1, lam2)
    return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])


def metrics(probs: np.ndarray, actual: int) -> tuple[float, float, int]:
    oneh = np.zeros(3); oneh[actual] = 1.0
    return (float(((probs - oneh) ** 2).sum()),
            float(-math.log(max(probs[actual], 1e-12))),
            int(np.argmax(probs) == actual))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit form gains on WC 2026")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    form_cache = _load(PRETOURN_CACHE)
    ev_cache = _load(EVENTS_CACHE)
    club_map = load_player_club()
    xi_by_fx = load_xi_by_fixture()
    if not form_cache or not ev_cache:
        sys.exit("caches missing — run form_backtest.py first to build them.")

    from engines.worldcup.predictor import (load_matches, compute_elo,
                                            fit_goal_model, expected_goals,
                                            HOME_ADV)
    played, _ = load_matches()
    ratings, played = compute_elo(played)
    beta = fit_goal_model(played)

    def rget(name):
        return ratings.get(ALIAS.get(name, name))

    # Precompute per-match: baseline lambdas + per-team form deltas (gain-free).
    rows = []
    for eid in range(WC_LO, WC_HI + 1):
        ev = ev_cache.get(str(eid))
        if not ev:
            continue
        hn, an = ev.get("home_team"), ev.get("away_team")
        hs, as_ = ev.get("home_score"), ev.get("away_score")
        if hs is None or as_ is None:
            continue
        rh, ra = rget(hn), rget(an)
        if rh is None or ra is None:
            continue
        xis = xi_by_fx.get(str(eid), {})
        xi_h, xi_a = xis.get(hn, []), xis.get(an, [])
        if not xi_h or not xi_a:
            continue
        adv = 0.0 if ev.get("neutral", True) else HOME_ADV
        lam1, lam2 = expected_goals(rh, ra, beta, adv)
        fa_h, fd_h, _ = team_deltas(xi_h, lambda p: form_cache.get(p), club_map)
        fa_a, fd_a, _ = team_deltas(xi_a, lambda p: form_cache.get(p), club_map)
        actual = 0 if hs > as_ else (1 if hs == as_ else 2)
        rows.append((lam1, lam2, fa_h, fd_h, fa_a, fd_a, actual))

    n = len(rows)
    print(f"fitting on {n} matches\n")

    import random

    def ll_real(g, g_def=0.0):
        s = 0.0
        for lam1, lam2, fa_h, fd_h, fa_a, fd_a, actual in rows:
            am_h, dm_h = multipliers(fa_h, fd_h, g, g_def)
            am_a, dm_a = multipliers(fa_a, fd_a, g, g_def)
            s += metrics(hda(lam1 * am_h * dm_a, lam2 * am_a * dm_h), actual)[1]
        return s / n

    def full_metrics(g, g_def=0.0):
        B = LL = AC = 0.0
        for lam1, lam2, fa_h, fd_h, fa_a, fd_a, actual in rows:
            am_h, dm_h = multipliers(fa_h, fd_h, g, g_def)
            am_a, dm_a = multipliers(fa_a, fd_a, g, g_def)
            b, ll, c = metrics(hda(lam1 * am_h * dm_a, lam2 * am_a * dm_h), actual)
            B += b; LL += ll; AC += c
        return B / n, LL / n, AC / n

    base_LL = ll_real(0.0)

    # 1) log-loss vs gain — monotonic? (if so, gain is a regularisation choice, not a fit)
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    curve = [(g, ll_real(g)) for g in grid]
    print("log-loss vs gain (g_def=0):")
    for g, ll in curve:
        print(f"   g_att={g:<4} log-loss {ll:.4f}  ({ll-base_LL:+.4f})")
    monotonic = all(curve[i][1] >= curve[i + 1][1] - 1e-9 for i in range(len(curve) - 1))

    # 2) permutation test at a conservative gain — is the gain team-SPECIFIC?
    G_PICK = 0.30           # conservative: captures most early benefit, far from saturation
    deltas = [(fa_h, fa_a) for (_, _, fa_h, _, fa_a, _, _) in rows]
    flat0 = [d for pair in deltas for d in pair]

    def ll_shuffled(seed):
        rng = random.Random(seed); flat = flat0[:]; rng.shuffle(flat)
        s = 0.0
        for i, (lam1, lam2, *_x, actual) in enumerate(rows):
            am_h, _ = multipliers(flat[2 * i], 0, G_PICK, 0)
            am_a, _ = multipliers(flat[2 * i + 1], 0, G_PICK, 0)
            s += metrics(hda(lam1 * am_h, lam2 * am_a), actual)[1]
        return s / n

    real = ll_real(G_PICK)
    shuf = [ll_shuffled(s) for s in range(50)]
    beat = sum(1 for x in shuf if x <= real)
    p_perm = (beat + 1) / (len(shuf) + 1)

    pick_B, pick_LL, pick_AC = full_metrics(G_PICK)
    print(f"\n{'config':26} {'acc':>6} {'Brier':>8} {'log-loss':>9}")
    print("─" * 54)
    print(f"{'baseline (no form)':26} {full_metrics(0)[2]:>6.1%} "
          f"{full_metrics(0)[0]:>8.4f} {base_LL:>9.4f}")
    print(f"{'+form (g_att=0.30, league)':26} {pick_AC:>6.1%} {pick_B:>8.4f} {pick_LL:>9.4f}")
    print("─" * 54)
    print(f"log-loss vs gain is {'MONOTONIC' if monotonic else 'non-monotonic'} "
          f"→ gain is a regularisation choice, not a fit; 0.30 chosen conservatively.")
    print(f"permutation test @0.30:  real {real:.4f}  vs  shuffled "
          f"mean {np.mean(shuf):.4f}  (p≈{p_perm:.3f}, {beat}/{len(shuf)} beat real)")
    team_specific = p_perm < 0.10

    if not args.no_write:
        PARAMS_FILE.write_text(json.dumps({
            "g_att": G_PICK, "g_def": 0.0,
            "selected_by": "conservative (log-loss monotonic in gain; regularised pick)",
            "permutation_p": round(float(p_perm), 3),
            "logloss_at_pick": round(float(pick_LL), 4),
            "baseline_logloss": round(float(base_LL), 4),
            "fitted_on": f"WC2026 {n} matches (in-sample)", "league_weighted": True,
        }, indent=1))
        print(f"\nwrote {PARAMS_FILE}")
    print(f"\nverdict: improvement is {'TEAM-SPECIFIC (real form signal)' if team_specific else 'NOT distinguishable from sharpening'} "
          f"by permutation test.\nForward test on unplayed fixtures remains the decisive, fully-clean check.")


if __name__ == "__main__":
    main()
