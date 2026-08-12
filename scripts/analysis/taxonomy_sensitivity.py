#!/usr/bin/env python3
"""Sensitivity of ratings and prices to the competition-weight taxonomy (plan §4).

What plan §4 demanded, and revision 3 failed to deliver
-------------------------------------------------------
R3 tested ONE invented weight table, reported a median rating shift of 1.9 Elo
points, and called the risk low. That was not an auditable test:

  * one plausible table is not the space of plausible tables;
  * median is the wrong safety statistic — the tail is what moves a bet;
  * rating shifts are not the quantity anyone cares about. **Probability** shifts
    and **bet-selection** changes are.

This script fixes all three. It runs several defensible weight tables, reports
max / p95 / p99 shifts restricted to FIFA members, and translates each into the
1X2 probability change and the number of fixtures whose implied edge would cross a
betting threshold.

Profiles
--------
  legacy       the current 12-entry table (the incumbent)
  v1           international/taxonomy.py, the proposed challenger
  v1_flat      v1 but every non-friendly competitive match weighted equally --
               tests whether the fine-grained weights matter at all
  v1_aggressive v1 with continental qualifying pushed to World Cup qualifying's
               weight -- an upper bound on reasonable disagreement
  friendly_20  legacy, but friendlies alone corrected -- isolates the single
               change most likely to matter

Usage:
  python3 scripts/analysis/taxonomy_sensitivity.py
  python3 scripts/analysis/taxonomy_sensitivity.py --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.worldcup import predictor as p     # noqa: E402
from international import registry as R         # noqa: E402
from international import taxonomy as T         # noqa: E402

RESULTS = ROOT / "data" / "results.csv"
EDGE_THRESHOLD = 0.03      # 3% edge — the repo's own "treat smaller as noise" line


def profile_table(name: str) -> dict:
    """The K table for a named profile."""
    if name == "legacy":
        return dict(T.LEGACY_K)

    labels = pd.read_csv(RESULTS, usecols=["tournament"]).tournament.dropna().unique()
    table = {}
    for label in labels:
        comp = T.classify(label)
        if comp is None:
            continue
        k = comp.k
        if name == "v1":
            pass
        elif name == "v1_flat":
            if comp.category not in (T.FRIENDLY, T.MINOR):
                k = 45
        elif name == "v1_aggressive":
            if comp.category == T.CONTINENTAL_QUAL:
                k = 50
        elif name == "friendly_20":
            return {**T.LEGACY_K}          # handled below
        else:
            raise ValueError(f"unknown profile {name!r}")
        table[str(label)] = k
    return table


def ratings_for(table: dict, default_k: int, played: pd.DataFrame) -> dict:
    orig_table, orig_default = p.K_BY_TOURNAMENT, p.DEFAULT_K
    p.K_BY_TOURNAMENT, p.DEFAULT_K = table, default_k
    try:
        ratings, _ = p.compute_elo(played)
    finally:
        p.K_BY_TOURNAMENT, p.DEFAULT_K = orig_table, orig_default
    return ratings


def probs(elo_h: float, elo_a: float, beta, neutral: bool) -> tuple:
    adv = 0.0 if neutral else p.HOME_ADV
    lam_h, lam_a = p.expected_goals(elo_h + adv, elo_a, beta)
    M = p.score_matrix(lam_h, lam_a)
    w = float(np.tril(M, -1).sum()); d = float(np.trace(M)); l = float(np.triu(M, 1).sum())
    s = w + d + l
    return w / s, d / s, l / s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    print(f"results.csv sha256[:16] = "
          f"{hashlib.sha256(RESULTS.read_bytes()).hexdigest()[:16]}")

    played, _ = p.load_matches()
    base_ratings = ratings_for(dict(T.LEGACY_K), T.LEGACY_DEFAULT_K, played)
    _, played_elo = p.compute_elo(played)
    beta = p.fit_goal_model(played_elo)

    # In-scope population: FIFA members only. R3 measured across all 338 identities
    # while recommending a FIFA-only product, so its tail statistics were dominated
    # by Island Games sides the product will never price.
    fifa = {t for t in base_ratings
            if R.status(t) == R.FIFA and not R.load()[t].member_to}
    recent = played[played.date >= played.date.max() - pd.DateOffset(years=4)]
    active = (set(recent.home_team.dropna()) | set(recent.away_team.dropna())) & fifa
    print(f"population: {len(active)} active FIFA members "
          f"(R3 measured 257 teams including non-FIFA sides)\n")

    # Recent fixtures between in-scope teams, for the probability impact.
    sample = recent[recent.home_team.isin(active) & recent.away_team.isin(active)]
    sample = sample.tail(600)
    print(f"probability impact measured on the last {len(sample)} in-scope matches\n")

    profiles = ["v1", "v1_flat", "v1_aggressive", "friendly_20"]
    out = {}
    hdr = (f"{'profile':<16}{'maxΔelo':>9}{'p99':>8}{'p95':>8}{'median':>8}"
           f"{'maxΔp':>9}{'p95Δp':>8}{'flips':>7}")
    print(hdr); print("-" * len(hdr))

    for name in profiles:
        table = profile_table(name)
        default = T.LEGACY_DEFAULT_K if name in ("legacy", "friendly_20") else 30
        if name == "friendly_20":
            table = {**T.LEGACY_K, "Friendly": 12}   # halve friendly influence
        new = ratings_for(table, default, played)

        d = pd.Series({t: new.get(t, 0) - base_ratings.get(t, 0) for t in active})
        shifts = d.abs()

        dp, flips = [], 0
        for r in sample.itertuples(index=False):
            h, aw, neutral = r.home_team, r.away_team, bool(r.neutral)
            b = probs(base_ratings[h], base_ratings[aw], beta, neutral)
            n = probs(new.get(h, base_ratings[h]), new.get(aw, base_ratings[aw]),
                      beta, neutral)
            delta = max(abs(x - y) for x, y in zip(b, n))
            dp.append(delta)
            if delta >= EDGE_THRESHOLD:
                flips += 1
        dp = np.array(dp)

        out[name] = {
            "max_elo": float(shifts.max()), "p99_elo": float(shifts.quantile(.99)),
            "p95_elo": float(shifts.quantile(.95)),
            "median_elo": float(shifts.median()),
            "max_prob": float(dp.max()), "p95_prob": float(np.quantile(dp, .95)),
            "fixtures_crossing_threshold": int(flips),
            "fixtures_tested": int(len(dp)),
        }
        print(f"{name:<16}{shifts.max():>9.1f}{shifts.quantile(.99):>8.1f}"
              f"{shifts.quantile(.95):>8.1f}{shifts.median():>8.1f}"
              f"{dp.max():>9.3f}{np.quantile(dp,.95):>8.3f}{flips:>7d}")

    print(f"\nmaxΔp / p95Δp are the largest and 95th-percentile change in any of the "
          f"three 1X2 probabilities.\n'flips' counts fixtures where some outcome moved "
          f"by >= {EDGE_THRESHOLD:.0%} — i.e. enough to change a bet decision at the "
          f"repo's own noise floor.")

    worst = max(out.values(), key=lambda v: v["fixtures_crossing_threshold"])
    share = worst["fixtures_crossing_threshold"] / max(worst["fixtures_tested"], 1)
    print(f"\nVERDICT: across every profile tested, at most "
          f"{worst['fixtures_crossing_threshold']}/{worst['fixtures_tested']} "
          f"({share:.1%}) of recent in-scope fixtures move enough to change a bet.")

    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
