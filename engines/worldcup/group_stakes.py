#!/usr/bin/env python3
"""Group-stage qualification stakes for WC2026 match predictions (--stakes flag).

For each upcoming group match, estimates how much each team's probability of
advancing to the Round of 32 changes depending on the result:

    swing_T = P(T advances | T wins this match) - P(T advances | T loses this match)

A team already qualified regardless of this result has swing ≈ 0 (dead rubber).
A team that must win to survive has swing ≈ 1.

The lambda multiplier applied in edge.py is:

    mult = exp(STAKES_COEF * (swing - 0.5))

so a full-stakes team (swing=1) gets exp(+STAKES_COEF/2) ≈ +7.8% on expected
goals (more intensive play), and a dead-rubber team (swing=0) gets
exp(-STAKES_COEF/2) ≈ -7.8% (rotated squad / reduced effort).
A team with swing=0.5 is unaffected. STAKES_COEF=0.15 by default.

This is opt-in via --stakes in edge.py. The coefficient is intentionally
conservative; it can be tuned via validate.py once historical data accumulates.

Usage (via edge.py --stakes):
    The map is computed automatically. To inspect it directly:

    from engines.worldcup.group_stakes import compute_stakes_map
    from engines.worldcup.simulate import MatchModel, load_group_matches
    from engines.worldcup.dixoncoles import build_sources
    import numpy as np

    sources, _ = build_sources("blend")
    model = MatchModel(sources)
    gm = load_group_matches()
    rng = np.random.default_rng(42)
    stakes = compute_stakes_map(model, gm, rng, n_sims=5000)
    for (h, a), (sh, sa) in sorted(stakes.items(), key=lambda x: -max(x[1])):
        print(f"  {h:25s} v {a:25s}  swing H={sh:.2f}  A={sa:.2f}")
"""
from collections import defaultdict

import numpy as np

from .simulate import GROUPS, rank_group

# Lambda-multiplier coefficient.
# exp(0.15 * 0.5) ≈ 1.078 → a must-win team gets ~+7.8% on expected goals.
# exp(0.15 * -0.5) ≈ 0.928 → a dead-rubber team gets ~-7.2% on expected goals.
# Increase to amplify the effect; decrease to dampen it.
STAKES_COEF = 0.15


def compute_stakes_map(model, group_matches, rng, n_sims=5000):
    """For each upcoming group match return (swing_home, swing_away).

    Runs n_sims full group-stage simulations (played matches are fixed; upcoming
    ones are sampled from the model). For each upcoming match, partitions the
    n_sims runs by whether the home team won/drew/lost and measures how much
    that changes each team's probability of advancing to the Round of 32.

    Parameters
    ----------
    model : simulate.MatchModel
        Blended Elo+Dixon-Coles scoreline model.
    group_matches : list[dict]
        Output of simulate.load_group_matches().
    rng : np.random.Generator
    n_sims : int
        Number of group-stage simulations. 5,000 gives ±1% sampling error.

    Returns
    -------
    dict[(home, away), (swing_home, swing_away)]
        swing values are in [0, 1].  Only covers upcoming (not-yet-played) matches.
    """
    upcoming_pairs = [(m["home"], m["away"]) for m in group_matches if not m["fixed"]]
    if not upcoming_pairs:
        return {}

    # sim_log[(home, away)] = list of (home_score, away_score, home_adv, away_adv)
    sim_log: dict[tuple, list] = {pair: [] for pair in upcoming_pairs}

    for _ in range(n_sims):
        by_group: dict[str, list] = defaultdict(list)
        match_scores: dict[tuple, tuple] = {}

        for m in group_matches:
            h, a = m["home"], m["away"]
            if m["fixed"]:
                hs, as_ = m["hs"], m["as"]
            else:
                h1 = 0.0 if m["neutral"] else 1.0
                hs, as_ = model.sample(h, a, h1, 0.0, rng)
                match_scores[(h, a)] = (hs, as_)
            by_group[m["group"]].append((h, a, hs, as_))

        # Rank all 12 groups and determine R32 qualifiers
        third_stats = []
        adv: set[str] = set()
        for g, teams in GROUPS.items():
            order, pts, gd, gf = rank_group(teams, by_group[g], rng)
            adv.add(order[0])
            adv.add(order[1])
            t3 = order[2]
            third_stats.append((pts[t3], gd[t3], gf[t3], rng.random(), t3))
        third_stats.sort(reverse=True)
        adv.update(t for _, _, _, _, t in third_stats[:8])

        for (h, a), (hs, as_) in match_scores.items():
            sim_log[(h, a)].append((hs, as_, h in adv, a in adv))

    stakes: dict[tuple, tuple] = {}
    for (home, away) in upcoming_pairs:
        records = sim_log[(home, away)]

        # Partition sims by result from home team's perspective
        hw = [(ha, aa) for hs, as_, ha, aa in records if hs > as_]   # home wins
        dr = [(ha, aa) for hs, as_, ha, aa in records if hs == as_]  # draw
        aw = [(ha, aa) for hs, as_, ha, aa in records if hs < as_]   # away wins

        def _rate(lst: list, idx: int) -> float:
            """Fraction of sims in `lst` where team[idx] advanced."""
            return sum(r[idx] for r in lst) / len(lst) if lst else 0.5

        # Home team: advances more when they win than when they lose
        ph_win  = _rate(hw, 0)
        ph_loss = _rate(aw, 0)

        # Away team: advances more when they win (hs < as_) than when they lose (hs > as_)
        pa_win  = _rate(aw, 1)
        pa_loss = _rate(hw, 1)

        stakes[(home, away)] = (
            float(np.clip(ph_win - ph_loss, 0.0, 1.0)),
            float(np.clip(pa_win - pa_loss, 0.0, 1.0)),
        )

    return stakes


def stakes_multipliers(
    swing_home: float,
    swing_away: float,
    coef: float = STAKES_COEF,
) -> tuple[float, float]:
    """Convert qualification swings to (mult_home, mult_away) lambda multipliers.

    mult > 1 → higher expected goals (team plays more intensively / must attack).
    mult < 1 → lower expected goals (dead rubber, squad rotation expected).
    Centred at swing=0.5: neither boosted nor penalised.

    Parameters
    ----------
    swing_home : float
        P(home advances | home wins) - P(home advances | home loses).
    swing_away : float
        P(away advances | away wins) - P(away advances | away loses).
    coef : float
        Strength of the effect. Default STAKES_COEF=0.15.

    Returns
    -------
    (mult_home, mult_away)
    """
    return (
        float(np.exp(coef * (swing_home - 0.5))),
        float(np.exp(coef * (swing_away - 0.5))),
    )
