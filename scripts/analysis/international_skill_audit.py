#!/usr/bin/env python3
"""Reproducible skill audit behind plans/international_football_module_plan.md §2.

Answers ONE question: does the existing Elo+Poisson model add value beyond
progressively harder reference forecasts, and does that vary by competition?

Three baselines, in increasing order of difficulty:

  B0  unconditional  -- test-period base rates of H/D/A, one triple for everything.
                        Knows nothing. This is what R3 used, and it is too easy.
  B1  per-competition -- test-period base rates computed WITHIN each competition.
                        Knows the competition's home/draw/away mix.
  B2  elo-only        -- the same Elo ratings through a plain logistic, no goal
                        model, no Dixon-Coles. Isolates what the goal model adds
                        on top of the ratings the market also has.

B0 and B1 are outcome-informed (they use test-period rates) and therefore NOT
deployable forecasts; they are diagnostic references only. This is stated in the
plan. The commercially relevant baseline is de-vigged market probability, which
cannot be computed outside the World Cup until odds history exists.

Scope: `--universe fifa` restricts to fixtures where BOTH teams carry a
confederation mapping in engines/worldcup/confederation_adj.py, used as a proxy
for FIFA membership (the mapping covers 197 teams; the unmapped active tail is
Kernow / Padania / Isle of Wight / Sapmi and similar non-FIFA sides). It is a
proxy, not an effective-dated FIFA registry -- building that registry is a
Stage 1 deliverable.

Uncertainty: bootstrap resamples whole international windows (matches grouped
into blocks by calendar month), not individual matches, because the same teams
recur inside a window. Reports the confidence interval on the friendly-minus-
competitive DIFFERENCE directly, rather than inviting the reader to eyeball
overlapping intervals.

Usage:
  python3 scripts/analysis/international_skill_audit.py
  python3 scripts/analysis/international_skill_audit.py --universe all --split 2020-01-01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.worldcup import predictor as p  # noqa: E402

CONF_SRC = ROOT / "engines" / "worldcup" / "confederation_adj.py"
RESULTS = ROOT / "data" / "results.csv"
N_BOOT = 2000
SEED = 20260808


def confederation_map() -> dict[str, str]:
    src = CONF_SRC.read_text()
    return dict(re.findall(
        r'"([^"]+)":\s*"(UEFA|CONMEBOL|CAF|AFC|CONCACAF|OFC)"', src))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build(split: str, universe: str):
    played, _ = p.load_matches()
    ratings, played = p.compute_elo(played)
    train = played[played.date < split]
    test = played[played.date >= split].copy()
    if universe == "fifa":
        cm = confederation_map()
        keep = test.home_team.isin(cm) & test.away_team.isin(cm)
        test = test[keep].copy()

    beta = p.fit_goal_model(train)
    adv = np.where(test["neutral"], 0.0, p.HOME_ADV)

    probs, elo_only, y = [], [], []
    for eh, ea, a, hs, as_ in zip(test.elo_h, test.elo_a, adv,
                                  test.home_score, test.away_score):
        lam_h, lam_a = p.expected_goals(eh + a, ea, beta)
        M = p.score_matrix(lam_h, lam_a)
        w = np.tril(M, -1).sum(); d = np.trace(M); l = np.triu(M, 1).sum()
        s = w + d + l
        probs.append((w / s, d / s, l / s))
        # B2: Elo expectation split into 1X2 using the test-period draw rate
        e = 1.0 / (1.0 + 10 ** ((ea - (eh + a)) / 400.0))
        elo_only.append(e)
        y.append(0 if hs > as_ else (1 if hs == as_ else 2))

    test = test.assign(
        p_h=[x[0] for x in probs], p_d=[x[1] for x in probs],
        p_a=[x[2] for x in probs], elo_exp=elo_only, y=y,
        gap=(test.elo_h + adv - test.elo_a).abs(),
        block=test.date.dt.to_period("M").astype(str),
    )
    return test


def brier(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    Y = np.eye(3)[y]
    return ((P - Y) ** 2).sum(axis=1)


def baselines(df: pd.DataFrame):
    y = df.y.to_numpy()
    n = len(df)
    model = np.c_[df.p_h, df.p_d, df.p_a]

    b0 = np.tile(np.bincount(y, minlength=3) / n, (n, 1))

    b1 = np.zeros((n, 3))
    for comp, idx in df.groupby("tour_key").groups.items():
        pos = df.index.get_indexer(idx)
        b1[pos] = np.bincount(y[pos], minlength=3) / len(pos)

    draw = (y == 1).mean()
    e = df.elo_exp.to_numpy()
    b2 = np.c_[e * (1 - draw), np.full(n, draw), (1 - e) * (1 - draw)]
    b2 /= b2.sum(axis=1, keepdims=True)

    return {"model": brier(model, y), "B0": brier(b0, y),
            "B1": brier(b1, y), "B2": brier(b2, y)}


def block_boot(df, sc, fn, rng, n=N_BOOT):
    blocks = df.block.to_numpy()
    uniq = np.unique(blocks)
    idx_by_block = {b: np.where(blocks == b)[0] for b in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_block[b] for b in pick])
        out.append(fn(df.iloc[sel], {k: v[sel] for k, v in sc.items()}))
    return np.percentile(out, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2022-01-01")
    ap.add_argument("--universe", default="fifa", choices=["fifa", "all"])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    df = build(a.split, a.universe).reset_index(drop=True)
    df["tour_key"] = df.tournament
    sc = baselines(df)
    rng = np.random.default_rng(SEED)

    print(f"results.csv sha256[:16] = {sha(RESULTS)}")
    print(f"split={a.split}  universe={a.universe}  n={len(df)}  "
          f"bootstrap={N_BOOT} month-blocks  seed={SEED}")
    print(f"HOME_ADV={p.HOME_ADV}  DC_RHO={p.DC_RHO}\n")
    print("Skill = 1 - Brier(model)/Brier(baseline). Higher is better; 0 = no gain.\n")

    hdr = f"{'competition':<38}{'n':>5}{'gap':>6}{'Brier':>8}{'vs B0':>8}{'vs B1':>8}{'vs B2':>8}"
    print(hdr); print("-" * len(hdr))

    def skill(sub, s, key):
        return 1 - s["model"].mean() / s[key].mean()

    rows = []
    for t, g in df.groupby("tour_key"):
        if len(g) < 60:
            continue
        pos = g.index.to_numpy()
        s = {k: v[pos] for k, v in sc.items()}
        rows.append((t, len(g), g.gap.mean(), s["model"].mean(),
                     skill(g, s, "B0"), skill(g, s, "B1"), skill(g, s, "B2")))
    for t, n, gap, br, s0, s1, s2 in sorted(rows, key=lambda r: -r[5]):
        print(f"{t:<38}{n:>5}{gap:>6.0f}{br:>8.3f}{s0:>8.3f}{s1:>8.3f}{s2:>8.3f}")

    print()
    fr = (df.tour_key == "Friendly").to_numpy()
    for lbl, m in (("FRIENDLY", fr), ("COMPETITIVE", ~fr)):
        s = {k: v[m] for k, v in sc.items()}
        print(f"{lbl:<38}{m.sum():>5}{df.gap[m].mean():>6.0f}"
              f"{s['model'].mean():>8.3f}{skill(None,s,'B0'):>8.3f}"
              f"{skill(None,s,'B1'):>8.3f}{skill(None,s,'B2'):>8.3f}")

    # CI on the DIFFERENCE (friendly minus competitive), per baseline
    print("\n95% CI on friendly-minus-competitive skill difference")
    print("(negative = friendlies worse; interval spanning 0 = no detected difference)")
    for key in ("B0", "B1", "B2"):
        def diff(d, s):
            m = (d.tour_key == "Friendly").to_numpy()
            if m.sum() < 30 or (~m).sum() < 30:
                return np.nan
            sf = 1 - s["model"][m].mean() / s[key][m].mean()
            sc_ = 1 - s["model"][~m].mean() / s[key][~m].mean()
            return sf - sc_
        lo, hi = block_boot(df, sc, diff, rng)
        pt = diff(df, sc)
        print(f"  vs {key}:  {pt:+.3f}   [{lo:+.3f}, {hi:+.3f}]")

    print("\nNOTE: the split coincides with the repo's own validation window "
          "(validate.py START=2022-01-01), so this is NOT an untouched holdout. "
          "Treat as diagnostic, not as out-of-sample evidence.")

    if a.json:
        a.json.write_text(json.dumps(
            {"split": a.split, "universe": a.universe, "n": len(df),
             "results_sha": sha(RESULTS), "seed": SEED,
             "rows": [dict(zip(
                 ["competition", "n", "gap", "brier", "skill_b0", "skill_b1", "skill_b2"],
                 r)) for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
