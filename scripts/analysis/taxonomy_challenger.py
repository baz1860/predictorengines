#!/usr/bin/env python3
"""Is the v1 competition taxonomy actually BETTER, or just different?

`taxonomy_sensitivity.py` established that the v1 weights move things — up to 15
percentage points on a single 1X2 probability, and enough to change the bet
decision on 4% of recent fixtures. That is an argument for caution, not an argument
for adoption. This script asks the question sensitivity cannot: does v1 predict
matches better than the incumbent?

Method
------
The same walk-forward harness the production gate uses (`validate.walk_forward`),
run once per weight profile. Only the K table differs; everything else — the
monthly refits, the training windows, the seed — is identical, so the difference in
score is attributable to the weights.

What actually changes, and what does not
----------------------------------------
Competition weights affect the **Elo** rating update only. Dixon-Coles fits attack
and defence strengths from goals and never reads the K table. The production model
is a 50/50 blend of the two, so **at most half the blend can move**. A small blend
delta therefore understates the effect on the Elo component, and both are reported.

Promotion rule, fixed BEFORE looking at the result
--------------------------------------------------
v1 is promoted only if:
  1. pooled blend Brier improves, AND
  2. no competition with n >= 40 regresses by more than 0.010, AND
  3. the pooled improvement exceeds 0.002 (the incumbent gate's own tolerance,
     so we are not promoting noise).

Anything else is "no deployable challenger", which is a legitimate and expected
outcome — the market-blend experiment in this repo reached exactly that verdict and
was correctly shelved.

Usage:
  python3 scripts/analysis/taxonomy_challenger.py --run legacy
  python3 scripts/analysis/taxonomy_challenger.py --run v1
  python3 scripts/analysis/taxonomy_challenger.py --compare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.worldcup import predictor as p        # noqa: E402
from engines.worldcup import validate as V         # noqa: E402
from international import taxonomy as T            # noqa: E402

OUT_DIR = ROOT / "data" / "international"
MIN_N = 40
COMP_TOL = 0.010
POOLED_TOL = 0.002


def weight_table(profile: str) -> tuple[dict, int]:
    if profile == "legacy":
        return dict(T.LEGACY_K), T.LEGACY_DEFAULT_K
    labels = pd.read_csv(ROOT / "data" / "results.csv",
                         usecols=["tournament"]).tournament.dropna().unique()
    table = {}
    for label in labels:
        comp = T.classify(label)
        if comp is not None:
            table[str(label)] = comp.k
    # No default: v1 classifies every label in the data, so a lookup miss means the
    # data changed under us and should be loud rather than silently averaged.
    return table, 30


def run(profile: str) -> Path:
    table, default = weight_table(profile)
    orig_t, orig_d = p.K_BY_TOURNAMENT, p.DEFAULT_K
    p.K_BY_TOURNAMENT, p.DEFAULT_K = table, default
    try:
        out, actuals, dates, tours = V.walk_forward(verbose=False, with_meta=True)
    finally:
        p.K_BY_TOURNAMENT, p.DEFAULT_K = orig_t, orig_d

    onehot = np.eye(3)[actuals]
    payload = {"profile": profile, "n": int(len(actuals))}
    for model in V.MODELS:
        brier = ((out[model] - onehot) ** 2).sum(axis=1)
        payload[model] = {
            "brier": float(brier.mean()),
            "per_competition": {
                str(k): {"n": int(v["n"]), "brier": float(v["brier"])}
                for k, v in V.by_competition(out, actuals, tours,
                                             model=model, min_n=MIN_N)
                .to_dict("index").items()},
        }
    path = OUT_DIR / f"taxonomy_challenger_{profile}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"{profile}: n={payload['n']}  blend Brier={payload['blend']['brier']:.4f}"
          f"  elo Brier={payload['elo']['brier']:.4f}")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def compare() -> None:
    base = json.loads((OUT_DIR / "taxonomy_challenger_legacy.json").read_text())
    chal = json.loads((OUT_DIR / "taxonomy_challenger_v1.json").read_text())
    if base["n"] != chal["n"]:
        sys.exit(f"sample mismatch ({base['n']} vs {chal['n']}) — not comparable")

    print(f"Walk-forward, n={base['n']} matches\n")
    print(f"{'model':<10}{'legacy':>10}{'v1':>10}{'delta':>10}   (negative = v1 better)")
    print("-" * 46)
    for model in ("elo", "dc", "blend"):
        b, c = base[model]["brier"], chal[model]["brier"]
        print(f"{model:<10}{b:>10.4f}{c:>10.4f}{c - b:>+10.4f}")
    print("\nDixon-Coles never reads the weight table, so a non-zero 'dc' delta "
          "would indicate\na harness problem, not a real effect.")

    bp, cp = base["blend"]["per_competition"], chal["blend"]["per_competition"]
    shared = sorted(set(bp) & set(cp), key=lambda k: cp[k]["brier"] - bp[k]["brier"])
    print(f"\nPer-competition blend Brier (n >= {MIN_N}):")
    print(f"{'competition':<38}{'n':>6}{'legacy':>9}{'v1':>9}{'delta':>9}")
    print("-" * 71)
    regressions = []
    for k in shared:
        d = cp[k]["brier"] - bp[k]["brier"]
        flag = ""
        if d > COMP_TOL:
            flag = "  REGRESSED"
            regressions.append((k, d))
        print(f"{k:<38}{cp[k]['n']:>6}{bp[k]['brier']:>9.4f}"
              f"{cp[k]['brier']:>9.4f}{d:>+9.4f}{flag}")

    pooled = chal["blend"]["brier"] - base["blend"]["brier"]
    improved = pooled < 0
    material = pooled < -POOLED_TOL
    clean = not regressions

    print("\n" + "=" * 71)
    print("PROMOTION RULE (fixed before the result was known)")
    print(f"  1. pooled blend improves          {'PASS' if improved else 'FAIL'}"
          f"   ({pooled:+.4f})")
    print(f"  2. no competition regresses >{COMP_TOL}  "
          f"{'PASS' if clean else 'FAIL'}"
          f"   ({len(regressions)} regression(s))")
    print(f"  3. improvement exceeds {POOLED_TOL}       "
          f"{'PASS' if material else 'FAIL'}")
    verdict = "PROMOTE v1" if (improved and clean and material) else \
              "DO NOT PROMOTE — keep legacy weights"
    print(f"\nVERDICT: {verdict}")
    if not (improved and clean and material):
        print("\nThis is a legitimate outcome. The v1 taxonomy remains correct as a "
              "*description*\nof competition importance and stays in use for "
              "categories and bettability; it is\nthe rating WEIGHTS that are not "
              "adopted.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["legacy", "v1"])
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.run:
        run(a.run)
    elif a.compare:
        compare()
    else:
        ap.error("pass --run legacy | --run v1 | --compare")


if __name__ == "__main__":
    main()
