"""Compatibility harness: freeze legacy World Cup behaviour (plan §8 step 1).

The rule the plan insists on: **"World Cup output unchanged" means identical output
from the legacy path on frozen inputs with fixed seeds** — not identical live
numbers. Live numbers legitimately move when results.csv gains matches. So the
goldens here pin:

  * the legacy competition-weight table, byte for byte;
  * Elo ratings and 1X2 probabilities computed from a FROZEN fixture set, not from
    the live file;
  * the engine adapter's identity and declared capabilities;
  * the canonical edge-row contract.

Regenerate deliberately with `--update` after a change you intend. A golden that
changes without `--update` is the harness doing its job.

Why this exists before the refactor: the plan's stage order puts the compatibility
harness FIRST, ahead of the taxonomy challenger and any rename, because a harness
written after a behavioural change freezes the bug rather than the behaviour.
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

GOLDEN = Path(__file__).parent / "goldens" / "legacy_worldcup.json"
FAIL = 0

# A frozen mini-history. Deliberately synthetic and tiny: it exercises the rating
# update, the tournament-weight lookup, home advantage, neutral venues and the
# scoreline matrix without depending on data/results.csv, which changes daily.
FROZEN_MATCHES = [
    # date, home, away, hs, as, tournament, neutral
    ("2020-01-05", "Alpha", "Bravo", 2, 1, "Friendly", False),
    ("2020-03-11", "Bravo", "Charlie", 0, 0, "FIFA World Cup qualification", False),
    ("2020-06-20", "Alpha", "Charlie", 3, 0, "FIFA World Cup", True),
    ("2021-02-14", "Charlie", "Alpha", 1, 1, "UEFA Euro qualification", False),
    ("2021-09-02", "Bravo", "Alpha", 2, 3, "Friendly", False),
    ("2022-05-19", "Charlie", "Bravo", 4, 2, "UEFA Nations League", False),
    ("2023-01-30", "Alpha", "Bravo", 1, 0, "Island Games", True),
    ("2023-11-11", "Charlie", "Alpha", 0, 2, "FIFA World Cup", False),
]
PAIRS = [("Alpha", "Bravo", False), ("Alpha", "Charlie", True),
         ("Bravo", "Charlie", False)]


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


def _frozen_df() -> pd.DataFrame:
    df = pd.DataFrame(FROZEN_MATCHES, columns=[
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "neutral"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute() -> dict:
    """Everything the goldens pin, computed from frozen inputs only."""
    from engines.worldcup import predictor as p
    from international import taxonomy as T

    played = _frozen_df()
    ratings, played = p.compute_elo(played)
    beta = p.fit_goal_model(played) if len(played) > 4 else None

    probs = {}
    for h, a, neutral in PAIRS:
        adv = 0.0 if neutral else p.HOME_ADV
        rh, ra = ratings.get(h, p.BASE_RATING), ratings.get(a, p.BASE_RATING)
        lam_h, lam_a = p.expected_goals(rh + adv, ra, beta)
        M = p.score_matrix(lam_h, lam_a)
        w = float(np.tril(M, -1).sum()); d = float(np.trace(M))
        l = float(np.triu(M, 1).sum())
        s = w + d + l
        probs[f"{h}|{a}|{'N' if neutral else 'H'}"] = [
            round(w / s, 10), round(d / s, 10), round(l / s, 10),
            round(float(lam_h), 10), round(float(lam_a), 10)]

    from contracts import CANONICAL_EDGE_FIELDS

    return {
        "legacy_k_table": dict(sorted(p.K_BY_TOURNAMENT.items())),
        "legacy_default_k": p.DEFAULT_K,
        "base_rating": p.BASE_RATING,
        "elo_home_adv": p.ELO_HOME_ADV,
        "max_goals": p.MAX_GOALS,
        "ratings": {k: round(float(v), 10) for k, v in sorted(ratings.items())},
        "probabilities": probs,
        # The taxonomy's legacy profile must agree with the engine, or every
        # golden below is measuring the wrong thing.
        "taxonomy_legacy_agrees": all(
            T.k_for(k, "legacy") == v for k, v in p.K_BY_TOURNAMENT.items()),
        "canonical_edge_fields": list(CANONICAL_EDGE_FIELDS),
    }


def adapter_identity() -> dict:
    from app.engines.worldcup import WorldCupAdapter
    a = WorldCupAdapter()
    return {"id": a.id, "sport": a.sport,
            "capabilities": sorted(a.capabilities)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="rewrite the goldens (use only for intended changes)")
    a = ap.parse_args()

    current = compute()
    current["adapter"] = adapter_identity()

    if a.update or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(current, indent=2, sort_keys=True))
        print(f"wrote goldens -> {GOLDEN.relative_to(ROOT)}")
        if not a.update:
            print("(first run: baseline established)")
        return 0

    golden = json.loads(GOLDEN.read_text())
    print("legacy goldens")
    for key in sorted(golden):
        check(f"{key} unchanged", golden[key] == current.get(key),
              f"expected {json.dumps(golden[key])[:120]}, "
              f"got {json.dumps(current.get(key))[:120]}")

    check("taxonomy legacy profile agrees with the engine",
          current["taxonomy_legacy_agrees"])

    print(f"\n{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURE(S)'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
