#!/usr/bin/env python3
"""Backtest the *locked* prediction snapshots against actual results.

This reads frozen predictions from `data/prediction_snapshots.csv` (written by
`snapshot_predictions.py`) and scores them against real scores in
`data/results.csv`. Because it never calls the live model, retraining or model
changes cannot move these numbers -- only adding new locked snapshots or new
actual results can. That is the whole point of snapshotting.

If a match was locked more than once, the EARLIEST snapshot is used (the
prediction as it stood before kickoff).

Usage:
  python backtest_snapshots.py
  python backtest_snapshots.py --csv out.csv   # also write per-match detail
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
ARCHIVE = ROOT / "data" / "prediction_snapshots.csv"
RESULTS = ROOT / "data" / "results.csv"


def outcome(hs, a):
    return 0 if hs > a else (1 if hs == a else 2)


def main():
    snaps = pd.read_csv(ARCHIVE)
    # earliest lock per match
    snaps = (snaps.sort_values("snapshot_ts")
                  .drop_duplicates(["match_date", "home", "away"], keep="first"))

    res = pd.read_csv(RESULTS)
    res["home_score"] = pd.to_numeric(res["home_score"], errors="coerce")
    res["away_score"] = pd.to_numeric(res["away_score"], errors="coerce")
    res = res.dropna(subset=["home_score", "away_score"])
    # Key on (date, home, away) so we match the exact fixture, never a prior
    # meeting between the same two nations.
    actual = {(str(r.date)[:10], r.home_team, r.away_team):
              (int(r.home_score), int(r.away_score))
              for r in res.itertuples(index=False)}

    rows, brier = [], 0.0
    ll = acc = shit = 0.0
    n = 0
    pending = 0
    for s in snaps.itertuples(index=False):
        key = (str(s.match_date)[:10], s.home, s.away)
        if key not in actual:
            pending += 1
            continue
        hs, as_ = actual[key]
        p = np.array([s.p_home, s.p_draw, s.p_away])
        ai = outcome(hs, as_)
        av = np.zeros(3); av[ai] = 1
        pick = int(np.argmax(p))
        sh = int(str(s.likely_score) == f"{hs}-{as_}")
        brier += np.sum((p - av) ** 2)
        ll += -np.log(max(p[ai], 1e-12))
        acc += int(pick == ai)
        shit += sh
        n += 1
        rows.append(dict(match_date=s.match_date,
                         match=f"{s.home} {hs}-{as_} {s.away}",
                         model_version=s.model_version,
                         p_home=s.p_home, p_draw=s.p_draw, p_away=s.p_away,
                         pick=["HOME", "DRAW", "AWAY"][pick],
                         actual=["HOME", "DRAW", "AWAY"][ai],
                         hit="Y" if pick == ai else "N",
                         p_on_actual=round(float(p[ai]), 3),
                         likely=s.likely_score, score=f"{hs}-{as_}",
                         score_hit="Y" if sh else "N"))

    if n == 0:
        print("No locked snapshots have actual results yet.")
        return
    df = pd.DataFrame(rows).sort_values("match_date")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nScored {n} locked predictions ({pending} still pending a result).")
    print(f"  3-way accuracy : {acc/n:.1%}   (chance 33.3%)")
    print(f"  Brier (avg)    : {brier/n:.4f}  (chance 0.667)")
    print(f"  Log-loss (avg) : {ll/n:.4f}  (chance 1.099)")
    print(f"  Exact scoreline: {int(shit)}/{n}")
    print(f"  Avg p(actual)  : {df.p_on_actual.mean():.3f}")

    if "--csv" in sys.argv:
        dest = sys.argv[sys.argv.index("--csv") + 1]
        df.to_csv(dest, index=False)
        print(f"\nPer-match detail -> {dest}")


if __name__ == "__main__":
    main()
