#!/usr/bin/env python3
"""Lock World Cup predictions at the moment they are made.

Appends the current `predictions_worldcup_2026.csv` into an append-only archive
(`data/prediction_snapshots.csv`). The FIRST time a match is seen it is locked;
later runs never overwrite it. Retraining, recalibration, or any other model
change therefore cannot alter the prediction that backtesting will score against
-- the snapshot is the point-in-time record.

Run this regularly (ideally once a day, before each matchday kicks off) so every
fixture is frozen while it is still unplayed.

Usage:
  python snapshot_predictions.py            # lock any new (unseen) fixtures
  python snapshot_predictions.py --status   # show counts only, write nothing
"""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
PRED = ROOT / "predictions_worldcup_2026.csv"
ARCHIVE = ROOT / "data" / "prediction_snapshots.csv"

KEY = ["match_date", "home", "away"]
COLS = ["snapshot_ts", "model_version", "match_date", "home", "away",
        "xg_home", "xg_away", "p_home", "p_draw", "p_away", "p_btts",
        "likely_score"]


def model_version():
    """Short git hash so each locked row records the model state that made it."""
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=ROOT,
            stderr=subprocess.DEVNULL) != 0
        return f"git:{h}{'+dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def load_archive():
    if ARCHIVE.exists():
        return pd.read_csv(ARCHIVE)
    return pd.DataFrame(columns=COLS)


def main():
    status = "--status" in sys.argv
    if not PRED.exists():
        sys.exit(f"No predictions file at {PRED}")

    arc = load_archive()
    cur = pd.read_csv(PRED).rename(columns={"date": "match_date"})
    locked = set(map(tuple, arc[KEY].values.tolist())) if len(arc) else set()
    mask = [(r.match_date, r.home, r.away) not in locked
            for r in cur.itertuples(index=False)]
    new = cur[mask].copy()

    print(f"Archive: {len(arc)} locked | predictions file: {len(cur)} rows | "
          f"new to lock: {len(new)}")
    if status or new.empty:
        return

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new["snapshot_ts"] = ts
    new["model_version"] = model_version()
    out = new[COLS]
    out.to_csv(ARCHIVE, mode="a", header=not ARCHIVE.exists(), index=False)
    print(f"Locked {len(out)} new prediction(s) at {ts} "
          f"({new['model_version'].iloc[0]}) -> {ARCHIVE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
