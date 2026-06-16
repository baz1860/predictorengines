#!/usr/bin/env python3
"""Lock edge (value-bet) recommendations at the moment they are made.

`edge.py` regenerates `edge_report.csv` every run, and that file is gitignored,
so the model's recommendation for a given match -- its model probability, the
book price, the computed edge and the suggested stake -- is otherwise lost the
moment the model is retrained or the odds move. This freezes each recommendation
into an append-only archive (`data/edge_snapshots.csv`) so betting backtests stay
reproducible no matter how the model changes later.

The FIRST time a (date, home, away, market, side, bet) recommendation is seen it
is locked; later runs never overwrite it.

Run it right after `python edge.py ...` produces a report, before you place bets.

Usage:
  python snapshot_edges.py                  # lock new recommendations from edge_report.csv
  python snapshot_edges.py --source FILE     # lock from a different report file
  python snapshot_edges.py --status          # show counts only, write nothing
"""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DEFAULT_SOURCE = ROOT / "edge_report.csv"
ARCHIVE = ROOT / "data" / "edge_snapshots.csv"

# Identity of a recommendation (one fixture can yield several bets/markets).
KEY = ["date", "home", "away", "market", "side", "bet"]
SRC_COLS = ["date", "match", "home", "away", "side", "market", "bet", "odds",
            "p_book", "p_model", "edge", "ev_per_unit", "kelly_stake",
            "overround", "elo_gap", "stake_gbp"]
COLS = ["snapshot_ts", "model_version"] + SRC_COLS


def model_version():
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
    source = Path(sys.argv[sys.argv.index("--source") + 1]) if "--source" in sys.argv \
        else DEFAULT_SOURCE

    arc = load_archive()
    if not source.exists():
        print(f"No edge report at {source}. Run `python edge.py ...` first.")
        if not ARCHIVE.exists():
            arc.to_csv(ARCHIVE, index=False)  # create empty archive so it exists
            print(f"Created empty archive -> {ARCHIVE.relative_to(ROOT)}")
        return

    cur = pd.read_csv(source)
    if cur.empty:
        print(f"{source.name} has no recommendations to lock "
              f"(archive: {len(arc)} locked).")
        if not ARCHIVE.exists():
            arc.to_csv(ARCHIVE, index=False)
            print(f"Created empty archive -> {ARCHIVE.relative_to(ROOT)}")
        return

    locked = set(map(tuple, arc[KEY].values.tolist())) if len(arc) else set()
    mask = [tuple(getattr(r, k) for k in KEY) not in locked
            for r in cur.itertuples(index=False)]
    new = cur[mask].copy()
    print(f"Archive: {len(arc)} locked | report: {len(cur)} rows | "
          f"new to lock: {len(new)}")
    if status or new.empty:
        return

    new["snapshot_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new["model_version"] = model_version()
    out = new.reindex(columns=COLS)
    out.to_csv(ARCHIVE, mode="a", header=not ARCHIVE.exists(), index=False)
    print(f"Locked {len(out)} new recommendation(s) "
          f"({new['model_version'].iloc[0]}) -> {ARCHIVE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
