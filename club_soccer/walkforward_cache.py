#!/usr/bin/env python3
"""Fold-level cache for the walk-forward validation.

Why
---
The monthly-refit walk-forward is almost entirely redundant work. The fold for
2023-05 trains only on matches before 2023-05, and that data does not change
between yesterday's run and today's. Roughly 66 of 67 folds therefore recompute
byte-identical results on every run.

That was tolerable when the whole pass took ~30s. After the P3 expansion
(fixtures.csv 21,629 -> 52,074 rows) it takes ~2 minutes, and it runs on every
`update.sh` invocation — twice on Mondays, because season.py's weekly footer
called walk_forward again after the gate had already computed it.

This module caches each fold's prediction rows, keyed on a fingerprint of
everything that can change them. Unchanged folds are reloaded; only the current
month and any fold whose inputs actually moved are recomputed. The metric is
identical, not approximated.

The correctness risk, stated plainly
------------------------------------
A cache whose key is incomplete produces a gate that silently passes on stale
results — strictly worse than having no gate, because it looks like it ran. So
the fingerprint deliberately covers more than seems necessary:

  * every fixture column that feeds fitting or prediction, INCLUDING row order
    (Elo is sequential, so re-ordering the same matches can change ratings)
  * the source of model.py, competitions.py and validate.py
  * comp_strength.json and ensemble_weights.json, which are data files that
    change behaviour without any code edit
  * the fold's own options (min_train, league_adjustments)

If any of those move, every affected fold recomputes. When in doubt the cache
misses — an unnecessary recompute costs seconds, a false hit costs correctness.

What actually invalidates, in practice
--------------------------------------
Folds are chronological, so invalidation cascades FORWARD:

  * new matches in the current month  -> 1 fold recomputes
  * a new calendar month              -> 1 new fold
  * a backfill to a match on date D   -> every fold after D recomputes, since
                                         they all train on it
  * a model.py or comp_strength edit  -> all folds recompute

The third case is the one to watch. BSD backfills shot and xG data for recent
matches, which is normal and cheap (a few trailing folds). A retroactive
correction to an old season is not cheap — it invalidates everything after it,
and that is correct behaviour, not a bug. Expect an occasional slow run.

Measured on the post-P3 dataset (52,074 rows, 67 folds): ~2 min cold,
~0.4s fully warm, ~14s with a few trailing folds stale.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE_DIR = DATA / "walkforward_cache"

# Columns that can influence a fit or a prediction. Anything not listed here is
# assumed inert; add to this list rather than trusting that assumption.
KEY_COLUMNS = [
    "fixture_id", "date", "season", "competition", "type", "neutral",
    "home", "away", "home_goals", "away_goals", "status",
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners", "home_xg", "away_xg", "xg_source",
]

# Source and data files whose contents change model behaviour.
# Every file whose contents change a prediction must be here, or a stale fold
# is served after that file changes. league seeding (promoted, ON in
# production) reads uefa_registry.py + uefa_coefficients.json and
# league_strength.py; the identity layer (club_identity/names/club_registry)
# changes which team a row is priced as; schema.py governs status handling.
# Omitting any of these is a silent correctness hole — e.g. an annual
# coefficient refresh would otherwise be invisible to the cache.
_CODE_FILES = ("model.py", "competitions.py", "validate.py", "coverage.py",
               "uefa_registry.py", "league_strength.py", "club_identity.py",
               "names.py", "club_registry.py", "schema.py")
_DATA_FILES = ("comp_strength.json", "ensemble_weights.json",
               "uefa_coefficients.json", "uefa_coefficients_history.json",
               "club_alias_map.json")

# (signature, hash): the memo is keyed on a cheap (path, mtime, size) signature
# of every tracked file, so an on-disk edit WITHIN a process is picked up on the
# next call. The previous memo cached the first hash forever, so a coefficient
# or alias-map file edited mid-process stayed invisible and stale folds were
# served without any recompute. Recomputing the full hash only when the
# signature moves keeps it fast while closing that hole.
_fingerprint_memo: tuple[tuple, str] | None = None


def _tracked_paths() -> list[Path]:
    return ([HERE / name for name in _CODE_FILES]
            + [DATA / name for name in _DATA_FILES])


def _file_signature() -> tuple:
    sig = []
    for path in _tracked_paths():
        try:
            st = path.stat()
            sig.append((path.name, st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((path.name, 0, 0))
    return tuple(sig)


def code_fingerprint() -> str:
    """Hash of the code and data files that determine prediction behaviour.

    Memoized on a (path, mtime, size) signature so repeated calls are cheap but
    an on-disk change is never missed within the same process."""
    global _fingerprint_memo
    sig = _file_signature()
    if _fingerprint_memo is None or _fingerprint_memo[0] != sig:
        h = hashlib.sha256()
        for path in _tracked_paths():
            h.update(path.read_bytes() if path.exists() else b"")
        _fingerprint_memo = (sig, h.hexdigest()[:16])
    return _fingerprint_memo[1]


def reset_code_fingerprint() -> None:
    global _fingerprint_memo
    _fingerprint_memo = None


def row_hashes(df: pd.DataFrame) -> np.ndarray:
    """Per-row content hash, in the frame's current order."""
    present = [c for c in KEY_COLUMNS if c in df.columns]
    return pd.util.hash_pandas_object(df[present], index=False).to_numpy(dtype="uint64")


def fold_key(month: str, train_hashes: np.ndarray, test_hashes: np.ndarray,
             options: dict) -> str:
    """Fingerprint for one fold.

    Row ORDER is included deliberately: the Elo pass is sequential, so the same
    set of matches in a different order can produce different ratings. Hashing
    an order-insensitive summary would be a subtle way to get a false hit.
    """
    h = hashlib.sha256()
    h.update(month.encode())
    h.update(code_fingerprint().encode())
    h.update(json.dumps(options, sort_keys=True, default=str).encode())
    h.update(train_hashes.tobytes())
    h.update(b"|")
    h.update(test_hashes.tobytes())
    return h.hexdigest()[:32]


def _path(month: str, key: str) -> Path:
    return CACHE_DIR / f"{month}_{key}.json"


def load(month: str, key: str) -> list[dict] | None:
    path = _path(month, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["rows"]
    except Exception:
        # A corrupt entry must behave as a miss, never as an empty fold —
        # silently returning [] would drop a month from the metric.
        return None


def store(month: str, key: str, rows: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"month": month, "key": key, "n": len(rows),
               "stored_at": time.time(), "rows": rows}
    tmp = _path(month, key).with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(_path(month, key))


def prune(seen: set[tuple[str, str]]) -> int:
    """Drop superseded entries for months this run actually processed.

    Deliberately scoped to processed months. Removing everything the current
    run did not reference would mean a windowed call — `walk_forward(
    test_from="2026-01-01")`, as the eval harness and tuning helpers make —
    silently wiped every fold outside its window, so the next full run would
    recompute from scratch. Only a stale key for a month we just recomputed is
    genuinely dead.

    Best-effort: a filesystem that refuses the unlink must not break
    validation. A leftover file costs disk, never correctness — entries are
    only ever read back by exact key.
    """
    if not CACHE_DIR.exists():
        return 0
    months = {m for m, _ in seen}
    wanted = {f"{m}_{k}.json" for m, k in seen}
    removed = 0
    for path in CACHE_DIR.glob("*.json"):
        month = path.name.split("_", 1)[0]
        if month in months and path.name not in wanted:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def clear() -> int:
    """Best-effort wipe. A cache that cannot be deleted is a disk-space
    problem; failing the caller over it would turn that into an outage."""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for path in CACHE_DIR.glob("*.*"):
        try:
            path.unlink()
            n += 1
        except OSError:
            pass
    return n


def stats() -> dict:
    if not CACHE_DIR.exists():
        return {"entries": 0, "bytes": 0}
    files = list(CACHE_DIR.glob("*.json"))
    return {"entries": len(files),
            "bytes": sum(f.stat().st_size for f in files),
            "code_fingerprint": code_fingerprint()}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Walk-forward fold cache.")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    if args.clear:
        print(f"removed {clear()} cache file(s)")
        return
    s = stats()
    print(f"entries          : {s['entries']}")
    print(f"size             : {s.get('bytes', 0) / 1e6:.1f} MB")
    print(f"code fingerprint : {s.get('code_fingerprint')}")


if __name__ == "__main__":
    main()
