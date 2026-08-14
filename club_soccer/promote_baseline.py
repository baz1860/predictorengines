#!/usr/bin/env python3
"""Re-pin the promotion gate reference. PROMOTER ACTION — deliberate, audited.

    python3 -m club_soccer.promote_baseline [--force]

Why this is its own module
--------------------------
``validate.py`` is DESCRIPTIVE: it measures where the model stands and may
freely rewrite validation_latest.json. It must never move the gate, and
tests/club_soccer/test_baseline_ownership.py enforces that by asserting the
string ``PROMOTION_BASELINE.write_text`` never appears in validate.py. The
module that measures must not be the module that moves the goalposts, even
behind a flag — so the writer lives here instead.

When re-pinning is legitimate
-----------------------------
Only when the evaluation POPULATION changed and the old reference therefore
describes a sample that no longer exists: rows deduplicated, club identities
merged, a data correction applied. The 2026-08-12 identity merge did exactly
that — 27,181 evaluation rows became 26,969 once duplicated matches were
reconciled — and until the baseline is re-pinned the gate cannot say anything
at all, because it is comparing two different samples.

When it is not
--------------
When a METRIC regressed. Same sample, worse model, which is the single thing
the gate exists to catch; re-pinning would erase it. That case is refused
unless you pass --force, and --force records the override in the file.

The superseded reference is retained inside the new baseline so that later
nobody has to take on trust whether a re-pin was honest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import model as M
from . import validate as V
from . import walkforward_cache as WFC

METRICS = ("brier", "log_loss", "brier_ou25", "brier_btts")


def build_payload(rows: list[dict], measured: dict, previous: dict | None,
                  force: bool = False) -> dict:
    """The new baseline document, or raise explaining why it is refused."""
    if previous is None:
        raise ValueError(
            f"no existing {V.PROMOTION_BASELINE.name} to supersede; establish "
            "the first baseline deliberately in a commit"
        )
    failures = V.gate_failures(rows, measured, previous)
    if not failures:
        raise ValueError(
            "gate already passes against the current baseline; nothing to "
            "re-baseline"
        )
    population, metric = V.partition_gate_failures(failures)
    if metric and not force:
        raise ValueError(
            "refusing to re-baseline: these are metric regressions, not a "
            "population change, and re-pinning would hide them:\n  "
            + "\n  ".join(metric)
            + "\nRe-run with --force only if you have understood why."
        )

    fixture_frame = M.played(M.load_fixtures()).sort_values(
        "date"
    ).reset_index(drop=True)
    data_hash = hashlib.sha256(
        WFC.row_hashes(fixture_frame).tobytes()
    ).hexdigest()[:20]

    payload = dict(previous)
    payload.update({name: float(measured[name]) for name in METRICS})
    payload.update({
        "n": int(measured["n"]),
        "evaluation_hash": V._evaluation_hash(rows),
        "code_hash": WFC.code_fingerprint(),
        "fixture_data_hash": data_hash,
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "promoted_on_host": socket.gethostname(),
        "superseded": {
            "n": previous.get("n"),
            "evaluation_hash": previous.get("evaluation_hash"),
            "brier": previous.get("brier"),
            "reason": population or ["forced past metric regression"],
            "forced": bool(metric and force),
        },
    })
    return payload


def promote(force: bool = False, verbose: bool = True) -> dict:
    previous = V.load_promotion_baseline()
    if previous is None:
        raise ValueError(f"--needs an existing {V.PROMOTION_BASELINE.name}")
    rows, measured = V.walk_forward(
        verbose=verbose, test_from=previous.get("test_from"),
        test_to=previous.get("test_to"),
    )
    payload = build_payload(rows, measured, previous, force=force)
    V.PROMOTION_BASELINE.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="re-pin even when a metric has regressed")
    args = ap.parse_args()

    previous = V.load_promotion_baseline() or {}
    try:
        payload = promote(force=args.force)
    except ValueError as exc:
        sys.exit(str(exc))

    print(f"\nRe-pinned {V.PROMOTION_BASELINE.name}")
    print(f"  n            {previous.get('n')} -> {payload['n']}")
    print(f"  population   {previous.get('evaluation_hash')} -> "
          f"{payload['evaluation_hash']}")
    print(f"  brier        {previous.get('brier'):.6f} -> {payload['brier']:.6f}")
    print(f"  superseded because: "
          f"{'; '.join(payload['superseded']['reason'])}")
    print("\n  Commit this file — it is the promotion gate reference.")


if __name__ == "__main__":
    main()
