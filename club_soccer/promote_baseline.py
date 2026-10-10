#!/usr/bin/env python3
"""Re-pin the promotion gate reference. PROMOTER ACTION — deliberate, audited.

    ./club_soccer/update.sh                        # let the data settle FIRST
    python3 -m club_soccer.promote_baseline [--force]
    python3 -m club_soccer.validate --gate         # confirm it took

Order matters. The pipeline's fd.co.uk refresh replaces historical rows inside
the pinned evaluation window, so a baseline pinned BEFORE a run is stale by the
end of it. Pin after a completed run, then verify.

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

Validated improvement (``--validated-improvement EVIDENCE.json``)
-----------------------------------------------------------------
A promoted model change (e.g. E1's pooled component) leaves the gate PASSING
but loosened: the baseline still holds the incumbent's metrics, so the model
may drift back by the gain plus tolerance before anything fails. That is
neither a population change nor a regression, so the cases above refuse it.

This third case ratchets the reference forward, and only on evidence. The
evidence file must be a ``promote`` verdict from a §12 A/B (status
"promoted", §12.1-2 headline and §12.3 time-splits passed) AND it must be
about exactly this comparison: same evaluation window, row count and
population hash as the pinned baseline; its incumbent metrics must equal the
pinned reference; its candidate metrics must equal what the current model
measures now. The measured model must also be strictly better on 1X2 Brier
and no worse on log-loss. Anything else is refused. The evidence file's name
and digest are recorded in the new baseline.
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
PROMOTION_POPULATION = V.DATA / "promotion_population.json"


# Metrics in the evidence file must reproduce the pinned / measured values to
# this absolute tolerance. Both sides come from the same deterministic
# walk-forward, so anything looser would admit a different sample or model.
EVIDENCE_MATCH_TOL = 1e-4


def verify_improvement_evidence(evidence: dict, measured: dict,
                                previous: dict, population_hash: str) -> list[str]:
    """Reasons the evidence does NOT justify a ratchet (empty = justified)."""
    problems: list[str] = []
    if evidence.get("status") != "promoted" or evidence.get("decision") != "promote":
        problems.append("evidence verdict is not a promotion "
                        f"(status={evidence.get('status')!r}, "
                        f"decision={evidence.get('decision')!r})")
    if not evidence.get("gate_12_1_2_headline"):
        problems.append("evidence did not pass the §12.1-2 headline gate")
    splits = evidence.get("gate_12_3_time_splits") or {}
    if not splits.get("pass"):
        problems.append("evidence did not pass the §12.3 time-split gate")
    for key in ("test_from", "test_to", "n"):
        if evidence.get(key) != previous.get(key):
            problems.append(f"evidence {key} {evidence.get(key)!r} != "
                            f"baseline {previous.get(key)!r}")
    for label, value in (("evidence", evidence.get("evaluation_hash")),
                         ("measured population", population_hash)):
        if value != previous.get("evaluation_hash"):
            problems.append(f"{label} hash {value} != baseline "
                            f"{previous.get('evaluation_hash')}")
    incumbent = evidence.get("incumbent") or {}
    candidate = evidence.get("hierarchical") or evidence.get("candidate") or {}
    for name in METRICS:
        try:
            if abs(float(incumbent[name]) - float(previous[name])) > EVIDENCE_MATCH_TOL:
                problems.append(f"evidence incumbent {name} does not match "
                                "the pinned baseline")
            if abs(float(candidate[name]) - float(measured[name])) > EVIDENCE_MATCH_TOL:
                problems.append(f"evidence candidate {name} does not match "
                                "the current model's measurement")
        except (KeyError, TypeError, ValueError):
            problems.append(f"evidence lacks a usable {name}")
    try:
        if not float(measured["brier"]) < float(previous["brier"]):
            problems.append("current 1X2 Brier is not strictly better than "
                            "the baseline")
        if float(measured["log_loss"]) > float(previous["log_loss"]):
            problems.append("current log-loss is worse than the baseline")
    except (KeyError, TypeError, ValueError):
        problems.append("measured/baseline Brier or log-loss malformed")
    return problems


def build_payload(rows: list[dict], measured: dict, previous: dict | None,
                  force: bool = False, improvement: dict | None = None,
                  improvement_source: Path | None = None) -> dict:
    """The new baseline document, or raise explaining why it is refused."""
    if previous is None:
        raise ValueError(
            f"no existing {V.PROMOTION_BASELINE.name} to supersede; establish "
            "the first baseline deliberately in a commit"
        )
    failures = V.gate_failures(rows, measured, previous)
    ratchet_reason: list[str] | None = None
    if not failures:
        if improvement is None:
            raise ValueError(
                "gate already passes against the current baseline; nothing "
                "to re-baseline (to ratchet after a validated model "
                "promotion, pass --validated-improvement EVIDENCE.json)"
            )
        problems = verify_improvement_evidence(
            improvement, measured, previous, V._evaluation_hash(rows))
        if problems:
            raise ValueError("refusing to ratchet; the evidence does not "
                             "justify it:\n  " + "\n  ".join(problems))
        digest = hashlib.sha256(
            json.dumps(improvement, sort_keys=True).encode()).hexdigest()[:20]
        name = improvement_source.name if improvement_source else "<inline>"
        ratchet_reason = [f"validated improvement: {name} ({digest}) — "
                          f"brier {previous.get('brier'):.6f} -> "
                          f"{float(measured['brier']):.6f}"]
        population, metric = [], []
    else:
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
        "evaluation_hash_schema": V.EVALUATION_HASH_SCHEMA,
        "population_snapshot": {
            "path": PROMOTION_POPULATION.name,
            "schema": V.EVALUATION_HASH_SCHEMA,
        },
        "code_hash": WFC.code_fingerprint(),
        "fixture_data_hash": data_hash,
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "promoted_on_host": socket.gethostname(),
        "superseded": {
            "n": previous.get("n"),
            "evaluation_hash": previous.get("evaluation_hash"),
            "brier": previous.get("brier"),
            "reason": (ratchet_reason or population
                       or ["forced past metric regression"]),
            "forced": bool(metric and force),
        },
    })
    if ratchet_reason:
        payload["superseded"]["kind"] = "validated_improvement"
    return payload


def _window_fingerprint(previous: dict) -> str:
    """Cheap digest of the evaluation window's rows, for drift detection."""
    frame = M.played(M.load_fixtures()).sort_values("date").reset_index(drop=True)
    window = frame[(frame["date"] >= str(previous.get("test_from")))
                   & (frame["date"] < str(previous.get("test_to")))]
    return hashlib.sha256(WFC.row_hashes(window).tobytes()).hexdigest()[:20]


def promote(force: bool = False, verbose: bool = True,
            improvement_path: Path | None = None) -> dict:
    """Pin a new baseline, and refuse if the data moved while measuring.

    The walk-forward takes minutes. fixtures.csv is written by the pipeline and
    by Syncthing, and the fd.co.uk refresh in particular replaces historical
    rows INSIDE the pinned window — the 2026-08-14 run reported "reconciled 33
    cross-provider duplicate row(s)" and "fixtures.csv 64886 -> 64886 (+0)",
    which is a changed population at an unchanged row count.

    A baseline pinned to a population that has already moved is worse than no
    baseline: the gate fails on the very next run and the operator reasonably
    concludes the re-pin did not work. So fingerprint the window before and
    after, and refuse rather than write something already stale.
    """
    previous = V.load_promotion_baseline()
    if previous is None:
        raise ValueError(f"needs an existing {V.PROMOTION_BASELINE.name}")

    before = _window_fingerprint(previous)
    rows, measured = V.walk_forward(
        verbose=verbose, test_from=previous.get("test_from"),
        test_to=previous.get("test_to"),
    )
    improvement = (json.loads(Path(improvement_path).read_text())
                   if improvement_path else None)
    payload = build_payload(rows, measured, previous, force=force,
                            improvement=improvement,
                            improvement_source=improvement_path)

    M.load_fixtures.cache_clear() if hasattr(
        M.load_fixtures, "cache_clear") else None
    after = _window_fingerprint(previous)
    if after != before:
        raise ValueError(
            "refusing to pin: fixtures.csv changed inside the evaluation "
            f"window while measuring ({before} -> {after}).\n"
            "A pipeline run almost certainly overlapped this one. Let "
            "./club_soccer/update.sh finish, then re-run this command so the "
            "baseline pins to settled data."
        )

    snapshot = V.build_population_snapshot(rows)
    if snapshot["evaluation_hash"] != payload.get("evaluation_hash"):
        raise ValueError("refusing to pin: population snapshot hash disagrees "
                         "with the proposed baseline")

    # Write both promoter-owned artifacts through temporary files. The
    # snapshot goes first: if the process stops between replacements, the old
    # baseline rejects the unmatched snapshot rather than trusting it.
    population_tmp = PROMOTION_POPULATION.with_suffix(".tmp")
    baseline_tmp = V.PROMOTION_BASELINE.with_suffix(".tmp")
    population_tmp.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
    )
    baseline_tmp.write_text(json.dumps(payload, indent=2) + "\n")
    population_tmp.replace(PROMOTION_POPULATION)
    baseline_tmp.replace(V.PROMOTION_BASELINE)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="re-pin even when a metric has regressed")
    ap.add_argument("--validated-improvement", metavar="EVIDENCE.json",
                    type=Path,
                    help="ratchet the reference after a promoted model "
                         "change, justified by a §12 A/B evidence file")
    args = ap.parse_args()

    previous = V.load_promotion_baseline() or {}
    try:
        payload = promote(force=args.force,
                          improvement_path=args.validated_improvement)
    except ValueError as exc:
        sys.exit(str(exc))

    print(f"\nRe-pinned {V.PROMOTION_BASELINE.name}")
    print(f"  n            {previous.get('n')} -> {payload['n']}")
    print(f"  population   {previous.get('evaluation_hash')} -> "
          f"{payload['evaluation_hash']}")
    print(f"  brier        {previous.get('brier'):.6f} -> {payload['brier']:.6f}")
    print(f"  superseded because: "
          f"{'; '.join(payload['superseded']['reason'])}")
    print(f"  population audit -> {PROMOTION_POPULATION.name}")
    print("\n  Commit both promoter-owned files — they are one gate reference.")
    print("  Confirm it took:  python3 -m club_soccer.validate --gate")


if __name__ == "__main__":
    main()
