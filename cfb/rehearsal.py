"""Offline CFB Week 0 production rehearsal and machine-readable evidence."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import preflight

from . import generate_docs
from . import identity
from . import live_evidence
from . import season

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STATUS_JSON = DATA / "rehearsal_status.json"
HISTORY_JSON = DATA / "rehearsal_history.json"
SCHEDULE_REVIEW_JSON = DATA / "reviewed_schedule.json"

WEEK_ZERO_PROVIDER_NAMES = [
    "TCU Horned Frogs", "North Carolina Tar Heels",
    "San Jose State Spartans", "USC Trojans", "NC State Wolfpack",
    "Virginia Cavaliers", "Jacksonville State Gamecocks",
    "North Dakota State Bison", "Sacramento State Hornets",
    "Eastern Michigan Eagles", "New Mexico State Aggies",
    "Florida State Seminoles", "Hawaii Rainbow Warriors",
    "Stanford Cardinal", "Memphis Tigers", "UNLV Rebels",
]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path: str | Path, payload: object) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def verify_card(card_path: str | Path, manifest_path: str | Path) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    actual_hash = _sha256(card_path)
    result = manifest.get("result") or {}
    total_stake = float(result.get("total_stake", -1))
    return {
        "card_sha256": actual_hash,
        "manifest_hash_matches": actual_hash == manifest.get("card_sha256"),
        "betting_eligible": bool(result.get("betting_eligible")),
        "value_bets": int(result.get("value_bets", -1)),
        "total_stake": total_stake,
        "safe_diagnostic_card": (
            not result.get("betting_eligible")
            and int(result.get("value_bets", -1)) == 0
            and total_stake == 0.0
        ),
    }


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run() -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    checks: list[dict] = []

    cfb = preflight.build_report()["engines"]["cfb"]
    checks.append(_check(
        "diagnostic_preflight", cfb["diagnostic_ready"],
        "; ".join(cfb["issues"]) if cfb["issues"] else "all readiness checks pass",
    ))

    gate = subprocess.run(
        [sys.executable, "-m", "cfb.validate", "--gate", "--quiet"],
        cwd=HERE.parent, text=True, capture_output=True, check=False,
    )
    gate_detail = (gate.stdout + gate.stderr).strip().splitlines()
    checks.append(_check(
        "frozen_validation_gate", gate.returncode == 0,
        gate_detail[-1] if gate_detail else f"exit {gate.returncode}",
    ))

    readme = generate_docs.README.read_text()
    docs_current = readme == generate_docs.replace(readme, generate_docs.render())
    checks.append(_check(
        "generated_documentation", docs_current,
        "README frozen metrics are current" if docs_current else
        "run python3 -m cfb.generate_docs --write",
    ))

    evidence = live_evidence.health()
    checks.append(_check(
        "live_evidence", evidence["passed"],
        (f"{evidence['quote_rows']} quote rows; {evidence['signal_rows']} paper signals"
         if evidence["passed"] else "; ".join(evidence["issues"])),
    ))

    resolved = identity.review_names(
        WEEK_ZERO_PROVIDER_NAMES, 2026, "the-odds-api")
    unresolved = [row["provider_name"] for row in resolved
                  if row["status"] != "resolved"]
    checks.append(_check(
        "week_zero_identity", not unresolved,
        f"{len(resolved) - len(unresolved)}/{len(resolved)} provider names resolved"
        + (f"; unresolved: {', '.join(unresolved)}" if unresolved else ""),
    ))

    review = json.loads(SCHEDULE_REVIEW_JSON.read_text())
    # Compare decision-relevant schedule content, not raw provider bytes:
    # informational fields (CFBD's own pregame Elo, etc.) must not force a
    # re-review, but any identity/kickoff/scope change still must.
    schedule_hash = identity.schedule_identity_sha256(2026)
    reviewed = review.get("schedule_identity_sha256")
    if reviewed is None:  # pre-migration record: fall back to the raw hash
        reviewed = review.get("schedule_sha256")
        schedule_hash = _sha256(DATA / "schedule_2026.json")
    schedule_ok = schedule_hash == reviewed
    checks.append(_check(
        "reviewed_schedule", schedule_ok,
        f"current {schedule_hash[:16]}; reviewed {str(reviewed or 'missing')[:16]}",
    ))

    card_result: dict = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            card_result = season.build_card(days=1, bankroll=100.0)
        card_check = verify_card(card_result["card"], card_result["manifest"])
        card_ok = (
            card_result["model_state"].get("model_season") == 2026
            and card_result["model_state"].get("prior_mode") == "regression_only"
            and card_check["manifest_hash_matches"]
            and card_check["safe_diagnostic_card"]
        )
        checks.append(_check(
            "golden_card", card_ok,
            f"{card_result['games']} games; state "
            f"{card_result['model_state'].get('prior_mode')}; "
            f"stake £{card_check['total_stake']:.2f}; "
            f"manifest {card_check['card_sha256'][:16]}",
        ))
    except Exception as exc:
        checks.append(_check("golden_card", False, f"card build failed: {exc}"))

    passed = all(check["passed"] for check in checks)
    prior_history: list[dict] = []
    try:
        loaded = json.loads(HISTORY_JSON.read_text())
        if isinstance(loaded, list):
            prior_history = loaded
    except (OSError, json.JSONDecodeError):
        pass
    entry = {
        "run_at": now,
        "status": "pass" if passed else "failure",
        "card_sha256": next((c["detail"].split("manifest ", 1)[-1]
                             for c in checks if c["name"] == "golden_card"
                             and "manifest " in c["detail"]), None),
    }
    history = (prior_history + [entry])[-30:]
    consecutive = 0
    for prior in reversed(history):
        if prior.get("status") != "pass":
            break
        consecutive += 1
    latest_by_day: dict[str, dict] = {}
    for prior in history:
        latest_by_day[str(prior.get("run_at", ""))[:10]] = prior
    consecutive_days = 0
    for day in sorted(latest_by_day, reverse=True):
        if latest_by_day[day].get("status") != "pass":
            break
        consecutive_days += 1
    payload = {
        "schema_version": 1,
        "run_at": now,
        "status": "pass" if passed else "failure",
        "release_posture": "no_go" if not cfb.get("ready") else "review_required",
        "consecutive_clean_rehearsals": consecutive,
        "consecutive_clean_rehearsal_days": consecutive_days,
        "checks": checks,
        "preflight_issues": cfb["issues"],
        "model_state": card_result.get("model_state"),
    }
    _atomic_json(HISTORY_JSON, history)
    _atomic_json(STATUS_JSON, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"CFB rehearsal: {report['status']} "
              f"({report['consecutive_clean_rehearsals']} consecutive clean runs; "
              f"{report['consecutive_clean_rehearsal_days']} day(s))")
        for check in report["checks"]:
            print(f"  {'PASS' if check['passed'] else 'FAIL'} "
                  f"{check['name']}: {check['detail']}")
        if report["release_posture"] == "no_go":
            print("  NO-GO: betting readiness is not satisfied; diagnostic output only")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
