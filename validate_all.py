#!/usr/bin/env python3
"""Run every engine's validation gate and summarize (V3 M3).

Each engine ships its own walk-forward validation with a stored baseline:
  * World Cup    – python -m engines.worldcup.validate --quiet --gate
  * Club Soccer  – python -m club_soccer.validate --gate
  * CFB          – python -m cfb.validate --quiet --gate   (new in V3 M3)
  * Golf         – golf/validate.py --quiet --gate --sims <small default>

Each runs in its own working dir with PYTHONPATH pointed at the engine folder, so
the flat module names don't collide (same isolation the app runners use). A
per-engine table is printed and a machine-readable summary is written to
data/validation_suite.json. Exit code is non-zero if ANY engine regresses or
errors, so this doubles as a CI gate.

Usage:
  python3 validate_all.py --gate          # gate every engine
  python3 validate_all.py --gate --sims 5000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "data" / "validation_suite.json"


def _engines(sims: int, cfb_since: int) -> list[dict]:
    return [
        {"id": "worldcup", "cwd": ROOT,
         "cmd": ["-m", "engines.worldcup.validate", "--quiet", "--gate"], "timeout": 600},
        {"id": "club_soccer", "cwd": ROOT,
         "cmd": ["-m", "club_soccer.validate", "--gate"], "timeout": 600},
        {"id": "cfb", "cwd": ROOT,
         "cmd": ["-m", "cfb.validate", "--quiet", "--gate", "--since", str(cfb_since)],
         "timeout": 600},
        {"id": "golf", "cwd": ROOT / "golf",
         "cmd": ["validate.py", "--quiet", "--gate", "--sims", str(sims)],
         "timeout": 900},
    ]


def _run(engine: dict) -> dict:
    cwd = engine["cwd"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, *engine["cmd"]],
            cwd=str(cwd), env=env, capture_output=True, text=True,
            timeout=engine["timeout"])
        rc = proc.returncode
        out = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return {"status": "ERROR", "returncode": None,
                "seconds": round(time.time() - start, 1), "detail": "timeout"}
    except Exception as e:  # noqa: BLE001
        return {"status": "ERROR", "returncode": None,
                "seconds": round(time.time() - start, 1), "detail": str(e)[:120]}
    status = "PASS" if rc == 0 else ("FAIL" if rc == 1 else "ERROR")
    tail = "\n".join(l for l in out.strip().splitlines() if l.strip())[-400:]
    return {"status": status, "returncode": rc,
            "seconds": round(time.time() - start, 1), "detail": tail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true",
                    help="(default behaviour) gate every engine")
    ap.add_argument("--sims", type=int, default=5000,
                    help="golf validation sims (small default for a fast gate)")
    ap.add_argument("--cfb-since", type=int, default=2023,
                    help="CFB first validation season")
    args = ap.parse_args()

    results: dict[str, dict] = {}
    for engine in _engines(args.sims, args.cfb_since):
        print(f"▶ {engine['id']} …", flush=True)
        results[engine["id"]] = _run(engine)

    SUMMARY.parent.mkdir(exist_ok=True)
    SUMMARY.write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "engines": results},
        indent=2))

    print(f"\n{'engine':<14s}{'status':>8s}{'secs':>8s}  notes")
    print("-" * 60)
    worst = 0
    for eid, r in results.items():
        if r["status"] != "PASS":
            worst = 1
        note = ""
        if r["status"] == "FAIL":
            note = "regression vs baseline"
        elif r["status"] == "ERROR":
            note = r["detail"].splitlines()[-1] if r["detail"] else "error"
        print(f"{eid:<14s}{r['status']:>8s}{r['seconds']:>8.1f}  {note[:40]}")
    print("-" * 60)
    print(f"summary written to {SUMMARY.relative_to(ROOT)}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
