#!/usr/bin/env python3
"""Offline preflight: report missing data files and missing API keys (V3 M2).

Pure local check — never makes a network call. Tells you, per engine, which key
inputs and fitted-model files are present and how stale they are, and which API
keys are configured (masked). Use it before a refresh/predict session, or as a
quick "is this checkout ready?" smoke test.

Usage:
  python3 preflight.py            # human-readable table
  python3 preflight.py --json     # machine-readable
Exit code is 0 always (missing data must not block offline operation); read the
report to decide what to refresh.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Per-engine key inputs. (label, relative path, required?) — required files that
# are missing show as ✗; optional ones as ·.
ENGINE_FILES: dict[str, list[tuple[str, str, bool]]] = {
    "worldcup": [
        ("results", "data/results.csv", True),
        ("goal model", "data/dc_params.json", True),
        ("calibration", "data/calibration.json", False),
        ("market blend", "data/market_blend.json", False),
    ],
    "club_soccer": [
        ("fixtures", "club_soccer/data/fixtures.csv", True),
        ("model params", "club_soccer/data/model_params.json", True),
        ("calibration", "club_soccer/data/calibration.json", False),
        ("odds", "club_soccer/data/odds.csv", False),
    ],
    "cfb": [
        ("games", "cfb/data/games.csv", True),
        ("power params", "cfb/data/power_params.json", True),
        ("upcoming", "cfb/data/upcoming.csv", False),
        ("odds", "cfb/odds.csv", False),
    ],
    "golf": [
        ("rounds", "golf/data/rounds.csv", True),
        ("model params", "golf/data/model_params.json", True),
        ("field", "golf/data/field.csv", False),
        ("odds", "golf/data/odds.csv", False),
    ],
}

# Which API keys each engine can use (for the masked-key report).
ENGINE_KEYS: dict[str, list[str]] = {
    "worldcup": ["the-odds-api"],
    "club_soccer": ["api-football", "the-odds-api"],
    "cfb": [],
    "golf": ["datagolf", "the-odds-api"],
}


def _age(path: Path) -> str:
    secs = time.time() - path.stat().st_mtime
    days = secs / 86400
    if days >= 1:
        return f"{days:.0f}d"
    hours = secs / 3600
    return f"{hours:.0f}h" if hours >= 1 else f"{secs/60:.0f}m"


def build_report() -> dict:
    try:
        from app import settings_store
        keys_set = settings_store.public_view().get("odds_api_keys_set", {})
    except Exception:
        keys_set = {}

    engines = {}
    for eid, files in ENGINE_FILES.items():
        file_rows = []
        missing_required = 0
        for label, rel, required in files:
            p = ROOT / rel
            exists = p.exists()
            if required and not exists:
                missing_required += 1
            file_rows.append({
                "label": label, "path": rel, "exists": exists,
                "required": required,
                "age": _age(p) if exists else None,
            })
        key_rows = [{"source": k, "set": bool(keys_set.get(k))}
                    for k in ENGINE_KEYS.get(eid, [])]
        engines[eid] = {
            "files": file_rows,
            "keys": key_rows,
            "ready": missing_required == 0,
            "missing_required": missing_required,
        }
    return {"engines": engines}


def _print(report: dict) -> None:
    for eid, e in report["engines"].items():
        flag = "ready" if e["ready"] else f"MISSING {e['missing_required']} required"
        print(f"\n{eid}  [{flag}]")
        for f in e["files"]:
            mark = "✓" if f["exists"] else ("✗" if f["required"] else "·")
            age = f"  ({f['age']})" if f["age"] else ""
            print(f"  {mark} {f['label']:<14} {f['path']}{age}")
        for k in e["keys"]:
            print(f"  {'✓' if k['set'] else '·'} key: {k['source']}"
                  f"{'' if k['set'] else ' (not set)'}")


def main() -> int:
    report = build_report()
    if "--json" in sys.argv[1:]:
        print(json.dumps(report, indent=2))
    else:
        _print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
