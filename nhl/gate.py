"""NHL staking gate.

This module is deliberately conservative: pricing can run for analysis, but
recommended bets and stakes stay disabled unless an explicit validation artifact
marks the model as passed and staking-enabled.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
GATE_JSON = DATA_DIR / "validation_gate.json"

DEFAULT_GATE: dict[str, Any] = {
    "status": "FAIL",
    "staking_enabled": False,
    "reason": "NHL model has not passed leak-free validation; staking disabled.",
}


def load_gate(path: str | Path = GATE_JSON) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {**DEFAULT_GATE, "path": str(p)}
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return {
            **DEFAULT_GATE,
            "path": str(p),
            "reason": f"Could not read NHL validation gate: {e}",
        }
    if not isinstance(data, dict):
        return {**DEFAULT_GATE, "path": str(p), "reason": "Validation gate is not a JSON object."}
    gate = {**DEFAULT_GATE, **data, "path": str(p)}
    if gate.get("status") != "PASS":
        gate["staking_enabled"] = False
    return gate


def staking_enabled(gate: dict[str, Any] | None = None) -> bool:
    g = load_gate() if gate is None else gate
    return bool(g.get("staking_enabled") is True and g.get("status") == "PASS")


def apply_staking_gate(rows: list[dict[str, Any]], gate: dict[str, Any] | None = None) -> bool:
    """Return True when rows were forced to no-stake mode."""
    g = load_gate() if gate is None else gate
    if staking_enabled(g):
        return False
    for row in rows:
        row.setdefault("raw_kelly_frac", row.get("kelly_frac", 0.0))
        row.setdefault("raw_stake_gbp", row.get("stake_gbp", 0.0))
        row["kelly_frac"] = 0.0
        row["stake_gbp"] = 0.0
        row["recommended"] = False
    return True
