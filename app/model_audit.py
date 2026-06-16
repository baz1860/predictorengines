"""Per-engine model audit (V3 M8).

Assembles, offline, the compact "is this engine fit to bet?" picture the UI
shows: last validation status, model-params age, data freshness, and which
modelling flags are active. Reads local files only — no network, no engine run.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import provenance

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SUITE = ROOT / "data" / "validation_suite.json"


def _validation(engine: str) -> dict:
    """Status + one-line summary from the last validate_all.py run, if any."""
    if not VALIDATION_SUITE.exists():
        return {"status": "unknown", "summary": "no validation run yet "
                "(run `python3 validate_all.py --gate`)", "generated_at": None}
    try:
        suite = json.loads(VALIDATION_SUITE.read_text())
    except Exception:
        return {"status": "unknown", "summary": "validation summary unreadable",
                "generated_at": None}
    eng = (suite.get("engines") or {}).get(engine)
    if not eng:
        return {"status": "unknown", "summary": "not in last validation run",
                "generated_at": suite.get("generated_at")}
    detail = (eng.get("detail") or "").strip().splitlines()
    summary = next((ln.strip() for ln in reversed(detail) if ln.strip()), "")
    return {"status": eng.get("status", "unknown"), "summary": summary,
            "seconds": eng.get("seconds"), "generated_at": suite.get("generated_at")}


def _params_age_days(engine: str, fresh: list[dict]) -> float | None:
    ages = [f["age_days"] for f in fresh
            if f["role"] == "model" and f["age_days"] is not None]
    return max(ages) if ages else None


def _flags(engine: str) -> list[dict]:
    """Active modelling flags per engine, derived from local state (file
    presence + known defaults). Each: {label, active, note}."""
    out: list[dict] = []
    if engine == "cfb":
        bw = ROOT / "cfb" / "data" / "blend_weight.json"
        w = None
        if bw.exists():
            try:
                w = json.loads(bw.read_text()).get("w_elo")
            except Exception:
                w = None
        out.append({"label": "Elo/power blend weight",
                    "active": w is not None,
                    "note": f"w_elo={w}" if w is not None else "default 0.50 (50/50)"})
        out.append({"label": "Market blend", "active": False,
                    "note": "experimental, off by default"})
    elif engine == "club_soccer":
        calib = ROOT / "club_soccer" / "data" / "calibration.json"
        out.append({"label": "1X2 calibration", "active": calib.exists(),
                    "note": "calibration.json present" if calib.exists() else "not fitted"})
        out.append({"label": "Market blend", "active": False,
                    "note": "experimental, off by default"})
    elif engine == "golf":
        calib = ROOT / "golf" / "data" / "calibration.json"
        out.append({"label": "Calibration", "active": calib.exists(),
                    "note": "calibration.json present" if calib.exists() else "not fitted"})
        out.append({"label": "Market blend", "active": True, "note": "default on"})
    elif engine == "worldcup":
        mb = ROOT / "data" / "market_blend.json"
        out.append({"label": "1X2 market blend", "active": mb.exists(),
                    "note": "market_blend.json present" if mb.exists() else "not fitted"})
    return out


def audit(engine: str) -> dict:
    """Full audit payload for one engine (offline)."""
    fresh = provenance.freshness(engine)
    manifest = provenance.read_manifest(engine)
    return {
        "engine": engine,
        "validation": _validation(engine),
        "freshness": fresh,
        "freshness_warnings": [f["message"] for f in fresh
                               if f["status"] in ("stale", "missing")],
        "params_age_days": _params_age_days(engine, fresh),
        "flags": _flags(engine),
        "manifest_generated_at": (manifest or {}).get("generated_at"),
    }
