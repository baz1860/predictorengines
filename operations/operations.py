"""V6 operator health, daily-run planning, backups, and release status."""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import bankroll_store, provenance
from governance import registry as governance_registry
from governance import store as governance_store

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BACKUP_DIR = DATA / "backups"

CRITICAL_FILES = [
    "data/suite_bankroll.json",
    "data/suite_ledger.csv",
    "data/bankroll.json",
    "data/ledger.csv",
    "data/app_settings.json",
    "data/validation_suite.json",
    "data/wc_v4_validation.json",
    "data/v5_model_registry.json",
    "data/v5_feature_snapshots.csv",
    "data/v5_recommendations.csv",
    "data/v5_reviews.csv",
    "data/v5_drift_report.json",
    "data/v5_research_backlog.json",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default


def validation_status() -> dict[str, Any]:
    path = DATA / "validation_suite.json"
    suite = _read_json(path, {})
    engines = suite.get("engines") or {}
    rows = []
    worst = "unknown"
    order = {"PASS": 0, "unknown": 1, "FAIL": 2, "ERROR": 3}
    for engine in ("worldcup", "club_soccer", "cfb", "golf"):
        rec = engines.get(engine, {})
        status = rec.get("status", "unknown")
        if order.get(status, 1) > order.get(worst, 1):
            worst = status
        rows.append({
            "engine": engine,
            "status": status,
            "seconds": rec.get("seconds"),
            "detail": rec.get("detail", ""),
        })
    return {
        "status": "ok" if worst == "PASS" else ("fail" if worst in ("FAIL", "ERROR") else "unknown"),
        "generated_at": suite.get("generated_at"),
        "rows": rows,
        "path": str(path.relative_to(ROOT)),
    }


def freshness_status() -> dict[str, Any]:
    rows = []
    counts = {"ok": 0, "stale": 0, "missing": 0}
    for engine in provenance.ENGINE_INPUTS:
        for rec in provenance.freshness(engine):
            row = {"engine": engine, **rec}
            rows.append(row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
    status = "fail" if counts["missing"] else ("warn" if counts["stale"] else "ok")
    return {"status": status, "counts": counts, "rows": rows}


def bankroll_status() -> dict[str, Any]:
    s = bankroll_store.status_summary()
    br = float(s.get("bankroll", 0.0) or 0.0)
    open_stake = float((s.get("totals") or {}).get("open_stake", 0.0) or 0.0)
    ratio = open_stake / br if br else 0.0
    return {
        "status": "warn" if ratio > 0.25 else "ok",
        "bankroll": br,
        "open_stake": open_stake,
        "open_risk_ratio": round(ratio, 4),
        "open_count": int((s.get("totals") or {}).get("open_count", 0)),
    }


def v5_status() -> dict[str, Any]:
    summary = governance_registry.registry_summary()
    drift = _read_json(governance_store.DRIFT_REPORT, {"status": "not_run", "alerts": []})
    return {
        "status": "warn" if drift.get("status") == "drift" else "ok",
        "recommendations": summary.get("recommendations", {}),
        "feature_snapshots": summary.get("feature_snapshots", 0),
        "registry_engines": sorted((summary.get("registry", {}).get("engines") or {}).keys()),
        "drift": drift,
    }


def backup_status() -> dict[str, Any]:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backups = sorted(BACKUP_DIR.glob("sports_predictor_backup_*.zip"))
    latest = backups[-1] if backups else None
    return {
        "status": "ok" if latest else "warn",
        "count": len(backups),
        "latest": str(latest.relative_to(ROOT)) if latest else None,
    }


def health() -> dict[str, Any]:
    sections = {
        "validation": validation_status(),
        "freshness": freshness_status(),
        "bankroll": bankroll_status(),
        "v5": v5_status(),
        "backup": backup_status(),
    }
    severity = {"ok": 0, "unknown": 1, "warn": 2, "fail": 3, "not_run": 1}
    worst = max((s.get("status", "unknown") for s in sections.values()),
                key=lambda x: severity.get(x, 1))
    return {"generated_at": _now(), "status": worst, "sections": sections}


def daily_run_plan() -> dict[str, Any]:
    steps = [
        {"id": "manifests", "command": "python3 -m app.provenance --write",
         "safe": True, "description": "Write data manifests for every engine."},
        {"id": "clv_snapshot", "command": "python3 clv_suite.py --snapshot",
         "safe": True, "description": "Snapshot current odds for open bets."},
        {"id": "validate", "command": "python3 validate_all.py --gate",
         "safe": True, "description": "Run all engine validation gates."},
        {"id": "governance_report", "command": "python3 -m governance.report",
         "safe": True, "description": "Refresh governance drift/research summary."},
        {"id": "backup", "command": "POST /api/v6/backup",
         "safe": True, "description": "Create a local backup zip of critical artifacts."},
    ]
    return {
        "mode": "preview",
        "status": "advisory",
        "steps": steps,
        "note": "V6 starts with dry-run planning. Execution is intentionally explicit.",
    }


def create_backup(label: str | None = None) -> dict[str, Any]:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(ch for ch in (label or "manual") if ch.isalnum() or ch in ("-", "_"))[:40]
    path = BACKUP_DIR / f"sports_predictor_backup_{stamp}_{safe_label}.zip"
    included = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in CRITICAL_FILES:
            p = ROOT / rel
            if p.exists() and p.is_file():
                zf.write(p, arcname=rel)
                included.append(rel)
    manifest = {
        "created_at": _now(),
        "label": safe_label,
        "included": included,
        "missing": [rel for rel in CRITICAL_FILES if not (ROOT / rel).exists()],
    }
    try:
        zf_path = str(path.relative_to(ROOT))
    except ValueError:
        zf_path = str(path)
    return {"status": "ok", "path": zf_path, "manifest": manifest, "files": len(included)}


def release_status() -> dict[str, Any]:
    artifacts = {
        "v3_suite_ledger": DATA / "suite_ledger.csv",
        "v4_validation": DATA / "wc_v4_validation.json",
        "v5_registry": DATA / "v5_model_registry.json",
        "v5_recommendations": DATA / "v5_recommendations.csv",
        "v6_plan": ROOT / "V6_PLAN.md",
    }
    rows = []
    for key, path in artifacts.items():
        rows.append({
            "artifact": key,
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
        })
    return {
        "version": "6.0",
        "generated_at": _now(),
        "artifacts": rows,
        "ready": all(r["exists"] for r in rows if r["artifact"] != "v5_recommendations"),
    }


def report() -> dict[str, Any]:
    return {
        "health": health(),
        "daily_run": daily_run_plan(),
        "release": release_status(),
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(report(), indent=2))
