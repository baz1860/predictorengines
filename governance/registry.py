"""P2/M1 - model registry, feature snapshots, and recommendation audit trail."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from contracts import fixture_key, market_id

from . import store

RECOMMENDATION_COLS = [
    "recommendation_id", "created_at", "engine", "sport", "event_id",
    "match_date", "home", "away", "market", "side", "line", "bet",
    "odds", "p_model", "p_market", "edge", "ev_per_unit", "stake_gbp",
    "model_version", "feature_version", "source", "status", "reason_codes",
]
FEATURE_COLS = [
    "feature_version", "created_at", "engine", "event_id", "asof",
    "schema_version", "source", "fingerprint", "columns",
]


def _registry() -> dict[str, Any]:
    return store.read_json(store.REGISTRY, {"engines": {}})


def _save_registry(reg: dict[str, Any]) -> None:
    reg["updated_at"] = store.now_iso()
    store.write_json(store.REGISTRY, reg)


def _hash_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def register_model(engine: str, market: str, artifact: str | Path | None = None,
                   metrics: dict[str, Any] | None = None,
                   feature_schema: list[str] | None = None,
                   role: str = "challenger",
                   parent_version: str | None = None) -> dict[str, Any]:
    """Create a versioned model artifact record.

    Registering as champion is allowed only for the first version or when the
    caller has already run an explicit promotion gate through `promote()`.
    """
    reg = _registry()
    eng = reg.setdefault("engines", {}).setdefault(engine, {"models": {}, "champions": {}})
    market = market_id(market) or "default"
    artifact_path = str(artifact or "")
    artifact_hash = ""
    if artifact_path and Path(artifact_path).exists():
        artifact_hash = _hash_obj(Path(artifact_path).read_bytes().hex())
    version = f"{engine}:{market}:{store.now_iso()}:{_hash_obj([artifact_path, metrics, feature_schema])}"
    rec = {
        "version": version,
        "engine": engine,
        "market": market,
        "created_at": store.now_iso(),
        "artifact": artifact_path,
        "artifact_hash": artifact_hash,
        "feature_schema": feature_schema or [],
        "metrics": metrics or {},
        "parent_version": parent_version or "",
        "role": role if role in ("champion", "challenger", "retired") else "challenger",
        "status": "active",
        "promotion_history": [],
    }
    eng["models"][version] = rec
    key = market
    if role == "champion" and not eng["champions"].get(key):
        eng["champions"][key] = version
    elif not eng["champions"].get(key):
        rec["role"] = "champion"
        eng["champions"][key] = version
    _save_registry(reg)
    return rec


def champion(engine: str, market: str = "default") -> dict[str, Any] | None:
    reg = _registry()
    eng = reg.get("engines", {}).get(engine, {})
    version = (eng.get("champions") or {}).get(market_id(market) or "default")
    return (eng.get("models") or {}).get(version) if version else None


def promote(engine: str, market: str, challenger_version: str,
            gate_report: dict[str, Any]) -> dict[str, Any]:
    """Promote a challenger only when the supplied gate report passes."""
    passed = bool(gate_report.get("passed") or gate_report.get("v4_beats_v3"))
    reg = _registry()
    eng = reg.get("engines", {}).get(engine)
    if not eng or challenger_version not in eng.get("models", {}):
        raise ValueError("unknown challenger version")
    if not passed:
        eng["models"][challenger_version]["status"] = "rejected"
        eng["models"][challenger_version]["promotion_history"].append({
            "at": store.now_iso(), "decision": "rejected", "gate_report": gate_report})
        _save_registry(reg)
        return eng["models"][challenger_version]
    key = market_id(market) or "default"
    old = eng.setdefault("champions", {}).get(key)
    if old and old in eng["models"]:
        eng["models"][old]["role"] = "retired"
    eng["champions"][key] = challenger_version
    rec = eng["models"][challenger_version]
    rec["role"] = "champion"
    rec["promotion_history"].append({
        "at": store.now_iso(), "decision": "promoted", "previous": old or "",
        "gate_report": gate_report})
    _save_registry(reg)
    return rec


def feature_snapshot(engine: str, event_id: str, asof: str,
                     features: dict[str, Any], schema_version: Any = "",
                     source: str = "") -> dict[str, Any]:
    """Persist a reconstructable point-in-time feature snapshot fingerprint."""
    cols = sorted(features.keys())
    version = f"feat:{engine}:{event_id}:{asof}:{_hash_obj(features)}"
    row = {
        "feature_version": version, "created_at": store.now_iso(),
        "engine": engine, "event_id": event_id, "asof": asof,
        "schema_version": schema_version, "source": source,
        "fingerprint": _hash_obj(features), "columns": json.dumps(cols),
    }
    store.append_csv(store.FEATURE_SNAPSHOTS, [row], FEATURE_COLS)
    return row


def _recommendation_id(row: dict[str, Any]) -> str:
    return "rec:" + _hash_obj([
        row.get("engine"), row.get("event_id"), row.get("market"),
        row.get("side"), row.get("model_version"), row.get("feature_version"),
        row.get("odds"), row.get("created_at"),
    ])


def record_recommendation(row: dict[str, Any]) -> dict[str, Any]:
    """Append a recommendation record separate from placed-bet settlement."""
    engine = str(row.get("engine", ""))
    market = market_id(row.get("market", "")) or ""
    event_id = str(row.get("event_id") or fixture_key(
        row.get("match_date", ""), row.get("home", ""), row.get("away", "")))
    model_version = str(row.get("model_version") or (
        champion(engine, market or "default") or {}).get("version", "unregistered"))
    rec = {
        "created_at": store.now_iso(),
        "engine": engine,
        "sport": str(row.get("sport", "")),
        "event_id": event_id,
        "match_date": str(row.get("match_date", row.get("date", "")) or ""),
        "home": str(row.get("home", "")),
        "away": str(row.get("away", "")),
        "market": market,
        "side": str(row.get("side", "")),
        "line": str(row.get("line", "")),
        "bet": str(row.get("bet", "")),
        "odds": row.get("odds", ""),
        "p_model": row.get("p_model", ""),
        "p_market": row.get("p_market", row.get("p_book", "")),
        "edge": row.get("edge", ""),
        "ev_per_unit": row.get("ev_per_unit", ""),
        "stake_gbp": row.get("stake_gbp", ""),
        "model_version": model_version,
        "feature_version": str(row.get("feature_version", "")),
        "source": str(row.get("source", "")),
        "status": str(row.get("status", "recommended")),
        "reason_codes": json.dumps(row.get("reason_codes", [])),
    }
    rec["recommendation_id"] = _recommendation_id(rec)
    store.append_csv(store.RECOMMENDATIONS, [rec], RECOMMENDATION_COLS)
    return rec


def recommendations(engine: str | None = None) -> pd.DataFrame:
    df = store.read_csv(store.RECOMMENDATIONS, RECOMMENDATION_COLS)
    if engine:
        df = df[df["engine"] == engine]
    return df


def registry_summary() -> dict[str, Any]:
    reg = _registry()
    recs = recommendations()
    return {
        "registry": reg,
        "recommendations": {
            "n": int(len(recs)),
            "by_engine": recs.groupby("engine").size().to_dict() if not recs.empty else {},
        },
        "feature_snapshots": int(len(store.read_csv(store.FEATURE_SNAPSHOTS, FEATURE_COLS))),
    }
