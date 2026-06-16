"""FastAPI backend for Sports Predictor.

Generic, engine-agnostic routes. Each route dispatches to the selected engine
adapter via the registry, so the API surface never grows when engines are added.
Serves the static frontend from app/web.
"""
from __future__ import annotations

from pathlib import Path

import json
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import bankroll_store, dashboard_data, model_audit, settings_store
from .engines import registry

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Sports Predictor")

# --- API input bounds (V3 M2) ---------------------------------------------------
# Engine slugs are short identifiers, never paths. Anything outside this shape is
# rejected at the boundary, so path-traversal / injection never reaches the
# registry lookup or the filesystem.
_SLUG_RE = re.compile(r"^[a-z0-9_-]{1,40}$")
# A params object is small structured config, not a payload. Cap its serialized
# size so a request can't balloon memory before an engine ever runs.
_MAX_PARAMS_BYTES = 50_000
# Numeric params get clamped to sane ranges regardless of what the client sends.
_PARAM_CLAMPS: dict[str, tuple[float, float]] = {
    "sims": (1, 200_000),
    "seed": (0, 2**31 - 1),
    "kelly": (0.0, 1.0),
    "min_edge": (0.0, 1.0),
    "cut_rule": (1, 200),
}


class EngineRequest(BaseModel):
    engine: str
    params: dict = Field(default_factory=dict)

    @field_validator("engine")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _SLUG_RE.match(v or ""):
            raise ValueError("invalid engine id")
        return v

    @field_validator("params")
    @classmethod
    def _bounded(cls, v: dict) -> dict:
        if not isinstance(v, dict):
            raise ValueError("params must be an object")
        if len(json.dumps(v, default=str)) > _MAX_PARAMS_BYTES:
            raise ValueError("params too large")
        for key, (lo, hi) in _PARAM_CLAMPS.items():
            if key in v and isinstance(v[key], (int, float)) and not isinstance(v[key], bool):
                v[key] = min(max(v[key], lo), hi)
        return v


class BankrollAction(BaseModel):
    action: str            # "status" | "settle" | "reset"
    amount: float | None = Field(default=None, ge=0.0, le=1_000_000.0)


class SettingsPatch(BaseModel):
    odds_api_keys: dict | None = None
    default_kelly: float | None = Field(default=None, ge=0.0, le=1.0)
    default_model: str | None = None


def _dispatch(engine_id: str, cap: str, params: dict):
    try:
        engine = registry.get(engine_id)
    except KeyError:
        raise HTTPException(404, f"Unknown engine: {engine_id}")
    if cap not in engine.capabilities:
        raise HTTPException(400, f"{engine.name} does not support {cap}")
    try:
        return getattr(engine, cap)(params)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/dashboard")
def dashboard():
    """Live suite-level dashboard payload (KPIs, bankroll curve, CLV,
    calibration, fixtures, bet queue, title odds). Reads local files only."""
    return dashboard_data.build_dashboard()


@app.get("/api/history")
def history():
    """Full bet-history explorer: ledger rows, cumulative P&L, by-market/by-sport
    aggregates, and filter options."""
    return dashboard_data.build_history()


@app.get("/api/fixtures")
def fixtures():
    """Upcoming predicted fixtures grouped by date, with queued edge picks."""
    return dashboard_data.build_fixtures()


@app.get("/api/outrights")
def outrights():
    """Title-race: current champion % plus movement history per team."""
    return dashboard_data.build_outrights()


@app.get("/api/engines")
def list_engines():
    """Sidebar data: every engine and what it can do."""
    return {"engines": [e.info() for e in registry.all()]}


@app.get("/api/engines/{engine_id}")
def engine_info(engine_id: str):
    try:
        return registry.get(engine_id).info()
    except KeyError:
        raise HTTPException(404, f"Unknown engine: {engine_id}")


@app.get("/api/engines/{engine_id}/audit")
def engine_audit(engine_id: str):
    """Model-audit panel: validation status, params age, data freshness, flags.
    Offline — reads local files only."""
    try:
        registry.get(engine_id)
    except KeyError:
        raise HTTPException(404, f"Unknown engine: {engine_id}")
    return model_audit.audit(engine_id)


@app.post("/api/predict")
def predict(req: EngineRequest):
    return _dispatch(req.engine, "predict", req.params)


@app.post("/api/simulate")
def simulate(req: EngineRequest):
    return _dispatch(req.engine, "simulate", req.params)


@app.post("/api/edge")
def edge(req: EngineRequest):
    return _dispatch(req.engine, "edge", req.params)


@app.post("/api/edge/template")
def edge_template(req: EngineRequest):
    try:
        engine = registry.get(req.engine)
    except KeyError:
        raise HTTPException(404, f"Unknown engine: {req.engine}")
    if not hasattr(engine, "write_odds_template"):
        raise HTTPException(400, "Engine has no odds template")
    return engine.write_odds_template()


@app.post("/api/bankroll")
def bankroll(req: BankrollAction):
    if req.action == "status":
        return bankroll_store.status_summary()
    if req.action == "settle":
        result = bankroll_store.settle(registry)
        return {"result": result, **bankroll_store.status_summary()}
    if req.action == "settle_preview":
        # Dry run: report what *would* settle without writing the ledger.
        return {"result": bankroll_store.settle(registry, dry_run=True),
                **bankroll_store.status_summary()}
    if req.action == "reset":
        if req.amount is None:
            raise HTTPException(422, "reset needs an amount")
        bankroll_store.reset(req.amount)
        return bankroll_store.status_summary()
    raise HTTPException(422, f"Unknown action: {req.action}")


@app.get("/api/settings")
def get_settings():
    return settings_store.public_view()


@app.post("/api/settings")
def post_settings(patch: SettingsPatch):
    settings_store.save({k: v for k, v in patch.model_dump().items() if v is not None})
    return settings_store.public_view()


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


# Static assets (app.js, style.css). Mounted last so /api/* wins.
app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
