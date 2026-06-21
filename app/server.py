"""FastAPI backend for Sports Predictor.

Generic, engine-agnostic routes. Each route dispatches to the selected engine
adapter via the registry, so the API surface never grows when engines are added.
Serves the static frontend from app/web.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import json
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import bankroll_store, dashboard_data, model_audit, settings_store
from .engines import registry
from v5 import drift as v5_drift
from v5 import live as v5_live
from v5 import portfolio as v5_portfolio
from v5 import registry as v5_registry
from v5 import report as v5_report
from v5 import research as v5_research
from v5 import review as v5_review
from v5 import scenario as v5_scenario
from v6 import operations as v6_operations
from v6 import runner as v6_runner
from v6 import scheduler as v6_scheduler

WEB_DIR = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Start the in-app update scheduler when the server comes up.
    v6_scheduler.start_scheduler()
    yield


app = FastAPI(title="Sports Predictor", lifespan=lifespan)

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
    "round": (1, 4),
    "round_no": (1, 4),
    "season": (2000, 2100),
    "min_rounds": (1, 500),
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


class V5Recommendation(BaseModel):
    row: dict


class V5Review(BaseModel):
    recommendation_id: str
    state: str
    tags: list[str] = Field(default_factory=list)
    note: str = ""
    adjusted_stake_gbp: float | None = Field(default=None, ge=0.0, le=1_000_000.0)


class V5Scenario(BaseModel):
    home: str
    away: str
    asof: str
    home_elo_delta: float = Field(default=0.0, ge=-400.0, le=400.0)
    away_elo_delta: float = Field(default=0.0, ge=-400.0, le=400.0)
    market_move: dict | None = None


class V5LiveSoccer(BaseModel):
    home: str
    away: str
    asof: str
    minute: int = Field(ge=0, le=130)
    home_score: int = Field(ge=0, le=20)
    away_score: int = Field(ge=0, le=20)
    red_cards_home: int = Field(default=0, ge=0, le=5)
    red_cards_away: int = Field(default=0, ge=0, le=5)
    state_fetched_at: str | None = None


class V6Backup(BaseModel):
    label: str | None = Field(default=None, max_length=40)


class V6Run(BaseModel):
    mode: str

    @field_validator("mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        if v not in v6_runner.MODES:
            raise ValueError(f"unknown update mode: {v}")
        return v


class V6ScheduleEntry(BaseModel):
    id: str | None = None
    mode: str
    time: str = Field(pattern=r"^\d{2}:\d{2}$")
    enabled: bool = True


class V6Schedule(BaseModel):
    entries: list[V6ScheduleEntry] = Field(default_factory=list)


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


@app.post("/api/refresh")
def refresh(req: EngineRequest):
    return _dispatch(req.engine, "refresh", req.params)


@app.post("/api/edge")
def edge(req: EngineRequest):
    return _dispatch(req.engine, "edge", req.params)


@app.post("/api/round-3balls")
def round_3balls(req: EngineRequest):
    return _dispatch(req.engine, "round_3balls", req.params)


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


@app.get("/api/v5")
def v5_summary():
    return v5_report.build()


@app.get("/api/v5/registry")
def v5_registry_summary():
    return v5_registry.registry_summary()


@app.post("/api/v5/recommendation")
def v5_record_recommendation(req: V5Recommendation):
    return v5_registry.record_recommendation(req.row)


@app.get("/api/v5/drift")
def v5_drift_report(engine: str | None = None):
    return v5_drift.recommendation_drift(engine)


@app.get("/api/v5/portfolio")
def v5_portfolio_report(engine: str | None = None):
    return v5_portfolio.optimize_from_recommendations(engine)


@app.post("/api/v5/scenario/worldcup")
def v5_worldcup_scenario(req: V5Scenario):
    return v5_scenario.worldcup_line_lab(**req.model_dump())


@app.post("/api/v5/live/soccer")
def v5_live_soccer(req: V5LiveSoccer):
    return v5_live.soccer_live_1x2(**req.model_dump())


@app.post("/api/v5/review")
def v5_add_review(req: V5Review):
    try:
        return v5_review.add_review(**req.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/v5/review")
def v5_review_analytics():
    return v5_review.analytics()


@app.get("/api/v5/research")
def v5_research_backlog(engine: str | None = None):
    return v5_research.generate_backlog(engine)


@app.get("/api/v6")
def v6_report():
    return v6_operations.report()


@app.get("/api/v6/health")
def v6_health():
    return v6_operations.health()


@app.get("/api/v6/daily-run")
def v6_daily_run():
    return v6_operations.daily_run_plan()


@app.post("/api/v6/backup")
def v6_backup(req: V6Backup):
    return v6_operations.create_backup(req.label)


@app.get("/api/v6/release")
def v6_release():
    return v6_operations.release_status()


@app.get("/api/v6/run/modes")
def v6_run_modes():
    """The update flows the UI can launch (id, label, description)."""
    return {"modes": v6_runner.modes()}


@app.get("/api/v6/run")
def v6_run_status(since: int = 0):
    """Current/last update run: status, per-step progress, and any log lines
    after `since` (use the returned next_offset to poll incrementally)."""
    return v6_runner.status(since=max(since, 0))


@app.post("/api/v6/run")
def v6_run_start(req: V6Run):
    """Launch an update flow (one at a time). 409 if a run is in progress."""
    try:
        return v6_runner.start(req.mode, trigger="manual")
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/v6/run/history")
def v6_run_history():
    return {"runs": v6_runner.history()}


@app.get("/api/v6/schedule")
def v6_get_schedule():
    return v6_scheduler.get_schedule()


@app.post("/api/v6/schedule")
def v6_save_schedule(req: V6Schedule):
    return v6_scheduler.save_schedule([e.model_dump() for e in req.entries])


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


# Static assets (app.js, style.css). Mounted last so /api/* wins.
app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
