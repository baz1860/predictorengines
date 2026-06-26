"""FastAPI backend for Sports Predictor.

Generic, engine-agnostic routes. Each route dispatches to the selected engine
adapter via the registry, so the API surface never grows when engines are added.
Serves the static frontend from app/web.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import csv
import io
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


class TennisMatch(BaseModel):
    round: str = ""
    player_a: str
    player_b: str
    state: str = Field(default="", max_length=12)
    winner: str = Field(default="", max_length=80)
    score: str = Field(default="", max_length=80)
    match_id: str = Field(default="", max_length=40)
    odds_a: float | None = Field(default=None, ge=1.0, le=1000.0)
    odds_b: float | None = Field(default=None, ge=1.0, le=1000.0)


class TennisDrawPayload(BaseModel):
    tour: str = Field(default="atp", pattern=r"^(atp|wta)$")
    tourney_name: str = Field(default="", max_length=80)
    surface: str = Field(default="grass", pattern=r"^(hard|clay|grass|carpet)$")
    best_of: int = Field(default=3, ge=1, le=5)
    matches: list[TennisMatch] = Field(default_factory=list)


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


@app.get("/api/tennis/tournaments")
def tennis_tournaments(tour: str = "atp"):
    """List active tournaments available from ESPN for the given tour."""
    from tennis.providers import _espn_draw
    if tour.lower() not in ("atp", "wta"):
        raise HTTPException(400, "tour must be atp or wta")
    draws = _espn_draw(tour)
    return {"tournaments": [
        {"name": d.tourney_name, "surface": d.surface, "best_of": d.best_of,
         "upcoming": sum(1 for m in d.matches if m.state in ("pre", "in"))}
        for d in draws
    ]}


@app.post("/api/tennis/draw/fetch")
def tennis_draw_fetch(body: dict = None):
    """Fetch the draw for a tournament from ESPN and write draw.csv.
    Body params: tour (atp/wta), tourney_filter (optional name substring).
    """
    body = body or {}
    tour = str(body.get("tour") or "atp").lower()
    tourney_filter = str(body.get("tourney_filter") or "")
    if tour not in ("atp", "wta"):
        raise HTTPException(400, "tour must be atp or wta")

    from tennis.providers import fetch_draw, DATA_DIR as TENNIS_DATA

    draw = fetch_draw(tour=tour, tourney_filter=tourney_filter)
    if draw is None:
        raise HTTPException(503, "Could not fetch draw from ESPN — check network connection")

    draw_csv = TENNIS_DATA / "draw.csv"
    odds_csv = TENNIS_DATA / "odds.csv"

    # Write draw.csv (completed, live, and upcoming matches)
    draw_buf = io.StringIO()
    w = csv.writer(draw_buf)
    w.writerow(["tour", "tourney_name", "surface", "best_of", "round",
                "player_a", "player_b", "state", "winner", "score", "match_id"])
    for m in draw.matches:
        w.writerow([draw.tour, draw.tourney_name, draw.surface,
                    draw.best_of, m.round, m.player_a, m.player_b,
                    m.state, m.winner, m.score, m.match_id])
    draw_csv.write_text(draw_buf.getvalue())

    # Preserve existing odds for any players already in odds.csv
    existing_odds: dict[tuple, dict] = {}
    if odds_csv.exists():
        with open(odds_csv, newline="") as f:
            for row in csv.DictReader(f):
                pa, pb = row.get("player_a","").strip(), row.get("player_b","").strip()
                if pa and pb:
                    existing_odds[(pa, pb)] = row

    # Write updated odds.csv (keep existing, add blank rows for new matches)
    odds_buf = io.StringIO()
    w2 = csv.writer(odds_buf)
    w2.writerow(["tour", "surface", "best_of", "player_a", "player_b", "odds_a", "odds_b"])
    for m in draw.matches:
        key = (m.player_a, m.player_b)
        if "TBD" in (m.player_a.upper(), m.player_b.upper()):
            continue
        if key in existing_odds:
            r = existing_odds[key]
            w2.writerow([draw.tour, draw.surface, draw.best_of,
                         m.player_a, m.player_b,
                         r.get("odds_a",""), r.get("odds_b","")])
        elif m.state in ("pre", "in"):
            w2.writerow([draw.tour, draw.surface, draw.best_of,
                         m.player_a, m.player_b, "", ""])
    odds_csv.write_text(odds_buf.getvalue())

    return {
        "tourney_name": draw.tourney_name,
        "tour": draw.tour,
        "surface": draw.surface,
        "best_of": draw.best_of,
        "matches": [
            {"round": m.round, "player_a": m.player_a, "player_b": m.player_b,
             "odds_a": None, "odds_b": None, "state": m.state,
             "winner": m.winner, "score": m.score, "match_id": m.match_id}
            for m in draw.matches
        ],
    }


@app.get("/api/tennis/draw")
def tennis_draw_get():
    """Return the current tennis draw + odds as a combined JSON payload."""
    from tennis.providers import DATA_DIR as TENNIS_DATA
    draw_csv = TENNIS_DATA / "draw.csv"
    odds_csv = TENNIS_DATA / "odds.csv"

    # Read tournament-level fields from draw.csv
    tour, tourney_name, surface, best_of = "atp", "", "grass", 3
    draw_rows: dict[tuple, dict] = {}
    if draw_csv.exists():
        with open(draw_csv, newline="") as f:
            for row in csv.DictReader(f):
                pa, pb = (row.get("player_a") or "").strip(), (row.get("player_b") or "").strip()
                if not pa or not pb:
                    continue
                tour = (row.get("tour") or tour).lower()
                tourney_name = row.get("tourney_name") or tourney_name
                surface = (row.get("surface") or surface).lower()
                try:
                    best_of = int(float(row.get("best_of") or best_of))
                except (ValueError, TypeError):
                    pass
                key = (pa, pb)
                draw_rows[key] = {"round": row.get("round") or "", "player_a": pa, "player_b": pb,
                                  "odds_a": None, "odds_b": None,
                                  "state": row.get("state") or "",
                                  "winner": row.get("winner") or "",
                                  "score": row.get("score") or "",
                                  "match_id": row.get("match_id") or ""}

    # Merge odds
    if odds_csv.exists():
        with open(odds_csv, newline="") as f:
            for row in csv.DictReader(f):
                pa, pb = (row.get("player_a") or "").strip(), (row.get("player_b") or "").strip()
                if not pa or not pb:
                    continue
                key = (pa, pb)
                if key not in draw_rows:
                    draw_rows[key] = {"round": "", "player_a": pa, "player_b": pb,
                                      "odds_a": None, "odds_b": None,
                                      "state": "", "winner": "",
                                      "score": "", "match_id": ""}
                try:
                    draw_rows[key]["odds_a"] = float(row["odds_a"]) if row.get("odds_a") else None
                    draw_rows[key]["odds_b"] = float(row["odds_b"]) if row.get("odds_b") else None
                except (ValueError, KeyError):
                    pass

    return {
        "tour": tour, "tourney_name": tourney_name,
        "surface": surface, "best_of": best_of,
        "matches": list(draw_rows.values()),
    }


@app.post("/api/tennis/draw")
def tennis_draw_post(payload: TennisDrawPayload):
    """Write draw.csv and odds.csv from the UI draw editor."""
    from tennis.providers import DATA_DIR as TENNIS_DATA

    draw_csv = TENNIS_DATA / "draw.csv"
    odds_csv = TENNIS_DATA / "odds.csv"

    # Write draw.csv
    draw_buf = io.StringIO()
    w = csv.writer(draw_buf)
    w.writerow(["tour", "tourney_name", "surface", "best_of", "round",
                "player_a", "player_b", "state", "winner", "score", "match_id"])
    for m in payload.matches:
        pa, pb = m.player_a.strip(), m.player_b.strip()
        if pa and pb:
            w.writerow([payload.tour, payload.tourney_name, payload.surface,
                        payload.best_of, m.round, pa, pb, m.state,
                        m.winner.strip(), m.score, m.match_id])
    draw_csv.write_text(draw_buf.getvalue())

    # Write odds.csv (only rows where at least one odds value provided)
    odds_buf = io.StringIO()
    w2 = csv.writer(odds_buf)
    w2.writerow(["tour", "surface", "best_of", "player_a", "player_b", "odds_a", "odds_b"])
    for m in payload.matches:
        pa, pb = m.player_a.strip(), m.player_b.strip()
        if pa and pb and (m.odds_a is not None or m.odds_b is not None):
            w2.writerow([payload.tour, payload.surface, payload.best_of,
                         pa, pb,
                         round(m.odds_a, 3) if m.odds_a is not None else "",
                         round(m.odds_b, 3) if m.odds_b is not None else ""])
    odds_csv.write_text(odds_buf.getvalue())

    return {"saved": True, "matches": len(payload.matches),
            "odds_rows": sum(1 for m in payload.matches
                             if m.odds_a is not None or m.odds_b is not None)}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


# Static assets (app.js, style.css). Mounted last so /api/* wins.
app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
