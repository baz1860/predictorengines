"""In-process command boundary used by the desktop app adapter."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from . import edge as E
from . import model as M
from .schema import DataError, load_bundle, race_cutoff, runner_snapshot


def _race_ids() -> list[str]:
    try:
        bundle = load_bundle()
    except DataError:
        return []
    out = []
    for race in bundle.races.sort_values("scheduled_off_utc").itertuples(index=False):
        row = bundle.races[bundle.races["race_id"].astype(str) == str(race.race_id)].iloc[0]
        if len(runner_snapshot(bundle, str(race.race_id), race_cutoff(row))) >= 2:
            out.append(str(race.race_id))
    return out


def cmd_schema(_p: dict | None = None) -> dict[str, Any]:
    return {
        "kind": "race", "names": _race_ids(), "models": [M.MODEL_NAME],
        "team_label": "Race", "supports_home": False,
        "fitted": M.ARTIFACT_PATH.exists(),
    }


def _race_id(p: dict) -> str:
    rid = p.get("race_id") or p.get("team1") or p.get("home")
    if not rid:
        raise ValueError("Choose a race_id.")
    return str(rid).strip()


def cmd_predict(p: dict) -> dict[str, Any]:
    rid = _race_id(p)
    bundle = load_bundle()
    artifact = M.load_artifact()
    pred = M.predict_race(rid, bundle=bundle, artifact=artifact)
    race = bundle.races[bundle.races["race_id"].astype(str) == rid].iloc[0]
    rows = [{"runner_id": str(r.runner_id), "horse": str(r.horse_name),
             "prob": round(float(r.p_model), 6),
             "fair_odds": round(float(r.fair_odds), 2),
             "quality": str(r.data_quality)} for r in pred.itertuples(index=False)]
    outcomes = [{"label": row["horse"], "prob": row["prob"], "kind": "neutral"}
                for row in rows]
    columns = [
        {"key": "horse", "label": "Horse", "fmt": "text"},
        {"key": "prob", "label": "Win probability", "fmt": "pct1"},
        {"key": "fair_odds", "label": "Fair odds", "fmt": "num"},
        {"key": "quality", "label": "Data", "fmt": "text"},
    ]
    venue = str(race.get("course_name") or race.get("course_id") or "")
    off = pd.Timestamp(race["scheduled_off_utc"]).strftime("%Y-%m-%d %H:%M UTC")
    return {
        "headline": f"{venue} · {off} · {len(rows)} runners",
        "outcomes": outcomes,
        "stats": [{"label": "Race ID", "value": rid},
                  {"label": "Model", "value": artifact["model"]},
                  {"label": "Calibration", "value": f"T={artifact.get('temperature', 1.0):.3f}"}],
        "table": {"title": "Win market", "columns": columns, "rows": rows},
    }


def cmd_edge(p: dict) -> dict[str, Any]:
    rid = _race_id(p) if any(p.get(k) for k in ("race_id", "team1", "home")) else ""
    bundle = load_bundle()
    if not rid:
        candidates = sorted(set(bundle.odds["race_id"].astype(str)))
        if not candidates:
            raise ValueError("No odds rows. Add horse_racing/data/odds.csv or pass race_id.")
        rid = candidates[0]
    artifact = M.load_artifact()
    report = E.price_race(rid, bundle=bundle, artifact=artifact,
                          source=p.get("source"), min_edge=float(p.get("min_edge", E.DEFAULT_EDGE)))
    rows = []
    def number(value, digits):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return round(value, digits) if math.isfinite(value) else None

    for r in report.itertuples(index=False):
        rows.append({
            "event_id": str(r.event_id), "match_date": str(r.match_date),
            "home": str(r.home), "away": "", "runner_id": str(r.runner_id),
            "market": "win", "side": "win", "line": "", "bet": str(r.bet),
            "odds": number(r.odds, 3), "p_model": number(r.p_model, 6),
            "p_market": number(r.p_market, 6), "p_book": number(r.p_book, 6),
            "edge": number(r.edge, 6), "ev_per_unit": number(r.ev_per_unit, 6),
            "kelly_frac": number(r.kelly_frac, 6), "stake_gbp": 0.0,
            "source": str(r.source), "model": artifact["model"],
            "recommended": bool(r.recommended), "board_complete": bool(r.board_complete),
        })
    columns = [
        {"key": "home", "label": "Horse", "fmt": "text"},
        {"key": "odds", "label": "Odds", "fmt": "num"},
        {"key": "p_model", "label": "Model", "fmt": "pct1"},
        {"key": "p_market", "label": "Market", "fmt": "pct1"},
        {"key": "edge", "label": "Edge", "fmt": "signed_pct"},
        {"key": "ev_per_unit", "label": "EV", "fmt": "signed_num"},
    ]
    n = sum(1 for row in rows if row["recommended"])
    complete = bool(rows and rows[0]["board_complete"])
    return {"note": f"{n} analytical edge(s) · race {rid} · "
                    f"board {'complete' if complete else 'incomplete'} · staking disabled",
            "columns": columns, "rows": rows}


COMMANDS = {"schema": cmd_schema, "predict": cmd_predict, "edge": cmd_edge}
