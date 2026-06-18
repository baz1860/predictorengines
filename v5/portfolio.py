"""P3/M3 - cross-sport portfolio risk allocation."""
from __future__ import annotations

from typing import Any

import pandas as pd

from app import bankroll_store

HARD_CAPS = {
    "per_event": 0.03,
    "per_engine": 0.10,
    "per_day": 0.20,
    "per_market": 0.08,
}


def _team_set(row: dict[str, Any]) -> set[str]:
    h, a = str(row.get("home", "")), str(row.get("away", ""))
    if not a or "OUTRIGHT" in a.upper():
        return {h} if h else set()
    return {h, a}


def optimize(candidates: list[dict[str, Any]], bankroll: float | None = None,
             caps: dict[str, float] | None = None) -> dict[str, Any]:
    """Allocate stake suggestions under hard risk caps.

    Expected return ordering is secondary to caps: candidates are sorted by EV,
    but stake is clipped by event, engine, market, day, and team exposure.
    """
    bankroll = float(bankroll if bankroll is not None else bankroll_store.current_bankroll())
    caps = {**HARD_CAPS, **(caps or {})}
    rows = []
    used_event: dict[str, float] = {}
    used_engine: dict[str, float] = {}
    used_market: dict[str, float] = {}
    used_team: dict[str, float] = {}
    used_day = 0.0
    ordered = sorted(candidates, key=lambda r: float(r.get("ev_per_unit") or r.get("edge") or 0), reverse=True)
    for r in ordered:
        requested = float(r.get("stake_gbp") or r.get("stake") or 0.0)
        if requested <= 0:
            continue
        event = str(r.get("event_id", ""))
        engine = str(r.get("engine", ""))
        market = str(r.get("market", ""))
        teams = _team_set(r)
        headroom = [
            caps["per_event"] * bankroll - used_event.get(event, 0.0),
            caps["per_engine"] * bankroll - used_engine.get(engine, 0.0),
            caps["per_market"] * bankroll - used_market.get(market, 0.0),
            caps["per_day"] * bankroll - used_day,
        ]
        for t in teams:
            headroom.append(caps["per_event"] * bankroll - used_team.get(t, 0.0))
        stake = max(0.0, min(requested, *headroom))
        stake = round(stake, 2)
        out = dict(r)
        out["requested_stake_gbp"] = round(requested, 2)
        out["allocated_stake_gbp"] = stake
        out["marginal_risk_gbp"] = stake
        out["risk_limited"] = stake < requested
        rows.append(out)
        used_event[event] = used_event.get(event, 0.0) + stake
        used_engine[engine] = used_engine.get(engine, 0.0) + stake
        used_market[market] = used_market.get(market, 0.0) + stake
        used_day += stake
        for t in teams:
            used_team[t] = used_team.get(t, 0.0) + stake
    return {
        "bankroll": bankroll,
        "caps": caps,
        "rows": rows,
        "exposure": {
            "event": used_event, "engine": used_engine,
            "market": used_market, "team": used_team,
            "day": round(used_day, 2),
        },
        "status": "advisory",
    }


def optimize_from_recommendations(engine: str | None = None) -> dict[str, Any]:
    from .registry import recommendations
    df = recommendations(engine)
    if df.empty:
        return optimize([])
    rows = df.to_dict(orient="records")
    return optimize(rows)
