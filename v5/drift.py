"""P2/M2 - drift detection and safe downgrade reports."""
from __future__ import annotations

from typing import Any

import pandas as pd

from app import bankroll_store

from . import registry, store


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def recommendation_drift(engine: str | None = None,
                         min_n: int = 10,
                         edge_shift_threshold: float = 0.04) -> dict[str, Any]:
    """Detect recommendation-distribution and CLV/P&L drift.

    This is intentionally conservative: below `min_n`, the report is "thin data"
    and confidence is reduced rather than producing a false alarm.
    """
    recs = registry.recommendations(engine)
    if recs.empty:
        rep = {"status": "no_data", "alerts": [], "confidence": "low", "n": 0}
        store.write_json(store.DRIFT_REPORT, rep)
        return rep
    recs = recs.copy()
    recs["edge_n"] = _num(recs["edge"])
    recs["created_at"] = pd.to_datetime(recs["created_at"], errors="coerce", utc=True)
    alerts = []
    if len(recs) < min_n:
        alerts.append("thin_recommendation_sample")
    if recs["edge_n"].notna().sum() >= 4:
        ordered = recs.sort_values("created_at")
        half = max(1, len(ordered) // 2)
        early = ordered.head(half)["edge_n"].mean()
        late = ordered.tail(half)["edge_n"].mean()
        if pd.notna(early) and pd.notna(late) and abs(late - early) >= edge_shift_threshold:
            alerts.append("edge_distribution_shift")

    ledger = bankroll_store.load_ledger()
    if engine and not ledger.empty:
        ledger = ledger[ledger["engine"] == engine]
    settled = ledger[ledger["status"].isin(["won", "lost"])] if not ledger.empty else ledger
    clv_alert = None
    if not settled.empty and "closing_odds" in settled.columns:
        close = _num(settled["closing_odds"])
        odds = _num(settled["odds"])
        clv = (odds / close - 1.0).replace([float("inf"), -float("inf")], pd.NA).dropna()
        if len(clv) >= min_n and float(clv.tail(min_n).mean()) < -0.01:
            clv_alert = "negative_recent_clv"
            alerts.append(clv_alert)

    status = "drift" if any(a not in ("thin_recommendation_sample",) for a in alerts) else "ok"
    rep = {
        "generated_at": store.now_iso(),
        "engine": engine or "all",
        "status": status,
        "confidence": "low" if len(recs) < min_n else "normal",
        "n": int(len(recs)),
        "alerts": alerts,
        "actions": ["reduce_confidence", "keep_champion"] if alerts else [],
        "note": "Drift reports are advisory; promotion/demotion still requires explicit gates.",
    }
    store.write_json(store.DRIFT_REPORT, rep)
    return rep


def champion_health(engine: str, market: str = "default") -> dict[str, Any]:
    champ = registry.champion(engine, market)
    drift = recommendation_drift(engine)
    return {
        "engine": engine,
        "market": market,
        "champion": champ,
        "drift": drift,
        "downgrade_recommended": bool(drift["status"] == "drift"),
    }
