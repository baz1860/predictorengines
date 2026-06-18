"""P7/M8 - human decision review and override analytics."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from . import registry, store

REVIEW_COLS = [
    "review_id", "created_at", "recommendation_id", "state", "tags",
    "note", "adjusted_stake_gbp",
]
VALID_STATES = {"accepted", "rejected", "watched", "manually_adjusted"}


def add_review(recommendation_id: str, state: str, tags: list[str] | None = None,
               note: str = "", adjusted_stake_gbp: float | None = None) -> dict[str, Any]:
    if state not in VALID_STATES:
        raise ValueError(f"invalid review state: {state}")
    row = {
        "review_id": f"rev:{store.now_iso()}:{recommendation_id}",
        "created_at": store.now_iso(),
        "recommendation_id": recommendation_id,
        "state": state,
        "tags": json.dumps(tags or []),
        "note": note,
        "adjusted_stake_gbp": "" if adjusted_stake_gbp is None else float(adjusted_stake_gbp),
    }
    store.append_csv(store.REVIEWS, [row], REVIEW_COLS)
    return row


def analytics() -> dict[str, Any]:
    rev = store.read_csv(store.REVIEWS, REVIEW_COLS)
    rec = registry.recommendations()
    if rev.empty:
        return {"n": 0, "states": {}, "note": "no human reviews yet"}
    out = {"n": int(len(rev)), "states": rev.groupby("state").size().to_dict()}
    if not rec.empty:
        merged = rev.merge(rec, on="recommendation_id", how="left")
        merged["edge_n"] = pd.to_numeric(merged["edge"], errors="coerce")
        out["avg_edge_by_state"] = {
            k: round(float(v), 4)
            for k, v in merged.groupby("state")["edge_n"].mean().dropna().items()
        }
    out["training_use"] = "excluded_by_default"
    return out
