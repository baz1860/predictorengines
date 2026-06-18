"""Small local stores for V5 governance artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REGISTRY = DATA / "v5_model_registry.json"
FEATURE_SNAPSHOTS = DATA / "v5_feature_snapshots.csv"
RECOMMENDATIONS = DATA / "v5_recommendations.csv"
REVIEWS = DATA / "v5_reviews.csv"
DRIFT_REPORT = DATA / "v5_drift_report.json"
RESEARCH_BACKLOG = DATA / "v5_research_backlog.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default


def write_json(path: Path, data: Any) -> None:
    DATA.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path)
            for c in columns:
                if c not in df.columns:
                    df[c] = ""
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=columns)


def append_csv(path: Path, rows: list[dict], columns: list[str]) -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    old = read_csv(path, columns)
    add = pd.DataFrame(rows, columns=columns)
    out = pd.concat([old, add], ignore_index=True) if not old.empty else add
    out.to_csv(path, index=False)
    return out
