"""Per-market CFB recommendation policy.

Forecasts and diagnostic edges may be shown for every market, but only an
explicit ``eligible`` status can create a stake or recorded bet. Policy is kept
outside fitted model parameters so evidence review can promote/demote a market
without pretending the model itself changed.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY_JSON = HERE / "data" / "market_policy.json"
VALID = {"disabled", "diagnostic", "paper", "eligible"}
DEFAULT = {"ml": "diagnostic", "spread": "diagnostic", "total": "paper"}


def load_policy(path: str | Path = POLICY_JSON) -> dict[str, str]:
    policy = dict(DEFAULT)
    try:
        raw = json.loads(Path(path).read_text())
        for market in policy:
            status = str(raw.get(market, policy[market])).strip().lower()
            if status in VALID:
                policy[market] = status
    except Exception:
        pass
    return policy


def status(market: str, policy: dict[str, str] | None = None) -> str:
    return (policy or load_policy()).get(str(market).lower(), "disabled")


def recordable(market: str, policy: dict[str, str] | None = None) -> bool:
    return status(market, policy) == "eligible"

