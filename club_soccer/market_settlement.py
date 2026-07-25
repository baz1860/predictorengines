"""Shared market identity, closing-price and settlement primitives.

These functions belong to the frozen decision ledger and its settlement path.
The decision-time report consumes their outputs; production settlement must
not depend on a reporting module.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .club_identity import canonical_name

HERE = Path(__file__).resolve().parent
MARKET_HISTORY = HERE / "data" / "market_history.csv"


def devig(odds: dict[str, float]) -> dict[str, float]:
    """Remove proportional overround from a complete decimal-odds market."""
    inv = {k: 1.0 / v for k, v in odds.items() if v and v > 1.0}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()} if total > 0 else {}


def match_key(match_date, home, away) -> str:
    return (
        f"{str(match_date)[:10]}|{canonical_name(home)}|"
        f"{canonical_name(away)}"
    )


def closing_probs(path: Path | None = None) -> tuple[dict, dict]:
    """Return de-vigged Pinnacle closes for 1X2 and OU2.5 by match key."""
    source = MARKET_HISTORY if path is None else Path(path)
    if not source.exists():
        return {}, {}
    market = pd.read_csv(source, low_memory=False)
    one, totals = {}, {}
    has_1x2 = {"psc_h", "psc_d", "psc_a"}.issubset(market.columns)
    has_totals = {"psc_over25", "psc_under25"}.issubset(market.columns)

    def valid(*values) -> bool:
        return all(
            isinstance(value, (int, float)) and value and value > 1
            for value in values
        )

    for row in market.itertuples(index=False):
        key = match_key(row.match_date, row.home, row.away)
        if has_1x2:
            home = getattr(row, "psc_h", None)
            draw = getattr(row, "psc_d", None)
            away = getattr(row, "psc_a", None)
            if valid(home, draw, away):
                one[key] = devig({"home": home, "draw": draw, "away": away})
        if has_totals:
            over = getattr(row, "psc_over25", None)
            under = getattr(row, "psc_under25", None)
            if valid(over, under):
                totals[key] = devig({"over": over, "under": under})
    return one, totals


def clv(market: str, side: str, odds_executed: float,
        close: dict | None) -> float | None:
    """Log closing-line value; None when the market has no valid close."""
    if market not in ("1x2", "total25") or not close:
        return None
    probability = close.get(side)
    if not probability or probability <= 0 or odds_executed <= 1:
        return None
    return float(math.log(odds_executed * probability))


def side_won(market: str, side: str, home_goals: float,
             away_goals: float) -> bool:
    if market == "1x2":
        return (
            (side == "home" and home_goals > away_goals)
            or (side == "away" and away_goals > home_goals)
            or (side == "draw" and home_goals == away_goals)
        )
    total = home_goals + away_goals
    return (
        (side == "over" and total > 2.5)
        or (side == "under" and total < 2.5)
    )
