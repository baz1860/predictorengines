"""Competitions whose predictions may be presented to the user.

This is intentionally a presentation policy, not a collection, training, odds,
or evidence policy. The pipeline continues to ingest and score every supported
competition so future evaluation and expansion retain their history.
"""
from __future__ import annotations

from collections.abc import Iterable


SURFACED_COMPETITIONS = (
    # Big five domestic leagues.
    "Premier League",
    "La Liga",
    "Bundesliga",
    "Serie A",
    "Ligue 1",
    # Scotland means the top flight; lower divisions and domestic cups remain
    # shadow/data-only.
    "Scottish Premiership",
    # European continental competitions.
    "Champions League",
    "Europa League",
    "Conference League",
    "UEFA Super Cup",
    # South American continental competitions currently in the registry.
    "Copa Libertadores",
    "Copa Sudamericana",
)
SURFACED_COMPETITION_SET = frozenset(SURFACED_COMPETITIONS)


def is_surfaced(competition: object) -> bool:
    """Whether a normalized competition may appear on prediction surfaces."""
    return str(competition or "").strip() in SURFACED_COMPETITION_SET


def filter_rows(rows: Iterable[dict]) -> list[dict]:
    """Return surfaced rows without mutating or truncating the source rows."""
    return [row for row in rows if is_surfaced(row.get("competition"))]
