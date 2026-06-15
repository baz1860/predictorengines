"""Competition registry for the Club Soccer engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Competition:
    name: str
    country: str
    kind: str
    tier: int
    api_id: int
    strength: float


COMPETITIONS = [
    Competition("Premier League", "England", "league", 1, 39, 1.00),
    Competition("Championship", "England", "league", 2, 40, 0.72),
    Competition("League One", "England", "league", 3, 41, 0.50),
    Competition("League Two", "England", "league", 4, 42, 0.35),
    Competition("FA Cup", "England", "cup", 0, 45, 0.75),
    Competition("EFL Cup", "England", "cup", 0, 48, 0.70),
    Competition("Scottish Premiership", "Scotland", "league", 1, 179, 0.58),
    Competition("Scottish Championship", "Scotland", "league", 2, 180, 0.38),
    Competition("Scottish League One", "Scotland", "league", 3, 181, 0.28),
    Competition("Scottish League Two", "Scotland", "league", 4, 182, 0.22),
    Competition("Scottish Cup", "Scotland", "cup", 0, 183, 0.48),
    Competition("Scottish League Cup", "Scotland", "cup", 0, 184, 0.45),
    Competition("Bundesliga", "Germany", "league", 1, 78, 0.93),
    Competition("DFB-Pokal", "Germany", "cup", 0, 81, 0.72),
    Competition("Serie A", "Italy", "league", 1, 135, 0.91),
    Competition("Coppa Italia", "Italy", "cup", 0, 137, 0.70),
    Competition("Ligue 1", "France", "league", 1, 61, 0.86),
    Competition("Coupe de France", "France", "cup", 0, 66, 0.64),
    Competition("La Liga", "Spain", "league", 1, 140, 0.92),
    Competition("Copa del Rey", "Spain", "cup", 0, 143, 0.70),
    Competition("Champions League", "Europe", "europe", 0, 2, 1.08),
    Competition("Europa League", "Europe", "europe", 0, 3, 0.88),
    Competition("Conference League", "Europe", "europe", 0, 848, 0.70),
    Competition("UEFA Super Cup", "Europe", "europe", 0, 531, 1.00),
]

BY_NAME = {c.name: c for c in COMPETITIONS}
BY_API_ID = {c.api_id: c for c in COMPETITIONS}


def names() -> list[str]:
    return [c.name for c in COMPETITIONS]


def public_rows() -> list[dict]:
    rows = []
    for c in COMPETITIONS:
        row = asdict(c)
        row["api_football_id"] = c.api_id
        rows.append(row)
    return rows


def get(name: str | None) -> Competition | None:
    if not name:
        return None
    return BY_NAME.get(str(name).strip())


def strength(name: str | None) -> float:
    c = get(name)
    return c.strength if c else 0.75
