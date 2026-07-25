"""Shared tennis round vocabulary and draw-schema helpers."""
from __future__ import annotations

DRAW_COLUMNS = [
    "tour", "tourney_name", "event_id", "surface", "best_of", "round",
    "player_a", "player_b", "state", "winner", "score", "match_id",
]

# Shallowest (title) to deepest main-draw round.
ROUND_SEQ = ["F", "SF", "QF", "R16", "R32", "R64", "R128", "R256"]
ROUND_RANK = {round_name: i for i, round_name in enumerate(ROUND_SEQ)}

# Feed/display order, earliest round first.
ROUND_ORDER = [
    "Q1", "Q2", "QF-Q", "R256", "R128", "R64", "R32", "R16",
    "R1", "R2", "R3", "R4", "QF", "SF", "F",
]
FIELD_ROUND = {
    2: "F", 4: "SF", 8: "QF", 16: "R16", 32: "R32",
    64: "R64", 128: "R128", 256: "R256",
}
ROUND_LABEL = {
    "R256": "Round of 256", "R128": "Round of 128", "R64": "Round of 64",
    "R32": "Round of 32", "R16": "Round of 16", "QF": "Quarterfinals",
    "SF": "Semifinals", "F": "Final", "Q1": "Qualifying R1",
    "Q2": "Qualifying R2", "QF-Q": "Qualifying final",
    "R1": "Round 1", "R2": "Round 2", "R3": "Round 3", "R4": "Round 4",
}

QUALIFYING_ROUNDS = {"Q1", "Q2", "QF-Q"}

ESPN_ROUND_MAP = {
    "Qualifying 1st Round": "Q1",
    "Qualifying 2nd Round": "Q2",
    "Qualifying Final": "QF-Q",
    "Round 1": "R1",
    "Round 2": "R2",
    "Round 3": "R3",
    "Round 4": "R4",
    "Round of 128": "R128",
    "Round of 64": "R64",
    "Round of 32": "R32",
    "Round of 16": "R16",
    "Quarterfinal": "QF",
    "Semifinal": "SF",
    "Final": "F",
}
ESPN_MAJOR_ROUND_MAP = {
    "Round 1": "R128", "Round 2": "R64", "Round 3": "R32", "Round 4": "R16",
}


def is_tbd(name: str) -> bool:
    return not str(name or "").strip() or str(name).strip().upper() in {"TBD", "BYE"}


def canonical_round(round_name: str, field_size: int | None = None) -> str:
    """Return a main-draw round label, inferring generic R1/R2/R3 by field size."""
    round_name = str(round_name or "").strip()
    if round_name in ROUND_RANK or round_name in QUALIFYING_ROUNDS:
        return round_name
    if field_size:
        inferred = FIELD_ROUND.get(int(field_size))
        if inferred:
            return inferred
    return round_name


def round_rank(round_name: str, field_size: int | None = None) -> int:
    return ROUND_RANK.get(canonical_round(round_name, field_size), 99)


def round_sort_key(round_name: str, first_index: int = 0) -> tuple[int, int]:
    try:
        return (ROUND_ORDER.index(str(round_name or "")), first_index)
    except ValueError:
        return (len(ROUND_ORDER), first_index)


def ordered_rounds(round_names) -> list[str]:
    return sorted(round_names, key=lambda name: round_sort_key(name)[0])


def next_deeper(round_name: str) -> str | None:
    rank = ROUND_RANK.get(round_name)
    if rank is None or rank + 1 >= len(ROUND_SEQ):
        return None
    return ROUND_SEQ[rank + 1]
