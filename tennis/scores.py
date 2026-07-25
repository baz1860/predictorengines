"""Canonical tennis score parsing shared by providers and the model."""
from __future__ import annotations

from typing import Optional

STOP_TOKENS = {
    "RET", "RET.", "DEF", "DEF.", "W/O", "WO", "WALKOVER",
    "ABN", "ABD", "UNK",
}


def is_retirement_or_walkover(score: Optional[str]) -> bool:
    tokens = {token.upper() for token in str(score or "").split()}
    return bool(tokens & STOP_TOKENS)


def parsed_sets(score: Optional[str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for token in str(score or "").strip().split():
        if token.upper() in STOP_TOKENS:
            break
        core = token.split("(", 1)[0]
        if "-" not in core:
            continue
        a, _, b = core.partition("-")
        try:
            games = (int(a), int(b))
        except ValueError:
            continue
        if games[0] != games[1]:
            out.append(games)
    return out


def parse_set_score(score: Optional[str]) -> tuple[int, int]:
    """Sets won by the score's winner-first and loser-first sides."""
    winner_sets = loser_sets = 0
    for winner_games, loser_games in parsed_sets(score):
        if winner_games > loser_games:
            winner_sets += 1
        else:
            loser_sets += 1
    return winner_sets, loser_sets


def score_total_games(score: Optional[str]) -> int | None:
    """Total games in a completed match, excluding retirements/walkovers."""
    if is_retirement_or_walkover(score):
        return None
    sets = parsed_sets(score)
    return sum(a + b for a, b in sets) if sets else None
