"""Free-source provider stack for the golf engine.

The original single-file provider implementation now lives in
``golf.providers.legacy`` and is re-exported here so existing imports keep
working. New free-source providers sit beside it:

* ``espn`` - ESPN/golfastR-style event, leaderboard, field, and score data.
* ``pgatour_stats`` - public PGA Tour aggregate stat pages.
* ``weather`` - Open-Meteo course/weather features.
* ``odds_manual`` - pasted/manual bookmaker boards.
* ``odds_theoddsapi`` - free-tier major outright odds.
"""

from .legacy import (
    CACHE_DIR,
    DATA_DIR,
    ROUNDS_COLUMNS,
    ROUNDS_CSV,
    EspnProvider,
    FieldEntry,
    RoundRecord,
    RoundsProvider,
    TournamentMeta,
    accumulate_rounds,
    get_provider,
    load_rounds,
)
from .espn import EspnGolfProvider
from .odds_manual import ManualOddsProvider
from .odds_theoddsapi import TheOddsApiGolfProvider
from .pgatour_stats import PgaTourStatsProvider
from .weather import OpenMeteoProvider

__all__ = [
    "CACHE_DIR",
    "DATA_DIR",
    "ROUNDS_COLUMNS",
    "ROUNDS_CSV",
    "EspnProvider",
    "FieldEntry",
    "RoundRecord",
    "RoundsProvider",
    "TournamentMeta",
    "accumulate_rounds",
    "get_provider",
    "load_rounds",
    "EspnGolfProvider",
    "ManualOddsProvider",
    "TheOddsApiGolfProvider",
    "PgaTourStatsProvider",
    "OpenMeteoProvider",
]
