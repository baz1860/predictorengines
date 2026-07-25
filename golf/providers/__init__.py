"""Free-source provider stack for the golf engine.

Historical CSV records and accumulation helpers remain in ``legacy`` for API
compatibility. ESPN live and history fetching share the one implementation in
``espn``:

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
    FieldEntry,
    RoundRecord,
    RoundsProvider,
    TournamentMeta,
    accumulate_rounds,
    accumulate_tours,
    get_provider,
    load_rounds,
    rebuild_tours,
)
from .espn import EspnGolfProvider, EspnProvider
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
    "accumulate_tours",
    "get_provider",
    "load_rounds",
    "rebuild_tours",
    "EspnGolfProvider",
    "ManualOddsProvider",
    "TheOddsApiGolfProvider",
    "PgaTourStatsProvider",
    "OpenMeteoProvider",
]
