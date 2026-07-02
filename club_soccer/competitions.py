"""Competition registry for the Club Soccer engine."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_COMP_STRENGTH_FILE = Path(__file__).resolve().parent / "data" / "comp_strength.json"
_comp_strength_cache: dict | None = None


@dataclass(frozen=True)
class Competition:
    name: str
    country: str
    kind: str
    tier: int
    api_id: int        # api-football league id (kept for fixtures.csv backwards compat)
    strength: float
    fdorg_code: str    # football-data.org competition code ("PL", "BL1", …); "" = not covered free
    bsd_league: str    # BSD league name substring for filtering; "" = use self.name
    # League geometry — parameterises P4.3 motivation bands. Not legally
    # perfect (e.g. playoff spots folded into a single count); documented
    # approximation per the plan. Zero for cups/Europe (not a table).
    teams_n: int = 0
    releg_spots: int = 0
    promo_spots: int = 0
    euro_spots: int = 0


COMPETITIONS = [
    Competition("Premier League",       "England",  "league", 1,  39,  1.00, "PL",  "",
               teams_n=20, releg_spots=3, promo_spots=0, euro_spots=7),
    Competition("Championship",         "England",  "league", 2,  40,  0.72, "ELC", "Championship",
               teams_n=24, releg_spots=3, promo_spots=3, euro_spots=0),
    Competition("League One",           "England",  "league", 3,  41,  0.50, "EL1", "League One",
               teams_n=24, releg_spots=4, promo_spots=3, euro_spots=0),
    Competition("League Two",           "England",  "league", 4,  42,  0.35, "EL2", "League Two",
               teams_n=24, releg_spots=2, promo_spots=4, euro_spots=0),
    Competition("FA Cup",               "England",  "cup",    0,  45,  0.75, "FAC", "FA Cup"),
    Competition("EFL Cup",              "England",  "cup",    0,  48,  0.70, "",    "EFL Cup"),
    Competition("Scottish Premiership", "Scotland", "league", 1, 179,  0.58, "",    "Scottish Premiership",
               teams_n=12, releg_spots=1, promo_spots=1, euro_spots=4),
    Competition("Scottish Championship","Scotland", "league", 2, 180,  0.38, "",    "Scottish Championship",
               teams_n=10, releg_spots=1, promo_spots=1, euro_spots=0),
    Competition("Scottish League One",  "Scotland", "league", 3, 181,  0.28, "",    "Scottish League One",
               teams_n=10, releg_spots=1, promo_spots=1, euro_spots=0),
    Competition("Scottish League Two",  "Scotland", "league", 4, 182,  0.22, "",    "Scottish League Two",
               teams_n=10, releg_spots=1, promo_spots=1, euro_spots=0),
    Competition("Scottish Cup",         "Scotland", "cup",    0, 183,  0.48, "",    "Scottish Cup"),
    Competition("Scottish League Cup",  "Scotland", "cup",    0, 184,  0.45, "",    "Scottish League Cup"),
    Competition("Bundesliga",           "Germany",  "league", 1,  78,  0.93, "BL1", "Bundesliga",
               teams_n=18, releg_spots=3, promo_spots=0, euro_spots=7),
    Competition("DFB-Pokal",            "Germany",  "cup",    0,  81,  0.72, "DFB", "DFB-Pokal"),
    Competition("Serie A",              "Italy",    "league", 1, 135,  0.91, "SA",  "Serie A",
               teams_n=20, releg_spots=3, promo_spots=0, euro_spots=8),
    Competition("Coppa Italia",         "Italy",    "cup",    0, 137,  0.70, "",    "Coppa Italia"),
    Competition("Ligue 1",              "France",   "league", 1,  61,  0.86, "FL1", "Ligue 1",
               teams_n=18, releg_spots=3, promo_spots=0, euro_spots=7),
    Competition("Coupe de France",      "France",   "cup",    0,  66,  0.64, "",    "Coupe de France"),
    Competition("La Liga",              "Spain",    "league", 1, 140,  0.92, "PD",  "La Liga",
               teams_n=20, releg_spots=3, promo_spots=0, euro_spots=8),
    Competition("Copa del Rey",         "Spain",    "cup",    0, 143,  0.70, "",    "Copa del Rey"),
    Competition("Champions League",     "Europe",   "europe", 0,   2,  1.08, "CL",  "Champions League"),
    Competition("Europa League",        "Europe",   "europe", 0,   3,  0.88, "EL",  "Europa League"),
    Competition("Conference League",    "Europe",   "europe", 0, 848,  0.70, "",    "Conference League"),
    Competition("UEFA Super Cup",       "Europe",   "europe", 0, 531,  1.00, "",    "UEFA Super Cup"),
]

BY_NAME = {c.name: c for c in COMPETITIONS}
BY_API_ID = {c.api_id: c for c in COMPETITIONS}

# BSD league names that are covered free (football only requires a BSD key, all leagues free)
# Maps BSD league name (lowercase) -> Competition name in our registry
# Populated lazily — BSD may use slightly different names; this handles common variants
BSD_LEAGUE_ALIASES: dict[str, str] = {
    "premier league": "Premier League",
    "efl championship": "Championship",
    "championship": "Championship",
    "efl league one": "League One",
    "league one": "League One",
    "efl league two": "League Two",
    "league two": "League Two",
    "fa cup": "FA Cup",
    "efl cup": "EFL Cup",
    "carabao cup": "EFL Cup",
    "league cup": "EFL Cup",
    "scottish premiership": "Scottish Premiership",
    "scottish championship": "Scottish Championship",
    "scottish league one": "Scottish League One",
    "scottish league two": "Scottish League Two",
    "scottish cup": "Scottish Cup",
    "scottish league cup": "Scottish League Cup",
    "bundesliga": "Bundesliga",
    "1. bundesliga": "Bundesliga",
    "dfb-pokal": "DFB-Pokal",
    "dfb pokal": "DFB-Pokal",
    "serie a": "Serie A",
    "coppa italia": "Coppa Italia",
    "ligue 1": "Ligue 1",
    "coupe de france": "Coupe de France",
    "la liga": "La Liga",
    "laliga": "La Liga",
    "copa del rey": "Copa del Rey",
    "uefa champions league": "Champions League",
    "champions league": "Champions League",
    "uefa europa league": "Europa League",
    "europa league": "Europa League",
    "uefa europa conference league": "Conference League",
    "conference league": "Conference League",
    "europa conference league": "Conference League",
    "uefa super cup": "UEFA Super Cup",
}

# football-data.org: competitions covered on the free tier
FDORG_COMPETITIONS = {c.name: c for c in COMPETITIONS if c.fdorg_code}


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


def _load_comp_strength() -> dict:
    """Fitted competition-strength overrides (P4.4) — cached for the process
    lifetime. Consulted by strength() ONLY when the file's "active" flag is
    true (report-only/gated-OFF by default, per the promotion discipline)."""
    global _comp_strength_cache
    if _comp_strength_cache is None:
        if _COMP_STRENGTH_FILE.exists():
            try:
                _comp_strength_cache = json.loads(_COMP_STRENGTH_FILE.read_text())
            except Exception:
                _comp_strength_cache = {}
        else:
            _comp_strength_cache = {}
    return _comp_strength_cache


def reload_comp_strength() -> None:
    """Force the next strength() call to re-read comp_strength.json from
    disk (tests / --fit-comp-strength change the file mid-process)."""
    global _comp_strength_cache
    _comp_strength_cache = None


def strength(name: str | None) -> float:
    c = get(name)
    base = c.strength if c else 0.75
    fitted = _load_comp_strength()
    if fitted.get("active") and name in fitted:
        return float(fitted[name])
    return base


def comp_from_bsd_league(bsd_name: str) -> Competition | None:
    """Resolve a BSD league name to a Competition.

    Exact match only (via BSD_LEAGUE_ALIASES or the bsd_league hint field).
    A prior substring-containment fallback caused real cross-continent
    collisions once BSD's fuller league catalog was queried, e.g. "USL
    Championship" (USA) matching our "Championship" (England) because
    "championship" in "usl championship", and "CAF Champions League" /
    "Brasileirão Serie A" likewise matching "Champions League" / "Serie A".
    Unrecognised names are printed by callers and simply skipped — safer
    than silently blending in a different continent's results.
    """
    low = str(bsd_name).strip().lower()
    if low in BSD_LEAGUE_ALIASES:
        return BY_NAME.get(BSD_LEAGUE_ALIASES[low])
    for c in COMPETITIONS:
        hint = (c.bsd_league or c.name).lower()
        if hint and hint == low:
            return c
    return None
