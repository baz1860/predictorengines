"""Competition registry for the Club Soccer engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass


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


COMPETITIONS = [
    Competition("Premier League",       "England",  "league", 1,  39,  1.00, "PL",  ""),
    Competition("Championship",         "England",  "league", 2,  40,  0.72, "ELC", "Championship"),
    Competition("League One",           "England",  "league", 3,  41,  0.50, "EL1", "League One"),
    Competition("League Two",           "England",  "league", 4,  42,  0.35, "EL2", "League Two"),
    Competition("FA Cup",               "England",  "cup",    0,  45,  0.75, "FAC", "FA Cup"),
    Competition("EFL Cup",              "England",  "cup",    0,  48,  0.70, "",    "EFL Cup"),
    Competition("Scottish Premiership", "Scotland", "league", 1, 179,  0.58, "",    "Scottish Premiership"),
    Competition("Scottish Championship","Scotland", "league", 2, 180,  0.38, "",    "Scottish Championship"),
    Competition("Scottish League One",  "Scotland", "league", 3, 181,  0.28, "",    "Scottish League One"),
    Competition("Scottish League Two",  "Scotland", "league", 4, 182,  0.22, "",    "Scottish League Two"),
    Competition("Scottish Cup",         "Scotland", "cup",    0, 183,  0.48, "",    "Scottish Cup"),
    Competition("Scottish League Cup",  "Scotland", "cup",    0, 184,  0.45, "",    "Scottish League Cup"),
    Competition("Bundesliga",           "Germany",  "league", 1,  78,  0.93, "BL1", "Bundesliga"),
    Competition("DFB-Pokal",            "Germany",  "cup",    0,  81,  0.72, "DFB", "DFB-Pokal"),
    Competition("Serie A",              "Italy",    "league", 1, 135,  0.91, "SA",  "Serie A"),
    Competition("Coppa Italia",         "Italy",    "cup",    0, 137,  0.70, "",    "Coppa Italia"),
    Competition("Ligue 1",              "France",   "league", 1,  61,  0.86, "FL1", "Ligue 1"),
    Competition("Coupe de France",      "France",   "cup",    0,  66,  0.64, "",    "Coupe de France"),
    Competition("La Liga",              "Spain",    "league", 1, 140,  0.92, "PD",  "La Liga"),
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


def strength(name: str | None) -> float:
    c = get(name)
    return c.strength if c else 0.75


def comp_from_bsd_league(bsd_name: str) -> Competition | None:
    """Resolve a BSD league name to a Competition, using fuzzy alias matching."""
    low = str(bsd_name).strip().lower()
    # Exact alias lookup
    if low in BSD_LEAGUE_ALIASES:
        return BY_NAME.get(BSD_LEAGUE_ALIASES[low])
    # Substring match against bsd_league hint field
    for c in COMPETITIONS:
        hint = (c.bsd_league or c.name).lower()
        if hint and (hint in low or low in hint):
            return c
    return None
