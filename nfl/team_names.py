"""Canonical team identity for the NFL engine.

nflverse data keys games/stats to the *abbreviation in use at the time*, so a
relocated franchise shows up under two abbreviations (Oakland/Las Vegas,
San Diego/LA Chargers, St. Louis/LA Rams). We fold historical abbreviations
into today's abbreviation so the franchise carries one continuous rating
history, then map abbreviation -> a stable full display name.

Team identity does NOT get reset by a name-only change (Washington
Redskins/Football Team/Commanders is one team, one abbreviation, WAS
throughout) — only relocations that changed the abbreviation are folded.
"""

# historical abbreviation -> current abbreviation (relocations only)
ABBR_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    # nflverse also occasionally uses these older codes in very old seasons
    "LAR": "LA",
}

# current abbreviation -> stable full display name
FULL_NAME = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

NAME_TO_ABBR = {v: k for k, v in FULL_NAME.items()}


def canon_abbr(abbr: str) -> str:
    """Fold a (possibly historical) abbreviation to its current one."""
    a = str(abbr or "").strip().upper()
    return ABBR_ALIASES.get(a, a)


def full_name(abbr: str) -> str:
    """Current abbreviation (or historical one, folded) -> stable full name."""
    a = canon_abbr(abbr)
    return FULL_NAME.get(a, a)


def to_abbr(name: str) -> str:
    """Full display name -> current abbreviation. Also accepts an abbreviation
    (historical or current) directly, so callers don't need to know which
    they have."""
    n = str(name or "").strip()
    if n in NAME_TO_ABBR:
        return NAME_TO_ABBR[n]
    return canon_abbr(n)


def all_full_names() -> list[str]:
    return sorted(FULL_NAME.values())


_NAME_TO_ABBR_CI = {v.lower(): k for k, v in FULL_NAME.items()}


def normalize(name: str) -> str:
    """Accept a full display name (any case), a current abbreviation, or a
    historical abbreviation, and return the stable full display name."""
    n = str(name or "").strip()
    if not n:
        return n
    hit = _NAME_TO_ABBR_CI.get(n.lower())
    if hit:
        return FULL_NAME[hit]
    return full_name(n)
