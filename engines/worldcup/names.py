"""World Cup team-name normalization shared across provider parsers."""
from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

ALIASES = {
    "USA": "United States",
    "United States of America": "United States",
    "USMNT": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "DRC": "DR Congo",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "IR Iran": "Iran",
}


def canonical_team(name: object) -> str:
    """Return the repository spelling for a provider/bookmaker team name."""
    raw = str(name or "").strip()
    return ALIASES.get(raw, raw)


def known_teams() -> set[str]:
    """Teams present in World Cup inputs, used for strict provider validation."""
    teams: set[str] = set()
    for rel, cols in (
        ("data/results.csv", ("home_team", "away_team")),
        ("data/squads.csv", ("team",)),
    ):
        path = ROOT / rel
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=lambda c: c in cols)
        except Exception:
            continue
        for col in cols:
            if col in df.columns:
                teams.update(df[col].dropna().astype(str).map(canonical_team))
    return teams


def require_known_team(name: object, teams: Iterable[str] | None = None,
                       context: str = "provider") -> str:
    """Canonicalize and reject unknown names with a useful alias hint."""
    canon = canonical_team(name)
    known = set(teams) if teams is not None else known_teams()
    if known and canon not in known:
        close = get_close_matches(canon, sorted(known), n=3, cutoff=0.6)
        hint = f" Close matches: {', '.join(close)}." if close else ""
        raise ValueError(
            f"Unknown {context} team {name!r} canonicalized to {canon!r}."
            f" Add an alias in engines/worldcup/names.py.{hint}"
        )
    return canon
