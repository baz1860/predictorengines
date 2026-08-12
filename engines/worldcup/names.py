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
    # Added August 2026 from the BSD international spike: these spellings caused
    # 24 fixtures involving FIFA members to be silently dropped as "out of scope".
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Ireland": "Republic of Ireland",
    "Eire": "Republic of Ireland",
    "US Virgin Islands": "United States Virgin Islands",
    "U.S. Virgin Islands": "United States Virgin Islands",
    "Chinese Taipei": "Taiwan",
    "Cape Verde Is.": "Cape Verde",
    "Korea DPR": "North Korea",
    "Korea Republic": "South Korea",
    "China PR": "China",
    "St Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St Lucia": "Saint Lucia",
    "St. Lucia": "Saint Lucia",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Antigua & Barbuda": "Antigua and Barbuda",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Bosnia": "Bosnia and Herzegovina",
}

# Markers that a "team" from a provider feed is not a senior national side.
# BSD files club friendlies and youth internationals inside its
# "International Friendly Games" league, so the fixture parser must reject them
# rather than let a club into a national-team model.
NON_NATIONAL_MARKERS = (
    " U15", " U16", " U17", " U18", " U19", " U20", " U21", " U22", " U23",
    " B", " XI", " Olympic", " Women", " (W)", " Amateur", " Select",
)


def looks_like_national_team(name: object) -> bool:
    """Heuristic rejection of clubs and age-group sides.

    Deliberately conservative: it only rejects on explicit markers, and anything
    that survives still has to match the team registry, which is the real gate.
    Two layers, because a club slipping into an international rating model is
    both easy to miss and expensive.
    """
    raw = f" {str(name or '').strip()}"
    return not any(raw.endswith(m) or f"{m} " in raw for m in NON_NATIONAL_MARKERS)


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
