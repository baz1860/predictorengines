"""Competition registry: category and importance weight for every label.

The problem this fixes
----------------------
`engines/worldcup/predictor.py` weights rating updates with a 12-entry table and
sends everything else to `DEFAULT_K = 30`. Measured over data/results.csv that is
**34.3% of all played matches** (30.5% since 2010), and the affected set includes
UEFA Euro qualification (2,824 matches) and African Cup of Nations qualification
(2,327) — major competitive fixtures weighted identically to an unlabelled
regional cup, and MORE heavily than a declared friendly (K=20).

Design rules
------------
1. **Exhaustive or explicit.** Every label in results.csv is either mapped here or
   reported by `unmapped()`. Nothing is silently defaulted. Callers that want the
   old behaviour must ask for it.
2. **Weights are a challenger, not an edit.** `WEIGHTS` is versioned and
   `k_for()` takes a `profile` argument. The `"legacy"` profile reproduces the
   current 12-entry table EXACTLY, so the compatibility goldens keep passing while
   the `"v1"` profile is evaluated alongside it (plan §4).
3. **Categories are coarse on purpose.** 201 labels collapse to 9 categories. The
   category drives product decisions (is this bettable?); the weight drives rating
   updates. They are separate concerns and separately reviewable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results.csv"

WEIGHTS_VERSION = 1

# Categories, most to least important.
WORLD_CUP = "world_cup"
WORLD_CUP_QUAL = "world_cup_qualifying"
CONTINENTAL = "continental_championship"
CONTINENTAL_QUAL = "continental_qualifying"
NATIONS_LEAGUE = "nations_league"
REGIONAL_CUP = "regional_cup"
FRIENDLY = "friendly"
MINOR = "minor"          # non-FIFA / sub-national / games-style tournaments
UNKNOWN = "unknown"

# The EXACT legacy table from engines/worldcup/predictor.py. Do not edit: the
# compatibility goldens assert byte-equality of legacy predictions against it.
LEGACY_K = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50, "Copa América": 50, "African Cup of Nations": 50,
    "AFC Asian Cup": 50, "CONCACAF Championship": 50, "Gold Cup": 50,
    "UEFA Nations League": 40, "CONCACAF Nations League": 40,
    "Confederations Cup": 50,
    "Friendly": 20,
}
LEGACY_DEFAULT_K = 30


@dataclass(frozen=True)
class Competition:
    category: str
    k: int          # rating-update weight under the "v1" profile
    bettable: bool  # eligible for the betting product at all (plan §11)
    # True when the classification came from a PATTERN rather than an EXPLICIT
    # entry. Patterns keep the pipeline running when a feed introduces a label we
    # have never seen, but a pattern match is a GUESS. Provisional labels are
    # reported by provisional() and fail the strict gate until a human either adds
    # an EXPLICIT entry or acknowledges the guess. Without this, the broad
    # "Cup|Championship|Trophy|Tournament" rule would be exactly the kind of silent
    # default this registry exists to abolish.
    provisional: bool = False


# Explicit labels. Anything matching a PATTERN below is classified without an
# entry here; anything matching neither is reported by unmapped() and is a
# Stage 1 deliverable, not a silent default.
EXPLICIT: dict[str, Competition] = {
    "FIFA World Cup": Competition(WORLD_CUP, 60, True),
    "FIFA World Cup qualification": Competition(WORLD_CUP_QUAL, 40, True),
    "Confederations Cup": Competition(CONTINENTAL, 50, True),
    "UEFA Euro": Competition(CONTINENTAL, 50, True),
    "Copa América": Competition(CONTINENTAL, 50, True),
    "African Cup of Nations": Competition(CONTINENTAL, 50, True),
    "AFC Asian Cup": Competition(CONTINENTAL, 50, True),
    "CONCACAF Championship": Competition(CONTINENTAL, 50, True),
    "Gold Cup": Competition(CONTINENTAL, 50, True),
    "Oceania Nations Cup": Competition(CONTINENTAL, 40, True),
    "UEFA Nations League": Competition(NATIONS_LEAGUE, 40, True),
    "CONCACAF Nations League": Competition(NATIONS_LEAGUE, 40, True),
    "Friendly": Competition(FRIENDLY, 20, True),

    # Regional cups: real senior internationals, below continental level.
    "Gulf Cup": Competition(REGIONAL_CUP, 30, True),
    "Arab Cup": Competition(REGIONAL_CUP, 30, True),
    "AFF Championship": Competition(REGIONAL_CUP, 30, True),
    "SAFF Cup": Competition(REGIONAL_CUP, 25, True),
    "COSAFA Cup": Competition(REGIONAL_CUP, 25, True),
    "CECAFA Cup": Competition(REGIONAL_CUP, 25, True),
    "WAFF Championship": Competition(REGIONAL_CUP, 25, True),
    "CFU Caribbean Cup": Competition(REGIONAL_CUP, 25, True),
    "Copa Centroamericana": Competition(REGIONAL_CUP, 30, True),
    "UNCAF Cup": Competition(REGIONAL_CUP, 30, True),
    "Nations Cup": Competition(REGIONAL_CUP, 25, True),
    "British Home Championship": Competition(REGIONAL_CUP, 40, False),
    "Baltic Cup": Competition(REGIONAL_CUP, 20, True),
    "Nordic Championship": Competition(REGIONAL_CUP, 20, True),
    "FIFA Series": Competition(FRIENDLY, 20, True),
    "CONCACAF Series": Competition(FRIENDLY, 20, True),
    "Kirin Cup": Competition(FRIENDLY, 20, True),
    "King's Cup": Competition(FRIENDLY, 20, True),
    "Merdeka Tournament": Competition(FRIENDLY, 15, True),

    # Games-style and non-FIFA events: real matches in the dataset, but between
    # selections that are not senior FIFA national teams, or under formats that
    # make them poor rating evidence.
    "Island Games": Competition(MINOR, 10, False),
    "Asian Games": Competition(MINOR, 15, False),
    "Pacific Games": Competition(MINOR, 15, False),
    "African Games": Competition(MINOR, 15, False),
    "Mediterranean Games": Competition(MINOR, 10, False),
    "Viva World Cup": Competition(MINOR, 5, False),
    "ConIFA World Football Cup": Competition(MINOR, 5, False),
    "ConIFA European Football Cup": Competition(MINOR, 5, False),
    "FIFI Wild Cup": Competition(MINOR, 5, False),

    # Non-FIFA / territorial competitions.
    "Coupe de l'Outre-Mer": Competition(MINOR, 5, False),
    "Muratti Vase": Competition(MINOR, 5, False),
    "GaNEFo": Competition(MINOR, 5, False),

    # Invitational tournaments between senior sides — friendly in substance.
    "Mundialito": Competition(REGIONAL_CUP, 25, True),
    "Tournoi de France": Competition(FRIENDLY, 20, True),
    "Tri-Nations Series": Competition(FRIENDLY, 20, True),
    "Canadian Shield": Competition(FRIENDLY, 20, True),
    "Mukuru 4 Nations": Competition(FRIENDLY, 20, True),
    "Morocco, Capital of African Football": Competition(FRIENDLY, 20, True),
    "SKN Football Festival": Competition(FRIENDLY, 20, True),
    "The Other Final": Competition(FRIENDLY, 20, True),
}

# Bilateral trophies — two nations playing for a named cup. Competitively these
# are friendlies with a trophy attached, and the dataset labels each leg with the
# trophy name rather than "Friendly". Listed explicitly rather than pattern-matched
# on "Copa " so that a future CONMEBOL competition cannot be swept in silently.
_BILATERAL_TROPHIES = (
    "Copa Artigas", "Copa Bernardo O'Higgins", "Copa Carlos Dittborn",
    "Copa Chevallier Boutell", "Copa Confraternidad", "Copa Félix Bogado",
    "Copa Juan Pinto Durán", "Copa Lipton", "Copa Newton", "Copa Oswaldo Cruz",
    "Copa Paz del Chaco", "Copa Premio Honor Argentino",
    "Copa Premio Honor Uruguayo", "Copa Ramón Castilla", "Copa Rio Branco",
    "Copa Roca", "Copa del Pacífico", "Superclásico de las Américas",
    "Soccer Ashes",
)
EXPLICIT.update({name: Competition(FRIENDLY, 20, True) for name in _BILATERAL_TROPHIES})

# Continental qualifying. These are the labels the legacy 12-entry table missed
# most damagingly: Euro qualifying alone is 2,824 matches, and under the legacy
# weights it moved ratings by DEFAULT_K=30 — LESS than a World Cup qualifier (40)
# and only slightly more than a friendly (20). Explicit, not pattern-matched,
# because their weights are load-bearing.
EXPLICIT.update({
    "UEFA Euro qualification": Competition(CONTINENTAL_QUAL, 40, True),
    "African Cup of Nations qualification": Competition(CONTINENTAL_QUAL, 40, True),
    "AFC Asian Cup qualification": Competition(CONTINENTAL_QUAL, 40, True),
    "CONCACAF Championship qualification": Competition(CONTINENTAL_QUAL, 40, True),
    "Gold Cup qualification": Competition(CONTINENTAL_QUAL, 40, True),
    "Copa América qualification": Competition(CONTINENTAL_QUAL, 40, True),
    "Oceania Nations Cup qualification": Competition(CONTINENTAL_QUAL, 40, True),
    "CONCACAF Nations League qualification": Competition(NATIONS_LEAGUE, 35, True),
    "CFU Caribbean Cup qualification": Competition(CONTINENTAL_QUAL, 30, True),
    "AFF Championship qualification": Competition(REGIONAL_CUP, 25, True),
    "COSAFA Cup qualification": Competition(REGIONAL_CUP, 20, True),
    "Arab Cup qualification": Competition(REGIONAL_CUP, 25, True),
    "EAFF Championship qualification": Competition(REGIONAL_CUP, 25, True),
    "ASEAN Championship qualification": Competition(REGIONAL_CUP, 25, True),
    "AFC Challenge Cup qualification": Competition(REGIONAL_CUP, 20, True),

    # High-volume regional competitions.
    "ASEAN Championship": Competition(REGIONAL_CUP, 30, True),
    "EAFF Championship": Competition(REGIONAL_CUP, 25, True),
    "CAFA Nations Cup": Competition(REGIONAL_CUP, 25, True),
    "AFC Challenge Cup": Competition(REGIONAL_CUP, 20, True),
    "AFC Solidarity Cup": Competition(REGIONAL_CUP, 20, True),
    "Amílcar Cabral Cup": Competition(REGIONAL_CUP, 25, True),
    "CCCF Championship": Competition(REGIONAL_CUP, 30, True),
    "NAFC Championship": Competition(REGIONAL_CUP, 30, True),
    "NAFU Championship": Competition(REGIONAL_CUP, 30, True),
    "Central European International Cup": Competition(REGIONAL_CUP, 35, True),
    "Windward Islands Tournament": Competition(REGIONAL_CUP, 20, True),
    "Korea Cup": Competition(FRIENDLY, 20, True),
    "Nehru Cup": Competition(FRIENDLY, 20, True),
    "Intercontinental Cup": Competition(FRIENDLY, 20, True),

    # Multi-sport games and non-FIFA circuits: senior sides are frequently not
    # first-choice, so these are poor rating evidence and are never bettable.
    "Olympic Games": Competition(MINOR, 15, False),
    "South Pacific Games": Competition(MINOR, 10, False),
    "South Pacific Mini Games": Competition(MINOR, 10, False),
    "Southeast Asian Games": Competition(MINOR, 10, False),
    "Southeast Asian Peninsular Games": Competition(MINOR, 10, False),
    "South Asian Games": Competition(MINOR, 10, False),
    "East Asian Games": Competition(MINOR, 10, False),
    "Indian Ocean Island Games": Competition(MINOR, 10, False),
    "Central American and Caribbean Games": Competition(MINOR, 10, False),
    "All-African Games": Competition(MINOR, 10, False),
    "Bolivarian Games": Competition(MINOR, 10, False),
    "Pacific Mini Games": Competition(MINOR, 10, False),
    "Pacific Games": Competition(MINOR, 10, False),
    "Afro-Asian Games": Competition(MINOR, 10, False),
    "Far Eastern Championship Games": Competition(MINOR, 10, False),
    "Inter-Allied Games": Competition(MINOR, 5, False),
    "Inter Games": Competition(MINOR, 5, False),
    "CONIFA World Football Cup": Competition(MINOR, 5, False),
    "CONIFA World Cup qualification": Competition(MINOR, 5, False),
    "CONIFA World Football Cup qualification": Competition(MINOR, 5, False),
    "CONIFA Africa Football Cup": Competition(MINOR, 5, False),
    "CONIFA Asia Cup": Competition(MINOR, 5, False),
    "CONIFA European Football Cup": Competition(MINOR, 5, False),
    "CONIFA South America Football Cup": Competition(MINOR, 5, False),
    "ConIFA Challenger Cup": Competition(MINOR, 5, False),
})

# The remaining pattern-matched labels: a long tail of one-off invitationals and
# defunct regional cups, none with meaningful volume. Listed so that a NEW label
# arriving from a feed is detectable — `unacknowledged_provisional()` returns it,
# and the strict gate fails. Acknowledging a guess is not the same as reviewing it;
# these carry pattern-derived weights and should be revisited if any becomes active.
ACKNOWLEDGED_PROVISIONAL: frozenset[str] = frozenset({
    "ABCS Tournament",                                      # 20
    "African Friendship Games",                             # 55
    "Al Ain International Cup",                             # 4
    "Atlantic Cup",                                         # 11
    "Atlantic Heritage Cup",                                # 2
    "Balkan Cup",                                           # 87
    "Beijing International Friendship Tournament",          # 10
    "Benedikt Fontana Cup",                                 # 1
    "Brazil Independence Cup",                              # 37
    "CONMEBOL–UEFA Cup of Champions",                       # 3
    "Corsica Cup",                                          # 4
    "Cup of Ancient Civilizations",                         # 2
    "Cyprus International Tournament",                      # 70
    "Dakar Tournament",                                     # 4
    "Diamond Jubilee International Football Tournament",    # 5
    "Dragon Cup",                                           # 9
    "Dunhill Cup",                                          # 15
    "Dynasty Cup",                                          # 29
    "ELF Cup",                                              # 16
    "FIFA 75th Anniversary Cup",                            # 1
    "Four Nations Tournament",                              # 8
    "Four Nations' Cup",                                    # 2
    "Great Wall Cup",                                       # 4
    "Guangzhou International Friendship Tournament",        # 4
    "Hungary Heritage Cup",                                 # 3
    "Indonesia Tournament",                                 # 75
    "International Tournament of Peoples, Cultures and Tribes",# 7
    "Joe Robbie Cup",                                       # 4
    "Jordan International Tournament",                      # 17
    "KTFF 50th Anniversary Cup",                            # 3
    "King Hassan II Tournament",                            # 12
    "Kirin Challenge Cup",                                  # 22
    "Kuneitra Cup",                                         # 15
    "Lunar New Year Cup",                                   # 25
    "MSG Prime Minister's Cup",                             # 23
    "Mahinda Rajapaksa Cup",                                # 7
    "Malta International Tournament",                       # 53
    "Mapinduzi Cup",                                        # 7
    "Marianas Cup",                                         # 2
    "Marlboro Cup",                                         # 8
    "Matthews Cup",                                         # 4
    "Mauritius Four Nations Cup",                           # 5
    "Melanesia Cup",                                        # 64
    "Merlion Cup",                                          # 25
    "Miami Cup",                                            # 12
    "Millennium Cup",                                       # 15
    "Navruz Cup",                                           # 4
    "Niamh Challenge Cup",                                  # 3
    "Nile Basin Tournament",                                # 14
    "OSN Cup",                                              # 4
    "Open International Championship",                      # 1
    "Outrigger Challenge Cup",                              # 3
    "Palestine Cup",                                        # 58
    "Palestine International Championship",                 # 11
    "Pan American Championship",                            # 39
    "Peace Cup",                                            # 6
    "Philippine Peace Cup",                                 # 12
    "Phillip Seaga Cup",                                    # 3
    "Prime Minister's Cup",                                 # 22
    "Real Madrid 75th Anniversary Cup",                     # 1
    "Rous Cup",                                             # 11
    "Scania 100 Tournament",                                # 4
    "Simba Tournament",                                     # 17
    "South Asian Super Cup",                                # 1
    "TIFOCO Tournament",                                    # 1
    "Three Nations Cup",                                    # 3
    "Tournament Burkina Faso",                              # 11
    "Trans-Tasman Cup",                                     # 12
    "Tri Nation Tournament",                                # 7
    "Tri-Nations Cup",                                      # 2
    "Tynwald Hill Tournament",                              # 6
    "UDEAC Cup",                                            # 70
    "UNIFFAC Cup",                                          # 15
    "USA Cup",                                              # 37
    "United Arab Emirates Friendship Tournament",           # 31
    "Unity Cup",                                            # 12
    "VFF Cup",                                              # 9
    "Vietnam Independence Cup",                             # 60
    "West African Cup",                                     # 54
    "World Unity Cup",                                      # 3
    "Zambian Independence Tournament",                      # 3
    "Évence Coppée Trophy",                                 # 1
})

# Ordered: first match wins. Applied only when EXPLICIT has no entry.
PATTERNS: list[tuple[re.Pattern, Competition]] = [
    (re.compile(r"^FIFA World Cup qualification", re.I), Competition(WORLD_CUP_QUAL, 40, True)),
    (re.compile(r"qualification|qualifier", re.I), Competition(CONTINENTAL_QUAL, 40, True)),
    (re.compile(r"^(UEFA Euro|Copa América|African Cup of Nations|AFC Asian Cup|"
                r"CONCACAF Championship|Gold Cup|Oceania Nations Cup)", re.I),
     Competition(CONTINENTAL, 50, True)),
    (re.compile(r"Nations League", re.I), Competition(NATIONS_LEAGUE, 40, True)),
    (re.compile(r"Games$|Games ", re.I), Competition(MINOR, 10, False)),
    (re.compile(r"ConIFA|Viva World Cup|Wild Cup", re.I), Competition(MINOR, 5, False)),
    (re.compile(r"Cup|Championship|Trophy|Tournament", re.I), Competition(REGIONAL_CUP, 25, True)),
]


def classify(label: object) -> Competition | None:
    """Category + weight for a competition label, or None if unclassifiable."""
    name = str(label or "").strip()
    if not name:
        return None
    if name in EXPLICIT:
        return EXPLICIT[name]
    for pat, comp in PATTERNS:
        if pat.search(name):
            return Competition(comp.category, comp.k, comp.bettable, provisional=True)
    return None


def k_for(label: object, profile: str = "legacy") -> int:
    """Rating-update weight.

    profile="legacy" reproduces engines/worldcup/predictor.py exactly, including
    DEFAULT_K=30 for unmapped labels. profile="v1" uses this registry. The
    challenger is never the default (plan §4).
    """
    if profile == "legacy":
        return LEGACY_K.get(str(label), LEGACY_DEFAULT_K)
    if profile != "v1":
        raise ValueError(f"unknown weight profile {profile!r}")
    comp = classify(label)
    if comp is None:
        raise KeyError(
            f"competition {label!r} is not classified. Add it to "
            f"international/taxonomy.py EXPLICIT or PATTERNS — the v1 profile "
            f"refuses to silently default (plan §4)."
        )
    return comp.k


def category(label: object) -> str:
    comp = classify(label)
    return comp.category if comp else UNKNOWN


def is_bettable(label: object) -> bool:
    comp = classify(label)
    return bool(comp and comp.bettable)


@lru_cache(maxsize=1)
def _labels() -> tuple[str, ...]:
    df = pd.read_csv(RESULTS, usecols=["tournament"])
    return tuple(sorted(df.tournament.dropna().astype(str).unique()))


def unmapped(labels: object = None) -> list[str]:
    """Labels present in the data that no rule classifies. Should be empty."""
    src = _labels() if labels is None else [str(x) for x in labels]
    return sorted({name for name in src if classify(name) is None})


def provisional(labels: object = None) -> list[str]:
    """Labels classified by PATTERN rather than an explicit entry — i.e. guessed.

    Non-empty is normal for the long tail of historical regional cups; what matters
    is that the set is *known* and does not grow silently when a feed introduces a
    new competition. The strict gate fails on anything here that is not listed in
    ACKNOWLEDGED_PROVISIONAL.
    """
    src = _labels() if labels is None else [str(x) for x in labels]
    out = []
    for name in src:
        comp = classify(name)
        if comp is not None and comp.provisional:
            out.append(name)
    return sorted(set(out))


def unacknowledged_provisional(labels: object = None) -> list[str]:
    return [n for n in provisional(labels) if n not in ACKNOWLEDGED_PROVISIONAL]


def coverage_report() -> pd.DataFrame:
    """Per-category match counts, for review of the mapping."""
    df = pd.read_csv(RESULTS, usecols=["tournament", "home_score"])
    df["category"] = df.tournament.map(category)
    df["played"] = df.home_score.notna()
    out = (df.groupby("category")
             .agg(labels=("tournament", "nunique"),
                  matches=("tournament", "size"),
                  played=("played", "sum"))
             .sort_values("matches", ascending=False))
    return out


if __name__ == "__main__":
    miss = unmapped()
    print(coverage_report().to_string())
    print(f"\nunclassified labels: {len(miss)}")
    for name in miss:
        print(f"  {name}")
