"""CAF / AFCON 2027 qualifying — a FREE source, derived not fetched.

Why this module is different from the others
--------------------------------------------
Every commercial route to AFCON 2027 qualifying is paid, verified 8 August 2026:

    BSD                no such league in a 79-league catalogue
    api-football       league id 36 exists; free plan capped at 2022-2024 seasons
    football-data.org  competition 2193 exists; TIER_FOUR, 403 with our key
    openfootball       AFCON qualifying only to 2024
    TheSportsDB        not discoverable on the free tier

But the fixtures themselves are public. A 4-team double round-robin is
**deterministic**: given the seeded group order, every matchday pairing follows
from the standard template. So rather than buy the fixture list, we derive it from
the group draw and cross-check it against a published list.

That is more robust than either source alone. The published list at
africasoccer.com contains at least one transcription error — Group A MD6 reads
"Morocco v Gabon; Niger v Gabon", which has Gabon playing twice and repeats the MD1
fixture. `cross_check()` surfaces exactly that class of mistake.

What this CANNOT give you
-------------------------
CAF has published matchday *windows*, not per-fixture kick-off times or venues:

    MD1-2   21 September - 6 October 2026
    MD3-4   9 - 17 November 2026
    MD5-6   22 - 30 March 2027

So every fixture here carries the window start as its date and says so in
`conflict`. They are correct about WHO plays WHOM and roughly when; they are not
precise enough to time an odds poll or settle a bet against. Treat them as
prediction inputs, not as a betting schedule, until a dated source appears.
"""
from __future__ import annotations

from dataclasses import dataclass

# Seeded group order from the draw of 19 May 2026 (Wikipedia, sourced to CAF).
# Position in the list IS the seeding, and the template below depends on it.
GROUPS: dict[str, list[str]] = {
    "A": ["Morocco", "Gabon", "Niger", "Lesotho"],
    "B": ["Egypt", "Angola", "Malawi", "South Sudan"],
    "C": ["Ivory Coast", "Ghana", "Gambia", "Somalia"],
    "D": ["South Africa", "Guinea", "Kenya", "Eritrea"],
    "E": ["DR Congo", "Equatorial Guinea", "Sierra Leone", "Zimbabwe"],
    "F": ["Burkina Faso", "Benin", "Mauritania", "Central African Republic"],
    "G": ["Cameroon", "Comoros", "Namibia", "Congo"],
    "H": ["Tunisia", "Uganda", "Libya", "Botswana"],
    "I": ["Algeria", "Zambia", "Togo", "Burundi"],
    "J": ["Senegal", "Mozambique", "Sudan", "Ethiopia"],
    "K": ["Mali", "Cape Verde", "Rwanda", "Liberia"],
    "L": ["Nigeria", "Madagascar", "Tanzania", "Guinea-Bissau"],
}

# Double round-robin, 1-indexed seed positions, home side first.
#
# The reverse-leg ORDER is not the obvious one and was corrected against the
# published list on 8 August 2026. CAF runs:
#     MD4 = reverse of MD3      MD5 = reverse of MD1      MD6 = reverse of MD2
# The intuitive guess (MD4 reverses MD1, MD5 reverses MD3) is wrong, and produced
# 20 incorrect fixtures before `cross_check()` caught it. Verified against both
# Group A and Group C of the published list, which agree with each other.
TEMPLATE: dict[int, list[tuple[int, int]]] = {
    1: [(1, 2), (3, 4)],
    2: [(2, 3), (4, 1)],
    3: [(1, 3), (2, 4)],
    4: [(3, 1), (4, 2)],      # reverse of MD3
    5: [(2, 1), (4, 3)],      # reverse of MD1
    6: [(3, 2), (1, 4)],      # reverse of MD2
}

# Window start dates. CAF has published windows, not per-fixture dates.
MATCHDAY_WINDOW: dict[int, tuple[str, str]] = {
    1: ("2026-09-21", "2026-10-06"),
    2: ("2026-09-21", "2026-10-06"),
    3: ("2026-11-09", "2026-11-17"),
    4: ("2026-11-09", "2026-11-17"),
    5: ("2027-03-22", "2027-03-30"),
    6: ("2027-03-22", "2027-03-30"),
}

COMPETITION = "African Cup of Nations qualification"
PROVIDER = "caf_derived"


@dataclass(frozen=True)
class DerivedFixture:
    group: str
    matchday: int
    home: str
    away: str
    window_start: str
    window_end: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.home, self.away, self.matchday)


def derive() -> list[DerivedFixture]:
    """All 144 group-stage fixtures from the draw plus the round-robin template."""
    out: list[DerivedFixture] = []
    for group, teams in GROUPS.items():
        if len(teams) != 4:
            raise ValueError(f"group {group} has {len(teams)} teams, expected 4")
        for md, pairs in TEMPLATE.items():
            start, end = MATCHDAY_WINDOW[md]
            for h, a in pairs:
                out.append(DerivedFixture(group, md, teams[h - 1], teams[a - 1],
                                          start, end))
    return out


def validate_template() -> list[str]:
    """Each team must play every other exactly twice, once at home."""
    problems = []
    for group, teams in GROUPS.items():
        played: dict[frozenset, int] = {}
        home_count = {t: 0 for t in teams}
        for f in derive():
            if f.group != group:
                continue
            played[frozenset((f.home, f.away))] = \
                played.get(frozenset((f.home, f.away)), 0) + 1
            home_count[f.home] += 1
        expected_pairs = 6          # C(4,2)
        if len(played) != expected_pairs:
            problems.append(f"group {group}: {len(played)} distinct pairings, "
                            f"expected {expected_pairs}")
        for pair, n in played.items():
            if n != 2:
                problems.append(f"group {group}: {sorted(pair)} meet {n} times")
        for team, n in home_count.items():
            if n != 3:
                problems.append(f"group {group}: {team} has {n} home matches, "
                                f"expected 3")
    return problems


def cross_check(published: dict[tuple[str, int], list[tuple[str, str]]]) -> list[str]:
    """Compare derived pairings against a published list.

    `published` maps (group, matchday) -> [(home, away), ...]. Differences are
    reported, not resolved: a mismatch means either our template assumption or the
    published list is wrong, and that is a question for a human.
    """
    mine: dict[tuple[str, int], set[frozenset]] = {}
    for f in derive():
        mine.setdefault((f.group, f.matchday), set()).add(frozenset((f.home, f.away)))

    problems = []
    for (group, md), pairs in sorted(published.items()):
        theirs = {frozenset(p) for p in pairs}
        ours = mine.get((group, md), set())
        if len(theirs) != len(pairs):
            problems.append(f"group {group} MD{md}: published list has a duplicate "
                            f"or repeated team: {pairs}")
        if theirs != ours:
            only_theirs = [sorted(p) for p in theirs - ours]
            only_ours = [sorted(p) for p in ours - theirs]
            problems.append(
                f"group {group} MD{md}: published {only_theirs} vs derived "
                f"{only_ours}")
    return problems


def to_fixture_rows() -> list[dict]:
    """Canonical fixture rows. Dated to the window start, and flagged as such."""
    from engines.worldcup.names import canonical_team

    from ..store import normalize_fixture

    rows = []
    for f in derive():
        row = normalize_fixture(
            home=canonical_team(f.home), away=canonical_team(f.away),
            competition=COMPETITION, local_date=f.window_start,
            provider=PROVIDER,
            provider_event_id=f"caf2027-{f.group}-md{f.matchday}-"
                              f"{f.home}-{f.away}".replace(" ", "_"),
        )
        row["conflict"] = (
            f"date is the MATCHDAY WINDOW START ({f.window_start} to "
            f"{f.window_end}), not a published kick-off. Group {f.group}, "
            f"matchday {f.matchday}. Derived from the draw, not observed.")
        rows.append(row)
    return rows
