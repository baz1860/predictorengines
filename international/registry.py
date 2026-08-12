"""Team registry: who is in scope, which confederation, FIFA member or not.

Why this exists
---------------
Plan §3. Previous revisions of the plan used six incompatible definitions of "all
national teams" — 338, 257, "about 230", 211, "top 60". None was testable. The
dataset mixes FIFA members with confederation associates (Guadeloupe, Sint
Maarten), non-FIFA territories (Greenland, Isle of Man) and sub-national or
stateless sides (Kernow, Padania, Sápmi, Tamil Eelam).

Contract
--------
* `status` is one of: fifa | non_fifa | unclassified
* `in_scope(team)` FAILS CLOSED — an *active* team with status `unclassified`
  raises rather than being silently included or excluded. That is the quarantine
  pattern the plan requires, and it means adding a new team to the feed is a
  deliberate act, not an accident.
* Membership is **effective-dated**: `member_from` / `member_to` (ISO dates, blank
  = open interval), so a historical fixture is judged against membership *at the
  time*, not today's list.

Known limitation, recorded rather than hidden: the seed classifies teams by
current status only. Dated joins are populated for the small number of cases that
matter to post-2000 fixtures; earlier history uses the open interval. Backfilling
full accession dates is a Stage 1 completion item, and `dated_coverage()` reports
how much is still open-ended.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_CSV = ROOT / "data" / "international" / "team_registry.csv"
RESULTS = ROOT / "data" / "results.csv"

FIFA = "fifa"
NON_FIFA = "non_fifa"
UNCLASSIFIED = "unclassified"

# A team is "active" if it has played inside this window. Dormant teams are not
# required to be classified, which stops 1930s one-off sides blocking the gate.
ACTIVE_YEARS = 4


class ScopeError(RuntimeError):
    """An active team is not classified. Refuse to guess (plan §3)."""


@dataclass(frozen=True)
class Team:
    name: str
    status: str
    confederation: str
    member_from: str
    member_to: str

    def is_member_on(self, date: object) -> bool:
        if self.status != FIFA:
            return False
        d = str(pd.Timestamp(date).date())
        if self.member_from and d < self.member_from:
            return False
        if self.member_to and d > self.member_to:
            return False
        return True


@lru_cache(maxsize=1)
def load() -> dict[str, Team]:
    if not REGISTRY_CSV.exists():
        raise FileNotFoundError(
            f"{REGISTRY_CSV} missing. Generate it with:\n"
            f"  python3 -m scripts.international.seed_team_registry --write")
    df = pd.read_csv(REGISTRY_CSV, dtype=str, keep_default_na=False)
    return {r.team: Team(r.team, r.status, r.confederation,
                         r.member_from, r.member_to)
            for r in df.itertuples(index=False)}


@lru_cache(maxsize=1)
def active_teams() -> frozenset[str]:
    df = pd.read_csv(RESULTS, usecols=["date", "home_team", "away_team"],
                     parse_dates=["date"])
    cutoff = df.date.max() - pd.DateOffset(years=ACTIVE_YEARS)
    recent = df[df.date >= cutoff]
    return frozenset(set(recent.home_team.dropna()) | set(recent.away_team.dropna()))


def status(team: object) -> str:
    entry = load().get(str(team))
    return entry.status if entry else UNCLASSIFIED


def confederation(team: object) -> str:
    entry = load().get(str(team))
    return entry.confederation if entry else ""


def unclassified_active() -> list[str]:
    reg = load()
    return sorted(t for t in active_teams()
                  if reg.get(t, None) is None or reg[t].status == UNCLASSIFIED)


def assert_scope_complete() -> None:
    """Gate hook: refuse to proceed while an active team is unclassified."""
    missing = unclassified_active()
    if missing:
        raise ScopeError(
            f"{len(missing)} active team(s) unclassified in {REGISTRY_CSV.name}: "
            f"{', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}. "
            f"Classify them (fifa / non_fifa) — scope fails closed by design.")


def in_scope(team: object, date: object = None, universe: str = "fifa") -> bool:
    """Is this team in the product universe?

    universe="fifa"  -> FIFA members only, honouring effective dates if `date` given
    universe="all"   -> everything in the dataset
    """
    if universe == "all":
        return True
    if universe != "fifa":
        raise ValueError(f"unknown universe {universe!r}")
    name = str(team)
    entry = load().get(name)
    if entry is None or entry.status == UNCLASSIFIED:
        if name in active_teams():
            raise ScopeError(
                f"team {name!r} is active but unclassified — add it to "
                f"{REGISTRY_CSV.name} (plan §3, fail closed).")
        return False
    return entry.is_member_on(date) if date is not None else entry.status == FIFA


def fixture_in_scope(home: object, away: object, date: object = None,
                     universe: str = "fifa", require_both: bool = True) -> bool:
    """Scope test for a fixture.

    `require_both=True` is the recommended default and the one the plan asks to be
    written down: a match counts only if BOTH sides are in the universe. Setting it
    False includes matches where a member plays a non-member, which keeps e.g.
    England v a non-FIFA invitational side in the product.
    """
    h = in_scope(home, date, universe)
    a = in_scope(away, date, universe)
    return (h and a) if require_both else (h or a)


def dated_coverage() -> dict[str, int]:
    """How much of the registry carries real accession dates vs open intervals."""
    reg = load()
    fifa = [t for t in reg.values() if t.status == FIFA]
    return {"fifa_members": len(fifa),
            "with_join_date": sum(1 for t in fifa if t.member_from),
            "open_interval": sum(1 for t in fifa if not t.member_from),
            "non_fifa": sum(1 for t in reg.values() if t.status == NON_FIFA)}


if __name__ == "__main__":
    print(f"registry: {len(load())} teams")
    print(f"active (last {ACTIVE_YEARS}y): {len(active_teams())}")
    for k, v in dated_coverage().items():
        print(f"  {k:<16} {v}")
    missing = unclassified_active()
    print(f"\nunclassified active: {len(missing)}")
    for name in missing:
        print(f"  {name}")
