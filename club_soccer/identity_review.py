#!/usr/bin/env python3
"""Export a report of unresolved club identities.

This module is deliberately read-only. Similar names are useful review hints,
not authority to rewrite ratings history. Confirmed aliases belong in the
reviewed ``data/club_alias_map.json`` artifact and are applied only by
``fetch.write_fixtures``.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
from pathlib import Path

import pandas as pd

from . import club_identity as CI
from .competitions import get as comp_get, teams_n_for

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REVIEW_CSV = DATA / "identity_review.csv"
DUPLICATE_CSV = DATA / "identity_duplicates.csv"
VERDICTS = DATA / "identity_verdicts.json"
FIELDS = [
    "assessment",
    "europe_only_name",
    "registry_country",
    "suggested_match",
    "suggested_league",
    "confidence",
    "n_matches",
    "seasons",
    "reason",
]


def _load_prior_decisions() -> dict:
    """Read historical decisions only to avoid relisting settled identities."""
    if not VERDICTS.exists():
        return {}
    try:
        value = json.loads(VERDICTS.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def domestic_index(df: pd.DataFrame) -> dict[str, str]:
    """Domestic club display name -> competition."""
    out: dict[str, str] = {}
    for comp_name, home, away in zip(df["competition"], df["home"], df["away"]):
        comp = comp_get(comp_name)
        if comp is None or comp.kind != "league":
            continue
        out.setdefault(str(home), str(comp_name))
        out.setdefault(str(away), str(comp_name))
    return out


def _registry_country(name: str) -> str:
    try:
        from .club_registry import country_of

        return str(country_of(name) or "")
    except (FileNotFoundError, ValueError):
        return ""


def _registry_confirms(a: str, b: str) -> bool:
    try:
        from .club_registry import confirms_same_club

        return bool(confirms_same_club(a, b))
    except (FileNotFoundError, ValueError):
        return False


def _suggest(name: str, domestic: dict[str, str],
             opponents: set[frozenset]) -> tuple[str, float, str]:
    """Best report hint; never consumed by an automatic writer."""
    confirmed = [
        candidate for candidate in domestic
        if frozenset((name, candidate)) not in opponents
        and _registry_confirms(name, candidate)
    ]
    if confirmed:
        candidate = max(
            confirmed,
            key=lambda value: difflib.SequenceMatcher(
                None, CI._core(name), CI._core(value)
            ).ratio(),
        )
        return candidate, 1.0, "club registry confirms the same identity"

    best = ("", 0.0, "")
    for candidate in domestic:
        if frozenset((name, candidate)) in opponents:
            continue
        plausible, reason = CI._affinity(name, candidate)
        if not plausible:
            continue
        score = difflib.SequenceMatcher(
            None, CI._core(name), CI._core(candidate)
        ).ratio()
        if score > best[1]:
            best = (candidate, round(score, 3), f"name-only hint: {reason}")
    return best


def build_rows(fixtures: pd.DataFrame | None = None) -> list[dict]:
    df = (
        pd.read_csv(CI.FIXTURES, low_memory=False)
        if fixtures is None else fixtures.copy()
    )
    europe = CI.europe_only_teams(df)
    domestic = domestic_index(df)
    opponents = CI._head_to_head(df)
    prior = _load_prior_decisions()
    counts = pd.concat([df["home"], df["away"]]).value_counts().to_dict()

    rows: list[dict] = []
    for name, seasons in europe.items():
        if name in prior:
            continue
        candidate, confidence, reason = _suggest(name, domestic, opponents)
        country = _registry_country(name)
        if confidence == 1.0:
            assessment = "confirmed registry alias"
        elif candidate:
            assessment = "manual review required"
        elif country:
            assessment = "no domestic identity found"
            reason = f"registry country: {country}"
        else:
            assessment = "unresolved"
            reason = "no registry identity or plausible domestic name"
        rows.append({
            "assessment": assessment,
            "europe_only_name": name,
            "registry_country": country,
            "suggested_match": candidate,
            "suggested_league": domestic.get(candidate, ""),
            "confidence": confidence if candidate else "",
            "n_matches": int(counts.get(name, 0)),
            "seasons": ",".join(str(value) for value in sorted(seasons)),
            "reason": reason,
        })
    rows.sort(key=lambda row: (-row["n_matches"], row["europe_only_name"]))
    return rows


def export(path: Path = REVIEW_CSV) -> Path:
    rows = build_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    confirmed = sum(row["assessment"] == "confirmed registry alias" for row in rows)
    print(f"wrote read-only identity report -> {path}")
    print(f"  unresolved={len(rows)} registry-confirmed suggestions={confirmed}")
    print("  To accept an alias, review and edit data/club_alias_map.json explicitly.")
    return path


DUPLICATE_FIELDS = [
    "competition",
    "season",
    "name_a",
    "name_b",
    "n_matches_a",
    "n_matches_b",
    "teams_seen",
    "teams_expected",
    "confidence",
    "reason",
]


def _same_league_affinity(a: str, b: str) -> tuple[bool, str]:
    """Stricter than ``CI._affinity``, for pairs already inside one league.

    ``CI._affinity`` scores a Europe-only name against a small pool of
    plausible domestic clubs, so a loose hint is cheap and a miss is
    expensive. Sweeping every pair within a league inverts that: the pool is
    every club in the division, and ``_affinity``'s "shared token of >= 5
    characters" rule fires on the city, not the club. It paired CSKA Moscow
    with Dynamo Moscow, Levski Sofia with Lokomotiv Sofia, and DC United with
    Minnesota United — 30-odd rows of noise that bury the real findings.

    Here the only accepted evidence is core equivalence: identical cores after
    dropping club tokens and years, or one core being a strict token subset of
    the other (the "Sønderjyske" / "Sønderjyske Fodbold" shape). City-sharing
    and fuzzy ratios are deliberately not accepted.
    """
    core_a, core_b = CI._core(a), CI._core(b)
    if not core_a or not core_b:
        return False, "empty core"
    if core_a == core_b:
        return True, "identical core"
    tokens_a, tokens_b = set(core_a.split()), set(core_b.split())
    if not tokens_a or not tokens_b or tokens_a == tokens_b:
        return False, "no distinguishing tokens"
    if tokens_a < tokens_b or tokens_b < tokens_a:
        # A strict subset is only meaningful if the shared part is the
        # distinctive bit of the name. "Aalborg" ⊂ "Aalborg Freja" qualifies
        # structurally but they are different clubs, so this stays a hint for
        # review rather than a merge instruction.
        return True, "core token subset"
    return False, "cores differ"


def duplicate_league_identities(
    fixtures: pd.DataFrame | None = None,
) -> list[dict]:
    """Two identities that look like ONE club inside a single league-season.

    ``build_rows`` only inspects Europe-only clubs — names seen in a European
    competition with no domestic league to anchor them. That misses the
    opposite failure: a club whose domestic fixtures arrive from two sources
    under two spellings, so BOTH identities sit in the same league table. The
    Danish Superliga carried three such pairs (Brøndby, Sønderjyske,
    Nordsjælland) and showed 14 teams in a 12-team league for two seasons
    without anything flagging it.

    A pair is reported when the two names never met each other that season (a
    club cannot be its own opponent) and either the club registry confirms
    them as one identity or ``CI._affinity`` finds a name-level hint. Team
    count against ``Competition.teams_n`` is reported as corroboration, not
    used as a filter: an over-count is strong evidence something is wrong, but
    mid-season data gaps mean an exactly-sized league can still be split.

    Read-only, in keeping with the rest of this module. Confirmed pairs belong
    in ``data/club_alias_map.json``.
    """
    df = (
        pd.read_csv(CI.FIXTURES, low_memory=False)
        if fixtures is None else fixtures.copy()
    )
    counts = pd.concat([df["home"], df["away"]]).value_counts().to_dict()
    rows: list[dict] = []

    for (comp_name, season), group in df.groupby(["competition", "season"],
                                                 dropna=False):
        comp = comp_get(str(comp_name))
        if comp is None or comp.kind != "league":
            continue
        teams = sorted({str(t) for t in group["home"]}
                       | {str(t) for t in group["away"]})
        if len(teams) < 2:
            continue
        opponents = CI._head_to_head(group)

        for index, name_a in enumerate(teams):
            for name_b in teams[index + 1:]:
                if frozenset((name_a, name_b)) in opponents:
                    continue
                if _registry_confirms(name_a, name_b):
                    confidence, reason = 1.0, "club registry confirms the same identity"
                else:
                    plausible, hint = _same_league_affinity(name_a, name_b)
                    if not plausible:
                        continue
                    confidence = round(difflib.SequenceMatcher(
                        None, CI._core(name_a), CI._core(name_b)
                    ).ratio(), 3)
                    reason = f"name-only hint: {hint}"
                rows.append({
                    "competition": str(comp_name),
                    "season": "" if pd.isna(season) else int(season),
                    "name_a": name_a,
                    "name_b": name_b,
                    "n_matches_a": int(counts.get(name_a, 0)),
                    "n_matches_b": int(counts.get(name_b, 0)),
                    "teams_seen": len(teams),
                    "teams_expected": teams_n_for(str(comp_name), season) or "",
                    "confidence": confidence,
                    "reason": reason,
                })

    rows.sort(key=lambda row: (-float(row["confidence"]), row["competition"],
                               row["season"], row["name_a"]))
    return rows


def oversized_league_seasons(
    fixtures: pd.DataFrame | None = None,
) -> list[dict]:
    """League-seasons holding more distinct clubs than the competition has.

    A cheap structural invariant that needs no name matching at all: if
    ``Competition.teams_n`` is 12 and the season shows 14 clubs, the data is
    wrong whether or not the extra names look similar to anything.
    """
    df = (
        pd.read_csv(CI.FIXTURES, low_memory=False)
        if fixtures is None else fixtures.copy()
    )
    out: list[dict] = []
    for (comp_name, season), group in df.groupby(["competition", "season"],
                                                 dropna=False):
        comp = comp_get(str(comp_name))
        if comp is None or comp.kind != "league":
            continue
        # Season-aware: a league that has changed size is not corrupt in the
        # seasons before the change (see competitions.TEAMS_N_BY_SEASON).
        expected = teams_n_for(str(comp_name), season)
        if not expected:
            continue
        teams = {str(t) for t in group["home"]} | {str(t) for t in group["away"]}
        if len(teams) > expected:
            out.append({
                "competition": str(comp_name),
                "season": "" if pd.isna(season) else int(season),
                "teams_seen": len(teams),
                "teams_expected": expected,
                "excess": len(teams) - expected,
                "teams": sorted(teams),
            })
    out.sort(key=lambda row: (-row["excess"], row["competition"], row["season"]))
    return out


def export_duplicates(path: Path = DUPLICATE_CSV) -> Path:
    rows = duplicate_league_identities()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DUPLICATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    confirmed = sum(row["confidence"] == 1.0 for row in rows)
    oversized = oversized_league_seasons()
    print(f"wrote same-league duplicate report -> {path}")
    print(f"  suspected pairs={len(rows)} registry-confirmed={confirmed}")
    for row in oversized:
        print(f"  OVERSIZED {row['competition']} {row['season']}: "
              f"{row['teams_seen']} clubs, expected {row['teams_expected']}")
    if rows:
        print("  To merge a pair, add it to data/club_alias_map.json explicitly.")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REVIEW_CSV)
    parser.add_argument("--duplicates", action="store_true",
                        help="report same-league duplicate identities instead")
    parser.add_argument("--duplicates-out", type=Path, default=DUPLICATE_CSV)
    args = parser.parse_args()
    if args.duplicates:
        export_duplicates(args.duplicates_out)
    else:
        export(args.out)


if __name__ == "__main__":
    main()
