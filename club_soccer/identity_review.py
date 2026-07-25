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
from .competitions import get as comp_get

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REVIEW_CSV = DATA / "identity_review.csv"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REVIEW_CSV)
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
