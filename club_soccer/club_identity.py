#!/usr/bin/env python3
"""Canonical club identity at the fixture write boundary.

Every fixture writer passes through ``fetch.write_fixtures``. This module keeps
that boundary deliberately small:

* resolve reviewed aliases from ``club_alias_map.json``;
* use the openfootball registry for a portable identity key and country guard;
* preserve unknown names instead of guessing;
* expose a report-only view of unresolved Europe-only identities.

Historical fuzzy merge builders were removed. They duplicated ingest logic,
created local backups, and could mutate production data from statistical
coincidences. Ambiguous new identities are reported by ``identity_review`` and
must be added to the reviewed alias artifact explicitly.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from .competitions import get as _get_comp
from .normalization import normalise_club_text

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
ALIAS_MAP = DATA / "club_alias_map.json"

_CLUB_TOKENS = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "bsc", "gnk", "sk",
    "rc", "vfl", "vfb", "tsg", "fsv", "bv", "sv", "us", "ud", "cd", "rcd",
    "sd", "aik", "if", "ik", "bk", "fk", "nk", "hk", "kv", "rsc", "kaa",
    "psv", "calcio", "balompie", "futbol", "football", "club", "clube", "de",
    "the",
}
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")

_BORDERLESS_ASSOCIATION_LEAGUES = {
    "Monaco": {"France"},
    "Liechtenstein": {"Switzerland"},
}
_CROSS_BORDER_CLUB_LEAGUE = {
    "Cardiff": "England",
    "Cardiff City": "England",
    "Swansea": "England",
    "Swansea City": "England",
    "Newport County": "England",
    "Newport": "England",
    "Wrexham": "England",
    "Wrexham AFC": "England",
    "Merthyr Town": "England",
    "Derry City": "Ireland",
    "Andorra": "Spain",
    "FC Andorra": "Spain",
    "Toronto FC": "USA",
    "Vancouver Whitecaps": "USA",
    "CF Montreal": "USA",
    "CF Montréal": "USA",
    "Montreal Impact": "USA",
}

_resolver_cache: tuple[dict[str, str], dict[str, str]] | None = None
_country_index_cache: dict[str, str] | None = None


def _norm(name) -> str:
    return normalise_club_text(name)


def _core(name) -> str:
    """Conservative comparison key used only by the ambiguity report."""
    text = _YEAR_RE.sub(" ", _norm(name))
    tokens = [token for token in text.split() if token not in _CLUB_TOKENS]
    return " ".join(tokens).strip() or text.strip()


def _affinity(a: str, b: str) -> tuple[bool, str]:
    """Return a report hint, never an automatic merge decision."""
    ca, cb = _core(a), _core(b)
    if not ca or not cb:
        return False, "empty core"
    if ca == cb:
        return True, "identical core"
    ta, tb = set(ca.split()), set(cb.split())
    if ta and tb and (ta <= tb or tb <= ta):
        return True, "core token subset"
    shared = {token for token in ta & tb if len(token) >= 5}
    if shared:
        return True, f"shared token {sorted(shared)[0]!r}"
    ratio = difflib.SequenceMatcher(None, ca, cb).ratio()
    return ratio >= 0.72, f"similarity {ratio:.2f}"


def _head_to_head(df: pd.DataFrame) -> set[frozenset]:
    return {
        frozenset((home, away))
        for home, away in zip(df["home"], df["away"])
        if home != away
    }


def europe_only_teams(df: pd.DataFrame) -> dict[str, set[int]]:
    """Club -> seasons for clubs observed in Europe but no domestic league."""
    domestic: set[str] = set()
    europe: dict[str, set[int]] = {}
    for row in df.itertuples(index=False):
        comp = _get_comp(getattr(row, "competition", ""))
        teams = (getattr(row, "home", ""), getattr(row, "away", ""))
        if comp is None:
            continue
        if comp.kind == "league":
            domestic.update(team for team in teams if team)
        elif comp.kind == "europe":
            raw_season = getattr(row, "season", None)
            try:
                season = int(raw_season) if not pd.isna(raw_season) else 0
            except (TypeError, ValueError):
                season = 0
            for team in teams:
                if team:
                    europe.setdefault(team, set()).add(season)
    return {team: seasons for team, seasons in europe.items() if team not in domestic}


def _load_resolver() -> tuple[dict[str, str], dict[str, str]]:
    """Return reviewed literal aliases and normalised aliases."""
    global _resolver_cache
    if _resolver_cache is not None:
        return _resolver_cache
    alias: dict[str, str] = {}
    if ALIAS_MAP.exists():
        raw = json.loads(ALIAS_MAP.read_text())
        loaded = raw.get("alias", {})
        if not isinstance(loaded, dict):
            raise ValueError("club_alias_map.json alias must be an object")
        alias = {str(source): str(target) for source, target in loaded.items()}

    # Alias chains make resolution depend on how often it is called. Refuse
    # them at load time instead of silently producing unstable identities.
    for source, target in alias.items():
        if target in alias:
            raise ValueError(
                f"club alias chain is not allowed: {source!r} -> {target!r} "
                f"-> {alias[target]!r}"
            )

    by_norm: dict[str, str] = {}
    for target in sorted(set(alias.values())):
        by_norm.setdefault(_norm(target), target)
    for source, target in sorted(alias.items()):
        by_norm.setdefault(_norm(source), target)
    _resolver_cache = (alias, by_norm)
    return _resolver_cache


def reload_resolver() -> None:
    global _resolver_cache
    _resolver_cache = None


def team_countries(refresh: bool = False) -> dict[str, str]:
    """Observed domestic-league country for canonical display identities."""
    global _country_index_cache
    if _country_index_cache is not None and not refresh:
        return _country_index_cache
    index: dict[str, str] = {}
    conflicts: set[str] = set()
    if FIXTURES.exists():
        frame = pd.read_csv(FIXTURES, usecols=["competition", "home", "away"])
        for row in frame.itertuples(index=False):
            comp = _get_comp(row.competition)
            if comp is None or comp.kind == "europe" or not comp.country:
                continue
            for team in (row.home, row.away):
                prior = index.get(team)
                if prior is not None and prior != comp.country:
                    conflicts.add(team)
                else:
                    index[team] = comp.country
    for team in conflicts:
        index.pop(team, None)
    _country_index_cache = index
    return index


def reset_country_index() -> None:
    global _country_index_cache
    _country_index_cache = None


def _registry_record(name: str) -> dict | None:
    try:
        from .club_registry import lookup

        record = lookup(name)
    except (FileNotFoundError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _country_allows(target: str, country: str | None,
                    registry_country: str | None = None) -> bool:
    if not country:
        return True
    observed = team_countries().get(target)
    association = observed or registry_country
    if not association or association == country:
        return True
    if country in _BORDERLESS_ASSOCIATION_LEAGUES.get(association, set()):
        return True
    return _CROSS_BORDER_CLUB_LEAGUE.get(target) == country


def canonical_name(name, country: str | None = None):
    """Map a provider spelling onto the reviewed project display identity.

    Unknown or ambiguous names pass through. The registry is used to recognise
    aliases only when its canonical record already maps to a reviewed project
    identity; adopting all registry display names would rename hundreds of
    established teams without improving identity.
    """
    if not name:
        return name
    text = str(name)
    alias, by_norm = _load_resolver()
    target = alias.get(text) or by_norm.get(_norm(text))
    record = _registry_record(text)
    if target is None and record and not record.get("ambiguous"):
        registry_name = str(record.get("canonical") or "")
        target = alias.get(registry_name) or by_norm.get(_norm(registry_name))
    if target is None:
        return text
    registry_country = None if not record else record.get("country")
    return target if _country_allows(target, country, registry_country) else text


def canonicalise(raw_name, country_hint: str | None = None):
    """The only ingest-facing display-name canonicaliser."""
    return canonical_name(raw_name, country=country_hint)


def canonical_id(raw_name, country_hint: str | None = None) -> str:
    """Portable club key backed by registry identity where unambiguous."""
    canonical = canonical_name(raw_name, country=country_hint)
    record = _registry_record(str(raw_name)) or _registry_record(str(canonical))
    identity_name = canonical
    country = None
    if record:
        countries = {str(value) for value in record.get("countries", []) if value}
        record_country = record.get("country")
        observed_country = team_countries().get(canonical)
        # A registry hit is not permission to ignore the country guard. For
        # example, the German registry record for "FC Bayern München" must not
        # give a Brazilian club with that literal name the German club's ID
        # after canonical_name() correctly refused the display-name alias.
        registry_country_allowed = _country_allows(
            canonical, country_hint, record_country
        )
        can_disambiguate = (
            registry_country_allowed
            and (
                not record.get("ambiguous")
                or bool(country_hint and country_hint in countries)
                or bool(observed_country and observed_country in countries)
            )
        )
        if can_disambiguate:
            identity_name = str(record.get("canonical") or canonical)
            country = country_hint if country_hint in countries else (
                observed_country if observed_country in countries else record_country
            )
    country = country or team_countries().get(canonical) or country_hint
    scope = _norm(country or "")
    name_key = _norm(identity_name)
    return hashlib.sha256(f"{scope}|{name_key}".encode()).hexdigest()[:20]
