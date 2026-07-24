#!/usr/bin/env python3
"""Club identity resolution — one club, one identity, across every source.

The problem
-----------
fixtures.csv is assembled from several providers (BSD, openfootball,
football-data.org, football-data.co.uk) that spell clubs differently. names.py
defines a canon for this, but it is applied only on some ingest paths, so the
same club exists under two names:

    Bayern Munich            156 matches   2022-08-05 .. 2026-05-16
    FC Bayern München        138 matches   2024-08-25 .. 2027-05-22

Those date ranges OVERLAP, and inspection shows the overlap is the same matches
recorded twice with identical scores. identities.dedupe_fixtures keys on
(date, competition, home, away) — with different spellings the keys differ, so
deduplication silently misses every one of them.

Two distinct harms, and the second is the serious one:

  1. Fragmentation: a club's rating is split across two identities, each fitted
     on part of its history. Bayern, Inter, Hearts and Ajax are all affected.
  2. Double-counting: ~2,100 corroborated duplicate rows feed the fit TWICE,
     inflating their weight in the attack/defence stats and running them
     through the Elo sequence twice.

This is upstream of every modelling question. Improving the model while the
inputs are duplicated is not worth doing.

Matching strategy
-----------------
Candidates come from HARD EVIDENCE, not from string similarity: two names are
candidate aliases when a match on the same date, in the same competition, with
the same score, has one side spelled identically and the other differently.
That is close to proof — but only close, so it is then filtered by guards.

Guards (all must pass; each exists because it caught a real false positive or
a real risk):

  G1  Head-to-head veto. Two identities that ever appear as opponents in the
      same match are different clubs. This alone killed the observed false
      positive "Bolton Wanderers == Wolverhampton" (2 Championship meetings),
      which chance-collided on date+competition+score with a shared opponent.
  G2  Country veto. Names resolving to different countries never merge.
  G3  Reserve/youth veto. "II", "B", "U19/U21/U23", "Reserves" never merge into
      the senior side — they are genuinely different teams.
  G4  Name affinity. Beyond evidence, the names must plausibly denote the same
      club: equal cores after stripping club-type tokens, or one core's
      distinctive tokens contained in the other, or difflib ratio >= 0.72.
  G5  Evidence floor. At least MIN_EVIDENCE corroborating collisions, unless
      the normalised cores are exactly equal.

Everything is written to a reviewable artifact (data/club_alias_map.json)
before it is applied. The map is a pure function of fixtures.csv, so it can be
re-derived and re-run at any time.

CLI:
  python3 -m club_soccer.club_identity --report      # propose, do not write
  python3 -m club_soccer.club_identity --write       # write the alias map
  python3 -m club_soccer.club_identity --apply       # rewrite fixtures.csv
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .competitions import get as _get_comp

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
ALIAS_MAP = DATA / "club_alias_map.json"

MIN_EVIDENCE = 3
SIMILARITY_CUTOFF = 0.72

# Human-reviewed merges the automatic rules decline, kept explicit rather than
# handled by loosening the heuristic. Each was rejected by G4 (abbreviations
# and "Stade X"/"X" forms have low string affinity) or by the G5 evidence
# floor, and each was checked by hand.
#
# The threshold is NOT loosened because a genuine false positive sits right
# beside these in similarity space: "Reims" / "Stade Rennais" scores 0.44 with
# 18 corroborating collisions, and they are different clubs. "Rennes" /
# "Stade Rennais" scores 0.53. Any cutoff loose enough to catch the real pairs
# automatically is uncomfortably close to admitting the false one, so the
# residue is curated instead.
MANUAL_ALIASES = {
    "Stade Rennais": "Rennes",
    "Queens Park Rangers": "QPR",
    "RCD Espanyol de Barcelona": "Espanyol",
    "Espanol": "Espanyol",
    "PSV Eindhoven": "PSV",
    "Stade Brestois 29": "Brest",
    "Stade Brestois": "Brest",
    "PAE Olympiakos SFP": "Olympiacos FC",
    "Paphos FC": "Pafos FC",
    # Direction fixes: without these the most-seen spelling wins and a
    # misspelling or abbreviation becomes the canonical identity.
    "Sheffield Utd": "Sheffield United",
    "Sheffield Weds": "Sheffield Wednesday",
    "Vallecano": "Rayo Vallecano",
    "Hamburg": "Hamburger SV",
    "Espanol": "Espanyol",
}

# Cross-border league membership (adversarial finding 11). A club's ASSOCIATION
# country (what the registry knows) can legitimately differ from the LEAGUE it
# plays in. Before the domestic-league index is built (bootstrap), or for a club
# only ever seen in continental play, canonical_name falls back to the
# association country and would then WRONGLY refuse these real aliases — AS
# Monaco -> Monaco was refused because the registry says "Monaco", not "France".
# League membership is modelled here explicitly and as an ALLOWLIST: an unknown
# club keeps the full guard, so no new cross-country weld is possible.
#
# Borderless associations have no domestic league of their own — every one of
# their clubs plays in a neighbouring pyramid, so any such mismatch is fine.
_BORDERLESS_ASSOCIATION_LEAGUES = {
    "Monaco": {"France"},
    "Liechtenstein": {"Switzerland"},
}
# Specific clubs that opt into a foreign pyramid while their association keeps
# its own league (Welsh & NI clubs in England, Derry in the Republic of Ireland,
# Canadian MLS sides in the US league, FC Andorra in Spain).
_CROSS_BORDER_CLUB_LEAGUE = {
    "Cardiff": "England", "Cardiff City": "England",
    "Swansea": "England", "Swansea City": "England",
    "Newport County": "England", "Newport": "England",
    "Wrexham": "England", "Wrexham AFC": "England",
    "Merthyr Town": "England",
    "Derry City": "Ireland",
    "Andorra": "Spain", "FC Andorra": "Spain",
    "Toronto FC": "USA", "Vancouver Whitecaps": "USA",
    "CF Montreal": "USA", "CF Montréal": "USA", "Montreal Impact": "USA",
}

# Club-type tokens and legal/founding cruft. Stripped when comparing names, so
# "Bologna FC 1909" and "Bologna" share a core. Deliberately conservative:
# tokens that can distinguish real clubs (e.g. "Real", "Athletic", "Sporting",
# "Dynamo") are NOT here, because "Real Sociedad"/"Sociedad" is a safe strip
# but "Sporting CP"/"Sporting Braga" is not.
_CLUB_TOKENS = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "bsc", "gnk", "sk", "rc",
    "vfl", "vfb", "tsg", "fsv", "bv", "sv", "us", "ud", "cd", "rcd", "sd",
    "aik", "if", "ik", "bk", "fk", "nk", "hk", "kv", "rsc", "kaa", "psv",
    "calcio", "balompie", "futbol", "football", "club", "clube", "de", "the",
}
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")
_RESERVE_RE = re.compile(r"\b(ii|iii|b|u\s?1[6-9]|u\s?2[0-3]|reserves?|youth|academy)\b")


def _norm(name) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _core(name) -> str:
    """Normalised name minus club-type tokens and founding years."""
    text = _YEAR_RE.sub(" ", _norm(name))
    tokens = [t for t in text.split() if t not in _CLUB_TOKENS]
    return " ".join(tokens).strip() or text.strip()


def _is_reserve(name) -> bool:
    return bool(_RESERVE_RE.search(_norm(name)))


def _affinity(a: str, b: str) -> tuple[bool, str]:
    """G4: do these two names plausibly denote the same club?"""
    ca, cb = _core(a), _core(b)
    if not ca or not cb:
        return False, "empty core"
    if ca == cb:
        return True, "identical core"
    ta, tb = set(ca.split()), set(cb.split())
    if ta and tb and (ta <= tb or tb <= ta):
        return True, "core token subset"
    # A shared distinctive token (>= 5 chars) carries most of the signal for
    # cases like "Brest" / "Stade Brestois" or "Lyon" / "Olympique Lyonnais",
    # where raw string similarity is low but the club is obvious.
    shared = {t for t in (ta & tb) if len(t) >= 5}
    if shared:
        return True, f"shared token {sorted(shared)[0]!r}"
    for x in ta:
        for y in tb:
            if len(x) >= 4 and len(y) >= 4 and (x.startswith(y) or y.startswith(x)):
                return True, f"token stem {x!r}/{y!r}"
    ratio = difflib.SequenceMatcher(None, ca, cb).ratio()
    if ratio >= SIMILARITY_CUTOFF:
        return True, f"similarity {ratio:.2f}"
    return False, f"no affinity (similarity {ratio:.2f})"


def _team_countries(df: pd.DataFrame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = collections.defaultdict(set)
    comp_country: dict[str, str] = {}
    for comp in df["competition"].dropna().unique():
        c = _get_comp(comp)
        # European competitions say nothing about a club's country.
        if c and c.kind != "europe" and c.country:
            comp_country[comp] = c.country
    for comp, home, away in zip(df["competition"], df["home"], df["away"]):
        country = comp_country.get(comp)
        if country:
            out[home].add(country)
            out[away].add(country)
    return out


def _head_to_head(df: pd.DataFrame) -> set[frozenset]:
    return {frozenset((h, a)) for h, a in zip(df["home"], df["away"]) if h != a}


def collect_evidence(df: pd.DataFrame) -> collections.Counter:
    """Candidate alias pairs, weighted by corroborating duplicate matches."""
    slots: dict[tuple, list] = collections.defaultdict(list)
    for r in df.itertuples(index=False):
        if pd.isna(getattr(r, "home_goals", None)):
            continue
        key = (str(r.date)[:10], r.competition, r.home_goals, r.away_goals)
        slots[key].append((r.home, r.away))
    pairs: collections.Counter = collections.Counter()
    for _key, entries in slots.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                (h1, a1), (h2, a2) = entries[i], entries[j]
                if _norm(h1) == _norm(h2) and _norm(a1) != _norm(a2):
                    pairs[tuple(sorted((a1, a2)))] += 1
                elif _norm(a1) == _norm(a2) and _norm(h1) != _norm(h2):
                    pairs[tuple(sorted((h1, h2)))] += 1
    return pairs


def normalisation_groups(counts: collections.Counter) -> list[list[str]]:
    """Names identical once accents, punctuation and case are normalised.

    These need no match evidence — they are the same string, so they are the
    same club by definition. They also cannot be found by collect_evidence,
    which compares normalised names and therefore treats them as already
    equal, so they were invisible in the first pass despite being distinct
    identities to the model.

    They matter: "Atletico Madrid" (128 matches) and "Atlético Madrid" (109)
    split one club almost down the middle. dedupe_fixtures normalises its
    match key, so the duplicate ROWS were already collapsing — but it keeps
    whichever spelling it saw first, leaving the TEAM identity split. That is
    the fragmentation this module exists to remove.
    """
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for name in counts:
        groups[_norm(name)].append(name)
    return [sorted(v) for v in groups.values() if len(v) > 1]


def build_alias_map(df: pd.DataFrame | None = None) -> dict:
    """Derive the alias map. Pure function of the fixture frame."""
    df = pd.read_csv(FIXTURES, low_memory=False) if df is None else df
    evidence = collect_evidence(df)
    h2h = _head_to_head(df)
    countries = _team_countries(df)
    counts = collections.Counter()
    for side in ("home", "away"):
        counts.update(df[side].dropna())

    accepted, rejected = [], []
    for (a, b), n in evidence.most_common():
        if frozenset((a, b)) in h2h:
            rejected.append({"a": a, "b": b, "evidence": n,
                             "reason": "G1 head-to-head: they have played each other"})
            continue
        ca, cb = countries.get(a, set()), countries.get(b, set())
        if ca and cb and not (ca & cb):
            rejected.append({"a": a, "b": b, "evidence": n,
                             "reason": f"G2 country mismatch {sorted(ca)} vs {sorted(cb)}"})
            continue
        if _is_reserve(a) != _is_reserve(b):
            rejected.append({"a": a, "b": b, "evidence": n,
                             "reason": "G3 reserve/youth vs senior side"})
            continue
        ok, why = _affinity(a, b)
        if not ok:
            rejected.append({"a": a, "b": b, "evidence": n, "reason": f"G4 {why}"})
            continue
        if n < MIN_EVIDENCE and _core(a) != _core(b):
            rejected.append({"a": a, "b": b, "evidence": n,
                             "reason": f"G5 evidence below floor ({n} < {MIN_EVIDENCE})"})
            continue
        accepted.append({"a": a, "b": b, "evidence": n, "affinity": why})

    # Canonical name: prefer the spelling names.py already treats as the canon
    # TARGET (a value in its alias tables but not itself a key), so this merge
    # agrees with the existing seeders instead of fighting them. Fall back to
    # the most-seen spelling, which minimises downstream churn in odds joins.
    from . import names as _names
    _canon_keys = set(_names.OVERRIDES) | set(_names.FDCOUK_ALIASES)
    _canon_values = set(_names.OVERRIDES.values()) | set(_names.FDCOUK_ALIASES.values())

    def _rank(name: str) -> tuple:
        established = name in _canon_values and name not in _canon_keys
        return (1 if established else 0, 1 if name in _canon_values else 0,
                counts[name])

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    # Pure normalisation variants first — no evidence needed, no guards
    # applicable (a name cannot have played itself, and accents never change
    # the club).
    norm_groups = normalisation_groups(counts)
    for group in norm_groups:
        for other in group[1:]:
            ra, rb = find(group[0]), find(other)
            if ra != rb:
                parent[rb] = ra
        accepted.append({"a": group[0], "b": " / ".join(group[1:]),
                         "evidence": sum(counts[g] for g in group),
                         "affinity": "identical after normalisation"})

    for rec in accepted:
        if rec.get("affinity") == "identical after normalisation":
            continue
        ra, rb = find(rec["a"]), find(rec["b"])
        if ra != rb:
            parent[rb] = ra

    # Human-reviewed merges join their groups too.
    present = set(counts)
    for src, dst in MANUAL_ALIASES.items():
        if src not in present and dst not in present:
            continue
        rs, rd = find(src), find(dst)
        if rs != rd:
            parent[rs] = rd
        accepted.append({"a": src, "b": dst, "evidence": int(counts.get(src, 0)),
                         "affinity": "manual (human-reviewed)"})

    # Choose each group's canonical name in a SEPARATE pass. Doing it during
    # union-find made the result depend on merge order, which is why "Espanol"
    # (a misspelling) and "Sheffield Utd" beat their proper forms. A reviewed
    # name always wins; otherwise the established-canon/most-seen ranking does.
    members: dict[str, set[str]] = collections.defaultdict(set)
    for name in present:
        members[find(name)].add(name)

    _manual_targets = set(MANUAL_ALIASES.values())
    alias: dict[str, str] = {}
    for group in members.values():
        reviewed = sorted(group & _manual_targets)
        canon = reviewed[0] if reviewed else max(group, key=_rank)
        for name in group:
            if name != canon and name in present:
                alias[name] = canon
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for src, dst in alias.items():
        groups[dst].append(src)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_rows": int(len(df)),
        "min_evidence": MIN_EVIDENCE,
        "similarity_cutoff": SIMILARITY_CUTOFF,
        "alias": dict(sorted(alias.items())),
        "groups": {k: sorted(v) for k, v in sorted(groups.items())},
        "accepted": accepted,
        "rejected": rejected,
    }


# ── cross-source reconciliation (P3) ──────────────────────────────────────
# A different problem from build_alias_map, needing a different mechanism.
#
# When a domestic league is ingested for the first time, its clubs arrive under
# the new source's spelling while the SAME clubs may already exist in
# fixtures.csv under a European-coverage spelling: fd.co.uk says "Sturm Graz",
# BSD already gave us "SK Sturm Graz". Left alone this re-creates the exact
# fragmentation P1 removed, on the very club that prompted the work.
#
# The P1 evidence matcher cannot help. It keys on same date+competition+score
# collisions, and an Austrian Bundesliga match never collides with a Champions
# League one — the two identities share no fixture. canonical_name() cannot
# help either: it passes unknown names through by design, which is correct for
# genuinely new clubs and useless here.
#
# So match on name affinity, scoped by country, with corroboration from season
# overlap: a club playing in Europe in season S was, by construction, in its
# domestic league in season S or S-1. Absence of overlap is strong evidence
# that two similarly-named clubs are not the same club.

# Cross-source merges confirmed by hand against the ingested league rosters.
# Everything here was proposed by the matcher but scored below the auto-accept
# bar; each was checked against the actual team list for that country.
DOMESTIC_MANUAL_MERGES = {
    # Portugal
    "Sporting Clube de Portugal": "Sp Lisbon",
    "Sporting CP": "Sp Lisbon",
    "Sporting Braga": "Sp Braga",
    "Sport Lisboa e Benfica": "Benfica",
    "Vitória Guimarães": "Guimaraes",
    # Belgium
    "Union Saint-Gilloise": "St. Gilloise",
    "Royale Union Saint-Gilloise": "St. Gilloise",
    "KRC Genk": "Genk",
    # Austria
    "Rapid Wien": "SK Rapid",
    "Red Bull Salzburg": "Salzburg",
    "FC Red Bull Salzburg": "Salzburg",
    # Turkey — fd.co.uk spells Başakşehir "Buyuksehyr"
    "İstanbul Başakşehir": "Buyuksehyr",
    "Istanbul Basaksehir": "Buyuksehyr",
    "Beşiktaş JK": "Besiktas",
    # Greece
    "Olympiakos Piraeus": "Olympiakos",
    "Olympiacos FC": "Olympiakos",
    # Netherlands
    "NEC Nijmegen": "Nijmegen",
    # Sweden / Norway — accent-only differences the matcher scored below the
    # auto bar because the source strips diacritics entirely.
    "Mjällby AIF": "Mjallby",
    "Tromsø IL": "Tromso",
    # Romania. Three Craiova spellings exist in the source and they are NOT
    # one club: "Univ. Craiova" (CSU Craiova, the European participant) and
    # "U Craiova 1948" (FCU Craiova 1948) are genuinely different. "U Craiova"
    # is a spelling fd.co.uk used for 7 matches in 2020 before switching to
    # "Univ. Craiova" — no shared dates, no head-to-head, so same club.
    "U Craiova": "Univ. Craiova",
    "Universitatea Craiova": "Univ. Craiova",
    "FC Universitatea Cluj": "U. Cluj",
}

# Proposals rejected on review. Recorded explicitly so a future re-run cannot
# quietly re-propose them, and so the reasoning survives.
DOMESTIC_MERGE_BLOCKLIST = {
    ("AEK Larnaca", "AEK"),                    # Cyprus vs AEK Athens
    ("AC Sparta Praha", "Sparta Rotterdam"),   # Czechia vs Netherlands
    ("FC Spartak Trnava", "Sparta Rotterdam"),  # Slovakia vs Netherlands
    ("Ħamrun Spartans FC", "Sparta Rotterdam"),  # Malta vs Netherlands
    ("IF Vestri", "Estoril"),                  # Iceland vs Portugal
    ("İstanbul Başakşehir", "Istanbulspor"),   # two different Istanbul clubs
    ("Samsunspor", "Sivasspor"),               # two different Turkish clubs
    ("Cercle Brugge", "Club Brugge"),          # two different Bruges clubs
    ("CFR 1907 Cluj", "U. Cluj"),              # CFR Cluj vs Universitatea Cluj
    ("Víkingur Gøta", "Viking"),               # Faroe Islands vs Norway
    ("GNK Dinamo Zagreb", "Dinamo Bucuresti"),  # Croatia vs Romania
}


def europe_only_teams(df: pd.DataFrame) -> dict[str, set[int]]:
    """Existing teams seen ONLY in UEFA competitions -> seasons they appear in."""
    kinds = {}
    for comp in df["competition"].dropna().unique():
        c = _get_comp(comp)
        kinds[comp] = c.kind if c else "league"
    seasons: dict[str, set[int]] = collections.defaultdict(set)
    domestic: set[str] = set()
    for comp, season, home, away in zip(df["competition"], df.get("season", []),
                                        df["home"], df["away"]):
        kind = kinds.get(comp, "league")
        for team in (home, away):
            if kind == "europe":
                try:
                    seasons[team].add(int(season))
                except (TypeError, ValueError):
                    pass
            else:
                domestic.add(team)
    return {t: s for t, s in seasons.items() if t not in domestic}


def propose_domestic_merges(existing: pd.DataFrame, incoming: pd.DataFrame,
                            season_slack: int = 1) -> dict:
    """Match newly ingested domestic clubs onto existing Europe-only identities.

    Returns accepted/rejected proposals for review. Nothing is applied here.
    """
    euro = europe_only_teams(existing)
    h2h = _head_to_head(existing)

    incoming_seasons: dict[str, set[int]] = collections.defaultdict(set)
    incoming_country: dict[str, set[str]] = collections.defaultdict(set)
    for comp, season, home, away in zip(incoming["competition"], incoming["season"],
                                        incoming["home"], incoming["away"]):
        c = _get_comp(comp)
        for team in (home, away):
            try:
                incoming_seasons[team].add(int(season))
            except (TypeError, ValueError):
                pass
            if c and c.country:
                incoming_country[team].add(c.country)

    existing_names = set(existing["home"].dropna()) | set(existing["away"].dropna())
    incoming_norm = {_norm(t) for t in incoming_seasons}

    # Match dates per identity — the decisive guard. One club cannot play two
    # matches on the same day, so a date collision between a Europe-only
    # identity and a candidate domestic identity proves they are different
    # clubs. Without this, "AC Sparta Praha" merges into "Sparta Rotterdam"
    # and "FC Spartak Trnava" and "Ħamrun Spartans FC" follow it, because
    # nothing else in the data distinguishes clubs across countries.
    def _dates(df: pd.DataFrame) -> dict[str, set[str]]:
        out: dict[str, set[str]] = collections.defaultdict(set)
        for date, home, away in zip(df["date"], df["home"], df["away"]):
            day = str(date)[:10]
            out[home].add(day)
            out[away].add(day)
        return out

    euro_dates = _dates(existing)
    new_dates = _dates(incoming)

    try:
        from .club_registry import confirms_same_club as reg_confirms
    except Exception:
        reg_confirms = None

    accepted, rejected, review = [], [], []
    claimed: dict[str, tuple[float, str]] = {}
    # Cross-source matching has NO fixture-level corroboration — the two
    # identities share no match — so only near-certain pairs auto-merge.
    # Everything between MIN_SCORE and AUTO_SCORE goes to review rather than
    # being applied. The mid-range is where the cross-country false positives
    # live: "AC Sparta Praha"/"Sparta Rotterdam" scored 0.64 and
    # "AEK Larnaca"/"AEK" (Cyprus vs Athens) 0.43, both entirely plausible to
    # a string matcher and both wrong. Guessing here corrupts a club's whole
    # rating history, so the residue is curated instead.
    MIN_SCORE = 0.62
    AUTO_SCORE = 0.80

    for new_name in sorted(incoming_seasons):
        # Already a known identity — nothing to reconcile.
        if new_name in existing_names:
            continue
        for euro_name, euro_seasons in euro.items():
            # If the Europe-only club also appears in the incoming league under
            # its own spelling, it is already reconciled and must not be
            # absorbed by a similarly-named neighbour. This is what stopped
            # "Cercle Brugge" being swallowed by "Club Brugge".
            if _norm(euro_name) in incoming_norm:
                continue
            if (euro_name, new_name) in DOMESTIC_MERGE_BLOCKLIST:
                rejected.append({"new": new_name, "existing": euro_name,
                                 "reason": "blocklisted on review"})
                continue
            # External reference veto. openfootball/clubs knows the country of
            # 3,552 clubs including ones we have never seen, and it catches
            # every cross-country false positive this project found by hand
            # (Sparta Praha/Sparta Rotterdam, AEK Larnaca/AEK, Dinamo Zagreb/
            # Dinamo Bucuresti) with no false vetoes on the true merges. It
            # cannot separate same-country rivals — that still needs review.
            try:
                from .club_registry import same_club_possible
                possible, why = same_club_possible(euro_name, new_name)
            except Exception:
                possible, why = True, "reference unavailable"
            if not possible:
                rejected.append({"new": new_name, "existing": euro_name,
                                 "reason": f"club registry: {why}"})
                continue
            if frozenset((new_name, euro_name)) in h2h:
                rejected.append({"new": new_name, "existing": euro_name,
                                 "reason": "G1 head-to-head"})
                continue
            ok, why = _affinity(new_name, euro_name)
            if not ok:
                continue
            # Same-day collisions are RECORDED, not vetoed. The original design
            # treated any collision as proof of two distinct clubs — one club
            # cannot play twice in a day. Sound in principle, wrong on this
            # data. Measured against known pairs:
            #
            #   FC Twente / Twente                    1 clash  (SAME club)
            #   Panathinaikos FC / Panathinaikos      1 clash  (SAME club)
            #   Legia Warszawa / Legia                2 clashes (SAME club)
            #   AC Sparta Praha / Sparta Rotterdam    0 clashes (different)
            #   GNK Dinamo Zagreb / Dinamo Bucuresti  0 clashes (different)
            #   AEK Larnaca / AEK                     0 clashes (different)
            #
            # Exactly backwards. Collisions come from the two sources
            # disagreeing about a date, which only happens when both sources
            # describe the SAME club; clubs from different countries never
            # coincide by chance. The veto was blocking true merges — it is why
            # FC Twente stayed split through the P3 ingest — while the false
            # positives were being caught by the affinity floor and the
            # already-present veto instead.
            clash = euro_dates.get(euro_name, set()) & new_dates.get(new_name, set())
            # Season corroboration: a European campaign implies domestic play
            # in the same or adjacent season.
            overlap = any(abs(a - b) <= season_slack
                          for a in euro_seasons for b in incoming_seasons[new_name])
            if not overlap:
                rejected.append({
                    "new": new_name, "existing": euro_name,
                    "reason": (f"no season overlap (europe {sorted(euro_seasons)} vs "
                               f"domestic {sorted(incoming_seasons[new_name])})")})
                continue
            score = difflib.SequenceMatcher(None, _core(new_name), _core(euro_name)).ratio()
            if score < MIN_SCORE:
                rejected.append({"new": new_name, "existing": euro_name,
                                 "reason": f"score {score:.2f} below {MIN_SCORE} ({why})"})
                continue
            # Auto-merge fails CLOSED. The registry must POSITIVELY confirm the
            # two are the same club (both known, same country) before a merge
            # applies without review. "not in reference" and "ambiguous" are NOT
            # confirmation. This is the guard that was missing: "CSKA Sofia" and
            # "FC CSKA 1948 Sofia" both reduce to core "cska sofia" (FC and the
            # founding year are stripped), so the identical-core fast-path
            # auto-accepted two genuine Bulgarian rivals the registry couldn't
            # object to. And a same-day collision — ambiguous on its own — is
            # only trusted when the registry also confirms; otherwise it is a
            # review trigger, restoring the veto that was wrongly dropped.
            confirmed = bool(reg_confirms and reg_confirms(new_name, euro_name))
            if not confirmed or clash:
                review.append({"new": new_name, "existing": euro_name,
                               "score": round(score, 3), "affinity": why,
                               "same_day_collisions": len(clash),
                               "reason": ("registry could not confirm same club"
                                          if not confirmed else
                                          "same-day collision without confirmation")})
                continue
            prev = claimed.get(euro_name)
            if prev and prev[0] >= score:
                continue
            claimed[euro_name] = (score, new_name)
            accepted.append({"new": new_name, "existing": euro_name,
                             "affinity": why, "score": round(score, 3),
                             "same_day_collisions": len(clash),
                             "europe_seasons": sorted(euro_seasons)})

    # One-to-one both ways: best claim per existing identity, and a new
    # identity may absorb at most one Europe-only spelling.
    best = {euro: name for euro, (_score, name) in claimed.items()}
    accepted = [a for a in accepted if best.get(a["existing"]) == a["new"]]
    by_new: dict[str, dict] = {}
    for a in sorted(accepted, key=lambda x: -x["score"]):
        by_new.setdefault(a["new"], a)
    accepted = list(by_new.values())

    # Reviewed merges are applied regardless of score, provided both identities
    # are actually present in this data.
    incoming_names = set(incoming_seasons)
    auto_targets = {a["existing"] for a in accepted}
    for euro_name, new_name in DOMESTIC_MANUAL_MERGES.items():
        if euro_name in auto_targets:
            continue          # already resolved automatically
        if euro_name in euro and new_name in incoming_names:
            accepted.append({"new": new_name, "existing": euro_name,
                             "affinity": "manual (reviewed against roster)",
                             "score": 1.0, "europe_seasons": sorted(euro[euro_name])})
    review = [r for r in review
              if (r["existing"], r["new"]) not in DOMESTIC_MERGE_BLOCKLIST
              and DOMESTIC_MANUAL_MERGES.get(r["existing"]) != r["new"]]
    # Map the OLD europe-only spelling onto the NEW domestic one: the domestic
    # league is now the club's primary evidence base, and its spelling is what
    # every future fetch of that league will use.
    alias = {a["existing"]: a["new"] for a in accepted}
    return {"alias": alias, "accepted": accepted, "rejected": rejected,
            "review": review, "europe_only_before": len(euro)}


# ── live ingest resolver ──────────────────────────────────────────────────
# Without this the daily fetch silently rebuilds the duplication: fetch.py
# writes BSD's spelling straight into fixtures.csv with no canon applied
# (seed_real.py accepts a `canon` callable, fetch.py never had one), so every
# run re-creates "FC Bayern München" alongside "Bayern Munich".
_resolver_cache: tuple[dict[str, str], dict[str, str]] | None = None


def _load_resolver() -> tuple[dict[str, str], dict[str, str]]:
    """(alias map, normalised-form -> canonical) — cached for the process."""
    global _resolver_cache
    if _resolver_cache is None:
        alias: dict[str, str] = {}
        if ALIAS_MAP.exists():
            try:
                alias = json.loads(ALIAS_MAP.read_text()).get("alias", {})
            except Exception:
                alias = {}
        by_norm: dict[str, str] = {}
        # Canonical targets win the normalised slot, so an accent-only variant
        # of a canonical name resolves onto it rather than forking a new one.
        for canon in set(alias.values()):
            by_norm.setdefault(_norm(canon), canon)
        for src, dst in alias.items():
            by_norm.setdefault(_norm(src), dst)
        _resolver_cache = (alias, by_norm)
    return _resolver_cache


def reload_resolver() -> None:
    global _resolver_cache
    _resolver_cache = None


_country_index_cache: dict[str, str] | None = None


def team_countries(refresh: bool = False) -> dict[str, str]:
    """Canonical club -> the country of the domestic league it plays in.

    Built from fixtures.csv via each competition's country. Clubs seen only in
    continental competition have no entry, which is treated as "unknown" and
    never blocks an alias.
    """
    global _country_index_cache
    if _country_index_cache is None or refresh:
        index: dict[str, str] = {}
        try:
            df = pd.read_csv(FIXTURES, low_memory=False,
                             usecols=["competition", "home", "away"])
            comp_country = {}
            for comp in df["competition"].dropna().unique():
                c = _get_comp(comp)
                if c and c.kind == "league" and c.country:
                    comp_country[comp] = c.country
            for comp, home, away in zip(df["competition"], df["home"], df["away"]):
                country = comp_country.get(comp)
                if not country:
                    continue
                index.setdefault(home, country)
                index.setdefault(away, country)
        except Exception:
            index = {}
        _country_index_cache = index
    return _country_index_cache


def canonical_name(name, country: str | None = None):
    """Map a provider spelling onto the canonical club identity.

    Order matters: an exact alias hit is authoritative; otherwise fall back to
    the normalised form, which catches accent/punctuation variants the alias
    map has never literally seen ("Atlético Madrid" -> "Atletico Madrid").
    Unknown names pass through unchanged — a new club must not be silently
    bent onto an existing identity.

    `country` scopes the mapping. Club names repeat across confederations, and
    the alias map is global: Brazil's "Athletic Club" (Athletic Club de São
    João del-Rei) resolves onto Spain's **Athletic Bilbao**, because names.py
    maps that exact string. Ingesting South American football without this
    guard would weld a Brazilian club's results onto Athletic Bilbao's rating
    history — and Brazil, Portugal and Spain share many more names (América,
    Nacional, Inter, Sport, Vitória, Santos).

    When `country` is supplied and the alias target belongs to a DIFFERENT
    country, the mapping is refused and the original name passes through. A
    target with no known country (continental-only clubs) never blocks.
    """
    if not name:
        return name
    text = str(name)
    alias, by_norm = _load_resolver()
    target = alias.get(text) or by_norm.get(_norm(text), text)
    if target == text or not country:
        return target
    # LEAGUE membership (where the club is actually observed to play) is
    # authoritative: if we have it, it decides, full stop.
    league_country = team_countries().get(target)
    if league_country is not None:
        return target if league_country == country else text
    # League membership unknown (bootstrap, or a continental-only club). Fall
    # back to the registry's ASSOCIATION country — but that can legitimately
    # differ from the league a cross-border club plays in, so a verified
    # cross-border membership overrides the mismatch rather than refusing a real
    # alias (AS Monaco -> Monaco, Cardiff -> England, Vaduz -> Switzerland).
    try:
        from .club_registry import country_of
        assoc_country = country_of(target)
    except Exception:
        assoc_country = None
    if assoc_country and assoc_country != country:
        if (country in _BORDERLESS_ASSOCIATION_LEAGUES.get(assoc_country, ())
                or _CROSS_BORDER_CLUB_LEAGUE.get(target) == country):
            return target        # verified cross-border membership
        return text              # cross-country collision — refuse the mapping
    return target


def reset_country_index() -> None:
    global _country_index_cache
    _country_index_cache = None


def apply_alias_map(df: pd.DataFrame, alias: dict[str, str],
                    scope_by_country: bool = True) -> pd.DataFrame:
    """Rewrite home/away through the alias map.

    Country-scoped by default. The map is global, and applying it globally
    rewrote Brazil's "Athletic Club" (Athletic Club de São João del-Rei) onto
    Spain's Athletic Bilbao — the alias exists because names.py canonicalises
    that exact string for La Liga. The row's own competition tells us which
    country it belongs to, so a mapping whose target lives elsewhere is
    refused.

    Continental competitions are unscoped: a Champions League row's country
    field is "Europe", which says nothing about either club.
    """
    out = df.copy()
    if not scope_by_country or "competition" not in out.columns:
        for side in ("home", "away"):
            out[side] = out[side].map(lambda v: alias.get(v, v))
        return out

    comp_country: dict[str, str | None] = {}
    for comp in out["competition"].dropna().unique():
        c = _get_comp(comp)
        comp_country[comp] = c.country if (c and c.kind == "league") else None

    known = team_countries()
    try:
        from .club_registry import country_of as _registry_country
    except Exception:
        _registry_country = lambda _n: None    # noqa: E731

    def _target_country(target: str) -> str | None:
        return known.get(target) or _registry_country(target)

    for side in ("home", "away"):
        values = []
        for name, comp in zip(out[side], out["competition"]):
            target = alias.get(name)
            if target is None:
                values.append(name)
                continue
            row_country = comp_country.get(comp)
            tgt_country = _target_country(target) if row_country else None
            if row_country and tgt_country and tgt_country != row_country:
                values.append(name)          # cross-country — refuse
            else:
                values.append(target)
        out[side] = values
    return out


def _load_alias() -> dict[str, str]:
    if not ALIAS_MAP.exists():
        raise SystemExit(f"{ALIAS_MAP} not found — run --write first.")
    return json.loads(ALIAS_MAP.read_text())["alias"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve duplicate club identities.")
    ap.add_argument("--report", action="store_true", help="propose merges only")
    ap.add_argument("--write", action="store_true", help="write the alias map")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite fixtures.csv using the stored alias map")
    ap.add_argument("--show-rejected", action="store_true")
    ap.add_argument("--apply-domestic", action="store_true",
                    help="apply reviewed cross-source merges to fixtures.csv")
    args = ap.parse_args()

    if args.apply_domestic:
        from .identities import dedupe_fixtures
        df = pd.read_csv(FIXTURES, low_memory=False)
        present = set(df["home"].dropna()) | set(df["away"].dropna())
        alias = {src: dst for src, dst in DOMESTIC_MANUAL_MERGES.items()
                 if src in present and dst in present}
        if not alias:
            print("no reviewed cross-source merges apply to the current data")
            return
        backup = FIXTURES.with_suffix(".csv.bak.pre_domestic")
        shutil.copy2(FIXTURES, backup)
        before_teams = len(present)
        merged = dedupe_fixtures(apply_alias_map(df, alias))
        merged.to_csv(FIXTURES, index=False)
        after_teams = len(set(merged["home"]) | set(merged["away"]))
        print(f"backup -> {backup.name}")
        for src, dst in sorted(alias.items()):
            print(f"  {src!r} -> {dst!r}")
        print(f"rows {len(df)} -> {len(merged)}; "
              f"identities {before_teams} -> {after_teams}")
        return

    if args.apply:
        from .identities import dedupe_fixtures
        alias = _load_alias()
        df = pd.read_csv(FIXTURES, low_memory=False)
        before = len(df)
        backup = FIXTURES.with_suffix(".csv.bak.pre_identity")
        shutil.copy2(FIXTURES, backup)
        merged = apply_alias_map(df, alias)
        deduped = dedupe_fixtures(merged)
        deduped.to_csv(FIXTURES, index=False)
        print(f"backup      -> {backup.name}")
        print(f"rows before : {before}")
        print(f"rows after  : {len(deduped)}  ({before - len(deduped)} duplicate rows removed)")
        print(f"identities  : {len(set(df['home']) | set(df['away']))} -> "
              f"{len(set(deduped['home']) | set(deduped['away']))}")
        return

    result = build_alias_map()
    print(f"candidate pairs accepted: {len(result['accepted'])}")
    print(f"candidate pairs rejected: {len(result['rejected'])}")
    print(f"identities merged away  : {len(result['alias'])}\n")
    for canon, aliases in result["groups"].items():
        print(f"  {canon!r}")
        for a in aliases:
            print(f"      <- {a!r}")
    if args.show_rejected:
        print("\nrejected:")
        for r in result["rejected"]:
            print(f"  {r['a']!r} / {r['b']!r}  (n={r['evidence']})  {r['reason']}")

    if args.write:
        DATA.mkdir(exist_ok=True)
        # The map must be CUMULATIVE. build_alias_map only ever proposes merges
        # it can still see evidence for, so once a merge has been applied to
        # fixtures.csv the losing spelling disappears and the pair stops being
        # proposed. Overwriting would therefore silently shrink the map — and
        # canonical_name() reads it on every daily fetch, so the P1 canon would
        # quietly stop being enforced and the duplication would return.
        prior: dict[str, str] = {}
        if ALIAS_MAP.exists():
            try:
                prior = json.loads(ALIAS_MAP.read_text()).get("alias", {})
            except Exception:
                prior = {}
        merged = dict(prior)
        merged.update(result["alias"])
        # Collapse chains (a -> b, b -> c  =>  a -> c) so the resolver is flat.
        def _resolve(name: str, depth: int = 0) -> str:
            nxt = merged.get(name)
            if nxt is None or nxt == name or depth > 10:
                return name
            return _resolve(nxt, depth + 1)
        merged = {k: _resolve(v) for k, v in merged.items()}
        merged = {k: v for k, v in merged.items() if k != v}
        added = len(merged) - len(prior)
        result["alias"] = dict(sorted(merged.items()))
        result["cumulative"] = True
        result["prior_entries"] = len(prior)
        ALIAS_MAP.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        reload_resolver()
        print(f"\nwrote {ALIAS_MAP} ({len(prior)} prior + {added} new "
              f"= {len(merged)} entries)")


if __name__ == "__main__":
    main()
