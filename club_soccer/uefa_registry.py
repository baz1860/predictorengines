#!/usr/bin/env python3
"""UEFA association registry — all 55 members, with honest source coverage.

Purpose
-------
P2 of the league-expansion plan. The plan's stated scope was "all 55 UEFA
associations, top and second tier". Probing the actual data sources shows that
is NOT achievable on free feeds, so this module records all 55 and marks
precisely which are ingestible and from where. The gap is tracked, not hidden.

What the sources actually carry (verified 2026-07-22, not assumed)
------------------------------------------------------------------
BSD (/api/v2/leagues/) returns 72 leagues TOTAL, of which 18 are UEFA-member
domestic competitions. Austria — Sturm Graz's league, the fixture that started
this whole investigation — is NOT among them. The earlier feasibility note in
the plan was written off a sample of the local BSD cache and was wrong.

football-data.co.uk is the stronger source for this job:
  * /mmz4281/{season}/{code}.csv — England x5, Scotland x4, Germany x2,
    Italy x2, Spain x2, France x2, Netherlands, Belgium, Portugal, Turkey,
    Greece. Second tiers for every big league.
  * /new/{COUNTRY}.csv — Austria, Denmark, Finland, Ireland, Norway, Poland,
    Romania, Russia, Sweden, Switzerland. Austria alone carries 14 seasons
    (2012/13-2025/26, 2,638 matches), far more than the 5 required, and the
    files include closing odds.

Combined reach: 22 of 55 associations. Notably still missing, all regular
European participants: Czechia (rank 10), Israel (11), Ukraine (19),
Serbia (20), Croatia (21), Hungary (23), Slovakia (24), Cyprus (26).

Clubs from unavailable associations are not silently mispriced: they fall to
the P0 low-evidence path, which flags them at the point of bet suggestion.

Coefficients
------------
data/uefa_coefficients.json, all 55 associations, refreshed annually. Used as
a RELATIVE prior only — the source's final season was partial at capture, so
absolute values are mildly depressed, but the ordering cross-checks exactly
against UEFA's published top ten.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
COEFFICIENTS = DATA / "uefa_coefficients.json"

# Source kinds
SRC_FDCOUK_MAIN = "fdcouk_main"      # /mmz4281/{season}/{code}.csv
SRC_FDCOUK_NEW = "fdcouk_new"        # /new/{COUNTRY}.csv
SRC_BSD = "bsd"                      # BSD /api/v2/leagues/
SRC_NONE = "none"                    # no free source identified


@dataclass(frozen=True)
class LeagueSource:
    """Where one division's history can actually be fetched from."""
    tier: int
    name: str                 # our canonical Competition name
    source: str
    code: str = ""            # fd.co.uk division code, or BSD league name
    source_league: str = ""   # the league string as the SOURCE spells it
    note: str = ""


@dataclass(frozen=True)
class Association:
    country: str
    rank: int
    coefficient: float
    leagues: tuple[LeagueSource, ...] = field(default_factory=tuple)
    note: str = ""

    @property
    def available(self) -> bool:
        return any(l.source != SRC_NONE for l in self.leagues)

    @property
    def tiers_available(self) -> int:
        return sum(1 for l in self.leagues if l.source != SRC_NONE)


def _load_coefficients() -> dict[str, dict]:
    doc = json.loads(COEFFICIENTS.read_text())
    return {a["country"]: a for a in doc["associations"]}


_COEF = _load_coefficients()


def _assoc(country: str, leagues: tuple[LeagueSource, ...] = (), note: str = "") -> Association:
    rec = _COEF[country]
    return Association(country=country, rank=rec["rank"],
                       coefficient=rec["coefficient"], leagues=leagues, note=note)


# ── the registry ──────────────────────────────────────────────────────────
# Division codes and source-league strings below were verified against live
# responses, not recalled. Leagues already present in competitions.py keep
# their existing Competition names so nothing downstream is renamed.

ASSOCIATIONS: tuple[Association, ...] = (
    _assoc("England", (
        LeagueSource(1, "Premier League", SRC_FDCOUK_MAIN, "E0"),
        LeagueSource(2, "Championship", SRC_FDCOUK_MAIN, "E1"),
        LeagueSource(3, "League One", SRC_FDCOUK_MAIN, "E2"),
        LeagueSource(4, "League Two", SRC_FDCOUK_MAIN, "E3"),
        LeagueSource(5, "National League", SRC_FDCOUK_MAIN, "EC"),
    )),
    _assoc("Italy", (
        LeagueSource(1, "Serie A", SRC_FDCOUK_MAIN, "I1"),
        LeagueSource(2, "Serie B", SRC_FDCOUK_MAIN, "I2"),
    )),
    _assoc("Spain", (
        LeagueSource(1, "La Liga", SRC_FDCOUK_MAIN, "SP1"),
        LeagueSource(2, "Segunda División", SRC_FDCOUK_MAIN, "SP2",
                     source_league="Segunda División"),
    )),
    _assoc("Germany", (
        LeagueSource(1, "Bundesliga", SRC_FDCOUK_MAIN, "D1"),
        LeagueSource(2, "2. Bundesliga", SRC_FDCOUK_MAIN, "D2"),
    )),
    _assoc("France", (
        LeagueSource(1, "Ligue 1", SRC_FDCOUK_MAIN, "F1"),
        LeagueSource(2, "Ligue 2", SRC_FDCOUK_MAIN, "F2"),
    )),
    _assoc("Netherlands", (
        LeagueSource(1, "Eredivisie", SRC_FDCOUK_MAIN, "N1"),
    ), note="second tier (Eerste Divisie) not on any free source checked"),
    _assoc("Portugal", (
        LeagueSource(1, "Liga Portugal", SRC_FDCOUK_MAIN, "P1"),
        LeagueSource(3, "Liga 3", SRC_BSD, source_league="Liga 3",
                     note="third tier; BSD carries it, second tier unavailable"),
    )),
    _assoc("Belgium", (
        LeagueSource(1, "Belgian Pro League", SRC_FDCOUK_MAIN, "B1",
                     source_league="Pro League"),
    )),
    _assoc("Turkey", (
        LeagueSource(1, "Super Lig", SRC_FDCOUK_MAIN, "T1",
                     source_league="Trendyol Super Lig"),
    )),
    _assoc("Czech Republic", (), note="no free source identified — rank 10, regular participant"),
    _assoc("Israel", (), note="no free source identified — rank 11, regular participant"),
    _assoc("Austria", (
        LeagueSource(1, "Austrian Bundesliga", SRC_FDCOUK_NEW, "AUT",
                     source_league="Bundesliga",
                     note="14 seasons available (2012/13-2025/26)"),
    )),
    _assoc("Norway", (
        LeagueSource(1, "Eliteserien", SRC_FDCOUK_NEW, "NOR",
                     source_league="Eliteserien"),
    )),
    _assoc("Scotland", (
        LeagueSource(1, "Scottish Premiership", SRC_FDCOUK_MAIN, "SC0"),
        LeagueSource(2, "Scottish Championship", SRC_FDCOUK_MAIN, "SC1"),
        LeagueSource(3, "Scottish League One", SRC_FDCOUK_MAIN, "SC2"),
        LeagueSource(4, "Scottish League Two", SRC_FDCOUK_MAIN, "SC3"),
    )),
    _assoc("Greece", (
        LeagueSource(1, "Greek Super League", SRC_FDCOUK_MAIN, "G1",
                     source_league="Stoiximan Super League"),
    )),
    _assoc("Switzerland", (
        LeagueSource(1, "Swiss Super League", SRC_FDCOUK_NEW, "SWZ",
                     source_league="Super League"),
        LeagueSource(2, "Swiss Challenge League", SRC_FDCOUK_NEW, "SWZ",
                     source_league="Challenge League",
                     note="only 2 rows present at capture — treat as unavailable in practice"),
    )),
    _assoc("Denmark", (
        LeagueSource(1, "Danish Superliga", SRC_FDCOUK_NEW, "DNK",
                     source_league="Superliga"),
    )),
    _assoc("Poland", (
        LeagueSource(1, "Ekstraklasa", SRC_FDCOUK_NEW, "POL",
                     source_league="Ekstraklasa"),
    )),
    _assoc("Ukraine", (), note="no free source identified — rank 19"),
    _assoc("Serbia", (), note="no free source identified — rank 20"),
    _assoc("Croatia", (), note="no free source identified — rank 21"),
    _assoc("Russia", (
        LeagueSource(1, "Russian Premier League", SRC_FDCOUK_NEW, "RUS",
                     source_league="Premier League"),
    ), note="UEFA-suspended; clubs do not enter European competition. Ingest is "
            "optional and contributes no cross-league links."),
    _assoc("Hungary", (), note="no free source identified — rank 23"),
    _assoc("Slovakia", (), note="no free source identified — rank 24"),
    _assoc("Azerbaijan", (), note="no free source identified — rank 25"),
    _assoc("Cyprus", (), note="no free source identified — rank 26"),
    _assoc("Bulgaria", (
        LeagueSource(1, "Parva Liga", SRC_BSD, source_league="Parva Liga"),
    )),
    _assoc("Romania", (
        LeagueSource(1, "Romanian Superliga", SRC_FDCOUK_NEW, "ROU",
                     source_league="Superliga"),
    )),
    _assoc("Sweden", (
        LeagueSource(1, "Allsvenskan", SRC_FDCOUK_NEW, "SWE",
                     source_league="Allsvenskan"),
    )),
    _assoc("Moldova", (), note="no free source identified"),
    _assoc("Slovenia", (), note="no free source identified"),
    _assoc("Kosovo", (), note="no free source identified"),
    _assoc("Ireland", (
        LeagueSource(1, "League of Ireland Premier Division", SRC_FDCOUK_NEW, "IRL",
                     source_league="Premier Division"),
    )),
    _assoc("Finland", (
        LeagueSource(1, "Veikkausliiga", SRC_FDCOUK_NEW, "FIN",
                     source_league="Veikkausliiga"),
    )),
    _assoc("Faroe Islands", (), note="no free source identified"),
    _assoc("Iceland", (), note="no free source identified"),
    _assoc("Bosnia-Herzegovina", (), note="no free source identified"),
    _assoc("Latvia", (), note="no free source identified"),
    _assoc("Armenia", (), note="no free source identified"),
    _assoc("Kazakhstan", (), note="no free source identified"),
    _assoc("Malta", (), note="no free source identified"),
    _assoc("Lithuania", (), note="no free source identified"),
    _assoc("Liechtenstein", (), note="no domestic league — clubs play in the Swiss pyramid"),
    _assoc("Albania", (), note="no free source identified"),
    _assoc("Northern Ireland", (), note="no free source identified"),
    _assoc("Estonia", (), note="no free source identified"),
    _assoc("Luxembourg", (), note="no free source identified"),
    _assoc("North Macedonia", (), note="no free source identified"),
    _assoc("Wales", (), note="no free source identified"),
    _assoc("Georgia", (), note="no free source identified"),
    _assoc("Montenegro", (), note="no free source identified"),
    _assoc("Andorra", (), note="no free source identified"),
    _assoc("Belarus", (), note="no free source identified"),
    _assoc("Gibraltar", (), note="no free source identified"),
    _assoc("San Marino", (), note="no free source identified"),
)

BY_COUNTRY = {a.country: a for a in ASSOCIATIONS}


# ── strength priors ───────────────────────────────────────────────────────
# Anchored on the two ends of the EXISTING hand-set scale in competitions.py
# (Premier League 1.00 at coefficient 89.160; Scottish Premiership 0.58 at
# 27.500) and linear in the coefficient between them. Checked against the
# other hand-set values before adoption:
#
#   Serie A   coef 79.106 -> 0.931 (hand-set 0.91)
#   La Liga   coef 73.989 -> 0.896 (hand-set 0.92)
#   Bundesliga coef 71.660 -> 0.881 (hand-set 0.93)
#   Ligue 1   coef 57.736 -> 0.786 (hand-set 0.86)
#
# Close enough to be a sane prior, not close enough to justify overwriting the
# hand-set values — so this is used ONLY to seed leagues that have no strength
# yet. Existing competitions keep theirs, and nothing here changes production
# pricing: P4's fitted strengths remain gated off.
# Anchors are RELATIVE, not absolute: England -> 1.00, Scotland -> 0.58, using
# each snapshot's OWN coefficients. The source rescales its absolute numbers
# between captures (England read 89.16 mid-season once, 119.5 complete later),
# so a fixed absolute anchor would drift; anchoring on England/Scotland within
# the snapshot makes the prior scale-invariant and lets the same code price any
# historical snapshot correctly.
_ANCHOR_HI_COUNTRY, _ANCHOR_HI_VALUE = "England", 1.00
_ANCHOR_LO_COUNTRY, _ANCHOR_LO_VALUE = "Scotland", 0.58
# The two anchors must be meaningfully separated for the linear rescale to be
# stable. Real England-Scotland coefficient separation is ~87; a separation
# near zero (junk/degenerate snapshot) makes the slope explode and every prior
# clip to the rails, so treat it as "no usable anchor" and fall back.
_MIN_ANCHOR_SEP = 1.0

# The coefficient-history file spells some associations differently from the
# competition/registry side ("Türkiye" vs "Turkey", "Czechia" vs "Czech
# Republic"). An unnormalised lookup then misses and the association silently
# falls back to DEFAULT_PRIOR — the Turkish clubs' spurious +88 Elo jump. Both
# sides are folded to one spelling before any coefficient lookup.
_ASSOC_ALIASES = {
    "turkiye": "Turkey",
    "turkey": "Turkey",
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
}


def _norm_assoc(name: str | None) -> str | None:
    """Fold an association name to one canonical spelling (accent- and
    alias-normalised) so history and registry names always meet."""
    if not name:
        return name
    key = unicodedata.normalize("NFKD", str(name))
    key = "".join(c for c in key if not unicodedata.combining(c)).casefold().strip()
    return _ASSOC_ALIASES.get(key, name)

# Second/third tier discount, from the observed ratios in the existing
# registry: England 0.72/1.00, Scotland 0.38/0.58 = 0.655, Spain-style
# second tiers sit around the same. One shared factor per tier step.
TIER_DISCOUNT = {1: 1.00, 2: 0.70, 3: 0.50, 4: 0.36, 5: 0.26}

DEFAULT_PRIOR = 0.75

COEFFICIENTS_HISTORY = DATA / "uefa_coefficients_history.json"
_history_cache: list | None = None


def _load_history() -> list:
    """[(published_on 'YYYY-MM-DD', {country: coef}), ...], newest last."""
    global _history_cache
    if _history_cache is None:
        try:
            doc = json.loads(COEFFICIENTS_HISTORY.read_text())
            snaps = doc.get("snapshots", {})
            rows = [(v["published_on"], v["coefficients"]) for v in snaps.values()]
            _history_cache = sorted(rows, key=lambda r: r[0])
        except Exception:
            _history_cache = []
    return _history_cache


def reload() -> None:
    """Drop memoized coefficient state so a mid-process edit to the coefficient
    files is reflected. `_COEF` and the history snapshots are otherwise read
    once (at import / first use), so the walk-forward cache could invalidate a
    fold on a changed file and then recompute it from the STALE in-memory
    coefficients. Call this after editing a coefficient file in a long-lived
    process (tests, notebooks) to keep the two in step."""
    global _history_cache, _COEF
    _history_cache = None
    try:
        _COEF = _load_coefficients()
    except Exception:
        pass


def _snapshot_for(as_of: str | None) -> dict:
    """The coefficient map to use. `as_of` (a date) selects the latest snapshot
    published on or before it — so a fold never sees future coefficients. None
    returns the most recent snapshot (production)."""
    hist = _load_history()
    if not hist:
        # Fall back to the flat registry values (pre-history behaviour).
        return {a.country: a.coefficient for a in ASSOCIATIONS}
    if as_of is None:
        return hist[-1][1]
    as_of = str(as_of)[:10]
    # Before the FIRST snapshot there is no point-in-time prior. The old code
    # defaulted to hist[0] and so returned a snapshot published in the future
    # relative to `as_of` (e.g. a 2020 query got the 2021-06-15 ranking) —
    # exactly the leak the history file exists to prevent. Return an empty map
    # so strength_prior falls back to DEFAULT_PRIOR ("no historical prior yet").
    chosen: dict | None = None
    for published_on, coeffs in hist:
        if published_on <= as_of:
            chosen = coeffs
        else:
            break
    return chosen if chosen is not None else {}


def strength_prior(country: str, tier: int = 1, as_of: str | None = None) -> float:
    """Initial competition strength implied by the UEFA coefficient.

    `as_of` (a date string) picks the coefficient snapshot that existed then, so
    a walk-forward fold seeds from period-correct coefficients rather than
    today's — the leak fix. None uses the latest snapshot (live production).
    """
    coeffs = {_norm_assoc(k): v for k, v in _snapshot_for(as_of).items()}
    c = coeffs.get(_norm_assoc(country))
    hi = coeffs.get(_norm_assoc(_ANCHOR_HI_COUNTRY))
    lo = coeffs.get(_norm_assoc(_ANCHOR_LO_COUNTRY))
    if c is None or hi is None or lo is None or (hi - lo) < _MIN_ANCHOR_SEP:
        return DEFAULT_PRIOR
    slope = (_ANCHOR_HI_VALUE - _ANCHOR_LO_VALUE) / (hi - lo)
    top = _ANCHOR_LO_VALUE + slope * (c - lo)
    top = max(0.15, min(1.05, top))
    return round(top * TIER_DISCOUNT.get(tier, 0.25), 4)


def prior_is_informative(country: str) -> bool:
    """True when the prior comes from a real UEFA coefficient, False when it
    is the flat DEFAULT_PRIOR fallback.

    This matters for shrinkage. A coefficient-backed prior is a genuine
    external estimate and deserves real weight; the 0.75 default is a
    placeholder that carries no information, so leagues resting on it should
    follow their OWN observed data instead. Measured: the leagues stuck on the
    default (non-UEFA divisions, unlisted second tiers) are exactly the ones
    whose fitted strength most overshoots their observed mean Elo — Brasileirão
    Serie B by 0.131, the Championship by 0.099 — because the bad prior was
    given full weight.
    """
    return country in BY_COUNTRY


def coefficient(country: str) -> float:
    a = BY_COUNTRY.get(country)
    return a.coefficient if a else 0.0


def rank(country: str) -> int:
    a = BY_COUNTRY.get(country)
    return a.rank if a else 99


def available_associations() -> list[Association]:
    return [a for a in ASSOCIATIONS if a.available]


def missing_associations() -> list[Association]:
    return [a for a in ASSOCIATIONS if not a.available]


def planned_leagues() -> list[tuple[Association, LeagueSource]]:
    """Every ingestible division, in coefficient-rank then tier order."""
    out = []
    for a in sorted(ASSOCIATIONS, key=lambda x: x.rank):
        for l in sorted(a.leagues, key=lambda x: x.tier):
            if l.source != SRC_NONE:
                out.append((a, l))
    return out


def coverage_summary() -> dict:
    avail = available_associations()
    return {
        "associations_total": len(ASSOCIATIONS),
        "associations_available": len(avail),
        "associations_missing": len(ASSOCIATIONS) - len(avail),
        "divisions_available": len(planned_leagues()),
        "missing_top_30": [f"{a.country} (rank {a.rank})"
                           for a in missing_associations() if a.rank <= 30],
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="UEFA association coverage report.")
    ap.add_argument("--missing", action="store_true", help="list unavailable only")
    args = ap.parse_args()

    s = coverage_summary()
    print(f"UEFA associations       : {s['associations_total']}")
    print(f"  with a usable source  : {s['associations_available']}")
    print(f"  no source identified  : {s['associations_missing']}")
    print(f"ingestible divisions    : {s['divisions_available']}")
    print(f"\nmissing inside the top 30 (regular European participants):")
    for m in s["missing_top_30"]:
        print(f"  - {m}")

    if args.missing:
        return
    print("\ningest plan (by coefficient rank):")
    for a, l in planned_leagues():
        src = f"{l.source}:{l.code}" if l.code else l.source
        print(f"  [{a.rank:>2}] {a.country:<20} T{l.tier}  {l.name:<36} "
              f"{src:<20} prior={strength_prior(a.country, l.tier)}")


if __name__ == "__main__":
    main()
