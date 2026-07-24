#!/usr/bin/env python3
"""Canonical club -> country registry, built from openfootball/clubs.

Why
---
`club_identity.team_countries()` derives a club's country from the domestic
league it appears in, inside our own fixtures.csv. That works for clubs we
already have league data for and fails exactly where it matters: a club seen
only in continental competition has no country, so the cross-confederation
guard has nothing to compare against and cannot block a bad merge.

github.com/openfootball/clubs is a public-domain dataset covering 101
countries. Crucially it carries more than country:

    Atlético Madrid,    Madrid
      | Atlético | Atl. Madrid | Atlético de Madrid | Club Atlético de Madrid
      | Ath Madrid [en]
    ii) Atlético Madrid B

  * the canonical club name
  * its ALTERNATIVE SPELLINGS — "Ath Madrid" is football-data.co.uk's, and
    "Rayo | Vallecano" is a merge this project previously made by hand
  * reserve/B sides marked `ii)`, which must never merge into the senior club

So it supplies a country index AND an externally-maintained alias table,
covering clubs we have never seen. That turns the identity guard from
"block when the target's country is known" into "block whenever either side's
country is known", and gives new-club aliases for free instead of by review.

Trust model
-----------
This is a REFERENCE, not an authority. It never overrides a decision recorded
in identity_verdicts.json or the curated tables in club_identity.py — those
were made against our actual data. It fills gaps and it vetoes; it does not
overrule a human.

CLI:
  python3 -m club_soccer.club_registry --build     # fetch + cache
  python3 -m club_soccer.club_registry --lookup "Sturm Graz"
  python3 -m club_soccer.club_registry --coverage  # how much of fixtures.csv it covers
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REGISTRY = DATA / "club_registry.json"

TREE_URL = ("https://api.github.com/repos/openfootball/clubs/"
            "git/trees/master?recursive=1")
RAW = "https://raw.githubusercontent.com/openfootball/clubs/master/{path}"

# Directory name -> the country string used in competitions.py. Only entries
# that differ from a simple title-case of the directory need listing.
_COUNTRY_ALIASES = {
    "united-states": "USA",
    "england": "England",
    "scotland": "Scotland",
    "wales": "Wales",
    "northern-ireland": "Northern Ireland",
    "ireland": "Ireland",
    "czech-republic": "Czech Republic",
    "bosnia-n-herzegovina": "Bosnia-Herzegovina",
    "north-macedonia": "North Macedonia",
    "south-korea": "South Korea",
    "south-africa": "South Africa",
    "saudi-arabia": "Saudi Arabia",
    "congo-dr": "Congo DR",
    "faroe-islands": "Faroe Islands",
    "united-arab-emirates": "United Arab Emirates",
    "el-salvador": "El Salvador",
    "costa-rica": "Costa Rica",
    "trinidad-n-tobago": "Trinidad and Tobago",
    "new-zealand": "New Zealand",
}

_SECTION_RE = re.compile(r"^\s*=+")
_RESERVE_RE = re.compile(r"^\s*(ii+|b)\)\s*", re.I)
_LANG_TAG_RE = re.compile(r"\s*\[[a-z]{2}\]\s*$", re.I)


def _country_from_path(path: str) -> str:
    parts = path.split("/")
    if len(parts) < 3:
        return ""
    slug = parts[1]
    return _COUNTRY_ALIASES.get(slug, slug.replace("-", " ").title())


def _strip_comment(line: str) -> str:
    # '#' starts a comment, but only outside a club name. openfootball uses
    # '##' for editorial asides and '#' for short notes; both follow the data.
    cut = line.find("#")
    return (line[:cut] if cut >= 0 else line).rstrip()


def _clean(name: str) -> str:
    name = _LANG_TAG_RE.sub("", name.strip())
    return name.strip(" ,|")


def parse_clubs_file(text: str, country: str) -> list[dict]:
    """Parse one openfootball clubs.txt into club records."""
    clubs: list[dict] = []
    current: dict | None = None
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line)
        if not line.strip() or _SECTION_RE.match(line):
            continue

        if line.lstrip().startswith("|"):
            # Continuation: alternative spellings for the club above.
            if current is None:
                continue
            for part in line.strip().lstrip("|").split("|"):
                alias = _clean(part)
                if alias and alias != current["name"]:
                    current["aliases"].append(alias)
            continue

        stripped = line.strip()
        reserve = bool(_RESERVE_RE.match(stripped))
        stripped = _RESERVE_RE.sub("", stripped)
        # "Club Name, City" or "Club Name, 1976, City" — the name is the first
        # comma-separated field.
        name = _clean(stripped.split(",")[0])
        if not name:
            continue
        current = {"name": name, "country": country,
                   "reserve": reserve, "aliases": []}
        clubs.append(current)
    return clubs


def _fetch(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def list_files() -> list[str]:
    doc = json.loads(_fetch(TREE_URL))
    return [x["path"] for x in doc.get("tree", [])
            if x["path"].endswith("clubs.txt")]


def build(paths: list[str] | None = None, verbose: bool = True) -> dict:
    """Fetch and parse the dataset into a lookup artifact."""
    paths = list_files() if paths is None else paths
    clubs: list[dict] = []
    failures: list[str] = []
    for path in paths:
        country = _country_from_path(path)
        if not country:
            continue
        try:
            text = _fetch(RAW.format(path=path))
        except Exception as exc:
            failures.append(f"{path}: {exc}")
            continue
        found = parse_clubs_file(text, country)
        clubs.extend(found)
        if verbose:
            print(f"  {country:<24}{len(found):>5} clubs   ({path})")

    by_norm: dict[str, dict] = {}
    for club in clubs:
        for name in [club["name"]] + club["aliases"]:
            key = _norm(name)
            if not key:
                continue
            existing = by_norm.get(key)
            if existing and existing["country"] != club["country"]:
                # Genuine cross-country name clash — exactly what the guard is
                # for. Mark it ambiguous so it can never be used to merge.
                existing["ambiguous"] = True
                existing.setdefault("countries", [existing["country"]])
                if club["country"] not in existing["countries"]:
                    existing["countries"].append(club["country"])
                continue
            if existing:
                continue
            by_norm[key] = {"canonical": club["name"], "country": club["country"],
                            "reserve": club["reserve"]}

    return {"source": "github.com/openfootball/clubs (public domain)",
            "files": len(paths), "clubs": len(clubs),
            "index_entries": len(by_norm),
            "ambiguous": sum(1 for v in by_norm.values() if v.get("ambiguous")),
            "failures": failures,
            "index": by_norm}


def _norm(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_cache: dict | None = None


def load() -> dict:
    global _cache
    if _cache is None:
        if REGISTRY.exists():
            try:
                _cache = json.loads(REGISTRY.read_text())
            except Exception:
                _cache = {"index": {}}
        else:
            _cache = {"index": {}}
    return _cache


def reload_registry() -> None:
    global _cache
    _cache = None


def lookup(name: str) -> dict | None:
    """Registry record for a club spelling, or None.

    A name that maps to clubs in more than one country returns a record marked
    `ambiguous`, which callers must treat as "do not merge" rather than as a
    country assignment.
    """
    return load().get("index", {}).get(_norm(name))


def country_of(name: str) -> str | None:
    rec = lookup(name)
    if not rec or rec.get("ambiguous"):
        return None
    return rec.get("country")


def same_club_possible(a: str, b: str) -> tuple[bool, str]:
    """Could these two spellings be one club, per the reference data?

    Returns (possible, reason). Used as a VETO only: a False is trustworthy
    (the reference says they are in different countries), a True means "no
    objection", never "these are the same club".
    """
    ra, rb = lookup(a), lookup(b)
    if not ra or not rb:
        return True, "not in the reference dataset"
    if ra.get("ambiguous") or rb.get("ambiguous"):
        return True, "name is ambiguous across countries"
    if ra["country"] != rb["country"]:
        return False, f"different countries ({ra['country']} vs {rb['country']})"
    if ra.get("reserve") != rb.get("reserve"):
        return False, "one is a reserve/B side"
    return True, f"both in {ra['country']}"


def confirms_same_club(a: str, b: str) -> bool:
    """POSITIVE confirmation: both spellings resolve to the SAME registry
    canonical identity. Absence or ambiguity is NOT confirmation.

    same_club_possible() is a veto (False = trustworthy no); this is its
    complement for the auto-merge fail-closed rule (True = trustworthy yes).
    A merge that relies only on 'the registry could not object' must go to
    human review, not apply automatically.

    Same country is NOT same club: "Manchester United" and "Manchester City"
    are both non-reserve English sides, so a country/reserve match alone welds
    genuine rivals. Confirmation therefore requires the two spellings to map to
    the same non-ambiguous canonical name — the registry's actual club identity.
    """
    ra, rb = lookup(a), lookup(b)
    if not ra or not rb:
        return False
    if ra.get("ambiguous") or rb.get("ambiguous"):
        return False
    ca, cb = ra.get("canonical"), rb.get("canonical")
    if not ca or not cb:
        return False
    return (ca == cb
            and ra["country"] == rb["country"]
            and ra.get("reserve") == rb.get("reserve"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, help="only fetch N files (for testing)")
    ap.add_argument("--lookup", metavar="NAME")
    ap.add_argument("--coverage", action="store_true")
    args = ap.parse_args()

    if args.lookup:
        rec = lookup(args.lookup)
        print(json.dumps(rec, indent=2, ensure_ascii=False) if rec
              else f"{args.lookup!r} not in the registry")
        return

    if args.coverage:
        import pandas as pd
        from .club_identity import FIXTURES
        df = pd.read_csv(FIXTURES, low_memory=False)
        names = set(df["home"].dropna()) | set(df["away"].dropna())
        hit = sum(1 for n in names if lookup(n))
        print(f"clubs in fixtures.csv : {len(names)}")
        print(f"  found in registry   : {hit} ({hit / max(len(names), 1):.1%})")
        print(f"  not found           : {len(names) - hit}")
        missing = sorted(n for n in names if not lookup(n))[:20]
        print("\nsample not found:")
        for n in missing:
            print(f"  {n}")
        return

    if args.build:
        paths = list_files()
        if args.limit:
            paths = paths[:args.limit]
        result = build(paths)
        DATA.mkdir(exist_ok=True)
        REGISTRY.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        reload_registry()
        print(f"\nclubs parsed   : {result['clubs']}")
        print(f"index entries  : {result['index_entries']}")
        print(f"ambiguous names: {result['ambiguous']}")
        if result["failures"]:
            print(f"failures       : {len(result['failures'])}")
        print(f"wrote {REGISTRY}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
