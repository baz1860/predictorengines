#!/usr/bin/env python3
"""Standby ingest for the 247Sports College Team Talent Composite.

Why this exists
---------------
The Elo preseason prior needs team talent (``priors.talent_z``), and it is the
dominant term: sd ~90 Elo vs ~33 for returning production. CFBD's ``/talent``
endpoint is the normal source, but it depends on 247 publishing first, and 247
finalises the composite on a fixed annual date — **27 August** in both 2024 and
2025. In 2026 the first kickoff is 29 August, so CFBD has roughly a two-day
window to ingest before Week 1. This module is the fallback for when it doesn't.

CFBD's ``talent`` field is 247's composite "Points" value verbatim (verified:
Alabama 2025 = 993.55 in both), so ingesting here is a drop-in, not an
approximation.

Provenance is never laundered: output carries ``"source": "247sports"`` per row
and a sidecar manifest, so a snapshot built from this is distinguishable from
one built from CFBD.

Usage:
  python3 -m cfb.fetch_247_talent --year 2026            # fetch + write
  python3 -m cfb.fetch_247_talent --year 2026 --dry-run  # parse, don't write
  python3 -m cfb.fetch_247_talent --year 2025 --verify   # check vs CFBD data
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFBD_DIR = HERE / "data" / "cfbd"
URL = "https://247sports.com/season/{year}-football/collegeteamtalentcomposite/"
PARTIAL_VIEW = "~/Views/SkyNet/InstitutionRanking/_SimpleSetForSeason.ascx"
USER_AGENT = "Mozilla/5.0 (compatible; cfb-model/1.0)"

MAX_PAGES = 12           # 50 rows/page; FBS+FCS needs ~6, cap guards a loop
MIN_TEAMS = 100          # FBS is ~134; refuse to publish a partial scrape
MIN_TALENT, MAX_TALENT = 100.0, 1500.0


def fetch_page(year: int, page: int = 1, timeout: int = 30) -> str:
    """Fetch one page. The list shows 50 teams; page 2+ is a partial view."""
    url = URL.format(year=int(year))
    if page > 1:
        url = f"{url}?ViewPath={urllib.parse.quote(PARTIAL_VIEW, safe='')}&Page={int(page)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def collect_rows(year: int, max_pages: int = MAX_PAGES,
                 pause: float = 1.0) -> list[dict]:
    """Page through the composite until a page adds no new teams."""
    rows: list[dict] = []
    seen: set[str] = set()
    for page in range(1, int(max_pages) + 1):
        if page > 1:
            time.sleep(pause)          # be a considerate client
        batch = parse_talent(fetch_page(year, page))
        fresh = [r for r in batch if r["slug"] not in seen]
        if not fresh:
            break
        seen.update(r["slug"] for r in fresh)
        rows.extend(fresh)
        print(f"  page {page}: +{len(fresh)} teams (total {len(rows)})")
    return rows


# Each row renders as:
#   <div class="team"><a class="rankings-page__name-link" href="...">Alabama </a></div>
#   ... <div class="avg"> 92.59 </div> ...
#   <div class="points"><a class="number" href="...">993.55 </a></div>
# The display name is already the clean school name (no mascot) — use it rather
# than the URL slug, which is "alabama-crimson-tide". Points are comma-grouped
# above 1000 ("1,002.98"), so strip separators before parsing.
_ROW = re.compile(r'rankings-page__list-item', re.I)
_NAME = re.compile(r'rankings-page__name-link"[^>]*>\s*([^<]+?)\s*<', re.I)
_POINTS = re.compile(
    r'class="points".*?class="number"[^>]*>\s*([\d,]+\.\d{1,2})\s*<',
    re.I | re.S)
_SLUG = re.compile(
    r'/college/([a-z0-9\-]+)/team/[a-z0-9\-]+-football-\d+/roster/', re.I)


def parse_talent(html: str) -> list[dict]:
    """Extract [{slug, team, talent}] from the composite page."""
    rows: list[dict] = []
    seen: set[str] = set()
    starts = [m.start() for m in _ROW.finditer(html)]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(html)
        block = html[start:end]
        name_match = _NAME.search(block)
        points_match = _POINTS.search(block)
        if not name_match or not points_match:
            continue
        try:
            talent = float(points_match.group(1).replace(",", ""))
        except ValueError:
            continue
        if not MIN_TALENT <= talent <= MAX_TALENT:
            continue
        name = name_match.group(1).strip()
        slug_match = _SLUG.search(block)
        slug = slug_match.group(1) if slug_match else name.lower().replace(" ", "-")
        if slug in seen:
            continue
        seen.add(slug)
        rows.append({"slug": slug, "team": name, "talent": talent})
    return rows


def resolve_names(rows: list[dict], year: int,
                  identity_season: int | None = None) -> tuple[list[dict], list[str]]:
    """Map 247 team names onto canonical CFBD identities. Never guesses.

    ``identity_season`` selects which schedule catalog supplies the canonical
    names; it defaults to ``year``. Needed to validate a past season's parse
    when only the current schedule file is on disk.
    """
    from . import identity as IDENTITY

    season = int(identity_season or year)
    resolved, unresolved = [], []
    for row in rows:
        match = (IDENTITY.resolve(row["team"], season, provider="247sports")
                 or IDENTITY.resolve(row["slug"].replace("-", " "), season,
                                     provider="247sports"))
        if match is None:
            unresolved.append(row["team"])
            continue
        resolved.append({"year": int(year), "team": match["canonical"],
                         "talent": row["talent"], "source": "247sports"})
    return resolved, unresolved


def write_talent(rows: list[dict], year: int) -> Path:
    """Publish in CFBD's schema so priors.load_features() needs no change."""
    if len(rows) < MIN_TEAMS:
        raise SystemExit(
            f"refusing to publish a partial scrape: {len(rows)} teams "
            f"(minimum {MIN_TEAMS}). The composite may not be published yet.")
    dest = CFBD_DIR / f"talent_{int(year)}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(rows, handle)
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    manifest = dest.with_name(f"talent_{int(year)}.source.json")
    manifest.write_text(json.dumps({
        "source": "247sports",
        "url": URL.format(year=int(year)),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "teams": len(rows),
        "note": ("Standby ingest used because CFBD /talent had no data for this "
                 "season. Values are 247's composite Points, the same figure "
                 "CFBD republishes. Replace with the CFBD pull when available."),
    }, indent=2) + "\n")
    return dest


def verify_against_cfbd(rows: list[dict], year: int) -> int:
    """Compare a parse against CFBD's stored values. Returns mismatch count."""
    path = CFBD_DIR / f"talent_{int(year)}.json"
    try:
        official = {r["team"]: float(r["talent"]) for r in
                    json.loads(path.read_text())}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        print(f"no usable CFBD talent_{year}.json to verify against")
        return -1
    if not official:
        print(f"CFBD talent_{year}.json is empty — nothing to verify against")
        return -1
    scraped = {r["team"]: float(r["talent"]) for r in rows}
    shared = sorted(set(official) & set(scraped))
    mismatches = [(t, official[t], scraped[t]) for t in shared
                  if abs(official[t] - scraped[t]) > 0.005]
    print(f"verify {year}: CFBD {len(official)} teams, parsed {len(scraped)}, "
          f"overlap {len(shared)}, value mismatches {len(mismatches)}")
    for team, a, b in mismatches[:10]:
        print(f"  MISMATCH {team}: CFBD {a} vs parsed {b}")
    missing = sorted(set(official) - set(scraped))
    if missing:
        print(f"  {len(missing)} CFBD team(s) not parsed, e.g. {missing[:5]}")
    return len(mismatches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and report, write nothing")
    parser.add_argument("--verify", action="store_true",
                        help="compare the parse against the stored CFBD file")
    parser.add_argument("--html", type=Path,
                        help="parse a saved page instead of fetching")
    parser.add_argument("--identity-season", type=int, default=None,
                        help="schedule catalog to resolve names against "
                             "(defaults to --year; use for back-validation)")
    args = parser.parse_args()

    raw = (parse_talent(args.html.read_text()) if args.html
           else collect_rows(args.year))
    if not raw:
        raise SystemExit(
            f"no team rows parsed for {args.year}. If the page shows "
            f"'No Results', 247 has not published this season's composite yet "
            f"(historically finalised 27 August).")
    rows, unresolved = resolve_names(raw, args.year, args.identity_season)
    print(f"parsed {len(raw)} team rows; resolved {len(rows)} to CFBD identities")
    if unresolved:
        print(f"  {len(unresolved)} unresolved name(s), excluded not guessed: "
              f"{', '.join(unresolved[:8])}"
              + ("..." if len(unresolved) > 8 else ""))
        print("  add reviewed aliases to cfb/data/team_aliases.json "
              "(provider '247sports') to include them")

    if args.verify:
        sys.exit(1 if verify_against_cfbd(rows, args.year) > 0 else 0)
    if args.dry_run:
        for row in sorted(rows, key=lambda r: -r["talent"])[:10]:
            print(f"  {row['team']:<24s} {row['talent']:.2f}")
        return
    print(f"wrote {write_talent(rows, args.year)}")
    print("Re-run readiness:  python3 preflight.py --engine cfb --require-ready")


if __name__ == "__main__":
    main()
