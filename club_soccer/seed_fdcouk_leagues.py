#!/usr/bin/env python3
"""P3 — ingest the UEFA expansion leagues from football-data.co.uk.

BSD carries only 18 UEFA domestic competitions and does not carry Austria at
all (see uefa_registry.py), so fd.co.uk is the primary source for this phase.
Two file shapes, with materially different richness:

  /mmz4281/{season}/{code}.csv   per-season, per-division. RICH: shots (HS/AS),
                                 shots on target (HST/AST), corners (HC/AC),
                                 cards, half-time score.
  /new/{COUNTRY}.csv             one file per country, all seasons. SPARSE:
                                 goals only, plus closing odds. No shot data.

The sparse files matter for modelling: the xg ensemble component is built
from shots-on-target, so leagues arriving via /new/
(Austria, Denmark, Norway, Poland, Romania, Russia, Sweden, Switzerland,
Finland, Ireland) contribute to the goals and Elo components only. Their teams
will be rated, but on a thinner signal than a Premier League side — which is
exactly what the P0 coverage tiers are there to express, and a reason not to
treat "league ingested" as "league fully modelled".

Identity is resolved at the shared ``fetch.write_fixtures`` boundary. Provider
aliases are mapped through the curated/openfootball registry there; this
seeder does not run a second fuzzy reconciliation workflow.

CLI:
  python3 -m club_soccer.seed_fdcouk_leagues --report          # what would be fetched
  python3 -m club_soccer.seed_fdcouk_leagues --wave 1 --dry-run
  python3 -m club_soccer.seed_fdcouk_leagues --wave 1 --write
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import schema
from .competitions import COMPETITIONS, get as comp_get
from .identities import dedupe_fixtures

DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
CACHE = DATA / "fdcouk_league_cache"

MAIN_URL = "https://www.football-data.co.uk/mmz4281/{ss}/{code}.csv"
NEW_URL = "https://www.football-data.co.uk/new/{code}.csv"

# 5 seasons of history, per the agreed scope.
SEASONS = ["2122", "2223", "2324", "2425", "2526", "2627"]
MIN_SEASON_START = 2021

# football-data.co.uk's per-season files (/mmz4281/{season}/) are IMMUTABLE once
# a season ends — only the current (and, across the Aug rollover, the previous)
# season file gains new results. So a daily refresh must re-download ONLY these,
# and read every completed season from the on-disk cache. Re-fetching all six
# files for every BSD-less league on every run was ~24 network round-trips with
# a 40s timeout each, which is what made the "Refresh BSD-less leagues" step
# appear to hang on a slow link.
REFRESHABLE_SEASONS = set(SEASONS[-2:])

# A refresh only cares about rows that could have changed since the last run.
# Generous enough to absorb a missed week plus late score corrections.
REFRESH_LOOKBACK_DAYS = 30

# Ingest waves, by UEFA coefficient rank. Each wave is independently
# shippable and independently reversible.
WAVES: dict[int, list[str]] = {
    1: ["Eredivisie", "Liga Portugal", "Belgian Pro League", "Super Lig",
        "Greek Super League", "Austrian Bundesliga"],
    2: ["Swiss Super League", "Danish Superliga", "Ekstraklasa", "Eliteserien",
        "Allsvenskan", "Romanian Superliga"],
    3: ["Serie B", "Segunda División", "2. Bundesliga", "Ligue 2",
        "National League"],
    4: ["Veikkausliiga", "League of Ireland", "Russian Premier League"],
}


def _fetch(url: str, cache_name: str, refresh: bool = False) -> str | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / cache_name
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            text = resp.read().decode("utf-8-sig", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"    fetch failed {url}: {exc}")
        return None
    path.write_text(text, encoding="utf-8")
    return text


def _fid(comp_name: str, date: str, home: str, away: str) -> int:
    """Deterministic synthetic fixture id.

    fd.co.uk supplies no match id, and the id must be stable across re-runs so
    a re-ingest updates rows rather than duplicating them.
    """
    h = hashlib.md5(f"fdcouk|{comp_name}|{date}|{home}|{away}".encode()).hexdigest()
    # Keep clear of BSD's id space; fixture ids are only required to be unique.
    return 8_000_000_000 + int(h[:9], 16)


def _num(value):
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _season_of(date: pd.Timestamp, comp=None) -> int:
    """Season label for a match date — delegated to the ONE canonical helper.

    Kept as a thin wrapper (call sites pass a pd.Timestamp) so the winter/
    calendar-season rule lives in exactly one place; a divergent inline copy is
    what mislabelled calendar-year leagues before.
    """
    from .fetch import season_for_date
    return season_for_date(date.strftime("%Y-%m-%d"), comp)


def _blank_row() -> dict:
    return {c: "" for c in schema.FIXTURE_COLUMNS}


def _make_row(comp, date: pd.Timestamp, home: str, away: str,
              hg, ag, extras: dict | None = None) -> dict:
    row = _blank_row()
    played = hg is not None and ag is not None
    row.update({
        "fixture_id": _fid(comp.name, date.strftime("%Y-%m-%d"), home, away),
        "kickoff_utc": "",
        "date": date.strftime("%Y-%m-%d"),
        "season": _season_of(date, comp),
        "competition": comp.name,
        "competition_id": comp.api_id,
        "country": comp.country,
        "type": comp.kind,
        "home_id": "", "away_id": "",
        "home": home, "away": away,
        "home_goals": hg if played else None,
        "away_goals": ag if played else None,
        # fd.co.uk publishes results only; an unplayed fixture simply is not
        # in the file. Every row here is therefore a finished match.
        "status": "FT",
        "status_raw": "",
        "result_scope": "FT",
        "neutral": 0,
        "xg_source": "",
    })
    if extras:
        row.update({k: v for k, v in extras.items() if v is not None})
    return row


def load_main(comp, refresh: bool = False,
              refresh_seasons: set | None = None) -> list[dict]:
    """Parse /mmz4281/ season files for one division.

    `refresh_seasons` limits which season files bypass the cache when
    `refresh` is set. None means "all" (a full backfill). The daily refresh
    passes REFRESHABLE_SEASONS so only the current/previous files are
    re-downloaded and completed seasons are served from cache — the fix for the
    hang on the BSD-less refresh step. A season with no cache yet is fetched
    regardless, so the first run still populates everything.
    """
    rows: list[dict] = []
    for ss in SEASONS:
        do_refresh = refresh and (refresh_seasons is None or ss in refresh_seasons)
        text = _fetch(MAIN_URL.format(ss=ss, code=comp.fdcouk_code),
                      f"{comp.fdcouk_code}_{ss}.csv", do_refresh)
        if text is None:
            continue
        frame = pd.read_csv(io.StringIO(text), low_memory=False)
        for r in frame.to_dict("records"):
            raw_date = str(r.get("Date") or "").strip()
            home = str(r.get("HomeTeam") or "").strip()
            away = str(r.get("AwayTeam") or "").strip()
            if not raw_date or not home or not away:
                continue
            date = pd.to_datetime(raw_date, dayfirst=True, errors="coerce")
            if pd.isna(date):
                continue
            hg, ag = _num(r.get("FTHG")), _num(r.get("FTAG"))
            if hg is None or ag is None:
                continue
            rows.append(_make_row(comp, date, home, away, hg, ag, {
                "home_shots": _num(r.get("HS")), "away_shots": _num(r.get("AS")),
                "home_sot": _num(r.get("HST")), "away_sot": _num(r.get("AST")),
                "home_corners": _num(r.get("HC")), "away_corners": _num(r.get("AC")),
                "home_yellow_cards": _num(r.get("HY")), "away_yellow_cards": _num(r.get("AY")),
                "home_red_cards": _num(r.get("HR")), "away_red_cards": _num(r.get("AR")),
                "home_goals_ht": _num(r.get("HTHG")), "away_goals_ht": _num(r.get("HTAG")),
                "home_goals_ft": hg, "away_goals_ft": ag,
            }))
    return rows


def load_new(comp, refresh: bool = False) -> list[dict]:
    """Parse a /new/{COUNTRY}.csv file, filtered to one league."""
    text = _fetch(NEW_URL.format(code=comp.fdcouk_new),
                  f"new_{comp.fdcouk_new}.csv", refresh)
    if text is None:
        return []
    frame = pd.read_csv(io.StringIO(text), low_memory=False)
    # League values carry stray trailing whitespace in several files
    # ("Superliga " vs "Superliga", "Allsvenskan " vs "Allsvenskan"), which
    # would silently drop roughly a third of Denmark's rows on an exact match.
    frame["_league"] = frame["League"].astype(str).str.strip()
    want = comp.fdcouk_league.strip()
    frame = frame[frame["_league"] == want]
    rows: list[dict] = []
    for r in frame.to_dict("records"):
        home = str(r.get("Home") or "").strip()
        away = str(r.get("Away") or "").strip()
        raw_date = str(r.get("Date") or "").strip()
        if not home or not away or not raw_date:
            continue
        date = pd.to_datetime(raw_date, dayfirst=True, errors="coerce")
        if pd.isna(date):
            continue
        if date.year < MIN_SEASON_START:
            continue
        hg, ag = _num(r.get("HG")), _num(r.get("AG"))
        if hg is None or ag is None:
            continue
        rows.append(_make_row(comp, date, home, away, hg, ag, {
            "home_goals_ft": hg, "away_goals_ft": ag,
        }))
    return rows


def load_competition(comp, refresh: bool = False,
                     refresh_seasons: set | None = None) -> list[dict]:
    if comp.fdcouk_code:
        return load_main(comp, refresh, refresh_seasons)
    if comp.fdcouk_new:
        return load_new(comp, refresh)
    return []


def wave_competitions(wave: int) -> list:
    names = WAVES.get(wave, [])
    comps = []
    for name in names:
        comp = comp_get(name)
        if comp is None:
            raise SystemExit(f"wave {wave} references unknown competition {name!r}")
        comps.append(comp)
    return comps


def ingest(wave: int, write: bool = False, refresh: bool = False) -> pd.DataFrame:
    comps = wave_competitions(wave)
    all_rows: list[dict] = []
    print(f"Wave {wave}: {len(comps)} competition(s)")
    for comp in comps:
        rows = load_competition(comp, refresh)
        teams = {r["home"] for r in rows} | {r["away"] for r in rows}
        seasons = sorted({r["season"] for r in rows})
        src = comp.fdcouk_code or f"new/{comp.fdcouk_new}"
        print(f"  {comp.name:<28} {src:<12} rows={len(rows):<6} "
              f"teams={len(teams):<4} seasons={seasons}")
        all_rows.extend(rows)

    new = pd.DataFrame(all_rows)
    if new.empty:
        print("  nothing fetched")
        return new

    # Respect the canon established in P1 before anything else. This alone
    # resolves cases like fd.co.uk's "PSV Eindhoven" onto the existing "PSV",
    # shrinking what the cross-source matcher has to guess at.
    from .club_identity import canonical_name
    for side in ("home", "away"):
        new[side] = [canonical_name(v, country=c)
                     for v, c in zip(new[side], new["country"])]

    if not write:
        print(f"\n  DRY RUN — {len(new)} rows parsed, fixtures.csv untouched")
        return new

    existing = pd.read_csv(FIXTURES, low_memory=False)
    before = len(existing)
    before_teams = len(set(existing["home"]) | set(existing["away"]))

    combined = pd.concat([existing, new], ignore_index=True)
    combined = dedupe_fixtures(combined)
    from .fetch import write_fixtures as _write_fixtures
    combined = _write_fixtures(combined)
    after_teams = len(set(combined["home"]) | set(combined["away"]))
    print(f"  fixtures.csv {before} -> {len(combined)} (+{len(combined) - before})")
    print(f"  identities   {before_teams} -> {after_teams}")
    return new


def needs_fdcouk_refresh() -> list:
    """Competitions with NO BSD path — fd.co.uk is their only source of updates.

    Derived from the registry, not hand-listed, so adding a BSD alias later
    automatically drops a league out of this set.

    This is the difference between a one-off backfill and a live league. The
    P3 ingest loaded 5 seasons of history for 20 competitions; the daily
    pipeline fetches from BSD, which resolves 14 of them via the P2 aliases
    and has never heard of the other 8. Without a refresh path those 8 freeze
    at their last ingest — no new results, no upcoming fixtures, ratings
    slowly going stale while still being priced. Austria is in this set, which
    is the league the whole expansion existed to fix.
    """
    from .competitions import BSD_LEAGUE_ALIASES
    covered = set(BSD_LEAGUE_ALIASES.values()) | {c.name for c in COMPETITIONS if c.bsd_league}
    return [c for c in COMPETITIONS
            if (c.fdcouk_code or c.fdcouk_new) and c.name not in covered]


def refresh(comps: list | None = None, write: bool = True,
            verbose: bool = True) -> dict:
    """Re-fetch current-season results for the BSD-less leagues and merge.

    Safe to run on every pipeline invocation: fixture ids are deterministic
    (`_fid`), so a re-fetch updates rows in place rather than duplicating, and
    dedupe_fixtures is the backstop. Cache is bypassed — the whole point is to
    pick up results that landed since the last run.
    """
    comps = needs_fdcouk_refresh() if comps is None else comps
    rows: list[dict] = []
    per_comp: dict[str, int] = {}
    for i, comp in enumerate(comps, 1):
        if verbose:
            # Print BEFORE the network fetch so a slow link looks like progress,
            # not a hang. Only the current/previous season files are downloaded;
            # completed seasons come from cache.
            print(f"  [{i}/{len(comps)}] {comp.name} …", flush=True)
        try:
            fetched = load_competition(comp, refresh=True,
                                       refresh_seasons=REFRESHABLE_SEASONS)
        except Exception as exc:
            if verbose:
                print(f"  {comp.name}: refresh FAILED ({exc})")
            per_comp[comp.name] = -1
            continue
        # Only recent rows matter for a refresh; history is already ingested.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=REFRESH_LOOKBACK_DAYS)).date()
        recent = [r for r in fetched if str(r["date"]) >= str(cutoff)]
        per_comp[comp.name] = len(recent)
        rows.extend(recent)
        if verbose:
            print(f"  {comp.name:<26} {len(recent):>4} row(s) in the last "
                  f"{REFRESH_LOOKBACK_DAYS}d  (file has {len(fetched)})")

    result = {"competitions": len(comps), "rows": len(rows), "per_competition": per_comp}
    if not rows or not write:
        if verbose and not rows:
            print("  nothing new")
        return result

    new = pd.DataFrame(rows)
    from .club_identity import canonical_name
    for side in ("home", "away"):
        new[side] = [canonical_name(v, country=c)
                     for v, c in zip(new[side], new["country"])]

    existing = pd.read_csv(FIXTURES, low_memory=False)
    before = len(existing)
    combined = dedupe_fixtures(pd.concat([existing, new], ignore_index=True))
    from .fetch import write_fixtures as _write_fixtures
    combined = _write_fixtures(combined)
    result["fixtures_before"] = before
    result["fixtures_after"] = len(combined)
    if verbose:
        print(f"  fixtures.csv {before} -> {len(combined)} (+{len(combined) - before})")
    return result


def _active_months(dates: pd.Series, peak_fraction: float = 0.25) -> set[int]:
    """Calendar months a league actually plays in.

    Derived from its own history: a month counts as in-season if it carries at
    least `peak_fraction` of the busiest month's matches. A flat "any match
    ever" test fails — 14 seasons accumulate a stray friendly or rescheduled
    tie in every month — so Austria would read as playing year-round when it in
    fact breaks over June/July.
    """
    months = pd.to_datetime(dates, errors="coerce").dt.month.dropna().astype(int)
    if months.empty:
        return set(range(1, 13))     # unknown — assume always in season
    counts = months.value_counts()
    peak = counts.max()
    return {int(m) for m, n in counts.items() if n >= peak_fraction * peak}


def staleness(days_warn: int = 21) -> list[dict]:
    """Days since the last recorded result, per competition.

    Surfaces the failure this module exists to prevent: a league quietly
    ceasing to update while still being priced.

    A stale league only WARNS when it is currently in its own playing season.
    Off-season dormancy is normal and must not alarm — every European winter
    league is idle in June/July, and warning on all of them trains the operator
    to ignore the warning, so a genuinely broken mid-season fetch hides in the
    noise. "In season" is read from each league's historical match calendar,
    not hard-coded.
    """
    df = pd.read_csv(FIXTURES, low_memory=False)
    played = df[df["home_goals"].notna()]
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    this_month = today.month
    out = []
    for comp in COMPETITIONS:
        sub = played[played["competition"] == comp.name]
        if sub.empty:
            continue
        last = pd.Timestamp(str(sub["date"].max())[:10])
        upcoming = int((pd.to_datetime(df[df["competition"] == comp.name]["date"],
                                       errors="coerce") > today).sum())
        days = int((today - last).days)
        in_season = this_month in _active_months(sub["date"])
        out.append({"competition": comp.name, "last_result": str(last.date()),
                    "days_stale": days, "upcoming": upcoming,
                    "in_season": in_season,
                    "has_bsd_path": comp.name not in {c.name for c in needs_fdcouk_refresh()},
                    # Warn only when the league should be producing results now
                    # (in season, nothing scheduled ahead, and gone quiet).
                    "warn": days > days_warn and upcoming == 0 and in_season})
    return sorted(out, key=lambda r: -r["days_stale"])


def refresh_health(comps: list | None = None) -> list[dict]:
    """Authoritative staleness: is the SOURCE ahead of what we hold?

    The season heuristic in staleness() is a cheap offline proxy. This is the
    real test and needs network: fetch each BSD-less league's source file and
    compare its latest result date to ours. A league is genuinely behind only
    if the source has results we have not ingested — which distinguishes the
    three cases the offline check conflates:

        source ahead of us   -> BROKEN: the refresh should have caught this
        source == us         -> healthy, whatever the calendar says
        source has nothing   -> off-season or pre-season; nothing to fetch

    "Danish Superliga is 66 days stale" looked alarming; this shows the source
    is also at 2026-05-17, so the new season simply has not published. Not a
    fault.
    """
    comps = needs_fdcouk_refresh() if comps is None else comps
    df = pd.read_csv(FIXTURES, low_memory=False)
    out = []
    for comp in comps:
        try:
            rows = load_competition(comp, refresh=True)
        except Exception as exc:
            out.append({"competition": comp.name, "error": str(exc),
                        "behind": True})
            continue
        source_latest = max((str(r["date"]) for r in rows), default="")
        ours = df[df["competition"] == comp.name]["date"]
        our_latest = str(ours.max())[:10] if not ours.empty else ""
        behind = bool(source_latest and source_latest > our_latest)
        out.append({"competition": comp.name, "source_latest": source_latest,
                    "our_latest": our_latest, "behind": behind})
    return out


def report() -> None:
    print(f"{'wave':<6}{'competition':<30}{'source':<16}{'seasons'}")
    for wave in sorted(WAVES):
        for comp in wave_competitions(wave):
            src = comp.fdcouk_code or f"new/{comp.fdcouk_new}"
            print(f"{wave:<6}{comp.name:<30}{src:<16}{len(SEASONS)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wave", type=int, help="ingest wave number")
    ap.add_argument("--write", action="store_true", help="write to fixtures.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="bypass the local cache")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--refresh-leagues", action="store_true",
                    help="re-fetch current results for leagues BSD cannot serve")
    ap.add_argument("--staleness", action="store_true",
                    help="days since last result, per competition")
    args = ap.parse_args()

    if args.staleness:
        rows = staleness()
        print(f"{'competition':<26}{'last result':>13}{'days':>6}{'upcoming':>10}  source")
        for r in rows:
            flag = " STALE" if r["warn"] else ""
            src = "BSD" if r["has_bsd_path"] else "fd.co.uk only"
            print(f"  {r['competition']:<24}{r['last_result']:>13}"
                  f"{r['days_stale']:>6}{r['upcoming']:>10}  {src}{flag}")
        return

    if args.refresh_leagues:
        comps = needs_fdcouk_refresh()
        print(f"Refreshing {len(comps)} league(s) with no BSD path:")
        refresh(comps, write=not args.dry_run)
        return

    if args.report or args.wave is None:
        report()
        return
    ingest(args.wave, write=args.write and not args.dry_run, refresh=args.refresh)


if __name__ == "__main__":
    main()
