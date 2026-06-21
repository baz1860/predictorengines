"""tennis/providers.py — data-source abstraction for the tennis engine.

One interface (`MatchProvider`) so the model never knows where its rows came
from. Free, no-key implementations:

  - `TMLProvider` (ATP): Tennismylife/TML-Database on GitHub — per-season CSVs
    in the same column format as the former Sackmann ATP repo (1968–present).
    https://github.com/Tennismylife/TML-Database

  - `MatchChartingProvider` (ATP + WTA supplementary): JeffSackmann's
    tennis_MatchChartingProject — a single flat CSV per tour with hand-charted
    match metadata (player names, surface, round, date). Set scores and player
    ranks are not available; Player 1 is treated as the winner (heuristic).
    https://github.com/JeffSackmann/tennis_MatchChartingProject

  - `CompositeProvider` (default): routes ATP → TMLProvider, WTA →
    MatchChartingProvider.

  - `SackmannProvider` (legacy): the original Sackmann ATP+WTA repos; kept for
    reference but both repos returned HTTP 404 as of June 2026.

All normalise to the canonical `matches.csv` schema:

    date, tourney_id, tourney_name, tour, surface, round, best_of,
    winner, loser, winner_rank, loser_rank, winner_sets, loser_sets, score

A paid feed (Sportradar, Tennis Abstract API, …) can drop in later behind the
same interface without touching model/simulate/edge. Raw season CSVs are cached
under tennis/data/api_cache/ so re-seeding is offline-safe and cheap.
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, runtime_checkable

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "api_cache"
MATCHES_CSV = DATA_DIR / "matches.csv"

# Canonical store columns (one row per completed match).
MATCH_COLUMNS = [
    "date", "tourney_id", "tourney_name", "tour", "surface", "round",
    "best_of", "winner", "loser", "winner_rank", "loser_rank",
    "winner_sets", "loser_sets", "score",
]

VALID_SURFACES = ("hard", "clay", "grass", "carpet")


# ─────────────────────────────────────────────
# Records
# ─────────────────────────────────────────────

@dataclass
class MatchRecord:
    date: str            # ISO YYYY-MM-DD (from Sackmann's YYYYMMDD tourney_date)
    tourney_id: str
    tourney_name: str
    tour: str            # "atp" | "wta"
    surface: str         # hard | clay | grass | carpet
    round: str           # F, SF, QF, R16, R32, R64, R128, RR, …
    best_of: int         # 3 or 5
    winner: str
    loser: str
    winner_rank: int     # 9999 when unknown
    loser_rank: int
    winner_sets: int
    loser_sets: int
    score: str


@runtime_checkable
class MatchProvider(Protocol):
    name: str

    def seasons_available(self) -> list[int]: ...
    def matches_for(self, year: int, tour: str) -> list[MatchRecord]: ...


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _http_text(url: str, retries: int = 3, timeout: int = 30) -> Optional[str]:
    """GET a text body with retries. Returns None on persistent failure so the
    caller can fall back to cache (offline-safe)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  fetch failed ({url[:70]}…): {exc}", file=sys.stderr)
                return None
            time.sleep(1.5)
    return None


def normalise_surface(raw: Optional[str]) -> str:
    s = (raw or "").strip().lower()
    return s if s in VALID_SURFACES else ("hard" if not s else s)


def _iso_date(yyyymmdd: Optional[str]) -> str:
    """Sackmann tourney_date is an 8-digit YYYYMMDD. → ISO YYYY-MM-DD."""
    s = str(yyyymmdd or "").strip()
    if len(s) >= 8 and s[:8].isdigit():
        y, m, d = s[:4], s[4:6], s[6:8]
        # Guard against the odd 00 month/day Sackmann sometimes carries.
        m = m if m != "00" else "01"
        d = d if d != "00" else "01"
        try:
            return _dt.date(int(y), int(m), int(d)).isoformat()
        except ValueError:
            return f"{y}-01-01"
    return ""


def _int_rank(raw: Optional[str], default: int = 9999) -> int:
    s = str(raw or "").strip()
    if not s:
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def parse_set_score(score: Optional[str]) -> tuple[int, int]:
    """Sets won by (winner, loser) from a Sackmann score string.

    Handles tiebreak annotations ("7-6(5)"), retirements/walkovers ("6-3 RET",
    "W/O"), and unfinished strings by counting only completed sets where one
    side has strictly more games. The winner is always listed first per set, so
    a set counts for whoever has the higher game count.
    """
    s = (score or "").strip()
    if not s or s.upper() in ("W/O", "WO", "DEF", "WALKOVER"):
        return (0, 0)
    w = l = 0
    for tok in s.split():
        t = tok.upper()
        if t in ("RET", "RET.", "DEF", "DEF.", "W/O", "WO", "ABN", "ABD", "UNK"):
            break
        # Strip a trailing tiebreak annotation: "7-6(5)" -> "7-6"
        core = tok.split("(", 1)[0]
        if "-" not in core:
            continue
        a, _, b = core.partition("-")
        try:
            ga, gb = int(a), int(b)
        except ValueError:
            continue
        if ga > gb:
            w += 1
        elif gb > ga:
            l += 1
    return (w, l)


# ─────────────────────────────────────────────
# Sackmann provider (free, no key)
# ─────────────────────────────────────────────

_SACKMANN_BASE = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master",
}
_FILE_PREFIX = {"atp": "atp_matches", "wta": "wta_matches"}


class SackmannProvider:
    """Per-season ATP/WTA match archives from Jeff Sackmann's GitHub repos.

    One HTTP call per (tour, year) returns every completed match for that
    season; raw payloads are cached under data/api_cache/ so a completed past
    season is fetched at most once.
    """

    name = "sackmann"

    def __init__(self, tours: Iterable[str] = ("atp", "wta")):
        self.tours = [t.lower() for t in tours if t.lower() in _SACKMANN_BASE]

    def seasons_available(self) -> list[int]:
        # Sackmann data runs 1968→present; callers pass explicit --seed years.
        return list(range(1968, _dt.date.today().year + 1))

    def _season_csv(self, year: int, tour: str) -> Optional[str]:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / f"{tour}_matches_{year}.csv"
        is_complete_past = year < _dt.date.today().year
        if cache.exists() and is_complete_past:
            try:
                return cache.read_text(encoding="utf-8")
            except OSError:
                pass
        url = f"{_SACKMANN_BASE[tour]}/{_FILE_PREFIX[tour]}_{year}.csv"
        text = _http_text(url)
        if text and text.lstrip().lower().startswith(("tourney_id", "﻿tourney_id")):
            try:
                cache.write_text(text, encoding="utf-8")
            except OSError:
                pass
            return text
        # offline / not-found → stale cache if we have one
        if cache.exists():
            try:
                return cache.read_text(encoding="utf-8")
            except OSError:
                return None
        return None

    def matches_for(self, year: int, tour: str) -> list[MatchRecord]:
        tour = tour.lower()
        text = self._season_csv(year, tour)
        if not text:
            return []
        out: list[MatchRecord] = []
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            winner = (row.get("winner_name") or "").strip()
            loser = (row.get("loser_name") or "").strip()
            if not winner or not loser:
                continue
            score = (row.get("score") or "").strip()
            ws, ls = parse_set_score(score)
            try:
                best_of = int(float(row.get("best_of") or 3))
            except ValueError:
                best_of = 3
            out.append(MatchRecord(
                date=_iso_date(row.get("tourney_date")),
                tourney_id=str(row.get("tourney_id") or "").strip(),
                tourney_name=(row.get("tourney_name") or "").strip(),
                tour=tour,
                surface=normalise_surface(row.get("surface")),
                round=(row.get("round") or "").strip(),
                best_of=best_of,
                winner=winner,
                loser=loser,
                winner_rank=_int_rank(row.get("winner_rank")),
                loser_rank=_int_rank(row.get("loser_rank")),
                winner_sets=ws,
                loser_sets=ls,
                score=score,
            ))
        return out


# ─────────────────────────────────────────────
# TML-Database provider (free, no key, ATP only)
# ─────────────────────────────────────────────

_TML_BASE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"


class TMLProvider:
    """ATP match archives from Tennismylife/TML-Database (1968–present).

    Same column layout as the former Sackmann ATP repo so row normalisation is
    identical. ATP only — no WTA data in this repo.
    """

    name = "tml"

    def seasons_available(self) -> list[int]:
        return list(range(1968, _dt.date.today().year + 1))

    def _season_csv(self, year: int) -> Optional[str]:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / f"tml_atp_{year}.csv"
        is_complete_past = year < _dt.date.today().year
        if cache.exists() and is_complete_past:
            try:
                return cache.read_text(encoding="utf-8")
            except OSError:
                pass
        url = f"{_TML_BASE}/{year}.csv"
        text = _http_text(url)
        if text and "tourney_id" in text[:300]:
            try:
                cache.write_text(text, encoding="utf-8")
            except OSError:
                pass
            return text
        if cache.exists():
            try:
                return cache.read_text(encoding="utf-8")
            except OSError:
                return None
        return None

    def matches_for(self, year: int, tour: str) -> list[MatchRecord]:
        if tour.lower() != "atp":
            return []
        text = self._season_csv(year)
        if not text:
            return []
        out: list[MatchRecord] = []
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            winner = (row.get("winner_name") or "").strip()
            loser = (row.get("loser_name") or "").strip()
            if not winner or not loser:
                continue
            score = (row.get("score") or "").strip()
            ws, ls = parse_set_score(score)
            try:
                best_of = int(float(row.get("best_of") or 3))
            except ValueError:
                best_of = 3
            out.append(MatchRecord(
                date=_iso_date(row.get("tourney_date")),
                tourney_id=str(row.get("tourney_id") or "").strip(),
                tourney_name=(row.get("tourney_name") or "").strip(),
                tour="atp",
                surface=normalise_surface(row.get("surface")),
                round=(row.get("round") or "").strip(),
                best_of=best_of,
                winner=winner,
                loser=loser,
                winner_rank=_int_rank(row.get("winner_rank")),
                loser_rank=_int_rank(row.get("loser_rank")),
                winner_sets=ws,
                loser_sets=ls,
                score=score,
            ))
        return out


# ─────────────────────────────────────────────
# MatchCharting provider (free, no key, ATP + WTA)
# ─────────────────────────────────────────────

_MCP_BASE = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"
)
_MCP_FILES = {"atp": "charting-m-matches.csv", "wta": "charting-w-matches.csv"}


class MatchChartingProvider:
    """Match metadata from JeffSackmann/tennis_MatchChartingProject (ATP + WTA).

    Coverage is selective (hand-charted matches only), and set scores / player
    ranks are not present in the matches metadata file. Player 1 is treated as
    the winner — a heuristic that holds for the majority of charted matches but
    is not guaranteed. Best used as a supplementary WTA source.
    """

    name = "match_charting"

    def __init__(self, tours: Iterable[str] = ("wta",)):
        self.tours = [t.lower() for t in tours if t.lower() in _MCP_FILES]

    def seasons_available(self) -> list[int]:
        return list(range(2000, _dt.date.today().year + 1))

    def _tour_csv(self, tour: str) -> Optional[str]:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / f"mcp_{tour}_matches.csv"
        # Re-fetch once per day — this is a single growing file.
        stale = True
        if cache.exists():
            age = (_dt.date.today() - _dt.date.fromtimestamp(cache.stat().st_mtime)).days
            stale = age >= 1
        if cache.exists() and not stale:
            try:
                return cache.read_text(encoding="utf-8")
            except OSError:
                pass
        url = f"{_MCP_BASE}/{_MCP_FILES[tour]}"
        text = _http_text(url)
        if text and "match_id" in text[:200]:
            try:
                cache.write_text(text, encoding="utf-8")
            except OSError:
                pass
            return text
        if cache.exists():
            try:
                return cache.read_text(encoding="utf-8")
            except OSError:
                return None
        return None

    def matches_for(self, year: int, tour: str) -> list[MatchRecord]:
        tour = tour.lower()
        if tour not in self.tours:
            return []
        text = self._tour_csv(tour)
        if not text:
            return []
        year_prefix = str(year)
        out: list[MatchRecord] = []
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            date_raw = str(row.get("Date") or "").strip()
            if not date_raw.startswith(year_prefix):
                continue
            winner = (row.get("Player 1") or "").strip()
            loser = (row.get("Player 2") or "").strip()
            if not winner or not loser:
                continue
            tourney = (row.get("Tournament") or "").strip()
            tourney_id = f"mcp-{date_raw[:6]}-{tourney.replace(' ', '_').lower()}"
            try:
                best_of = int(str(row.get("Best of") or "3").strip())
            except ValueError:
                best_of = 3
            out.append(MatchRecord(
                date=_iso_date(date_raw),
                tourney_id=tourney_id,
                tourney_name=tourney,
                tour=tour,
                surface=normalise_surface(row.get("Surface")),
                round=(row.get("Round") or "").strip(),
                best_of=best_of,
                winner=winner,
                loser=loser,
                winner_rank=9999,
                loser_rank=9999,
                winner_sets=0,
                loser_sets=0,
                score="",
            ))
        return out


# ─────────────────────────────────────────────
# Composite provider (default: TML for ATP, MatchCharting for WTA)
# ─────────────────────────────────────────────

class CompositeProvider:
    """Routes ATP → TMLProvider and WTA → MatchChartingProvider.

    Default provider after the original Sackmann repos became unavailable
    (both returned HTTP 404 as of June 2026).
    """

    name = "composite"

    def __init__(self) -> None:
        self._atp = TMLProvider()
        self._wta = MatchChartingProvider(tours=("wta",))

    def seasons_available(self) -> list[int]:
        atp = set(self._atp.seasons_available())
        wta = set(self._wta.seasons_available())
        return sorted(atp | wta)

    def matches_for(self, year: int, tour: str) -> list[MatchRecord]:
        t = tour.lower()
        if t == "atp":
            return self._atp.matches_for(year, tour)
        if t == "wta":
            return self._wta.matches_for(year, tour)
        return []


# ─────────────────────────────────────────────
# Store I/O
# ─────────────────────────────────────────────

def load_matches() -> list[dict]:
    """Read matches.csv into a list of dict rows (empty if absent)."""
    if not MATCHES_CSV.exists():
        return []
    with open(MATCHES_CSV, newline="") as f:
        return list(csv.DictReader(f))


def accumulate_matches(provider: Optional[MatchProvider] = None,
                       years: Optional[Iterable[int]] = None,
                       tours: Iterable[str] = ("atp", "wta"),
                       verbose: bool = True) -> int:
    """Append any new completed matches to matches.csv.

    Idempotent (dedupes on tour+tourney_id+winner+loser+round) and offline-safe
    (writes nothing, returns 0 when the provider yields no rows). When `years`
    is None, refreshes the current and previous season. Returns rows added.
    """
    provider = provider or CompositeProvider()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if years is None:
        yr = _dt.date.today().year
        years = [yr - 1, yr]
    years = sorted({int(y) for y in years})
    tours = [t.lower() for t in tours]

    existing = load_matches()
    seen = {
        (r.get("tour", ""), r.get("tourney_id", ""), r.get("winner", ""),
         r.get("loser", ""), r.get("round", ""))
        for r in existing
    }

    new_rows: list[dict] = []
    for tour in tours:
        for year in years:
            recs = provider.matches_for(year, tour)
            if verbose:
                print(f"[{provider.name}] {tour} {year}: {len(recs)} match(es)")
            for rec in recs:
                key = (rec.tour, rec.tourney_id, rec.winner, rec.loser, rec.round)
                if key in seen:
                    continue
                seen.add(key)
                new_rows.append(asdict(rec))

    if not new_rows:
        if verbose:
            print("  no new matches")
        return 0

    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: (str(r.get("date", "")), str(r.get("tour", "")),
                                 str(r.get("tourney_id", "")), str(r.get("round", ""))))
    with open(MATCHES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MATCH_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    if verbose:
        print(f"  +{len(new_rows)} matches → {MATCHES_CSV} ({len(all_rows)} total)")
    return len(new_rows)


if __name__ == "__main__":
    # quick manual check: python -m tennis.providers 2023 2024
    yrs = [int(a) for a in sys.argv[1:]] or None
    n = accumulate_matches(years=yrs)
    print(f"added {n} matches")
