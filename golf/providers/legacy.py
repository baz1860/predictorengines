"""
golf/providers/legacy.py  –  original round-history provider implementation.

One interface (`RoundsProvider`) so the model never knows where its data came
from.  Two implementations:

  • EspnProvider     – free. Season scoreboards discover event ids; per-event
                       leaderboards supply authoritative results and metadata.
  • DataGolfProvider – retained only for backward compatibility. The production
                       free-source stack should prefer the ESPN/PGA Tour/Open-
                       Meteo providers in this package.

`get_provider()` returns DataGolf when a key is configured *and* it supports the
requested capability, otherwise ESPN — so adding a key later enriches the model
without touching `model.py` / `validate.py`.

Source-of-truth store written by fetch.py --accumulate: golf/data/rounds.csv.
Season scoreboards are used only to discover event ids. Every event is then
read from ESPN's per-event leaderboard, which is the payload that contains
competitor statuses, real course metadata, tee times, and tournament rules.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, runtime_checkable

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "api_cache"
ROUNDS_CSV = DATA_DIR / "rounds.csv"

ROOT = Path(__file__).resolve().parents[2]
# Append (don't insert at 0): root only provides api_keys; inserting it ahead of
# golf/ would shadow golf-local modules (edge, model, …) with the root engine's.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from api_keys import get_key  # noqa: E402

ROUNDS_COLUMNS = [
    "tournament_id", "event_name", "date", "tour", "is_major",
    "course", "course_id", "course_name", "course_par", "course_yards",
    "round", "tee_time", "player", "dg_id", "score_to_par", "field_size",
    "made_cut", "finish", "cut_round", "cut_score", "cut_count",
    "total_rounds", "no_cut", "holes_scored", "birdies_or_better",
    "bogeys", "double_bogeys_or_worse", "par3_to_par", "par3_holes",
    "par4_to_par", "par4_holes", "par5_to_par", "par5_holes",
]

# Exact men's-major names. Substring matching misclassified the Australian PGA,
# BMW PGA, SA Open, and other DP World Tour events as majors and dropped them.
_MAJOR_NAMES = {
    "masters tournament", "pga championship", "u.s. open", "us open",
    "the open championship", "the open",
}


# ─────────────────────────────────────────────
# Records
# ─────────────────────────────────────────────

@dataclass
class TournamentMeta:
    tournament_id: str
    name: str
    date: str            # ISO start date, YYYY-MM-DD
    tour: str = "pga"
    is_major: bool = False
    course: str = ""
    course_id: str = ""
    course_par: int = 0
    course_yards: int = 0
    cut_round: int = 2
    cut_score: float | None = None
    cut_count: int = 65
    total_rounds: int = 4
    no_cut: bool = False


@dataclass
class RoundRecord:
    tournament_id: str
    date: str            # ISO date of the round (start + round-1)
    tour: str
    is_major: int        # 0/1 (stored as int for clean CSV)
    course: str
    round: int
    player: str
    dg_id: str
    score_to_par: float
    field_size: int
    made_cut: int
    finish: int
    event_name: str = ""
    course_id: str = ""
    course_name: str = ""
    course_par: int = 0
    course_yards: int = 0
    tee_time: str = ""
    cut_round: int = 2
    cut_score: float | None = None
    cut_count: int = 65
    total_rounds: int = 4
    no_cut: int = 0
    holes_scored: int = 0
    birdies_or_better: int = 0
    bogeys: int = 0
    double_bogeys_or_worse: int = 0
    par3_to_par: float = 0.0
    par3_holes: int = 0
    par4_to_par: float = 0.0
    par4_holes: int = 0
    par5_to_par: float = 0.0
    par5_holes: int = 0


@dataclass
class FieldEntry:
    name: str
    dg_id: str = ""
    world_rank: int = 0
    status: str = "active"


@runtime_checkable
class RoundsProvider(Protocol):
    name: str
    supports_history: bool

    def recent_tournaments(self, since: Optional[str] = None) -> list[TournamentMeta]: ...
    def rounds_for(self, tournament_id: str) -> list[RoundRecord]: ...
    def field_for(self, event: Optional[str] = None) -> list[FieldEntry]: ...
    def pretournament_preds(self, event: Optional[str] = None) -> Optional[dict]: ...
    def sg_categories(self, player: str, asof: Optional[str] = None) -> Optional[dict]: ...


def _is_major(name: str) -> bool:
    n = " ".join((name or "").lower().split())
    return n in _MAJOR_NAMES


# ─────────────────────────────────────────────
# ESPN provider compatibility exports
# ─────────────────────────────────────────────

# Live and historical ESPN ingestion now share the implementation in espn.py.
# These aliases preserve the original provider API used by fetch.py and callers.
from .espn import EspnGolfProvider, TOUR_SLUGS, _hole_features  # noqa: E402

EspnProvider = EspnGolfProvider


def _int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float_or_none(value) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────
# DataGolf provider (paid drop-in upgrade)
# ─────────────────────────────────────────────

class DataGolfProvider:
    """DataGolf-backed provider. Field/predictions work today; round-by-round
    history is the one remaining TODO, so `supports_history` is False and
    `get_provider()` keeps ESPN as the historical spine until it is wired.

    Filling rounds_for() with the `historical-raw-data/rounds` feed (true SG +
    categories) is the entire 'DataGolf later' upgrade — no other module changes.
    """

    name = "datagolf"
    supports_history = False  # flip to True once rounds_for() is implemented

    def __init__(self, api_key: str, seasons: Optional[Iterable[int]] = None):
        self.api_key = api_key
        self.seasons = list(seasons) if seasons else None

    def recent_tournaments(self, since: Optional[str] = None) -> list[TournamentMeta]:
        return []  # TODO: schedule feed

    def rounds_for(self, tournament_id: str) -> list[RoundRecord]:
        return []  # TODO: historical-raw-data/rounds → RoundRecord with SG cats

    def field_for(self, event: Optional[str] = None) -> list[FieldEntry]:
        from ..fetch import fetch_dg_field
        out = []
        for p in fetch_dg_field(self.api_key):
            name = p.get("player_name") or p.get("name", "")
            if name:
                out.append(FieldEntry(name=name, dg_id=str(p.get("dg_id", "")),
                                      world_rank=int(p.get("owgr", 0) or 0)))
        return out

    def pretournament_preds(self, event: Optional[str] = None) -> Optional[dict]:
        from ..fetch import fetch_dg_predictions
        try:
            return {"baseline": fetch_dg_predictions(self.api_key)}
        except Exception:  # noqa: BLE001
            return None

    def sg_categories(self, player: str, asof: Optional[str] = None) -> Optional[dict]:
        return None  # TODO: from historical rounds


# ─────────────────────────────────────────────
# Selection + store I/O
# ─────────────────────────────────────────────

def get_provider(seasons: Optional[Iterable[int]] = None,
                 need: str = "history") -> RoundsProvider:
    """Best provider for the requested capability.

    need="history" → must support round history; DataGolf falls back to ESPN
    until its history feed is wired.  need="field" → DataGolf used if keyed.
    """
    dg_key = get_key("datagolf", env="DG_API_KEY")
    if dg_key:
        dg = DataGolfProvider(dg_key, seasons=seasons)
        if need != "history" or dg.supports_history:
            return dg
    return EspnProvider(seasons=seasons)


def load_rounds() -> "list[dict]":
    """Read rounds.csv into a list of dict rows (empty if absent)."""
    import csv
    if not ROUNDS_CSV.exists():
        return []
    with open(ROUNDS_CSV) as f:
        return list(csv.DictReader(f))


def _normalise_round_row(row: dict) -> dict:
    """Canonical typed row used on both sides of accumulation comparisons."""
    course_name = str(row.get("course_name") or row.get("course") or "")
    return asdict(RoundRecord(
        tournament_id=str(row.get("tournament_id") or ""),
        event_name=str(row.get("event_name") or ""),
        date=str(row.get("date") or "")[:10],
        tour=str(row.get("tour") or "pga"),
        is_major=_int(row.get("is_major"), 0),
        course=course_name,
        course_id=str(row.get("course_id") or ""),
        course_name=course_name,
        course_par=_int(row.get("course_par"), 0),
        course_yards=_int(row.get("course_yards"), 0),
        round=_int(row.get("round"), 0),
        tee_time=str(row.get("tee_time") or ""),
        player=str(row.get("player") or ""),
        dg_id=str(row.get("dg_id") or ""),
        score_to_par=float(row.get("score_to_par") or 0.0),
        field_size=_int(row.get("field_size"), 0),
        made_cut=_int(row.get("made_cut"), 0),
        finish=_int(row.get("finish"), 999),
        cut_round=_int(row.get("cut_round"), 2),
        cut_score=_float_or_none(row.get("cut_score")),
        cut_count=_int(row.get("cut_count"), 65),
        total_rounds=_int(row.get("total_rounds"), 4),
        no_cut=_int(row.get("no_cut"), 0),
        holes_scored=_int(row.get("holes_scored"), 0),
        birdies_or_better=_int(row.get("birdies_or_better"), 0),
        bogeys=_int(row.get("bogeys"), 0),
        double_bogeys_or_worse=_int(row.get("double_bogeys_or_worse"), 0),
        par3_to_par=float(row.get("par3_to_par") or 0.0),
        par3_holes=_int(row.get("par3_holes"), 0),
        par4_to_par=float(row.get("par4_to_par") or 0.0),
        par4_holes=_int(row.get("par4_holes"), 0),
        par5_to_par=float(row.get("par5_to_par") or 0.0),
        par5_holes=_int(row.get("par5_holes"), 0),
    ))


def _round_key(row: dict) -> tuple[str, str, str]:
    pid = str(row.get("dg_id") or "").strip()
    identity = f"id:{pid}" if pid else f"name:{row.get('player', '')}"
    return str(row.get("tournament_id", "")), identity, str(row.get("round", ""))


def _sort_rounds(rows: list[dict]) -> None:
    rows.sort(key=lambda r: (
        r["date"], r["tournament_id"], int(r["round"]), int(r["finish"]),
        r["player"], r.get("dg_id", ""),
    ))


def _write_rounds(rows: list[dict], path: Path | None = None) -> None:
    from ..io_utils import atomic_write_csv
    path = path or ROUNDS_CSV
    atomic_write_csv(path, ROUNDS_COLUMNS, rows, extrasaction="ignore")


def accumulate_rounds(provider: Optional[RoundsProvider] = None,
                      since: Optional[str] = None,
                      verbose: bool = True) -> int:
    """Append any new (tournament,player,round) records to rounds.csv.

    Idempotent (dedupes on tournament_id+player+round) and offline-safe (writes
    nothing and returns 0 when the provider yields no data). Returns rows added.
    """
    provider = provider or get_provider(need="history")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing = [_normalise_round_row(r) for r in load_rounds()]
    merged = {_round_key(r): r for r in existing}

    new_rows: list[dict] = []
    tournaments = provider.recent_tournaments(since=since)
    if verbose:
        print(f"[{provider.name}] {len(tournaments)} tournament(s) to scan"
              + (f" since {since}" if since else ""))
    for meta in tournaments:
        for rec in provider.rounds_for(meta.tournament_id):
            row = _normalise_round_row(asdict(rec))
            key = _round_key(row)
            if merged.get(key) == row:
                continue
            merged[key] = row
            new_rows.append(row)

    if not new_rows:
        if verbose:
            print("  no new rounds")
        return 0

    all_rows = list(merged.values())
    _sort_rounds(all_rows)
    _write_rounds(all_rows)
    if verbose:
        print(f"  +{len(new_rows)} rounds → {ROUNDS_CSV} ({len(all_rows)} total)")
    return len(new_rows)


def rebuild_tours(tours: Iterable[str], seasons: Iterable[int],
                  verbose: bool = True) -> dict[str, int]:
    """Rebuild rounds.csv only from corrected per-event payloads.

    The existing CSV is never read and is replaced atomically only after at
    least one complete event was parsed. This is the safe repair path for
    histories written by the old status-less season parser.
    """
    seasons = list(seasons)
    all_rows: dict[tuple[str, str, str], dict] = {}
    results: dict[str, int] = {}
    for tour in tours:
        slug = TOUR_SLUGS.get(str(tour).lower(), str(tour).lower())
        provider = EspnProvider(seasons=seasons, tour=slug)
        tournaments = provider.recent_tournaments()
        if verbose:
            print(f"── tour: {slug} · {len(tournaments)} event(s) ──")
        before = len(all_rows)
        for idx, meta in enumerate(tournaments, 1):
            rows = provider.rounds_for(meta.tournament_id)
            for rec in rows:
                row = _normalise_round_row(asdict(rec))
                all_rows[_round_key(row)] = row
            if verbose and (idx % 10 == 0 or idx == len(tournaments)):
                print(f"  {idx:>3}/{len(tournaments)} events · "
                      f"{len(all_rows):,} rounds")
        results[slug] = len(all_rows) - before
    if not all_rows:
        raise RuntimeError("rebuild produced no rounds; existing rounds.csv preserved")
    rows = list(all_rows.values())
    _sort_rounds(rows)
    _write_rounds(rows)
    if verbose:
        print(f"Rebuilt {ROUNDS_CSV} atomically ({len(rows):,} rows)")
    return results


def accumulate_tours(tours: Iterable[str],
                     seasons: Optional[Iterable[int]] = None,
                     since: Optional[str] = None,
                     verbose: bool = True) -> dict[str, int]:
    """Accumulate round history for several tours into the one rounds.csv.

    ``tours`` accepts slugs or aliases (``pga``, ``liv``, ``eur``/``dpwt``).
    Each tour is fetched with its own ESPN feed and appended via the same
    idempotent, offline-safe ``accumulate_rounds`` path. PGA always uses the
    keyed provider when available (DataGolf) so nothing regresses; LIV/DPWT go
    straight to ESPN, which is the only free source that carries them.

    Returns ``{tour_slug: rows_added}``.
    """
    seasons = list(seasons) if seasons is not None else None
    results: dict[str, int] = {}
    for tour in tours:
        slug = TOUR_SLUGS.get(str(tour).lower(), str(tour).lower())
        if slug == "pga":
            provider: RoundsProvider = get_provider(seasons=seasons, need="history")
        else:
            provider = EspnProvider(seasons=seasons, tour=slug)
        if verbose:
            print(f"── tour: {slug} ──")
        results[slug] = accumulate_rounds(provider, since=since, verbose=verbose)
    return results


if __name__ == "__main__":
    # quick manual check: python providers.py 2023 2024
    yrs = [int(a) for a in sys.argv[1:]] or None
    n = accumulate_rounds(EspnProvider(seasons=yrs) if yrs else None)
    print(f"added {n} rounds")
