"""
golf/providers.py  –  Data-source abstraction for the golf engine (v2).

One interface (`RoundsProvider`) so the model never knows where its data came
from.  Two implementations:

  • EspnProvider     – free.  One scoreboard call per season returns every PGA
                       event with full round-by-round linescores embedded.
  • DataGolfProvider – paid drop-in upgrade (true strokes-gained + categories +
                       course history).  Field/odds work today; round history is
                       a thin TODO so `get_provider()` keeps using ESPN for the
                       historical spine until it is filled in.

`get_provider()` returns DataGolf when a key is configured *and* it supports the
requested capability, otherwise ESPN — so adding a key later enriches the model
without touching `model.py` / `validate.py`.

Source-of-truth store written by fetch.py --accumulate: golf/data/rounds.csv
  tournament_id, date, tour, is_major, course, round,
  player, dg_id, score_to_par, field_size, made_cut, finish
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, runtime_checkable

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "api_cache"
ROUNDS_CSV = DATA_DIR / "rounds.csv"

ROOT = Path(__file__).resolve().parents[1]
# Append (don't insert at 0): root only provides api_keys; inserting it ahead of
# golf/ would shadow golf-local modules (edge, model, …) with the root engine's.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from api_keys import get_key  # noqa: E402

ROUNDS_COLUMNS = [
    "tournament_id", "date", "tour", "is_major", "course", "round",
    "player", "dg_id", "score_to_par", "field_size", "made_cut", "finish",
]

# Name fragments that mark a men's major (rotating venues handled at fit time).
_MAJOR_FRAGMENTS = (
    "masters tournament", "pga championship", "u.s. open", "us open",
    "the open championship", "the open", "open championship",
)


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


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _http_json(url: str, retries: int = 3, timeout: int = 25) -> Optional[dict]:
    """GET JSON with retries. Returns None on persistent failure (offline-safe)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  fetch failed ({url[:60]}…): {exc}", file=sys.stderr)
                return None
            time.sleep(1.5)
    return None


def _parse_to_par(disp: Optional[str]) -> Optional[float]:
    """ESPN round displayValue → strokes vs par. 'E'→0, '-6'→-6, '+2'→2."""
    s = (disp or "").strip()
    if s in ("", "E", "e", "-", "--", "WD", "DQ", "CUT", "MC"):
        return 0.0 if s in ("E", "e") else None
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None


def _is_major(name: str) -> bool:
    n = (name or "").lower()
    return any(frag in n for frag in _MAJOR_FRAGMENTS)


def _add_days(iso_date: str, days: int) -> str:
    try:
        d = _dt.date.fromisoformat(iso_date[:10])
        return (d + _dt.timedelta(days=days)).isoformat()
    except ValueError:
        return iso_date[:10]


# ─────────────────────────────────────────────
# ESPN provider (free)
# ─────────────────────────────────────────────

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
ESPN_LEADERBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/leaderboard"


class EspnProvider:
    """Round history + current field from ESPN's unofficial API.

    One call per season (`scoreboard?dates=YYYY`) returns every completed event
    with each competitor's per-round linescores embedded, so a few calls seed
    years of history. Raw season payloads are cached under data/api_cache/.
    """

    name = "espn"
    supports_history = True

    def __init__(self, seasons: Optional[Iterable[int]] = None):
        if seasons is None:
            yr = _dt.date.today().year
            seasons = [yr - 1, yr]
        self.seasons = sorted(set(int(s) for s in seasons))
        self._events: dict[str, dict] = {}   # tournament_id → raw event payload
        self._meta: dict[str, TournamentMeta] = {}

    # -- season payloads (cached) --
    def _season_payload(self, year: int) -> Optional[dict]:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / f"espn_pga_{year}.json"
        is_complete_past = year < _dt.date.today().year
        if cache.exists() and is_complete_past:
            try:
                return json.loads(cache.read_text())
            except (ValueError, OSError):
                pass
        data = _http_json(f"{ESPN_SCOREBOARD}?dates={year}")
        if data is not None and data.get("events"):
            try:
                cache.write_text(json.dumps(data))
            except OSError:
                pass
            return data
        # fall back to a stale cache rather than nothing (offline-safe)
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except (ValueError, OSError):
                return None
        return None

    def _load_all(self) -> None:
        if self._meta:
            return
        for yr in self.seasons:
            payload = self._season_payload(yr)
            if not payload:
                continue
            for ev in payload.get("events", []):
                tid = str(ev.get("id") or "")
                if not tid:
                    continue
                name = ev.get("name", "")
                date = (ev.get("date") or "")[:10]
                comp = (ev.get("competitions") or [{}])[0]
                course = _espn_course(ev, comp) or name
                self._events[tid] = ev
                self._meta[tid] = TournamentMeta(
                    tournament_id=tid, name=name, date=date, tour="pga",
                    is_major=_is_major(name), course=course)

    def recent_tournaments(self, since: Optional[str] = None) -> list[TournamentMeta]:
        self._load_all()
        out = [m for m in self._meta.values()
               if (since is None or m.date >= since)]
        return sorted(out, key=lambda m: m.date)

    def rounds_for(self, tournament_id: str) -> list[RoundRecord]:
        self._load_all()
        ev = self._events.get(str(tournament_id))
        meta = self._meta.get(str(tournament_id))
        if not ev or not meta:
            return []
        comp = (ev.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        field_size = len(competitors)
        out: list[RoundRecord] = []
        for c in competitors:
            name = (c.get("athlete") or {}).get("displayName", "").strip()
            if not name:
                continue
            dg_id = str((c.get("athlete") or {}).get("id") or "")
            try:
                finish = int(c.get("order") or 999)
            except (TypeError, ValueError):
                finish = 999
            lss = c.get("linescores") or []
            rounds = []
            for ls in lss:
                rnd = int(ls.get("period") or 0)
                stp = _parse_to_par(ls.get("displayValue"))
                if rnd and stp is not None:
                    rounds.append((rnd, stp))
            made_cut = 1 if len(rounds) >= 3 else 0
            for rnd, stp in rounds:
                out.append(RoundRecord(
                    tournament_id=meta.tournament_id,
                    date=_add_days(meta.date, rnd - 1),
                    tour=meta.tour, is_major=int(meta.is_major),
                    course=meta.course, round=rnd, player=name, dg_id=dg_id,
                    score_to_par=float(stp), field_size=field_size,
                    made_cut=made_cut, finish=finish))
        return out

    def field_for(self, event: Optional[str] = None) -> list[FieldEntry]:
        """Current/most-recent event field from the live leaderboard endpoint."""
        data = _http_json(ESPN_LEADERBOARD)
        out: list[FieldEntry] = []
        if not data:
            return out
        for ev in data.get("events", []):
            for comp in ev.get("competitions", []):
                for c in comp.get("competitors", []):
                    name = (c.get("athlete") or {}).get("displayName", "").strip()
                    if not name:
                        continue
                    status = (c.get("status") or {}).get("type", {}).get("name", "active")
                    out.append(FieldEntry(
                        name=name,
                        dg_id=str((c.get("athlete") or {}).get("id") or ""),
                        status=status))
        return out

    def pretournament_preds(self, event: Optional[str] = None) -> Optional[dict]:
        return None  # not available on the free tier

    def sg_categories(self, player: str, asof: Optional[str] = None) -> Optional[dict]:
        return None  # ESPN gives no strokes-gained categories


def _espn_course(ev: dict, comp: dict) -> str:
    for src in (comp.get("course"), ev.get("courses"), comp.get("venue")):
        if isinstance(src, dict) and src.get("name"):
            return src["name"]
        if isinstance(src, list) and src and isinstance(src[0], dict) and src[0].get("name"):
            return src[0]["name"]
    return ""


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
        from .fetch import fetch_dg_field
        out = []
        for p in fetch_dg_field(self.api_key):
            name = p.get("player_name") or p.get("name", "")
            if name:
                out.append(FieldEntry(name=name, dg_id=str(p.get("dg_id", "")),
                                      world_rank=int(p.get("owgr", 0) or 0)))
        return out

    def pretournament_preds(self, event: Optional[str] = None) -> Optional[dict]:
        from .fetch import fetch_dg_predictions
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


def accumulate_rounds(provider: Optional[RoundsProvider] = None,
                      since: Optional[str] = None,
                      verbose: bool = True) -> int:
    """Append any new (tournament,player,round) records to rounds.csv.

    Idempotent (dedupes on tournament_id+player+round) and offline-safe (writes
    nothing and returns 0 when the provider yields no data). Returns rows added.
    """
    import csv
    provider = provider or get_provider(need="history")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_rounds()
    seen = {(r["tournament_id"], r["player"], str(r["round"])) for r in existing}

    new_rows: list[dict] = []
    tournaments = provider.recent_tournaments(since=since)
    if verbose:
        print(f"[{provider.name}] {len(tournaments)} tournament(s) to scan"
              + (f" since {since}" if since else ""))
    for meta in tournaments:
        for rec in provider.rounds_for(meta.tournament_id):
            key = (rec.tournament_id, rec.player, str(rec.round))
            if key in seen:
                continue
            seen.add(key)
            new_rows.append(asdict(rec))

    if not new_rows:
        if verbose:
            print("  no new rounds")
        return 0

    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: (r["date"], r["tournament_id"], int(r["round"]),
                                 int(r["finish"])))
    with open(ROUNDS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROUNDS_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    if verbose:
        print(f"  +{len(new_rows)} rounds → {ROUNDS_CSV} ({len(all_rows)} total)")
    return len(new_rows)


if __name__ == "__main__":
    # quick manual check: python providers.py 2023 2024
    yrs = [int(a) for a in sys.argv[1:]] or None
    n = accumulate_rounds(EspnProvider(seasons=yrs) if yrs else None)
    print(f"added {n} rounds")
