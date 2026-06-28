"""ESPN/golfastR-style provider for free PGA event data.

ESPN's public golf JSON endpoints are the current free source of truth for this
engine's event spine: schedule, leaderboard, field, round scores, and embedded
hole-by-hole scorecards when present. The implementation is intentionally cache
first so parser failures can be debugged from saved payloads.
"""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .. import provider_qa as qa

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "api_cache" / "espn"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
# ESPN's older /leaderboard endpoint has returned 404 in current checks. The
# site scoreboard endpoint carries the same event/competitor/linescore payload.
ESPN_LEADERBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/leaderboard"


@dataclass(frozen=True)
class EspnEvent:
    event_id: str
    name: str
    start_date: str
    end_date: str = ""
    course_name: str = ""
    status: str = ""
    tour: str = "pga"
    source: str = "espn"
    source_event_id: str = ""

    def as_store_row(self) -> dict:
        row = asdict(self)
        row["source_event_id"] = self.source_event_id or self.event_id
        row["event_id"] = self.event_id
        return row


@dataclass(frozen=True)
class EspnFieldEntry:
    name: str
    source_player_id: str = ""
    status: str = "active"
    country: str = ""
    world_rank: int | None = None
    source: str = "espn"

    def as_store_row(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HoleScore:
    event_id: str
    player_name: str
    player_id: str
    round_no: int
    hole: int
    score: int
    score_type: str = ""
    source: str = "espn"


class EspnGolfProvider:
    name = "espn"

    def __init__(self, cache_dir: Path | None = None, ttl_seconds: int = 900):
        self.cache_dir = cache_dir or CACHE_DIR
        self.ttl_seconds = ttl_seconds

    def schedule(self, season: int | None = None, use_cache: bool = True) -> list[EspnEvent]:
        season = season or dt.date.today().year
        payload = self._json("scoreboard", ESPN_SCOREBOARD, {"dates": str(season)}, use_cache)
        events = []
        for ev in payload.get("events", []) or []:
            events.append(_event_from_payload(ev))
        return sorted(events, key=lambda e: e.start_date)

    def current_event_payload(self, event_id: str | None = None,
                              use_cache: bool = False) -> dict:
        params = {"event": event_id} if event_id else {}
        label = "scoreboard_current" if not event_id else "scoreboard_event"
        return self._json(label, ESPN_SCOREBOARD, params, use_cache)

    def current_event(self, event_id: str | None = None,
                      use_cache: bool = False) -> EspnEvent | None:
        payload = self.current_event_payload(event_id, use_cache=use_cache)
        events = payload.get("events", []) or []
        return _event_from_payload(events[0]) if events else None

    def field(self, event_id: str | None = None,
              use_cache: bool = False) -> list[EspnFieldEntry]:
        payload = self.current_event_payload(event_id, use_cache=use_cache)
        out = []
        for ev in payload.get("events", []) or []:
            for comp in ev.get("competitions", []) or []:
                for c in comp.get("competitors", []) or []:
                    athlete = c.get("athlete") or {}
                    name = (athlete.get("displayName") or athlete.get("fullName") or "").strip()
                    if not name:
                        continue
                    status = _status_name(c) or "active"
                    flag = athlete.get("flag") or {}
                    out.append(EspnFieldEntry(
                        name=name,
                        source_player_id=str(athlete.get("id") or c.get("id") or ""),
                        status=status,
                        country=str(flag.get("alt") or ""),
                        world_rank=_safe_int(c.get("rank")),
                    ))
        return out

    def leaderboard_rows(self, event_id: str | None = None,
                         use_cache: bool = False) -> list[dict]:
        payload = self.current_event_payload(event_id, use_cache=use_cache)
        rows = []
        for ev in payload.get("events", []) or []:
            eid = str(ev.get("id") or event_id or "")
            for comp in ev.get("competitions", []) or []:
                for c in comp.get("competitors", []) or []:
                    athlete = c.get("athlete") or {}
                    lines = c.get("linescores") or []
                    rows.append({
                        "event_id": eid,
                        "player_id": str(athlete.get("id") or c.get("id") or ""),
                        "name": athlete.get("displayName") or "",
                        "position": c.get("order") or "",
                        "score": c.get("score") or c.get("displayValue") or "",
                        "status": _status_name(c),
                        "rounds_played": len(lines),
                    })
        return rows

    def hole_scores(self, event_id: str | None = None,
                    use_cache: bool = False) -> list[HoleScore]:
        payload = self.current_event_payload(event_id, use_cache=use_cache)
        out = []
        for ev in payload.get("events", []) or []:
            eid = str(ev.get("id") or event_id or "")
            for comp in ev.get("competitions", []) or []:
                for c in comp.get("competitors", []) or []:
                    athlete = c.get("athlete") or {}
                    name = (athlete.get("displayName") or "").strip()
                    pid = str(athlete.get("id") or c.get("id") or "")
                    if not name:
                        continue
                    for round_line in c.get("linescores", []) or []:
                        rnd = _safe_int(round_line.get("period"))
                        if not rnd:
                            continue
                        for hole_line in round_line.get("linescores", []) or []:
                            hole = _safe_int(hole_line.get("period"))
                            score = _safe_int(hole_line.get("value"))
                            if not hole or score is None:
                                continue
                            st = hole_line.get("scoreType") or {}
                            out.append(HoleScore(
                                event_id=eid,
                                player_name=name,
                                player_id=pid,
                                round_no=rnd,
                                hole=hole,
                                score=score,
                                score_type=str(st.get("displayValue") or ""),
                            ))
        return out

    def completed_round_scores(
        self, event_id: str | None = None, use_cache: bool = False
    ) -> tuple[list[dict], int]:
        """Build a between-rounds scores snapshot from the live leaderboard.

        Returns ``(rows, rounds_done)`` where each row is
        ``{"name", "score", "made_cut", "completed"}``:

          * ``score``    – cumulative strokes-to-par through the completed rounds
                           (the number shown on the leaderboard between rounds).
          * ``completed``– how many rounds this player has fully finished.
          * ``made_cut`` – 1 unless the player is cut/withdrawn/disqualified, or
                           has played fewer rounds than the field (i.e. is out).

        ``rounds_done`` is the number of rounds completed by the bulk of the
        field — the largest round R that at least half of the started players
        have finished. A round counts as "finished" for a player only when all
        18 holes are present, so an in-progress round is never double-counted.
        This is intentionally a *between-rounds* view: run it after a round
        completes and before the next tees off.
        """
        payload = self.current_event_payload(event_id, use_cache=use_cache)
        players: list[dict] = []
        for ev in payload.get("events", []) or []:
            for comp in ev.get("competitions", []) or []:
                for c in comp.get("competitors", []) or []:
                    athlete = c.get("athlete") or {}
                    name = (athlete.get("displayName") or athlete.get("fullName") or "").strip()
                    if not name:
                        continue
                    cum = 0.0
                    completed = 0
                    for rline in c.get("linescores") or []:
                        holes = rline.get("linescores") or []
                        if len(holes) < 18:
                            continue  # round in progress — don't count it
                        completed += 1
                        cum += _to_par(rline.get("displayValue"))
                    players.append({
                        "name": name,
                        "score": cum,
                        "completed": completed,
                        "_cut_flag": _is_out(c),
                    })

        started = [p for p in players if p["completed"] > 0]
        rounds_done = 0
        if started:
            for r in (3, 2, 1):
                if sum(1 for p in started if p["completed"] >= r) >= 0.5 * len(started):
                    rounds_done = r
                    break

        rows = []
        for p in players:
            made_cut = 1
            if p["_cut_flag"] or (rounds_done and p["completed"] < rounds_done):
                made_cut = 0
            rows.append({
                "name": p["name"],
                "score": int(p["score"]) if float(p["score"]).is_integer() else p["score"],
                "made_cut": made_cut,
                "completed": p["completed"],
            })
        return rows, rounds_done

    def qa_checks(self, field_rows: Iterable[EspnFieldEntry]) -> list[qa.SourceCheck]:
        rows = [r.as_store_row() for r in field_rows]
        return [
            qa.require_columns("espn.field", rows, ["name", "status", "source_player_id"]),
            qa.min_rows("espn.field", rows, 20),
        ]

    def _json(self, label: str, url: str, params: dict | None = None,
              use_cache: bool = True) -> dict:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        params = {k: v for k, v in (params or {}).items() if v not in ("", None)}
        cache_key = label
        if params:
            suffix = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
            cache_key = f"{label}_{suffix}"
        cache = self.cache_dir / f"{cache_key}.json"
        if use_cache and cache.exists() and time.time() - cache.stat().st_mtime <= self.ttl_seconds:
            return json.loads(cache.read_text())
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}" if query else url
        req = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = json.load(resp)
        cache.write_text(json.dumps(payload))
        return payload


def _event_from_payload(ev: dict) -> EspnEvent:
    comp = (ev.get("competitions") or [{}])[0]
    status = ((ev.get("status") or {}).get("type") or {}).get("name") or \
        ((comp.get("status") or {}).get("type") or {}).get("name") or ""
    return EspnEvent(
        event_id=str(ev.get("id") or ""),
        source_event_id=str(ev.get("id") or ""),
        name=str(ev.get("name") or ev.get("shortName") or ""),
        start_date=str(ev.get("date") or comp.get("date") or "")[:10],
        end_date=str(ev.get("endDate") or "")[:10],
        course_name=_course_name(ev, comp),
        status=status,
    )


def _course_name(ev: dict, comp: dict) -> str:
    for src in (comp.get("course"), ev.get("courses"), comp.get("venue")):
        if isinstance(src, dict) and src.get("name"):
            return str(src["name"])
        if isinstance(src, list) and src and isinstance(src[0], dict) and src[0].get("name"):
            return str(src[0]["name"])
    return str(ev.get("name") or "")


def _status_name(comp: dict) -> str:
    return str(((comp.get("status") or {}).get("type") or {}).get("name") or "")


def _to_par(value) -> float:
    """Parse an ESPN to-par display ('-7', 'E', '+2', '') into a number."""
    s = str(value or "").strip()
    if s in ("", "E", "e", "EVEN", "Even", "even", "-", "--"):
        return 0.0
    try:
        return float(s.replace("+", ""))
    except (TypeError, ValueError):
        return 0.0


def _is_out(competitor: dict) -> bool:
    """True if the competitor is cut, withdrawn, or disqualified."""
    blob = " ".join(str(x) for x in (
        _status_name(competitor),
        ((competitor.get("status") or {}).get("type") or {}).get("description"),
        competitor.get("displayValue"),
    )).upper()
    return any(tok in blob for tok in ("CUT", "WD", "WITHDR", "DQ", "DISQ", "MDF"))


def _safe_int(value) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
