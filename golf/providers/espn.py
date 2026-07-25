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
from typing import Iterable, Optional

from .. import provider_qa as qa

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "api_cache" / "espn"
HISTORY_CACHE_DIR = DATA_DIR / "api_cache"
ESPN_SCOREBOARD_TMPL = "https://site.api.espn.com/apis/site/v2/sports/golf/{tour}/scoreboard"
ESPN_LEADERBOARD = "https://site.web.api.espn.com/apis/site/v2/sports/golf/leaderboard"
TOUR_SLUGS = {
    "pga": "pga",
    "liv": "liv",
    "eur": "eur",
    "dpwt": "eur",
    "euro": "eur",
    "european": "eur",
}
_MAJOR_NAMES = {
    "masters tournament",
    "pga championship",
    "u.s. open",
    "us open",
    "the open championship",
    "the open",
}


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
    cut_rule: int = 65
    no_cut: bool = False
    total_rounds: int = 4
    course_id: str = ""
    course_par: int = 0
    course_yards: int = 0
    par3_holes: int = 0
    par4_holes: int = 0
    par5_holes: int = 0
    cut_round: int = 2
    cut_score: float | None = None
    realized_cut_count: int = 0
    multi_course: bool = False

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
    tee_time_r1: str = ""
    tee_time_r2: str = ""
    start_hole_r1: str = ""
    start_hole_r2: str = ""
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
    """The single ESPN implementation for live data and historical rounds.

    Season scoreboards discover event ids only. Every field, leaderboard, and
    historical result is parsed from the event-specific leaderboard payload.
    """

    name = "espn"
    supports_history = True

    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_seconds: int = 900,
        seasons: Optional[Iterable[int]] = None,
        tour: str = "pga",
        history_cache_dir: Path | None = None,
    ):
        self.cache_dir = cache_dir or CACHE_DIR
        self.history_cache_dir = history_cache_dir or HISTORY_CACHE_DIR
        self.ttl_seconds = ttl_seconds
        if seasons is None:
            year = dt.date.today().year
            seasons = [year - 1, year]
        self.seasons = sorted(set(int(season) for season in seasons))
        self.tour_slug = TOUR_SLUGS.get(str(tour).lower(), str(tour).lower())
        self.name = f"espn-{self.tour_slug}"
        self._events: dict[str, dict] = {}
        self._meta: dict[str, object] = {}

    def schedule(self, season: int | None = None, use_cache: bool = True) -> list[EspnEvent]:
        season = season or dt.date.today().year
        scoreboard = ESPN_SCOREBOARD_TMPL.format(tour=self.tour_slug)
        payload = self._json(
            f"scoreboard_{self.tour_slug}",
            scoreboard,
            {"dates": str(season)},
            use_cache,
        )
        events = []
        for ev in payload.get("events", []) or []:
            events.append(_event_from_payload(ev, tour=self.tour_slug))
        return sorted(events, key=lambda e: e.start_date)

    def current_event_payload(self, event_id: str | None = None,
                              use_cache: bool = False) -> dict:
        params = {"league": self.tour_slug, "event": event_id}
        label = "leaderboard_current" if not event_id else f"leaderboard_{event_id}"
        return self._json(label, ESPN_LEADERBOARD, params, use_cache)

    def _resolve_event_dates(self, query: str) -> tuple[str, str]:
        """Map an event id or name to (YYYYMMDD, cache_label) via the season
        schedule. Returns ('', '') when nothing matches (caller falls back to the
        current-week board)."""
        q = query.strip().lower()
        if not q:
            return "", ""
        try:
            events = self.schedule(use_cache=True)
        except Exception:  # noqa: BLE001 — schedule offline → fall back to current
            return "", ""
        for ev in events:
            if q == ev.event_id.lower() or q in ev.name.lower():
                return ev.start_date.replace("-", ""), f"scoreboard_ev_{ev.event_id}"
        return "", ""

    def current_event(self, event_id: str | None = None,
                      use_cache: bool = False) -> EspnEvent | None:
        payload = self.current_event_payload(event_id, use_cache=use_cache)
        events = self._events_for(payload, event_id)
        return _event_from_payload(events[0], tour=self.tour_slug) if events else None

    # Historical provider protocol -------------------------------------------------

    def _season_payload(self, year: int) -> dict | None:
        """Cached season schedule used strictly for event-id discovery."""
        self.history_cache_dir.mkdir(parents=True, exist_ok=True)
        cache = self.history_cache_dir / f"espn_{self.tour_slug}_{year}.json"
        if cache.exists() and year < dt.date.today().year:
            try:
                return json.loads(cache.read_text())
            except (ValueError, OSError):
                pass
        url = ESPN_SCOREBOARD_TMPL.format(tour=self.tour_slug)
        data = _http_json(url, {"dates": year})
        if data is not None and data.get("events"):
            try:
                cache.write_text(json.dumps(data))
            except OSError:
                pass
            return data
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except (ValueError, OSError):
                pass
        return None

    def _load_all(self) -> None:
        if self._meta:
            return
        from .legacy import TournamentMeta

        for year in self.seasons:
            payload = self._season_payload(year)
            if not payload:
                continue
            for event in payload.get("events", []) or []:
                event_id = str(event.get("id") or "")
                if not event_id:
                    continue
                name = str(event.get("name") or "")
                if self.tour_slug != "pga" and _is_major(name):
                    continue
                date = str(event.get("date") or "")[:10]
                self._events[event_id] = event
                self._meta[event_id] = TournamentMeta(
                    tournament_id=event_id,
                    name=name,
                    date=date,
                    tour=self.tour_slug,
                    is_major=_is_major(name),
                )

    def _event_payload(self, tournament_id: str) -> dict | None:
        """Authoritative event payload, cached permanently once final."""
        self.history_cache_dir.mkdir(parents=True, exist_ok=True)
        event_id = str(tournament_id)
        cache = self.history_cache_dir / (
            f"espn_event_{self.tour_slug}_{event_id}.json"
        )
        cached = None
        if cache.exists():
            try:
                cached = json.loads(cache.read_text())
            except (ValueError, OSError):
                pass
        cached_events = (cached or {}).get("events") or []
        cached_final = bool(
            cached_events
            and (
                ((cached_events[0].get("status") or {}).get("type") or {}).get(
                    "completed"
                )
            )
        )
        if cached_final:
            return cached
        data = _http_json(
            ESPN_LEADERBOARD,
            {"league": self.tour_slug, "event": event_id},
        )
        if data is not None and data.get("events"):
            try:
                cache.write_text(json.dumps(data))
            except OSError:
                pass
            return data
        return cached

    def recent_tournaments(self, since: Optional[str] = None) -> list:
        self._load_all()
        tournaments = [
            meta
            for meta in self._meta.values()
            if since is None or meta.date >= since
        ]
        return sorted(tournaments, key=lambda meta: meta.date)

    def rounds_for(self, tournament_id: str) -> list:
        from .legacy import RoundRecord, TournamentMeta

        self._load_all()
        schedule_meta = self._meta.get(str(tournament_id))
        payload = self._event_payload(str(tournament_id))
        events = (payload or {}).get("events") or []
        if not events or not schedule_meta:
            return []
        event = next(
            (
                row
                for row in events
                if str(row.get("id") or "") == str(tournament_id)
            ),
            events[0],
        )
        tournament = event.get("tournament") or {}
        scoring = str((tournament.get("scoringSystem") or {}).get("name") or "")
        if scoring and scoring.lower() != "medal":
            return []
        competition = _first_competition(event)
        status = str(
            ((event.get("status") or {}).get("type") or {}).get("name")
            or ((competition.get("status") or {}).get("type") or {}).get("name")
            or ""
        ).upper()
        if not any(token in status for token in ("FINAL", "COMPLETE", "POST")):
            return []
        course = _course(event, competition)
        course_name = str(course.get("name") or schedule_meta.name)
        total_rounds = _int(tournament.get("numberOfRounds"), 4)
        cut_round = _int(tournament.get("cutRound"), 0)
        no_cut = cut_round <= 0
        realized_cut_count = (
            0 if no_cut else _int(tournament.get("cutCount"), 0)
        )
        cut_rule = _pre_event_cut_rule(
            str(tournament.get("displayName") or event.get("name") or schedule_meta.name),
            schedule_meta.tour,
            competition,
            event,
        )
        multi_course = len([
            row for row in (event.get("courses") or []) if isinstance(row, dict)
        ]) > 1
        meta = TournamentMeta(
            tournament_id=schedule_meta.tournament_id,
            name=str(
                tournament.get("displayName")
                or event.get("name")
                or schedule_meta.name
            ),
            date=str(event.get("date") or schedule_meta.date)[:10],
            tour=schedule_meta.tour,
            is_major=bool(tournament.get("major", schedule_meta.is_major)),
            course=course_name,
            course_id=str(course.get("id") or ""),
            course_par=_int(course.get("shotsToPar"), 0),
            course_yards=_int(course.get("totalYards"), 0),
            cut_round=cut_round,
            cut_score=_safe_float(tournament.get("cutScore")),
            cut_rule=cut_rule,
            realized_cut_count=realized_cut_count,
            total_rounds=total_rounds,
            no_cut=no_cut,
            multi_course=multi_course,
        )
        competitors = competition.get("competitors", []) or []
        field_size = len(competitors)
        hole_features = _hole_features(
            self._events.get(str(tournament_id)) or {}
        )
        rows = []
        for competitor in competitors:
            athlete = competitor.get("athlete") or {}
            name = str(athlete.get("displayName") or "").strip()
            if not name:
                continue
            finish = _int(
                competitor.get("order")
                or (
                    ((competitor.get("status") or {}).get("position") or {}).get(
                        "id"
                    )
                ),
                999,
            )
            rounds = []
            for line in competitor.get("linescores") or []:
                round_no = _int(line.get("period"), 0)
                score_to_par = _to_par(line.get("displayValue"))
                tee_time = str(line.get("teeTime") or "")
                if (
                    round_no
                    and round_no <= total_rounds
                    and score_to_par is not None
                    and _round_complete(competitor, line)
                    and _valid_round_score(score_to_par)
                ):
                    rounds.append((round_no, score_to_par, tee_time))
            player_status = _status_name(competitor).upper()
            if no_cut:
                made_cut = 1
            elif "CUT" in player_status:
                made_cut = 0
            elif any(
                token in player_status
                for token in ("WD", "WITHDR", "DQ", "DISQUAL")
            ):
                made_cut = int(len(rounds) > cut_round)
            else:
                made_cut = 1
            for round_no, score_to_par, tee_time in rounds:
                rows.append(
                    RoundRecord(
                        tournament_id=meta.tournament_id,
                        event_name=meta.name,
                        date=(
                            tee_time[:10]
                            if tee_time
                            else _add_days(meta.date, round_no - 1)
                        ),
                        tour=meta.tour,
                        is_major=int(meta.is_major),
                        course=meta.course,
                        course_id=meta.course_id,
                        course_name=meta.course,
                        course_par=meta.course_par,
                        course_yards=meta.course_yards,
                        round=round_no,
                        tee_time=tee_time,
                        player=name,
                        dg_id=str(athlete.get("id") or ""),
                        score_to_par=float(score_to_par),
                        field_size=field_size,
                        made_cut=made_cut,
                        finish=finish,
                        cut_round=meta.cut_round,
                        cut_score=meta.cut_score,
                        cut_rule=meta.cut_rule,
                        realized_cut_count=meta.realized_cut_count,
                        total_rounds=meta.total_rounds,
                        no_cut=int(meta.no_cut),
                        multi_course=int(meta.multi_course),
                        **hole_features.get((name, round_no), {}),
                    )
                )
        # ESPN occasionally emits the same display-name competitor twice under
        # different ids. Event outcomes are display-name keyed, so identical
        # duplicates collapse here; contradictory duplicates are unsafe to guess.
        deduped = {}
        for row in rows:
            key = (row.player.casefold(), row.round)
            old = deduped.get(key)
            if old is not None and old.score_to_par != row.score_to_par:
                raise ValueError(
                    f"conflicting ESPN identities for {row.player!r}, "
                    f"event {row.tournament_id}, round {row.round}"
                )
            if old is None or (row.dg_id and row.dg_id < old.dg_id):
                deduped[key] = row
        return list(deduped.values())

    def field_for(self, event: Optional[str] = None) -> list:
        """Compatibility view of :meth:`field` for the history protocol."""
        from .legacy import FieldEntry

        return [
            FieldEntry(
                name=row.name,
                dg_id=row.source_player_id,
                world_rank=row.world_rank or 0,
                status=row.status,
            )
            for row in self.field(event_id=event, use_cache=False)
        ]

    def pretournament_preds(self, event: Optional[str] = None) -> None:
        return None

    def sg_categories(
        self, player: str, asof: Optional[str] = None
    ) -> None:
        return None

    @staticmethod
    def _events_for(payload: dict, event_id: str | None) -> list[dict]:
        """The single target event's payload entries.

        ESPN's scoreboard returns the whole week's board — several concurrent
        tournaments (e.g. The Open alongside the Barracuda Championship). All the
        per-competitor readers below must scope to one event or they merge fields
        from every event that week. Match the requested id; otherwise fall back to
        the featured (first) event, matching current_event's behaviour.
        """
        events = payload.get("events", []) or []
        if event_id:
            matched = [ev for ev in events if str(ev.get("id") or "") == str(event_id)]
            return matched
        return events[:1]

    def field(self, event_id: str | None = None,
              use_cache: bool = False) -> list[EspnFieldEntry]:
        payload = self.current_event_payload(event_id, use_cache=use_cache)
        out = []
        for ev in self._events_for(payload, event_id):
            for comp in _competitions(ev):
                for c in comp.get("competitors", []) or []:
                    athlete = c.get("athlete") or {}
                    name = (athlete.get("displayName") or athlete.get("fullName") or "").strip()
                    if not name:
                        continue
                    status = _status_name(c) or "active"
                    flag = athlete.get("flag") or {}
                    round_meta = _field_round_meta(c)
                    out.append(EspnFieldEntry(
                        name=name,
                        source_player_id=str(athlete.get("id") or c.get("id") or ""),
                        status=status,
                        country=str(flag.get("alt") or ""),
                        world_rank=_safe_int(c.get("rank")),
                        tee_time_r1=round_meta.get("tee_time_r1", ""),
                        tee_time_r2=round_meta.get("tee_time_r2", ""),
                        start_hole_r1=round_meta.get("start_hole_r1", ""),
                        start_hole_r2=round_meta.get("start_hole_r2", ""),
                    ))
        return out

    def leaderboard_rows(self, event_id: str | None = None,
                         use_cache: bool = False) -> list[dict]:
        payload = self.current_event_payload(event_id, use_cache=use_cache)
        rows = []
        for ev in self._events_for(payload, event_id):
            eid = str(ev.get("id") or event_id or "")
            for comp in _competitions(ev):
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
        for ev in self._events_for(payload, event_id):
            eid = str(ev.get("id") or event_id or "")
            for comp in _competitions(ev):
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
        self, event_id: str | None = None, use_cache: bool = False,
        cut_size: int = 65, no_cut: bool = False,
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

        Cut handling: ESPN's scoreboard feed does NOT flag the 36-hole cut — the
        competitor ``status`` is null even for players well outside the line. So
        once round 2 is complete we apply the standard PGA rule ourselves: keep
        the lowest ``cut_size`` 36-hole scores plus ties, mark the rest
        ``made_cut = 0``. From round 3 on, cut players simply have fewer completed
        rounds than the field and are excluded on that basis.
        """
        payload = self.current_event_payload(event_id, use_cache=use_cache)
        players: list[dict] = []
        for ev in self._events_for(payload, event_id):
            for comp in _competitions(ev):
                for c in comp.get("competitors", []) or []:
                    athlete = c.get("athlete") or {}
                    name = (athlete.get("displayName") or athlete.get("fullName") or "").strip()
                    if not name:
                        continue
                    cum = 0.0
                    completed = 0
                    for rline in c.get("linescores") or []:
                        if not _round_complete(c, rline):
                            continue  # round in progress — don't count it
                        score = _to_par(rline.get("displayValue"))
                        if score is None:
                            continue
                        completed += 1
                        cum += score
                    players.append({
                        "name": name,
                        "score": cum,
                        "completed": completed,
                        "_cut_flag": _is_out(c),
                        "_status": _status_name(c).upper(),
                    })

        active_started = [
            p for p in players if p["completed"] > 0 and not p["_cut_flag"]
        ]
        started = active_started or [p for p in players if p["completed"] > 0]
        rounds_done = 0
        if started:
            for r in (4, 3, 2, 1):
                if sum(1 for p in started if p["completed"] >= r) >= 0.5 * len(started):
                    rounds_done = r
                    break

        # After the cut round (R2), derive the cut line from the 36-hole scores,
        # since the feed won't tell us. Top `cut_size` and ties survive.
        cut_line = None
        explicit_cut = any("CUT" in p["_status"] for p in players)
        if rounds_done == 2 and not no_cut and not explicit_cut:
            r36 = sorted(p["score"] for p in players
                         if p["completed"] >= 2 and not p["_cut_flag"])
            if len(r36) > cut_size:
                cut_line = r36[cut_size - 1]

        rows = []
        for p in players:
            missed = (
                p["_cut_flag"]
                or (rounds_done and p["completed"] < rounds_done)
                or (cut_line is not None and p["completed"] >= 2
                    and p["score"] > cut_line)
            )
            made_cut = 0 if missed else 1
            rows.append({
                "name": p["name"],
                "score": int(p["score"]) if float(p["score"]).is_integer() else p["score"],
                "made_cut": made_cut,
                "completed": p["completed"],
            })
        return rows, rounds_done

    def qa_checks(self, field_rows: Iterable[EspnFieldEntry]) -> list[qa.SourceCheck]:
        rows = [
            r.as_store_row() if hasattr(r, "as_store_row") else dict(r)
            for r in field_rows
        ]
        return [
            qa.require_columns("espn.field", rows, ["name", "status", "source_player_id"]),
            qa.min_rows("espn.field", rows, 20),
            qa.field_size("espn.field", rows),
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


def _event_from_payload(ev: dict, tour: str = "pga") -> EspnEvent:
    comp = _first_competition(ev)
    status = ((ev.get("status") or {}).get("type") or {}).get("name") or \
        ((comp.get("status") or {}).get("type") or {}).get("name") or ""
    tournament = ev.get("tournament") or {}
    fmt = comp.get("format") or ev.get("format") or {}
    cut_round = _safe_int(tournament.get("cutRound"))
    if tournament:
        no_cut = cut_round == 0 or (
            cut_round is None and _explicit_no_cut(comp, ev, fmt)
        )
        realized_cut_count = (
            0 if no_cut else (_safe_int(tournament.get("cutCount")) or 0)
        )
        cut_rule = _pre_event_cut_rule(
            str(ev.get("name") or ev.get("shortName") or ""),
            tour,
            comp,
            ev,
        )
        total_rounds = _safe_int(tournament.get("numberOfRounds")) or 4
    else:
        realized_cut_count = 0
        no_cut = _explicit_no_cut(comp, ev, fmt)
        cut_rule = (
            _safe_int(comp.get("cutRule", ev.get("cutRule", fmt.get("cutRule"))))
            or 65
        )
        total_rounds = (
            _safe_int(comp.get(
                "numberOfRounds", ev.get("numberOfRounds", fmt.get("rounds"))
            ))
            or 4
        )
        cut_round = 0 if no_cut else 2
    course = _course(ev, comp)
    hole_counts = _course_hole_counts(course)
    return EspnEvent(
        event_id=str(ev.get("id") or ""),
        source_event_id=str(ev.get("id") or ""),
        name=str(ev.get("name") or ev.get("shortName") or ""),
        start_date=str(ev.get("date") or comp.get("date") or "")[:10],
        end_date=str(ev.get("endDate") or "")[:10],
        course_name=str(course.get("name") or ev.get("name") or ""),
        status=status,
        cut_rule=cut_rule,
        no_cut=no_cut,
        total_rounds=total_rounds,
        course_id=str(course.get("id") or ""),
        course_par=_safe_int(course.get("shotsToPar")) or 0,
        course_yards=_safe_int(course.get("totalYards")) or 0,
        par3_holes=hole_counts[3],
        par4_holes=hole_counts[4],
        par5_holes=hole_counts[5],
        cut_round=cut_round or 0,
        cut_score=_safe_float(tournament.get("cutScore")),
        realized_cut_count=realized_cut_count or 0,
        multi_course=len([
            row for row in (ev.get("courses") or []) if isinstance(row, dict)
        ]) > 1,
        tour=tour,
    )


def _course_name(ev: dict, comp: dict) -> str:
    return str(_course(ev, comp).get("name") or ev.get("name") or "")


def _first_competition(ev: dict) -> dict:
    return next(iter(_competitions(ev)), {})


def _competitions(ev: dict):
    for item in ev.get("competitions") or []:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, list):
            yield from (row for row in item if isinstance(row, dict))


def _course(ev: dict, comp: dict) -> dict:
    courses = [c for c in (ev.get("courses") or []) if isinstance(c, dict)]
    if courses:
        return next((c for c in courses if c.get("host")), courses[0])
    for src in (comp.get("course"), comp.get("venue")):
        if isinstance(src, dict):
            return src
    return {}


def _course_hole_counts(course: dict) -> dict[int, int]:
    counts = {3: 0, 4: 0, 5: 0}
    for hole in course.get("holes") or []:
        par = _safe_int(hole.get("shotsToPar"))
        if par in counts:
            counts[par] += 1
    return counts


def _status_name(comp: dict) -> str:
    return str(((comp.get("status") or {}).get("type") or {}).get("name") or "")


def _to_par(display) -> float | None:
    """ESPN round/aggregate to-par string → float; invalid values are absent."""
    s = str(display if display is not None else "").strip()
    if s in ("E", "e"):
        return 0.0
    if s in ("", "-", "--", "—"):
        return None
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None


def _valid_round_score(score: float) -> bool:
    """Reject impossible display-value artifacts before they reach history."""
    return -15.0 <= float(score) <= 30.0


def _explicit_no_cut(comp: dict, ev: dict, fmt: dict) -> bool:
    return bool(
        comp.get("noCut", ev.get("noCut", False))
        or fmt.get("noCut", False)
        or str(fmt.get("name") or fmt.get("description") or "").strip().lower()
        == "no cut"
    )


def _pre_event_cut_rule(name: str, tour: str, comp: dict, ev: dict) -> int:
    """Return a rule known before play; never derive it from cutCount.

    ESPN's tournament.cutCount is the realized number of players surviving ties.
    Explicit format cutRule is accepted when plausible, otherwise current men's
    major and tour rules are used for the 2022+ history this project fits.
    """
    fmt = comp.get("format") or ev.get("format") or {}
    explicit = _safe_int(
        comp.get("cutRule", ev.get("cutRule", fmt.get("cutRule")))
    )
    if explicit is not None and 1 <= explicit <= 70:
        return explicit
    folded = " ".join(str(name or "").lower().split())
    if folded == "masters tournament":
        return 50
    if folded in {"u.s. open", "us open"}:
        return 60
    if folded in {
        "pga championship", "the open championship", "the open"
    }:
        return 70
    return 65


def _is_out(competitor: dict) -> bool:
    """True when a competitor has left the tournament (cut, WD or DQ)."""
    status = _status_name(competitor).upper()
    return any(k in status for k in ("CUT", "WD", "WITHDR", "DQ", "DISQUAL"))


def _round_complete(competitor: dict, round_line: dict) -> bool:
    """Recognise complete rounds in both old and per-event payload shapes."""
    holes = round_line.get("linescores") or []
    if holes:
        return len(holes) >= 18
    if round_line.get("value") in ("", None):
        return False
    rnd = _safe_int(round_line.get("period")) or 0
    status = competitor.get("status") or {}
    current = _safe_int(status.get("period")) or 0
    name = _status_name(competitor).upper()
    display = str(status.get("displayValue") or "").upper()
    detail = str(status.get("detail") or "").upper()
    return (
        rnd < current
        or any(k in name for k in ("FINISH", "CUT"))
        or display in {"F", "CUT"}
        or detail.endswith("(F)")
    )


def _field_round_meta(comp: dict) -> dict[str, str]:
    """Best-effort tee metadata from ESPN linescore statistics.

    ESPN has changed this shape more than once. We scan display values rather
    than depending on stat names so missing/changed metadata simply degrades to
    blanks in field.csv.
    """
    out: dict[str, str] = {}
    for round_line in comp.get("linescores", []) or []:
        rnd = _safe_int(round_line.get("period"))
        if rnd not in (1, 2):
            continue
        tee_time = str(round_line.get("teeTime") or "").strip()
        if tee_time:
            out[f"tee_time_r{rnd}"] = tee_time
        vals: list[str] = []
        for cat in (round_line.get("statistics") or {}).get("categories", []) or []:
            for stat in cat.get("stats", []) or []:
                display = str(stat.get("displayValue") or "").strip()
                if display:
                    vals.append(display)
        for val in vals:
            low = val.lower()
            if ("am" in low or "pm" in low or "gmt" in low or "utc" in low
                    or _looks_like_iso_time(val)):
                out.setdefault(f"tee_time_r{rnd}", val)
                break
        for val in vals:
            if val in {"1", "10"}:
                out[f"start_hole_r{rnd}"] = val
                break
    return out


def _looks_like_iso_time(value: str) -> bool:
    return len(value) >= 16 and value[4:5] == "-" and "T" in value


def _safe_int(value) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _http_json(url: str, params: dict | None = None, retries: int = 3) -> dict | None:
    """GET JSON with retry, returning ``None`` on persistent failure."""
    query = urllib.parse.urlencode(
        {key: value for key, value in (params or {}).items() if value is not None}
    )
    full_url = f"{url}?{query}" if query else url
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                full_url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.load(response)
        except Exception:  # noqa: BLE001 - offline-safe provider boundary
            if attempt + 1 < retries:
                time.sleep(1.5)
    return None


def _is_major(name: str) -> bool:
    return " ".join(str(name).lower().split()) in _MAJOR_NAMES


def _add_days(iso_date: str, days: int) -> str:
    try:
        date = dt.date.fromisoformat(iso_date[:10])
        return (date + dt.timedelta(days=days)).isoformat()
    except ValueError:
        return iso_date[:10]


def _int(value, default: int = 0) -> int:
    parsed = _safe_int(value)
    return default if parsed is None else parsed


def _hole_features(event: dict) -> dict[tuple[str, int], dict]:
    """Round-level score-shape features from season payload scorecards."""
    out: dict[tuple[str, int], dict] = {}
    competition = _first_competition(event)
    for competitor in competition.get("competitors") or []:
        athlete = competitor.get("athlete") or {}
        name = str(athlete.get("displayName") or "").strip()
        for round_line in competitor.get("linescores") or []:
            round_no = _int(round_line.get("period"), 0)
            holes = round_line.get("linescores") or []
            if not round_no or len(holes) < 18:
                continue
            features = {
                "holes_scored": 0,
                "birdies_or_better": 0,
                "bogeys": 0,
                "double_bogeys_or_worse": 0,
                "par3_to_par": 0.0,
                "par3_holes": 0,
                "par4_to_par": 0.0,
                "par4_holes": 0,
                "par5_to_par": 0.0,
                "par5_holes": 0,
            }
            for hole in holes:
                gross = _safe_float(hole.get("value"))
                relative = _to_par(
                    (hole.get("scoreType") or {}).get("displayValue")
                )
                if gross is None or relative is None:
                    continue
                par = int(round(gross - relative))
                features["holes_scored"] += 1
                features["birdies_or_better"] += int(relative <= -1)
                features["bogeys"] += int(relative == 1)
                features["double_bogeys_or_worse"] += int(relative >= 2)
                if par in (3, 4, 5):
                    features[f"par{par}_to_par"] += float(relative)
                    features[f"par{par}_holes"] += 1
            if features["holes_scored"] >= 18:
                out[(name, round_no)] = features
    return out


# Short compatibility name. Both names resolve to the same class object.
EspnProvider = EspnGolfProvider
