"""SQLite store for ephemeral/live golf provider state.

``data/rounds.csv`` is the sole authoritative historical round store and is
never mirrored into SQLite. This database is a rebuildable cache for the
current event, field, public-stat snapshots, odds, and provider-run metadata.
CSV exports remain the compatibility boundary for the engine and app adapter.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Iterable, Mapping

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "golf.db"
FIELD_CSV = DATA_DIR / "field.csv"

# ── database location resolution ─────────────────────────────────────────────
# golf.db normally lives next to the CSV contract in golf/data. Some filesystems
# (network shares, fuse mounts like the desktop-app sandbox) cannot service the
# fsync/locking SQLite needs and raise "disk I/O error". When that happens we
# transparently relocate the cache to local scratch so the engine still runs —
# the CSVs in golf/data remain the source of truth the model consumes, so a
# fresh local cache loses nothing (refresh re-imports them every run). Set
# GOLF_DB_PATH to force a specific location.
_ACTIVE_DB_PATH: Path | None = None


def _sqlite_usable(path: Path) -> bool:
    """True if SQLite can actually create/write/commit at this location.

    Probes the real DB file (creating/dropping a throwaway table) rather than a
    sidecar, because some mounts that fail SQLite I/O also block file deletion —
    a separate probe file would become undeletable litter.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
        try:
            con.execute("CREATE TABLE IF NOT EXISTS _ioprobe(x)")
            con.execute("INSERT INTO _ioprobe(x) VALUES (1)")
            con.commit()
            con.execute("DROP TABLE _ioprobe")
            con.commit()
        finally:
            con.close()
        return True
    except sqlite3.Error:
        return False


def active_db_path() -> Path:
    """Resolve the writable DB path once per process.

    Order: GOLF_DB_PATH env override → canonical golf/data/golf.db if SQLite is
    usable there → local scratch fallback (tempdir/golf_engine/golf.db).
    """
    global _ACTIVE_DB_PATH
    if _ACTIVE_DB_PATH is not None:
        return _ACTIVE_DB_PATH

    env = os.environ.get("GOLF_DB_PATH")
    if env:
        _ACTIVE_DB_PATH = Path(env)
        _ACTIVE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        return _ACTIVE_DB_PATH

    # Probe a sidecar so we never touch the real golf.db while testing the mount.
    probe = DB_PATH.parent / ".sqlite_ioprobe.db"
    usable = _sqlite_usable(probe)
    for leftover in (probe, probe.with_name(probe.name + "-journal")):
        try:
            leftover.unlink(missing_ok=True)
        except OSError:
            pass

    if usable:
        _ACTIVE_DB_PATH = DB_PATH
    else:
        _ACTIVE_DB_PATH = Path(tempfile.gettempdir()) / "golf_engine" / "golf.db"
        _ACTIVE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _ACTIVE_DB_PATH
MANIFEST_JSON = DATA_DIR / "free_source_manifest.json"


SCHEMA = """
PRAGMA foreign_keys = ON;

-- One-time cleanup for databases created before rounds.csv became the sole
-- historical store. This table was only ever a rebuildable CSV mirror.
DROP TABLE IF EXISTS rounds;

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT '',
    source_event_id TEXT NOT NULL DEFAULT '',
    tour TEXT NOT NULL DEFAULT 'pga',
    season INTEGER,
    name TEXT NOT NULL,
    start_date TEXT NOT NULL DEFAULT '',
    end_date TEXT NOT NULL DEFAULT '',
    course_id TEXT NOT NULL DEFAULT '',
    course_name TEXT NOT NULL DEFAULT '',
    course_par INTEGER NOT NULL DEFAULT 0,
    course_yards INTEGER NOT NULL DEFAULT 0,
    par3_holes INTEGER NOT NULL DEFAULT 0,
    par4_holes INTEGER NOT NULL DEFAULT 0,
    par5_holes INTEGER NOT NULL DEFAULT 0,
    city TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT '',
    country TEXT NOT NULL DEFAULT '',
    latitude REAL,
    longitude REAL,
    timezone TEXT NOT NULL DEFAULT '',
    is_major INTEGER NOT NULL DEFAULT 0,
    cut_rule INTEGER NOT NULL DEFAULT 65,
    no_cut INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT '',
    source_player_id TEXT NOT NULL DEFAULT '',
    canonical_name TEXT NOT NULL,
    display_name TEXT NOT NULL,
    country TEXT NOT NULL DEFAULT '',
    amateur INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS field_entries (
    event_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    entry_source TEXT NOT NULL DEFAULT '',
    tee_time_r1 TEXT NOT NULL DEFAULT '',
    tee_time_r2 TEXT NOT NULL DEFAULT '',
    start_hole_r1 TEXT NOT NULL DEFAULT '',
    start_hole_r2 TEXT NOT NULL DEFAULT '',
    world_rank INTEGER,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (event_id, player_id)
);

CREATE TABLE IF NOT EXISTS stat_snapshots (
    season INTEGER NOT NULL,
    stat_id TEXT NOT NULL,
    stat_name TEXT NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    rank INTEGER,
    value REAL,
    raw_json TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'pgatour',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (season, stat_id, player_id)
);

CREATE TABLE IF NOT EXISTS odds_quotes (
    quote_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL DEFAULT '',
    market TEXT NOT NULL,
    round_no INTEGER,
    group_id TEXT NOT NULL DEFAULT '',
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    book TEXT NOT NULL DEFAULT '',
    decimal_odds REAL NOT NULL,
    timestamp TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    settlement_rule TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS provider_runs (
    run_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    action TEXT NOT NULL,
    ok INTEGER NOT NULL,
    rows INTEGER NOT NULL DEFAULT 0,
    warnings TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    p = Path(path) if path is not None else active_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    return con


def init_db(path: Path | str | None = None) -> Path:
    p = Path(path) if path is not None else active_db_path()
    should_vacuum = False
    with connect(p) as con:
        con.executescript(SCHEMA)
        # Existing live-cache databases predate the current course context.
        # SQLite has no portable ADD COLUMN IF NOT EXISTS, so migrate explicitly.
        for name, definition in {
            "course_id": "TEXT NOT NULL DEFAULT ''",
            "course_par": "INTEGER NOT NULL DEFAULT 0",
            "course_yards": "INTEGER NOT NULL DEFAULT 0",
            "par3_holes": "INTEGER NOT NULL DEFAULT 0",
            "par4_holes": "INTEGER NOT NULL DEFAULT 0",
            "par5_holes": "INTEGER NOT NULL DEFAULT 0",
        }.items():
            _ensure_column(con, "events", name, definition)
        con.execute(
            "DELETE FROM odds_quotes "
            "WHERE datetime(timestamp) < datetime('now', '-180 days')"
        )
        pages = int(con.execute("PRAGMA page_count").fetchone()[0])
        free = int(con.execute("PRAGMA freelist_count").fetchone()[0])
        should_vacuum = pages > 100 and free / pages > 0.25
    if should_vacuum:
        with connect(p) as con:
            con.execute("VACUUM")
    return Path(p)


def _ensure_column(
    con: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    existing = {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _pid(name: str, source_id: str = "") -> str:
    """Stable player identity. Provider IDs prevent same-name collisions."""
    if str(source_id or "").strip():
        return "source:" + str(source_id).strip()
    return "name:" + " ".join(str(name or "").casefold().split())


def upsert_players(con: sqlite3.Connection, players: Iterable[Mapping]) -> int:
    rows = list(players)
    for r in rows:
        name = str(r.get("display_name") or r.get("name") or r.get("player") or "").strip()
        if not name:
            continue
        source_id = str(r.get("source_player_id") or r.get("player_id") or r.get("espn_id") or "").strip()
        player_id = str(r.get("player_id") or _pid(name, source_id))
        con.execute(
            """
            INSERT INTO players(player_id, source, source_player_id, canonical_name,
                                display_name, country, amateur, updated_at)
            VALUES(?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(player_id) DO UPDATE SET
                source=excluded.source,
                source_player_id=excluded.source_player_id,
                canonical_name=excluded.canonical_name,
                display_name=excluded.display_name,
                country=excluded.country,
                amateur=excluded.amateur,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                player_id,
                str(r.get("source") or ""),
                source_id,
                str(r.get("canonical_name") or name),
                name,
                str(r.get("country") or ""),
                int(bool(r.get("amateur", False))),
            ),
        )
    return len(rows)


def upsert_events(con: sqlite3.Connection, events: Iterable[Mapping]) -> int:
    rows = list(events)
    for r in rows:
        event_id = str(r.get("event_id") or r.get("tournament_id") or r.get("id") or "").strip()
        name = str(r.get("name") or r.get("event") or r.get("tournament_name") or "").strip()
        if not event_id or not name:
            continue
        con.execute(
            """
            INSERT INTO events(event_id, source, source_event_id, tour, season, name,
                               start_date, end_date, course_id, course_name,
                               course_par, course_yards, par3_holes, par4_holes,
                               par5_holes, city, state, country, latitude,
                               longitude, timezone, is_major, cut_rule, no_cut,
                               status, updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(event_id) DO UPDATE SET
                source=excluded.source,
                source_event_id=excluded.source_event_id,
                tour=excluded.tour,
                season=excluded.season,
                name=excluded.name,
                start_date=excluded.start_date,
                end_date=excluded.end_date,
                course_id=excluded.course_id,
                course_name=excluded.course_name,
                course_par=excluded.course_par,
                course_yards=excluded.course_yards,
                par3_holes=excluded.par3_holes,
                par4_holes=excluded.par4_holes,
                par5_holes=excluded.par5_holes,
                city=excluded.city,
                state=excluded.state,
                country=excluded.country,
                latitude=excluded.latitude,
                longitude=excluded.longitude,
                timezone=excluded.timezone,
                is_major=excluded.is_major,
                cut_rule=excluded.cut_rule,
                no_cut=excluded.no_cut,
                status=excluded.status,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                event_id,
                str(r.get("source") or ""),
                str(r.get("source_event_id") or event_id),
                str(r.get("tour") or "pga"),
                _int_or_none(r.get("season")),
                name,
                str(r.get("start_date") or r.get("date") or "")[:10],
                str(r.get("end_date") or "")[:10],
                str(r.get("course_id") or ""),
                str(r.get("course_name") or r.get("course") or ""),
                _int_or_none(r.get("course_par")) or 0,
                _int_or_none(r.get("course_yards")) or 0,
                _int_or_none(r.get("par3_holes")) or 0,
                _int_or_none(r.get("par4_holes")) or 0,
                _int_or_none(r.get("par5_holes")) or 0,
                str(r.get("city") or ""),
                str(r.get("state") or ""),
                str(r.get("country") or ""),
                _float_or_none(r.get("latitude")),
                _float_or_none(r.get("longitude")),
                str(r.get("timezone") or ""),
                int(bool(r.get("is_major", False))),
                int(r.get("cut_rule") or 65),
                int(bool(r.get("no_cut", False))),
                str(r.get("status") or ""),
            ),
        )
    return len(rows)


def upsert_field(con: sqlite3.Connection, event_id: str, rows: Iterable[Mapping]) -> int:
    rows = list(rows)
    upsert_players(con, rows)
    seen_ids: set[str] = set()
    for r in rows:
        name = str(r.get("display_name") or r.get("name") or r.get("player") or "").strip()
        if not name:
            continue
        source_id = str(r.get("source_player_id") or r.get("player_id") or r.get("espn_id") or "").strip()
        player_id = str(r.get("player_id") or _pid(name, source_id))
        seen_ids.add(player_id)
        con.execute(
            """
            INSERT INTO field_entries(event_id, player_id, status, entry_source,
                                      tee_time_r1, tee_time_r2, start_hole_r1,
                                      start_hole_r2, world_rank, updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(event_id, player_id) DO UPDATE SET
                status=excluded.status,
                entry_source=excluded.entry_source,
                tee_time_r1=excluded.tee_time_r1,
                tee_time_r2=excluded.tee_time_r2,
                start_hole_r1=excluded.start_hole_r1,
                start_hole_r2=excluded.start_hole_r2,
                world_rank=excluded.world_rank,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                event_id,
                player_id,
                str(r.get("status") or "active"),
                str(r.get("source") or r.get("entry_source") or ""),
                str(r.get("tee_time_r1") or ""),
                str(r.get("tee_time_r2") or ""),
                str(r.get("start_hole_r1") or ""),
                str(r.get("start_hole_r2") or ""),
                _int_or_none(r.get("world_rank")),
            ),
        )
    # Make the field authoritative: drop entries no longer in the fetched field
    # (withdrawals, and — importantly — players that a previous buggy merge of a
    # concurrent event left behind). Only prune when we actually have a field, so
    # a partial/failed fetch never wipes it.
    if seen_ids:
        placeholders = ",".join("?" for _ in seen_ids)
        con.execute(
            f"DELETE FROM field_entries WHERE event_id = ? "
            f"AND player_id NOT IN ({placeholders})",
            (event_id, *seen_ids),
        )
    return len(rows)


def export_field_csv(event_id: str, path: Path | str = FIELD_CSV,
                     db_path: Path | str | None = None) -> Path:
    db_path = active_db_path() if db_path is None else db_path
    init_db(db_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as con:
        rows = con.execute(
            """
            SELECT p.display_name AS name, p.source_player_id AS dg_id,
                   f.world_rank, f.status,
                   e.event_id, e.name AS event, e.course_id,
                   e.course_name AS course, e.course_par, e.course_yards,
                   e.par3_holes, e.par4_holes, e.par5_holes,
                   e.cut_rule, e.no_cut,
                   f.tee_time_r1, f.tee_time_r2,
                   f.start_hole_r1, f.start_hole_r2,
                   '' AS course_sigma,
                   '' AS odds_win, '' AS odds_top5, '' AS odds_top10,
                   '' AS odds_top20, '' AS odds_cut
            FROM field_entries f
            JOIN players p ON p.player_id = f.player_id
            LEFT JOIN events e ON e.event_id = f.event_id
            WHERE f.event_id = ?
            ORDER BY p.display_name
            """,
            (event_id,),
        ).fetchall()
    cols = [
        "name", "dg_id", "world_rank", "status", "event_id", "event",
        "course_id", "course", "course_par", "course_yards",
        "par3_holes", "par4_holes", "par5_holes",
        "cut_rule", "no_cut",
        "tee_time_r1", "tee_time_r2", "start_hole_r1", "start_hole_r2",
        "course_sigma",
        "odds_win", "odds_top5", "odds_top10", "odds_top20", "odds_cut",
    ]
    from .io_utils import atomic_write_csv
    atomic_write_csv(path, cols, [dict(row) for row in rows])
    return path


def record_provider_run(provider: str, action: str, ok: bool, rows: int = 0,
                        warnings: list | None = None,
                        db_path: Path | str | None = None) -> None:
    db_path = active_db_path() if db_path is None else db_path
    init_db(db_path)
    run_id = f"{provider}:{action}:{int(time.time() * 1000)}"
    with connect(db_path) as con:
        con.execute(
            """
            INSERT INTO provider_runs(run_id, provider, action, ok, rows, warnings)
            VALUES(?,?,?,?,?,?)
            """,
            (run_id, provider, action, int(ok), int(rows), json.dumps(warnings or [])),
        )


def upsert_odds_quotes(con: sqlite3.Connection, quotes: Iterable[Mapping]) -> int:
    rows = list(quotes)
    for r in rows:
        name = str(r.get("player_name") or r.get("name") or r.get("player") or "").strip()
        market = str(r.get("market") or "").strip()
        try:
            odds = float(r.get("decimal_odds") or r.get("odds"))
        except (TypeError, ValueError):
            continue
        if not name or not market or odds <= 1:
            continue
        player_id = str(r.get("player_id") or _pid(name, str(r.get("source_player_id") or "")))
        quote_id = str(r.get("quote_id") or "|".join([
            str(r.get("event_id") or ""),
            market,
            str(r.get("round_no") or ""),
            str(r.get("group_id") or ""),
            player_id,
            str(r.get("book") or ""),
            str(r.get("timestamp") or ""),
            f"{odds:.4f}",
        ]))
        con.execute(
            """
            INSERT OR REPLACE INTO odds_quotes(
                quote_id, event_id, market, round_no, group_id, player_id,
                player_name, book, decimal_odds, timestamp, source,
                settlement_rule, updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            """,
            (
                quote_id,
                str(r.get("event_id") or ""),
                market,
                _int_or_none(r.get("round_no")),
                str(r.get("group_id") or ""),
                player_id,
                name,
                str(r.get("book") or ""),
                odds,
                str(r.get("timestamp") or ""),
                str(r.get("source") or ""),
                str(r.get("settlement_rule") or ""),
            ),
        )
    return len(rows)


def upsert_stat_rows(con: sqlite3.Connection, rows: Iterable[Mapping]) -> int:
    rows = list(rows)
    for r in rows:
        name = str(r.get("player_name") or "").strip()
        stat_id = str(r.get("stat_id") or "").strip()
        if not name or not stat_id:
            continue
        player_id = str(r.get("player_id") or _pid(name))
        raw_json = r.get("raw_json")
        if not isinstance(raw_json, str):
            raw_json = json.dumps(raw_json or {})
        con.execute(
            """
            INSERT INTO stat_snapshots(season, stat_id, stat_name, player_id,
                                       player_name, rank, value, raw_json,
                                       source, updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(season, stat_id, player_id) DO UPDATE SET
                stat_name=excluded.stat_name,
                player_name=excluded.player_name,
                rank=excluded.rank,
                value=excluded.value,
                raw_json=excluded.raw_json,
                source=excluded.source,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                _int_or_none(r.get("season")) or 0,
                stat_id,
                str(r.get("stat_name") or ""),
                player_id,
                name,
                _int_or_none(r.get("rank")),
                _float_or_none(r.get("value")),
                raw_json,
                str(r.get("source") or "pgatour"),
            ),
        )
    return len(rows)


def write_manifest(payload: Mapping, path: Path | str = MANIFEST_JSON) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **dict(payload)}
    from .io_utils import atomic_write_text
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")
    return path


def _float_or_none(value) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
