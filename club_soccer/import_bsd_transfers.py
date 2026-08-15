#!/usr/bin/env python3
"""Resumable importer for Bzzoiro's global football transfer feed.

The initial backfill is built outside ``club_soccer/data`` so Syncthing never
sees a partial database.  Once complete and verified, it is atomically moved
to ``data/transfers_bsd.sqlite3``.  Incremental refreshes use the same pattern:
copy the live database to a private work directory, update and verify the copy,
then atomically replace the live file.

Mini commands, from the project root::

    python3 -m club_soccer.import_bsd_transfers --backfill
    python3 -m club_soccer.import_bsd_transfers --update
    python3 -m club_soccer.import_bsd_transfers --status
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_keys import get_key
from bsd_client import _get

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DEFAULT_DB = DATA / "transfers_bsd.sqlite3"
DEFAULT_WORK_DIR = HERE / ".transfer_import"

PAGE_SIZE = 100  # endpoint maximum
DEFAULT_OVERLAP_DAYS = 7
SOURCE = "bzzoiro_v2"

FetchPage = Callable[..., dict]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transfers (
    id               INTEGER PRIMARY KEY,
    player_id        INTEGER,
    player_name      TEXT NOT NULL,
    transfer_date    TEXT,
    from_team_id     INTEGER,
    from_team_name   TEXT,
    to_team_id       INTEGER,
    to_team_name     TEXT,
    transfer_type    INTEGER,
    fee_eur          INTEGER,
    fee_description  TEXT,
    source           TEXT NOT NULL,
    retrieved_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transfers_player_date
    ON transfers(player_id, transfer_date);
CREATE INDEX IF NOT EXISTS idx_transfers_date
    ON transfers(transfer_date);
CREATE INDEX IF NOT EXISTS idx_transfers_from_team
    ON transfers(from_team_id, transfer_date);
CREATE INDEX IF NOT EXISTS idx_transfers_to_team
    ON transfers(to_team_id, transfer_date);
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_UPSERT = """
INSERT INTO transfers (
    id, player_id, player_name, transfer_date,
    from_team_id, from_team_name, to_team_id, to_team_name,
    transfer_type, fee_eur, fee_description, source, retrieved_at_utc
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    player_id=excluded.player_id,
    player_name=excluded.player_name,
    transfer_date=excluded.transfer_date,
    from_team_id=excluded.from_team_id,
    from_team_name=excluded.from_team_name,
    to_team_id=excluded.to_team_id,
    to_team_name=excluded.to_team_name,
    transfer_type=excluded.transfer_type,
    fee_eur=excluded.fee_eur,
    fee_description=excluded.fee_description,
    source=excluded.source,
    retrieved_at_utc=excluded.retrieved_at_utc
"""


def get_transfers_page(api_key: str, **params) -> dict:
    """Fetch one page of BSD's cross-player v2 transfer feed."""
    return _get("/api/v2/transfers/", api_key, **params)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    return conn


def _meta_get(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return str(row[0]) if row else default


def _meta_set(conn: sqlite3.Connection, key: str, value: object) -> None:
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def _as_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise(row: dict, retrieved_at: str) -> tuple | None:
    transfer_id = _as_int(row.get("id"))
    player = row.get("player") if isinstance(row.get("player"), dict) else {}
    player_name = str(
        player.get("name") or row.get("player_name") or ""
    ).strip()
    if transfer_id is None or not player_name:
        return None
    return (
        transfer_id,
        _as_int(player.get("id") or row.get("player_id")),
        player_name,
        str(row.get("transfer_date") or "") or None,
        _as_int(row.get("from_team_id")),
        str(row.get("from_team_name") or "") or None,
        _as_int(row.get("to_team_id")),
        str(row.get("to_team_name") or "") or None,
        _as_int(row.get("transfer_type")),
        _as_int(row.get("fee_eur")),
        str(row.get("fee_description") or "") or None,
        SOURCE,
        retrieved_at,
    )


def _upsert_page(
    conn: sqlite3.Connection,
    rows: list[dict],
    retrieved_at: str,
    *,
    seen_table: bool = False,
) -> tuple[int, int]:
    valid: list[tuple] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalised = _normalise(row, retrieved_at)
        if normalised is not None:
            valid.append(normalised)
    conn.executemany(_UPSERT, valid)
    if seen_table:
        conn.executemany(
            "INSERT OR IGNORE INTO import_seen(id) VALUES (?)",
            ((row[0],) for row in valid),
        )
    return len(valid), len(rows) - len(valid)


def _fetch_with_retry(
    fetch_page: FetchPage,
    api_key: str,
    *,
    retries: int,
    retry_delay: float,
    **params,
) -> dict:
    for attempt in range(retries + 1):
        try:
            page = fetch_page(api_key, **params)
            if not isinstance(page, dict) or not isinstance(page.get("results"), list):
                raise RuntimeError("BSD transfer response is not a paginated result")
            return page
        except Exception:
            if attempt >= retries:
                raise
            wait = min(retry_delay * (2 ** attempt), 30.0)
            if wait:
                time.sleep(wait)
    raise AssertionError("unreachable")


def _verify(conn: sqlite3.Connection) -> None:
    result = conn.execute("PRAGMA integrity_check").fetchone()
    if not result or result[0] != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {result!r}")
    invalid = conn.execute(
        "SELECT COUNT(*) FROM transfers "
        "WHERE id IS NULL OR player_name='' OR source!=?",
        (SOURCE,),
    ).fetchone()[0]
    if invalid:
        raise RuntimeError(f"transfer database contains {invalid} invalid row(s)")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _publish(staging: Path, live: Path) -> None:
    """Atomically publish a closed, verified SQLite file."""
    live.parent.mkdir(parents=True, exist_ok=True)
    _fsync_file(staging)
    os.replace(staging, live)
    directory_fd = os.open(live.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _backup_database(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)


@contextlib.contextmanager
def _exclusive_lock(work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / "import.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another BSD transfer import is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _progress(offset: int, total: int | None, started: float) -> str:
    elapsed = max(time.monotonic() - started, 0.001)
    rate = offset / elapsed
    if total:
        pct = min(offset / total, 1.0)
        remaining = max(total - offset, 0)
        eta = remaining / rate if rate else 0
        return (
            f"{offset:,}/{total:,} ({pct:.1%}), "
            f"{rate:.1f} rows/s, ETA {eta / 60:.1f} min"
        )
    return f"{offset:,} rows, {rate:.1f} rows/s"


def backfill(
    api_key: str,
    *,
    db_path: Path = DEFAULT_DB,
    work_dir: Path = DEFAULT_WORK_DIR,
    page_size: int = PAGE_SIZE,
    retries: int = 5,
    retry_delay: float = 2.0,
    request_delay: float = 0.0,
    progress_every: int = 25,
    replace: bool = False,
    today: date | None = None,
    fetch_page: FetchPage | None = None,
) -> dict:
    """Build full history, resuming the private staging DB if interrupted."""
    if not 1 <= page_size <= PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {PAGE_SIZE}")
    fetch_page = fetch_page or get_transfers_page
    cutoff = (today or date.today()).isoformat()
    staging = work_dir / "transfers_bsd.backfill.sqlite3"

    with _exclusive_lock(work_dir):
        if db_path.exists() and not replace:
            raise FileExistsError(
                f"{db_path} already exists; use --update, or --backfill --replace"
            )
        conn = _connect(staging)
        try:
            mode = _meta_get(conn, "import_mode")
            if mode and mode != "backfill":
                raise RuntimeError(f"staging database has unexpected mode {mode!r}")
            stored_cutoff = _meta_get(conn, "backfill_date_to")
            if stored_cutoff:
                cutoff = stored_cutoff
            else:
                _meta_set(conn, "import_mode", "backfill")
                _meta_set(conn, "backfill_date_to", cutoff)
                _meta_set(conn, "backfill_offset", 0)
                conn.commit()

            offset = int(_meta_get(conn, "backfill_offset", "0"))
            invalid_total = int(_meta_get(conn, "invalid_rows", "0"))
            total: int | None = None
            started = time.monotonic()
            page_number = offset // page_size

            while True:
                page = _fetch_with_retry(
                    fetch_page,
                    api_key,
                    retries=retries,
                    retry_delay=retry_delay,
                    limit=page_size,
                    offset=offset,
                    ordering="transfer_date",
                    date_to=cutoff,
                )
                if isinstance(page.get("count"), int):
                    total = page["count"]
                batch = page["results"]
                if not batch:
                    if page.get("next"):
                        raise RuntimeError("BSD returned an empty page with a next link")
                    break

                retrieved_at = datetime.now(timezone.utc).isoformat()
                _valid, invalid = _upsert_page(conn, batch, retrieved_at)
                invalid_total += invalid
                offset += len(batch)
                page_number += 1
                _meta_set(conn, "backfill_offset", offset)
                _meta_set(conn, "invalid_rows", invalid_total)
                conn.commit()

                if progress_every and page_number % progress_every == 0:
                    print("  " + _progress(offset, total, started), flush=True)
                if not page.get("next"):
                    break
                if request_delay:
                    time.sleep(request_delay)

            _meta_set(conn, "backfill_complete", "1")
            _meta_set(conn, "last_successful_sync_date", cutoff)
            _meta_set(conn, "last_successful_sync_utc",
                      datetime.now(timezone.utc).isoformat())
            _meta_set(conn, "backfill_offset", offset)
            conn.commit()
            _verify(conn)
            count = conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
        except BaseException:
            conn.close()
            raise
        else:
            conn.close()
        _publish(staging, db_path)

    return {
        "rows": count,
        "invalid_rows": invalid_total,
        "date_to": cutoff,
        "database": str(db_path),
    }


def update(
    api_key: str,
    *,
    db_path: Path = DEFAULT_DB,
    work_dir: Path = DEFAULT_WORK_DIR,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
    page_size: int = PAGE_SIZE,
    retries: int = 5,
    retry_delay: float = 2.0,
    request_delay: float = 0.0,
    progress_every: int = 25,
    today: date | None = None,
    fetch_page: FetchPage | None = None,
) -> dict:
    """Refresh from the last success with overlap, then atomically publish."""
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} does not exist; run --backfill first")
    if overlap_days < 0:
        raise ValueError("overlap_days must be non-negative")
    if not 1 <= page_size <= PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {PAGE_SIZE}")
    fetch_page = fetch_page or get_transfers_page
    through = today or date.today()
    staging = work_dir / "transfers_bsd.update.sqlite3"

    with _exclusive_lock(work_dir):
        _backup_database(db_path, staging)
        conn = _connect(staging)
        try:
            last_sync_raw = _meta_get(conn, "last_successful_sync_date")
            if last_sync_raw:
                last_sync = date.fromisoformat(last_sync_raw)
            else:
                raw = conn.execute(
                    "SELECT MAX(transfer_date) FROM transfers"
                ).fetchone()[0]
                last_sync = date.fromisoformat(raw) if raw else through
            start = min(last_sync, through) - timedelta(days=overlap_days)
            date_from = start.isoformat()
            date_to = through.isoformat()

            conn.execute("DROP TABLE IF EXISTS import_seen")
            conn.execute("CREATE TEMP TABLE import_seen (id INTEGER PRIMARY KEY)")
            offset = invalid_total = 0
            total: int | None = None
            page_number = 0
            started = time.monotonic()

            while True:
                page = _fetch_with_retry(
                    fetch_page,
                    api_key,
                    retries=retries,
                    retry_delay=retry_delay,
                    limit=page_size,
                    offset=offset,
                    ordering="transfer_date",
                    date_from=date_from,
                    date_to=date_to,
                )
                if isinstance(page.get("count"), int):
                    total = page["count"]
                batch = page["results"]
                if not batch:
                    if page.get("next"):
                        raise RuntimeError("BSD returned an empty page with a next link")
                    break

                retrieved_at = datetime.now(timezone.utc).isoformat()
                _valid, invalid = _upsert_page(
                    conn, batch, retrieved_at, seen_table=True
                )
                invalid_total += invalid
                offset += len(batch)
                page_number += 1
                conn.commit()

                if progress_every and page_number % progress_every == 0:
                    print("  " + _progress(offset, total, started), flush=True)
                if not page.get("next"):
                    break
                if request_delay:
                    time.sleep(request_delay)

            # The window is complete at this point. Remove provider rows that
            # disappeared from that same window, but never touch older history.
            removed = conn.execute(
                "DELETE FROM transfers "
                "WHERE source=? AND transfer_date BETWEEN ? AND ? "
                "AND id NOT IN (SELECT id FROM import_seen)",
                (SOURCE, date_from, date_to),
            ).rowcount
            _meta_set(conn, "last_successful_sync_date", date_to)
            _meta_set(conn, "last_successful_sync_utc",
                      datetime.now(timezone.utc).isoformat())
            _meta_set(conn, "last_update_date_from", date_from)
            _meta_set(conn, "last_update_rows", offset)
            _meta_set(conn, "invalid_rows_last_update", invalid_total)
            conn.commit()
            _verify(conn)
            count = conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
        except BaseException:
            conn.close()
            raise
        else:
            conn.close()
        _publish(staging, db_path)

    return {
        "rows": count,
        "fetched": offset,
        "removed": removed,
        "invalid_rows": invalid_total,
        "date_from": date_from,
        "date_to": date_to,
        "database": str(db_path),
    }


def status(db_path: Path = DEFAULT_DB) -> dict:
    if not db_path.exists():
        return {"exists": False, "database": str(db_path)}
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        count, first_date, last_date, players = conn.execute(
            "SELECT COUNT(*), MIN(transfer_date), MAX(transfer_date), "
            "COUNT(DISTINCT player_id) FROM transfers"
        ).fetchone()
        return {
            "exists": True,
            "database": str(db_path),
            "rows": count,
            "players": players,
            "first_transfer_date": first_date,
            "last_transfer_date": last_date,
            "last_successful_sync_date": _meta_get(
                conn, "last_successful_sync_date"
            ),
            "last_successful_sync_utc": _meta_get(
                conn, "last_successful_sync_utc"
            ),
            "invalid_rows": int(_meta_get(conn, "invalid_rows", "0")),
        }


def _print_result(result: dict) -> None:
    for key, value in result.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import BSD transfer history into an atomic SQLite cache."
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--backfill", action="store_true",
                         help="resumable full-history import")
    actions.add_argument("--update", action="store_true",
                         help="incremental refresh of a completed database")
    actions.add_argument("--status", action="store_true",
                         help="show local database status; no API call")
    parser.add_argument("--replace", action="store_true",
                        help="allow --backfill to replace an existing live DB")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--api-key", default="",
                        help="BSD key override; normally read from BSD_API_KEY/api_keys.json")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument("--overlap-days", type=int, default=DEFAULT_OVERLAP_DAYS)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=25,
                        help="print progress every N pages (0 disables)")
    args = parser.parse_args()

    if args.status:
        _print_result(status(args.db))
        return
    from .runtime_safety import assert_writer_host
    assert_writer_host()
    api_key = args.api_key or get_key("bsd", env="BSD_API_KEY")
    if not api_key:
        parser.error(
            "No BSD key. Set BSD_API_KEY or add 'bsd' to data/api_keys.json."
        )

    common = {
        "db_path": args.db,
        "page_size": args.page_size,
        "retries": args.retries,
        "retry_delay": args.retry_delay,
        "request_delay": args.request_delay,
        "progress_every": args.progress_every,
    }
    if args.backfill:
        result = backfill(api_key, replace=args.replace, **common)
    else:
        result = update(
            api_key, overlap_days=args.overlap_days, **common
        )
    _print_result(result)


if __name__ == "__main__":
    main()
