from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from club_soccer import import_bsd_transfers as importer


def transfer(
    transfer_id: int,
    player_id: int,
    day: str,
    *,
    player_name: str = "Player",
    from_team: str = "Old FC",
    to_team: str = "New FC",
) -> dict:
    return {
        "id": transfer_id,
        "player": {"id": player_id, "name": player_name},
        "transfer_date": day,
        "from_team_id": 10,
        "from_team_name": from_team,
        "to_team_id": 20,
        "to_team_name": to_team,
        "transfer_type": 3,
        "fee_eur": 1_000_000,
        "fee_description": "1M €",
    }


def paged(rows: list[dict], calls: list[dict], *, fail_offset: int | None = None):
    def fetch(_api_key: str, **params):
        calls.append(params)
        offset = params["offset"]
        if fail_offset is not None and offset == fail_offset:
            raise RuntimeError("simulated API failure")
        limit = params["limit"]
        batch = rows[offset:offset + limit]
        next_link = "next" if offset + len(batch) < len(rows) else None
        return {
            "count": len(rows),
            "next": next_link,
            "previous": None,
            "results": batch,
        }

    return fetch


def test_backfill_paginates_deduplicates_and_publishes_atomically(tmp_path):
    db = tmp_path / "data" / "transfers.sqlite3"
    work = tmp_path / "work"
    rows = [
        transfer(1, 101, "2020-01-01", player_name="One"),
        transfer(2, 102, "2021-01-01", player_name="Two"),
        transfer(2, 102, "2021-01-01", player_name="Two corrected"),
        transfer(3, 103, "2022-01-01", player_name="Three"),
    ]
    calls: list[dict] = []

    result = importer.backfill(
        "key",
        db_path=db,
        work_dir=work,
        page_size=2,
        progress_every=0,
        today=date(2026, 7, 26),
        fetch_page=paged(rows, calls),
    )

    assert db.exists()
    assert not (work / "transfers_bsd.backfill.sqlite3").exists()
    assert result["rows"] == 3
    assert [call["offset"] for call in calls] == [0, 2]
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT player_name FROM transfers WHERE id=2"
        ).fetchone()[0] == "Two corrected"
        assert conn.execute(
            "SELECT value FROM metadata WHERE key='backfill_complete'"
        ).fetchone()[0] == "1"


def test_backfill_failure_keeps_live_database_and_resumes(tmp_path):
    db = tmp_path / "data" / "transfers.sqlite3"
    work = tmp_path / "work"
    rows = [
        transfer(1, 101, "2020-01-01"),
        transfer(2, 102, "2021-01-01"),
        transfer(3, 103, "2022-01-01"),
    ]
    failing_calls: list[dict] = []
    with pytest.raises(RuntimeError, match="simulated"):
        importer.backfill(
            "key",
            db_path=db,
            work_dir=work,
            page_size=2,
            retries=0,
            progress_every=0,
            today=date(2026, 7, 26),
            fetch_page=paged(rows, failing_calls, fail_offset=2),
        )

    assert not db.exists()
    assert (work / "transfers_bsd.backfill.sqlite3").exists()

    resume_calls: list[dict] = []
    result = importer.backfill(
        "key",
        db_path=db,
        work_dir=work,
        page_size=2,
        retries=0,
        progress_every=0,
        today=date(2026, 7, 27),
        fetch_page=paged(rows, resume_calls),
    )
    assert resume_calls[0]["offset"] == 2
    assert resume_calls[0]["date_to"] == "2026-07-26"
    assert result["rows"] == 3


def test_incremental_failure_does_not_modify_live_database(tmp_path):
    db = tmp_path / "data" / "transfers.sqlite3"
    work = tmp_path / "work"
    importer.backfill(
        "key",
        db_path=db,
        work_dir=work,
        progress_every=0,
        today=date(2026, 7, 20),
        fetch_page=paged([transfer(1, 101, "2026-07-19")], []),
    )
    before = db.read_bytes()

    with pytest.raises(RuntimeError, match="simulated"):
        importer.update(
            "key",
            db_path=db,
            work_dir=work,
            retries=0,
            progress_every=0,
            today=date(2026, 7, 26),
            fetch_page=paged([], [], fail_offset=0),
        )

    assert db.read_bytes() == before


def test_incremental_update_uses_overlap_upserts_and_reconciles_window(tmp_path):
    db = tmp_path / "data" / "transfers.sqlite3"
    work = tmp_path / "work"
    importer.backfill(
        "key",
        db_path=db,
        work_dir=work,
        progress_every=0,
        today=date(2026, 7, 20),
        fetch_page=paged(
            [
                transfer(1, 101, "2020-01-01", player_name="Old history"),
                transfer(2, 102, "2026-07-19", player_name="Removed correction"),
            ],
            [],
        ),
    )
    calls: list[dict] = []
    result = importer.update(
        "key",
        db_path=db,
        work_dir=work,
        overlap_days=7,
        progress_every=0,
        today=date(2026, 7, 26),
        fetch_page=paged(
            [
                transfer(3, 103, "2026-07-25", player_name="New"),
                transfer(3, 103, "2026-07-25", player_name="New corrected"),
            ],
            calls,
        ),
    )

    assert calls[0]["date_from"] == "2026-07-13"
    assert calls[0]["date_to"] == "2026-07-26"
    assert result["rows"] == 2
    assert result["removed"] == 1
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT id, player_name FROM transfers ORDER BY id"
        ).fetchall()
    assert rows == [(1, "Old history"), (3, "New corrected")]


def test_status_does_not_require_an_api_key(tmp_path):
    missing = importer.status(tmp_path / "missing.sqlite3")
    assert missing["exists"] is False
