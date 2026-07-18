#!/usr/bin/env python3
"""Offline adversarial contracts for the Betfair Historical BASIC adapter."""
from __future__ import annotations

import bz2
import io
import json
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

from horse_racing.providers import betfair_historical as B
from horse_racing.schema import load_bundle


def _message(pt, market_id, market_definition=None, changes=None):
    market = {"id": market_id}
    if market_definition is not None:
        market["marketDefinition"] = market_definition
    if changes is not None:
        market["rc"] = changes
    return json.dumps({"op": "mcm", "pt": pt, "mc": [market]}).encode() + b"\n"


def _dataset(root: Path, days: list[int]) -> None:
    races, runners = [], []
    for day in days:
        off = pd.Timestamp(f"2023-06-{day:02d}T14:00:00Z")
        race_id = f"rp:race:{day}"
        races.append({"race_id": race_id, "meeting_date": str(off.date()),
                      "scheduled_off_utc": off.isoformat(), "course_id": "rp:course:1",
                      "course_name": "Goodwood", "jurisdiction": "GB", "code": "flat",
                      "surface": "turf", "going": "good", "distance_metres": 1600,
                      "race_class": 2, "handicap_flag": 0, "prediction_cutoff": "",
                      "source_updated_at": (off-pd.Timedelta(hours=2)).isoformat(),
                      "record_version": 1})
        for selection, name in ((11, "Alpha's Boy"), (22, "Beta Girl (IRE)")):
            runners.append({"race_id": race_id,
                            "runner_id": f"{race_id}:runner:{selection}",
                            "horse_id": f"horse:{selection}", "horse_name": name,
                            "trainer_id": f"trainer:{selection}", "trainer_name": "T",
                            "jockey_id": f"jockey:{selection}", "jockey_name": "J",
                            "draw": selection // 11, "age": 4, "weight_carried_kg": 58,
                            "official_rating": 90, "declared_status": "active",
                            "non_runner_status": "", "field_version": 7,
                            "source_updated_at": (off-pd.Timedelta(hours=2)).isoformat(),
                            "record_version": 1})
    pd.DataFrame(races).to_csv(root / "races.csv", index=False)
    pd.DataFrame(runners).to_csv(root / "runners.csv", index=False)
    pd.DataFrame(columns=["race_id", "runner_id", "finish_position", "completion_status",
                          "result_published_at", "result_updated_at", "record_version"]).to_csv(
                              root / "results.csv", index=False)
    pd.DataFrame(columns=["race_id", "runner_id", "market_id", "source", "decimal_odds",
                          "available_size", "captured_at", "field_version",
                          "market_status"]).to_csv(root / "odds.csv", index=False)


def _archive(path: Path, day: int, *, stale=False, delayed=False) -> None:
    off = pd.Timestamp(f"2023-06-{day:02d}T14:00:00Z")
    market_id = f"1.2345678{day:02d}"
    definition = {"eventTypeId": "7", "marketType": "WIN", "numberOfWinners": 1,
                  "countryCode": "GB", "venue": "Goodwood", "eventId": f"event-{day}",
                  "eventName": f"Goodwood {day}", "name": "1m Flat",
                  "marketTime": off.isoformat(), "status": "OPEN", "inPlay": False,
                  "runners": [{"id": 11, "name": "Alphas Boy", "status": "ACTIVE"},
                              {"id": 22, "name": "Beta Girl", "status": "ACTIVE"}]}
    price_at = off - pd.Timedelta(hours=7) if stale else off - pd.Timedelta(minutes=16)
    first_at = price_at - pd.Timedelta(seconds=1)
    snapshot_at = off - pd.Timedelta(minutes=16)
    messages = [
        _message(int(first_at.timestamp()*1000), market_id, definition),
        _message(int(price_at.timestamp()*1000), market_id,
                 changes=[{"id": 11, "ltp": 3.5}, {"id": 22, "ltp": 1.8}]),
    ]
    if stale:
        messages.append(_message(int(snapshot_at.timestamp()*1000), market_id, changes=[]))
    if delayed:
        moved = definition | {"marketTime": (off + pd.Timedelta(minutes=2)).isoformat()}
        messages.append(_message(int((off-pd.Timedelta(minutes=14)).timestamp()*1000),
                                 market_id, moved))
    else:
        messages.append(_message(int((off-pd.Timedelta(minutes=14)).timestamp()*1000),
                                 market_id, changes=[{"id": 11, "ltp": 99.0}]))
    compressed = bz2.compress(b"".join(messages))
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo(
            f"BASIC/2023/Jun/{day}/123/{market_id}.bz2")
        info.size = len(compressed)
        tar.addfile(info, io.BytesIO(compressed))


def _trim_at_member_boundary(source: Path, target: Path) -> None:
    payload = source.read_bytes()
    while payload.endswith(b"\0" * 512):
        payload = payload[:-512]
    target.write_bytes(payload)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # Additive upsert and idempotency across disjoint archives.
        root = base / "multi"
        root.mkdir()
        _dataset(root, [1, 2])
        archive_a, archive_b = base / "a.tar", base / "b.tar"
        _archive(archive_a, 1)
        _archive(archive_b, 2)
        first = B.ingest_basic_archive(archive_a, root)
        assert first["stats"]["written_boards"] == 1
        B.ingest_basic_archive(archive_b, root)
        odds = load_bundle(root).odds
        assert len(odds) == 4 and odds["race_id"].nunique() == 2
        B.ingest_basic_archive(archive_a, root)
        odds = load_bundle(root).odds
        assert len(odds) == 4 and odds["race_id"].nunique() == 2
        manifest = json.loads((root / "betfair_manifest.json").read_text())
        assert len(manifest["archives"]) == 2
        assert manifest["dataset"] == {"odds_rows": 4, "races": 2}
        assert odds["available_size"].isna().all()
        assert set(odds["field_version"]) == {7}
        assert set(odds["market_status"]) == {"open"}
        assert odds["provider_archive_sha256"].astype(str).str.len().eq(64).all()

        # An old component price must not inherit a fresh board heartbeat.
        stale_root = base / "stale"
        stale_root.mkdir()
        _dataset(stale_root, [3])
        stale_archive = base / "stale.tar"
        _archive(stale_archive, 3, stale=True)
        stale = B.ingest_basic_archive(stale_archive, stale_root)
        assert stale["stats"]["written_boards"] == 0
        assert stale["quarantine"]["stale_component_ltp"] == 1
        assert load_bundle(stale_root).odds.empty

        # A delayed Betfair marketTime must not move beyond canonical cutoff.
        drift_root = base / "drift"
        drift_root.mkdir()
        _dataset(drift_root, [4])
        drift_archive = base / "drift.tar"
        _archive(drift_archive, 4, delayed=True)
        drift = B.ingest_basic_archive(drift_archive, drift_root)
        assert drift["quarantine"]["snapshot_after_canonical_cutoff"] == 1
        assert load_bundle(drift_root).odds.empty

        # Clean member-boundary truncation is incomplete even without ReadError.
        boundary_root = base / "boundary"
        boundary_root.mkdir()
        _dataset(boundary_root, [5])
        complete_archive, boundary_archive = base / "complete.tar", base / "boundary.tar"
        _archive(complete_archive, 5)
        _trim_at_member_boundary(complete_archive, boundary_archive)
        boundary = B.ingest_basic_archive(boundary_archive, boundary_root)
        assert boundary["archive"]["complete"] is False
        assert boundary["archive"]["end_markers_present"] is False

        # Compressed member limits are enforced before decompression.
        limit_root = base / "limit"
        limit_root.mkdir()
        _dataset(limit_root, [6])
        limit_archive = base / "limit.tar"
        _archive(limit_archive, 6)
        original_limit = B.MAX_COMPRESSED_MEMBER_BYTES
        try:
            B.MAX_COMPRESSED_MEMBER_BYTES = 1
            limited = B.ingest_basic_archive(limit_archive, limit_root)
        finally:
            B.MAX_COMPRESSED_MEMBER_BYTES = original_limit
        assert limited["quarantine"]["oversized_compressed_member"] == 1
        assert load_bundle(limit_root).odds.empty

    print("horse racing Betfair adapter tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
