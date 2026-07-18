"""Stream Betfair Historical BASIC archives into canonical cutoff win odds.

BASIC supplies sampled last-traded prices, not an executable price/size ladder.
The adapter therefore produces an analytical point-in-time benchmark with blank
available_size.  Identity joins are deliberately strict and ambiguous markets
are quarantined rather than guessed.
"""
from __future__ import annotations

import bz2
import hashlib
import json
import re
import tarfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DEFAULT_CUTOFF_MINUTES
from ..schema import DataError, load_bundle, race_cutoff, runner_snapshot

PROVIDER = "betfair_historical_basic"
SOURCE = "betfair_historical_basic"
_MARKET_FILE = re.compile(r"^1\.\d+\.bz2$")
MAX_COMPRESSED_MEMBER_BYTES = 8 * 1024 * 1024
MAX_DECOMPRESSED_MARKET_BYTES = 64 * 1024 * 1024
MAX_JSON_LINE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_COMPONENT_STALENESS_SECONDS = 3600


def _key(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\([^)]*\)\s*$", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _utc_ms(value) -> pd.Timestamp:
    return pd.Timestamp(int(value), unit="ms", tz="UTC")


def _parse_market(fileobj, cutoff_minutes: int) -> dict | None:
    """Return the market state and runner LTPs as known at prediction cutoff."""
    definition: dict = {}
    runners: dict[int, dict] = {}
    prices: dict[int, tuple[float, pd.Timestamp]] = {}
    cutoff: pd.Timestamp | None = None
    status = ""
    in_play = False
    market_id = ""
    snapshot_at: pd.Timestamp | None = None

    try:
        stream = bz2.BZ2File(fileobj)
        decompressed = 0
        while True:
            raw_line = stream.readline(MAX_JSON_LINE_BYTES + 1)
            if not raw_line:
                break
            if len(raw_line) > MAX_JSON_LINE_BYTES:
                raise DataError("Betfair market JSON line exceeds safety limit")
            decompressed += len(raw_line)
            if decompressed > MAX_DECOMPRESSED_MARKET_BYTES:
                raise DataError("Betfair market stream exceeds decompression safety limit")
            message = json.loads(raw_line)
            captured = _utc_ms(message["pt"])
            if cutoff is not None and captured <= cutoff:
                snapshot_at = captured
            for change in message.get("mc", []):
                market_id = str(change.get("id") or market_id)
                update = change.get("marketDefinition") or {}
                if update:
                    if update.get("marketTime"):
                        market_time = pd.Timestamp(update["marketTime"])
                        cutoff = market_time - pd.Timedelta(minutes=cutoff_minutes)
                    if cutoff is not None and captured <= cutoff:
                        snapshot_at = captured
                        definition.update(update)
                        status = str(update.get("status", status)).lower()
                        in_play = bool(update.get("inPlay", in_play))
                        for runner in update.get("runners", []):
                            selection_id = int(runner["id"])
                            current = runners.setdefault(selection_id, {})
                            current.update(runner)
                if cutoff is None or captured > cutoff:
                    continue
                for runner_change in change.get("rc", []):
                    value = runner_change.get("ltp")
                    if value is None:
                        continue
                    price = float(value)
                    if np.isfinite(price) and price > 1:
                        prices[int(runner_change["id"])] = (price, captured)
            if cutoff is not None and captured > cutoff:
                break
    except (OSError, EOFError, ValueError, json.JSONDecodeError) as exc:
        raise DataError(f"invalid Betfair BASIC market stream: {exc}") from exc

    if not definition or cutoff is None or snapshot_at is None:
        return None
    if (str(definition.get("eventTypeId")) != "7"
            or str(definition.get("marketType", "")).upper() != "WIN"
            or int(definition.get("numberOfWinners", 0)) != 1
            or str(definition.get("countryCode", "")).upper() not in {"GB", "IE"}):
        return None
    active = {
        selection_id: data for selection_id, data in runners.items()
        if str(data.get("status", "ACTIVE")).upper() == "ACTIVE"
    }
    return {
        "betfair_market_id": market_id,
        "event_id": str(definition.get("eventId", "")),
        "market_time": pd.Timestamp(definition["marketTime"]),
        "cutoff": cutoff,
        "snapshot_at": snapshot_at,
        "jurisdiction": str(definition["countryCode"]).upper(),
        "venue": str(definition.get("venue", "")),
        "event_name": str(definition.get("eventName", "")),
        "market_name": str(definition.get("name", "")),
        "status": status or str(definition.get("status", "")).lower(),
        "in_play": in_play,
        "runners": active,
        "prices": prices,
    }


def _canonical_index(bundle, cutoff_minutes: int) -> list[dict]:
    indexed = []
    for _index, row in bundle.races.iterrows():
        cutoff = race_cutoff(row, cutoff_minutes)
        race_id = str(row["race_id"])
        runners = runner_snapshot(bundle, race_id, cutoff)
        names = {_key(item.horse_name): str(item.runner_id)
                 for item in runners.itertuples(index=False)}
        if len(names) != len(runners) or len(names) < 2:
            continue
        indexed.append({
            "race_id": race_id,
            "off": pd.Timestamp(row["scheduled_off_utc"]),
            "date": pd.Timestamp(row["scheduled_off_utc"]).date(),
            "jurisdiction": str(row["jurisdiction"]).upper(),
            "course": _key(row["course_name"]),
            "runners": names,
            "field_versions": runners.set_index("runner_id")["field_version"].to_dict(),
            "cutoff": cutoff,
        })
    return indexed


def _match_market(market: dict, canonical: list[dict], tolerance_minutes: int) -> tuple[dict | None, str]:
    market_names = {_key(data.get("name")): selection_id
                    for selection_id, data in market["runners"].items()}
    if "" in market_names or len(market_names) != len(market["runners"]):
        return None, "duplicate_or_blank_market_runner_name"
    scoped = [race for race in canonical
              if race["jurisdiction"] == market["jurisdiction"]
              and race["date"] == market["market_time"].date()]
    if not scoped:
        return None, "no_canonical_date_scope"
    timed = []
    for race in scoped:
        delta = abs((race["off"] - market["market_time"]).total_seconds()) / 60
        if delta <= tolerance_minutes:
            timed.append((delta, race))
    if not timed:
        return None, "no_scheduled_off_match"
    course = [(delta, race) for delta, race in timed
              if race["course"] == _key(market["venue"])]
    if not course:
        return None, "no_course_match"
    candidates = [(delta, race) for delta, race in course
                  if set(race["runners"]) == set(market_names)]
    if len(candidates) != 1:
        return None, "runner_field_mismatch" if not candidates else "ambiguous_identity_match"
    race = candidates[0][1]
    mapping = {market_names[name]: runner_id for name, runner_id in race["runners"].items()}
    return race | {"selection_to_runner": mapping}, "matched"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_tar_end_markers(path: Path) -> bool:
    if path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        handle.seek(-1024, 2)
        return handle.read(1024) == b"\0" * 1024


def ingest_basic_archive(archive: str | Path, data_dir: str | Path,
                         cutoff_minutes: int = DEFAULT_CUTOFF_MINUTES,
                         tolerance_minutes: int = 20,
                         max_component_staleness_seconds: int =
                         DEFAULT_MAX_COMPONENT_STALENESS_SECONDS) -> dict:
    archive = Path(archive).resolve()
    root = Path(data_dir)
    if not archive.is_file():
        raise DataError(f"Betfair archive does not exist: {archive}")
    if max_component_staleness_seconds <= 0:
        raise ValueError("max component staleness must be positive")
    archive_sha256 = _sha256(archive)
    bundle = load_bundle(root)
    if bundle.races.empty:
        raise DataError("canonical races must be ingested before Betfair odds")
    canonical = _canonical_index(bundle, cutoff_minutes)
    wanted_dates = {item["date"] for item in canonical}
    stats = {"members_read": 0, "market_files_considered": 0, "win_markets": 0,
             "matched_markets": 0, "written_boards": 0}
    quarantine: dict[str, int] = {}
    archive_dates: dict[str, int] = {}
    odds_rows = []
    matches = []
    accepted_races: set[str] = set()
    blocked_duplicate_races: set[str] = set()
    archive_complete = True
    archive_error = ""
    last_member = ""

    try:
        with tarfile.open(archive, "r:") as tar:
            while True:
                member = tar.next()
                if member is None:
                    break
                stats["members_read"] += 1
                last_member = member.name
                basename = member.name.rsplit("/", 1)[-1]
                if not member.isfile() or not _MARKET_FILE.match(basename):
                    continue
                if member.size > MAX_COMPRESSED_MEMBER_BYTES:
                    quarantine["oversized_compressed_member"] = \
                        quarantine.get("oversized_compressed_member", 0) + 1
                    continue
                parts = member.name.split("/")
                try:
                    path_date = datetime.strptime(" ".join(parts[1:4]), "%Y %b %d").date()
                except (ValueError, IndexError):
                    quarantine["invalid_archive_path"] = quarantine.get("invalid_archive_path", 0) + 1
                    continue
                date_key = path_date.isoformat()
                archive_dates[date_key] = archive_dates.get(date_key, 0) + 1
                if path_date not in wanted_dates:
                    continue
                stats["market_files_considered"] += 1
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                try:
                    market = _parse_market(extracted, cutoff_minutes)
                except DataError:
                    quarantine["invalid_market_stream"] = quarantine.get("invalid_market_stream", 0) + 1
                    continue
                if market is None:
                    continue
                stats["win_markets"] += 1
                race, reason = _match_market(market, canonical, tolerance_minutes)
                if race is None:
                    quarantine[reason] = quarantine.get(reason, 0) + 1
                    continue
                stats["matched_markets"] += 1
                prices = market["prices"]
                selection_ids = set(race["selection_to_runner"])
                if set(prices) & selection_ids != selection_ids:
                    quarantine["incomplete_cutoff_prices"] = quarantine.get("incomplete_cutoff_prices", 0) + 1
                    continue
                if market["in_play"]:
                    quarantine["in_play_at_cutoff"] = quarantine.get("in_play_at_cutoff", 0) + 1
                    continue
                race_id = race["race_id"]
                if market["snapshot_at"] > race["cutoff"]:
                    quarantine["snapshot_after_canonical_cutoff"] = \
                        quarantine.get("snapshot_after_canonical_cutoff", 0) + 1
                    continue
                max_staleness = max(
                    (market["snapshot_at"] - prices[selection_id][1]).total_seconds()
                    for selection_id in selection_ids)
                if max_staleness < 0 or max_staleness > max_component_staleness_seconds:
                    quarantine["stale_component_ltp"] = \
                        quarantine.get("stale_component_ltp", 0) + 1
                    continue
                if race_id in blocked_duplicate_races:
                    quarantine["duplicate_canonical_market"] = \
                        quarantine.get("duplicate_canonical_market", 0) + 1
                    continue
                if race_id in accepted_races:
                    # Two otherwise-valid exchange markets for one canonical race
                    # are ambiguous. Remove the first as well and fail closed.
                    quarantine["duplicate_canonical_market"] = \
                        quarantine.get("duplicate_canonical_market", 0) + 2
                    blocked_duplicate_races.add(race_id)
                    accepted_races.remove(race_id)
                    odds_rows = [row for row in odds_rows if row["race_id"] != race_id]
                    matches = [item for item in matches if item["race_id"] != race_id]
                    stats["written_boards"] -= 1
                    continue
                for selection_id, runner_id in race["selection_to_runner"].items():
                    price, _price_observed_at = prices[selection_id]
                    odds_rows.append({
                        "race_id": race["race_id"], "runner_id": runner_id,
                        "market_id": "win", "source": SOURCE,
                        "decimal_odds": price, "available_size": "",
                        "captured_at": market["snapshot_at"].isoformat(),
                        "field_version": race["field_versions"].get(runner_id, ""),
                        "market_status": market["status"],
                        "provider_archive_sha256": archive_sha256,
                    })
                stats["written_boards"] += 1
                accepted_races.add(race_id)
                matches.append({"race_id": race_id,
                                "betfair_market_id": market["betfair_market_id"],
                                "snapshot_at": market["snapshot_at"].isoformat(),
                                "max_component_ltp_staleness_seconds": max_staleness})
    except tarfile.ReadError as exc:
        archive_complete = False
        archive_error = str(exc)

    end_markers_present = _has_tar_end_markers(archive)
    if archive_complete and not end_markers_present:
        archive_complete = False
        archive_error = "POSIX tar end markers are absent"

    # Migrate the previous single-archive manifest so stricter re-ingestion can
    # remove rows that did not yet carry provider_archive_sha256.
    manifest_path = root / "betfair_manifest.json"
    previous_manifest = {}
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            previous_manifest = {}
    previous_owned_races = set()
    if (previous_manifest.get("archive") or {}).get("sha256") == archive_sha256:
        previous_owned_races = {str(item.get("race_id"))
                                for item in previous_manifest.get("matches", [])}

    existing = bundle.odds.copy()
    accepted_or_previous = accepted_races | previous_owned_races
    if not existing.empty:
        source_rows = existing["source"].astype(str) == SOURCE
        owned = (existing.get("provider_archive_sha256", pd.Series("", index=existing.index))
                 .astype(str) == archive_sha256)
        replaced_race = existing["race_id"].astype(str).isin(accepted_or_previous)
        existing = existing[~(source_rows & (owned | replaced_race))]
    new_odds = pd.DataFrame(odds_rows, columns=[
        "race_id", "runner_id", "market_id", "source", "decimal_odds",
        "available_size", "captured_at", "field_version", "market_status",
        "provider_archive_sha256",
    ])
    combined = pd.concat([existing, new_odds], ignore_index=True)
    combined = combined.drop_duplicates(["race_id", "runner_id", "source"], keep="last")
    if not combined.empty:
        captured = pd.to_datetime(combined["captured_at"], utc=True, errors="raise")
        combined["captured_at"] = captured.map(pd.Timestamp.isoformat)
    odds_path = root / "odds.csv"
    odds_tmp = root / "odds.csv.tmp"
    combined.to_csv(odds_tmp, index=False)
    odds_tmp.replace(odds_path)
    load_bundle(root)

    manifest = {
        "provider": PROVIDER,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "validation_grade": "point_in_time_observational",
        "odds_executable": False,
        "available_size_present": False,
        "price_semantics": "sampled last traded price as known at prediction cutoff",
        "timestamp_semantics": "reconstructed full-board snapshot; unchanged LTP deltas persist",
        "cutoff_minutes": cutoff_minutes,
        "max_component_ltp_staleness_seconds": max_component_staleness_seconds,
        "identity_policy": "exact jurisdiction/date/course/runner-set; ambiguous joins quarantined",
        "archive": {"path": str(archive), "bytes": archive.stat().st_size,
                    "sha256": archive_sha256, "complete": archive_complete,
                    "end_markers_present": end_markers_present,
                    "warning": "",
                    "error": archive_error,
                    "last_read_member": last_member,
                    "path_date_min": min(archive_dates) if archive_dates else None,
                    "path_date_max": max(archive_dates) if archive_dates else None,
                    "distinct_path_dates": len(archive_dates),
                    "market_files_by_year": {
                        year: sum(count for day, count in archive_dates.items()
                                  if day.startswith(year + "-"))
                        for year in sorted({day[:4] for day in archive_dates})
                    }},
        "stats": stats,
        "quarantine": quarantine,
        "matches": matches,
    }
    archive_manifest_dir = root / "raw" / "betfair" / "manifests"
    archive_manifest_dir.mkdir(parents=True, exist_ok=True)
    archive_manifest_path = archive_manifest_dir / f"{archive_sha256}.json"
    archive_manifest_tmp = archive_manifest_path.with_suffix(".json.tmp")
    archive_manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    archive_manifest_tmp.replace(archive_manifest_path)
    archive_summaries = []
    for path in sorted(archive_manifest_dir.glob("*.json")):
        try:
            item = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        archive_item = item.get("archive") or {}
        archive_summaries.append({
            "sha256": archive_item.get("sha256"), "path": archive_item.get("path"),
            "bytes": archive_item.get("bytes"), "complete": archive_item.get("complete"),
            "path_date_min": archive_item.get("path_date_min"),
            "path_date_max": archive_item.get("path_date_max"),
            "stats": item.get("stats", {}), "quarantine": item.get("quarantine", {}),
            "manifest_path": str(path),
        })
    manifest["archives"] = archive_summaries
    manifest["dataset"] = {
        "odds_rows": int((combined["source"].astype(str) == SOURCE).sum()),
        "races": int(combined.loc[combined["source"].astype(str) == SOURCE,
                                   "race_id"].astype(str).nunique()),
    }
    manifest_tmp = root / "betfair_manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_tmp.replace(manifest_path)
    return manifest
