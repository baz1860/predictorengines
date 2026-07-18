"""The Racing API client and canonical GB/IE flat-racing adapter.

The provider supplies stable entity IDs, historical racecards/results and
timestamped bookmaker price changes. It does not supply executable available
size. Consequently imported odds are useful as a validation baseline but are
deliberately ineligible for edge recommendations.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..config import DEFAULT_CUTOFF_MINUTES
from ..schema import (ODDS_COLUMNS, RACE_COLUMNS, RESULT_COLUMNS, RUNNER_COLUMNS,
                      DataError, load_bundle)

BASE_URL = "https://api.theracingapi.com/v1"
PROVIDER = "the_racing_api"
USERNAME_ENV = "THE_RACING_API_USERNAME"
PASSWORD_ENV = "THE_RACING_API_PASSWORD"


def _utc(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise DataError(f"The Racing API returned an invalid timestamp: {value!r}")
    return pd.Timestamp(parsed)


def _iso(value: Any) -> str:
    return _utc(value).isoformat()


def _number(value: Any, default: str = "") -> Any:
    parsed = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(parsed) else parsed


def _field_version(race: dict) -> int:
    horses = sorted(str(r.get("horse_id") or "") for r in race.get("runners", []))
    return zlib.crc32("\x1f".join(horses).encode()) & 0x7fffffff


def _jurisdiction(region: Any) -> str:
    value = str(region or "").strip().upper()
    mapping = {"GB": "GB", "GREAT BRITAIN": "GB", "IRE": "IE", "IE": "IE",
               "IRELAND": "IE"}
    return mapping.get(value, value)


def _surface(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"turf", "grass"}:
        return "turf"
    if raw in {"all_weather", "allweather", "aw", "synthetic", "polytrack",
               "tapeta", "fibresand"}:
        return "all_weather"
    return raw


def _distance_metres(race: dict) -> Any:
    direct = _number(race.get("distance") or race.get("dist_m"), None)
    if direct is not None:
        # Racecard `distance` is metres, while some examples use a display value.
        if float(direct) >= 400:
            return round(float(direct))
    furlongs = _number(race.get("distance_f") or race.get("dist_f"), None)
    if furlongs is not None:
        text = str(furlongs).lower().replace("f", "").strip()
        parsed = _number(text, None)
        if parsed is not None:
            return round(float(parsed) * 201.168)
    yards = _number(race.get("dist_y"), None)
    return "" if yards is None else round(float(yards) * 0.9144)


def _race_class(value: Any) -> Any:
    text = str(value or "").strip().lower().replace("class", "").replace("_", " ")
    token = text.strip().split(" ")[0] if text.strip() else ""
    return _number(token)


def _position(value: Any) -> Any:
    text = str(value or "").strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else ""


def _completion(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "finished" if _position(text) != "" else (text or "not_finished")


@dataclass
class ArchivedResponse:
    endpoint: str
    params: dict[str, Any]
    retrieved_at: str
    payload: dict[str, Any]
    path: Path


class Client:
    def __init__(self, username: str | None = None, password: str | None = None,
                 base_url: str = BASE_URL, timeout: int = 30,
                 session: requests.Session | None = None, min_interval: float = 0.55):
        self.username = username or os.getenv(USERNAME_ENV, "")
        self.password = password or os.getenv(PASSWORD_ENV, "")
        if not self.username or not self.password:
            raise DataError(f"set {USERNAME_ENV} and {PASSWORD_ENV}")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.min_interval = max(0.0, float(min_interval))
        self._last_request = 0.0

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        last: Exception | None = None
        for attempt in range(4):
            try:
                delay = self.min_interval - (time.monotonic() - self._last_request)
                if delay > 0:
                    time.sleep(delay)
                response = self.session.get(url, params=clean,
                                            auth=(self.username, self.password),
                                            timeout=self.timeout,
                                            headers={"Accept": "application/json"})
                self._last_request = time.monotonic()
                if response.status_code == 429 or response.status_code >= 500:
                    raise RuntimeError(f"provider HTTP {response.status_code}")
                if response.status_code in {401, 403}:
                    raise DataError("The Racing API credentials or plan permissions were rejected")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise DataError("The Racing API returned a non-object response")
                return payload
            except DataError:
                raise
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last = exc
                if attempt < 3:
                    time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"The Racing API request failed: {last}")


def archive_payload(raw_root: Path, endpoint: str, params: dict[str, Any],
                    payload: dict[str, Any], retrieved_at: str) -> ArchivedResponse:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    checksum = hashlib.sha256(body).hexdigest()
    stamp = _utc(retrieved_at)
    folder = raw_root / PROVIDER / endpoint.replace("/", "_") / stamp.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{stamp.strftime('%Y%m%dT%H%M%S%fZ')}_{checksum[:16]}.json"
    record = {"provider": PROVIDER, "endpoint": endpoint, "params": params,
              "retrieved_at": stamp.isoformat(), "payload_sha256": checksum,
              "payload": payload}
    if not target.exists():
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return ArchivedResponse(endpoint, params, stamp.isoformat(), payload, target)


def normalize(racecards: list[dict], results: list[dict], retrieved_at: str,
              bookmaker: str | None = None) -> dict[str, pd.DataFrame]:
    """Convert provider payloads into canonical tables.

    Historical cards do not expose declaration publication timestamps. For a
    historical card, source_updated_at is conservatively pinned to its decision
    cutoff and the manifest marks the assumption. This is research-grade replay,
    not proof of field-level point-in-time provenance.
    """
    result_by_race = {str(row.get("race_id")): row for row in results}
    races_out: list[dict] = []
    runners_out: list[dict] = []
    results_out: list[dict] = []
    odds_out: list[dict] = []
    assumptions: list[str] = []
    skipped = {"out_of_scope": 0, "unsupported_surface": 0, "missing_stable_ids": 0}
    retrieved = _utc(retrieved_at)
    record_version = int(retrieved.timestamp())

    for card in racecards:
        rid = str(card.get("race_id") or "").strip()
        if not rid or str(card.get("type") or "").strip().lower() != "flat":
            skipped["out_of_scope"] += 1
            continue
        jurisdiction = _jurisdiction(card.get("region"))
        if jurisdiction not in {"GB", "IE"}:
            skipped["out_of_scope"] += 1
            continue
        surface = _surface(card.get("surface"))
        if surface not in {"turf", "all_weather"}:
            skipped["unsupported_surface"] += 1
            continue
        card_runners = card.get("runners") or []
        stable = [r for r in card_runners
                  if str(r.get("horse_id") or "").strip()
                  and str(r.get("trainer_id") or "").strip()
                  and str(r.get("jockey_id") or "").strip()]
        if len(stable) != len(card_runners) or len(stable) < 2:
            skipped["missing_stable_ids"] += 1
            continue
        off = _utc(card.get("off_dt"))
        cutoff = off - pd.Timedelta(minutes=DEFAULT_CUTOFF_MINUTES)
        historical = retrieved > cutoff
        source_at = cutoff if historical else retrieved
        if historical:
            assumptions.append("historical racecard fields pinned to the decision cutoff")
        version = _field_version(card)
        course = str(card.get("course") or "").strip()
        races_out.append({
            "race_id": rid, "meeting_date": str(card.get("date") or off.date()),
            "scheduled_off_utc": off.isoformat(), "course_id": str(card.get("course_id") or ""),
            "course_name": course, "jurisdiction": jurisdiction, "code": "flat",
            "surface": surface, "going": str(card.get("going") or ""),
            "distance_metres": _distance_metres(card), "race_class": _race_class(card.get("race_class")),
            "handicap_flag": int("handicap" in str(card.get("race_name") or "").lower()),
            "prediction_cutoff": cutoff.isoformat(), "source_updated_at": source_at.isoformat(),
            "record_version": record_version,
        })
        for runner in card_runners:
            horse_id = str(runner.get("horse_id") or "").strip()
            if not horse_id:
                continue
            runner_id = f"{rid}:{horse_id}"
            lbs = _number(runner.get("lbs"), None)
            runners_out.append({
                "race_id": rid, "runner_id": runner_id, "horse_id": horse_id,
                "horse_name": str(runner.get("horse") or ""),
                "trainer_id": str(runner.get("trainer_id") or ""),
                "trainer_name": str(runner.get("trainer") or ""),
                "jockey_id": str(runner.get("jockey_id") or ""),
                "jockey_name": str(runner.get("jockey") or ""),
                "draw": _number(runner.get("draw")), "age": _number(runner.get("age")),
                "weight_carried_kg": "" if lbs is None else round(float(lbs) * 0.45359237, 3),
                "official_rating": _number(runner.get("ofr")), "declared_status": "active",
                "non_runner_status": "", "field_version": version,
                "source_updated_at": source_at.isoformat(),
                "record_version": record_version,
            })
            for quote in runner.get("odds") or []:
                source = str(quote.get("bookmaker") or "").strip()
                if bookmaker and source.casefold() != bookmaker.casefold():
                    continue
                for item in quote.get("history") or []:
                    when = item.get("changed_at")
                    decimal = _number(item.get("decimal"), None)
                    if when and decimal is not None:
                        odds_out.append({
                            "race_id": rid, "runner_id": runner_id, "market_id": "win",
                            "source": source, "decimal_odds": decimal, "available_size": "",
                            "captured_at": _iso(when), "field_version": version,
                            "market_status": "open",
                        })

        result = result_by_race.get(rid)
        if result:
            result_off = _utc(result.get("off_dt") or card.get("off_dt"))
            # The endpoint has no publication timestamp. A next-day availability
            # assumption prevents same-day leakage and is recorded in the manifest.
            published = (result_off + pd.Timedelta(days=1)).normalize()
            assumptions.append("result availability inferred as 00:00 UTC on the next day")
            result_horses = {str(r.get("horse_id")): r for r in result.get("runners") or []}
            for runner in card_runners:
                horse_id = str(runner.get("horse_id") or "")
                rr = result_horses.get(horse_id)
                if not rr:
                    continue
                results_out.append({
                    "race_id": rid, "runner_id": f"{rid}:{horse_id}",
                    "finish_position": _position(rr.get("position")),
                    "completion_status": _completion(rr.get("position")),
                    "result_published_at": published.isoformat(), "result_updated_at": "",
                    "record_version": record_version,
                })

    frames = {
        "races": pd.DataFrame(races_out, columns=RACE_COLUMNS),
        "runners": pd.DataFrame(runners_out, columns=RUNNER_COLUMNS),
        "results": pd.DataFrame(results_out, columns=RESULT_COLUMNS),
        "odds": pd.DataFrame(odds_out, columns=ODDS_COLUMNS),
    }
    frames["manifest"] = {
        "provider": PROVIDER, "retrieved_at": retrieved.isoformat(),
        "validation_grade": "research_only" if assumptions else "point_in_time",
        "assumptions": sorted(set(assumptions)),
        "odds_executable": False,
        "reason_odds_not_executable": "provider does not publish available_size",
        "skipped_races": skipped,
        "rows": {name: int(len(frame)) for name, frame in frames.items()},
    }
    return frames


def _upsert(path: Path, incoming: pd.DataFrame, keys: list[str]) -> None:
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        combined = pd.concat([existing, incoming.astype(str)], ignore_index=True)
    else:
        combined = incoming.astype(str)
    combined = combined.drop_duplicates(keys, keep="last")
    combined.to_csv(path, index=False)


def ingest(client: Client, start_date: date, end_date: date, data_dir: str | Path,
           bookmaker: str | None = "Betfair Exchange") -> dict:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw_root = root / "raw"
    cards: list[dict] = []
    results: list[dict] = []
    raw_files: list[str] = []
    day = start_date
    while day <= end_date:
        retrieved = datetime.now(timezone.utc).isoformat()
        card_params = {"date": day.isoformat(), "region_codes": ["gb", "ire"],
                       "limit": 500, "skip": 0}
        card_payload = client.get("racecards/pro", card_params)
        raw_files.append(str(archive_payload(raw_root, "racecards_pro", card_params,
                                             card_payload, retrieved).path))
        cards.extend(card_payload.get("racecards") or [])
        skip = 0
        while True:
            result_params = {"start_date": day.isoformat(), "end_date": day.isoformat(),
                             "region": ["gb", "ire"], "type": ["flat"],
                             "limit": 100, "skip": skip}
            payload = client.get("results", result_params)
            raw_files.append(str(archive_payload(raw_root, "results", result_params,
                                                 payload, retrieved).path))
            batch = payload.get("results") or []
            results.extend(batch)
            skip += len(batch)
            if not batch or skip >= int(payload.get("total") or 0):
                break
        day += timedelta(days=1)
    fetched_at = datetime.now(timezone.utc).isoformat()
    normalized = normalize(cards, results, fetched_at, bookmaker=bookmaker)
    _upsert(root / "races.csv", normalized["races"], ["race_id"])
    _upsert(root / "runners.csv", normalized["runners"],
            ["race_id", "runner_id", "record_version"])
    _upsert(root / "results.csv", normalized["results"],
            ["race_id", "runner_id", "record_version"])
    _upsert(root / "odds.csv", normalized["odds"],
            ["race_id", "runner_id", "source", "captured_at"])
    manifest = normalized["manifest"] | {
        "date_range": [start_date.isoformat(), end_date.isoformat()],
        "bookmaker": bookmaker, "raw_files": raw_files,
    }
    previous_path = root / "provider_manifest.json"
    if previous_path.exists():
        try:
            previous = json.loads(previous_path.read_text())
        except (OSError, ValueError):
            previous = {}
        if previous.get("validation_grade") == "research_only":
            manifest["validation_grade"] = "research_only"
        manifest["assumptions"] = sorted(set(previous.get("assumptions", []))
                                          | set(manifest.get("assumptions", [])))
        manifest["raw_files"] = sorted(set(previous.get("raw_files", []))
                                        | set(manifest["raw_files"]))
    (root / "provider_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    bundle = load_bundle(root)
    manifest["rows"] = {"races": len(bundle.races), "runners": len(bundle.runners),
                        "results": len(bundle.results), "odds": len(bundle.odds)}
    (root / "provider_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
