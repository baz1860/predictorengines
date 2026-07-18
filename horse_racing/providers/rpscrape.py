"""Adapter for a pinned external rpscrape checkout.

rpscrape output is retrospective. Racing Post IDs are retained, but declaration
and result publication timestamps are conservatively inferred and explicitly
graded research-only. Starting prices are kept outside canonical cutoff odds.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import DEFAULT_CUTOFF_MINUTES
from ..schema import (ODDS_COLUMNS, RACE_COLUMNS, RESULT_COLUMNS, RUNNER_COLUMNS,
                      DataError, load_bundle)

PROVIDER = "rpscrape_racing_post"
PINNED_COMMIT = "f2c6977cd9a8b6ca20af823b3b2fac0ca39bce51"
# Reviewed safety patch:
# - network.py: bounded transport-error retries with linear backoff (upstream
#   only retried HTTP 406, so one curl timeout aborted a multi-month scrape).
# - race.py: a page persistently served without the race marker raises a fatal
#   PersistentMarkerError with backoff between attempts, instead of
#   VoidRaceError. rpscrape silently skips VoidRaceError races, so throttled or
#   challenge responses were being recorded as done and permanently dropped;
#   fatal-and-resume converges to full coverage instead.
EXPECTED_PATCH_SHA256 = "4555e974ee19cd74b0a2702b82f347d6baac995dfe00d91873592591708fdb0a"
PATCHED_FILES = ("scripts/utils/network.py", "scripts/utils/race.py")
REQUIRED_PATCHED_FILE = "scripts/utils/race.py"
REQUIRED = {
    "date", "region", "course_id", "course", "race_id", "off", "race_name",
    "type", "class", "dist_m", "going", "surface", "ran", "pos", "draw",
    "horse_id", "horse", "age", "lbs", "dec", "jockey_id", "jockey",
    "trainer_id", "trainer", "or",
}


def _id(kind: str, value) -> str:
    text = str(value or "").strip()
    return f"rp:{kind}:{text}" if text else ""


def _number(value, default=""):
    parsed = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(parsed) or not np.isfinite(parsed) else parsed


def _surface(value) -> str:
    raw = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    if raw in {"turf", "grass"}:
        return "turf"
    if raw in {"aw", "all weather", "polytrack", "tapeta", "fibresand", "synthetic"}:
        return "all_weather"
    return raw


def _race_class(value):
    text = str(value or "").lower().replace("class", "").strip()
    return _number(text.split()[0] if text else "")


def _off_utc(row: pd.Series) -> pd.Timestamp:
    region = str(row.get("region") or "").strip().upper()
    zone = ZoneInfo("Europe/Dublin" if region in {"IRE", "IE"} else "Europe/London")
    raw = f"{str(row['date']).strip()} {str(row['off']).strip()}"
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        raise DataError(f"rpscrape returned invalid date/off: {raw!r}")
    return pd.Timestamp(parsed).tz_localize(zone, ambiguous="raise",
                                            nonexistent="raise").tz_convert("UTC")


def _finish_position(value):
    parsed = _number(value, None)
    return "" if parsed is None or float(parsed) <= 0 else int(parsed)


def normalize_csv(path: str | Path, ingested_at: str | None = None) -> dict:
    source = Path(path)
    if not source.exists():
        raise DataError(f"rpscrape output does not exist: {source}")
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED - set(raw.columns))
    if missing:
        raise DataError(f"rpscrape output missing columns: {', '.join(missing)}")
    raw = raw.rename(columns={"or": "official_rating"})
    ingested = pd.Timestamp(ingested_at or datetime.now(timezone.utc).isoformat())
    if ingested.tzinfo is None:
        ingested = ingested.tz_localize("UTC")
    version = int(ingested.timestamp())
    races, runners, results, starting_prices = [], [], [], []
    skipped = {"out_of_scope": 0, "invalid_surface": 0, "missing_ids": 0,
               "incomplete_result": 0}

    for provider_race_id, group in raw.groupby("race_id", sort=False):
        first = group.iloc[0]
        region = str(first["region"]).strip().upper()
        if str(first["type"]).strip().lower() != "flat" or region not in {"GB", "IRE", "IE"}:
            skipped["out_of_scope"] += 1
            continue
        surface = _surface(first["surface"])
        if surface not in {"turf", "all_weather"}:
            skipped["invalid_surface"] += 1
            continue
        id_cols = ["race_id", "course_id", "horse_id", "trainer_id", "jockey_id"]
        if any(group[col].astype(str).str.strip().eq("").any() for col in id_cols):
            skipped["missing_ids"] += 1
            continue
        positions = pd.to_numeric(group["pos"], errors="coerce")
        if len(group) < 2 or int((positions == 1).sum()) < 1:
            skipped["incomplete_result"] += 1
            continue
        rid = _id("race", provider_race_id)
        off = _off_utc(first)
        cutoff = off - pd.Timedelta(minutes=DEFAULT_CUTOFF_MINUTES)
        field_key = "\x1f".join(sorted(group["horse_id"].astype(str)))
        field_version = int(hashlib.sha256(field_key.encode()).hexdigest()[:7], 16)
        jurisdiction = "IE" if region in {"IRE", "IE"} else "GB"
        races.append({
            "race_id": rid, "meeting_date": str(first["date"]),
            "scheduled_off_utc": off.isoformat(), "course_id": _id("course", first["course_id"]),
            "course_name": str(first["course"]), "jurisdiction": jurisdiction,
            "code": "flat", "surface": surface, "going": str(first["going"]),
            "distance_metres": _number(first["dist_m"]),
            "race_class": _race_class(first["class"]),
            "handicap_flag": int("handicap" in str(first["race_name"]).lower()),
            "prediction_cutoff": cutoff.isoformat(),
            "source_updated_at": cutoff.isoformat(), "record_version": version,
        })
        published = (off + pd.Timedelta(days=1)).normalize()
        for row in group.itertuples(index=False):
            horse_id = _id("horse", row.horse_id)
            runner_id = f"{rid}:{horse_id}"
            lbs = _number(row.lbs, None)
            runners.append({
                "race_id": rid, "runner_id": runner_id, "horse_id": horse_id,
                "horse_name": str(row.horse), "trainer_id": _id("trainer", row.trainer_id),
                "trainer_name": str(row.trainer), "jockey_id": _id("jockey", row.jockey_id),
                "jockey_name": str(row.jockey), "draw": _number(row.draw),
                "age": _number(row.age),
                "weight_carried_kg": "" if lbs is None else round(float(lbs) * 0.45359237, 3),
                "official_rating": _number(row.official_rating),
                "declared_status": "active", "non_runner_status": "",
                "field_version": field_version, "source_updated_at": cutoff.isoformat(),
                "record_version": version,
            })
            position = _finish_position(row.pos)
            results.append({
                "race_id": rid, "runner_id": runner_id, "finish_position": position,
                "completion_status": "finished" if position != "" else str(row.pos).lower(),
                "result_published_at": published.isoformat(), "result_updated_at": "",
                "record_version": version,
            })
            sp = _number(row.dec, None)
            if sp is not None and float(sp) > 1:
                starting_prices.append({
                    "race_id": rid, "runner_id": runner_id,
                    "decimal_odds": float(sp), "price_type": "official_starting_price",
                    "available_at": published.isoformat(), "source": "racing_post_via_rpscrape",
                })

    frames = {
        "races": pd.DataFrame(races, columns=RACE_COLUMNS),
        "runners": pd.DataFrame(runners, columns=RUNNER_COLUMNS),
        "results": pd.DataFrame(results, columns=RESULT_COLUMNS),
        "odds": pd.DataFrame(columns=ODDS_COLUMNS),
        "starting_prices": pd.DataFrame(starting_prices),
    }
    frames["manifest"] = {
        "provider": PROVIDER, "rpscrape_commit": PINNED_COMMIT,
        "ingested_at": ingested.isoformat(), "validation_grade": "research_only",
        "assumptions": [
            "retrospective runner declarations pinned to prediction cutoff",
            "result availability inferred as 00:00 UTC on the following day",
            "Racing Post IDs accepted as provider-stable within this dataset",
        ],
        "starting_prices_are_cutoff_odds": False,
        "rows": {name: len(frame) for name, frame in frames.items()},
        "skipped_races": skipped,
    }
    return frames


def _write_dataset(frames: dict, data_dir: Path, raw_sources: list[Path]) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw" / "rpscrape"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archived = []
    for raw_source in raw_sources:
        checksum = hashlib.sha256(raw_source.read_bytes()).hexdigest()
        raw_target = raw_dir / f"{raw_source.stem}_{checksum[:16]}.csv"
        if not raw_target.exists():
            shutil.copy2(raw_source, raw_target)
        archived.append({"path": str(raw_target), "sha256": checksum})
    # A racecard/result refresh must not destroy independently ingested market
    # history. Preserve only odds whose canonical race/runner pair still exists.
    odds_path = data_dir / "odds.csv"
    if odds_path.exists():
        existing_odds = pd.read_csv(odds_path, dtype=str, keep_default_na=False)
        valid = frames["runners"][["race_id", "runner_id"]].astype(str).drop_duplicates()
        existing_odds = existing_odds.merge(valid, on=["race_id", "runner_id"], how="inner")
        frames["odds"] = pd.concat([existing_odds, frames["odds"]], ignore_index=True)
        if not frames["odds"].empty:
            frames["odds"] = frames["odds"].drop_duplicates(
                ["race_id", "runner_id", "source", "captured_at"], keep="last")
    for name in ("races", "runners", "results", "odds", "starting_prices"):
        target = data_dir / f"{name}.csv"
        temporary = target.with_suffix(".csv.tmp")
        frames[name].to_csv(temporary, index=False)
        temporary.replace(target)
    frames["manifest"]["rows"] = {
        name: len(frames[name]) for name in ("races", "runners", "results", "odds",
                                             "starting_prices")}
    manifest = frames["manifest"] | {
        "raw_files": archived,
    }
    (data_dir / "provider_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    load_bundle(data_dir)
    return manifest


def _verify_checkout(home: Path) -> str:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=home,
                              capture_output=True, text=True, check=False).stdout.strip()
    if revision != PINNED_COMMIT:
        raise DataError(f"rpscrape revision {revision or 'unknown'} is not pinned {PINNED_COMMIT}")
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                            cwd=home, capture_output=True, text=True,
                            check=False).stdout.splitlines()
    allowed = {f" M {path}" for path in PATCHED_FILES}
    unexpected = sorted(line for line in status if line not in allowed)
    if unexpected or f" M {REQUIRED_PATCHED_FILE}" not in status:
        detail = unexpected[0] if unexpected else "reviewed patch is missing or staged"
        raise DataError(f"rpscrape checkout has unreviewed changes: {detail}")
    diff = subprocess.run(["git", "diff", "HEAD", "--", *PATCHED_FILES], cwd=home,
                          capture_output=True, check=False).stdout
    patch_sha = hashlib.sha256(diff).hexdigest()
    if patch_sha != EXPECTED_PATCH_SHA256:
        raise DataError("rpscrape safety patch differs from the reviewed bounded-retry patch")
    return patch_sha


def scrape_and_ingest(start: date, end: date, region: str, data_dir: str | Path,
                      checkout: str | Path, clean: bool = False) -> dict:
    home = Path(checkout).resolve()
    script = home / "scripts" / "rpscrape.py"
    python = home / ".venv" / "bin" / "python"
    if not script.exists() or not python.exists():
        raise DataError(f"pinned rpscrape checkout/venv missing at {home}")
    patch_sha = _verify_checkout(home)
    region = region.strip().lower()
    if region not in {"gb", "ire", "both"}:
        raise ValueError("rpscrape region must be gb, ire or both")
    value = start.strftime("%Y/%m/%d")
    if end != start:
        value += "-" + end.strftime("%Y/%m/%d")
    normalized, raw_sources = [], []
    for item in (["gb", "ire"] if region == "both" else [region]):
        command = [str(python), "rpscrape.py", "--date", value, "--region", item,
                   "--type", "flat"]
        if clean:
            command.append("--clean")
        run = subprocess.run(command, cwd=script.parent, capture_output=True, text=True,
                             timeout=7200, check=False)
        if run.returncode != 0:
            raise RuntimeError(f"rpscrape failed for {item}: "
                               f"{(run.stderr or run.stdout)[-2000:]}")
        marker = next((line.split("=", 1)[1] for line in run.stdout.splitlines()
                       if line.startswith("OUTPUT_CSV=")), "")
        if not marker:
            raise RuntimeError(f"rpscrape did not report OUTPUT_CSV for {item}")
        raw_sources.append(Path(marker))
        normalized.append(normalize_csv(marker))
    names = ("races", "runners", "results", "odds", "starting_prices")
    frames = {name: pd.concat([part[name] for part in normalized], ignore_index=True)
              for name in names}
    skipped = {key: sum(part["manifest"]["skipped_races"][key] for part in normalized)
               for key in normalized[0]["manifest"]["skipped_races"]}
    frames["manifest"] = normalized[0]["manifest"] | {
        "rows": {name: len(frames[name]) for name in names}, "skipped_races": skipped,
    }
    manifest = _write_dataset(frames, Path(data_dir), raw_sources)
    manifest["command_scope"] = {"start": start.isoformat(), "end": end.isoformat(),
                                 "region": region, "clean": bool(clean)}
    manifest["rpscrape_safety_patch_sha256"] = patch_sha
    (Path(data_dir) / "provider_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest
