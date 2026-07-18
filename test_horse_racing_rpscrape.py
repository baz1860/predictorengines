#!/usr/bin/env python3
"""Offline contract tests for the pinned rpscrape output adapter."""
from __future__ import annotations

import tempfile
import subprocess
import hashlib
from pathlib import Path

import pandas as pd

from horse_racing.providers import rpscrape as R
from horse_racing.schema import DataError, load_bundle


COLUMNS = [
    "date", "region", "course_id", "course", "race_id", "off", "race_name",
    "type", "class", "pattern", "rating_band", "age_band", "sex_rest", "dist",
    "dist_f", "dist_m", "dist_y", "going", "surface", "ran", "num", "pos",
    "draw", "ovr_btn", "btn", "horse_id", "horse", "age", "sex", "lbs", "hg",
    "sp", "dec", "jockey_id", "jockey", "trainer_id", "trainer", "or",
]


def row(horse: int, position: str, *, surface="Turf", jockey_id=None) -> dict:
    return dict(zip(COLUMNS, [
        "2026-07-13", "GB", "3", "Ayr", "923054", "15:00", "Test Handicap",
        "Flat", "Class 6", "", "0-65", "3yo+", "", "6f", "6", "1207", "1320",
        "Good To Firm", surface, "2", str(horse), position, str(horse), "0", "0",
        str(5000 + horse), f"Horse {horse} (GB)", "4", "G", "138", "", "7/2",
        "4.5", str(jockey_id if jockey_id is not None else 9000 + horse),
        f"Jockey {horse}", str(8000 + horse), f"Trainer {horse}", "64",
    ]))


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "raw.csv"
        pd.DataFrame([row(1, "1"), row(2, "2")], columns=COLUMNS).to_csv(source, index=False)
        frames = R.normalize_csv(source, "2026-07-14T20:00:00Z")
        assert frames["manifest"]["validation_grade"] == "research_only"
        assert frames["manifest"]["starting_prices_are_cutoff_odds"] is False
        assert len(frames["races"]) == 1 and len(frames["runners"]) == 2
        assert frames["odds"].empty and len(frames["starting_prices"]) == 2
        assert frames["runners"]["horse_id"].str.startswith("rp:horse:").all()
        assert frames["races"].iloc[0]["scheduled_off_utc"] == "2026-07-13T14:00:00+00:00"
        dataset = root / "dataset"
        dataset.mkdir()
        for name in ("races", "runners", "results", "odds"):
            frames[name].to_csv(dataset / f"{name}.csv", index=False)
        load_bundle(dataset)

        # Refreshing retrospective cards/results must preserve independently
        # ingested odds for canonical race/runner pairs that still exist.
        first_runner = frames["runners"].iloc[0]
        pd.DataFrame([{
            "race_id": first_runner["race_id"], "runner_id": first_runner["runner_id"],
            "market_id": "win", "source": "betfair_historical_basic",
            "decimal_odds": 4.2, "available_size": "",
            "captured_at": "2026-07-13T13:44:00+00:00",
            "field_version": first_runner["field_version"], "market_status": "open",
            "provider_archive_sha256": "a" * 64,
        }]).to_csv(dataset / "odds.csv", index=False)
        R._write_dataset(frames, dataset, [source])
        refreshed = load_bundle(dataset)
        assert len(refreshed.odds) == 1
        assert refreshed.odds.iloc[0]["source"] == "betfair_historical_basic"
        assert refreshed.odds.iloc[0]["provider_archive_sha256"] == "a" * 64

        invalid = root / "invalid.csv"
        pd.DataFrame([row(1, "1", jockey_id=""), row(2, "2")],
                     columns=COLUMNS).to_csv(invalid, index=False)
        rejected = R.normalize_csv(invalid, "2026-07-14T20:00:00Z")
        assert rejected["races"].empty
        assert rejected["manifest"]["skipped_races"]["missing_ids"] == 1

        # Only the reviewed bounded-retry patch may dirty the pinned checkout.
        checkout = root / "checkout"
        (checkout / "scripts" / "utils").mkdir(parents=True)
        race_py = checkout / "scripts" / "utils" / "race.py"
        scrape_py = checkout / "scripts" / "rpscrape.py"
        race_py.write_text("VALUE = 1\n")
        scrape_py.write_text("VALUE = 1\n")
        subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
        subprocess.run(["git", "add", "."], cwd=checkout, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                        "commit", "-qm", "initial"], cwd=checkout, check=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout,
                                capture_output=True, text=True, check=True).stdout.strip()
        race_py.write_text("VALUE = 2\n")
        diff = subprocess.run(["git", "diff", "HEAD", "--", "scripts/utils/race.py"],
                              cwd=checkout, capture_output=True, check=True).stdout
        original_commit, original_patch = R.PINNED_COMMIT, R.EXPECTED_PATCH_SHA256
        try:
            R.PINNED_COMMIT = commit
            R.EXPECTED_PATCH_SHA256 = hashlib.sha256(diff).hexdigest()
            assert R._verify_checkout(checkout) == R.EXPECTED_PATCH_SHA256
            scrape_py.write_text("VALUE = 2\n")
            try:
                R._verify_checkout(checkout)
                raise AssertionError("unreviewed checkout modification should be rejected")
            except DataError:
                pass
        finally:
            R.PINNED_COMMIT, R.EXPECTED_PATCH_SHA256 = original_commit, original_patch
    print("horse_racing_rpscrape: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
