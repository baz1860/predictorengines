#!/usr/bin/env python3
"""Contract tests for The Racing API canonical adapter (no network required)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from horse_racing.providers.the_racing_api import archive_payload, normalize
from horse_racing.schema import load_bundle
from horse_racing.validate import walk_forward


def main() -> int:
    card = {
        "race_id": "rac_1", "course": "Ascot", "course_id": "crs_52",
        "date": "2026-06-20", "off_dt": "2026-06-20T15:00:00+01:00",
        "race_name": "Test Handicap", "distance": "1207", "distance_f": "6.0",
        "region": "GB", "race_class": "Class 3", "type": "Flat",
        "going": "Good", "surface": "Turf",
        "runners": [
            {"horse_id": "hrs_1", "horse": "Alpha", "trainer_id": "trn_1",
             "trainer": "Trainer A", "jockey_id": "jky_1", "jockey": "Jockey A",
             "draw": "1", "age": "4", "lbs": "126", "ofr": "92",
             "odds": [{"bookmaker": "Betfair Exchange", "history": [
                 {"changed_at": "2026-06-20T13:40:00+00:00", "decimal": "3.0"}]}]},
            {"horse_id": "hrs_2", "horse": "Beta", "trainer_id": "trn_2",
             "trainer": "Trainer B", "jockey_id": "jky_2", "jockey": "Jockey B",
             "draw": "2", "age": "5", "lbs": "129", "ofr": "90",
             "odds": [{"bookmaker": "Betfair Exchange", "history": [
                 {"changed_at": "2026-06-20T13:41:00+00:00", "decimal": "2.0"}]}]},
        ],
    }
    result = {
        "race_id": "rac_1", "off_dt": "2026-06-20T15:00:00+01:00",
        "runners": [{"horse_id": "hrs_1", "position": "1"},
                    {"horse_id": "hrs_2", "position": "2"}],
    }
    tables = normalize([card], [result], "2026-07-01T00:00:00+00:00")
    assert tables["manifest"]["validation_grade"] == "research_only"
    assert tables["manifest"]["odds_executable"] is False
    assert len(tables["races"]) == 1 and len(tables["runners"]) == 2
    assert len(tables["results"]) == 2 and len(tables["odds"]) == 2
    assert set(tables["runners"]["runner_id"]) == {"rac_1:hrs_1", "rac_1:hrs_2"}
    assert tables["odds"]["available_size"].eq("").all()
    cutoff = pd.Timestamp(tables["races"].iloc[0]["prediction_cutoff"])
    source = pd.Timestamp(tables["races"].iloc[0]["source_updated_at"])
    published = pd.Timestamp(tables["results"].iloc[0]["result_published_at"])
    assert source == cutoff
    assert published > pd.Timestamp(tables["races"].iloc[0]["scheduled_off_utc"])

    bad_surface = dict(card, race_id="rac_bad_surface", surface="Dirt")
    bad_ids = dict(card, race_id="rac_bad_ids",
                   runners=[dict(card["runners"][0], jockey_id=""), card["runners"][1]])
    filtered = normalize([bad_surface, bad_ids], [], "2026-07-01T00:00:00+00:00")
    assert filtered["races"].empty
    assert filtered["manifest"]["skipped_races"]["unsupported_surface"] == 1
    assert filtered["manifest"]["skipped_races"]["missing_stable_ids"] == 1

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name in ("races", "runners", "results", "odds"):
            tables[name].to_csv(root / f"{name}.csv", index=False)
        bundle = load_bundle(root)
        assert len(bundle.races) == 1 and len(bundle.runners) == 2
        a = archive_payload(root / "raw", "racecards_pro", {"date": "2026-06-20"},
                            {"racecards": [card]}, "2026-07-01T00:00:00+00:00")
        b = archive_payload(root / "raw", "racecards_pro", {"date": "2026-06-20"},
                            {"racecards": [card]}, "2026-07-01T00:00:00+00:00")
        assert a.path == b.path and a.path.exists()

        replay = root / "replay"
        replay.mkdir()
        cards, results = [], []
        for i in range(40):
            day = pd.Timestamp("2025-01-01T14:00:00Z") + pd.Timedelta(days=i)
            runners = []
            result_runners = []
            for j in range(4):
                horse = (i + j) % 8
                runners.append({
                    "horse_id": f"hrs_{horse}", "horse": f"Horse {horse}",
                    "trainer_id": f"trn_{horse % 3}", "trainer": f"Trainer {horse % 3}",
                    "jockey_id": f"jky_{horse % 4}", "jockey": f"Jockey {horse % 4}",
                    "draw": str(j + 1), "age": str(3 + horse % 5),
                    "lbs": str(122 + horse), "ofr": str(100 - horse * 3),
                    "odds": [{"bookmaker": "Betfair Exchange", "history": [{
                        "changed_at": (day - pd.Timedelta(minutes=20)).isoformat(),
                        "decimal": str(2.5 + j),
                    }]}],
                })
                result_runners.append({"horse_id": f"hrs_{horse}",
                                       "position": str(j + 1)})
            rid = f"rac_replay_{i:03d}"
            cards.append(dict(card, race_id=rid, date=str(day.date()),
                              off_dt=day.isoformat(), runners=runners))
            results.append({"race_id": rid, "off_dt": day.isoformat(),
                            "runners": result_runners})
        replay_tables = normalize(cards, results, "2026-07-01T00:00:00Z")
        for name in ("races", "runners", "results", "odds"):
            replay_tables[name].to_csv(replay / f"{name}.csv", index=False)
        (replay / "provider_manifest.json").write_text(
            json.dumps(replay_tables["manifest"]))
        _predictions, report = walk_forward(replay, min_train=30, test_size=5)
        assert report["metrics"]["p_model"]["races"] == 10
        assert report["data_provenance"]["validation_grade"] == "research_only"
    print("horse_racing_provider: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
