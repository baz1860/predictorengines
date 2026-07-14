#!/usr/bin/env python3
"""Offline correctness tests for the horse-racing V1 engine."""
from __future__ import annotations

import sys
import tempfile
import subprocess
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts import validate_edge_rows, validate_prediction
from app.engines._inproc import run_inprocess
from horse_racing.edge import price_race
from horse_racing.features import FEATURES, build_feature_frame
from horse_racing.model import (fit, git_provenance, load_artifact, predict_race,
                                save_artifact)
from horse_racing.schema import DataError, load_bundle
from horse_racing.validate import walk_forward


def _synthetic(root: Path, n_completed: int = 72) -> str:
    rng = np.random.default_rng(11)
    horses = [f"h{i:02d}" for i in range(12)]
    ability = {h: 1.8 - 0.32 * i for i, h in enumerate(horses)}
    races, runners, results, odds = [], [], [], []
    for race_n in range(n_completed + 1):
        off = pd.Timestamp("2024-01-01T14:00:00Z") + pd.Timedelta(days=race_n)
        rid = f"race-{race_n:03d}"
        field = [horses[(race_n * 2 + j) % len(horses)] for j in range(6)]
        races.append({
            "race_id": rid, "meeting_date": str(off.date()),
            "scheduled_off_utc": off.isoformat(), "course_id": f"c{race_n % 3}",
            "course_name": f"Course {race_n % 3}", "jurisdiction": "GB",
            "code": "flat", "surface": "turf" if race_n % 2 == 0 else "all_weather",
            "going": "good", "distance_metres": 1200 + 400 * (race_n % 3),
            "race_class": 3, "handicap_flag": 1, "prediction_cutoff": "",
            "source_updated_at": (off - pd.Timedelta(days=1)).isoformat(),
            "record_version": 1,
        })
        noise = rng.normal(0, 0.55, len(field))
        scores = np.array([ability[h] for h in field]) + noise
        order = np.argsort(-scores)
        finish = {field[int(idx)]: pos + 1 for pos, idx in enumerate(order)}
        true_logits = np.array([ability[h] for h in field])
        true_p = np.exp(true_logits - np.logaddexp.reduce(true_logits))
        for j, horse in enumerate(field):
            runner_id = f"{rid}-{horse}"
            runners.append({
                "race_id": rid, "runner_id": runner_id, "horse_id": horse,
                "horse_name": f"Horse {horse[1:]}", "trainer_id": f"t-{horse}",
                "trainer_name": f"Trainer {horse}", "jockey_id": f"j-{horse}",
                "jockey_name": f"Jockey {horse}", "draw": j + 1,
                "age": 3 + (j % 3), "weight_carried_kg": 56 + j * 0.6,
                "official_rating": 72 + ability[horse] * 9 + rng.normal(0, 1.5),
                "declared_status": "declared", "non_runner_status": "",
                "field_version": 1,
                "source_updated_at": (off - pd.Timedelta(hours=4)).isoformat(),
                "record_version": 1,
            })
            odds.append({
                "race_id": rid, "runner_id": runner_id, "market_id": "win",
                "source": "testbook", "decimal_odds": round(1.0 / (true_p[j] * 1.08), 4),
                "available_size": 100, "captured_at": (off - pd.Timedelta(minutes=20)).isoformat(),
                "field_version": 1, "market_status": "open",
            })
            if race_n < n_completed:
                results.append({
                    "race_id": rid, "runner_id": runner_id,
                    "finish_position": finish[horse], "completion_status": "finished",
                    "result_published_at": (off + pd.Timedelta(minutes=8)).isoformat(),
                    "result_updated_at": "", "record_version": 1,
                })
    pd.DataFrame(races).to_csv(root / "races.csv", index=False)
    pd.DataFrame(runners).to_csv(root / "runners.csv", index=False)
    pd.DataFrame(results).to_csv(root / "results.csv", index=False)
    pd.DataFrame(odds).to_csv(root / "odds.csv", index=False)
    return f"race-{n_completed:03d}"


def _correction_replay(root: Path) -> None:
    base = load_bundle(root)
    before_r1 = build_feature_frame(base, ["race-001"], include_labels=False)
    before_r2 = build_feature_frame(base, ["race-002"], include_labels=False)
    results = pd.read_csv(root / "results.csv", dtype=str, keep_default_na=False)
    original = results[results["race_id"] == "race-000"].copy()
    positions = pd.to_numeric(original["finish_position"])
    first = int(positions.idxmin())
    last = int(positions.idxmax())
    original.loc[first, "finish_position"] = str(int(positions.max()))
    original.loc[last, "finish_position"] = "1"
    original["record_version"] = "2"
    original["result_updated_at"] = "2024-01-02T15:00:00+00:00"
    pd.concat([results, original], ignore_index=True).to_csv(root / "results.csv", index=False)
    corrected = load_bundle(root)
    after_r1 = build_feature_frame(corrected, ["race-001"], include_labels=False)
    after_r2 = build_feature_frame(corrected, ["race-002"], include_labels=False)
    keys = ["runner_id", *FEATURES]
    pd.testing.assert_frame_equal(before_r1[keys].reset_index(drop=True),
                                  after_r1[keys].reset_index(drop=True))
    common = before_r2.merge(after_r2, on="runner_id", suffixes=("_before", "_after"))
    changed = any(not np.allclose(common[f"{f}_before"], common[f"{f}_after"])
                  for f in FEATURES)
    assert changed, "a published correction should affect later, not earlier, race features"


def _nonmonotone_cutoff_replay(root: Path) -> None:
    full = root / "full"
    reference = root / "reference"
    full.mkdir()
    reference.mkdir()
    _synthetic(full, n_completed=12)
    _synthetic(reference, n_completed=12)
    cutoff = "2024-01-10T13:50:00+00:00"
    for data_dir in (full, reference):
        races = pd.read_csv(data_dir / "races.csv")
        races["prediction_cutoff"] = races["prediction_cutoff"].astype("object")
        races.loc[races["race_id"] == "race-011", "prediction_cutoff"] = cutoff
        races.loc[races["race_id"] == "race-011", "source_updated_at"] = \
            "2024-01-09T12:00:00+00:00"
        races.to_csv(data_dir / "races.csv", index=False)
        runners = pd.read_csv(data_dir / "runners.csv")
        runners.loc[runners["race_id"] == "race-011", "source_updated_at"] = \
            "2024-01-10T12:00:00+00:00"
        runners.to_csv(data_dir / "runners.csv", index=False)
    results = pd.read_csv(reference / "results.csv")
    available = pd.to_datetime(results["result_published_at"], utc=True) <= pd.Timestamp(cutoff)
    results[available].to_csv(reference / "results.csv", index=False)
    observed = build_feature_frame(load_bundle(full), ["race-011"], include_labels=False)
    expected = build_feature_frame(load_bundle(reference), ["race-011"], include_labels=False)
    keys = ["runner_id", *FEATURES]
    pd.testing.assert_frame_equal(observed[keys].reset_index(drop=True),
                                  expected[keys].reset_index(drop=True))


def _git_dirty_provenance(root: Path) -> None:
    repo = root / "git-repo"
    (repo / "horse_racing").mkdir(parents=True)
    source = repo / "horse_racing" / "model.py"
    source.write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "horse_racing/model.py"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "initial"], cwd=repo, check=True)
    assert git_provenance(repo)["dirty"] is False
    source.write_text("VALUE = 2\n")
    assert git_provenance(repo)["dirty"] is True


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        current = _synthetic(root)
        bundle = load_bundle(root)
        artifact = fit(bundle=bundle, min_races=30)
        save_artifact(artifact, root / "model_params.json")
        assert artifact["scope"]["jurisdictions"] == ["GB", "IE"]
        assert artifact["code_sha256"] and artifact["input_checksums"]["results"]
        assert isinstance(artifact["git_dirty"], bool)
        mismatched = dict(artifact)
        mismatched["code_sha256"] = "0" * 64
        mismatch_path = root / "mismatched_model.json"
        mismatch_path.write_text(json.dumps(mismatched))
        try:
            load_artifact(mismatch_path)
            raise AssertionError("code-mismatched model artifact should be rejected")
        except DataError:
            pass

        pred1 = predict_race(current, bundle=bundle, artifact=artifact)
        pred2 = predict_race(current, bundle=bundle, artifact=artifact)
        assert len(pred1) == 6
        assert np.isclose(pred1["p_model"].sum(), 1.0, atol=1e-12)
        assert (pred1["p_model"] > 0).all() and np.isfinite(pred1["fair_odds"]).all()
        pd.testing.assert_frame_equal(pred1, pred2)

        edge = price_race(current, bundle=bundle, artifact=artifact, source="testbook")
        assert edge["board_complete"].all()
        assert np.isclose(edge["p_market"].sum(), 1.0, atol=1e-12)
        assert (edge["stake_gbp"] == 0).all(), "V1 staking must remain disabled"

        # A quote after cutoff must not replace the executable cutoff quote.
        odds = pd.read_csv(root / "odds.csv")
        late = odds[(odds["race_id"] == current)].iloc[[0]].copy()
        old_odds = float(late.iloc[0]["decimal_odds"])
        late["decimal_odds"] = 99.0
        late["captured_at"] = "2024-03-13T13:50:00+00:00"  # off 14:00, cutoff 13:45
        pd.concat([odds, late], ignore_index=True).to_csv(root / "odds.csv", index=False)
        bundle_late = load_bundle(root)
        edge_late = price_race(current, bundle=bundle_late, artifact=artifact,
                               source="testbook")
        selected = edge_late[edge_late["runner_id"] == late.iloc[0]["runner_id"]].iloc[0]
        assert np.isclose(float(selected["odds"]), old_odds)

        incomplete_root = root / "incomplete"
        incomplete_root.mkdir()
        _synthetic(incomplete_root)
        incomplete_odds = pd.read_csv(incomplete_root / "odds.csv")
        drop_idx = incomplete_odds[incomplete_odds["race_id"] == current].index[0]
        incomplete_odds.drop(index=drop_idx).to_csv(incomplete_root / "odds.csv", index=False)
        incomplete = price_race(current, bundle=load_bundle(incomplete_root), artifact=artifact,
                                source="testbook")
        assert not incomplete["board_complete"].any()
        assert not incomplete["recommended"].any()

        suspended_root = root / "suspended"
        suspended_root.mkdir()
        _synthetic(suspended_root)
        suspended_odds = pd.read_csv(suspended_root / "odds.csv")
        suspended = suspended_odds[suspended_odds["race_id"] == current].copy()
        suspended["captured_at"] = "2024-03-13T13:44:00+00:00"
        suspended["market_status"] = "suspended"
        pd.concat([suspended_odds, suspended], ignore_index=True).to_csv(
            suspended_root / "odds.csv", index=False)
        try:
            price_race(current, bundle=load_bundle(suspended_root), artifact=artifact,
                       source="testbook")
            raise AssertionError("suspended latest board should not fall back to old quotes")
        except DataError:
            pass

        unknown_size_root = root / "unknown-size"
        unknown_size_root.mkdir()
        _synthetic(unknown_size_root)
        unknown_odds = pd.read_csv(unknown_size_root / "odds.csv")
        unknown_odds.loc[unknown_odds["race_id"] == current, "available_size"] = float("nan")
        unknown_odds.to_csv(unknown_size_root / "odds.csv", index=False)
        unknown_size = price_race(current, bundle=load_bundle(unknown_size_root),
                                  artifact=artifact, source="testbook")
        assert not unknown_size["board_complete"].any()
        assert not unknown_size["recommended"].any()

        infinite_size_root = root / "infinite-size"
        infinite_size_root.mkdir()
        _synthetic(infinite_size_root)
        infinite_odds = pd.read_csv(infinite_size_root / "odds.csv")
        infinite_odds["available_size"] = infinite_odds["available_size"].astype(float)
        infinite_odds.loc[infinite_odds["race_id"] == current, "available_size"] = \
            float("inf")
        infinite_odds.to_csv(infinite_size_root / "odds.csv", index=False)
        infinite_size = price_race(current, bundle=load_bundle(infinite_size_root),
                                   artifact=artifact, source="testbook")
        assert not infinite_size["board_complete"].any()
        assert not infinite_size["recommended"].any()

        _preds, report = walk_forward(root, min_train=30, test_size=14)
        assert report["metrics"]["p_model"]["races"] >= 35
        assert report["metrics"]["p_model"]["log_loss"] \
            < report["metrics"]["p_uniform"]["log_loss"]
        assert report["metrics"]["p_market"]["races"] >= 35
        assert report["paired_logloss"]["model_minus_uniform"]["races"] >= 35

        # Use a separate copy because this deliberately appends a correction.
        correction_root = root / "correction"
        correction_root.mkdir()
        _synthetic(correction_root, n_completed=8)
        _correction_replay(correction_root)

        cutoff_root = root / "nonmonotone"
        cutoff_root.mkdir()
        _nonmonotone_cutoff_replay(cutoff_root)

        leak_root = root / "self-result-leak"
        leak_root.mkdir()
        _synthetic(leak_root, n_completed=40)
        leak_results = pd.read_csv(leak_root / "results.csv")
        leak_results[["result_published_at", "result_updated_at"]] = \
            leak_results[["result_published_at", "result_updated_at"]].astype("object")
        leak_results.loc[leak_results["race_id"] == "race-020",
                         ["result_published_at", "result_updated_at"]] = \
            ["2024-01-21T13:30:00+00:00", "2024-01-21T13:30:00+00:00"]
        leak_results.to_csv(leak_root / "results.csv", index=False)
        try:
            build_feature_frame(load_bundle(leak_root), ["race-020"])
            raise AssertionError("self-result available before cutoff should be rejected")
        except DataError:
            pass
        try:
            fit(data_dir=leak_root, min_races=30)
            raise AssertionError("fit should reject a pre-cutoff self-result")
        except DataError:
            pass

        # Engine payloads satisfy the shared contracts when pointed at the test
        # data through narrowly scoped module constants.
        import horse_racing.engine as he
        import horse_racing.model as hm
        import horse_racing.schema as hs
        old_data, old_artifact = hs.DATA_DIR, hm.ARTIFACT_PATH
        try:
            hs.DATA_DIR = root
            hm.ARTIFACT_PATH = root / "model_params.json"
            payload = he.cmd_predict({"race_id": current})
            validate_prediction(payload)
            edge_payload = he.cmd_edge({"race_id": current, "source": "testbook"})
            validate_edge_rows(edge_payload["rows"])
            hs.DATA_DIR = incomplete_root
            partial_payload = run_inprocess(
                he.COMMANDS, "edge", {"race_id": current, "source": "testbook"})
            validate_edge_rows(partial_payload["rows"])
            assert not any(row["recommended"] for row in partial_payload["rows"])
            assert any(row["odds"] is None for row in partial_payload["rows"])
        finally:
            hs.DATA_DIR, hm.ARTIFACT_PATH = old_data, old_artifact

        _git_dirty_provenance(root)

        # A race metadata row written after cutoff is rejected, not backfilled.
        bad_races = pd.read_csv(root / "races.csv")
        bad_races.loc[bad_races["race_id"] == current, "source_updated_at"] = \
            "2024-03-13T13:55:00+00:00"
        bad_races.to_csv(root / "races.csv", index=False)
        try:
            build_feature_frame(load_bundle(root), [current], include_labels=False)
            raise AssertionError("post-cutoff race metadata should be rejected")
        except DataError:
            pass

        bad_races.loc[bad_races["race_id"] == current, "source_updated_at"] = \
            "2024-03-12T13:00:00+00:00"
        bad_races.loc[bad_races["race_id"] == current, "jurisdiction"] = "US"
        bad_races.to_csv(root / "races.csv", index=False)
        try:
            load_bundle(root)
            raise AssertionError("out-of-scope jurisdiction should be rejected")
        except DataError:
            pass

    print("horse_racing: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
