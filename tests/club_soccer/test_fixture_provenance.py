#!/usr/bin/env python3
"""fixtures.csv records which alias map produced it, and a host whose map is
behind must not overwrite it.

club_soccer/data/.stignore deliberately syncs the generated artifacts, so more
than one host can regenerate the same shared fixtures.csv. A host whose
club_alias_map.json has not yet synced writes pre-merge identities and wins on
mtime; every local check passes, because that host's own map agrees with what
it wrote. The Danish Superliga identity merge was reverted this way on
2026-08-11 and again on 2026-08-12, leaving only a Syncthing conflict file.

`human_verdicts_merged_at` orders the two maps — a digest alone proves they
differ, not which is newer.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from club_soccer import club_identity as CI
from club_soccer import fetch


@pytest.fixture
def fixtures_file(tmp_path):
    path = tmp_path / "fixtures.csv"
    frame = pd.DataFrame([{
        "competition": "Premier League", "season": 2026, "home": "Arsenal",
        "away": "Chelsea", "date": "2026-08-01", "home_goals": 1,
        "away_goals": 0, "status": "FT", "fixture_id": 1, "type": "league",
        "neutral": 0,
    }])
    return path, frame


def _provenance(path):
    return json.loads(fetch._provenance_path(path).read_text())


def test_first_write_stamps_provenance(fixtures_file):
    path, frame = fixtures_file
    fetch.write_fixtures(frame, path=path)
    stamp = _provenance(path)
    assert stamp["rows"] == 1
    assert stamp["host"]
    assert stamp["alias_map_sha256"] == CI.alias_map_fingerprint()["sha256"]


def test_same_map_writes_freely(fixtures_file):
    path, frame = fixtures_file
    fetch.write_fixtures(frame, path=path)
    fetch.write_fixtures(frame, path=path)      # must not raise


def test_older_local_map_is_refused(fixtures_file):
    path, frame = fixtures_file
    fetch.write_fixtures(frame, path=path)
    stamp = _provenance(path)
    stamp.update({
        "alias_map_sha256": "deadbeefdeadbeef",
        "alias_map_updated_at": "2099-01-01T00:00:00+00:00",
        "alias_map_entries": 999,
        "host": "mac-mini.local",
    })
    fetch._provenance_path(path).write_text(json.dumps(stamp))
    with pytest.raises(ValueError, match="NEWER"):
        fetch.write_fixtures(frame, path=path)


def test_newer_local_map_proceeds_and_restamps(fixtures_file):
    """The mirror case must NOT block, or the guard would deadlock the pair."""
    path, frame = fixtures_file
    fetch.write_fixtures(frame, path=path)
    stamp = _provenance(path)
    stamp.update({
        "alias_map_sha256": "0000000000000000",
        "alias_map_updated_at": "2020-01-01T00:00:00+00:00",
    })
    fetch._provenance_path(path).write_text(json.dumps(stamp))
    fetch.write_fixtures(frame, path=path)
    assert _provenance(path)["alias_map_sha256"] == \
        CI.alias_map_fingerprint()["sha256"]


def test_override_env_var_bypasses_the_guard(fixtures_file, monkeypatch):
    path, frame = fixtures_file
    fetch.write_fixtures(frame, path=path)
    stamp = _provenance(path)
    stamp.update({"alias_map_sha256": "deadbeefdeadbeef",
                  "alias_map_updated_at": "2099-01-01T00:00:00+00:00"})
    fetch._provenance_path(path).write_text(json.dumps(stamp))
    monkeypatch.setenv("CLUB_SOCCER_ALLOW_STALE_ALIAS_MAP", "1")
    fetch.write_fixtures(frame, path=path)      # must not raise


def test_missing_provenance_does_not_block_a_first_stamped_write(fixtures_file):
    """Existing deployments have an unstamped fixtures.csv; they must not
    deadlock on upgrade."""
    path, frame = fixtures_file
    frame.to_csv(path, index=False)             # unstamped, as if pre-upgrade
    assert not fetch._provenance_path(path).exists()
    fetch.write_fixtures(frame, path=path)
    assert fetch._provenance_path(path).exists()


def test_provenance_is_synced():
    """A guard nobody else can see is not a guard."""
    stignore = (CI.DATA / ".stignore").read_text().splitlines()
    assert "!/fixtures_provenance.json" in stignore


def test_alias_map_fingerprint_is_stable_and_ordered():
    first = CI.alias_map_fingerprint()
    second = CI.alias_map_fingerprint()
    assert first == second
    assert first["sha256"] and first["updated_at"]
    assert first["entries"] > 0
