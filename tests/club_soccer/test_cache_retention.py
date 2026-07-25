from __future__ import annotations

from club_soccer.cache_retention import Policy, prune


def test_prune_removes_expired_files(tmp_path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    now = 2_000_000_000.0
    old.touch()
    new.touch()
    import os
    os.utime(old, (now - 11 * 86400, now - 11 * 86400))
    os.utime(new, (now - 1 * 86400, now - 1 * 86400))
    result = prune(Policy(tmp_path, max_age_days=10, max_bytes=100), now=now)
    assert result["files_removed"] == 1
    assert not old.exists()
    assert new.exists()


def test_prune_enforces_size_quota_oldest_first(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"a" * 8)
    second.write_bytes(b"b" * 8)
    import os
    os.utime(first, (100, 100))
    os.utime(second, (200, 200))
    result = prune(
        Policy(tmp_path, max_age_days=100_000, max_bytes=8),
        now=300,
    )
    assert result["bytes_remaining"] == 8
    assert not first.exists()
    assert second.exists()
