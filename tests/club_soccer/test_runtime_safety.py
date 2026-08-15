from __future__ import annotations

import pytest

from club_soccer import runtime_safety as S


def test_writer_guard_allows_declared_host(monkeypatch):
    monkeypatch.setenv("CLUB_SOCCER_WRITER_HOST", "bingo.local")
    monkeypatch.setattr(S.socket, "gethostname", lambda: "BINGO.LOCAL")
    S.assert_writer_host()


def test_writer_guard_rejects_receive_only_host(monkeypatch):
    monkeypatch.setenv("CLUB_SOCCER_WRITER_HOST", "bingo.local")
    monkeypatch.setattr(S.socket, "gethostname", lambda: "lucky.local")
    with pytest.raises(RuntimeError, match="single-writer"):
        S.assert_writer_host()


def test_conflict_report_is_non_destructive(tmp_path):
    conflict = tmp_path / "ledger.sync-conflict-20260815.csv"
    conflict.write_text("preserve me")
    report = S.conflict_report(tmp_path)
    assert report["ok"] is False
    assert report["conflict_count"] == 1
    assert report["conflicts"][0]["name"] == conflict.name
    assert conflict.read_text() == "preserve me"
