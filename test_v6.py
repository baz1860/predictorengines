#!/usr/bin/env python3
"""Tests for the V6 product/operations layer."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v6 import operations


def test_health_report_has_operator_sections():
    h = operations.health()
    assert h["status"] in {"ok", "warn", "fail", "unknown"}
    for key in ("validation", "freshness", "bankroll", "v5", "backup"):
        assert key in h["sections"]


def test_daily_run_is_preview_and_non_execution_by_default():
    plan = operations.daily_run_plan()
    assert plan["mode"] == "preview"
    assert plan["status"] == "advisory"
    assert len(plan["steps"]) >= 4
    assert all(step["safe"] for step in plan["steps"])


def test_backup_includes_manifest_and_zip():
    old_dir = operations.BACKUP_DIR
    with tempfile.TemporaryDirectory() as tmp:
        operations.BACKUP_DIR = Path(tmp)
        try:
            res = operations.create_backup("test")
            assert res["status"] == "ok"
            assert res["path"].endswith(".zip")
            assert res["files"] >= 1
            assert "included" in res["manifest"]
        finally:
            operations.BACKUP_DIR = old_dir


def test_release_status_lists_required_artifacts():
    rel = operations.release_status()
    keys = {r["artifact"] for r in rel["artifacts"]}
    assert {"v3_suite_ledger", "v4_validation", "v5_registry", "v6_plan"} <= keys


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} V6 tests passed.")


if __name__ == "__main__":
    _run_all()
