from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())


@pytest.mark.gates
def test_validation_gates():
    sims = os.environ.get("VALIDATION_SIMS", "4000")
    proc = subprocess.run(
        [sys.executable, "validate_all.py", "--gate", "--sims", sims],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
