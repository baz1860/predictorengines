from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    """Everything under tests/ is a fast offline check unless explicitly gated."""
    for item in items:
        if item.get_closest_marker("gates"):
            continue
        item.add_marker(pytest.mark.fast)
