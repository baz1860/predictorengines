"""In-app execution of the update.sh flows.

The shell script `update.sh` remains the single source of truth for *what* a
daily update does. This module is the bridge that lets the app run those same
flows from the UI: it launches `update.sh <mode>` as a subprocess, streams the
output line-by-line into an in-memory buffer the frontend can poll, derives
per-step progress from the script's own `== ... ==` headers, and appends a
record to a small run-history file when each run finishes.

Only one run may be in flight at a time. State is shared between the worker
thread and API request threads under a single lock; readers always get a copy.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UPDATE_SCRIPT = ROOT / "update.sh"
HISTORY_FILE = DATA / "update_runs.json"

# Cap the live log so a long run can't grow memory without bound. Older lines
# scroll off; the persisted history keeps the summary, not the raw stream.
_MAX_LOG_LINES = 4000
_MAX_HISTORY = 50

# The four flows update.sh understands. "default" (the full daily run) takes no
# mode argument — passing the literal "default" would be read as a sim count.
MODES: dict[str, dict[str, str]] = {
    "morning": {
        "label": "Morning",
        "description": "Results, live feeds, refit, predictions, edge, manifest, dashboard.",
    },
    "prekickoff": {
        "label": "Pre-kickoff",
        "description": "Lineups / availability / odds, squad ratings, edge, manifest.",
    },
    "postmatch": {
        "label": "Post-match",
        "description": "Results & stats, settle bets, CLV, validation gate, dashboard.",
    },
    "default": {
        "label": "Full daily",
        "description": "Complete flow: refresh, settle, refit, predict, simulate, edge, tracker, CLV, validate.",
    },
}

_HEADER_RE = re.compile(r"^==\s*(.+?)\s*==\s*$")
_GATE_FAIL = "VALIDATION GATE FAILED"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _RunState:
    """Mutable state for the current/last run, guarded by `lock`."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.mode: str | None = None
        self.status: str = "idle"          # idle | running | success | failed
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self.exit_code: int | None = None
        self.lines: deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self.line_base: int = 0            # how many lines have scrolled off the deque
        self.steps: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self._proc: subprocess.Popen | None = None

    # -- snapshot (called by API threads) -------------------------------------
    def snapshot(self, since: int = 0) -> dict[str, Any]:
        with self.lock:
            total = self.line_base + len(self.lines)
            start = max(since - self.line_base, 0)
            new_lines = list(self.lines)[start:] if since < total else []
            return {
                "mode": self.mode,
                "mode_label": MODES.get(self.mode or "", {}).get("label", self.mode),
                "status": self.status,
                "running": self.status == "running",
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "exit_code": self.exit_code,
                "steps": [dict(s) for s in self.steps],
                "warnings": list(self.warnings),
                "lines": new_lines,
                "next_offset": total,
            }


_state = _RunState()


def is_running() -> bool:
    with _state.lock:
        return _state.status == "running"


def status(since: int = 0) -> dict[str, Any]:
    return _state.snapshot(since)


def _append_line(text: str) -> None:
    """Record one output line and update step/warning derived state."""
    with _state.lock:
        if len(_state.lines) == _state.lines.maxlen:
            _state.line_base += 1
        _state.lines.append(text)
        if _GATE_FAIL in text:
            _state.warnings.append(text.strip())
        m = _HEADER_RE.match(text.strip())
        if m:
            # A new "== header ==" closes the previous step and opens this one.
            for s in _state.steps:
                if s["status"] == "running":
                    s["status"] = "done"
                    s["ended_at"] = _now()
            _state.steps.append({
                "name": m.group(1),
                "status": "running",
                "started_at": _now(),
                "ended_at": None,
            })


def _finalize(exit_code: int) -> None:
    ok = exit_code == 0
    with _state.lock:
        for s in _state.steps:
            if s["status"] == "running":
                s["status"] = "done" if ok else "failed"
                s["ended_at"] = _now()
        _state.status = "success" if ok else "failed"
        _state.exit_code = exit_code
        _state.ended_at = _now()
        record = {
            "mode": _state.mode,
            "mode_label": MODES.get(_state.mode or "", {}).get("label", _state.mode),
            "status": _state.status,
            "started_at": _state.started_at,
            "ended_at": _state.ended_at,
            "exit_code": exit_code,
            "steps": len(_state.steps),
            "warnings": list(_state.warnings),
        }
    _append_history(record)


def _worker(mode: str, trigger: str) -> None:
    cmd = ["bash", str(UPDATE_SCRIPT)]
    if mode in ("morning", "prekickoff", "postmatch"):
        cmd.append(mode)
    # "default" => no extra arg (full daily flow with the script's own defaults).
    _append_line(f"$ {' '.join(cmd)}  ({trigger})")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except Exception as e:  # pragma: no cover - script missing / not executable
        _append_line(f"!! failed to launch update.sh: {e}")
        _finalize(127)
        return

    with _state.lock:
        _state._proc = proc
    assert proc.stdout is not None
    for raw in proc.stdout:
        _append_line(raw.rstrip("\n"))
    proc.wait()
    with _state.lock:
        _state._proc = None
    _finalize(proc.returncode if proc.returncode is not None else 1)


def start(mode: str, trigger: str = "manual") -> dict[str, Any]:
    """Begin a run. Returns the fresh status, or raises if one is in flight or
    the mode is unknown."""
    if mode not in MODES:
        raise ValueError(f"Unknown update mode: {mode}")
    if not UPDATE_SCRIPT.exists():
        raise ValueError("update.sh not found next to the app")
    with _state.lock:
        if _state.status == "running":
            raise RuntimeError(f"An update is already running ({_state.mode})")
        _state.mode = mode
        _state.status = "running"
        _state.started_at = _now()
        _state.ended_at = None
        _state.exit_code = None
        _state.lines = deque(maxlen=_MAX_LOG_LINES)
        _state.line_base = 0
        _state.steps = []
        _state.warnings = []
    threading.Thread(target=_worker, args=(mode, trigger), daemon=True).start()
    return _state.snapshot()


def modes() -> list[dict[str, str]]:
    return [{"id": k, **v} for k, v in MODES.items()]


# -- run history --------------------------------------------------------------
def _append_history(record: dict[str, Any]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    runs = history()
    runs.insert(0, record)
    runs = runs[:_MAX_HISTORY]
    try:
        HISTORY_FILE.write_text(json.dumps(runs, indent=2))
    except Exception:
        pass


def history() -> list[dict[str, Any]]:
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []
