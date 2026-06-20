"""Built-in scheduler for the update flows.

While the app is open, a single background thread wakes once a minute, and for
every enabled schedule entry whose `time` (local HH:MM) has just arrived, it
kicks off the matching update run via `runner.start`. Each entry remembers the
date it last fired so a run happens at most once per day per entry, and a missed
minute (app asleep, busy) still fires later the same day via a catch-up window.

Config lives in data/update_schedule.json:
    {
      "entries": [
        {"id": "...", "mode": "morning", "time": "07:30", "enabled": true,
         "last_fired": "2026-06-20"}
      ]
    }
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from . import runner

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCHEDULE_FILE = DATA / "update_schedule.json"

_lock = threading.Lock()
_thread: threading.Thread | None = None
# If the app was closed at the exact minute, still fire within this many minutes.
_CATCHUP_MINUTES = 30


def _read() -> dict[str, Any]:
    if SCHEDULE_FILE.exists():
        try:
            data = json.loads(SCHEDULE_FILE.read_text())
            if isinstance(data, dict) and isinstance(data.get("entries"), list):
                return data
        except Exception:
            pass
    return {"entries": []}


def _write(data: dict[str, Any]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    SCHEDULE_FILE.write_text(json.dumps(data, indent=2))


def get_schedule() -> dict[str, Any]:
    with _lock:
        return _read()


def _valid_time(t: str) -> bool:
    try:
        datetime.strptime(t, "%H:%M")
        return True
    except Exception:
        return False


def save_schedule(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace the entry list. Validates mode and time; preserves last_fired
    for entries the client echoes back with an id."""
    existing = {e.get("id"): e for e in _read().get("entries", [])}
    cleaned: list[dict[str, Any]] = []
    for e in entries or []:
        mode = e.get("mode")
        time_str = e.get("time", "")
        if mode not in runner.MODES or not _valid_time(time_str):
            continue
        eid = e.get("id") or uuid.uuid4().hex[:8]
        cleaned.append({
            "id": eid,
            "mode": mode,
            "time": time_str,
            "enabled": bool(e.get("enabled", True)),
            "last_fired": existing.get(eid, {}).get("last_fired"),
        })
    data = {"entries": cleaned}
    with _lock:
        _write(data)
    return data


def _tick() -> None:
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    with _lock:
        data = _read()
        changed = False
        for e in data.get("entries", []):
            if not e.get("enabled"):
                continue
            if e.get("last_fired") == today:
                continue
            if not _valid_time(e.get("time", "")):
                continue
            target = datetime.strptime(e["time"], "%H:%M").replace(
                year=now.year, month=now.month, day=now.day)
            delta_min = (now - target).total_seconds() / 60.0
            if 0 <= delta_min <= _CATCHUP_MINUTES and not runner.is_running():
                try:
                    runner.start(e["mode"], trigger=f"scheduled {e['time']}")
                    e["last_fired"] = today
                    changed = True
                except Exception:
                    # Another run in flight, or launch failed; retry next tick.
                    pass
        if changed:
            _write(data)


def _loop() -> None:
    import time
    while True:
        try:
            _tick()
        except Exception:
            pass
        time.sleep(60)


def start_scheduler() -> None:
    """Idempotently start the background thread (called on server startup)."""
    global _thread
    with _lock:
        if _thread and _thread.is_alive():
            return
        _thread = threading.Thread(target=_loop, daemon=True, name="update-scheduler")
        _thread.start()
