"""In-app command console.

The app already runs its own update flows as subprocesses (see ``v6/runner.py``).
This module generalises that idea into a full terminal: it runs *any* command the
user types in a single persistent ``bash`` session rooted at the project folder,
streams the output into a per-command buffer the frontend can poll, and lets the
user stop a running command.

Design notes:

* One persistent shell means ``cd`` and environment variables persist between
  commands, exactly like a real terminal.
* Only one command runs at a time (a terminal is sequential); a second command
  queues behind the first via ``run_lock``.
* Each command is its own object with its own line buffer, so the UI can render
  every command in its own card and poll it incrementally (``since`` / ``next_offset``)
  just like the Ops run log.
* Stop kills the whole shell process group, so long-running or hung commands
  (and anything they spawned) are terminated; the shell is recreated lazily on
  the next command.

This is deliberately an arbitrary-command surface — that is the feature. It is
gated by :data:`ENABLED` (disable by setting ``SP_CONSOLE=0``) and the server
only binds to localhost.
"""
from __future__ import annotations

import itertools
import os
import re
import secrets
import signal
import subprocess
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Feature gate. On by default; set SP_CONSOLE=0 to disable the routes.
ENABLED = os.environ.get("SP_CONSOLE", "1") != "0"

# Bound memory: cap lines kept per command and total commands remembered.
_MAX_LINES = 5000
_MAX_COMMANDS = 100

# Strip ANSI colour/cursor sequences so the browser renders clean text.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# Per-boot marker used to detect a command's end and read back its exit code.
# Randomised so ordinary command output can't accidentally match it.
_SENTINEL = "__SPCONSOLE_DONE_%s__" % secrets.token_hex(8)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _int_or_none(s: str):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


class _Command:
    """One command execution and its output buffer, guarded by the parent lock."""

    def __init__(self, cid: str, cmd: str) -> None:
        self.id = cid
        self.cmd = cmd
        self.status = "running"          # running | done | stopped | error
        self.exit_code: int | None = None
        self.started_at = _now()
        self.ended_at: str | None = None
        self.lines: deque[str] = deque(maxlen=_MAX_LINES)
        self.line_base = 0               # lines that scrolled off the deque

    def append(self, text: str) -> None:
        if len(self.lines) == self.lines.maxlen:
            self.line_base += 1
        self.lines.append(text)

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        total = self.line_base + len(self.lines)
        start = max(since - self.line_base, 0)
        new_lines = list(self.lines)[start:] if since < total else []
        return {
            "id": self.id,
            "cmd": self.cmd,
            "status": self.status,
            "running": self.status == "running",
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "lines": new_lines,
            "next_offset": total,
        }

    def full(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cmd": self.cmd,
            "status": self.status,
            "running": self.status == "running",
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "output": "".join(self.lines),
        }


class _Console:
    def __init__(self) -> None:
        self.lock = threading.Lock()        # guards command bookkeeping + shell handle
        self.run_lock = threading.Lock()    # serialises command execution
        self.shell: subprocess.Popen | None = None
        self.commands: deque[_Command] = deque(maxlen=_MAX_COMMANDS)
        self.by_id: dict[str, _Command] = {}
        self._counter = itertools.count(1)
        self.cwd = str(ROOT)

    # -- shell lifecycle -----------------------------------------------------
    def _ensure_shell(self) -> subprocess.Popen:
        if self.shell is None or self.shell.poll() is not None:
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["TERM"] = "dumb"
            self.shell = subprocess.Popen(
                ["bash"],
                cwd=str(ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,   # own process group -> we can kill the tree
            )
        return self.shell

    def _update_cwd(self) -> None:
        sh = self.shell
        if sh is not None and sh.poll() is None and os.path.isdir("/proc"):
            try:
                self.cwd = os.readlink("/proc/%d/cwd" % sh.pid)
            except OSError:
                pass

    # -- running -------------------------------------------------------------
    def start(self, cmd: str) -> dict[str, Any]:
        cmd = (cmd or "").strip()
        cid = str(next(self._counter))
        command = _Command(cid, cmd)
        with self.lock:
            self.commands.append(command)
            self.by_id[cid] = command
            # Forget ids that fell off the deque so the map can't grow unbounded.
            live = {c.id for c in self.commands}
            for k in [k for k in self.by_id if k not in live]:
                del self.by_id[k]
        if not cmd:
            command.status = "done"
            command.exit_code = 0
            command.ended_at = _now()
            return command.snapshot()
        threading.Thread(target=self._worker, args=(command,), daemon=True).start()
        return command.snapshot()

    def _worker(self, command: _Command) -> None:
        with self.run_lock:
            try:
                sh = self._ensure_shell()
                assert sh.stdin is not None and sh.stdout is not None
                # Send the command, then a sentinel line carrying the exit code.
                sh.stdin.write(command.cmd + "\n")
                sh.stdin.write("printf '%s%%s\\n' \"$?\"\n" % _SENTINEL)
                sh.stdin.flush()

                for raw in sh.stdout:
                    if _SENTINEL in raw:
                        pre_text, _, rest = raw.partition(_SENTINEL)
                        if pre_text:
                            command.append(_ANSI_RE.sub("", pre_text))
                        command.exit_code = _int_or_none(rest.strip())
                        command.status = "done"
                        break
                    command.append(_ANSI_RE.sub("", raw))
                else:
                    # Reached EOF without a sentinel -> shell was killed (Stop) or died.
                    command.status = "stopped"
            except Exception as e:                       # pragma: no cover - defensive
                command.append("[console error: %s]\n" % e)
                command.status = "error"
            finally:
                command.ended_at = _now()
                self._update_cwd()

    def stop(self) -> bool:
        with self.lock:
            sh = self.shell
            self.shell = None
        if sh is None or sh.poll() is not None:
            return False
        try:
            os.killpg(os.getpgid(sh.pid), signal.SIGKILL)
        except OSError:
            try:
                sh.kill()
            except OSError:
                pass
        return True

    # -- reading -------------------------------------------------------------
    def poll(self, cid: str = "", since: int = 0) -> dict[str, Any]:
        with self.lock:
            command = self.by_id.get(cid) or (self.commands[-1] if self.commands else None)
            busy = self.run_lock.locked()
            cwd = self.cwd
        if command is None:
            return {"id": None, "status": "idle", "running": False,
                    "lines": [], "next_offset": 0, "cwd": cwd, "busy": busy}
        snap = command.snapshot(since)
        snap["cwd"] = cwd
        snap["busy"] = busy
        return snap

    def history(self, limit: int = 50) -> dict[str, Any]:
        with self.lock:
            cmds = list(self.commands)[-limit:]
            cwd = self.cwd
            busy = self.run_lock.locked()
        return {"commands": [c.full() for c in cmds], "cwd": cwd, "busy": busy}


_console = _Console()


# -- module-level API (thin, like v6.runner) ---------------------------------
def start(cmd: str) -> dict[str, Any]:
    return _console.start(cmd)


def poll(cid: str = "", since: int = 0) -> dict[str, Any]:
    return _console.poll(cid, max(since, 0))


def stop() -> dict[str, Any]:
    return {"stopped": _console.stop()}


def history(limit: int = 50) -> dict[str, Any]:
    return _console.history(limit)
