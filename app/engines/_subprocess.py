"""Helper to invoke an engine runner in an isolated subprocess.

Each engine lives in its own folder with flat module names that collide across
engines, so we run its runner with cwd + PYTHONPATH pointed at the engine dir.
Results come back as JSON on the final stdout line.

Hardening (V3 M2):
  * only a fixed allowlist of commands may be launched;
  * the subprocess gets a curated environment (see security.safe_runner_env),
    not the whole parent environment;
  * stdout is parsed as strict, finite JSON — NaN/Inf or noisy output is a hard
    error rather than a silently-poisoned payload;
  * stderr snippets and engine error strings are redacted before they surface.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..security import collect_secrets, redact, safe_runner_env
from contracts import ContractError, assert_finite_json

# Every command any runner is allowed to receive. Reject anything else before
# we ever spawn a process.
ALLOWED_COMMANDS = {"schema", "predict", "simulate", "edge", "edge_template"}


def run_engine(engine_dir: Path, runner: Path, command: str,
               params: dict | None = None, timeout: int = 180,
               allowed_commands: set[str] | None = None) -> dict:
    allowed = allowed_commands or ALLOWED_COMMANDS
    if command not in allowed:
        raise ValueError(f"Unknown runner command: {command!r}")
    if not runner.exists():
        raise RuntimeError(f"Runner not found: {runner.name}")

    env = safe_runner_env({
        "PYTHONPATH": str(engine_dir) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    })
    proc = subprocess.run(
        [sys.executable, str(runner), command],
        input=json.dumps(params or {}),
        cwd=str(engine_dir), env=env,
        capture_output=True, text=True, timeout=timeout)

    secrets = collect_secrets()
    out = (proc.stdout or "").strip()
    try:
        data = json.loads(out.splitlines()[-1]) if out else {}
    except (json.JSONDecodeError, IndexError):
        raise RuntimeError(
            f"Engine runner failed (exit {proc.returncode}).\n"
            f"stderr: {redact((proc.stderr or '').strip()[:500], secrets)}")
    if isinstance(data, dict) and "error" in data:
        raise ValueError(redact(str(data["error"]), secrets))
    try:
        assert_finite_json(data)
    except ContractError as e:
        raise RuntimeError(f"Engine runner emitted invalid JSON: {e}")
    return data
