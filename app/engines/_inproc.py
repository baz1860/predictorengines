"""In-process engine dispatch (refactor Phase 4).

Replaces the per-engine subprocess (`_subprocess.run_engine`) now that the sport
packages are importable without module-name collisions. Preserves the subprocess's
security guarantees in-process:

  * only an allowlisted command may run;
  * any error is **redacted** (stored/env secret values masked) before it surfaces,
    so an engine exception can't leak a key into the UI;
  * the payload is validated as strict, finite JSON (no NaN/Inf).

What it deliberately drops vs the subprocess: the curated-environment isolation
(`safe_runner_env`). The engines are first-party code and the worldcup engine already
ran in-process, so this makes the others consistent rather than introducing a new risk.
"""
from __future__ import annotations

from typing import Callable

from app.security import collect_secrets, redact
from contracts import ContractError, assert_finite_json

# Same allowlist the subprocess runner enforced.
ALLOWED_COMMANDS = {
    "schema",
    "refresh",
    "predict",
    "simulate",
    "edge",
    "edge_template",
    "round_3balls",
}


def run_inprocess(commands: dict[str, Callable[[dict], dict]], command: str,
                  params: dict | None = None,
                  allowed: set[str] | None = None) -> dict:
    allowed = allowed if allowed is not None else ALLOWED_COMMANDS
    if command not in allowed:
        raise ValueError(f"Unknown engine command: {command!r}")
    if command not in commands:
        raise ValueError(f"Engine does not implement: {command!r}")
    try:
        result = commands[command](params or {})
    except ValueError as e:
        # User-facing validation error (e.g. "Pick two clubs.") — redact in case a
        # value carries a secret, then re-raise as the same type the UI expects.
        raise ValueError(redact(str(e), collect_secrets()))
    except Exception as e:  # noqa: BLE001 — mirror the runner's catch-all
        raise ValueError(redact(f"{type(e).__name__}: {e}", collect_secrets()))
    try:
        assert_finite_json(result)
    except ContractError as e:
        raise RuntimeError(
            f"Engine emitted invalid JSON: {redact(str(e), collect_secrets())}") from None
    return result
