"""Security helpers for the suite (V3 M2).

Two concerns, both about reducing attack surface without changing model output:

  * `redact()`        – scrub known API-key values and key-looking tokens out of
                        any text before it reaches a log, API response, or error.
  * `safe_get()`      – thin requests wrapper with a timeout, provider label, and
                        redacted errors for engine network fetches.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

# API-key environment variable names the engines understand. Passed through so
# env-provided key values are treated as secrets by collect_secrets().
_KEY_ENV_VARS = {"THE_ODDS_API_KEY", "ODDS_API_KEY", "API_FOOTBALL_KEY",
                 "DG_API_KEY", "DATAGOLF_KEY"}

# A token that looks like an API key: 20+ chars of key alphabet, no spaces.
_KEY_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{20,}")
_REDACTION = "***"


def collect_secrets() -> list[str]:
    """Every concrete secret value we know about: stored keys + env key values."""
    secrets: set[str] = set()
    try:
        from api_keys import load_keys
        secrets.update(v for v in load_keys().values() if v)
    except Exception:
        pass
    for name in _KEY_ENV_VARS:
        v = os.environ.get(name, "").strip()
        if v:
            secrets.add(v)
    return [s for s in secrets if len(s) >= 4]


def redact(text: str | None, secrets: Iterable[str] | None = None) -> str:
    """Scrub known secret values, then mask any remaining key-looking tokens."""
    if not text:
        return text or ""
    out = str(text)
    for s in (secrets if secrets is not None else collect_secrets()):
        if s and len(s) >= 4:
            out = out.replace(s, _REDACTION)
    # Catch unknown keys too: any long opaque token becomes ***.
    out = _KEY_TOKEN_RE.sub(_REDACTION, out)
    return out


def safe_get(url: str, *, provider: str, timeout: float = 15.0, **kwargs):
    """GET with a hard timeout and provider-labelled, redacted errors.

    Engine network fetches should funnel through this so a failure never leaks a
    key embedded in the URL/params and always names the provider.
    """
    import requests
    try:
        resp = requests.get(url, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp
    except Exception as e:  # noqa: BLE001 - re-raised with a clean message
        raise RuntimeError(f"{provider} request failed: {redact(str(e))}") from None
