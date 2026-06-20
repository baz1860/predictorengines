"""Local API key lookup.

Keys live in data/api_keys.json, which is ignored by git. Environment variables
still win, and explicit CLI flags should keep winning over this helper.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEY_FILE = ROOT / "data" / "api_keys.json"

ALIASES = {
    "odds": "the-odds-api",
    "odds-api": "the-odds-api",
    "the_odds_api": "the-odds-api",
    "api_football": "api-football",
    "football": "api-football",
    "dg": "datagolf",
    "data-golf": "datagolf",
    # BSD (Bzzoiro Sports Data) — free football API replacement for api-football
    "bsd": "bsd",
    "bzzoiro": "bsd",
    "bzzoiro_sports": "bsd",
    # football-data.org — free REST API for major European competitions
    "fdorg": "football-data-org",
    "football_data_org": "football-data-org",
    "footballdata": "football-data-org",
}


def _normalise(source: str) -> str:
    key = str(source).strip()
    return ALIASES.get(key, key)


def load_keys(path: Path = KEY_FILE) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        if isinstance(v, str) and v.strip():
            out[_normalise(k)] = v.strip()
    return out


def save_keys(keys: dict[str, str], path: Path = KEY_FILE) -> dict[str, str]:
    clean = {_normalise(k): str(v).strip() for k, v in keys.items()
             if str(v).strip()}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(clean, indent=2) + "\n")
    # Owner-only: the key file is a secret. Best-effort — chmod is a no-op /
    # unsupported on some filesystems and on Windows, so never fail the save.
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return clean


def get_key(source: str, env: str | None = None, path: Path = KEY_FILE) -> str:
    if env:
        val = os.environ.get(env, "").strip()
        if val:
            return val
    return load_keys(path).get(_normalise(source), "")
