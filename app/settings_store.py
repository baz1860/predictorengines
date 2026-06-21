"""Suite-level settings, stored locally at data/app_settings.json.

Defaults live in data/app_settings.json. API keys are editable in the Settings
tab but stored separately in data/api_keys.json, which is ignored by git.
"""
from __future__ import annotations

import json
from pathlib import Path

from api_keys import load_keys, save_keys

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "data" / "app_settings.json"

# Odds sources the UI offers a key field for.
ODDS_SOURCES = [
    {"id": "the-odds-api", "label": "The Odds API (major golf outrights / odds)"},
    {"id": "api-football", "label": "API-Football (injuries.py)"},
]

DEFAULTS = {
    "odds_api_keys": {},        # {source_id: key}
    "default_kelly": 0.25,
    "default_model": "blend",
}


def load() -> dict:
    d = dict(DEFAULTS)
    d["odds_api_keys"] = {}
    saved_keys = {}
    if SETTINGS.exists():
        try:
            saved = json.loads(SETTINGS.read_text())
            if isinstance(saved, dict):
                saved_keys = saved.pop("odds_api_keys", {}) or {}
                d.update(saved)
        except Exception:
            pass
    d["odds_api_keys"] = {**saved_keys, **load_keys()}
    return d


def save(patch: dict) -> dict:
    d = load()
    if "odds_api_keys" in patch and isinstance(patch["odds_api_keys"], dict):
        keys = dict(load_keys())
        for k, v in patch["odds_api_keys"].items():
            if v:
                keys[k] = v
            else:
                keys.pop(k, None)   # empty value clears the key
        save_keys(keys)
        d["odds_api_keys"] = load_keys()
    for k in ("default_kelly", "default_model"):
        if k in patch:
            d[k] = patch[k]
    SETTINGS.parent.mkdir(exist_ok=True)
    settings_only = dict(d)
    settings_only["odds_api_keys"] = {}
    SETTINGS.write_text(json.dumps(settings_only, indent=2))
    return d


def odds_api_key(source_id: str) -> str | None:
    return load().get("odds_api_keys", {}).get(source_id)


def public_view() -> dict:
    """Settings for the UI. API keys are masked — never send raw keys back."""
    d = load()
    keys = d.get("odds_api_keys", {})
    return {
        "sources": ODDS_SOURCES,
        "odds_api_keys_set": {s["id"]: bool(keys.get(s["id"])) for s in ODDS_SOURCES},
        "odds_api_keys_masked": {
            s["id"]: (("…" + keys[s["id"]][-4:]) if keys.get(s["id"]) else "")
            for s in ODDS_SOURCES
        },
        "default_kelly": d.get("default_kelly", 0.25),
        "default_model": d.get("default_model", "blend"),
    }
