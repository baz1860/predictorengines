#!/usr/bin/env python3
"""Offline app-wrapper checks for the free-source golf commands."""

from __future__ import annotations

from app.engines._inproc import ALLOWED_COMMANDS
from app.engines.golf import GolfAdapter
from app.server import EngineRequest, refresh, round_3balls
from app.settings_store import public_view
from golf import engine

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_adapter_advertises_free_source_commands():
    adapter = GolfAdapter()
    caps = adapter.capabilities
    check("golf adapter exposes refresh", "refresh" in caps, str(caps))
    check("golf adapter exposes round 3-balls", "round_3balls" in caps, str(caps))
    check("in-process allowlist permits refresh", "refresh" in ALLOWED_COMMANDS,
          str(ALLOWED_COMMANDS))
    check("in-process allowlist permits round 3-balls", "round_3balls" in ALLOWED_COMMANDS,
          str(ALLOWED_COMMANDS))


def test_server_routes_dispatch_to_adapter():
    old_refresh = engine.COMMANDS.get("refresh")
    old_round = engine.COMMANDS.get("round_3balls")
    try:
        engine.COMMANDS["refresh"] = lambda p: {
            "note": "refresh ok",
            "columns": [],
            "rows": [],
            "provider_rows": {},
            "qa": {},
            "manifest": {},
        }
        engine.COMMANDS["round_3balls"] = lambda p: {
            "note": "round ok",
            "columns": [],
            "rows": [],
        }
        r = refresh(EngineRequest(engine="golf", params={"use_cache": True}))
        check("refresh route dispatches", r.get("note") == "refresh ok", str(r))
        r = round_3balls(EngineRequest(engine="golf", params={"round_no": 1}))
        check("round 3-ball route dispatches", r.get("note") == "round ok", str(r))
    finally:
        if old_refresh is None:
            engine.COMMANDS.pop("refresh", None)
        else:
            engine.COMMANDS["refresh"] = old_refresh
        if old_round is None:
            engine.COMMANDS.pop("round_3balls", None)
        else:
            engine.COMMANDS["round_3balls"] = old_round


def test_settings_are_free_source_first():
    source_ids = [s["id"] for s in public_view()["sources"]]
    check("settings no longer prompts for datagolf", "datagolf" not in source_ids,
          str(source_ids))
    check("settings still offers The Odds API", "the-odds-api" in source_ids,
          str(source_ids))


def main():
    print("Golf app-wrapper tests")
    test_adapter_advertises_free_source_commands()
    test_server_routes_dispatch_to_adapter()
    test_settings_are_free_source_first()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
