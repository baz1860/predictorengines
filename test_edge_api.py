#!/usr/bin/env python3
"""World Cup edge API selection tests.

Run: python3 test_edge_api.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.worldcup import edge


PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_worldcup_sport_selection_prefers_soccer():
    sports = [
        {"key": "cricket_t20_world_cup_womens", "title": "T20 Women's World Cup"},
        {"key": "soccer_fifa_world_cup", "title": "FIFA World Cup"},
        {"key": "soccer_fifa_world_cup_winner", "title": "FIFA World Cup Winner"},
    ]
    check("prefers FIFA soccer World Cup over cricket World Cup",
          edge._select_worldcup_sport_key(sports) == "soccer_fifa_world_cup")


def test_worldcup_sport_selection_fallback():
    sports = [
        {"key": "cricket_t20_world_cup_womens", "title": "T20 Women's World Cup"},
        {"key": "soccer_other_world_cup", "title": "Other World Cup"},
        {"key": "soccer_other_world_cup_winner", "title": "Other World Cup Winner"},
    ]
    check("fallback still requires soccer match market",
          edge._select_worldcup_sport_key(sports) == "soccer_other_world_cup")


def main():
    print("World Cup edge API tests")
    test_worldcup_sport_selection_prefers_soccer()
    test_worldcup_sport_selection_fallback()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
