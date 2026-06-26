"""tennis/fetch.py — data layer CLI for the tennis engine.

Source of truth is data/matches.csv, seeded from Jeff Sackmann's free archives
(no API key). Upcoming fixtures (draw.csv) and book prices (odds.csv) are loaded
from manual CSV templates by default — a paid fixtures/odds feed can replace the
template writers later without touching the model.

Usage:
  python -m tennis.fetch --seed 2019 2020 2021 2022 2023 2024 2025
  python -m tennis.fetch --accumulate              # current + previous season
  python -m tennis.fetch --tours atp               # ATP only
  python -m tennis.fetch --draw-template           # write a draw.csv skeleton
  python -m tennis.fetch --odds-template           # write an odds.csv skeleton
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .providers import (
    DATA_DIR,
    accumulate_matches,
)

DRAW_CSV = DATA_DIR / "draw.csv"
ODDS_CSV = DATA_DIR / "odds.csv"

DRAW_COLUMNS = ["tour", "tourney_name", "surface", "best_of", "round",
                "player_a", "player_b", "state", "winner", "score", "match_id"]
ODDS_COLUMNS = ["tour", "surface", "best_of", "player_a", "player_b",
                "odds_a", "odds_b"]


def write_template(path: Path, columns: list[str], sample: dict) -> Path:
    """Write a header-plus-one-example CSV the user fills in by hand. Never
    clobbers an existing file the user may already have populated."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"  {path} already exists — leaving it untouched")
        return path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerow({**{c: "" for c in columns}, **sample})
    print(f"  template → {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch tennis data (matches, draw, odds)")
    ap.add_argument("--seed", nargs="*", type=int, metavar="YEAR",
                    help="Backfill data/matches.csv for the given seasons "
                         "(e.g. --seed 2020 2021 2022 2023 2024 2025) and exit")
    ap.add_argument("--accumulate", action="store_true",
                    help="Append new completed matches (current + previous season)")
    ap.add_argument("--tours", nargs="*", default=["atp", "wta"],
                    choices=["atp", "wta"],
                    help="Tours to fetch (default: both)")
    ap.add_argument("--draw-template", action="store_true",
                    help="Write a blank data/draw.csv to fill in by hand")
    ap.add_argument("--odds-template", action="store_true",
                    help="Write a blank data/odds.csv to fill in by hand")
    args = ap.parse_args()

    if args.draw_template:
        write_template(DRAW_CSV, DRAW_COLUMNS,
                        {"tour": "atp", "tourney_name": "Wimbledon",
                        "surface": "grass", "best_of": 5, "round": "R128",
                        "player_a": "Carlos Alcaraz", "player_b": "Jannik Sinner",
                        "state": "pre"})
        return
    if args.odds_template:
        write_template(ODDS_CSV, ODDS_COLUMNS,
                       {"tour": "atp", "surface": "grass", "best_of": 5,
                        "player_a": "Carlos Alcaraz", "player_b": "Jannik Sinner",
                        "odds_a": "1.85", "odds_b": "2.00"})
        return

    if args.seed is not None or args.accumulate:
        years = args.seed if args.seed else None
        added = accumulate_matches(years=years, tours=tuple(args.tours))
        print(f"Done. {added} new match(es) recorded.")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
