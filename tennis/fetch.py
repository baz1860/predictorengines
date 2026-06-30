"""tennis/fetch.py — data layer CLI for the tennis engine.

Source of truth is data/matches.csv, seeded from Jeff Sackmann's free archives
(no API key). Upcoming fixtures are pulled by tennis.season from ESPN into
draw.csv. Book prices can be written from a manual template or fetched from
The Odds API into odds.csv.

Usage:
  python -m tennis.fetch --seed 2019 2020 2021 2022 2023 2024 2025
  python -m tennis.fetch --accumulate              # current + previous season
  python -m tennis.fetch --tours atp               # ATP only
  python -m tennis.fetch --draw-template           # write a draw.csv skeleton
  python -m tennis.fetch --odds-template           # write an odds.csv skeleton
  python -m tennis.fetch --odds-api --tours atp --event Wimbledon
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import urllib.parse
import urllib.request
from pathlib import Path

from api_keys import get_key

from .model import fold_name
from .providers import (
    DATA_DIR,
    _infer_surface,
    accumulate_matches,
)

DRAW_CSV = DATA_DIR / "draw.csv"
ODDS_CSV = DATA_DIR / "odds.csv"

DRAW_COLUMNS = ["tour", "tourney_name", "surface", "best_of", "round",
                "player_a", "player_b", "state", "winner", "score", "match_id"]
ODDS_COLUMNS = ["tour", "surface", "best_of", "player_a", "player_b",
                "odds_a", "odds_b"]
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_ODDS_API_KEY = get_key("the-odds-api", env="THE_ODDS_API_KEY")

_MAJORS = ("wimbledon", "australian open", "french open", "us open")


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


def _odds_get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((fold_name(a), fold_name(b))))


def _draw_index(tour: str) -> dict[tuple[str, str], dict]:
    """Current draw keyed by folded player pair, used to preserve ESPN spelling."""
    if not DRAW_CSV.exists():
        return {}
    out = {}
    with open(DRAW_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("tour") or "").lower() not in ("", tour):
                continue
            a = (r.get("player_a") or "").strip()
            b = (r.get("player_b") or "").strip()
            if not a or not b or "TBD" in (a.upper(), b.upper()):
                continue
            key = _pair_key(a, b)
            try:
                best_of = int(float(r.get("best_of") or 3))
            except ValueError:
                best_of = 3
            out[key] = {
                "player_a": a,
                "player_b": b,
                "surface": (r.get("surface") or "hard").lower(),
                "best_of": best_of,
            }
    return out


def _select_tennis_sport_keys(api_key: str, tour: str, event: str = "",
                              sport_key: str = "") -> list[dict]:
    """Return active Odds API tennis sport descriptors for a tour/event."""
    if sport_key:
        return [{"key": sport_key, "title": sport_key}]
    sports = _odds_get(f"{ODDS_API_BASE}/sports/?apiKey={api_key}")
    prefix = f"tennis_{tour.lower()}_"
    event_l = event.lower().strip()
    matches = []
    for s in sports:
        key = str(s.get("key") or "")
        title = str(s.get("title") or "")
        group = str(s.get("group") or "")
        if not s.get("active", True):
            continue
        if not key.startswith(prefix):
            continue
        if group.lower() != "tennis" and "tennis" not in key.lower():
            continue
        haystack = f"{key} {title}".lower()
        if event_l and event_l not in haystack:
            continue
        matches.append(s)
    return matches


def _fallback_meta(tour: str, title: str, key: str) -> tuple[str, int]:
    label = f"{title} {key}".lower()
    surface = _infer_surface(title or key)
    best_of = 5 if tour == "atp" and any(m in label for m in _MAJORS) else 3
    return surface, best_of


def fetch_odds_api(tour: str = "atp", event: str = "", api_key: str | None = None,
                   regions: str = "eu", sport_key: str = "") -> list[dict]:
    """Fetch h2h odds from The Odds API and return odds.csv-shaped rows.

    The provider names are matched back to draw.csv when possible, so the card's
    exact-name lookup works even when ESPN and the bookmaker differ in accents or
    punctuation. Odds are the median decimal price across available bookmakers.
    """
    api_key = (api_key or DEFAULT_ODDS_API_KEY or "").strip()
    if not api_key:
        raise ValueError("No The Odds API key. Set THE_ODDS_API_KEY or add "
                         "data/api_keys.json with key 'the-odds-api'.")

    draw = _draw_index(tour)
    sport_keys = _select_tennis_sport_keys(api_key, tour, event, sport_key)
    if not sport_keys:
        label = f"{tour.upper()}" + (f" matching {event!r}" if event else "")
        raise ValueError(f"No active Odds API tennis sport found for {label}.")

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for sport in sport_keys:
        key = sport["key"]
        title = str(sport.get("title") or key)
        query = urllib.parse.urlencode({
            "apiKey": api_key,
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": "decimal",
        })
        events = _odds_get(f"{ODDS_API_BASE}/sports/{key}/odds/?{query}")
        fallback_surface, fallback_best_of = _fallback_meta(tour, title, key)
        for ev in events:
            a = str(ev.get("home_team") or "").strip()
            b = str(ev.get("away_team") or "").strip()
            if not a or not b:
                continue
            prices = {fold_name(a): [], fold_name(b): []}
            for book in ev.get("bookmakers") or []:
                for market in book.get("markets") or []:
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes") or []:
                        name = fold_name(outcome.get("name") or "")
                        try:
                            price = float(outcome.get("price"))
                        except (TypeError, ValueError):
                            continue
                        if price > 1.0 and name in prices:
                            prices[name].append(price)
            if not prices[fold_name(a)] or not prices[fold_name(b)]:
                continue

            pair = _pair_key(a, b)
            meta = draw.get(pair)
            if meta:
                out_a, out_b = meta["player_a"], meta["player_b"]
                surface, best_of = meta["surface"], meta["best_of"]
                # Preserve draw order. Provider home/away can be opposite.
                if _pair_key(out_a, out_b) != _pair_key(a, b):
                    continue
                odds_by_fold = {
                    fold_name(a): statistics.median(prices[fold_name(a)]),
                    fold_name(b): statistics.median(prices[fold_name(b)]),
                }
                odds_a = odds_by_fold[fold_name(out_a)]
                odds_b = odds_by_fold[fold_name(out_b)]
            else:
                out_a, out_b = a, b
                surface, best_of = fallback_surface, fallback_best_of
                odds_a = statistics.median(prices[fold_name(a)])
                odds_b = statistics.median(prices[fold_name(b)])

            row_key = _pair_key(out_a, out_b)
            if row_key in seen:
                continue
            seen.add(row_key)
            rows.append({
                "tour": tour.lower(),
                "surface": surface,
                "best_of": best_of,
                "player_a": out_a,
                "player_b": out_b,
                "odds_a": round(float(odds_a), 3),
                "odds_b": round(float(odds_b), 3),
            })
    return rows


def write_odds_csv(rows: list[dict], path: Path = ODDS_CSV,
                   preserve_other_tours: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_tours = {str(r.get("tour") or "").lower() for r in rows
                     if str(r.get("tour") or "").strip()}
    out_rows: list[dict] = []
    if preserve_other_tours and path.exists() and replace_tours:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("tour") or "").lower() not in replace_tours:
                    out_rows.append(row)
    out_rows.extend(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ODDS_COLUMNS)
        w.writeheader()
        for row in out_rows:
            w.writerow({c: row.get(c, "") for c in ODDS_COLUMNS})
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
    ap.add_argument("--odds-api", action="store_true",
                    help="Fetch h2h odds from The Odds API into data/odds.csv")
    ap.add_argument("--event", default="",
                    help="Tournament name filter for --odds-api, e.g. Wimbledon")
    ap.add_argument("--api-key", default=DEFAULT_ODDS_API_KEY,
                    help="The Odds API key; defaults to THE_ODDS_API_KEY/data/api_keys.json")
    ap.add_argument("--sport-key", default="",
                    help="Explicit Odds API sport key, e.g. tennis_atp_wimbledon")
    ap.add_argument("--regions", default="eu",
                    help="The Odds API regions for odds fetch (default: eu)")
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

    if args.odds_api:
        rows = []
        tours = list(args.tours)
        if args.sport_key:
            if "_wta_" in args.sport_key:
                tours = ["wta"]
            elif "_atp_" in args.sport_key:
                tours = ["atp"]
        for tour in tours:
            fetched = fetch_odds_api(tour=tour, event=args.event,
                                     api_key=args.api_key,
                                     regions=args.regions,
                                     sport_key=args.sport_key)
            rows.extend(fetched)
            print(f"  {tour.upper()}: fetched {len(fetched)} h2h odds row(s)")
        if not rows:
            print("  no odds rows fetched — leaving odds.csv untouched")
            return
        out = write_odds_csv(rows)
        print(f"  odds → {out}")
        return

    if args.seed is not None or args.accumulate:
        years = args.seed if args.seed else None
        added = accumulate_matches(years=years, tours=tuple(args.tours))
        print(f"Done. {added} new match(es) recorded.")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
