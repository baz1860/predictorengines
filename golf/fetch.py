"""
golf/fetch.py  –  Data layer for the golf prediction engine.

Sources (in priority order):
  1. DataGolf API  (--dg-key)  →  field + SG ratings + course fit
  2. ESPN unofficial API       →  current event field + leaderboard
  3. Manual CSV fallback       →  data/field.csv, data/players.csv

Usage:
  python fetch.py [--espn] [--dg-key KEY] [--odds-key KEY]
                  [--tournament-id N] [--no-odds]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
# Append (not insert at 0): keeps golf-local modules ahead of the root engine's
# same-named modules; root only needs to supply api_keys.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from api_keys import get_key

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# ESPN unofficial API helpers
# ─────────────────────────────────────────────

ESPN_LEADERBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/golf/pga/leaderboard"
)
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
)


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt+1}/{retries}: {exc}")
            time.sleep(2)


def fetch_espn_field() -> list[dict]:
    """
    Pull current PGA Tour event field from ESPN leaderboard API.
    Returns list of dicts with keys: name, world_rank, status.
    """
    print("Fetching ESPN field...")
    data = _get(ESPN_LEADERBOARD)

    players = []
    events = data.get("events", [])
    if not events:
        print("  No active event found on ESPN.")
        return players

    event = events[0]
    event_name = event.get("name", "Unknown")
    print(f"  Event: {event_name}")

    for comp in event.get("competitions", []):
        for comp_player in comp.get("competitors", []):
            athlete = comp_player.get("athlete", {})
            name = athlete.get("displayName", "")
            rank = athlete.get("displayName", "")  # ESPN doesn't expose OWGR here
            status = comp_player.get("status", {}).get("type", {}).get("name", "active")
            if name:
                players.append({
                    "name": name,
                    "world_rank": comp_player.get("rank", ""),
                    "status": status,
                    "event": event_name,
                })

    print(f"  {len(players)} players found.")
    return players


def fetch_espn_leaderboard() -> list[dict]:
    """
    Pull live or final leaderboard from ESPN.
    Returns list of dicts: name, position, score, thru, today.
    """
    print("Fetching ESPN leaderboard...")
    data = _get(ESPN_LEADERBOARD)

    rows = []
    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            for p in comp.get("competitors", []):
                athlete = p.get("athlete", {})
                rows.append({
                    "name": athlete.get("displayName", ""),
                    "position": p.get("status", {}).get("position", {}).get("displayName", ""),
                    "score": p.get("score", {}).get("displayValue", "E"),
                    "thru": p.get("status", {}).get("thru", ""),
                    "today": p.get("linescores", [{}])[-1].get("displayValue", "") if p.get("linescores") else "",
                })
    return rows


# ─────────────────────────────────────────────
# DataGolf API helpers
# ─────────────────────────────────────────────

DG_BASE = "https://feeds.datagolf.com"


def fetch_dg_field(api_key: str) -> list[dict]:
    """Pre-tournament player list with DataGolf skill ratings."""
    print("Fetching DataGolf field + ratings...")
    url = f"{DG_BASE}/field-updates"
    data = _get(url, params={"tour": "pga", "file_format": "json", "key": api_key})
    field = data.get("field", [])
    print(f"  {len(field)} players from DataGolf.")
    return field


def fetch_dg_predictions(api_key: str, add_position: int = 10) -> list[dict]:
    """Pre-tournament win/finish probabilities from DataGolf."""
    print("Fetching DataGolf predictions...")
    url = f"{DG_BASE}/preds/pre-tournament"
    data = _get(
        url,
        params={
            "tour": "pga",
            "add_position": add_position,
            "file_format": "json",
            "key": api_key,
        },
    )
    probs = data.get("baseline", []) or data.get("probs", [])
    print(f"  {len(probs)} player predictions.")
    return probs


def fetch_dg_historical_rounds(api_key: str, event_id: int, year: int) -> list[dict]:
    """Historical round-by-round data for a specific event."""
    url = f"{DG_BASE}/historical-raw-data/rounds"
    data = _get(
        url,
        params={
            "tour": "pga",
            "event_id": event_id,
            "year": year,
            "file_format": "json",
            "key": api_key,
        },
    )
    return data.get("scores", [])


# ─────────────────────────────────────────────
# The Odds API helper
# ─────────────────────────────────────────────

ODDS_BASE = "https://api.the-odds-api.com/v4"

# The Odds API uses event-specific sport keys for golf outrights, e.g.:
#   golf_masters_tournament_winner, golf_us_open_winner,
#   golf_the_open_championship_winner, golf_pga_championship,
#   golf_fedex_cup_winner, golf_rbc_canadian_open, etc.
# Run: python fetch.py --list-sports --odds-key KEY  to see all available keys.
GOLF_SPORT_DEFAULT = "golf_rbc_canadian_open"


def list_sports(api_key: str, filter_golf: bool = True) -> list[dict]:
    """List all available sport keys from The Odds API."""
    url = f"{ODDS_BASE}/sports"
    data = _get(url, params={"apiKey": api_key})
    if filter_golf:
        data = [s for s in data if "golf" in s.get("key", "").lower()]
    return data


def fetch_odds(api_key: str, market: str = "outrights", sport: str = GOLF_SPORT_DEFAULT) -> list[dict]:
    """
    Fetch current outright odds for a golf event from The Odds API.

    market: 'outrights' for tournament winner (most golf events use this)
    sport:  The Odds API sport key — use --list-sports to find the right one.
            Common golf keys:
              golf_rbc_canadian_open
              golf_us_open_winner
              golf_masters_tournament_winner
              golf_the_open_championship_winner
              golf_pga_championship
    """
    print(f"Fetching odds (sport={sport}, market={market})...")
    url = f"{ODDS_BASE}/sports/{sport}/odds"
    try:
        data = _get(
            url,
            params={
                "apiKey": api_key,
                "regions": "uk,eu,au",
                "markets": market,
                "oddsFormat": "decimal",
            },
        )
        print(f"  {len(data)} events with odds.")
        return data
    except Exception as exc:
        print(f"  Odds API error: {exc}")
        return []


# ─────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────

def write_field_csv(players: list[dict], path: Path | None = None) -> Path:
    """Write field.csv from a list of player dicts."""
    import csv
    path = path or DATA_DIR / "field.csv"
    cols = ["name", "world_rank", "status", "event", "odds_win", "odds_top5", "odds_top10", "odds_top20", "odds_cut"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for p in players:
            w.writerow({**{c: "" for c in cols}, **p})
    print(f"  Written {len(players)} rows → {path}")
    return path


def write_players_csv(players: list[dict], path: Path | None = None) -> Path:
    """Write or update players.csv with SG ratings."""
    import csv
    path = path or DATA_DIR / "players.csv"

    # Load existing to preserve manual edits
    existing = {}
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                existing[row["name"]] = row

    cols = ["name", "dg_id", "sg_total", "sg_ott", "sg_app", "sg_atg", "sg_putt", "driving_dist", "driving_acc", "datagolf_skill", "owgr", "country"]
    for p in players:
        name = p.get("player_name") or p.get("name", "")
        if not name:
            continue
        row = existing.get(name, {c: "" for c in cols})
        row["name"] = name
        # Map DataGolf fields
        if "dg_id" in p:
            row["dg_id"] = p["dg_id"]
        if "datagolf_skill" in p:
            row["datagolf_skill"] = p["datagolf_skill"]
        if "owgr" in p:
            row["owgr"] = p["owgr"]
        if "country" in p:
            row["country"] = p["country"]
        existing[name] = row

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in sorted(existing.values(), key=lambda r: r.get("name", "")):
            w.writerow(row)
    print(f"  {len(existing)} players → {path}")
    return path


def write_odds_csv(events: list[dict], path: Path | None = None, bookmaker_pref: list[str] | None = None) -> Path:
    """
    Parse The Odds API response into odds.csv in the format edge.py expects:
      name, odds_win, odds_top5, odds_top10, odds_top20, odds_cut, odds_nocut

    Picks the best available bookmaker per player (preference order in bookmaker_pref,
    default: pinnacle > bet365 > betfair > first available).
    """
    import csv
    path = path or DATA_DIR / "odds.csv"

    # bookmaker priority
    bm_pref = bookmaker_pref or ["pinnacle", "bet365", "betfair_ex_eu", "betfair_ex_uk", "unibet", "williamhill"]

    # Collect: player → market → {bookmaker: odds}
    collected: dict[str, dict[str, dict[str, float]]] = {}

    MARKET_MAP = {
        "outrights":  "odds_win",
        "h2h":        "odds_win",   # sometimes outrights come back as h2h
        "winner":     "odds_win",
        "top_5":      "odds_top5",
        "top_10":     "odds_top10",
        "top_20":     "odds_top20",
        "make_cut":   "odds_cut",
        "miss_cut":   "odds_nocut",
    }

    for event in events:
        for bm in event.get("bookmakers", []):
            bm_key = bm.get("key", "")
            for market in bm.get("markets", []):
                mkt_key = market.get("key", "")
                col = MARKET_MAP.get(mkt_key)
                if not col:
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "").strip()
                    price = outcome.get("price", 0)
                    if not name or not price:
                        continue
                    collected.setdefault(name, {}).setdefault(col, {})[bm_key] = float(price)

    # Flatten: pick best bookmaker per player per market
    def best_odds(bm_dict: dict[str, float]) -> float:
        for bm in bm_pref:
            if bm in bm_dict:
                return bm_dict[bm]
        return max(bm_dict.values())  # take highest if preferred not available

    cols = ["name", "odds_win", "odds_top5", "odds_top10", "odds_top20", "odds_cut", "odds_nocut"]
    rows = []
    for name, markets in sorted(collected.items()):
        row = {"name": name}
        for col in cols[1:]:
            if col in markets:
                row[col] = f"{best_odds(markets[col]):.2f}"
            else:
                row[col] = ""
        rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows)} players with odds → {path}")
    return path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fetch golf data (field, ratings, odds)")
    ap.add_argument("--espn", action="store_true", help="Fetch current field from ESPN")
    ap.add_argument("--leaderboard", action="store_true", help="Fetch live leaderboard from ESPN")
    ap.add_argument("--dg-key", default=get_key("datagolf", env="DG_API_KEY"), help="DataGolf API key")
    ap.add_argument("--odds-key", default=get_key("the-odds-api", env="THE_ODDS_API_KEY"), help="The Odds API key")
    ap.add_argument("--tournament-id", type=int, default=None, help="DataGolf tournament ID")
    ap.add_argument("--sport", default=GOLF_SPORT_DEFAULT,
                    help="The Odds API sport key (default: %(default)s)")
    ap.add_argument("--list-sports", action="store_true",
                    help="Print available golf sport keys from The Odds API and exit")
    ap.add_argument("--no-odds", action="store_true", help="Skip odds fetch")
    ap.add_argument("--accumulate", action="store_true",
                    help="Append new round-by-round results to data/rounds.csv "
                         "(current + previous season) and exit")
    ap.add_argument("--seed", nargs="*", type=int, metavar="YEAR",
                    help="Backfill data/rounds.csv for the given seasons "
                         "(e.g. --seed 2022 2023 2024 2025) and exit")
    args = ap.parse_args()

    # ── Round-history accumulation (v2 data store) ──
    if args.accumulate or args.seed is not None:
        from providers import accumulate_rounds, get_provider
        seasons = args.seed if args.seed else None
        provider = get_provider(seasons=seasons, need="history")
        added = accumulate_rounds(provider)
        print(f"Done. {added} new round(s) recorded.")
        return

    fetched_field = []

    # ── List sports ──
    if args.list_sports:
        if not args.odds_key:
            print("--list-sports requires --odds-key, THE_ODDS_API_KEY, or data/api_keys.json")
            sys.exit(1)
        sports = list_sports(args.odds_key)
        print(f"\nAvailable golf sport keys ({len(sports)}):")
        for s in sports:
            print(f"  {s.get('key'):<50} {s.get('title')}")
        sys.exit(0)

    # ── DataGolf (highest quality) ──
    if args.dg_key:
        try:
            dg_field = fetch_dg_field(args.dg_key)
            fetched_field = dg_field
            write_players_csv(dg_field)
        except Exception as exc:
            print(f"DataGolf field error: {exc}")
        try:
            dg_preds = fetch_dg_predictions(args.dg_key)
            write_players_csv(dg_preds)
        except Exception as exc:
            print(f"DataGolf predictions error: {exc}")

    # ── ESPN fallback ──
    if args.espn or not args.dg_key:
        try:
            espn_field = fetch_espn_field()
            if espn_field:
                if not fetched_field:
                    fetched_field = espn_field
                write_field_csv(espn_field)
        except Exception as exc:
            print(f"ESPN field error: {exc}")

    if args.leaderboard:
        try:
            lb = fetch_espn_leaderboard()
            import csv
            p = DATA_DIR / "leaderboard.csv"
            if lb:
                with open(p, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(lb[0].keys()))
                    w.writeheader()
                    w.writerows(lb)
                print(f"  Leaderboard → {p}")
        except Exception as exc:
            print(f"ESPN leaderboard error: {exc}")

    # ── Odds ──
    if not args.no_odds and args.odds_key:
        try:
            odds_data = fetch_odds(args.odds_key, market="outrights", sport=args.sport)
            write_odds_csv(odds_data)
        except Exception as exc:
            print(f"Odds API error: {exc}")
    elif not args.no_odds:
        odds_path = DATA_DIR / "odds.csv"
        if not odds_path.exists():
            print(f"No odds key provided. Create {odds_path} manually (see template).")

    if not fetched_field:
        print("\nNo field data fetched. Populate data/field.csv and data/players.csv manually.")
        print("See README for data sources.")
    else:
        print(f"\nDone. {len(fetched_field)} players in field.")


if __name__ == "__main__":
    main()
