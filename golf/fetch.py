"""
golf/fetch.py  –  Data layer for the golf prediction engine.

Sources (in priority order):
  1. DataGolf API  (--dg-key)  →  field + SG ratings + course fit
  2. ESPN unofficial API       →  current event field + leaderboard
  3. Manual CSV fallback       →  data/field.csv, data/players.csv

Usage:
  python -m golf.fetch [--espn] [--dg-key KEY] [--odds-key KEY]
                  [--tournament-id N] [--no-odds]
"""

import argparse
import datetime as dt
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
    "https://site.web.api.espn.com/apis/site/v2/sports/golf/leaderboard"
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
    Pull current PGA Tour event field from ESPN scoreboard API.
    Returns list of dicts with keys: name, world_rank, status.

    Only considers events that are upcoming or in progress. On Mondays the
    scoreboard still shows last week's finished event; writing that field
    would poison downstream name checks (season.py validates matchups
    against field.csv), so completed events return an empty field instead.
    """
    print("Fetching ESPN field...")
    data = _get(ESPN_LEADERBOARD, {"league": "pga"})

    players = []
    events = data.get("events", [])
    if not events:
        print("  No active event found on ESPN.")
        return players

    live = [e for e in events
            if not e.get("status", {}).get("type", {}).get("completed", False)]
    if not live:
        done = ", ".join(e.get("name", "?") for e in events)
        print(f"  Scoreboard only shows completed event(s): {done}.")
        print("  Next field not yet published — field.csv left unchanged.")
        return players

    event = live[0]
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
    data = _get(ESPN_LEADERBOARD, {"league": "pga"})

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
# Keys only exist while the event has an open market, so the right key is
# discovered per run (find_golf_sport_key) rather than hardcoded.
# Run: python -m golf.fetch --list-sports --odds-key KEY  to see all available keys.


def list_sports(api_key: str, filter_golf: bool = True) -> list[dict]:
    """List all available sport keys from The Odds API."""
    url = f"{ODDS_BASE}/sports"
    data = _get(url, params={"apiKey": api_key})
    if filter_golf:
        data = [s for s in data if "golf" in s.get("key", "").lower()]
    return data


# Words that carry no identity: "Masters Tournament Winner" ≡ "Masters".
_SPORT_KEY_STOPWORDS = {"the", "a", "winner", "tournament", "championship", "golf"}


def _name_tokens(text: str) -> set[str]:
    text = text.lower().replace(".", "").replace("'", "")  # "U.S." → "us"
    words = "".join(c if c.isalnum() else " " for c in text).split()
    return {w for w in words if w not in _SPORT_KEY_STOPWORDS}


def find_golf_sport_key(api_key: str, event_name: str) -> str | None:
    """
    Pick The Odds API sport key for the current event by name-matching the
    active golf keys against the ESPN event name. Returns None when no key
    matches — e.g. the API doesn't carry this week's event, or the only
    active keys belong to other tournaments (fetching those would silently
    price the wrong event, which is worse than no odds).

    Raises on network failure so the caller can tell "confirmed no market"
    (None) apart from "couldn't check" (exception).
    """
    sports = list_sports(api_key)

    want = _name_tokens(event_name)
    if not want:
        return None

    best_key, best_score = None, 0.0
    for s in sports:
        for label in (s.get("title", ""), s.get("key", "")):
            have = _name_tokens(label)
            if not have:
                continue
            # Jaccard keeps "US Open" from matching "The Open" (both share
            # only "open") while exact-identity names score 1.0.
            score = len(want & have) / len(want | have)
            if score > best_score:
                best_key, best_score = s.get("key"), score

    # Strictly above 0.5: "US Open" vs "The Open Winner" scores exactly 0.5
    # and must not match (The Open's market opens weeks early).
    if best_score > 0.5:
        return best_key
    if sports:
        keys = ", ".join(s.get("key", "?") for s in sports)
        print(f"  No Odds API market matches '{event_name}' (active golf keys: {keys}).")
    else:
        print(f"  No active golf markets on The Odds API for '{event_name}'.")
    return None


def fetch_odds(api_key: str, market: str = "outrights", sport: str = "") -> list[dict]:
    """
    Fetch current outright odds for a golf event from The Odds API.

    market: 'outrights' for tournament winner (most golf events use this)
    sport:  The Odds API sport key — normally discovered via
            find_golf_sport_key(); use --list-sports to inspect manually.
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


def write_odds_csv(events: list[dict], path: Path | None = None,
                   bookmaker_pref: list[str] | None = None,
                   event_name: str = "") -> Path:
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

    captured_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    cols = ["name", "odds_win", "odds_top5", "odds_top10", "odds_top20", "odds_cut",
            "odds_nocut", "event", "captured_at"]
    rows = []
    for name, markets in sorted(collected.items()):
        row = {"name": name, "event": event_name, "captured_at": captured_at}
        for col in ("odds_win", "odds_top5", "odds_top10", "odds_top20", "odds_cut", "odds_nocut"):
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
    ap.add_argument("--use-datagolf", action="store_true",
                    help="compatibility only: use DataGolf field/prediction endpoints when keyed")
    ap.add_argument("--odds-key", default=get_key("the-odds-api", env="THE_ODDS_API_KEY"), help="The Odds API key")
    ap.add_argument("--tournament-id", type=int, default=None, help="DataGolf tournament ID")
    ap.add_argument("--sport", default=None,
                    help="The Odds API sport key (default: auto-detect from the current event name)")
    ap.add_argument("--list-sports", action="store_true",
                    help="Print available golf sport keys from The Odds API and exit")
    ap.add_argument("--no-odds", action="store_true", help="Skip odds fetch")
    ap.add_argument("--accumulate", action="store_true",
                    help="Append new round-by-round results to data/rounds.csv "
                         "(current + previous season) and exit")
    ap.add_argument("--seed", nargs="*", type=int, metavar="YEAR",
                    help="Backfill data/rounds.csv for the given seasons "
                         "(e.g. --seed 2022 2023 2024 2025) and exit")
    ap.add_argument("--rebuild", nargs="+", type=int, metavar="YEAR",
                    help="Atomically rebuild rounds.csv from corrected per-event "
                         "payloads for these seasons; never reads existing rows")
    ap.add_argument("--tours", default="pga",
                    help="Comma-separated tours to accumulate/seed: "
                         "pga, liv, eur (DP World Tour). Default: pga. "
                         "Example: --tours pga,liv,eur")
    args = ap.parse_args()

    # ── Round-history accumulation (v2 data store) ──
    if args.accumulate or args.seed is not None or args.rebuild is not None:
        from .providers import (
            accumulate_rounds, accumulate_tours, get_provider, rebuild_tours,
        )
        seasons = args.seed if args.seed else None
        tours = [t.strip() for t in str(args.tours).split(",") if t.strip()]
        if args.rebuild is not None:
            results = rebuild_tours(tours, seasons=args.rebuild)
            total = sum(results.values())
            summary = ", ".join(f"{t}: {n}" for t, n in results.items())
            print(f"Done. Rebuilt {total} round(s) ({summary}).")
            return
        if tours == ["pga"]:
            # Unchanged default path: keyed provider (DataGolf) when available.
            provider = get_provider(seasons=seasons, need="history")
            added = accumulate_rounds(provider)
            print(f"Done. {added} new round(s) recorded.")
        else:
            results = accumulate_tours(tours, seasons=seasons)
            total = sum(results.values())
            summary = ", ".join(f"{t}: {n}" for t, n in results.items())
            print(f"Done. {total} new round(s) recorded ({summary}).")
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

    # ── DataGolf compatibility path ──
    if args.use_datagolf and args.dg_key:
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

    # ── ESPN free-source path ──
    if args.espn or not fetched_field:
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
            sport = args.sport
            event_name = next(
                (p.get("event") for p in fetched_field if p.get("event")), "")
            if not sport and event_name:
                sport = find_golf_sport_key(args.odds_key, event_name)
            if sport:
                odds_data = fetch_odds(args.odds_key, market="outrights", sport=sport)
                write_odds_csv(odds_data, event_name=event_name)
            elif event_name:
                # Confirmed: no market for this event. Any existing odds.csv
                # belongs to an earlier event, and edge.py matches odds to
                # predictions by player name alone — the same pros play every
                # week, so stale odds would silently price the wrong event.
                write_odds_csv([], event_name=event_name)
                print(f"  Cleared odds.csv — no market for '{event_name}' yet.")
            else:
                print("  No current event to match odds against — odds.csv left unchanged.")
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
