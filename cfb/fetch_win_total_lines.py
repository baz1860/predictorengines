#!/usr/bin/env python3
"""Fetch NCAAF preseason win total lines from The Odds API.

Hits the /v4/sports/americanfootball_ncaaf/odds endpoint with the
team_totals market (season wins), saves raw JSON and a tidy CSV.

Usage:
  python3 fetch_win_total_lines.py
  python3 fetch_win_total_lines.py --api-key YOUR_KEY

Output: data/win_totals_lines_2026.csv (team, line, over_odds, under_odds, books)
        data/win_totals_raw_2026.json  (full API response for debugging)
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api_keys import get_key

DEFAULT_KEY = get_key("the-odds-api", env="THE_ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4"

# The Odds API market key for season win totals
# Try these in order until one returns data
MARKETS_TO_TRY = ["team_totals", "wins"]


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        sys.exit(f"HTTP {e.code}: {body}")
    except Exception as e:
        sys.exit(f"Request failed: {e}")


def fetch_sports(api_key):
    """List available sports and find the NCAAF key."""
    data = _get(f"{BASE}/sports/?apiKey={api_key}")
    ncaaf = [s for s in data if "ncaaf" in s.get("key", "").lower()
             or "college football" in s.get("title", "").lower()]
    if not ncaaf:
        print("Available sport keys:")
        for s in data:
            if "football" in s.get("title", "").lower():
                print(f"  {s['key']} — {s['title']}")
        sys.exit("No NCAAF sport found. Season may not be listed yet.")
    return ncaaf[0]["key"]


def fetch_odds(api_key, sport_key, market):
    url = (f"{BASE}/sports/{sport_key}/odds/"
           f"?apiKey={api_key}&regions=us&markets={market}&oddsFormat=american")
    return _get(url)


def parse_win_totals(events):
    """Extract median over/under line + odds per team across all bookmakers."""
    team_data = {}  # team -> list of (line, over_odds, under_odds, book)
    for event in events:
        for bm in event.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                for outcome in mkt.get("outcomes", []):
                    name = outcome.get("name", "")
                    side = outcome.get("description", "").lower()
                    price = outcome.get("price")
                    point = outcome.get("point")
                    if point is None or price is None:
                        continue
                    team_data.setdefault(name, []).append({
                        "line": point,
                        "side": side,
                        "price": price,
                        "book": bm["title"],
                    })

    rows = []
    for team, entries in team_data.items():
        overs = [e for e in entries if e["side"] in ("over", "")]
        unders = [e for e in entries if e["side"] == "under"]
        if not overs:
            continue
        line = median(e["line"] for e in overs)
        over_odds = median(e["price"] for e in overs) if overs else -110
        under_odds = median(e["price"] for e in unders) if unders else -110
        books = len(set(e["book"] for e in overs))
        rows.append({
            "team": team,
            "line": line,
            "over_odds": int(round(over_odds)),
            "under_odds": int(round(under_odds)),
            "books": books,
        })
    return sorted(rows, key=lambda r: -r["line"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=DEFAULT_KEY)
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()
    if not args.api_key:
        sys.exit("No Odds API key. Pass --api-key, set THE_ODDS_API_KEY, "
                 "or add data/api_keys.json.")

    print("Finding NCAAF sport key...")
    sport_key = fetch_sports(args.api_key)
    print(f"  Using: {sport_key}")

    events = None
    used_market = None
    for market in MARKETS_TO_TRY:
        print(f"Trying market: {market}...")
        data = fetch_odds(args.api_key, sport_key, market)
        if data:
            events = data
            used_market = market
            print(f"  Got {len(data)} events with market '{market}'")
            break
        print(f"  No data for market '{market}'")

    if not events:
        sys.exit("No win-total odds found. Lines may not be posted yet (usually July).")

    # Save raw
    raw_path = os.path.join(HERE, "data", f"win_totals_raw_{args.year}.json")
    with open(raw_path, "w") as f:
        json.dump(events, f, indent=1)
    print(f"Raw response -> {raw_path}")

    rows = parse_win_totals(events)
    if not rows:
        print("Raw data saved but couldn't parse team win total structure.")
        print("The API may be using a different format — check win_totals_raw_2026.json")
        print("and run: python3 compare_win_totals.py --raw")
        return

    import csv
    out_path = os.path.join(HERE, "data", f"win_totals_lines_{args.year}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["team", "line", "over_odds", "under_odds", "books"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} teams -> {out_path}")


if __name__ == "__main__":
    main()
