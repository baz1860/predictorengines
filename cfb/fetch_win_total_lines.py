#!/usr/bin/env python3
"""Fetch NCAAF preseason season-win totals from The Odds API.

Season win totals are a futures market, not a per-game market. The previous
version of this script requested ``team_totals``/``wins`` against the featured
`/odds` endpoint — ``team_totals`` is a per-game market (points in ONE game)
available only on the per-event endpoint, and ``wins`` is not a market key at
all. Every call 4xx'd and the script exited on the first error, so this file
never produced output; anything it *had* produced would have been per-game
team points mislabelled as season wins.

This version:
  * discovers the live sport keys instead of hardcoding a guess, and reports
    what the account can actually see when nothing matches;
  * tries each candidate (sport key, market) pair and keeps going past a 4xx
    instead of exiting on the first one;
  * refuses to write rows whose points don't look like season wins (a season
    win total is ~1.5-13.5; a game total is ~30-80). A wrong-market response
    is a hard failure, never a silently mislabelled CSV.

Usage:
  python3 -m cfb.fetch_win_total_lines
  python3 -m cfb.fetch_win_total_lines --list        # show visible sport keys
  python3 -m cfb.fetch_win_total_lines --api-key KEY --year 2026

Output: data/win_totals_lines_<year>.csv (team, line, over_odds, under_odds, books)
        data/win_totals_raw_<year>.json  (full API response, for debugging)
"""
import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASE = "https://api.the-odds-api.com/v4"

# Season-win markets live on their own futures sport key. As of 2026-08 The
# Odds API lists only `americanfootball_ncaaf` (per-game) and
# `americanfootball_ncaaf_championship_winner` (an outright winner market) —
# neither carries season win totals. Probing the per-game key for `totals`
# would burn quota to return game totals we'd reject anyway, so we only ever
# request a key that actually looks like a season-wins market.
WIN_TOTAL_MARKETS = ("team_season_wins", "season_wins", "totals")
GAME_SPORT_KEY = "americanfootball_ncaaf"
# Substrings identifying an outright "who wins the title" market — these are
# NOT win totals, despite containing "win".
OUTRIGHT_MARKERS = ("championship_winner", "super_bowl_winner", "_winner")
SEASON_WIN_MARKERS = ("season_wins", "win_totals", "team_wins", "regular_season_wins")

# A season win total for a 12-13 game schedule. Anything outside this is a
# different market (per-game totals sit around 30-80).
MIN_PLAUSIBLE_WINS = 0.5
MAX_PLAUSIBLE_WINS = 14.5


def _get(url, *, allow_error=False):
    """GET JSON. With allow_error, return None on a 4xx instead of exiting."""
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        if allow_error and 400 <= e.code < 500:
            return None
        sys.exit(f"HTTP {e.code}: {body}")
    except Exception as e:
        if allow_error:
            return None
        sys.exit(f"Request failed: {e}")


def list_sports(api_key):
    return _get(f"{BASE}/sports/?apiKey={api_key}") or []


def ncaaf_sport_keys(sports):
    """All NCAAF-related sport keys the account can see."""
    keys = [s["key"] for s in sports
            if "ncaaf" in s.get("key", "").lower()
            or "college football" in s.get("title", "").lower()]
    return sorted(dict.fromkeys(keys))


def season_win_keys(keys):
    """Keys that plausibly carry SEASON win totals.

    Excludes outright title-winner futures (they contain "win" but price a
    single champion, not per-team win counts) and the per-game key (its
    `totals` market is game points).
    """
    out = []
    for key in keys:
        low = key.lower()
        if low == GAME_SPORT_KEY or any(m in low for m in OUTRIGHT_MARKERS):
            continue
        if any(m in low for m in SEASON_WIN_MARKERS):
            out.append(key)
    return out


def fetch_odds(api_key, sport_key, market):
    query = urllib.parse.urlencode({
        "apiKey": api_key, "regions": "us", "markets": market,
        "oddsFormat": "american",
    })
    return _get(f"{BASE}/sports/{sport_key}/odds/?{query}", allow_error=True)


def _points(events):
    values = []
    for event in events or []:
        for bm in event.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                for out in mkt.get("outcomes", []):
                    if out.get("point") is not None:
                        values.append(float(out["point"]))
    return values


def looks_like_win_totals(events):
    """True only if the response's points are in season-win range."""
    values = _points(events)
    if len(values) < 10:
        return False
    inside = sum(MIN_PLAUSIBLE_WINS <= v <= MAX_PLAUSIBLE_WINS for v in values)
    return inside / len(values) >= 0.9


def parse_win_totals(events):
    """Median line + over/under odds per team across bookmakers.

    Each outcome names the team and uses `description`/`name` for the side.
    A row without an identifiable Over side is dropped rather than guessed —
    the old code treated a missing side as "over" and could median over- and
    under-prices together.
    """
    team_data = {}
    for event in events:
        for bm in event.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                for out in mkt.get("outcomes", []):
                    point, price = out.get("point"), out.get("price")
                    if point is None or price is None:
                        continue
                    name = str(out.get("name", "")).strip()
                    desc = str(out.get("description", "")).strip()
                    # Providers vary: sometimes name=team & description=Over,
                    # sometimes the reverse. Identify the side explicitly.
                    side = None
                    for candidate in (desc, name):
                        if candidate.lower() in ("over", "under"):
                            side = candidate.lower()
                            break
                    if side is None:
                        continue
                    team = desc if name.lower() in ("over", "under") else name
                    if not team:
                        continue
                    team_data.setdefault(team, []).append(
                        {"line": float(point), "side": side,
                         "price": float(price), "book": bm.get("title", "")})

    rows = []
    for team, entries in team_data.items():
        overs = [e for e in entries if e["side"] == "over"]
        unders = [e for e in entries if e["side"] == "under"]
        if not overs:
            continue
        rows.append({
            "team": team,
            "line": median(e["line"] for e in overs),
            "over_odds": int(round(median(e["price"] for e in overs))),
            "under_odds": int(round(median(e["price"] for e in unders)))
            if unders else -110,
            "books": len({e["book"] for e in overs}),
        })
    return sorted(rows, key=lambda r: -r["line"])


def write_template(year, model_path=None):
    """Seed a hand-fillable lines CSV from the model's projected teams.

    The Odds API does not currently list an NCAAF season-wins market, so the
    supported way to run the comparison is to paste book lines in by hand.
    """
    out_path = os.path.join(HERE, "data", f"win_totals_lines_{year}.csv")
    if os.path.exists(out_path):
        sys.exit(f"{out_path} already exists — edit it, or delete it to reseed.")
    teams = []
    model_path = model_path or os.path.join(
        HERE, f"projected_win_totals_{year}.csv")
    if os.path.exists(model_path):
        import csv as _csv
        with open(model_path) as f:
            teams = [row["team"] for row in _csv.DictReader(f)]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["team", "line", "over_odds", "under_odds", "books"])
        w.writeheader()
        for team in teams or ["Ohio State"]:
            w.writerow({"team": team, "line": "", "over_odds": -110,
                        "under_odds": -110, "books": 1})
    print(f"wrote {out_path} ({len(teams) or 1} rows) — fill in the `line` "
          f"column from your book; blank lines are skipped.")
    print("Team names must be canonical CFBD names or reviewed aliases "
          "(cfb/data/team_aliases.json).")


def main():
    from api_keys import get_key

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--list", action="store_true",
                    help="list the sport keys this account can see, then exit")
    ap.add_argument("--template", action="store_true",
                    help="write a hand-fillable lines CSV instead of fetching")
    args = ap.parse_args()
    if args.template:
        write_template(args.year)
        return
    api_key = args.api_key or get_key("the-odds-api", env="THE_ODDS_API_KEY")
    if not api_key:
        sys.exit("No Odds API key. Pass --api-key, set THE_ODDS_API_KEY, "
                 "or add data/api_keys.json.")

    # /sports is a free endpoint — this whole discovery step costs no quota.
    sports = list_sports(api_key)
    if args.list:
        for s in sports:
            print(f"  {s['key']:<45s} {s.get('title', '')}")
        return

    ncaaf = ncaaf_sport_keys(sports)
    keys = season_win_keys(ncaaf)
    if not keys:
        print("NCAAF sport keys visible to this account:")
        for key in ncaaf:
            note = ""
            if key == GAME_SPORT_KEY:
                note = "  (per-game markets — game totals, not season wins)"
            elif any(m in key.lower() for m in OUTRIGHT_MARKERS):
                note = "  (outright title winner — not per-team win totals)"
            print(f"  {key}{note}")
        sys.exit(
            "No NCAAF season-win-total market is offered on this account, so "
            "there is nothing to fetch (no quota was spent).\n"
            "Either the market is not posted yet, or the provider does not "
            "carry it. To run the comparison from book lines you enter by "
            f"hand:\n  python3 -m cfb.fetch_win_total_lines --template "
            f"--year {args.year}")

    events, used = None, None
    attempts = []
    for sport_key in keys:
        for market in WIN_TOTAL_MARKETS:
            data = fetch_odds(api_key, sport_key, market)
            if not data:
                attempts.append(f"{sport_key}/{market}: no data")
                continue
            if not looks_like_win_totals(data):
                sample = _points(data)[:5]
                attempts.append(
                    f"{sport_key}/{market}: {len(data)} events but points "
                    f"{sample} are not season wins — wrong market, skipped")
                continue
            events, used = data, f"{sport_key}/{market}"
            break
        if events:
            break

    if not events:
        print("Tried:")
        for line in attempts:
            print(f"  {line}")
        sys.exit(
            "No NCAAF season-win-total market returned usable data. Run with "
            "--list to see visible sport keys, or use --template to enter "
            "lines by hand.")
    print(f"Using {used}: {len(events)} event(s)")

    raw_path = os.path.join(HERE, "data", f"win_totals_raw_{args.year}.json")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(events, f, indent=1)
    print(f"Raw response -> {raw_path}")

    rows = parse_win_totals(events)
    if not rows:
        sys.exit(f"Raw data saved but no team/over/under structure was parsed. "
                 f"Inspect {raw_path}.")
    out_path = os.path.join(HERE, "data", f"win_totals_lines_{args.year}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["team", "line", "over_odds", "under_odds", "books"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} teams -> {out_path}")


if __name__ == "__main__":
    main()
