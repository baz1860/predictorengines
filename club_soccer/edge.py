#!/usr/bin/env python3
"""Club soccer edge finder: 1X2, over/under 2.5, and BTTS."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import model as M
from api_keys import get_key

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ODDS_CSV = DATA / "odds.csv"
REPORT = DATA / "edge_report.csv"
CACHE = DATA / "bsd_cache"
KELLY_FRACTION = 0.25
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
ODDS_API_SPORTS = {
    "Premier League": "soccer_epl",
    "Championship": "soccer_efl_champ",
    "Bundesliga": "soccer_germany_bundesliga",
    "Serie A": "soccer_italy_serie_a",
    "Ligue 1": "soccer_france_ligue_one",
    "La Liga": "soccer_spain_la_liga",
    "Champions League": "soccer_uefa_champs_league",
    "Europa League": "soccer_uefa_europa_league",
    "Conference League": "soccer_uefa_europa_conference_league",
}

MARKETS = {
    "1x2": [("home", None), ("draw", None), ("away", None)],
    "total": [("over", 2.5), ("under", 2.5)],
    "btts": [("yes", None), ("no", None)],
}


def devig(odds: list[float]) -> np.ndarray:
    inv = np.array([1.0 / float(o) for o in odds])
    return inv / inv.sum()


def kelly(p: float, odds: float) -> float:
    b = odds - 1.0
    return max(0.0, (p * b - (1.0 - p)) / b)


def side_prob(pred: dict, market: str, side: str) -> float:
    p = pred["probs"]
    if market == "1x2":
        return float(p[side])
    if market == "total":
        return float(p["over25" if side == "over" else "under25"])
    if market == "btts":
        return float(p["btts_yes" if side == "yes" else "btts_no"])
    raise ValueError(f"Unknown market: {market}")


def load_odds(path: Path = ODDS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Use --template or API odds.")
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df["line"] = pd.to_numeric(df.get("line", np.nan), errors="coerce")
    return df.dropna(subset=["odds"]).copy()


def write_template(path: Path = ODDS_CSV) -> None:
    fixtures = M.upcoming(M.load_fixtures())
    rows = []
    for r in fixtures.itertuples(index=False):
        for market, sides in MARKETS.items():
            for side, line in sides:
                rows.append({"date": r.date.date(), "competition": r.competition,
                             "home": r.home, "away": r.away, "market": market,
                             "side": side, "line": "" if line is None else line,
                             "odds": ""})
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def rows_from_odds(odds: pd.DataFrame, model_name: str = "ensemble",
                   bankroll: float = 100.0, calib_maps=None) -> list[dict]:
    out = []
    params = M.load_params()
    for (date, comp, home, away, market), grp in odds.groupby(
            ["date", "competition", "home", "away", "market"], dropna=False):
        priced = grp[np.isfinite(grp["odds"]) & (grp["odds"] > 1.0)]
        if len(priced) < 2:
            continue
        try:
            pred = M.predict(home, away, comp, model_name, params=params)
        except ValueError:
            continue
        if calib_maps is not None:
            from .calibrate import apply as _apply_calib
            ph, pdr, pa = _apply_calib(pred["probs"]["home"], pred["probs"]["draw"],
                                       pred["probs"]["away"], calib_maps)
            pred["probs"]["home"], pred["probs"]["draw"], pred["probs"]["away"] = ph, pdr, pa
        implied = devig(priced["odds"].tolist())
        for (_, r), p_book in zip(priced.iterrows(), implied):
            p_model = side_prob(pred, str(r["market"]), str(r["side"]))
            ev = p_model * float(r["odds"]) - 1.0
            kfrac = KELLY_FRACTION * kelly(p_model, float(r["odds"]))
            raw_line = r.get("line", np.nan)
            line = "" if pd.isna(raw_line) else float(raw_line)
            label = bet_label(home, away, str(r["market"]), str(r["side"]), line)
            out.append({"date": str(date), "competition": comp,
                        "match": f"{home} v {away}", "home": home, "away": away,
                        "market": r["market"], "side": r["side"], "line": line,
                        "bet": label, "odds": round(float(r["odds"]), 3),
                        "p_model": round(float(p_model), 3),
                        "p_book": round(float(p_book), 3),
                        "edge": round(float(p_model - p_book), 3),
                        "ev_per_unit": round(float(ev), 3),
                        "kelly_stake": round(float(kfrac), 4),
                        "stake_gbp": round(float(kfrac) * bankroll, 2)})
    out.sort(key=lambda x: -x["ev_per_unit"])
    return out


def bet_label(home: str, away: str, market: str, side: str, line) -> str:
    if market == "1x2":
        return {"home": f"{home} win", "draw": "Draw", "away": f"{away} win"}[side]
    if market == "total":
        return f"{'Over' if side == 'over' else 'Under'} {float(line):g} goals"
    if market == "btts":
        return "Both teams to score" if side == "yes" else "BTTS no"
    return f"{market} {side}"


def _decimal(v: object) -> float | None:
    """Parse a decimal odds value; return None if invalid."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if f > 1.0 else None
    except (TypeError, ValueError):
        return None


def fetch_bsd_odds(api_key: str | None = None) -> pd.DataFrame:
    """Fetch BSD odds for upcoming local fixtures.

    BSD embeds 1X2 odds directly in each event response
    (``odds_home``, ``odds_draw``, ``odds_away``).  Over/under 2.5 and
    BTTS odds are read from common BSD field names if present
    (``odds_over25``, ``odds_under25``, ``odds_btts_yes``, ``odds_btts_no``).

    BSD key: data/api_keys.json -> "bsd", or env BSD_API_KEY.
    Register free at https://sports.bzzoiro.com/register/
    """
    from bsd_client import get_all_events, league_name as bsd_league_name
    from .competitions import comp_from_bsd_league

    key = api_key or get_key("bsd", env="BSD_API_KEY")
    if not key:
        raise ValueError(
            "No BSD key. Register at https://sports.bzzoiro.com/register/ "
            "and add 'bsd' to data/api_keys.json, or set BSD_API_KEY."
        )

    # Match BSD events to our upcoming fixtures by (home, away, competition)
    fixtures = M.upcoming(M.load_fixtures())
    if fixtures.empty:
        raise ValueError("No upcoming fixtures found in fixtures.csv.")

    # Build lookup: (home_lower, away_lower, comp) -> fixture row
    fixture_lookup: dict[tuple[str, str, str], object] = {}
    for fx in fixtures.itertuples(index=False):
        key_t = (str(fx.home).lower(), str(fx.away).lower(), str(fx.competition))
        fixture_lookup[key_t] = fx

    events = get_all_events(key, status="upcoming")
    CACHE.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ev in events:
        comp = comp_from_bsd_league(bsd_league_name(ev))
        if comp is None:
            continue
        home_raw = str(ev.get("home_team") or "")
        away_raw = str(ev.get("away_team") or "")
        lookup_key = (home_raw.lower(), away_raw.lower(), comp.name)
        fixture = fixture_lookup.get(lookup_key)
        if fixture is None:
            # Try partial match (BSD names may differ slightly)
            for (fh, fa, fc), fx in fixture_lookup.items():
                if fc == comp.name and (fh in home_raw.lower() or home_raw.lower() in fh):
                    if fa in away_raw.lower() or away_raw.lower() in fa:
                        fixture = fx
                        break
        if fixture is None:
            continue

        date = getattr(fixture, "date", None)
        if hasattr(date, "date"):
            date = date.date()

        # 1X2 odds (always embedded in BSD events)
        odds_h = _decimal(ev.get("odds_home"))
        odds_d = _decimal(ev.get("odds_draw"))
        odds_a = _decimal(ev.get("odds_away"))

        # Over/under 2.5 and BTTS (BSD may provide these as top-level fields)
        odds_over = _decimal(ev.get("odds_over25") or ev.get("odds_over_2_5"))
        odds_under = _decimal(ev.get("odds_under25") or ev.get("odds_under_2_5"))
        odds_btts_y = _decimal(ev.get("odds_btts_yes") or ev.get("odds_btts"))
        odds_btts_n = _decimal(ev.get("odds_btts_no"))

        # Also try nested odds dict if present
        nested = ev.get("odds") or {}
        if isinstance(nested, dict):
            if odds_h is None:
                odds_h = _decimal(nested.get("home") or nested.get("1"))
            if odds_d is None:
                odds_d = _decimal(nested.get("draw") or nested.get("x"))
            if odds_a is None:
                odds_a = _decimal(nested.get("away") or nested.get("2"))
            if odds_over is None:
                odds_over = _decimal(nested.get("over25") or nested.get("over_2_5"))
            if odds_under is None:
                odds_under = _decimal(nested.get("under25") or nested.get("under_2_5"))
            if odds_btts_y is None:
                odds_btts_y = _decimal(nested.get("btts_yes"))
            if odds_btts_n is None:
                odds_btts_n = _decimal(nested.get("btts_no"))

        base = {
            "date": date,
            "competition": comp.name,
            "home": getattr(fixture, "home", home_raw),
            "away": getattr(fixture, "away", away_raw),
            "bookmaker": "bsd",
        }
        if odds_h is not None:
            rows.append({**base, "market": "1x2", "side": "home", "line": "", "odds": odds_h})
        if odds_d is not None:
            rows.append({**base, "market": "1x2", "side": "draw", "line": "", "odds": odds_d})
        if odds_a is not None:
            rows.append({**base, "market": "1x2", "side": "away", "line": "", "odds": odds_a})
        if odds_over is not None:
            rows.append({**base, "market": "total", "side": "over", "line": 2.5, "odds": odds_over})
        if odds_under is not None:
            rows.append({**base, "market": "total", "side": "under", "line": 2.5, "odds": odds_under})
        if odds_btts_y is not None:
            rows.append({**base, "market": "btts", "side": "yes", "line": "", "odds": odds_btts_y})
        if odds_btts_n is not None:
            rows.append({**base, "market": "btts", "side": "no", "line": "", "odds": odds_btts_n})

    if not rows:
        raise ValueError(
            "No BSD odds matched upcoming fixtures. "
            "Check your BSD key, or use manual club_soccer/data/odds.csv."
        )
    return pd.DataFrame(rows)


def fetch_the_odds_api(api_key: str | None = None) -> pd.DataFrame:
    """Best-effort The Odds API fallback for competitions it publishes."""
    key = api_key or get_key("the-odds-api", env="THE_ODDS_API_KEY")
    if not key:
        raise ValueError("No The Odds API key. Add data/api_keys.json key 'the-odds-api'.")
    fixtures = M.upcoming(M.load_fixtures())
    rows = []
    for comp, sport in ODDS_API_SPORTS.items():
        wanted = fixtures[fixtures["competition"] == comp]
        if wanted.empty:
            continue
        query = urllib.parse.urlencode({
            "apiKey": key, "regions": "uk,eu,us",
            "markets": "h2h,totals,btts", "oddsFormat": "decimal",
        })
        url = f"{ODDS_API_URL.format(sport=sport)}?{query}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for event in payload:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            match = wanted[(wanted["home"] == home) & (wanted["away"] == away)]
            if match.empty:
                continue
            fixture = match.iloc[0]
            for book in event.get("bookmakers", []) or []:
                for market in book.get("markets", []) or []:
                    key_name = market.get("key", "")
                    for outcome in market.get("outcomes", []) or []:
                        name = str(outcome.get("name", ""))
                        odds = outcome.get("price")
                        if key_name == "h2h":
                            mapped = _map_api_bet("Match Winner", name, odds, home, away)
                        elif key_name == "totals":
                            mapped = _map_api_bet("Goals Over/Under",
                                                  f"{name} {outcome.get('point', '')}", odds)
                        elif key_name == "btts":
                            mapped = _map_api_bet("Both Teams Score", name, odds)
                        else:
                            mapped = None
                        if not mapped:
                            continue
                        market_name, side, line, decimal = mapped
                        rows.append({"date": fixture["date"].date(),
                                     "competition": comp, "home": home, "away": away,
                                     "market": market_name, "side": side,
                                     "line": line, "odds": decimal,
                                     "bookmaker": book.get("title", "")})
    if not rows:
        raise ValueError("No The Odds API odds matched upcoming local fixtures; use BSD odds (--bsd-odds) or manual odds.csv.")
    return pd.DataFrame(rows)


def grade(side: str, market: str, line, home_goals: float, away_goals: float) -> str:
    hg, ag = int(home_goals), int(away_goals)
    if market == "1x2":
        actual = "home" if hg > ag else ("draw" if hg == ag else "away")
        return "won" if side == actual else "lost"
    if market == "total":
        total = hg + ag
        line = float(line)
        if total == line:
            return "push"
        return "won" if ((side == "over" and total > line) or
                         (side == "under" and total < line)) else "lost"
    if market == "btts":
        yes = hg > 0 and ag > 0
        return "won" if ((side == "yes" and yes) or (side == "no" and not yes)) else "lost"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--bsd-odds", action="store_true",
                    help="fetch live odds from BSD (free; recommended)")
    ap.add_argument("--the-odds-api", action="store_true",
                    help="fetch odds from The Odds API (paid)")
    ap.add_argument("--api-key",
                    help="API key override (BSD key for --bsd-odds, "
                         "Odds API key for --the-odds-api)")
    ap.add_argument("--model", choices=["ensemble", "goals", "elo", "xg"], default="ensemble")
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--calibrated", action="store_true",
                    help="apply fitted 1X2 calibration (needs validate.py --calibrate)")
    args = ap.parse_args()
    if args.template:
        write_template()
        print(f"Wrote {ODDS_CSV}")
        return
    calib_maps = None
    if args.calibrated:
        from .calibrate import load_maps
        calib_maps = load_maps()
        if calib_maps is None:
            sys.exit("--calibrated needs data/calibration.json. "
                     "Fit it first: python3 validate.py --calibrate")
    try:
        if args.bsd_odds:
            odds = fetch_bsd_odds(args.api_key)
        elif args.the_odds_api:
            odds = fetch_the_odds_api(args.api_key)
        else:
            odds = load_odds()
        rows = rows_from_odds(odds, args.model, args.bankroll, calib_maps)
    except Exception as e:
        sys.exit(str(e))
    DATA.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT, index=False)
    if rows:
        print(pd.DataFrame(rows).head(30).to_string(index=False))
    else:
        print("No priced edges found.")
    print(f"Saved -> {REPORT}")


if __name__ == "__main__":
    main()
