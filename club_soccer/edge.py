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

import model as M
from api_keys import get_key

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ODDS_CSV = DATA / "odds.csv"
REPORT = DATA / "edge_report.csv"
CACHE = DATA / "api_cache"
KELLY_FRACTION = 0.25
API_HOST = "v3.football.api-sports.io"
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
            from calibrate import apply as _apply_calib
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


def _api_get(path: str, api_key: str) -> dict:
    req = urllib.request.Request(
        f"https://{API_HOST}{path}",
        headers={"x-apisports-key": api_key, "x-rapidapi-host": API_HOST},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cached_fixture_odds(fixture_id: str, api_key: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"odds_{fixture_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    payload = _api_get(f"/odds?fixture={fixture_id}", api_key)
    path.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(),
                                "payload": payload}, indent=2))
    return {"payload": payload}


def _map_api_bet(bet_name: str, value_name: str, odd,
                 home: str = "", away: str = "") -> tuple[str, str, float | str, float] | None:
    bet = bet_name.strip().lower()
    value = value_name.strip().lower()
    try:
        decimal = float(odd)
    except (TypeError, ValueError):
        return None
    if decimal <= 1.0:
        return None
    if bet in {"match winner", "fulltime result", "1x2", "winner"}:
        if value in {"home", "1", home.strip().lower()}:
            return ("1x2", "home", "", decimal)
        if value in {"draw", "x"}:
            return ("1x2", "draw", "", decimal)
        if value in {"away", "2", away.strip().lower()}:
            return ("1x2", "away", "", decimal)
    if "over/under" in bet or bet in {"goals over/under", "total goals"}:
        parts = value.replace("goals", "").split()
        if len(parts) >= 2 and parts[0] in {"over", "under"}:
            try:
                line = float(parts[1])
            except ValueError:
                return None
            if abs(line - 2.5) < 1e-9:
                return ("total", parts[0], line, decimal)
    if bet in {"both teams score", "both teams to score", "btts"}:
        if value in {"yes", "no"}:
            return ("btts", value, "", decimal)
    return None


def fetch_api_odds(api_key: str | None = None) -> pd.DataFrame:
    """Fetch API-Football fixture odds for upcoming local fixtures."""
    key = api_key or get_key("api-football", env="API_FOOTBALL_KEY")
    if not key:
        raise ValueError("No API-Football key. Add data/api_keys.json or pass --api-key.")
    fixtures = M.upcoming(M.load_fixtures())
    rows = []
    for fixture in fixtures.itertuples(index=False):
        fixture_id = str(getattr(fixture, "fixture_id", "") or "").strip()
        if not fixture_id:
            continue
        payload = _cached_fixture_odds(fixture_id, key).get("payload", {})
        for item in payload.get("response", []) or []:
            bookmakers = item.get("bookmakers", []) or []
            for book in bookmakers:
                for bet in book.get("bets", []) or []:
                    for val in bet.get("values", []) or []:
                        mapped = _map_api_bet(str(bet.get("name", "")),
                                              str(val.get("value", "")),
                                              val.get("odd"),
                                              str(fixture.home),
                                              str(fixture.away))
                        if not mapped:
                            continue
                        market, side, line, odds = mapped
                        rows.append({"date": fixture.date.date(),
                                     "competition": fixture.competition,
                                     "home": fixture.home, "away": fixture.away,
                                     "market": market, "side": side,
                                     "line": line, "odds": odds,
                                     "bookmaker": book.get("name", "")})
    if not rows:
        raise ValueError("No API-Football odds found for upcoming local fixtures; use manual club_soccer/data/odds.csv.")
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
        raise ValueError("No The Odds API odds matched upcoming local fixtures; use API-Football or manual odds.csv.")
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
    ap.add_argument("--api-odds", action="store_true")
    ap.add_argument("--the-odds-api", action="store_true")
    ap.add_argument("--api-key")
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
        from calibrate import load_maps
        calib_maps = load_maps()
        if calib_maps is None:
            sys.exit("--calibrated needs data/calibration.json. "
                     "Fit it first: python3 validate.py --calibrate")
    try:
        if args.api_odds:
            odds = fetch_api_odds(args.api_key)
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
