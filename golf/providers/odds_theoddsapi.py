"""The Odds API provider for free-tier golf outrights.

The Odds API's own golf page describes current golf coverage as major-tournament
winner futures. Treat this provider as optional support for outrights, not as a
weekly PGA 3-ball/matchup source.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from api_keys import get_key

from .odds_manual import OddsQuote

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "api_cache" / "the_odds_api"
ODDS_BASE = "https://api.the-odds-api.com/v4"

MAJOR_SPORT_KEYS = {
    "masters": "golf_masters_tournament_winner",
    "pga_championship": "golf_pga_championship_winner",
    "us_open": "golf_us_open_winner",
    "the_open": "golf_the_open_championship_winner",
}


class TheOddsApiGolfProvider:
    name = "the_odds_api"

    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None,
                 ttl_seconds: int = 900):
        self.api_key = api_key or get_key("the-odds-api", env="THE_ODDS_API_KEY")
        self.cache_dir = cache_dir or CACHE_DIR
        self.ttl_seconds = ttl_seconds

    def list_golf_sports(self, use_cache: bool = True) -> list[dict]:
        if not self.api_key:
            return []
        data = self._request("sports", f"{ODDS_BASE}/sports", {"apiKey": self.api_key}, use_cache)
        return [s for s in data if "golf" in str(s.get("key", "")).lower()]

    def fetch_outrights(self, sport_key: str, event_id: str = "",
                        regions: str = "uk,eu,us",
                        bookmakers: str = "",
                        use_cache: bool = True) -> list[OddsQuote]:
        if not self.api_key:
            return []
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": "outrights",
            "oddsFormat": "decimal",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
            params.pop("regions", None)
        data = self._request(
            f"outrights_{sport_key}",
            f"{ODDS_BASE}/sports/{sport_key}/odds",
            params,
            use_cache,
        )
        return _parse_outrights(data, event_id=event_id)

    def _request(self, label: str, url: str, params: dict,
                 use_cache: bool = True):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = self.cache_dir / f"{label}.json"
        if use_cache and cache.exists() and time.time() - cache.stat().st_mtime <= self.ttl_seconds:
            return json.loads(cache.read_text())
        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
        cache.write_text(json.dumps(payload))
        return payload


def _parse_outrights(payload: list[dict], event_id: str = "") -> list[OddsQuote]:
    out = []
    for event in payload or []:
        eid = event_id or str(event.get("id") or "")
        for book in event.get("bookmakers", []) or []:
            book_key = str(book.get("key") or book.get("title") or "")
            for market in book.get("markets", []) or []:
                if market.get("key") not in {"outrights", "h2h"}:
                    continue
                for outcome in market.get("outcomes", []) or []:
                    name = str(outcome.get("name") or "").strip()
                    try:
                        price = float(outcome.get("price"))
                    except (TypeError, ValueError):
                        continue
                    if name and price > 1:
                        out.append(OddsQuote(
                            event_id=eid,
                            market="win",
                            player_name=name,
                            decimal_odds=price,
                            book=book_key,
                            source="the_odds_api",
                            timestamp=str(book.get("last_update") or ""),
                        ))
    return out
