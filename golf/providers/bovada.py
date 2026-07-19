"""Bovada golf coupon provider (free, keyless).

Bovada's public coupon JSON exposes the weekly markets The Odds API's free tier
does not: outright winner, tournament match-ups, and round 2-ball / 3-ball
markets for the current PGA / DP World events. We parse them into OddsQuote rows
and export to the same odds.csv / matchups.csv / threeballs.csv contract the
pricer already reads, so no downstream change is needed.

Best-effort and defensive: the endpoint is unofficial and geo-sensitive, so any
failure is surfaced as a QA warning and the existing CSVs (or a manual paste)
stand. Odds come straight from Bovada's `decimal` field (fractional fallback).
"""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Iterable

from .. import provider_qa as qa
from .odds_manual import OddsQuote, write_threeballs_csv, _ts

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "api_cache" / "bovada"
ODDS_CSV = DATA_DIR / "odds.csv"
MATCHUPS_CSV = DATA_DIR / "matchups.csv"

COUPON_URL = ("https://www.bovada.lv/services/sports/event/coupon/events/A/"
              "description/golf?marketFilterId=def&preMatchOnly=true&lang=en")

# Once a tournament tees off, Bovada moves its outright winner and tournament
# match-up markets from the pre-match coupon to the live one, so the pre-match
# feed above only carries next-round 2/3-balls mid-event. Fetching the live
# coupon as well keeps odds.csv / matchups.csv updating in-play instead of
# going stale after round 1 starts.
LIVE_COUPON_URL = ("https://www.bovada.lv/services/sports/event/coupon/events/A/"
                   "description/golf?marketFilterId=def&liveOnly=true&lang=en")


def _slug(name: str) -> str:
    """Slug used to match coupon event links, e.g. 'Travelers Championship'
    → 'travelers-championship' (matches /golf/pga-tour/travelers-championship…)."""
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


# Tokens that name a market type rather than a tournament: Bovada appends them
# to the tournament path segment ('scottish-open-2-3-balls', '…-match-ups',
# '…-specials'), and they also appear as their own segments ('cut-lines').
_MARKET_TOKENS = {
    "tournament", "match", "ups", "up", "round", "rounds", "ball", "balls",
    "cut", "lines", "specials", "main", "markets", "six", "shooter", "group",
    "1st", "2nd", "3rd", "4th", "final",
}
_STOP_TOKENS = {"the", "a"}


def _name_core(text: str) -> set[str]:
    """Identity tokens of a tournament name / link segment: slug words minus
    articles, market-type words and numbers (years, group sizes)."""
    return {t for t in _slug(text).split("-")
            if t and not t.isdigit()
            and t not in _STOP_TOKENS and t not in _MARKET_TOKENS}


def _link_matches_event(link: str, slug: str, event_core: set[str]) -> bool:
    """Does a coupon event link belong to `event_name`'s tournament?

    Exact slug substring first (historical behaviour), then a token test that
    tolerates sponsor-name drift: ESPN says 'Genesis Scottish Open' where
    Bovada files everything under 'scottish-open'. A link's tournament path
    segment matches when its identity tokens are a subset of the event's and
    cover at least half of them — subset rejects other tournaments outright
    ('isco-championship', 'the-open-championship' ⊄ {genesis, scottish,
    open}), and the half-coverage floor stops a bare shared word like 'open'
    from matching. The final path segment (player names + timestamp) is
    excluded from the test.
    """
    if slug and slug in (link or ""):
        return True
    if not event_core:
        return False
    parts = [p for p in str(link or "").split("/") if p]
    for seg in parts[1:-1]:  # skip the 'golf' root and the per-event leaf
        core = _name_core(seg)
        if core and core <= event_core and len(core) >= len(event_core) / 2:
            return True
    return False


_ROUND_WORDS = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "final": 4}


def _round_from_desc(desc: str) -> int | None:
    """Round number named in a market description, e.g. '2nd Round 2-Balls' → 2,
    'Final Round Match-Ups' → 4. Returns None when the description names no round,
    so the caller can fall back to its own round_no. Reading the round from the
    feed (rather than trusting the --round flag) means a board is labelled by what
    Bovada says it settles on, and two rounds posted at once never get conflated."""
    m = re.search(r"(1st|2nd|3rd|4th|final)\s+round", desc)
    return _ROUND_WORDS[m.group(1)] if m else None


def _decimal(price: dict) -> float | None:
    """Decimal odds from a Bovada price object (prefer its own decimal field,
    fall back to parsing the fractional)."""
    try:
        d = float(price.get("decimal"))
        if d > 1.0:
            return d
    except (TypeError, ValueError):
        pass
    frac = str(price.get("fractional") or "")
    if "/" in frac:
        try:
            num, den = frac.split("/")
            if float(den) > 0:
                return 1.0 + float(num) / float(den)
        except ValueError:
            pass
    return None


class BovadaGolfProvider:
    name = "bovada"

    def __init__(self, cache_dir: Path | None = None, ttl_seconds: int = 900):
        self.cache_dir = cache_dir or CACHE_DIR
        self.ttl_seconds = ttl_seconds

    def fetch_coupon(self, use_cache: bool = False, live: bool = False) -> list:
        """Fetch the golf coupon. `live=False` is the pre-match feed; `live=True`
        is the in-play feed, where outrights and tournament match-ups live once
        the event has teed off. Each feed keeps its own cache file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = self.cache_dir / ("coupon_golf_live.json" if live else "coupon_golf.json")
        if use_cache and cache.exists() and \
                time.time() - cache.stat().st_mtime <= self.ttl_seconds:
            return json.loads(cache.read_text())
        url = LIVE_COUPON_URL if live else COUPON_URL
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = json.load(resp)
        cache.write_text(json.dumps(payload))
        return payload

    # Markets worth taking from the live coupon: tournament-level prices that
    # settle on all 72 holes. Live round 2/3-ball groups are deliberately
    # excluded — mid-round in-play prices reflect holes already played and must
    # not feed the pre-round group pricer.
    LIVE_MARKETS = ("win", "tournament_matchup")

    def live_event_quotes(self, coupon: list, event_name: str,
                          event_id: str = "") -> list[OddsQuote]:
        """Tournament-level quotes (outright winner, tournament match-ups) from
        the live coupon, so those boards keep updating after the event tees off
        and drops out of the pre-match feed."""
        return [q for q in self.event_quotes(coupon, event_name, event_id=event_id)
                if q.market in self.LIVE_MARKETS]

    def event_quotes(self, coupon: list, event_name: str, event_id: str = "",
                     round_no: int = 1) -> list[OddsQuote]:
        """All quotes for `event_name`, matched on the coupon event link slug so
        only this tournament's markets are kept (the coupon mixes events from
        several tournaments under shared category groups)."""
        slug = _slug(event_name)
        if not slug:
            return []
        event_core = _name_core(event_name)
        quotes: list[OddsQuote] = []
        for grp in coupon or []:
            for ev in grp.get("events", []):
                if not _link_matches_event(ev.get("link") or "", slug, event_core):
                    continue
                for dg in ev.get("displayGroups", []):
                    for market in dg.get("markets", []):
                        quotes.extend(
                            self._market_quotes(ev, market, event_id, round_no))
        return _dedupe(quotes)

    def _market_quotes(self, ev: dict, market: dict, event_id: str,
                       round_no: int) -> list[OddsQuote]:
        desc = (market.get("description") or "").lower()
        outs = [((o.get("description") or "").strip(), _decimal(o.get("price") or {}))
                for o in market.get("outcomes", [])]
        outs = [(n, d) for n, d in outs if n and d]
        gid_base = str(ev.get("id") or ev.get("link") or desc)
        # Prefer the round the feed names ('2nd Round 2-Balls' → 2); fall back to
        # the caller's round_no only when the description is round-agnostic.
        round_no = _round_from_desc(desc) or round_no

        def rows(market_name, settlement, group_id=""):
            return [OddsQuote(event_id=event_id, market=market_name, player_name=n,
                              decimal_odds=o, round_no=(round_no if "ball" in market_name
                                                        or market_name == "round_matchup" else None),
                              group_id=group_id, book="bovada", source="bovada",
                              settlement_rule=settlement, timestamp=_ts())
                    for n, o in outs]

        if desc == "winner":
            return rows("win", "dead_heat")
        if "tournament match-up" in desc and len(outs) == 2:
            return rows("tournament_matchup", "push_tie", f"bovada-tmatch:{gid_base}")
        if "round match-up" in desc and len(outs) == 2:
            return rows("round_matchup", "push_tie", f"bovada-rmatch-r{round_no}:{gid_base}")
        if "2-ball" in desc and len(outs) == 2:
            return rows("2ball", "dead_heat", f"bovada-2ball-r{round_no}:{gid_base}")
        if "3-ball" in desc and len(outs) == 3:
            return rows("3ball", "dead_heat", f"bovada-3ball-r{round_no}:{gid_base}")
        return []  # ignore Scores, over/under, etc.

    def qa_checks(self, quotes: Iterable[OddsQuote], label: str = "bovada") -> list:
        rows = [q.as_dict() for q in quotes]
        return [
            qa.require_columns(label, rows, ["market", "player_name", "decimal_odds"]),
            qa.min_rows(label, rows, 1),
        ]


def _dedupe(quotes: list[OddsQuote]) -> list[OddsQuote]:
    seen, out = set(), []
    for q in quotes:
        key = (q.market, q.group_id, q.player_name)
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


# ── exporters to the CSV contract the pricer reads ──────────────────────────

def write_outrights_csv(quotes: Iterable[OddsQuote], path: Path | None = None,
                        event: str = "") -> Path | None:
    """Win-market quotes → odds.csv (name, odds_win). Returns None if no wins."""
    path = path or ODDS_CSV
    wins = [q for q in quotes if q.market == "win"]
    if not wins:
        return None
    cols = ["name", "odds_win", "odds_top5", "odds_top10", "odds_top20", "odds_cut",
            "event", "captured_at"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for q in wins:
            w.writerow({"name": q.player_name, "odds_win": round(q.decimal_odds, 3),
                        "event": event, "captured_at": q.timestamp or _ts()})
    return path


def write_matchups_csv(quotes: Iterable[OddsQuote], path: Path | None = None,
                       event: str = "") -> Path | None:
    """Tournament-matchup pairs → matchups.csv. Round match-ups are deliberately
    excluded — they settle on one round, not 72 holes, so they must not flow into
    the tournament-matchup pricer."""
    path = path or MATCHUPS_CSV
    groups: dict[str, list[OddsQuote]] = {}
    for q in quotes:
        if q.market == "tournament_matchup":
            groups.setdefault(q.group_id, []).append(q)
    pairs = {g: qs for g, qs in groups.items() if len(qs) == 2}
    if not pairs:
        return None
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "player_a", "player_b",
                                          "odds_a", "odds_b", "event", "captured_at"])
        w.writeheader()
        for gid, qs in pairs.items():
            w.writerow({"group_id": gid,
                        "player_a": qs[0].player_name, "player_b": qs[1].player_name,
                        "odds_a": round(qs[0].decimal_odds, 3),
                        "odds_b": round(qs[1].decimal_odds, 3),
                        "event": event, "captured_at": qs[0].timestamp or _ts()})
    return path


def export_csvs(quotes: list[OddsQuote], event: str = "") -> dict[str, int]:
    """Write the win / matchup / round-group CSVs and return rows written per
    file. Only writes a file when that market is present, so a partial coupon
    never blanks an existing board. Boards are tagged with the event so the
    pricers can refuse them once the tour moves on."""
    written = {}
    if write_outrights_csv(quotes, event=event):
        written["odds.csv"] = sum(1 for q in quotes if q.market == "win")
    if write_matchups_csv(quotes, event=event):
        written["matchups.csv"] = sum(1 for q in quotes if q.market == "tournament_matchup") // 2
    groups = [q for q in quotes if q.market in ("2ball", "3ball")]
    if groups:
        write_threeballs_csv(groups, event=event)
        written["threeballs.csv"] = len({q.group_id for q in groups})
    return written
