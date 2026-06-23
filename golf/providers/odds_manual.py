"""Manual bookmaker odds provider.

Free automation can model 3-ball probabilities, but free bookmaker odds for
3-balls and matchups are not reliably exposed. This provider makes pasted/CSV
boards first-class inputs with schema validation and normalized quote rows.
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .. import provider_qa as qa

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ODDS_CSV = DATA_DIR / "odds.csv"
MATCHUPS_CSV = DATA_DIR / "matchups.csv"
THREEBALLS_CSV = DATA_DIR / "threeballs.csv"
THREEBALLS_RAW = DATA_DIR / "threeballs_r1_raw.txt"


@dataclass(frozen=True)
class OddsQuote:
    market: str
    player_name: str
    decimal_odds: float
    event_id: str = ""
    round_no: int | None = None
    group_id: str = ""
    book: str = "manual"
    source: str = "manual"
    timestamp: str = ""
    settlement_rule: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


HEADER_RE = re.compile(r"^[23]\s*Ball.*-\s*(.+)$", re.I)
NUM_RE = re.compile(r"^\d+(\.\d+)?$")
FRAC_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$")


def _parse_odds(token: str) -> float | None:
    """Decimal odds from a pasted token. Accepts decimal (2.50), UK fractional
    (6/5, 13/8, 4/6) and evens. Returns None if the token isn't a price."""
    t = token.strip()
    if t.lower() in {"evens", "evs", "even"}:
        return 2.0
    if NUM_RE.match(t):
        return float(t)
    m = FRAC_RE.match(t)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den > 0:
            return 1.0 + num / den
    return None


def _group_market(n_players: int) -> str:
    """Round-group market tag by field size: twosomes vs threesomes."""
    return "2ball" if n_players == 2 else "3ball"


class ManualOddsProvider:
    name = "manual_odds"

    def load_outrights(self, path: Path | None = None, event_id: str = "") -> list[OddsQuote]:
        path = path or ODDS_CSV
        if not path.exists():
            return []
        out = []
        market_map = {
            "odds_win": "win",
            "odds_top5": "top5",
            "odds_top10": "top10",
            "odds_top20": "top20",
            "odds_cut": "make_cut",
            "odds_nocut": "miss_cut",
        }
        with path.open() as f:
            for row in csv.DictReader(f):
                name = (row.get("name") or row.get("player") or "").strip()
                if not name:
                    continue
                for col, market in market_map.items():
                    odds = _safe_float(row.get(col))
                    if odds and odds > 1:
                        out.append(OddsQuote(
                            event_id=event_id,
                            market=market,
                            player_name=name,
                            decimal_odds=odds,
                            timestamp=_ts(),
                            settlement_rule="dead_heat" if market.startswith("top") else "",
                        ))
        return out

    def load_matchups(self, path: Path | None = None, event_id: str = "",
                      round_no: int | None = None) -> list[OddsQuote]:
        path = path or MATCHUPS_CSV
        if not path.exists():
            return []
        out = []
        with path.open() as f:
            for i, row in enumerate(csv.DictReader(f), 1):
                a = (row.get("player_a") or "").strip()
                b = (row.get("player_b") or "").strip()
                oa, ob = _safe_float(row.get("odds_a")), _safe_float(row.get("odds_b"))
                if not (a and b and oa and ob):
                    continue
                gid = row.get("group_id") or f"matchup-{i}:{a}|{b}"
                market = "round_matchup" if round_no else "tournament_matchup"
                out.extend([
                    OddsQuote(
                        event_id=event_id,
                        market=market,
                        player_name=a,
                        decimal_odds=oa,
                        round_no=round_no,
                        group_id=gid,
                        settlement_rule="push_tie",
                        timestamp=_ts(),
                    ),
                    OddsQuote(
                        event_id=event_id,
                        market=market,
                        player_name=b,
                        decimal_odds=ob,
                        round_no=round_no,
                        group_id=gid,
                        settlement_rule="push_tie",
                        timestamp=_ts(),
                    ),
                ])
        return out

    def load_threeballs(self, path: Path | None = None, event_id: str = "",
                        round_no: int | None = 1) -> list[OddsQuote]:
        path = path or THREEBALLS_CSV
        if not path.exists():
            return []
        out = []
        with path.open() as f:
            for i, row in enumerate(csv.DictReader(f), 1):
                # Keep only filled, validly-priced slots so a twosome (empty
                # player_c/odds_c) loads as a 2-ball rather than being dropped.
                pairs = [
                    (nm, od)
                    for x in "abc"
                    for nm in [(row.get(f"player_{x}") or "").strip()]
                    for od in [_safe_float(row.get(f"odds_{x}"))]
                    if nm and od and od > 1
                ]
                if len(pairs) not in (2, 3):
                    continue
                names = [nm for nm, _ in pairs]
                odds = [od for _, od in pairs]
                gid = row.get("group_id") or f"{_group_market(len(names))}-r{round_no}-{i}:" + "|".join(names)
                for name, price in zip(names, odds):
                    out.append(OddsQuote(
                        event_id=event_id,
                        market=_group_market(len(names)),
                        player_name=name,
                        decimal_odds=float(price),
                        round_no=round_no,
                        group_id=gid,
                        settlement_rule=row.get("settlement_rule") or "dead_heat",
                        timestamp=_ts(),
                    ))
        return out

    def parse_threeball_text(self, text: str, event_id: str = "",
                             round_no: int = 1, book: str = "manual") -> list[OddsQuote]:
        groups = parse_skybet_threeball_text(text)
        out = []
        for group in groups:
            market = _group_market(len(group["players"]))
            gid = f"{market}-r{round_no}:{group['group']}"
            for name, odds in group["players"]:
                out.append(OddsQuote(
                    event_id=event_id,
                    market=market,
                    player_name=name,
                    decimal_odds=odds,
                    round_no=round_no,
                    group_id=gid,
                    book=book,
                    timestamp=_ts(),
                    settlement_rule="dead_heat",
                ))
        return out

    def qa_checks(self, quotes: Iterable[OddsQuote], label: str = "manual_odds") -> list[qa.SourceCheck]:
        rows = [q.as_dict() for q in quotes]
        return [
            qa.require_columns(label, rows, ["market", "player_name", "decimal_odds"]),
            qa.min_rows(label, rows, 1),
        ]


def parse_skybet_threeball_text(text: str) -> list[dict]:
    """Parse pasted Sky Bet-style 3-ball boards.

    Expected shape (odds may be decimal 2.50, fractional 6/5, or evens):
      3 Ball Round 1 - Player A / Player B / Player C
      Player A
      2.50
      Player B
      3.20
      Player C
      4.00
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    groups, cur, pending = [], None, []
    for ln in lines:
        h = HEADER_RE.match(ln)
        if h:
            if cur is not None:
                groups.append(cur)
            cur = {"group": h.group(1).strip(), "players": []}
            pending = []
            continue
        if cur is None:
            continue
        odds = _parse_odds(ln)
        if odds is not None:
            if pending:
                cur["players"].append((pending.pop(0), odds))
        else:
            pending.append(ln)
    if cur is not None:
        groups.append(cur)
    return [g for g in groups if len(g["players"]) in (2, 3)]


def write_threeballs_csv(quotes: Iterable[OddsQuote], path: Path | None = None) -> Path:
    path = path or THREEBALLS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    by_group: dict[str, list[OddsQuote]] = {}
    for q in quotes:
        if q.market in ("2ball", "3ball"):
            by_group.setdefault(q.group_id, []).append(q)
    with path.open("w", newline="") as f:
        cols = [
            "group_id", "player_a", "player_b", "player_c",
            "odds_a", "odds_b", "odds_c", "settlement_rule",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for gid, qs in by_group.items():
            if len(qs) not in (2, 3):
                continue
            row = {"group_id": gid,
                   "settlement_rule": qs[0].settlement_rule or "dead_heat"}
            for slot, q in zip("abc", qs):  # player_c/odds_c stay blank for 2-balls
                row[f"player_{slot}"] = q.player_name
                row[f"odds_{slot}"] = q.decimal_odds
            w.writerow(row)
    return path


def _safe_float(value) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
