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


HEADER_RE = re.compile(r"^3\s*Ball.*-\s*(.+)$", re.I)
NUM_RE = re.compile(r"^\d+(\.\d+)?$")


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
                names = [(row.get(f"player_{x}") or "").strip() for x in "abc"]
                odds = [_safe_float(row.get(f"odds_{x}")) for x in "abc"]
                if not all(names) or not all(o and o > 1 for o in odds):
                    continue
                gid = row.get("group_id") or f"3ball-r{round_no}-{i}:" + "|".join(names)
                for name, price in zip(names, odds):
                    out.append(OddsQuote(
                        event_id=event_id,
                        market="3ball",
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
            gid = f"3ball-r{round_no}:{group['group']}"
            for name, odds in group["players"]:
                out.append(OddsQuote(
                    event_id=event_id,
                    market="3ball",
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

    Expected shape:
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
        if NUM_RE.match(ln):
            if pending:
                cur["players"].append((pending.pop(0), float(ln)))
        else:
            pending.append(ln)
    if cur is not None:
        groups.append(cur)
    return [g for g in groups if len(g["players"]) == 3]


def write_threeballs_csv(quotes: Iterable[OddsQuote], path: Path | None = None) -> Path:
    path = path or THREEBALLS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    by_group: dict[str, list[OddsQuote]] = {}
    for q in quotes:
        if q.market == "3ball":
            by_group.setdefault(q.group_id, []).append(q)
    with path.open("w", newline="") as f:
        cols = [
            "group_id", "player_a", "player_b", "player_c",
            "odds_a", "odds_b", "odds_c", "settlement_rule",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for gid, qs in by_group.items():
            if len(qs) != 3:
                continue
            w.writerow({
                "group_id": gid,
                "player_a": qs[0].player_name,
                "player_b": qs[1].player_name,
                "player_c": qs[2].player_name,
                "odds_a": qs[0].decimal_odds,
                "odds_b": qs[1].decimal_odds,
                "odds_c": qs[2].decimal_odds,
                "settlement_rule": qs[0].settlement_rule or "dead_heat",
            })
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
