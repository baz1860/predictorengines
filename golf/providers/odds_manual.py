"""Manual bookmaker odds provider.

Free automation can model 3-ball probabilities, but free bookmaker odds for
3-balls and matchups are not reliably exposed. This provider makes pasted/CSV
boards first-class inputs with schema validation and normalized quote rows.
"""

from __future__ import annotations

import csv
import re
import datetime as dt
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .. import provider_qa as qa

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ODDS_CSV = DATA_DIR / "odds.csv"
MATCHUPS_CSV = DATA_DIR / "matchups.csv"
# Legacy single-round board paths. Round boards are now per-round files
# (threeballs_r{N}.csv / threeballs_r{N}_raw.txt) so a round-1 threesome board
# cannot bleed into a round-3 pricing run. These names are kept for back-compat
# and as the round-1 fallback.
THREEBALLS_CSV = DATA_DIR / "threeballs.csv"
THREEBALLS_RAW = DATA_DIR / "threeballs_r1_raw.txt"


def threeballs_csv_path(round_no) -> Path:
    """Parsed round-group board file for a given round."""
    try:
        n = int(round_no)
    except (TypeError, ValueError):
        n = 1
    return DATA_DIR / f"threeballs_r{n}.csv"


def threeballs_raw_path(round_no) -> Path:
    """Raw paste file for a given round's tee groups."""
    try:
        n = int(round_no)
    except (TypeError, ValueError):
        n = 1
    return DATA_DIR / f"threeballs_r{n}_raw.txt"


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


HEADER_RE = re.compile(r"^([23])\s*Ball.*-\s*(.+)$", re.I)
NUM_RE = re.compile(r"^\d+(\.\d+)?$")
FRAC_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$")


def _parse_odds(token: str) -> float | None:
    """Decimal odds from a pasted token. Accepts decimal (2.50), UK fractional
    (6/5, 13/8, 4/6) and evens. Returns None if the token isn't a price."""
    t = token.strip()
    if t.lower() in {"evens", "evs", "even"}:
        return 2.0
    if NUM_RE.match(t):
        value = float(t)
        return value if value > 1.0 else None
    m = FRAC_RE.match(t)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den > 0:
            value = 1.0 + num / den
            return value if value > 1.0 else None
    return None


def _group_market(n_players: int) -> str:
    """Round-group market tag by field size: twosomes vs threesomes."""
    return "2ball" if n_players == 2 else "3ball"


# Rounds played after the 36-hole cut. The field is halved and everyone goes
# out in twosomes, so a 3-ball group tagged for one of these rounds is an
# impossible market — in practice a leftover round-1/2 board being re-priced.
POST_CUT_ROUNDS = (3, 4)


def post_cut_round(round_no) -> bool:
    """True if round_no is a post-cut round (played in 2-balls)."""
    try:
        return int(round_no) in POST_CUT_ROUNDS
    except (TypeError, ValueError):
        return False


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
                            book=(row.get("book") or "manual").strip(),
                            source=(row.get("source") or "manual").strip(),
                            timestamp=(row.get("captured_at")
                                       or row.get("timestamp") or _ts()).strip(),
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
                captured_at = (row.get("captured_at")
                               or row.get("timestamp") or _ts()).strip()
                book = (row.get("book") or "manual").strip()
                source = (row.get("source") or "manual").strip()
                out.extend([
                    OddsQuote(
                        event_id=event_id,
                        market=market,
                        player_name=a,
                        decimal_odds=oa,
                        round_no=round_no,
                        group_id=gid,
                        book=book,
                        source=source,
                        settlement_rule="push_tie",
                        timestamp=captured_at,
                    ),
                    OddsQuote(
                        event_id=event_id,
                        market=market,
                        player_name=b,
                        decimal_odds=ob,
                        round_no=round_no,
                        group_id=gid,
                        book=book,
                        source=source,
                        settlement_rule="push_tie",
                        timestamp=captured_at,
                    ),
                ])
        return out

    def load_threeballs(self, path: Path | None = None, event_id: str = "",
                        round_no: int | None = 1) -> list[OddsQuote]:
        if path is None:
            path = threeballs_csv_path(round_no)
            # Round-1 back-compat: fall back to the legacy single-board file if
            # no per-round file has been written yet. Later rounds get no such
            # fallback, so a leftover round-1 board can't serve round 3/4.
            if not path.exists() and int(round_no or 1) == 1 and THREEBALLS_CSV.exists():
                path = THREEBALLS_CSV
        if not path.exists():
            return []
        out = []
        with path.open() as f:
            for i, row in enumerate(csv.DictReader(f), 1):
                populated = []
                malformed = False
                for slot in "abc":
                    name = (row.get(f"player_{slot}") or "").strip()
                    raw_odds = (row.get(f"odds_{slot}") or "").strip()
                    if not name and not raw_odds:
                        continue
                    odds = _safe_float(raw_odds)
                    if not name or odds is None or odds <= 1.0:
                        malformed = True
                        break
                    populated.append((name, odds))
                if malformed:
                    continue
                pairs = populated
                if len(pairs) not in (2, 3):
                    continue
                names = [nm for nm, _ in pairs]
                odds = [od for _, od in pairs]
                gid = row.get("group_id") or f"{_group_market(len(names))}-r{round_no}-{i}:" + "|".join(names)
                tagged = re.search(r"(?<!\d)([23])\s*ball", gid, re.I)
                if tagged and int(tagged.group(1)) != len(pairs):
                    continue
                for name, price in zip(names, odds):
                    out.append(OddsQuote(
                        event_id=event_id,
                        market=_group_market(len(names)),
                        player_name=name,
                        decimal_odds=float(price),
                        round_no=round_no,
                        group_id=gid,
                        book=(row.get("book") or "manual").strip(),
                        source=(row.get("source") or "manual").strip(),
                        settlement_rule=row.get("settlement_rule") or "dead_heat",
                        timestamp=(row.get("captured_at")
                                   or row.get("timestamp") or _ts()).strip(),
                    ))
        return out

    def parse_threeball_text(self, text: str, event_id: str = "",
                             round_no: int = 1, book: str = "manual") -> list[OddsQuote]:
        issues: list[str] = []
        groups = parse_skybet_threeball_text(text, issues=issues)
        self.last_parse_issues = issues
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


def parse_skybet_threeball_text(text: str, issues: list[str] | None = None) -> list[dict]:
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
    groups, cur = [], None

    def finish(group):
        if group is None:
            return
        expected = group.pop("_expected")
        invalid = group.pop("_invalid")
        pending_name = group.pop("_pending", None)
        header_names = group.pop("_header_names")
        parsed_names = [norm_event(n) for n, _ in group["players"]]
        if pending_name:
            invalid = f"missing odds after '{pending_name}'"
        if len(group["players"]) != expected:
            invalid = invalid or f"expected {expected} selections, parsed {len(group['players'])}"
        if len(header_names) != expected:
            invalid = invalid or f"header names {len(header_names)} disagree with {expected}-ball"
        header_keys = [norm_event(n) for n in header_names]
        if any(h not in p and p not in h for h, p in zip(header_keys, parsed_names)):
            invalid = invalid or "selection order/names disagree with the group header"
        if invalid:
            if issues is not None:
                issues.append(f"{group['group']}: {invalid}")
            return
        groups.append(group)

    for ln in lines:
        h = HEADER_RE.match(ln)
        if h:
            finish(cur)
            label = h.group(2).strip()
            cur = {"group": label, "players": [], "_expected": int(h.group(1)),
                   "_header_names": [p.strip() for p in label.split("/") if p.strip()],
                   "_pending": None, "_invalid": ""}
            continue
        if cur is None:
            continue
        odds = _parse_odds(ln)
        if odds is not None:
            if cur["_pending"] is None:
                cur["_invalid"] = cur["_invalid"] or f"odds '{ln}' without a player"
            else:
                cur["players"].append((cur["_pending"], odds))
                cur["_pending"] = None
        else:
            if cur["_pending"] is not None:
                cur["_invalid"] = cur["_invalid"] or (
                    f"missing odds after '{cur['_pending']}' before '{ln}'")
            cur["_pending"] = ln
    finish(cur)
    return groups


def write_threeballs_csv(quotes: Iterable[OddsQuote], path: Path | None = None,
                         event: str = "", round_no: int | None = None) -> Path:
    quotes = list(quotes)
    if path is None:
        # Default to the per-round board file. Prefer an explicit round_no, else
        # infer it from the quotes so each round's board lands in its own file.
        if round_no is None:
            round_no = next((q.round_no for q in quotes if q.round_no), 1)
        path = threeballs_csv_path(round_no)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_group: dict[str, list[OddsQuote]] = {}
    for q in quotes:
        if q.market in ("2ball", "3ball"):
            by_group.setdefault(q.group_id, []).append(q)
    with path.open("w", newline="") as f:
        cols = [
            "group_id", "player_a", "player_b", "player_c",
            "odds_a", "odds_b", "odds_c", "settlement_rule",
            "book", "source", "event", "captured_at",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for gid, qs in by_group.items():
            if len(qs) not in (2, 3):
                continue
            row = {"group_id": gid,
                   "settlement_rule": qs[0].settlement_rule or "dead_heat",
                   "book": qs[0].book,
                   "source": qs[0].source,
                   "event": event,
                   "captured_at": qs[0].timestamp or _ts()}
            for slot, q in zip("abc", qs):  # player_c/odds_c stay blank for 2-balls
                row[f"player_{slot}"] = q.player_name
                row[f"odds_{slot}"] = q.decimal_odds
            w.writerow(row)
    return path


def board_event(path: Path) -> str:
    """The event tag a board CSV was written under ('' when untagged).

    Boards are only priceable against the event they were captured for —
    player-name overlap cannot tell consecutive events apart when fields
    overlap (e.g. co-sanctioned weeks), so pricers compare this tag to the
    current event instead.
    """
    if not path.exists():
        return ""
    with path.open() as f:
        for row in csv.DictReader(f):
            ev = (row.get("event") or "").strip()
            if ev:
                return ev
    return ""


def board_captured_at(path: Path) -> dt.datetime | None:
    """Capture time embedded in a board; filesystem mtimes are not provenance."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            for row in csv.DictReader(f):
                raw = (row.get("captured_at") or row.get("timestamp") or "").strip()
                if raw:
                    parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
                    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except (OSError, csv.Error, ValueError):
        return None
    return None


def norm_event(name: str) -> str:
    """Case/spacing-insensitive event-name key for staleness comparisons."""
    return " ".join(str(name or "").split()).casefold()


def _safe_float(value) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _ts() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
