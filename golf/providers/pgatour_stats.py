"""Public PGA Tour stats-page provider.

This is a free-source replacement for paid strokes-gained feeds. It scrapes the
public PGA Tour stat detail pages and stores raw pages before parsing. The
parser deliberately accepts multiple page shapes because pgatour.com has changed
its frontend before.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from .. import provider_qa as qa

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "api_cache" / "pgatour_stats"
STATS_CSV = DATA_DIR / "pgatour_stats.csv"

# Public stat IDs. These are treated as configuration, not magic constants; if
# PGA Tour changes them, provider QA will fail loudly and the cache preserves
# the page that broke.
STAT_IDS = {
    "sg_total": "02675",
    "sg_t2g": "02674",
    "sg_ott": "02567",
    "sg_app": "02568",
    "sg_arg": "02569",
    "sg_putt": "02564",
    "scoring_average": "120",
    "birdie_average": "156",
    "bogey_avoidance": "02415",
    "driving_distance": "101",
    "driving_accuracy": "102",
    "gir": "103",
    "scrambling": "130",
    "owgr": "186",
}


@dataclass(frozen=True)
class StatRow:
    stat_id: str
    stat_name: str
    player_name: str
    rank: int | None
    value: float | None
    values: dict[str, Any]
    season: int | None = None
    source: str = "pgatour"

    def as_dict(self) -> dict:
        row = asdict(self)
        row["raw_json"] = json.dumps(self.values, sort_keys=True)
        return row


class PgaTourStatsProvider:
    name = "pgatour_stats"

    def __init__(self, cache_dir: Path | None = None, ttl_seconds: int = 86_400):
        self.cache_dir = cache_dir or CACHE_DIR
        self.ttl_seconds = ttl_seconds

    def fetch_stat(self, stat: str, season: int | None = None,
                   use_cache: bool = True) -> list[StatRow]:
        stat_id = STAT_IDS.get(stat, stat)
        raw = self._fetch_html(stat_id, season=season, use_cache=use_cache)
        rows = parse_stat_page(raw, stat_id=stat_id, stat_name=stat, season=season)
        return rows

    def fetch_default_stats(self, season: int | None = None,
                            use_cache: bool = True) -> dict[str, list[StatRow]]:
        out = {}
        for stat in (
            "sg_total", "sg_t2g", "sg_ott", "sg_app", "sg_arg", "sg_putt",
            "scoring_average", "birdie_average", "bogey_avoidance",
            "driving_distance", "driving_accuracy", "gir", "scrambling", "owgr",
        ):
            try:
                out[stat] = self.fetch_stat(stat, season=season, use_cache=use_cache)
            except Exception:
                out[stat] = []
        return out

    def qa_checks(self, rows: Iterable[StatRow], label: str = "pgatour_stats") -> list[qa.SourceCheck]:
        dicts = [r.as_dict() for r in rows]
        return [
            qa.require_columns(label, dicts, ["stat_id", "player_name", "value"]),
            qa.min_rows(label, dicts, 20),
        ]

    def _fetch_html(self, stat_id: str, season: int | None = None,
                    use_cache: bool = True) -> str:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{season}" if season else ""
        cache = self.cache_dir / f"stat_{stat_id}{suffix}.html"
        if use_cache and cache.exists() and time.time() - cache.stat().st_mtime <= self.ttl_seconds:
            return cache.read_text(errors="replace")
        params = {"year": season} if season else {}
        query = urllib.parse.urlencode(params)
        url = f"https://www.pgatour.com/stats/detail/{stat_id}"
        if query:
            url += f"?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        cache.write_text(raw)
        return raw


def parse_stat_page(raw_html: str, stat_id: str, stat_name: str = "",
                    season: int | None = None) -> list[StatRow]:
    rows = _parse_embedded_json(raw_html, stat_id, stat_name, season)
    if rows:
        return rows
    return _parse_text_table(raw_html, stat_id, stat_name, season)


def write_stats_csv(rows: Iterable[StatRow], path: Path | None = None) -> Path:
    import csv

    path = path or STATS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["season", "stat_id", "stat_name", "player_name", "rank", "value", "raw_json", "source"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r.as_dict())
    return path


def _parse_embedded_json(raw_html: str, stat_id: str, stat_name: str,
                         season: int | None) -> list[StatRow]:
    blobs = []
    for match in re.finditer(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>',
                             raw_html, flags=re.S | re.I):
        text = html.unescape(match.group(1)).strip()
        if not text:
            continue
        try:
            blobs.append(json.loads(text))
        except ValueError:
            continue
    next_match = re.search(r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                           raw_html, flags=re.S | re.I)
    if next_match:
        try:
            blobs.append(json.loads(html.unescape(next_match.group(1))))
        except ValueError:
            pass

    candidates = []
    for blob in blobs:
        _walk_json(blob, candidates)

    rows = []
    seen = set()
    for item in candidates:
        name = _extract_name(item)
        if not name:
            continue
        rank = _extract_rank(item)
        values = _extract_values(item)
        value = _primary_value(values)
        key = (name, rank, value)
        if key in seen:
            continue
        seen.add(key)
        rows.append(StatRow(stat_id, stat_name or stat_id, name, rank, value, values, season))
    rows.sort(key=lambda r: (r.rank is None, r.rank or 9999, r.player_name))
    return rows


def _parse_text_table(raw_html: str, stat_id: str, stat_name: str,
                      season: int | None) -> list[StatRow]:
    parser = _TextParser()
    parser.feed(raw_html)
    tokens = [t for t in parser.tokens if t and not t.startswith(".css-")]
    rows = []
    i = 0
    while i < len(tokens) - 2:
        rank = _safe_int(tokens[i])
        if rank is None:
            i += 1
            continue
        # Pages usually emit Rank, optional change, Player, value, label.
        window = tokens[i + 1:i + 8]
        player_idx = None
        for j, token in enumerate(window):
            if _looks_like_player(token):
                player_idx = j
                break
        if player_idx is None:
            i += 1
            continue
        name = window[player_idx]
        value = None
        values = {}
        for token in window[player_idx + 1:]:
            value = _safe_float(token)
            if value is not None:
                values["value"] = value
                break
        if value is not None:
            rows.append(StatRow(stat_id, stat_name or stat_id, name, rank, value, values, season))
            i += player_idx + 3
        else:
            i += 1
    return _dedupe_rows(rows)


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tokens: list[str] = []

    def handle_data(self, data: str) -> None:
        text = html.unescape(data)
        for part in re.split(r"[\n\r\t]+", text):
            clean = " ".join(part.split())
            if clean:
                self.tokens.append(clean)


def _walk_json(value: Any, out: list[dict]) -> None:
    if isinstance(value, dict):
        if _extract_name(value) and (_extract_values(value) or _extract_rank(value) is not None):
            out.append(value)
        for child in value.values():
            _walk_json(child, out)
    elif isinstance(value, list):
        for child in value:
            _walk_json(child, out)


def _extract_name(item: dict) -> str:
    for key in ("playerName", "player_name", "displayName", "name", "competitorName"):
        val = item.get(key)
        if isinstance(val, str) and _looks_like_player(val):
            return val.strip()
    for key in ("player", "athlete", "competitor"):
        val = item.get(key)
        if isinstance(val, dict):
            nested = _extract_name(val)
            if nested:
                return nested
    return ""


def _extract_rank(item: dict) -> int | None:
    for key in ("rank", "rankThisWeek", "position"):
        r = _safe_int(item.get(key))
        if r is not None:
            return r
    return None


def _extract_values(item: dict) -> dict:
    out = {}
    for key, value in item.items():
        if isinstance(value, (str, int, float)):
            num = _safe_float(value)
            if num is not None and key.lower() not in {"id", "playerid", "rank"}:
                out[key] = num
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, dict):
                    nm = child.get("statName") or child.get("name") or child.get("label")
                    val = _safe_float(child.get("statValue") or child.get("value"))
                    if nm and val is not None:
                        out[str(nm)] = val
    return out


def _primary_value(values: dict) -> float | None:
    for key in ("Avg", "AVG", "average", "value", "statValue"):
        if key in values:
            return _safe_float(values[key])
    for value in values.values():
        num = _safe_float(value)
        if num is not None:
            return num
    return None


def _looks_like_player(token: str) -> bool:
    token = str(token or "").strip()
    if len(token) < 4 or len(token) > 40:
        return False
    if token.lower() in {"rank", "player", "avg", "total", "view full standings"}:
        return False
    if _safe_float(token) is not None:
        return False
    return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", token)) and " " in token


def _dedupe_rows(rows: list[StatRow]) -> list[StatRow]:
    seen = set()
    out = []
    for row in rows:
        key = (row.stat_id, row.player_name, row.rank)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    out.sort(key=lambda r: (r.rank is None, r.rank or 9999, r.player_name))
    return out


def _safe_int(value) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    try:
        if value in ("", None):
            return None
        s = str(value).replace(",", "").replace("%", "").strip()
        if s in {"-", "--"}:
            return None
        return float(s)
    except (TypeError, ValueError):
        return None
