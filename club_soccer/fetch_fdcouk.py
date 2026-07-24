#!/usr/bin/env python3
"""Historical closing odds from football-data.co.uk -> data/market_history.csv.

Free, no key. Per-league-season CSVs at
  https://www.football-data.co.uk/mmz4281/{ss}/{code}.csv
e.g. 2526/E0.csv = Premier League 2025-26. Raw files are cached under
data/fdcouk_cache/{ss}_{code}.csv; only the current season's file is
refetched on each run (past seasons are immutable once the season ends).
404s (next season's file not published yet — usually mid-August) are
skipped silently, matching the offline-first / never-raise contract.
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .names import FDCOUK_ALIASES

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = DATA / "fdcouk_cache"
MARKET_HISTORY = DATA / "market_history.csv"

BASE_URL = "https://www.football-data.co.uk/mmz4281/{ss}/{code}.csv"

# fd.co.uk league code -> our Competition name (competitions.py)
CODE_TO_COMP = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "SP1": "La Liga",
    "F1": "Ligue 1",
}

SEASON_STRINGS = ["2223", "2324", "2425", "2526", "2627"]

MARKET_COLUMNS = [
    "match_date", "competition", "home", "away",
    "b365_h", "b365_d", "b365_a",
    "ps_h", "ps_d", "ps_a",
    "psc_h", "psc_d", "psc_a",
    "b365_over25", "b365_under25", "max_over25", "max_under25",
    # Pinnacle CLOSING over/under 2.5 — the totals equivalent of psc_* and the
    # CLV reference the staking gate needs to ever activate the OU2.5 market.
    # Absent for the non-UEFA leagues (they have no fd.co.uk closing feed), so
    # totals CLV is available for the European leagues only.
    "psc_over25", "psc_under25",
    "source_code",
]


def _current_season_string() -> str:
    """The fd.co.uk 'ss' token (e.g. '2526') for the season in progress today."""
    today = datetime.now(timezone.utc).date()
    start_year = today.year if today.month >= 7 else today.year - 1
    yy, yy1 = start_year % 100, (start_year + 1) % 100
    return f"{yy:02d}{yy1:02d}"


def _fetch_raw(ss: str, code: str) -> str | None:
    url = BASE_URL.format(ss=ss, code=code)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        print(f"  fetch_fdcouk: {ss}/{code} HTTP {exc.code} — skipped")
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  fetch_fdcouk: {ss}/{code} network error ({exc}) — skipped")
        return None


def _load_one(ss: str, code: str, refresh: bool) -> tuple[str | None, bool]:
    """Returns (csv_text_or_None, was_freshly_fetched)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ss}_{code}.csv"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8", errors="replace"), False
    raw = _fetch_raw(ss, code)
    if raw is None:
        cached = path.read_text(encoding="utf-8", errors="replace") if path.exists() else None
        return cached, False
    path.write_text(raw, encoding="utf-8")
    return raw, True


def _num(v) -> float | None:
    try:
        f = float(v)
        return f if f > 1.0 else None
    except (TypeError, ValueError):
        return None


def _canon_team(name: str) -> str:
    raw = str(name).strip()
    return FDCOUK_ALIASES.get(raw, raw)


def _parse_csv(raw: str, code: str) -> pd.DataFrame:
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(raw))
    except Exception as exc:
        print(f"  fetch_fdcouk: could not parse {code}.csv ({exc}) — skipped")
        return pd.DataFrame(columns=MARKET_COLUMNS)
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame(columns=MARKET_COLUMNS)

    comp = CODE_TO_COMP[code]
    dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    def col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([None] * len(df))

    rows = pd.DataFrame({
        "match_date": dates.dt.strftime("%Y-%m-%d"),
        "competition": comp,
        "home": col("HomeTeam").map(_canon_team),
        "away": col("AwayTeam").map(_canon_team),
        "b365_h": col("B365H").map(_num), "b365_d": col("B365D").map(_num),
        "b365_a": col("B365A").map(_num),
        "ps_h": col("PSH").map(_num), "ps_d": col("PSD").map(_num),
        "ps_a": col("PSA").map(_num),
        "psc_h": col("PSCH").map(_num), "psc_d": col("PSCD").map(_num),
        "psc_a": col("PSCA").map(_num),
        "b365_over25": col("B365>2.5").map(_num),
        "b365_under25": col("B365<2.5").map(_num),
        "max_over25": col("Max>2.5").map(_num),
        "max_under25": col("Max<2.5").map(_num),
        "psc_over25": col("PC>2.5").map(_num),
        "psc_under25": col("PC<2.5").map(_num),
        "source_code": code,
    })
    return rows.dropna(subset=["match_date"])


def build(refresh_current_only: bool = True, verbose: bool = True) -> pd.DataFrame:
    current_ss = _current_season_string()
    frames: list[pd.DataFrame] = []
    fetched, cached, missing = 0, 0, 0
    for ss in SEASON_STRINGS:
        for code in CODE_TO_COMP:
            refresh = refresh_current_only and ss == current_ss
            raw, was_fetched = _load_one(ss, code, refresh=refresh)
            if raw is None:
                missing += 1
                continue
            if was_fetched:
                fetched += 1
            else:
                cached += 1
            frames.append(_parse_csv(raw, code))
    if verbose:
        print(f"  fetch_fdcouk: {fetched} fetched, {cached} cached, {missing} unavailable "
              f"(404 — season not yet published or code not covered)")
    if not frames:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["match_date", "competition", "home", "away"], keep="last")
    return out[MARKET_COLUMNS]


def unmatched_names(df: pd.DataFrame) -> dict[str, set[str]]:
    """Team names per competition that had no FDCOUK_ALIASES entry AND don't
    already match a name in fixtures.csv — used to grow the alias dict."""
    from . import model as M
    fx = M.load_fixtures()
    result: dict[str, set[str]] = {}
    for comp, grp in df.groupby("competition"):
        known = set(fx[fx["competition"] == comp]["home"]) | set(fx[fx["competition"] == comp]["away"])
        bad = (set(grp["home"]) | set(grp["away"])) - known
        if bad:
            result[comp] = bad
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true",
                    help="use cached files even for the current season")
    ap.add_argument("--show-unmatched", action="store_true",
                    help="print team names with no fixtures.csv match, per competition")
    args = ap.parse_args()
    df = build(refresh_current_only=not args.no_refresh)
    DATA.mkdir(exist_ok=True)
    df.to_csv(MARKET_HISTORY, index=False)
    print(f"Wrote {len(df)} rows -> {MARKET_HISTORY}")
    if args.show_unmatched:
        for comp, names in sorted(unmatched_names(df).items()):
            print(f"  {comp}: {sorted(names)}")


if __name__ == "__main__":
    main()
