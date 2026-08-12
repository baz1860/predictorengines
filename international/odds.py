"""Odds snapshot store for international fixtures.

Why this exists
---------------
Plan §2 measures the model's skill against a naive baseline and finds it near zero
once the baseline is given the same Elo ratings. The question that actually decides
whether this project is worth building — does the model have *edge against market
prices* — cannot be answered from inside our own data. It needs odds history, and
outside the World Cup we have none: 1,196 price points across 59 matches.

Every week without capture is a week that question stays open. So this store exists
before there is anything to put in it.

Absence is data
---------------
As of 8 August 2026, BSD returns **no odds at all** for September internationals:
all eleven inline odds fields are null and `/api/odds/?event=` returns `count: 0`.
That is expected six weeks out, but it is also exactly the thing the evidence phase
has to measure, so `record_absence()` writes a real row saying "asked, got nothing"
rather than writing nothing. A gap in the data must be distinguishable from a gap in
the collection.

Shape
-----
Append-only snapshots, one row per (fixture, market, side, bookmaker, snapshot).
Never updated in place: the whole point is the time series from first price to
closing price, because closing-line value is the only honest measure of edge we can
eventually compute.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import timeutil as T

ROOT = Path(__file__).resolve().parents[1]
ODDS_CSV = ROOT / "data" / "international" / "odds_snapshots.csv"

COLUMNS = [
    "snapshot_at", "fixture_id", "provider_event_id", "kickoff_utc",
    "home_team", "away_team", "competition",
    "market", "side", "line", "odds", "bookmaker",
    "provider", "hours_to_kickoff", "status",
]

# `status` values — absence is a first-class observation.
PRICED = "priced"
NO_ODDS = "no_odds_returned"
FETCH_FAILED = "fetch_failed"

# BSD inline odds field -> (market, side, line)
BSD_INLINE = {
    "odds_home": ("h2h", "home", ""),
    "odds_draw": ("h2h", "draw", ""),
    "odds_away": ("h2h", "away", ""),
    "odds_btts_yes": ("btts", "yes", ""),
    "odds_btts_no": ("btts", "no", ""),
    "odds_over_15": ("totals", "over", "1.5"),
    "odds_under_15": ("totals", "under", "1.5"),
    "odds_over_25": ("totals", "over", "2.5"),
    "odds_under_25": ("totals", "under", "2.5"),
    "odds_over_35": ("totals", "over", "3.5"),
    "odds_under_35": ("totals", "under", "3.5"),
}


def _now() -> str:
    return T.now_iso()


def _hours_to(kickoff: object, at: object = None) -> float | str:
    hours = T.hours_between(kickoff, at)
    return "" if hours is None else round(hours, 2)


def parse_bsd_inline(event: dict, fixture: dict, snapshot_at: str | None = None,
                     bookmaker: str = "bsd_implied") -> list[dict]:
    """Rows from BSD's inline `odds_*` fields on a v1 event detail payload.

    These are a single implied price with no bookmaker attribution. Useful as a
    market *reference*, NOT as an executable price and not as a substitute for
    multi-book data — recorded with an explicit synthetic bookmaker name so nobody
    later mistakes it for a real book.
    """
    at = snapshot_at or _now()
    base = {
        "snapshot_at": at,
        "fixture_id": fixture.get("fixture_id", ""),
        "provider_event_id": fixture.get("provider_event_id", ""),
        "kickoff_utc": fixture.get("kickoff_utc", ""),
        "home_team": fixture.get("home_team", ""),
        "away_team": fixture.get("away_team", ""),
        "competition": fixture.get("competition", ""),
        "bookmaker": bookmaker,
        "provider": "bsd",
        "hours_to_kickoff": _hours_to(fixture.get("kickoff_utc"), at),
    }
    rows = []
    for field, (market, side, line) in BSD_INLINE.items():
        value = event.get(field)
        if value is None or value == "":
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if price <= 1.0:                      # decimal odds must exceed 1.0
            continue
        rows.append({**base, "market": market, "side": side, "line": line,
                     "odds": price, "status": PRICED})
    return rows


def parse_bsd_comparison(payload: dict, fixture: dict,
                         snapshot_at: str | None = None) -> list[dict]:
    """Rows from `/api/odds/?event=` — the multi-bookmaker endpoint.

    Returns `{"count": 0, "odds": []}` for every international fixture tested on
    8 August 2026. The parser exists so that the day it starts returning data we
    capture it without a code change; until then `count: 0` is recorded by
    `record_absence()`.
    """
    at = snapshot_at or _now()
    base = {
        "snapshot_at": at,
        "fixture_id": fixture.get("fixture_id", ""),
        "provider_event_id": fixture.get("provider_event_id", ""),
        "kickoff_utc": fixture.get("kickoff_utc", ""),
        "home_team": fixture.get("home_team", ""),
        "away_team": fixture.get("away_team", ""),
        "competition": fixture.get("competition", ""),
        "provider": "bsd",
        "hours_to_kickoff": _hours_to(fixture.get("kickoff_utc"), at),
    }
    rows = []
    for entry in (payload or {}).get("odds", []) or []:
        book = str(entry.get("bookmaker") or entry.get("bookmaker_name") or "unknown")
        for field, (market, side, line) in BSD_INLINE.items():
            value = entry.get(field)
            if value in (None, ""):
                continue
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if price <= 1.0:
                continue
            rows.append({**base, "bookmaker": book, "market": market,
                         "side": side, "line": line, "odds": price,
                         "status": PRICED})
    return rows


def record_absence(fixture: dict, snapshot_at: str | None = None,
                   provider: str = "bsd", status: str = NO_ODDS) -> dict:
    """One row meaning "we asked at this time and there were no prices".

    Without this the dataset cannot distinguish "no odds existed" from "we did not
    look", and the first is a finding while the second is a bug.
    """
    at = snapshot_at or _now()
    return {
        "snapshot_at": at,
        "fixture_id": fixture.get("fixture_id", ""),
        "provider_event_id": fixture.get("provider_event_id", ""),
        "kickoff_utc": fixture.get("kickoff_utc", ""),
        "home_team": fixture.get("home_team", ""),
        "away_team": fixture.get("away_team", ""),
        "competition": fixture.get("competition", ""),
        "market": "", "side": "", "line": "", "odds": "",
        "bookmaker": "", "provider": provider,
        "hours_to_kickoff": _hours_to(fixture.get("kickoff_utc"), at),
        "status": status,
    }


class OddsStore:
    """Append-only snapshot history."""

    def __init__(self, path: Path = ODDS_CSV) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        return pd.read_csv(self.path, dtype={"provider_event_id": str})

    def append(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        df = pd.DataFrame(rows).reindex(columns=COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.path.exists()
        df.to_csv(self.path, mode="a", header=header, index=False)
        return len(df)

    def coverage(self) -> dict:
        df = self.load()
        if df.empty:
            return {"snapshots": 0, "fixtures": 0, "priced_fixtures": 0,
                    "absence_rows": 0, "bookmakers": 0}
        priced = df[df.status == PRICED]
        return {
            "snapshots": int(df.snapshot_at.nunique()),
            "fixtures": int(df.fixture_id.nunique()),
            "priced_fixtures": int(priced.fixture_id.nunique()),
            "absence_rows": int((df.status != PRICED).sum()),
            "bookmakers": int(priced.bookmaker.nunique()) if len(priced) else 0,
        }

    def first_price_timing(self) -> pd.DataFrame:
        """When does a price first appear, per fixture and per competition?

        This is the measurement the "wait on BSD" decision rests on. If prices
        never appear, this frame stays empty and that IS the answer. If they appear
        at T-3 days, the polling schedule and any betting workflow have to be built
        around that, and we would rather learn it from data than assume it.
        """
        df = self.load()
        priced = df[df.status == PRICED]
        if priced.empty:
            return pd.DataFrame(columns=["fixture_id", "competition",
                                         "first_seen_at", "hours_before_kickoff"])
        first = (priced.sort_values("snapshot_at")
                       .groupby("fixture_id", as_index=False)
                       .first()[["fixture_id", "competition", "snapshot_at",
                                 "hours_to_kickoff"]])
        return first.rename(columns={"snapshot_at": "first_seen_at",
                                     "hours_to_kickoff": "hours_before_kickoff"})

    def absence_profile(self) -> pd.DataFrame:
        """How often we asked and got nothing, bucketed by time to kickoff.

        Distinguishes "prices do not exist this far out" from "prices never exist".
        """
        df = self.load()
        if df.empty:
            return pd.DataFrame(columns=["bucket", "asked", "priced", "priced_share"])
        hours = pd.to_numeric(df.hours_to_kickoff, errors="coerce")
        bucket = pd.cut(hours, [-1, 6, 24, 72, 168, 336, 720, 1e9],
                        labels=["<6h", "6-24h", "1-3d", "3-7d", "1-2w",
                                "2w-30d", ">30d"])
        out = (df.assign(bucket=bucket, is_priced=df.status == PRICED)
                 .groupby("bucket", observed=False)
                 .agg(asked=("status", "size"), priced=("is_priced", "sum")))
        # `replace(0, pd.NA)` turns the column to object dtype, and dividing by it
        # yields object, which `.round()` rejects. Divide then mask instead.
        share = out.priced.astype(float) / out.asked.astype(float)
        out["priced_share"] = share.where(out.asked > 0).round(3)
        return out.reset_index()

    def closing_prices(self) -> pd.DataFrame:
        """Last priced snapshot per (fixture, market, side, bookmaker).

        The teacher signal for closing-line value. Empty until prices exist.

        Note the NaN-key trap: `line` is blank for h2h and btts markets, which the
        CSV round-trip turns into NaN, and pandas `groupby` drops NaN keys by
        default. Without `dropna=False` this silently returned an EMPTY frame for
        exactly the market we care most about — the one used to price a match.
        """
        df = self.load()
        df = df[df.status == PRICED].copy()
        if df.empty:
            return df
        df["line"] = df["line"].fillna("").astype(str)
        df = df.sort_values("snapshot_at")
        return df.groupby(["fixture_id", "market", "side", "line", "bookmaker"],
                          as_index=False, dropna=False).last()
