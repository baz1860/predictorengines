"""Raw observation store + canonical fixture store (plan §7.2).

Why three stores and not one
----------------------------
`data/results.csv` is a fine historical record and a bad fixture diary. It has no
kick-off time, no timezone, no venue identifier, no provider event ID, no lifecycle
status, no source attribution, no retrieval timestamp and no cancellation history.
Writing future fixtures into it — as an earlier draft of the plan proposed — is how
the July 2026 duplicates happened.

  RawStore        append-only, exactly what a provider returned, timestamped.
                  Never edited. Makes ingest replayable without re-querying, which
                  is the whole basis of the evidence phase being auditable.
  FixtureStore    one row per real match: canonical ID, kick-off UTC, venue and its
                  timezone, neutral flag, lifecycle status, source, conflict note.
  results.csv     unchanged. A fixture is promoted here only once it has finished
                  and passed validation. That promotion step is deliberately NOT
                  implemented yet — it needs the evidence phase first.

Lifecycle
---------
    scheduled -> played      (normal)
              -> postponed   (new date expected; stays in the store)
              -> cancelled   (tombstoned, never silently deleted)
              -> abandoned   (started, not completed)

Tombstoning rather than deleting matters: a cancelled friendly that vanishes from a
feed must be actively retired, otherwise the next ingest re-adds it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from . import timeutil as T
from .identity import canonical_id, signature

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "international"
RAW_DIR = DATA / "raw"
FIXTURES_CSV = DATA / "fixtures.csv"

SCHEDULED, PLAYED, POSTPONED, CANCELLED, ABANDONED = (
    "scheduled", "played", "postponed", "cancelled", "abandoned")
LIFECYCLE = (SCHEDULED, PLAYED, POSTPONED, CANCELLED, ABANDONED)

FIXTURE_COLUMNS = [
    "fixture_id", "signature", "status",
    "kickoff_utc", "local_date", "venue_tz",
    "home_team", "away_team", "competition", "category",
    "city", "country", "neutral",
    "home_score", "away_score",
    "provider", "provider_event_id", "observed_at", "raw_ref", "conflict",
]


def _now() -> str:
    return T.now_iso()


@dataclass
class RawObservation:
    provider: str
    kind: str                 # "fixtures" | "odds" | "lineups" | ...
    payload: object
    observed_at: str = field(default_factory=_now)
    request: dict = field(default_factory=dict)

    def path(self, root: Path = RAW_DIR) -> Path:
        stamp = self.observed_at.replace(":", "").replace("-", "").replace("+0000", "Z")
        return root / self.provider / f"{self.kind}_{stamp}.json"


class RawStore:
    """Append-only. Writing the same observation twice creates two files."""

    def __init__(self, root: Path = RAW_DIR) -> None:
        self.root = root

    def write(self, obs: RawObservation) -> Path:
        path = obs.path(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():                       # collision within one second
            stem, n = path.stem, 2
            while path.exists():
                path = path.with_name(f"{stem}_{n}.json")
                n += 1
        path.write_text(json.dumps(
            {"provider": obs.provider, "kind": obs.kind,
             "observed_at": obs.observed_at, "request": obs.request,
             "payload": obs.payload}, ensure_ascii=False, indent=1))
        return path

    def replay(self, provider: str | None = None, kind: str | None = None):
        """Yield stored observations oldest-first, for offline reprocessing."""
        pattern = f"{provider or '*'}/{kind or '*'}_*.json"
        for path in sorted(self.root.glob(pattern)):
            yield json.loads(path.read_text())


def normalize_fixture(*, home: str, away: str, competition: str,
                      kickoff_utc: object = None, local_date: object = None,
                      venue_tz: str = "", city: str = "", country: str = "",
                      neutral: object = False, home_score: object = None,
                      away_score: object = None, provider: str = "",
                      provider_event_id: object = None, raw_ref: str = "",
                      status: str | None = None, observed_at: str | None = None,
                      ) -> dict:
    """One provider record -> one canonical fixture row.

    Requires at least one of kickoff_utc / local_date. `local_date` alone is
    accepted but flagged in `conflict`, because a date without a timezone is
    precisely what produced the July 2026 duplicates.
    """
    from . import taxonomy

    if kickoff_utc is None and local_date is None:
        raise ValueError("a fixture needs kickoff_utc or local_date")

    ko = T.to_utc(kickoff_utc) if kickoff_utc is not None else None

    ld = str(pd.Timestamp(local_date).date()) if local_date is not None else \
        str(ko.date())

    scored = home_score is not None and away_score is not None
    if status is None:
        status = PLAYED if scored else SCHEDULED
    if status not in LIFECYCLE:
        raise ValueError(f"status {status!r} not in {LIFECYCLE}")

    conflict = ""
    if ko is None:
        conflict = "no kickoff_utc supplied; identity falls back to local date"

    return {
        "fixture_id": canonical_id(home, away, competition,
                                   kickoff_utc=ko, provider_event_id=provider_event_id,
                                   date=ld),
        "signature": signature(home, away, competition),
        "status": status,
        "kickoff_utc": ko.isoformat() if ko is not None else "",
        "local_date": ld,
        "venue_tz": venue_tz,
        "home_team": home, "away_team": away, "competition": competition,
        "category": taxonomy.category(competition),
        "city": city, "country": country, "neutral": bool(neutral),
        "home_score": home_score, "away_score": away_score,
        "provider": provider, "provider_event_id": provider_event_id or "",
        "observed_at": observed_at or _now(),
        "raw_ref": raw_ref, "conflict": conflict,
    }


class FixtureStore:
    """Canonical fixtures, keyed on fixture_id, with lifecycle transitions."""

    def __init__(self, path: Path = FIXTURES_CSV) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=FIXTURE_COLUMNS)
        return pd.read_csv(self.path, dtype={"provider_event_id": str})

    def upsert(self, rows: list[dict]) -> dict[str, int]:
        """Insert or update by fixture_id. Returns a small change summary."""
        cur = self.load()
        incoming = pd.DataFrame(rows).reindex(columns=FIXTURE_COLUMNS)
        if cur.empty:
            merged, added, updated = incoming, len(incoming), 0
        else:
            known = set(cur.fixture_id)
            new = incoming[~incoming.fixture_id.isin(known)]
            upd = incoming[incoming.fixture_id.isin(known)]
            cur = cur.set_index("fixture_id")
            for r in upd.to_dict("records"):
                fid = r["fixture_id"]
                for k, v in r.items():
                    if k != "fixture_id" and v not in (None, ""):
                        cur.at[fid, k] = v
            # Cast both sides to object before concat: a column that is entirely
            # blank on one side loads as float64, and concatenating mismatched
            # dtypes is deprecated. The CSV round-trip re-infers types anyway.
            frames = [f.reindex(columns=FIXTURE_COLUMNS).astype(object)
                      for f in (cur.reset_index(), new) if not f.empty]
            merged = pd.concat(frames, ignore_index=True) if frames else cur.reset_index()
            added, updated = len(new), len(upd)

        merged = merged.reindex(columns=FIXTURE_COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged.sort_values(["local_date", "home_team"], kind="stable").to_csv(
            self.path, index=False)
        return {"added": added, "updated": updated, "total": len(merged)}

    def tombstone(self, fixture_id: str, status: str = CANCELLED,
                  note: str = "") -> bool:
        """Retire a fixture explicitly. Never delete — a deleted row comes back."""
        if status not in (CANCELLED, POSTPONED, ABANDONED):
            raise ValueError(f"{status!r} is not a retirement status")
        df = self.load()
        hit = df.fixture_id == fixture_id
        if not hit.any():
            return False
        # `conflict` may have loaded as float64 if every stored value was blank.
        df["conflict"] = df["conflict"].astype("object")
        df.loc[hit, "status"] = status
        df.loc[hit, "conflict"] = note or f"retired as {status}"
        df.to_csv(self.path, index=False)
        return True

    def upcoming(self, asof: object = None) -> pd.DataFrame:
        """Fixtures genuinely still to be played.

        Unlike `predictor.load_matches()`, this tests STATUS and DATE, not just a
        blank score — the two omissions behind the July 2026 zombie fixtures.
        """
        df = self.load()
        if df.empty:
            return df
        now = T.naive_utc(asof)          # local_date is a naive date string
        dates = pd.to_datetime(df.local_date, errors="coerce")
        return df[(df.status == SCHEDULED) & (dates >= now.normalize())]
