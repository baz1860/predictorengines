"""The single source of truth for time in the international module.

The rule
--------
**Every instant is stored, compared and reasoned about in UTC.** There is exactly
one exception, and it is a boundary, not a compromise:

    data/results.csv is dated in LOCAL time.

That is the upstream martj42 convention and we do not control it. Writing UTC dates
into that file would put us one day ahead of upstream on every evening kick-off in
the Americas, which is precisely the disagreement that duplicated two World Cup
fixtures. So `local_date()` exists, is used ONLY at that boundary, and always
records which timezone produced it.

Why this module exists
----------------------
Before it, thirty-plus ad-hoc conversions were scattered across ten files —
`tz_localize` here, `astimezone` there, `pd.Timestamp.utcnow()` in five places,
and a hardcoded `timezone(timedelta(hours=-7))` in the World Cup fetcher that
treated every venue on earth as Californian. Each re-implemented the same
normalisation slightly differently, and three silent bugs came out of the gaps:

  * a naive/aware comparison that raised only on a code path with no test;
  * `tz_of()` returning the string "nan", which failed conversion and fell back
    to UTC without saying so;
  * a `NaN >= interval` comparison that silently scheduled nothing.

Consolidating removes the gaps rather than patching each one.

Vocabulary — pick deliberately
------------------------------
    to_utc(x)      -> tz-aware UTC Timestamp, or None. Use for anything stored.
    naive_utc(x)   -> tz-NAIVE Timestamp holding UTC wall time. Use ONLY to
                      compare against legacy naive data (results.csv, Elo dates).
    utc_iso(x)     -> canonical string, always ending "+00:00".
    local_date(x)  -> (date string, tz used). The results.csv boundary.

Naive timestamps are a legacy concession, never a default. If you find yourself
reaching for `naive_utc` outside a comparison with results.csv, that is a smell.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

UTC = "UTC"
ISO_SUFFIX = "+00:00"


def now_utc() -> pd.Timestamp:
    """Current instant, tz-aware UTC. The only clock this module reads."""
    return pd.Timestamp.now(tz=UTC)


def now_iso() -> str:
    """Current instant as a canonical UTC string, second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_utc(value: object) -> pd.Timestamp | None:
    """Anything parseable -> tz-aware UTC. None/NaT/blank -> None.

    A NAIVE input is ASSUMED to be UTC. That assumption is safe here because every
    provider we ingest supplies an offset, and the one place naive values appear is
    results.csv, which carries dates only. If a provider ever sends a naive local
    timestamp, it must be localised at the provider boundary, not here.
    """
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)


def naive_utc(value: object = None) -> pd.Timestamp | None:
    """UTC wall time with the tzinfo stripped, for comparison with naive data.

    `results.csv` dates and the Elo pipeline are tz-naive. Comparing those against
    an aware Timestamp raises `TypeError: Cannot compare tz-naive and tz-aware`,
    which is how this bit us the first time. Passing None returns "now".
    """
    ts = now_utc() if value is None else to_utc(value)
    return None if ts is None else ts.tz_localize(None)


def utc_iso(value: object) -> str:
    """Canonical stored form. Always ends '+00:00', or '' when unparseable."""
    ts = to_utc(value)
    return "" if ts is None else ts.isoformat()


def is_utc_iso(value: object) -> bool:
    """Does this string carry an explicit UTC offset? The storage invariant."""
    text = str(value or "").strip()
    return bool(text) and (text.endswith(ISO_SUFFIX) or text.endswith("Z"))


def hours_between(later: object, earlier: object = None) -> float | None:
    """Hours from `earlier` (default now) to `later`. None if either is unusable."""
    a, b = to_utc(later), (now_utc() if earlier is None else to_utc(earlier))
    if a is None or b is None:
        return None
    return round((a - b).total_seconds() / 3600.0, 4)


def local_date(kickoff: object, tz: str = "") -> tuple[str, str]:
    """(local date string, timezone used) — the results.csv boundary.

    With no timezone the UTC date is returned and the second element is "", which
    callers MUST treat as a flag rather than a default: that date is right for
    roughly half the world and a day out for the rest.
    """
    ts = to_utc(kickoff)
    if ts is None:
        return "", ""
    zone = str(tz or "").strip()
    if not zone or zone.lower() == "nan":
        return str(ts.date()), ""
    try:
        return str(ts.tz_convert(zone).date()), zone
    except Exception:
        return str(ts.date()), ""


def series_to_utc(values: pd.Series) -> pd.Series:
    """Vectorised `to_utc` for a column. Unparseable entries become NaT."""
    return pd.to_datetime(values, utc=True, errors="coerce")


def series_naive_utc(values: pd.Series) -> pd.Series:
    """Vectorised `naive_utc`, for joining against naive legacy columns."""
    return series_to_utc(values).dt.tz_localize(None)


def audit_utc_column(values: pd.Series) -> dict:
    """How healthy is a stored timestamp column? Used by the gate."""
    total = len(values)
    text = values.astype(str)
    blank = int((text.str.strip().isin(["", "nan", "NaT", "None"])).sum())
    explicit = int(text.map(is_utc_iso).sum())
    parsed = int(series_to_utc(values).notna().sum())
    return {"rows": total, "blank": blank, "explicit_utc": explicit,
            "parsed": parsed, "implicit_or_bad": total - blank - explicit}
