"""Timezone handling — the single source of truth.

Before `international/timeutil.py`, thirty-plus ad-hoc conversions were spread
across ten files and three silent bugs came out of the gaps. These tests pin the
behaviour those bugs violated, plus the cases that actually bite in a global
fixture list: date-line crossings, DST transitions, and the naive/aware boundary
with results.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from international import timeutil as T   # noqa: E402

FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


def test_to_utc() -> None:
    print("\nto_utc")
    check("offset input converts to UTC",
          T.to_utc("2026-06-22T21:00:00+04:00").isoformat() == "2026-06-22T17:00:00+00:00")
    check("Z suffix parses",
          T.to_utc("2026-07-07T00:00:00Z").isoformat() == "2026-07-07T00:00:00+00:00")
    check("naive input is assumed UTC",
          T.to_utc("2026-07-07 00:00:00").isoformat() == "2026-07-07T00:00:00+00:00")
    check("already-UTC input is unchanged",
          T.to_utc(pd.Timestamp("2026-01-01", tz="UTC")).isoformat()
          == "2026-01-01T00:00:00+00:00")

    for bad in (None, "", "   ", "not a date", float("nan"), pd.NaT):
        check(f"unusable input {bad!r} -> None", T.to_utc(bad) is None)

    check("output is always tz-aware", T.to_utc("2026-01-01").tzinfo is not None)


def test_naive_boundary() -> None:
    print("\nnaive/aware boundary (the results.csv edge)")
    n = T.naive_utc("2026-07-07T00:00:00Z")
    check("naive_utc strips tzinfo", n.tzinfo is None)
    check("naive_utc preserves the UTC wall time", str(n) == "2026-07-07 00:00:00")

    # The original crash: comparing a naive results.csv date against an aware now.
    naive_date = pd.Timestamp("2026-07-06")
    try:
        _ = naive_date < T.naive_utc()
        check("naive results.csv date compares cleanly against naive_utc()", True)
    except TypeError as exc:
        check("naive results.csv date compares cleanly against naive_utc()", False, str(exc))

    try:
        _ = naive_date < T.now_utc()
        check("comparing naive against aware still raises (so the helper is needed)",
              False, "expected TypeError")
    except TypeError:
        check("comparing naive against aware still raises (so the helper is needed)",
              True)

    check("naive_utc(None) returns now", T.naive_utc() is not None)


def test_local_date_boundary() -> None:
    print("\nlocal_date — the one place local time is allowed")
    # The exact duplicate case: midnight UTC on 7 July is 19:00 on 6 July in Texas.
    check("US venue: UTC 7 July -> local 6 July",
          T.local_date("2026-07-07T00:00:00Z", "America/Chicago")
          == ("2026-07-06", "America/Chicago"))
    # Date line the other way: 09:00 UTC is already the next day in Auckland.
    check("date line: UTC 21:00 -> next day in Auckland",
          T.local_date("2026-06-01T21:00:00Z", "Pacific/Auckland")[0] == "2026-06-02")
    check("Tokyo evening kickoff does not slip a day",
          T.local_date("2026-06-01T10:00:00Z", "Asia/Tokyo")[0] == "2026-06-01")

    check("no timezone -> UTC date, flagged with an empty zone",
          T.local_date("2026-07-07T00:00:00Z") == ("2026-07-07", ""))
    check("the string 'nan' is treated as missing, not as a zone",
          T.local_date("2026-07-07T00:00:00Z", "nan") == ("2026-07-07", ""))
    check("an invalid zone falls back to UTC and says so",
          T.local_date("2026-07-07T00:00:00Z", "Mars/Olympus") == ("2026-07-07", ""))
    check("unparseable kickoff yields empty, not a crash",
          T.local_date("rubbish", "Europe/London") == ("", ""))


def test_dst() -> None:
    print("\nDST transitions")
    # Europe/London: BST (UTC+1) in summer, GMT in winter.
    check("summer: 23:30 UTC is already the next day in London",
          T.local_date("2026-06-15T23:30:00Z", "Europe/London")[0] == "2026-06-16")
    check("winter: 23:30 UTC is the same day in London",
          T.local_date("2026-12-15T23:30:00Z", "Europe/London")[0] == "2026-12-15")
    # A fixed offset would get one of those wrong; a real zone gets both right.
    check("hours_between is unaffected by DST (absolute instants)",
          T.hours_between("2026-06-15T12:00:00Z", "2026-06-15T06:00:00Z") == 6.0)


def test_iso_and_audit() -> None:
    print("\nstorage invariant")
    check("utc_iso always ends +00:00",
          T.utc_iso("2026-06-22T21:00:00+04:00").endswith("+00:00"))
    check("utc_iso of junk is empty", T.utc_iso("nope") == "")
    check("is_utc_iso accepts +00:00", T.is_utc_iso("2026-01-01T00:00:00+00:00"))
    check("is_utc_iso accepts Z", T.is_utc_iso("2026-01-01T00:00:00Z"))
    check("is_utc_iso REJECTS a naive timestamp",
          not T.is_utc_iso("2026-01-01T00:00:00"))
    check("is_utc_iso rejects a bare date", not T.is_utc_iso("2026-01-01"))
    check("is_utc_iso rejects blank", not T.is_utc_iso(""))

    col = pd.Series(["2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00Z",
                     "2026-01-03T00:00:00", "", None])
    a = T.audit_utc_column(col)
    check("audit counts explicit UTC values", a["explicit_utc"] == 2, str(a))
    check("audit counts blanks separately", a["blank"] == 2, str(a))
    check("audit flags the naive one", a["implicit_or_bad"] == 1, str(a))


def test_hours_between() -> None:
    print("\nhours_between")
    check("positive when later is in the future",
          T.hours_between("2026-01-01T12:00:00Z", "2026-01-01T00:00:00Z") == 12.0)
    check("negative when already past",
          T.hours_between("2026-01-01T00:00:00Z", "2026-01-01T12:00:00Z") == -12.0)
    check("mixed offsets handled",
          T.hours_between("2026-01-01T12:00:00+00:00", "2026-01-01T16:00:00+04:00") == 0.0)
    check("None on unusable input", T.hours_between("junk") is None)


def test_live_store_is_utc() -> None:
    print("\nlive fixture store")
    from international.store import FixtureStore
    df = FixtureStore().load()
    if df.empty:
        print("  SKIP  fixture store empty")
        return
    a = T.audit_utc_column(df.kickoff_utc)
    check("no stored kickoff lacks an explicit UTC offset",
          a["implicit_or_bad"] == 0, str(a))
    check("every non-blank kickoff parses", a["parsed"] == a["explicit_utc"], str(a))


def main() -> int:
    test_to_utc()
    test_naive_boundary()
    test_local_date_boundary()
    test_dst()
    test_iso_and_audit()
    test_hours_between()
    test_live_store_is_utc()
    print(f"\n{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURE(S)'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
