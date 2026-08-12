"""BSD fixture provider (plan §8 steps 3-4).

Every test here is OFFLINE. The parsers are pure functions over provider payloads,
and the payloads used are either synthetic fixtures defined below or real
observations replayed from the raw store — which is the point of writing raw
before parsing.

The regressions pinned here are the ones the first live spike actually found:
  * "Czechia" / "Türkiye" / "Ireland" / "Bosnia & Herzegovina" dropped 24 fixtures
    involving FIFA members because the parser did not canonicalise names;
  * a Japanese club (Albirex Niigata) and a youth side (Jamaica U20) appeared
    inside BSD's "International Friendly Games" league;
  * a UTC kick-off at a US venue produces the PREVIOUS local date — the exact
    mechanism that duplicated two World Cup fixtures in results.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.worldcup.names import canonical_team, looks_like_national_team  # noqa: E402
from international import venues as V                       # noqa: E402
from international.providers import bsd                     # noqa: E402
from international.store import PLAYED, RawStore, SCHEDULED  # noqa: E402

FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


LEAGUES = [
    {"id": 27, "name": "World Cup 2026", "country": "International", "is_women": False},
    {"id": 31, "name": "International Friendly Games", "country": "International",
     "is_women": False},
    {"id": 40, "name": "UEFA Nations League", "country": "Europe", "is_women": False},
    {"id": 41, "name": "World Cup Qualification CAF", "country": "Africa",
     "is_women": False},
    {"id": 50, "name": "UEFA European U19 Championship", "country": "Europe",
     "is_women": False},
    {"id": 51, "name": "NWSL", "country": "USA", "is_women": True},
    {"id": 52, "name": "Premier League", "country": "England", "is_women": False},
    {"id": 53, "name": "Some New Continental Cup", "country": "Africa",
     "is_women": False},
]


def _event(**kw) -> dict:
    base = {"id": 1, "league_id": 40, "home_team": "Spain", "away_team": "France",
            "event_date": "2026-09-24T18:45:00+00:00", "status": "notstarted",
            "home_score": None, "away_score": None, "venue_id": None,
            "is_neutral_ground": False}
    base.update(kw)
    return base


def test_league_mapping() -> None:
    print("\nleague mapping")
    m = bsd.international_league_ids(LEAGUES)
    check("senior internationals are mapped", set(m) == {27, 31, 40, 41}, str(sorted(m)))
    check("youth competitions excluded", 50 not in m)
    check("women's competitions excluded", 51 not in m)
    check("domestic leagues excluded", 52 not in m)
    check("all six qualifying confederations are in the map",
          sum(1 for v in bsd.LEAGUE_TO_COMPETITION.values()
              if v == "FIFA World Cup qualification") == 6)

    unm = bsd.unmapped_international_leagues(LEAGUES)
    check("an unmapped international-looking league is reported",
          unm == ["Some New Continental Cup"], str(unm))

    from international import taxonomy as T
    bad = [c for c in bsd.LEAGUE_TO_COMPETITION.values() if T.classify(c) is None]
    check("every mapped competition exists in the taxonomy", bad == [], str(bad))


def test_name_canonicalisation() -> None:
    print("\nname canonicalisation (the 24-fixture bug)")
    for raw, want in [("Czechia", "Czech Republic"), ("Türkiye", "Turkey"),
                      ("Ireland", "Republic of Ireland"),
                      ("Bosnia & Herzegovina", "Bosnia and Herzegovina"),
                      ("US Virgin Islands", "United States Virgin Islands")]:
        check(f"{raw!r} -> {want!r}", canonical_team(raw) == want,
              canonical_team(raw))

    m = bsd.international_league_ids(LEAGUES)
    row = bsd.parse_event(_event(home_team="Czechia", away_team="Türkiye"),
                          m[40], venue_lookup=False)
    check("parser canonicalises both sides",
          (row["home_team"], row["away_team"]) == ("Czech Republic", "Turkey"))

    from international import registry as R
    check("canonicalised names are now in FIFA scope",
          R.fixture_in_scope(row["home_team"], row["away_team"]))


def test_non_national_rejection() -> None:
    print("\nclub and youth rejection")
    # Honest about the limits of the heuristic: a club with no age/gender marker
    # in its name (Albirex Niigata) is NOT caught here. It is caught by the team
    # registry, which knows nothing by that name. Two layers, and this test pins
    # which layer does what — a vacuous assertion here would hide the gap.
    from international import registry as R
    check("the name heuristic does NOT catch a marker-less club",
          looks_like_national_team("Albirex Niigata"))
    check("the registry catches it instead",
          R.status("Albirex Niigata") == R.UNCLASSIFIED
          and not R.in_scope("Albirex Niigata"))

    check("a U20 side is rejected", not looks_like_national_team("Jamaica U20"))
    check("a B team is rejected", not looks_like_national_team("Spain B"))
    check("a women's side is rejected", not looks_like_national_team("Brazil Women"))
    check("a real nation passes", looks_like_national_team("Brazil"))
    check("a nation with a compound name passes",
          looks_like_national_team("Bosnia and Herzegovina"))

    m = bsd.international_league_ids(LEAGUES)
    try:
        bsd.parse_event(_event(league_id=31, home_team="Miami United",
                               away_team="Jamaica U20"), m[31], venue_lookup=False)
        check("parse_event raises on a youth side", False)
    except bsd.NotANationalTeam:
        check("parse_event raises on a youth side", True)

    rows, skipped = bsd.parse_events(
        [_event(id=1), _event(id=2, league_id=31, home_team="X", away_team="Spain B"),
         _event(id=3, league_id=52)], m)
    check("parse_events keeps the good row", len(rows) == 1)
    check("parse_events skips rather than crashes", len(skipped) == 2)
    check("skip reasons are recorded",
          all(s["reason"] for s in skipped), str(skipped))


def test_timezone_and_local_date() -> None:
    print("\nkick-off, timezone and local date")
    # The duplicate mechanism: 00:00 UTC on 7 July is 19:00 on 6 July in Texas.
    local, tz = V.local_date("2026-07-07T00:00:00Z", 1184)
    check("US venue: UTC 7 July -> local 6 July",
          (local, tz) == ("2026-07-06", "America/Chicago"), f"{local} {tz}")
    check("no venue -> UTC date, flagged with empty tz",
          V.local_date("2026-07-07T00:00:00Z", None) == ("2026-07-07", ""))
    check("tz_of returns '' not 'nan' for an unknown venue",
          V.tz_of(999999) == "")

    m = bsd.international_league_ids(LEAGUES)
    row = bsd.parse_event(_event(event_date="2026-09-24T18:45:00+00:00"),
                          m[40], venue_lookup=False)
    check("kickoff stored in UTC", row["kickoff_utc"].startswith("2026-09-24T18:45")
          and row["kickoff_utc"].endswith("+00:00"))
    check("missing timezone is flagged in conflict",
          "timezone unresolved" in row["conflict"], row["conflict"])

    # BSD v1 serves +04:00; conversion must not shift the calendar day wrongly.
    row2 = bsd.parse_event(_event(event_date="2026-06-22T21:00:00+04:00"),
                           m[40], venue_lookup=False)
    check("a +04:00 timestamp converts to the right UTC instant",
          row2["kickoff_utc"].startswith("2026-06-22T17:00"), row2["kickoff_utc"])


def test_status_and_scores() -> None:
    print("\nstatus and scores")
    m = bsd.international_league_ids(LEAGUES)
    r = bsd.parse_event(_event(status="finished", home_score=3, away_score=2),
                        m[40], venue_lookup=False)
    check("finished with scores -> played", r["status"] == PLAYED)
    check("scores parsed", (r["home_score"], r["away_score"]) == (3, 2))

    r2 = bsd.parse_event(_event(status="finished"), m[40], venue_lookup=False)
    check("finished WITHOUT scores stays scheduled, never a blank played row",
          r2["status"] == SCHEDULED)

    for raw, want in [("postponed", "postponed"), ("cancelled", "cancelled"),
                      ("abandoned", "abandoned"), ("notstarted", "scheduled")]:
        rr = bsd.parse_event(_event(status=raw), m[40], venue_lookup=False)
        check(f"status {raw!r} -> {want!r}", rr["status"] == want, rr["status"])

    check("unknown status falls back to scheduled",
          bsd.parse_event(_event(status="???"), m[40],
                          venue_lookup=False)["status"] == SCHEDULED)


def test_identity_stability() -> None:
    print("\nfixture identity")
    m = bsd.international_league_ids(LEAGUES)
    a = bsd.parse_event(_event(id=777), m[40], venue_lookup=False)
    # Same provider id, different date -> same fixture (a rescheduled match).
    b = bsd.parse_event(_event(id=777, event_date="2026-09-25T18:45:00+00:00"),
                        m[40], venue_lookup=False)
    check("provider event id anchors identity across a date change",
          a["fixture_id"] == b["fixture_id"])
    c = bsd.parse_event(_event(id=778), m[40], venue_lookup=False)
    check("different provider id -> different fixture", a["fixture_id"] != c["fixture_id"])


def test_against_recorded_payloads() -> None:
    print("\nreplay of recorded payloads")
    raw = RawStore()
    leagues = events = None
    for o in raw.replay("bsd", "leagues"):
        leagues = o["payload"]
    for o in raw.replay("bsd", "events"):
        events = o["payload"]
    if leagues is None or events is None:
        print("  SKIP  no recorded observations yet (run a live fetch)")
        return

    m = bsd.international_league_ids(leagues)
    rows, skipped = bsd.parse_events(events, m)
    check("replay produces international fixtures", len(rows) > 100, str(len(rows)))
    check("every row has a UTC kickoff",
          all(r["kickoff_utc"].endswith("+00:00") for r in rows))
    check("every row has a canonical competition",
          all(r["competition"] in bsd.LEAGUE_TO_COMPETITION.values() for r in rows))
    check("fixture ids are unique",
          len({r["fixture_id"] for r in rows}) == len(rows))

    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df.kickoff_utc, utc=True)
    win = df[(ts >= "2026-09-21T00:00:00Z") & (ts <= "2026-10-06T23:59:59Z")]
    check("the September window is covered", len(win) > 100, str(len(win)))
    check("no club or youth side survived",
          all(looks_like_national_team(n)
              for n in set(df.home_team) | set(df.away_team)))


def main() -> int:
    test_league_mapping()
    test_name_canonicalisation()
    test_non_national_rejection()
    test_timezone_and_local_date()
    test_status_and_scores()
    test_identity_stability()
    test_against_recorded_payloads()
    print(f"\n{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURE(S)'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
