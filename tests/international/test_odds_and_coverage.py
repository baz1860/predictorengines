"""Odds store and coverage tiers.

Both modules exist to make an absence visible:
  * the odds store records "we asked and there were no prices", because otherwise
    "no odds existed" and "we never looked" are indistinguishable later, and only
    one of those is a finding;
  * coverage tiers record "we barely know this team", because a rating lookup for
    an unknown team silently returns the average one — the failure that cost the
    club module a 4-0 loss at 18%.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from international import coverage as C     # noqa: E402
from international import odds as O         # noqa: E402

FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


FIXTURE = {
    "fixture_id": "abc123", "provider_event_id": "9074",
    "kickoff_utc": "2026-09-24T18:45:00+00:00",
    "home_team": "Spain", "away_team": "France", "competition": "UEFA Nations League",
}


def test_parsing() -> None:
    print("\nodds parsing")
    event = {"odds_home": 1.65, "odds_draw": 3.67, "odds_away": 5.21,
             "odds_over_25": 1.94, "odds_under_25": 1.83,
             "odds_btts_yes": 1.9, "odds_btts_no": 1.82,
             "odds_over_15": None, "unrelated": 7}
    rows = O.parse_bsd_inline(event, FIXTURE, snapshot_at="2026-09-20T12:00:00+00:00")
    check("one row per populated market/side", len(rows) == 7, str(len(rows)))
    check("null odds are skipped", all(r["side"] != "over" or r["line"] != "1.5"
                                       for r in rows))
    h2h = [r for r in rows if r["market"] == "h2h"]
    check("h2h has three sides", len(h2h) == 3)
    check("totals carry a line",
          all(r["line"] for r in rows if r["market"] == "totals"))
    check("every row is marked priced", all(r["status"] == O.PRICED for r in rows))
    check("the implied book is named honestly, not as a real bookmaker",
          all(r["bookmaker"] == "bsd_implied" for r in rows))
    check("hours to kickoff computed",
          abs(rows[0]["hours_to_kickoff"] - 102.75) < 0.1,
          str(rows[0]["hours_to_kickoff"]))

    check("odds of 1.0 or less are rejected as impossible",
          O.parse_bsd_inline({"odds_home": 1.0, "odds_draw": 0.5}, FIXTURE) == [])
    check("non-numeric odds are skipped",
          O.parse_bsd_inline({"odds_home": "n/a"}, FIXTURE) == [])

    check("empty comparison payload yields nothing",
          O.parse_bsd_comparison({"count": 0, "odds": []}, FIXTURE) == [])
    multi = {"odds": [{"bookmaker": "bet365", "odds_home": 1.7, "odds_away": 5.0},
                      {"bookmaker": "pinnacle", "odds_home": 1.72}]}
    rows2 = O.parse_bsd_comparison(multi, FIXTURE)
    check("multi-bookmaker payload keeps bookmaker identity",
          {r["bookmaker"] for r in rows2} == {"bet365", "pinnacle"})


def test_absence_is_recorded() -> None:
    print("\nabsence as evidence")
    row = O.record_absence(FIXTURE, snapshot_at="2026-08-08T12:00:00+00:00")
    check("absence row carries the fixture", row["fixture_id"] == "abc123")
    check("absence row has no price", row["odds"] == "")
    check("absence row is flagged", row["status"] == O.NO_ODDS)
    check("absence row records how far out we asked",
          row["hours_to_kickoff"] > 1000, str(row["hours_to_kickoff"]))
    check("a failed fetch is distinguishable from an empty one",
          O.record_absence(FIXTURE, status=O.FETCH_FAILED)["status"]
          != O.NO_ODDS)


def test_store() -> None:
    print("\nodds store")
    with tempfile.TemporaryDirectory() as td:
        store = O.OddsStore(Path(td) / "odds.csv")
        check("empty store reports zeros", store.coverage()["snapshots"] == 0)

        store.append(O.parse_bsd_inline({"odds_home": 2.0, "odds_draw": 3.4,
                                         "odds_away": 3.6}, FIXTURE,
                                        snapshot_at="2026-09-20T00:00:00+00:00"))
        store.append(O.parse_bsd_inline({"odds_home": 1.9, "odds_draw": 3.5,
                                         "odds_away": 3.9}, FIXTURE,
                                        snapshot_at="2026-09-24T18:00:00+00:00"))
        store.append([O.record_absence(dict(FIXTURE, fixture_id="zzz"))])

        cov = store.coverage()
        check("snapshots counted", cov["snapshots"] == 3, str(cov))
        check("priced fixtures counted separately from absences",
              cov["priced_fixtures"] == 1 and cov["absence_rows"] == 1, str(cov))

        close = store.closing_prices()
        check("closing price is the LAST snapshot, not the first",
              float(close[close.side == "home"].odds.iloc[0]) == 1.9,
              str(close[close.side == "home"].odds.tolist()))
        check("history is preserved, not overwritten", len(store.load()) == 7)


def test_coverage_tiers() -> None:
    print("\ncoverage tiers")
    rows = []
    # A fully connected clique: 13 teams each playing all the others twice. Every
    # member clears the match, opponent-count AND anchoring thresholds, which is
    # what "full" is supposed to mean. An earlier version of this fixture gave
    # Brazil plenty of matches against opponents who played nobody else — and the
    # module correctly refused to call that full, which is the Sturm Graz signature.
    clique = ["Brazil"] + [f"Opp{i}" for i in range(12)]
    day = 1
    for i, a_team in enumerate(clique):
        for b_team in clique[i + 1:]:
            for _ in range(2):
                rows.append((f"2025-01-{(day % 28) + 1:02d}", a_team, b_team,
                             2, 0, "Friendly"))
                day += 1
    # Isolated: 6 matches against two opponents who play nobody else -> thin
    for k in range(6):
        rows.append(("2025-03-%02d" % (k + 1), "Isolated", "Hermit", 0, 0, "Friendly"))
    # Ghost: one match -> defaulted
    rows.append(("2025-04-01", "Ghost", "Brazil", 0, 9, "Friendly"))

    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                     "home_score", "away_score", "tournament"])
    df["date"] = pd.to_datetime(df.date)
    table = C.build(df, asof="2025-06-01")

    check("a well-connected team is full", table["Brazil"].tier == C.FULL,
          table["Brazil"].reason)
    check("a team with few matches is not full",
          table["Isolated"].tier in (C.THIN, C.DEFAULTED))
    check("a one-match team is defaulted", table["Ghost"].tier == C.DEFAULTED)
    check("an unknown team is defaulted, never assumed average",
          C.tier_of("Nobody At All", table) == C.DEFAULTED)

    check("a fixture is only as evidenced as its weaker side",
          C.fixture_tier("Brazil", "Ghost", table) == C.DEFAULTED)
    check("two strong sides give a full fixture",
          C.fixture_tier("Brazil", "Opp1", table) == C.FULL)

    check("the reason is human-readable", bool(table["Isolated"].reason))
    s = C.summary(table)
    check("summary covers all three tiers", set(s.index) == {C.FULL, C.THIN, C.DEFAULTED})


def test_live_coverage() -> None:
    print("\ncoverage on live data")
    table = C.build()
    check("most teams are fully evidenced",
          sum(1 for t in table.values() if t.tier == C.FULL) > 150)
    from international import registry as R
    thin_fifa = [t for t in table.values()
                 if t.tier != C.FULL and R.status(t.team) == R.FIFA]
    check("thin FIFA members are identified for the betting gate",
          0 < len(thin_fifa) < 60, str(len(thin_fifa)))
    check("every thin team carries a reason",
          all(t.reason for t in thin_fifa))


def main() -> int:
    test_parsing()
    test_absence_is_recorded()
    test_store()
    test_coverage_tiers()
    test_live_coverage()
    print(f"\n{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURE(S)'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
