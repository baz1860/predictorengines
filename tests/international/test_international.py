"""Tests for the international module (plan §7, stages 1-2 and 4).

Run standalone (`python3 tests/international/test_international.py`) or under
pytest. The root conftest.py turns any check() failure into a pytest failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from international import fixtures as F          # noqa: E402
from international import gate as G              # noqa: E402
from international import identity as I          # noqa: E402
from international import registry as R          # noqa: E402
from international import store as S             # noqa: E402
from international import taxonomy as T          # noqa: E402

FAIL = 0
COLS = ["date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral"]


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


def _df(rows: list[list]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLS)


# ── taxonomy ────────────────────────────────────────────────────────────────
def test_taxonomy() -> None:
    print("\ntaxonomy")
    check("every label in results.csv is classified", T.unmapped() == [],
          f"unmapped: {T.unmapped()[:5]}")

    # The legacy profile must reproduce predictor.py EXACTLY, or the goldens lie.
    from engines.worldcup import predictor as p
    same = all(T.k_for(k, "legacy") == v for k, v in p.K_BY_TOURNAMENT.items())
    check("legacy profile matches predictor.K_BY_TOURNAMENT", same)
    check("legacy profile defaults to DEFAULT_K",
          T.k_for("Some Unheard-Of Cup", "legacy") == p.DEFAULT_K)

    # A label matching no rule at all must raise under v1.
    try:
        T.k_for("Zzzz Qqqq", "v1")
        check("v1 refuses wholly unrecognised labels", False, "no exception raised")
    except KeyError:
        check("v1 refuses wholly unrecognised labels", True)

    # A label matching only a broad PATTERN is classified so the pipeline keeps
    # running, but is flagged provisional so it cannot slip in unnoticed.
    novel = T.classify("Some Unheard-Of Cup")
    check("a pattern-matched novel label is marked provisional",
          novel is not None and novel.provisional)
    check("a novel label is unacknowledged, so the strict gate will catch it",
          "Some Unheard-Of Cup" in T.unacknowledged_provisional(["Some Unheard-Of Cup"]))
    check("explicit labels are never provisional",
          not T.classify("FIFA World Cup").provisional)
    check("no unacknowledged provisional labels in live data",
          T.unacknowledged_provisional() == [],
          str(T.unacknowledged_provisional()[:5]))

    check("v1 gives Euro qualifying a real weight, not the generic default",
          T.k_for("UEFA Euro qualification", "v1") == 40
          and T.k_for("UEFA Euro qualification", "legacy") == p.DEFAULT_K)
    check("friendlies stay below qualifiers under v1",
          T.k_for("Friendly", "v1") < T.k_for("UEFA Euro qualification", "v1"))
    check("Island Games is not bettable", not T.is_bettable("Island Games"))
    check("World Cup is bettable", T.is_bettable("FIFA World Cup"))


# ── registry ────────────────────────────────────────────────────────────────
def test_registry() -> None:
    print("\nregistry")
    reg = R.load()
    current = [t for t in reg.values() if t.status == R.FIFA and not t.member_to]
    check("exactly 211 current FIFA members", len(current) == 211, str(len(current)))

    by_conf: dict[str, int] = {}
    for t in current:
        by_conf[t.confederation] = by_conf.get(t.confederation, 0) + 1
    expected = {"UEFA": 55, "CAF": 54, "AFC": 46, "CONCACAF": 35, "OFC": 11,
                "CONMEBOL": 10}
    check("confederation counts match FIFA membership", by_conf == expected,
          str(by_conf))

    check("no active team is unclassified", R.unclassified_active() == [],
          str(R.unclassified_active()[:5]))
    check("Kernow is out of scope", not R.in_scope("Kernow"))
    check("Brazil is in scope", R.in_scope("Brazil"))
    check("everything is in scope under universe=all",
          R.in_scope("Kernow", universe="all"))

    # Effective dating: a defunct member is in scope for its own era only.
    check("Czechoslovakia in scope in 1990", R.in_scope("Czechoslovakia", "1990-06-01"))
    check("Czechoslovakia out of scope in 2020",
          not R.in_scope("Czechoslovakia", "2020-06-01"))

    check("fixture needs both sides in scope by default",
          not R.fixture_in_scope("Brazil", "Kernow"))
    check("require_both=False keeps mixed fixtures",
          R.fixture_in_scope("Brazil", "Kernow", require_both=False))

    try:
        R.in_scope("Not A Real Team At All")
        check("unknown inactive team is simply out of scope", True)
    except R.ScopeError:
        check("unknown inactive team is simply out of scope", False)


# ── identity ────────────────────────────────────────────────────────────────
def test_identity() -> None:
    print("\nidentity")
    check("signature ignores date",
          I.signature("Argentina", "Egypt", "FIFA World Cup")
          == I.signature("argentina", "egypt", "fifa world cup"))
    check("signature is order sensitive",
          I.signature("A", "B", "F") != I.signature("B", "A", "F"))

    a = pd.Series({"date": "2026-07-06", "home_score": None, "away_score": None,
                   "city": "Atlanta"})
    b = pd.Series({"date": "2026-07-07", "home_score": 3.0, "away_score": 2.0,
                   "city": "Atlanta"})
    v = I.classify_pair(a, b)
    check("blank + scored one day apart, same venue = same match",
          v.outcome == I.SAME_MATCH, v.reason)

    c = pd.Series({"date": "2026-07-07", "home_score": 1.0, "away_score": 0.0,
                   "city": "Atlanta"})
    d = pd.Series({"date": "2026-07-08", "home_score": 2.0, "away_score": 2.0,
                   "city": "Atlanta"})
    check("two different scores = two real matches",
          I.classify_pair(c, d).outcome == I.DISTINCT_MATCHES)

    e = pd.Series({"date": "2026-07-01", "home_score": None, "away_score": None,
                   "city": "X"})
    f = pd.Series({"date": "2026-07-20", "home_score": 1.0, "away_score": 0.0,
                   "city": "X"})
    check("weeks apart = distinct", I.classify_pair(e, f).outcome == I.DISTINCT_MATCHES)

    g = pd.Series({"date": "2026-07-06", "home_score": None, "away_score": None,
                   "city": "Lisbon"})
    check("one scored but venues differ = ambiguous, never auto-merged",
          I.classify_pair(g, b).outcome == I.AMBIGUOUS)

    check("provider id beats date for canonical id",
          I.canonical_id("A", "B", "F", date="2026-01-01", provider_event_id=99)
          == I.canonical_id("A", "B", "F", date="2026-01-02", provider_event_id=99))


# ── fixtures ────────────────────────────────────────────────────────────────
def test_fixtures() -> None:
    print("\nfixtures")
    dupe = _df([
        ["2026-07-06", "Argentina", "Egypt", None, None, "FIFA World Cup",
         "Atlanta", "United States", True],
        ["2026-07-07", "Argentina", "Egypt", 3, 2, "FIFA World Cup",
         "Atlanta", "United States", True],
        ["2026-06-01", "Brazil", "Chile", 1, 0, "Friendly", "Rio", "Brazil", False],
    ])
    pairs = F.find_duplicates(dupe)
    check("the July 2026 pattern is detected", len(pairs) == 1 and
          pairs[0].outcome == I.SAME_MATCH)

    kept, unresolved = F.reconcile(dupe, exceptions={})
    check("reconcile drops the blank row", len(kept) == 2)
    check("reconcile keeps the scored row",
          bool(kept[(kept.home_team == "Argentina")].home_score.notna().all()))
    check("nothing left unresolved for the blank/scored case", unresolved == [])

    # Both scored: must NOT be dropped automatically — that would delete history.
    both = _df([
        ["2021-11-13", "Nicaragua", "Cuba", 2, 0, "Friendly", "Managua",
         "Nicaragua", False],
        ["2021-11-14", "Nicaragua", "Cuba", 2, 0, "Friendly", "Managua",
         "Nicaragua", False],
    ])
    kept2, unresolved2 = F.reconcile(both, exceptions={})
    check("both-scored duplicates are NOT auto-dropped", len(kept2) == 2)
    check("both-scored duplicates are reported for review", len(unresolved2) == 1)

    kept3, _ = F.reconcile(
        both, exceptions={F._pair_id(F.find_duplicates(both)[0]): F.ACCEPTED_DUPLICATE})
    check("an adjudicated duplicate IS dropped", len(kept3) == 1)

    # Placeholder rows must be ignored, not flagged.
    ph = _df([
        ["2026-07-18", None, None, None, None, "FIFA World Cup", "Miami",
         "United States", True],
        ["2026-07-19", None, None, None, None, "FIFA World Cup", "Miami",
         "United States", True],
    ])
    check("unresolved placeholders are not duplicates", F.find_duplicates(ph) == [])
    check("unresolved placeholders are not stale", len(F.stale_blanks(ph)) == 0)

    stale = F.stale_blanks(dupe, asof="2026-08-08")
    check("a past blank row is stale", len(stale) == 1)
    check("a future blank row is not stale",
          len(F.stale_blanks(_df([["2026-09-05", "Wales", "Norway", None, None,
                                   "Friendly", "Cardiff", "Wales", False]]),
                             asof="2026-08-08")) == 0)

    try:
        F.assert_invariants(dupe, asof="2026-08-08", exceptions={})
        check("assert_invariants raises on a dirty table", False)
    except F.FixtureIntegrityError:
        check("assert_invariants raises on a dirty table", True)


# ── live data ───────────────────────────────────────────────────────────────
def test_live_results_clean() -> None:
    print("\nlive data/results.csv")
    df = pd.read_csv(ROOT / "data" / "results.csv")
    stale = F.stale_blanks(df, asof="2026-08-08")
    check("no past-kickoff blank rows remain", len(stale) == 0,
          str(stale[["date", "home_team", "away_team"]].values.tolist()[:3]))

    auto = [p for p in F.find_duplicates(df)
            if any(r in p.reason for r in F.AUTO_RESOLVABLE_REASONS)]
    check("no unreconciled blank/scored duplicates remain", auto == [],
          str([(p.home, p.away) for p in auto]))

    check("daily gate passes", G.run(strict=False, asof="2026-08-08") == [])
    check("strict gate still reports the review backlog",
          len(G.run(strict=True, asof="2026-08-08")) > 0)


# ── store ───────────────────────────────────────────────────────────────────
def test_store(tmp: Path | None = None) -> None:
    print("\nstore")
    import tempfile
    tmpdir = Path(tempfile.mkdtemp())

    raw = S.RawStore(tmpdir / "raw")
    p1 = raw.write(S.RawObservation("bsd", "fixtures", {"events": [1, 2]}))
    p2 = raw.write(S.RawObservation("bsd", "fixtures", {"events": [1, 2, 3]}))
    check("raw store is append-only", p1 != p2 and p1.exists() and p2.exists())
    check("raw store replays", len(list(raw.replay("bsd", "fixtures"))) == 2)

    row = S.normalize_fixture(
        home="Argentina", away="Egypt", competition="FIFA World Cup",
        kickoff_utc="2026-07-07T00:00:00Z", venue_tz="America/New_York",
        city="Atlanta", country="United States", neutral=True,
        provider="bsd", provider_event_id="8371")
    check("normalize sets scheduled status when unscored", row["status"] == S.SCHEDULED)
    check("normalize attaches a taxonomy category", row["category"] == T.WORLD_CUP)
    check("normalize records kickoff in UTC", row["kickoff_utc"].endswith("+00:00"))

    local_only = S.normalize_fixture(home="A", away="B", competition="Friendly",
                                     local_date="2026-09-05")
    check("a date-only fixture is flagged as a weak identity",
          "no kickoff_utc" in local_only["conflict"])

    st = S.FixtureStore(tmpdir / "fixtures.csv")
    check("upsert inserts", st.upsert([row])["added"] == 1)
    scored = dict(row, home_score=3, away_score=2, status=S.PLAYED)
    res = st.upsert([scored])
    check("upsert updates by fixture_id, not duplicating",
          res["updated"] == 1 and res["total"] == 1)
    check("status transitioned to played",
          st.load().iloc[0]["status"] == S.PLAYED)

    future = S.normalize_fixture(home="Wales", away="Norway", competition="Friendly",
                                 kickoff_utc="2026-09-05T18:00:00Z")
    st.upsert([future])
    up = st.upcoming(asof="2026-08-08")
    check("upcoming excludes played fixtures and includes future ones",
          len(up) == 1 and up.iloc[0]["home_team"] == "Wales")

    check("tombstone retires rather than deletes",
          st.tombstone(future["fixture_id"], S.CANCELLED, "friendly called off")
          and len(st.load()) == 2
          and len(st.upcoming(asof="2026-08-08")) == 0)


def main() -> int:
    test_taxonomy()
    test_registry()
    test_identity()
    test_fixtures()
    test_live_results_clean()
    test_store()
    print(f"\n{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURE(S)'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
