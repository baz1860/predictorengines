"""CAF AFCON 2027 qualifying, derived from the draw.

This is the one provider whose fixtures are computed rather than fetched, so the
tests carry more weight than usual: there is no upstream to blame if the template
is wrong.

The cross-check test encodes the published fixture list from africasoccer.com
(20 May 2026) INCLUDING its errors, and asserts that our comparison catches them.
A cross-check that passes against a flawed source is not a cross-check.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from international import registry as R              # noqa: E402
from international import taxonomy as T              # noqa: E402
from international.providers import caf              # noqa: E402

FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


# Published list as printed, Groups A and C. Group A MD6 is reproduced verbatim
# with its error: "Morocco v Gabon; Niger v Gabon" repeats MD1 and plays Gabon twice.
PUBLISHED = {
    ("A", 1): [("Morocco", "Gabon"), ("Niger", "Lesotho")],
    ("A", 2): [("Gabon", "Niger"), ("Lesotho", "Morocco")],
    ("A", 3): [("Gabon", "Lesotho"), ("Morocco", "Niger")],
    ("A", 4): [("Lesotho", "Gabon"), ("Niger", "Morocco")],
    ("A", 5): [("Gabon", "Morocco"), ("Lesotho", "Niger")],
    ("A", 6): [("Morocco", "Gabon"), ("Niger", "Gabon")],          # <- corrupt
    ("C", 1): [("Gambia", "Somalia"), ("Ivory Coast", "Ghana")],
    ("C", 2): [("Ghana", "Gambia"), ("Somalia", "Ivory Coast")],
    ("C", 3): [("Ghana", "Somalia"), ("Ivory Coast", "Gambia")],
    ("C", 4): [("Gambia", "Ivory Coast"), ("Somalia", "Ghana")],
    ("C", 5): [("Ghana", "Ivory Coast"), ("Somalia", "Gambia")],
    ("C", 6): [("Gambia", "Ghana"), ("Ivory Coast", "Somalia")],
}


def test_shape() -> None:
    print("\nderived fixture set")
    fx = caf.derive()
    check("12 groups x 6 matchdays x 2 matches = 144 fixtures", len(fx) == 144,
          str(len(fx)))
    check("every group has 4 teams",
          all(len(t) == 4 for t in caf.GROUPS.values()))
    check("48 distinct teams", len({t for g in caf.GROUPS.values() for t in g}) == 48)
    check("matchdays 1-6 all present", {f.matchday for f in fx} == set(range(1, 7)))

    md12 = [f for f in fx if f.matchday in (1, 2)]
    check("48 fixtures in the September window", len(md12) == 48, str(len(md12)))
    check("September window dated correctly",
          all(f.window_start == "2026-09-21" for f in md12))


def test_round_robin_validity() -> None:
    print("\nround-robin integrity")
    problems = caf.validate_template()
    check("every pairing occurs exactly twice, once at home each",
          problems == [], "; ".join(problems[:3]))


def test_cross_check_catches_the_published_error() -> None:
    print("\ncross-check against the published list")
    problems = caf.cross_check(PUBLISHED)
    check("a discrepancy is reported at all", problems != [])
    joined = " | ".join(problems)
    check("the corrupt Group A MD6 row is flagged", "group A MD6" in joined, joined)
    check("Group C — which is internally consistent — is NOT flagged",
          "group C" not in joined, joined)
    check("only the one bad matchday is flagged",
          sum(1 for p in problems if p.startswith("group A MD")) == 1, joined)


def test_scope_and_taxonomy() -> None:
    print("\nscope and taxonomy")
    check("competition is classified", T.classify(caf.COMPETITION) is not None)
    check("competition is bettable", T.is_bettable(caf.COMPETITION))
    check("competition maps to continental qualifying",
          T.category(caf.COMPETITION) == T.CONTINENTAL_QUAL)

    teams = {t for g in caf.GROUPS.values() for t in g}
    from engines.worldcup.names import canonical_team
    unknown = sorted(t for t in teams if R.status(canonical_team(t)) != R.FIFA)
    check("every one of the 48 teams is a known FIFA member",
          unknown == [], str(unknown))


def test_fixture_rows() -> None:
    print("\ncanonical rows")
    rows = caf.to_fixture_rows()
    check("144 rows produced", len(rows) == 144)
    check("fixture ids are unique", len({r["fixture_id"] for r in rows}) == 144)
    check("every row is flagged as window-dated, not a real kickoff",
          all("MATCHDAY WINDOW" in r["conflict"] for r in rows))
    check("every row carries the derived provider",
          all(r["provider"] == caf.PROVIDER for r in rows))
    check("rows are scheduled, not played",
          all(r["status"] == "scheduled" for r in rows))
    sept = [r for r in rows if r["local_date"] == "2026-09-21"]
    check("48 rows land in the September window", len(sept) == 48, str(len(sept)))


def main() -> int:
    test_shape()
    test_round_robin_validity()
    test_cross_check_catches_the_published_error()
    test_scope_and_taxonomy()
    test_fixture_rows()
    print(f"\n{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURE(S)'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
