"""Per-team home venue profiles.

Built to close two measured gaps: 202 of 255 fixtures had no venue timezone, and
the altitude model ran off a hand-typed 48-city dictionary.

The design point these tests defend is that **"national stadium" is the wrong
model**. Spain has 22 distinct home cities over a decade, the United States 39,
Germany 21 — while Kyrgyzstan and Liechtenstein have exactly one. A single-venue
column would be right for about half the world and quietly wrong for the rest, so
this stores a distribution and forces callers to look at `share`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from international import home_venues as HV   # noqa: E402

FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


def _results(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score",
                                     "away_score", "city", "country", "neutral"])
    df["date"] = pd.to_datetime(df.date)
    return df


def test_history_shape() -> None:
    print("\nhome-match history")
    rows = [
        ("2025-01-01", "Alpha", "Bravo", 1, 0, "Aville", "Alphaland", "FALSE"),
        ("2025-02-01", "Alpha", "Charlie", 2, 0, "Aville", "Alphaland", "FALSE"),
        ("2025-03-01", "Alpha", "Delta", 0, 0, "Btown", "Alphaland", "FALSE"),
        # neutral: must NOT count as a home venue
        ("2025-04-01", "Alpha", "Echo", 1, 1, "Neutralia", "Faraway", "TRUE"),
        # unplayed: must not count either
        ("2025-05-01", "Alpha", "Foxtrot", None, None, "Cport", "Alphaland", "FALSE"),
    ]
    h = HV.home_match_history(_results(rows), window_years=10)
    alpha = h[h.team == "Alpha"]
    check("only played non-neutral home matches counted", len(alpha) == 2,
          str(alpha.city.tolist()))
    check("neutral venue excluded", "Neutralia" not in set(alpha.city))
    check("unplayed fixture excluded", "Cport" not in set(alpha.city))
    top = alpha[alpha["rank"] == 1].iloc[0]
    check("modal venue identified", top.city == "Aville", top.city)
    check("share computed", abs(top.share - 2 / 3) < 0.01, str(top.share))
    check("ranks are dense from 1", sorted(alpha["rank"]) == [1, 2])


def test_rotation_is_visible() -> None:
    print("\nrotation is exposed, not hidden")
    single = [("2025-01-0%d" % d, "Solo", "X", 1, 0, "Onecity", "Sololand", "FALSE")
              for d in range(1, 6)]
    rotating = [("2025-02-0%d" % d, "Rota", "X", 1, 0, f"City{d}", "Rotaland", "FALSE")
                for d in range(1, 6)]
    h = HV.home_match_history(_results(single + rotating), window_years=10)

    solo = h[(h.team == "Solo") & (h["rank"] == 1)].iloc[0]
    rota = h[(h.team == "Rota") & (h["rank"] == 1)].iloc[0]
    check("single-venue team has share 1.0", abs(solo.share - 1.0) < 0.001)
    check("rotating team has a low share", rota.share <= 0.2, str(rota.share))

    v_solo = HV.Venue("Onecity", "Sololand", 5, float(solo.share), 0, 0, 0, "UTC")
    v_rota = HV.Venue("City1", "Rotaland", 1, float(rota.share), 0, 0, 0, "UTC")
    check("single-venue team is 'confident'", v_solo.confident)
    check("rotating team is NOT 'confident'", not v_rota.confident)


def test_live_table() -> None:
    print("\nlive venue table")
    cov = HV.coverage()
    if not cov.get("teams"):
        print("  SKIP  table not built")
        return
    check("covers most active national teams", cov["teams"] > 200, str(cov["teams"]))
    check("most venues are geocoded",
          cov["geocoded"] / cov["venues"] > 0.8, str(cov))
    check("teams with a single venue are identified", cov["teams_single_venue"] > 50)
    check("rotating teams are identified", cov["teams_rotating"] > 10)

    # The user's own example: Spain rotates and must not be given a false home.
    spain = HV.primary_venue("Spain")
    check("Spain resolves to a venue at all", spain is not None)
    if spain:
        check("Spain is flagged as NOT confident", not spain.confident,
              f"share {spain.share:.0%}")
        check("Spain still yields a usable timezone",
              spain.timezone == "Europe/Madrid", spain.timezone)

    for team, tz in [("England", "Europe/London"), ("Bolivia", "America/La_Paz"),
                     ("Japan", "Asia/Tokyo")]:
        v = HV.primary_venue(team)
        check(f"{team} timezone resolves to {tz}",
              v is not None and v.timezone == tz,
              v.timezone if v else "none")


def test_altitude_values() -> None:
    print("\naltitude sanity")
    cov = HV.coverage()
    if not cov.get("teams"):
        print("  SKIP  table not built")
        return
    known = {"Bolivia": (3500, 3900), "Ecuador": (2700, 2950),
             "Mexico": (2150, 2350), "England": (0, 100), "Peru": (0, 300)}
    for team, (lo, hi) in known.items():
        v = HV.primary_venue(team)
        ok = v is not None and v.elevation_m is not None and lo <= v.elevation_m <= hi
        check(f"{team} elevation within {lo}-{hi}m", ok,
              str(v.elevation_m) if v else "none")

    # 9999 is Open-Meteo's no-data sentinel and must never reach the table.
    table = HV._load()
    bad = table[table.elevation_m > 5000]
    check("no sentinel elevations survive", bad.empty,
          str(bad[["team", "city", "elevation_m"]].values.tolist()[:3]))


def test_fallback_chain() -> None:
    print("\nfallback chain")
    if not HV.coverage().get("teams"):
        print("  SKIP  table not built")
        return
    tz, basis = HV.timezone_for(team="Spain", city="Seville", country="Spain")
    check("a named city wins over the team guess", basis == "venue city", basis)

    tz, basis = HV.timezone_for(team="Spain")
    check("team fallback works", tz == "Europe/Madrid", tz)
    check("low confidence is stated in the basis", "low confidence" in basis, basis)

    tz, basis = HV.timezone_for(team="Bolivia")
    check("a confident team does not carry the warning",
          "low confidence" not in basis, basis)

    check("unknown team resolves to nothing, not a guess",
          HV.timezone_for(team="Nowhere United") == ("", "unresolved"))
    check("elevation falls back the same way",
          HV.elevation_for(team="Bolivia")[0] is not None)


def test_context_integration() -> None:
    print("\naltitude feeds the context model")
    from engines.worldcup.context import ALT_M, venue_alt_km
    check("hand-listed cities are unchanged (goldens stay stable)",
          abs(venue_alt_km("La Paz", "Bolivia") - ALT_M["La Paz"] / 1000) < 0.001)
    check("a city absent from ALT_M now resolves",
          venue_alt_km("Thimphu", "Bhutan") > 2.0,
          str(venue_alt_km("Thimphu", "Bhutan")))
    check("lowland cities stay at zero", venue_alt_km("London", "England") == 0.0)
    check("an unknown city is zero, not an error",
          venue_alt_km("Nowhereville", "Nowhere") == 0.0)


def main() -> int:
    test_history_shape()
    test_rotation_is_visible()
    test_live_table()
    test_altitude_values()
    test_fallback_chain()
    test_context_integration()
    print(f"\n{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURE(S)'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
