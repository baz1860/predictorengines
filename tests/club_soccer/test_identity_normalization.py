#!/usr/bin/env python3
"""Club-name normalisation must not silently delete letters.

`fold_accents` relies on NFKD + combining-mark stripping, which handles é/ä/ñ
but leaves ø, æ, å, ð, þ, ß, đ, ł, ı, œ, ə and ħ untouched — they are atomic
codepoints, not base + accent. `normalise_club_text` then applies [^a-z0-9 ]
and DELETED them, so "Brøndby IF" keyed as "br ndby if" and could never match
"Brondby". The Danish Superliga carried two identities each for Brøndby,
Sønderjyske and Nordsjælland as a result: 14 clubs in a 12-club league.
"""
from __future__ import annotations

import pandas as pd
import pytest

from club_soccer import identity_review as IR
from club_soccer.competitions import get as comp_get
from club_soccer.normalization import fold_accents, normalise_club_text


@pytest.mark.parametrize("raw,expected", [
    ("Brøndby IF", "brondby if"),
    ("Sønderjyske Fodbold", "sonderjyske fodbold"),
    ("FC Nordsjælland", "fc nordsjaelland"),
    ("Zagłębie Lubin", "zaglebie lubin"),
    ("Ħamrun Spartans FC", "hamrun spartans fc"),
    ("Stjarnan Garðabær", "stjarnan gardabaer"),
    ("Víkingur Gøta", "vikingur gota"),
    ("Zirə FK", "zire fk"),
    # Regular combining accents must keep working unchanged.
    ("Bayern München", "bayern munchen"),
    ("Atlético Madrid", "atletico madrid"),
])
def test_non_decomposing_letters_are_transliterated(raw, expected):
    assert normalise_club_text(raw) == expected


def test_no_alphabetic_character_is_dropped():
    """The strip must never be the thing that removes a letter."""
    for raw in ["Brøndby", "Sønderjyske", "Nordsjælland", "Wisła", "Ħamrun",
                "Garðabær", "Zirə", "Þór", "Borussia Mönchengladbach"]:
        folded = fold_accents(raw)
        assert all(not char.isalpha() or "a" <= char <= "z" for char in folded), \
            f"{raw!r} folded to {folded!r} with a non-ASCII letter still present"


def test_danish_spellings_now_share_one_key():
    for a, b in [("Brøndby IF", "Brondby IF"),
                 ("Sønderjyske", "Sonderjyske"),
                 ("Nordsjælland", "Nordsjaelland")]:
        assert normalise_club_text(a) == normalise_club_text(b)


# --- same-league duplicate detection -------------------------------------

def _fixture_frame(pairs, competition="Danish Superliga", season=2026):
    return pd.DataFrame([
        {"competition": competition, "season": season, "home": home,
         "away": away, "date": "2026-08-01", "home_goals": 1, "away_goals": 0}
        for home, away in pairs
    ])


def test_detector_flags_two_spellings_inside_one_league():
    df = _fixture_frame([
        ("Silkeborg", "Brondby"),
        ("Odense", "Brøndby IF"),      # same club, second spelling
        ("Silkeborg", "Odense"),
    ])
    rows = IR.duplicate_league_identities(df)
    pairs = {(row["name_a"], row["name_b"]) for row in rows}
    assert ("Brondby", "Brøndby IF") in pairs


def test_detector_ignores_clubs_that_merely_share_a_city():
    """CI._affinity's shared-token rule paired every Moscow club with every
    other. A same-league sweep compares the whole division, so that rule
    produced ~30 false rows and buried the real findings."""
    df = _fixture_frame([
        ("CSKA Moscow", "Spartak Moscow"),
        ("Dynamo Moscow", "Lokomotiv Moscow"),
        ("Levski Sofia", "Slavia Sofia"),
        ("DC United", "Minnesota United"),
    ], competition="Russian Premier League")
    assert IR.duplicate_league_identities(df) == []


def test_detector_ignores_clubs_that_played_each_other():
    """A club cannot be its own opponent, so a head-to-head rules a pair out
    however similar the names look."""
    df = _fixture_frame([("Brondby", "Brøndby IF")])
    assert IR.duplicate_league_identities(df) == []


def test_oversized_leagues_are_detected_without_name_matching():
    """The structural check catches misnamings the name check cannot — e.g.
    Wolverhampton's second half of 2023/24 duplicated under 'Bolton'."""
    teams = [f"Club {i}" for i in range(14)]
    pairs = [(teams[i], teams[(i + 1) % len(teams)]) for i in range(len(teams))]
    df = _fixture_frame(pairs)
    rows = IR.oversized_league_seasons(df)
    assert len(rows) == 1
    assert rows[0]["teams_seen"] == 14
    assert rows[0]["teams_expected"] == comp_get("Danish Superliga").teams_n
    assert rows[0]["excess"] == 14 - comp_get("Danish Superliga").teams_n


def test_production_leagues_have_no_registry_confirmed_duplicates():
    """Guards the merge that was applied on 2026-08-10. A new confirmed pair
    appearing here means a feed has started splitting a club again."""
    rows = IR.duplicate_league_identities()
    confirmed = [row for row in rows if row["confidence"] == 1.0]
    assert not confirmed, f"unmerged confirmed duplicates: {confirmed}"
