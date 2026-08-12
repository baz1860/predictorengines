"""Offline tests for the 247 talent standby ingest.

No network: the parser is exercised against inline fixtures matching the real
page markup. The live parse was validated separately against CFBD's stored
2025 values — 133 teams, zero value mismatches.
"""
from __future__ import annotations

import json

import pytest

from cfb import fetch_247_talent as T


def _row(team: str, slug: str, avg: str, points: str) -> str:
    return f'''
    <li class="rankings-page__list-item"> <div class="wrapper">
      <div class="rank-column"><div class="primary"> 1 </div></div>
      <div class="team"> <a class="rankings-page__name-link"
        href="https://247sports.com/college/{slug}/team/{slug}-football-164/roster/?year=2025">{team} </a> </div>
      <div class="total"> <a href="#">82 Commits </a> </div>
      <div class="avg"> {avg} </div>
      <ul class="star-commits-list"><li><h2>5-Star</h2><div class="gold"> 14 </div></li></ul>
      <div class="points"> <a class="number" href="#">{points} </a> </div>
    </div></li>'''


def test_parses_name_value_and_comma_grouped_totals():
    """Points above 1000 are comma-grouped; a naive numeric regex drops them."""
    html = (_row("Georgia", "georgia", "92.47", "1,002.98")
            + _row("Alabama", "alabama", "92.59", "993.55"))
    rows = T.parse_talent(html)
    assert [r["team"] for r in rows] == ["Georgia", "Alabama"]
    assert rows[0]["talent"] == pytest.approx(1002.98)
    assert rows[1]["talent"] == pytest.approx(993.55)


def test_uses_display_name_not_the_mascot_slug():
    """The roster slug is 'alabama-crimson-tide'; the school name is 'Alabama'."""
    html = _row("Alabama", "alabama", "92.59", "993.55").replace(
        "alabama-football-164", "alabama-crimson-tide-football-164")
    rows = T.parse_talent(html)
    assert rows[0]["team"] == "Alabama"


def test_average_rating_is_never_mistaken_for_the_composite():
    """The avg column (~85-95) sits before points and must not be picked up."""
    rows = T.parse_talent(_row("Alabama", "alabama", "92.59", "993.55"))
    assert rows[0]["talent"] == pytest.approx(993.55)


def test_empty_page_parses_to_nothing():
    """An unpublished season renders 'No Results' — must not fabricate rows."""
    assert T.parse_talent(
        "<div>No Results for 2026 Football</div>") == []


def test_duplicate_rows_are_collapsed():
    html = _row("Alabama", "alabama", "92.59", "993.55") * 3
    assert len(T.parse_talent(html)) == 1


def test_implausible_values_are_rejected():
    html = (_row("Bogus", "bogus", "92.5", "12.34")
            + _row("AlsoBogus", "alsobogus", "92.5", "99,999.00"))
    assert T.parse_talent(html) == []


def test_publish_refuses_a_partial_scrape(tmp_path, monkeypatch):
    """A half-loaded page must never overwrite the talent file."""
    monkeypatch.setattr(T, "CFBD_DIR", tmp_path)
    rows = [{"year": 2026, "team": f"T{i}", "talent": 800.0, "source": "247sports"}
            for i in range(T.MIN_TEAMS - 1)]
    with pytest.raises(SystemExit):
        T.write_talent(rows, 2026)
    assert not (tmp_path / "talent_2026.json").exists()


def test_publish_writes_cfbd_schema_and_records_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "CFBD_DIR", tmp_path)
    rows = [{"year": 2026, "team": f"T{i}", "talent": 800.0 + i,
             "source": "247sports"} for i in range(T.MIN_TEAMS)]
    dest = T.write_talent(rows, 2026)
    written = json.loads(dest.read_text())
    # CFBD schema so priors.load_features() needs no change
    assert {"year", "team", "talent"} <= set(written[0])
    assert all(r["source"] == "247sports" for r in written), "provenance dropped"
    manifest = json.loads((tmp_path / "talent_2026.source.json").read_text())
    assert manifest["source"] == "247sports"
    assert manifest["teams"] == T.MIN_TEAMS


def test_unresolvable_names_are_excluded_not_guessed():
    """Identity discipline: an unknown 247 spelling must not attach to a team."""
    raw = [{"slug": "ohio-state", "team": "Ohio State", "talent": 900.0},
           {"slug": "not-a-team", "team": "Definitely Not A Team", "talent": 800.0}]
    resolved, unresolved = T.resolve_names(raw, 2026)
    assert [r["team"] for r in resolved] == ["Ohio State"]
    assert unresolved == ["Definitely Not A Team"]


def test_reviewed_247_aliases_resolve():
    """The two names 247 spells differently from CFBD."""
    raw = [{"slug": "appalachian-state", "team": "Appalachian State", "talent": 700.0},
           {"slug": "louisiana-monroe", "team": "Louisiana-Monroe", "talent": 600.0}]
    resolved, unresolved = T.resolve_names(raw, 2026)
    assert unresolved == []
    assert sorted(r["team"] for r in resolved) == ["App State", "UL Monroe"]
