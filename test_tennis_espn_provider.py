"""Regression tests for the free ESPN ATP completed-results feed."""
from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import asdict

from tennis import providers as P


def _competition(match_id: str, date: str, state: str = "post") -> dict:
    completed = state == "post"
    return {
        "id": match_id,
        "date": f"{date}T14:00Z",
        "status": {
            "type": {
                "state": state,
                "completed": completed,
                "detail": "Final" if completed else "Scheduled",
            },
        },
        "round": {"displayName": "Quarterfinal"},
        "competitors": [
            {
                "winner": False,
                "athlete": {"fullName": "Beta Player"},
                "linescores": [
                    {"value": 6, "tiebreak": 5},
                    {"value": 3},
                    {"value": 4},
                ],
            },
            {
                "winner": True,
                "athlete": {"fullName": "Alpha Player"},
                "linescores": [
                    {"value": 7, "tiebreak": 7},
                    {"value": 6},
                    {"value": 6},
                ],
            },
        ],
    }


def _payload() -> dict:
    return {
        "events": [
            {
                "id": "999-2026",
                "name": "Test Argentina Open",
                "major": False,
                "groupings": [
                    {
                        "grouping": {
                            "slug": "mens-doubles",
                            "displayName": "Men's Doubles",
                        },
                        "competitions": [_competition("doubles", "2026-07-24")],
                    },
                    {
                        "grouping": {
                            "slug": "mens-singles",
                            "displayName": "Men's Singles",
                        },
                        "competitions": [
                            _competition("complete", "2026-07-24"),
                            _competition("future", "2026-07-25", state="pre"),
                        ],
                    },
                ],
            },
        ],
    }


def test_espn_provider_first_run_normalises_and_caches(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(P, "CACHE_DIR", tmp_path)
    calls: list[str] = []

    def fake_http(url: str, **_kwargs) -> str:
        calls.append(url)
        return json.dumps(_payload())

    monkeypatch.setattr(P, "_http_text", fake_http)
    provider = P.ESPNResultsProvider(
        today=dt.date(2026, 7, 25), refresh_hours=24,
    )
    rows = provider.matches_for(2026, "atp")

    assert len(rows) == 1
    row = rows[0]
    assert row.date == "2026-07-24"
    assert row.tourney_id == "espn-999-2026"
    assert row.surface == "clay"
    assert row.round == "QF"
    assert row.best_of == 3
    assert (row.winner, row.loser) == ("Alpha Player", "Beta Player")
    assert row.score == "7-6(5) 6-3 6-4"
    assert (row.winner_sets, row.loser_sets) == (3, 0)
    assert "dates=20260101-20260725" in calls[0]
    assert (tmp_path / "espn_atp_2026.json").exists()

    # A second call inside the refresh interval is offline and cache-backed.
    monkeypatch.setattr(
        P, "_http_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fresh cache should prevent another HTTP request")
        ),
    )
    assert provider.matches_for(2026, "atp") == rows


def test_espn_score_marks_retirements() -> None:
    competition = _competition("retired", "2026-07-24")
    competition["status"]["type"]["detail"] = "Retired"
    winner = competition["competitors"][1]
    loser = competition["competitors"][0]
    assert P._espn_score(winner, loser, competition).endswith(" RET")


def test_espn_provider_uses_stale_cache_on_network_failure(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(P, "CACHE_DIR", tmp_path)
    cache = {
        "source": P._ESPN_ATP_URL,
        "year": 2026,
        "fetched_through": "2026-07-20",
        "events": _payload()["events"],
    }
    (tmp_path / "espn_atp_2026.json").write_text(json.dumps(cache))
    monkeypatch.setattr(P, "_http_text", lambda *_args, **_kwargs: None)

    provider = P.ESPNResultsProvider(
        today=dt.date(2026, 7, 25), refresh_hours=0,
    )
    rows = provider.matches_for(2026, "atp")
    assert len(rows) == 1
    assert rows[0].winner == "Alpha Player"


def test_accumulate_dedupes_tml_espn_overlap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(P, "DATA_DIR", tmp_path)
    monkeypatch.setattr(P, "MATCHES_CSV", tmp_path / "matches.csv")
    existing = P.MatchRecord(
        date="2026-07-24",
        tourney_id="2026-999",
        tourney_name="Argentina Open",
        tour="atp",
        surface="clay",
        round="QF",
        best_of=3,
        winner="Álpha Player",
        loser="Beta Player",
        winner_rank=1,
        loser_rank=2,
        winner_sets=2,
        loser_sets=0,
        score="6-3 6-4",
    )
    with P.MATCHES_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=P.MATCH_COLUMNS)
        writer.writeheader()
        writer.writerow(asdict(existing))

    incoming = P.MatchRecord(
        **{
            **asdict(existing),
            "tourney_id": "espn-999-2026",
            "winner": "Alpha Player",
            "winner_rank": 9999,
            "loser_rank": 9999,
        },
    )

    class OneRowProvider:
        name = "one-row"

        def seasons_available(self):
            return [2026]

        def matches_for(self, year, tour):
            return [incoming] if year == 2026 and tour == "atp" else []

    status = P.accumulate_matches(
        provider=OneRowProvider(), years=[2026], tours=["atp"], verbose=False,
    )
    assert status["added"] == 0
    with P.MATCHES_CSV.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_accumulate_keeps_same_source_rematches(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(P, "DATA_DIR", tmp_path)
    monkeypatch.setattr(P, "MATCHES_CSV", tmp_path / "matches.csv")
    base = P.MatchRecord(
        date="2026-11-10",
        tourney_id="2026-finals",
        tourney_name="Tour Finals",
        tour="atp",
        surface="hard",
        round="RR",
        best_of=3,
        winner="Alpha Player",
        loser="Beta Player",
        winner_rank=1,
        loser_rank=2,
        winner_sets=2,
        loser_sets=0,
        score="6-3 6-4",
    )
    rematch = P.MatchRecord(
        **{
            **asdict(base),
            "tourney_id": "2026-finals-rematch",
            "round": "F",
        },
    )

    class RematchProvider:
        name = "archive"

        def seasons_available(self):
            return [2026]

        def matches_for(self, year, tour):
            return [base, rematch] if year == 2026 and tour == "atp" else []

    status = P.accumulate_matches(
        provider=RematchProvider(), years=[2026], tours=["atp"], verbose=False,
    )
    assert status["added"] == 2


def test_composite_routes_current_atp_to_espn(monkeypatch) -> None:
    provider = P.CompositeProvider()
    current_year = dt.date.today().year
    sentinel = P.MatchRecord(
        date=f"{current_year}-01-01",
        tourney_id="espn-test",
        tourney_name="Test",
        tour="atp",
        surface="hard",
        round="F",
        best_of=3,
        winner="A",
        loser="B",
        winner_rank=9999,
        loser_rank=9999,
        winner_sets=2,
        loser_sets=0,
        score="6-3 6-4",
    )
    monkeypatch.setattr(provider._atp_live, "matches_for", lambda *_args: [sentinel])
    monkeypatch.setattr(
        provider._atp_history, "matches_for",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("current season should prefer ESPN")
        ),
    )
    assert provider.matches_for(current_year, "atp") == [sentinel]
