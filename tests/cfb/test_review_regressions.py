"""Regressions for the 2026-08-07 adversarial-review fixes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfb import bankroll as BANKROLL
from cfb import elo as E
from cfb import epa as X
from cfb import identity as IDENTITY
from cfb import power as P
from cfb import season as S
from app.engines import _inproc


# ── unknown teams are domain errors, not process exits ──────────────────────

def test_predict_raises_valueerror_not_systemexit():
    """SystemExit is a BaseException; a stray one killed the host app."""
    with pytest.raises(ValueError):
        E.predict({"A": 1500.0}, 0.057, 13.0, "A", "Nope")
    params = {"teams": {"A": {"off": 1.0, "def": 1.0}}, "mu": 25.0,
              "hfa": 2.0, "sigma": 13.0, "sigma_total": 13.0}
    with pytest.raises(ValueError):
        P.predict(params, "A", "Nope")
    with pytest.raises(ValueError):
        X.predict({**params, "c0": 0.0, "c1": 1.0}, "A", "Nope")


def test_inproc_converts_systemexit_into_a_handled_error():
    def boom(_params):
        raise SystemExit("engine tried to exit")

    with pytest.raises(ValueError):
        _inproc.run_inprocess({"predict": boom}, "predict", {})


# ── a stale quote must not shadow an executable one ─────────────────────────

def test_load_market_prefers_executable_quote_over_better_stale_odds(monkeypatch):
    slate = pd.DataFrame([{
        "date": pd.Timestamp("2026-08-30"), "season": 2026,
        "home_team": "Alpha", "away_team": "Bravo", "neutral": False,
        "home_div": "fbs", "away_div": "fbs", "game_id": "1",
    }])
    prepared = pd.DataFrame([
        # stale but better priced
        {"date": "2026-08-30", "home": "Alpha", "away": "Bravo",
         "market": "ml", "side": "home", "line": np.nan, "odds": 1.95,
         "p_implied": 0.50, "bookmaker": "stale_book", "quote_eligible": False,
         "fixture_matched": True, "pair_complete": True},
        # fresh, executable, slightly worse price
        {"date": "2026-08-30", "home": "Alpha", "away": "Bravo",
         "market": "ml", "side": "home", "line": np.nan, "odds": 1.91,
         "p_implied": 0.52, "bookmaker": "live_book", "quote_eligible": True,
         "fixture_matched": True, "pair_complete": True},
    ])
    monkeypatch.setattr(S.os.path, "exists", lambda p: True)
    monkeypatch.setattr(S.pd, "read_csv", lambda *a, **k: prepared.copy())
    monkeypatch.setattr(S, "prepare_odds", lambda odds, seasons: prepared.copy())

    market = S.load_market(slate)
    quote = market[("2026-08-30", "Alpha", "Bravo")]["ml"]["home"]
    assert quote[3] == "live_book", "stale quote shadowed an executable one"
    assert quote[4] is True


# ── settlement must identify the exact game ─────────────────────────────────

def _rematch_games():
    return pd.DataFrame([
        {"game_id": "111", "season": 2026, "week": 10, "date": "2026-11-07",
         "home_team": "Alpha", "away_team": "Bravo",
         "home_points": 10, "away_points": 30},
        {"game_id": "222", "season": 2026, "week": 15, "date": "2026-12-06",
         "home_team": "Alpha", "away_team": "Bravo",
         "home_points": 40, "away_points": 3},
    ])


def test_settlement_uses_game_id_for_a_rematch():
    games = _rematch_games()
    bet = {"cfbd_game_id": "222", "date": "2026-12-06", "home": "Alpha",
           "away": "Bravo", "market": "ml", "side": "home", "line": None,
           "odds": 2.0, "stake": 10.0}
    assert BANKROLL.settle_bet(bet, games) == pytest.approx(10.0)
    losing = {**bet, "cfbd_game_id": "111", "date": "2026-11-07"}
    assert BANKROLL.settle_bet(losing, games) == pytest.approx(-10.0)


def test_settlement_refuses_ambiguous_legacy_match():
    """Without a game ID, an ambiguous name match must not settle at all."""
    games = _rematch_games()
    games["date"] = "2026-12-06"  # force two same-day candidates
    bet = {"cfbd_game_id": "", "date": "2026-12-06", "home": "Alpha",
           "away": "Bravo", "market": "ml", "side": "home", "line": None,
           "odds": 2.0, "stake": 10.0}
    assert BANKROLL.settle_bet(bet, games) is None


# ── identity caching must not change identity decisions ─────────────────────

def test_identity_cache_preserves_strict_resolution():
    first = IDENTITY.resolve("Ohio State Buckeyes", 2026, provider="the-odds-api")
    second = IDENTITY.resolve("Ohio State Buckeyes", 2026, provider="the-odds-api")
    assert first == second
    assert first["canonical"] == "Ohio State"
    # An unreviewed spelling must still be refused after caching.
    assert IDENTITY.resolve("Ohio St Buckeyes XYZ", 2026,
                            provider="the-odds-api") is None


def test_schedule_catalog_cache_actually_hits():
    IDENTITY.schedule_catalog(2026)
    calls = {"n": 0}
    real_loads = IDENTITY.json.loads

    def counting_loads(*a, **k):
        calls["n"] += 1
        return real_loads(*a, **k)

    IDENTITY.json.loads = counting_loads
    try:
        for _ in range(25):
            IDENTITY.schedule_catalog(2026)
    finally:
        IDENTITY.json.loads = real_loads
    assert calls["n"] == 0, "schedule catalog re-parsed despite the cache"


# ── reviewed-schedule gate: ignore noise, still catch real changes ──────────

def _schedule_fixture(tmp_path, **overrides):
    import json as _json
    game = {
        "id": 401856766, "season": 2026, "week": 1, "seasonType": "regular",
        "startDate": "2026-09-05T16:00:00.000Z",
        "homeId": 2628, "homeTeam": "TCU", "homeClassification": "fbs",
        "awayId": 153, "awayTeam": "North Carolina", "awayClassification": "fbs",
        "neutralSite": False, "completed": False,
        "homePregameElo": None, "awayPregameElo": None,
    }
    game.update(overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "schedule_2026.json"
    path.write_text(_json.dumps([game]))
    return path


def test_schedule_gate_ignores_fields_the_model_never_reads(tmp_path):
    """CFBD backfilling its own pregame Elo must not force a re-review."""
    base = _schedule_fixture(tmp_path / "a")
    noisy = _schedule_fixture(tmp_path / "b",
                              homePregameElo=1613, awayPregameElo=1380)
    assert (IDENTITY.schedule_identity_sha256(2026, base)
            == IDENTITY.schedule_identity_sha256(2026, noisy))


@pytest.mark.parametrize("field,value", [
    ("startDate", "2026-09-05T20:00:00.000Z"),   # kickoff moved
    ("homeTeam", "Texas Christian"),             # identity changed
    ("homeId", 9999),                            # different team ID
    ("awayClassification", "fcs"),               # reclassified
    ("neutralSite", True),                       # venue semantics
    ("week", 2),                                 # rescheduled week
    ("completed", True),                         # result state
])
def test_schedule_gate_still_catches_decision_relevant_changes(tmp_path, field, value):
    base = _schedule_fixture(tmp_path / "base")
    changed = _schedule_fixture(tmp_path / "changed", **{field: value})
    assert (IDENTITY.schedule_identity_sha256(2026, base)
            != IDENTITY.schedule_identity_sha256(2026, changed)), (
        f"a change to {field} no longer triggers schedule re-review")


def test_schedule_gate_catches_added_and_removed_events(tmp_path):
    import json as _json
    base = _schedule_fixture(tmp_path / "base")
    games = _json.loads(base.read_text())
    extra = dict(games[0], id=401999999, homeTeam="Alpha", awayTeam="Bravo")
    more = tmp_path / "more_schedule_2026.json"
    more.parent.mkdir(parents=True, exist_ok=True)
    more.write_text(_json.dumps(games + [extra]))
    assert (IDENTITY.schedule_identity_sha256(2026, base)
            != IDENTITY.schedule_identity_sha256(2026, more))


def test_schedule_identity_is_order_independent(tmp_path):
    import json as _json
    base = _schedule_fixture(tmp_path / "base")
    games = _json.loads(base.read_text())
    extra = dict(games[0], id=401999999, homeTeam="Alpha", awayTeam="Bravo")
    a = tmp_path / "a_schedule.json"
    b = tmp_path / "b_schedule.json"
    a.parent.mkdir(parents=True, exist_ok=True)
    a.write_text(_json.dumps([games[0], extra]))
    b.write_text(_json.dumps([extra, games[0]]))
    assert (IDENTITY.schedule_identity_sha256(2026, a)
            == IDENTITY.schedule_identity_sha256(2026, b))


def test_current_reviewed_schedule_record_matches_disk():
    """The committed review record must describe the schedule actually on disk."""
    import json as _json
    from pathlib import Path

    review = _json.loads(
        Path("cfb/data/reviewed_schedule.json").read_text())
    assert (review["schedule_identity_sha256"]
            == IDENTITY.schedule_identity_sha256(2026))


# ── re-freeze script safety rails ───────────────────────────────────────────

def test_refreeze_requires_explicit_confirmation():
    """Without --confirm it must refuse and exit non-zero, changing nothing."""
    import subprocess
    from pathlib import Path

    baseline = Path("cfb/data/validation_baseline.json")
    before = baseline.read_bytes()
    result = subprocess.run(["bash", "cfb/refreeze.sh"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "--confirm" in result.stdout
    assert baseline.read_bytes() == before, "refused run still mutated artifacts"


def test_refreeze_rejects_unknown_arguments():
    """A typo'd flag must not be silently treated as an unconfirmed run."""
    import subprocess

    result = subprocess.run(["bash", "cfb/refreeze.sh", "--yes"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "unknown argument" in result.stderr


def test_refreeze_is_not_invoked_by_the_routine_update():
    """update.sh must never rebaseline; that would defeat the frozen gate."""
    from pathlib import Path

    update = Path("cfb/update.sh").read_text()
    assert "refreeze" not in update
    assert "--update-baseline" not in update


# ── win-total plausibility guard ────────────────────────────────────────────

def test_game_totals_are_rejected_as_season_win_totals():
    from cfb import fetch_win_total_lines as F

    def payload(point):
        return [{"bookmakers": [{"title": "b", "markets": [
            {"outcomes": [{"point": point, "price": -110}] * 6}]}]}] * 3

    assert F.looks_like_win_totals(payload(8.5)) is True
    assert F.looks_like_win_totals(payload(52.5)) is False


def test_win_total_parser_drops_rows_without_an_identified_side():
    from cfb import fetch_win_total_lines as F

    no_side = [{"bookmakers": [{"title": "DK", "markets": [
        {"outcomes": [{"name": "Texas", "point": 9.5, "price": -110}]}]}]}]
    assert F.parse_win_totals(no_side) == []


def test_outright_winner_keys_are_not_treated_as_season_wins():
    """`americanfootball_ncaaf_championship_winner` contains "win" but prices a
    single champion — probing it for win totals wastes quota and can't work."""
    from cfb import fetch_win_total_lines as F

    live = ["americanfootball_ncaaf",
            "americanfootball_ncaaf_championship_winner"]
    assert F.season_win_keys(live) == []
    assert F.season_win_keys(
        live + ["americanfootball_ncaaf_team_season_wins"]
    ) == ["americanfootball_ncaaf_team_season_wins"]


# ── win-total line consolidation ────────────────────────────────────────────

def test_american_odds_are_medianed_in_probability_space():
    """-110 and +100 naively average to -5, which is not a valid price."""
    from cfb.compare_win_totals import (american_to_implied,
                                        implied_to_american,
                                        _median_american)

    assert round(implied_to_american(american_to_implied(-110))) == -110
    assert round(implied_to_american(american_to_implied(150))) == 150
    combined = _median_american([-110, 100])
    assert abs(combined) >= 100, "produced an impossible American price"
    assert combined == -105
    # An already-invalid price is treated as the -110 default, not propagated.
    assert american_to_implied(-5) == pytest.approx(american_to_implied(-110))


# ── the weekly refresh must not destroy imported line history ───────────────

def _line_row(season, week, home, away, line, juice=None):
    return {"season": season, "week": week, "home_team": home,
            "away_team": away, "home_line": line, "home_odds": juice,
            "away_odds": juice, "n_books": 3}


def test_refresh_keeps_imported_games_the_mirror_lacks():
    """Regression: once the mirror's coverage reached the present, the old
    `season > mirror.max()` rule kept nothing and every `update.sh` silently
    discarded CFBD-imported lines."""
    from cfb.fetch_data import merge_with_imported

    # Mirror now reaches the SAME max season as the imported data — the exact
    # condition that made the old rule a no-op.
    mirror = pd.DataFrame([
        _line_row(2019, 1, "Alpha", "Bravo", -3.5, -110),
        _line_row(2025, 1, "Alpha", "Bravo", -7.0),
    ])
    existing = pd.concat([mirror, pd.DataFrame([
        _line_row(2025, 2, "Charlie", "Delta", -1.5),   # mirror lacks this game
        _line_row(2025, 3, "Echo", "Foxtrot", +2.5),    # and this one
    ])], ignore_index=True)

    out = merge_with_imported(mirror, existing)
    assert len(out) == 4, "imported games absent from the mirror were dropped"
    games = set(zip(out["season"], out["week"], out["home_team"]))
    assert (2025, 2, "Charlie") in games
    assert (2025, 3, "Echo") in games
    # Mirror rows still win where both have the game (they carry the juice).
    kept = out[(out["season"] == 2019) & (out["home_team"] == "Alpha")]
    assert float(kept.iloc[0]["home_odds"]) == -110


def test_refresh_output_is_byte_stable_for_unchanged_data():
    """Row churn alone must not move the validation fingerprint."""
    from cfb.fetch_data import merge_with_imported

    mirror = pd.DataFrame([
        _line_row(2024, 5, "Alpha", "Bravo", -3.5),
        _line_row(2024, 1, "Charlie", "Delta", -1.5),
        _line_row(2023, 9, "Echo", "Foxtrot", +2.5),
    ])
    shuffled = mirror.iloc[::-1].reset_index(drop=True)
    first = merge_with_imported(mirror, mirror)
    second = merge_with_imported(shuffled, shuffled)
    pd.testing.assert_frame_equal(first, second)
    assert first.to_csv(index=False) == second.to_csv(index=False)


def test_alias_and_canonical_rows_consolidate_to_one_bet():
    """Two spellings of one team must not become two bets on one market."""
    from cfb.compare_win_totals import resolve_line_teams

    lines = pd.DataFrame([
        {"team": "Ohio State", "line": 11.5, "over_odds": -110,
         "under_odds": -110, "books": 1},
        {"team": "Ohio State Buckeyes", "line": 11.5, "over_odds": -120,
         "under_odds": 100, "books": 3},
    ])
    resolved, unresolved = resolve_line_teams(lines, 2026)
    assert unresolved == []
    assert list(resolved["team"]) == ["Ohio State"]
    assert int(resolved.iloc[0]["books"]) == 4
    assert abs(int(resolved.iloc[0]["under_odds"])) >= 100
