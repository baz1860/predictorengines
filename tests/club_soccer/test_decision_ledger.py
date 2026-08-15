#!/usr/bin/env python3
"""Decision ledger (adversarial blocker B) — frozen, executable, no hindsight."""
from __future__ import annotations

import csv

import pytest

from club_soccer import decision_ledger as DL


# ── executable quote selection ────────────────────────────────────────────

def test_selects_the_best_complete_market_book_never_a_median():
    side_odds = {
        "home": {"bookA": 2.00, "bookB": 2.10, "bookC": 1.95},
        "draw": {"bookA": 3.40, "bookB": 3.30},          # bookC missing -> incomplete
        "away": {"bookA": 3.80, "bookB": 3.70},
    }
    q = DL.select_executable_quote(side_odds, "home")
    assert q is not None
    book, odds, devig = q
    # bookC quotes only home, so it cannot win despite the best home price.
    assert book in ("bookA", "bookB")
    assert odds == 2.10 and book == "bookB", "best price among complete books"
    assert abs(sum(devig.values()) - 1.0) < 1e-9


def test_incomplete_market_yields_no_quote():
    side_odds = {"home": {"bookA": 2.0}, "draw": {}, "away": {"bookA": 3.5}}
    assert DL.select_executable_quote(side_odds, "home") is None


def test_devig_is_from_the_selected_book_only():
    side_odds = {"over": {"x": 1.90}, "under": {"x": 1.90}}
    _b, _o, devig = DL.select_executable_quote(side_odds, "over")
    assert devig["over"] == pytest.approx(0.5, abs=1e-9)


# ── cross-book consensus remains available as a diagnostic ────────────────

def test_consensus_benchmark_is_the_mean_of_complete_books():
    # This is reported for diagnostics. Production selection deliberately uses
    # the executing book's own de-vig so live and accumulated evidence measure
    # the same strategy.
    side_odds = {
        "home": {"A": 2.10, "B": 1.90},
        "away": {"A": 1.80, "B": 2.00},
    }
    q = DL.select_executable_quote(side_odds, "home")
    assert q[0] == "A" and q[1] == 2.10          # best home price = book A
    # Book A alone gives home devig 0.4615; consensus of A and B is 0.4872.
    cons = DL.market_consensus_devig(side_odds)
    assert cons["home"] == pytest.approx(0.48718, abs=1e-4)
    # A neutral p_model compared to the diagnostic consensus is +0.0128.
    assert 0.50 - cons["home"] == pytest.approx(0.01282, abs=1e-4)


def test_consensus_falls_back_to_single_book_when_only_one_quotes():
    side_odds = {"home": {"A": 2.00}, "away": {"A": 2.00}}
    cons = DL.market_consensus_devig(side_odds)
    assert cons == {"home": pytest.approx(0.5), "away": pytest.approx(0.5)}


def test_consensus_none_when_no_book_quotes_the_whole_market():
    side_odds = {"home": {"A": 2.0}, "away": {"B": 2.0}}   # no single complete book
    assert DL.market_consensus_devig(side_odds) is None


# ── provenance ────────────────────────────────────────────────────────────

def test_resolver_version_changes_with_the_alias_map(tmp_path, monkeypatch):
    monkeypatch.setattr(DL, "DATA", tmp_path)
    (tmp_path / "club_alias_map.json").write_text('{"alias": {}}')
    (tmp_path / "club_registry.json").write_text('{"index": {}}')
    v1 = DL.resolver_version()
    (tmp_path / "club_alias_map.json").write_text('{"alias": {"A": "B"}}')
    v2 = DL.resolver_version()
    assert v1 != v2, "a changed alias map must produce a new resolver version"


# ── settlement + join ─────────────────────────────────────────────────────

@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(DL, "DECISIONS", tmp_path / "dec.csv")
    monkeypatch.setattr(DL, "SETTLEMENTS", tmp_path / "set.csv")
    monkeypatch.setattr(DL, "CLOSING", tmp_path / "close.csv")
    monkeypatch.setattr(DL, "RAW_CLOSING", tmp_path / "raw_close.csv")
    monkeypatch.setattr(DL, "SETTLEMENT_CLV_V2", tmp_path / "clv_v2.csv")
    monkeypatch.setattr(DL, "DECISION_STRATEGIES", tmp_path / "strategies.csv")
    monkeypatch.setattr(DL, "IDENTITY_EXCLUSIONS", tmp_path / "identity_exclusions.csv")
    monkeypatch.setattr(DL, "DATA", tmp_path)
    return tmp_path


def _decision(**kw):
    base = {f: "" for f in DL.DECISION_FIELDS}
    base.update({"decision_id": "d1", "provider_fixture_id": "100",
                 "market": "1x2", "side": "home", "odds_executed": "2.0",
                 "p_model": "0.55", "p_book_devig": "0.48", "edge": "0.07",
                 "decision_lead_min": "90", "kickoff_utc": "2026-08-01T18:00:00+00:00",
                 "competition": "Premier League"})
    base.update(kw)
    return base


def test_settled_bets_joins_decision_and_settlement(ledger):
    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, [_decision()])
    DL._append(DL.SETTLEMENTS, DL.SETTLEMENT_FIELDS, [{
        "provider_fixture_id": "100", "market": "1x2", "side": "home",
        "won": 1, "clv": 0.03}])
    DL._append(DL.SETTLEMENT_CLV_V2, DL.SETTLEMENT_CLV_V2_FIELDS, [{
        "provider_fixture_id": "100", "market": "1x2", "side": "home",
        "fair_clv": 0.03, "fair_close_probability": 0.50,
        "devig_method": DL.CLV_DEVIG_METHOD,
        "schema_version": DL.CLV_SCHEMA_VERSION,
    }])
    bets = DL.settled_bets()
    assert len(bets) == 1
    assert bets[0]["won"] == 1 and bets[0]["clv"] == 0.03
    assert bets[0]["odds_executed"] == 2.0


def test_a_decision_without_settlement_is_not_a_bet(ledger):
    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, [_decision()])
    assert DL.settled_bets() == []


def test_capture_closing_snapshots_the_bsd_consensus(ledger, monkeypatch):
    """The near-kickoff pass stores a de-vigged CONSENSUS close for a fixture we
    have a decision on, so a league fd.co.uk never covers still earns a CLV
    reference."""
    from datetime import timedelta

    import api_keys
    import bsd_client

    from club_soccer import snapshot_odds as SO

    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, [_decision(
        provider_fixture_id="500", market="1x2", side="home")])

    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    ko = (now + timedelta(minutes=10)).isoformat()          # inside [1, 20] window
    monkeypatch.setattr(api_keys, "get_key", lambda *a, **k: "KEY")
    monkeypatch.setattr(bsd_client, "get_all_events",
                        lambda *a, **k: [{"id": "500", "event_date": ko}])
    monkeypatch.setattr(SO, "odds_comparison", lambda *a, **k: {"markets": {"1x2": {
        "HOME": {"bookmakers": {"A": {"decimal_odds": 2.00}, "B": {"decimal_odds": 2.10}}},
        "DRAW": {"bookmakers": {"A": {"decimal_odds": 3.40}, "B": {"decimal_odds": 3.30}}},
        "AWAY": {"bookmakers": {"A": {"decimal_odds": 3.80}, "B": {"decimal_odds": 3.70}}},
    }}})

    assert DL.capture_closing(api_key="KEY", verbose=False) == 3   # home/draw/away
    rows = {r["side"]: r for r in DL.load_closing()}
    assert set(rows) == {"home", "draw", "away"}
    assert rows["home"]["provider_fixture_id"] == "500"
    # consensus of the two complete books, de-vigged (~0.487 for home here)
    assert 0.45 < float(rows["home"]["p_close_devig"]) < 0.52
    raw = DL.load_raw_closing()
    assert len(raw) == 2
    assert all(r["schema_version"] == DL.CLV_SCHEMA_VERSION for r in raw)
    # idempotent: a second pass adds nothing
    assert DL.capture_closing(api_key="KEY", verbose=False) == 0


def test_settle_prefers_the_bsd_captured_close_over_fdcouk(ledger, monkeypatch):
    """When both exist, CLV is scored against the BSD close we captured for the
    exact fixture, not the fd.co.uk fallback map."""
    import pandas as pd

    from club_soccer import market_settlement as MS
    from club_soccer import model as M

    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, [_decision(
        provider_fixture_id="777", club_home="Alpha FC", club_away="Beta FC",
        market="1x2", side="home", odds_executed="2.5",
        kickoff_utc="2026-08-01T18:00:00+00:00")])
    DL._append(DL.CLOSING, DL.CLOSING_FIELDS, [{
        "provider_fixture_id": "777", "close_ts": "2026-08-01T17:50:00+00:00",
        "market": "1x2", "side": "home", "p_close_devig": "0.50",
        "close_lead_min": "10"}])

    fixtures = pd.DataFrame([{
        "fixture_id": "x", "date": "2026-08-01", "home": "Alpha FC",
        "away": "Beta FC", "home_goals": 2.0, "away_goals": 0.0}])
    monkeypatch.setattr(M, "load_fixtures", lambda: fixtures)
    # fd.co.uk fallback offers a DIFFERENT close; the BSD one must win.
    monkeypatch.setattr(MS, "closing_probs", lambda: (
        {MS.match_key("2026-08-01", "Alpha FC", "Beta FC"): {"home": 0.90}}, {}))

    assert DL.settle(verbose=False) == 1
    s = DL.load_settlements()[0]
    assert float(s["pinnacle_close_devig"]) == 0.50          # BSD close, not 0.90
    # CLV = log(2.5 * 0.50) = log(1.25) > 0
    import math
    assert float(s["clv"]) == round(math.log(2.5 * 0.50), 5)


def test_settle_matches_by_identity_not_the_discarded_fixture_id(ledger, monkeypatch):
    """Blocker 1: fixtures.csv.fixture_id is whichever provider row survived
    dedup, so it disagrees with the recorded BSD event id for most matches.
    Settlement must therefore join on canonical identity. Here the fixture's
    fixture_id (3252528302) differs from the decision's provider_fixture_id
    (198815); the result must still be found and settled."""
    import pandas as pd

    from club_soccer import market_settlement as MS
    from club_soccer import model as M

    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, [_decision(
        decision_id="d1", provider_fixture_id="198815",
        club_home="Hibernian", club_away="Motherwell",
        kickoff_utc="2026-08-01T18:00:00+00:00", market="1x2", side="home")])

    fixtures = pd.DataFrame([{
        "fixture_id": "3252528302", "date": "2026-08-01",
        "home": "Hibernian", "away": "Motherwell",
        "home_goals": 2.0, "away_goals": 1.0}])
    monkeypatch.setattr(M, "load_fixtures", lambda: fixtures)
    monkeypatch.setattr(M, "played", lambda df: df)
    monkeypatch.setattr(MS, "closing_probs", lambda: ({}, {}))

    assert DL.settle(verbose=False) == 1
    s = DL.load_settlements()[0]
    assert s["provider_fixture_id"] == "198815"   # row still keyed by the BSD id
    assert int(s["won"]) == 1                      # Hibernian won 2-1 at home


def test_settle_does_not_settle_a_fixture_that_has_not_finished(ledger, monkeypatch):
    """A decision with no matching finished fixture must not settle."""
    import pandas as pd

    from club_soccer import market_settlement as MS
    from club_soccer import model as M

    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, [_decision(
        club_home="Hibernian", club_away="Motherwell",
        kickoff_utc="2026-08-01T18:00:00+00:00")])
    monkeypatch.setattr(M, "load_fixtures", lambda: pd.DataFrame(
        columns=["fixture_id", "date", "home", "away", "home_goals", "away_goals"]))
    monkeypatch.setattr(M, "played", lambda df: df)
    monkeypatch.setattr(MS, "closing_probs", lambda: ({}, {}))
    assert DL.settle(verbose=False) == 0


def test_settle_requires_an_official_result_status(ledger, monkeypatch):
    """Finding 10: settlement grades an awarded (AWD) result but must NOT grade
    a live (in-play) row whose score is not final."""
    import pandas as pd

    from club_soccer import market_settlement as MS
    from club_soccer import model as M

    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, [
        _decision(decision_id="a", provider_fixture_id="1", club_home="Alpha FC",
                  club_away="Beta FC", kickoff_utc="2026-08-01T18:00:00+00:00",
                  side="home"),
        _decision(decision_id="b", provider_fixture_id="2", club_home="Gamma FC",
                  club_away="Delta FC", kickoff_utc="2026-08-02T18:00:00+00:00",
                  side="home"),
    ])
    fixtures = pd.DataFrame([
        {"date": "2026-08-01", "home": "Alpha FC", "away": "Beta FC",
         "home_goals": 3.0, "away_goals": 0.0, "status": "AWARDED"},
        {"date": "2026-08-02", "home": "Gamma FC", "away": "Delta FC",
         "home_goals": 1.0, "away_goals": 0.0, "status": "IN_PLAY"},
    ])
    monkeypatch.setattr(M, "load_fixtures", lambda: fixtures)
    monkeypatch.setattr(MS, "closing_probs", lambda: ({}, {}))

    assert DL.settle(verbose=False) == 1          # only the AWARDED fixture
    settled = DL.load_settlements()
    assert [s["provider_fixture_id"] for s in settled] == ["1"]


def test_decision_ids_are_unique_per_fixture_market_side():
    a = DL._decision_id(100, "1x2", "home")
    b = DL._decision_id(100, "1x2", "away")
    c = DL._decision_id(100, "1x2", "home")
    assert a == c and a != b


# ── the property that justifies the whole rewrite ─────────────────────────

def test_backtest_metrics_do_not_depend_on_the_alias_map(ledger, monkeypatch):
    """THE acceptance test for blocker B. The reconstruction backtest changed
    its sample with the alias-map version; the ledger backtest must not — it
    reads only frozen rows."""
    from club_soccer import decision_time_backtest as B

    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, [
        _decision(decision_id="d1", provider_fixture_id="1", side="home"),
        _decision(decision_id="d2", provider_fixture_id="2", side="over",
                  market="total25", odds_executed="1.9", p_model="0.6"),
    ])
    DL._append(DL.SETTLEMENTS, DL.SETTLEMENT_FIELDS, [
        {"provider_fixture_id": "1", "market": "1x2", "side": "home", "won": 1, "clv": 0.02},
        {"provider_fixture_id": "2", "market": "total25", "side": "over", "won": 0, "clv": -0.01},
    ])
    bets = B.build_bets()
    assert len(bets) == 2
    # The frozen path must not CALL the resolver or read the alias-map file.
    import inspect
    # Strip the docstring, then check for actual usage.
    src = inspect.getsource(B._settled_frame)
    body = src.split('"""')[2] if src.count('"""') >= 2 else src
    assert "canonical_name(" not in body
    assert "club_alias_map" not in body
    assert "settled_bets" in body, "must read from the frozen ledger"


def test_routine_model_refits_do_not_reset_the_evidence_cohort(ledger):
    """Parameter hashes are provenance, not a new betting strategy."""
    from club_soccer import decision_time_backtest as B

    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, [
        _decision(
            decision_id="d1", provider_fixture_id="1",
            decision_ts="2026-07-01T12:00:00+00:00",
            resolver_version="resolver", code_hash="strategy",
            model_hash="fit-one", strategy_eligible="1",
        ),
        _decision(
            decision_id="d2", provider_fixture_id="2",
            decision_ts="2026-07-02T12:00:00+00:00",
            resolver_version="resolver", code_hash="strategy",
            model_hash="fit-two", strategy_eligible="1",
        ),
    ])
    DL._append(DL.SETTLEMENTS, DL.SETTLEMENT_FIELDS, [
        {"provider_fixture_id": "1", "market": "1x2", "side": "home",
         "won": 1, "clv": 0.02},
        {"provider_fixture_id": "2", "market": "1x2", "side": "home",
         "won": 0, "clv": -0.01},
    ])
    bets = B.build_bets()
    assert len(bets) == 2
    assert set(bets["model_hash"]) == {"fit-one", "fit-two"}


def test_code_and_resolver_changes_do_not_reset_explicit_strategy(ledger):
    """Compatibility is a deliberate strategy contract, not byte equality."""
    from club_soccer import decision_time_backtest as B
    from club_soccer.strategy_contract import STRATEGY_VERSION, manifest_hash

    decisions = [
        _decision(decision_id="d1", provider_fixture_id="1",
                  decision_ts="2026-07-01T12:00:00+00:00",
                  resolver_version="aliases-a", code_hash="bytes-a"),
        _decision(decision_id="d2", provider_fixture_id="2",
                  decision_ts="2026-07-02T12:00:00+00:00",
                  resolver_version="aliases-b", code_hash="bytes-b"),
    ]
    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, decisions)
    DL._append(DL.DECISION_STRATEGIES, DL.DECISION_STRATEGY_FIELDS, [
        {"decision_id": d["decision_id"], "strategy_version": STRATEGY_VERSION,
         "strategy_manifest_hash": manifest_hash(),
         "identity_version": d["resolver_version"],
         "pricing_code_hash": d["code_hash"],
         "recorded_at_utc": d["decision_ts"]}
        for d in decisions
    ])
    DL._append(DL.SETTLEMENTS, DL.SETTLEMENT_FIELDS, [
        {"provider_fixture_id": "1", "market": "1x2", "side": "home",
         "won": 1, "clv": 0.01},
        {"provider_fixture_id": "2", "market": "1x2", "side": "home",
         "won": 0, "clv": -0.01},
    ])

    bets = B.build_bets(strategy_version=STRATEGY_VERSION)
    assert set(bets["key"]) == {"1", "2"}
    assert set(bets["resolver_version"]) == {"aliases-a", "aliases-b"}
    assert set(bets["code_hash"]) == {"bytes-a", "bytes-b"}


def test_incompatible_strategy_is_diagnostic_only(ledger):
    from club_soccer import decision_time_backtest as B
    from club_soccer.strategy_contract import STRATEGY_VERSION, manifest_hash

    decisions = [
        _decision(decision_id="old", provider_fixture_id="1",
                  decision_ts="2026-07-01T12:00:00+00:00"),
        _decision(decision_id="new", provider_fixture_id="2",
                  decision_ts="2026-07-02T12:00:00+00:00"),
    ]
    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, decisions)
    DL._append(DL.DECISION_STRATEGIES, DL.DECISION_STRATEGY_FIELDS, [
        {"decision_id": "old", "strategy_version": "retired-v0",
         "strategy_manifest_hash": "old", "identity_version": "r",
         "pricing_code_hash": "c", "recorded_at_utc": decisions[0]["decision_ts"]},
        {"decision_id": "new", "strategy_version": STRATEGY_VERSION,
         "strategy_manifest_hash": manifest_hash(), "identity_version": "r",
         "pricing_code_hash": "c", "recorded_at_utc": decisions[1]["decision_ts"]},
    ])
    DL._append(DL.SETTLEMENTS, DL.SETTLEMENT_FIELDS, [
        {"provider_fixture_id": fid, "market": "1x2", "side": "home",
         "won": 1, "clv": 0.01} for fid in ("1", "2")
    ])

    current = B.build_bets(strategy_version=STRATEGY_VERSION)
    views = B._diagnostic_views(current)
    assert list(current["key"]) == ["2"]
    assert views["all_history"]["n_rows"] == 2
    assert views["exclusions"]["incompatible_strategy"] == 1


def test_identity_review_excludes_only_affected_fixture_and_can_reinstate(
        ledger, monkeypatch):
    from club_soccer import decision_time_backtest as B

    monkeypatch.setattr(DL, "resolver_version", lambda: "review-v1")
    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, [
        _decision(decision_id="d1", provider_fixture_id="1"),
        _decision(decision_id="d2", provider_fixture_id="2"),
    ])
    DL._append(DL.SETTLEMENTS, DL.SETTLEMENT_FIELDS, [
        {"provider_fixture_id": fid, "market": "1x2", "side": "home",
         "won": 1, "clv": 0.01} for fid in ("1", "2")
    ])

    DL.review_identity(provider_fixture_id="1", action="exclude",
                       reason="manual identity mismatch review")
    assert list(B.build_bets()["key"]) == ["2"]
    DL.review_identity(provider_fixture_id="1", action="reinstate",
                       reason="provider identity confirmed")
    assert set(B.build_bets()["key"]) == {"1", "2"}


def test_ledger_backtest_produces_a_valid_gated_artifact(ledger):
    from club_soccer import decision_time_backtest as B
    DL._append(DL.DECISIONS, DL.DECISION_FIELDS, [_decision()])
    DL._append(DL.SETTLEMENTS, DL.SETTLEMENT_FIELDS, [{
        "provider_fixture_id": "100", "market": "1x2", "side": "home",
        "won": 1, "clv": 0.03}])
    df = B.build_bets()
    assert not df.empty
    assert set(["key", "market", "side", "odds", "p_model", "won"]).issubset(df.columns)


# ── wiring ────────────────────────────────────────────────────────────────

def test_recorded_in_the_daily_pipeline():
    import inspect
    from club_soccer import season
    src = inspect.getsource(season)
    assert "decision_ledger" in src
    assert "Record staking decisions" in src


def test_decision_window_is_within_gate_bounds():
    assert DL.MIN_LEAD_MIN >= 60
    assert DL.MAX_LEAD_MIN <= 7 * 24 * 60


def test_decision_window_supports_frequent_capture(tmp_path=None):
    """Finding 8: the window must be wide enough that a 15-min capture cadence
    (with the odd skipped run) never drops a fixture, yet stay a realistic
    pre-kickoff instant close to the 60-min floor."""
    assert DL.MIN_LEAD_MIN == 60
    assert 90 <= DL.MAX_LEAD_MIN <= 180
    # span must comfortably exceed a couple of 15-min intervals
    assert DL.MAX_LEAD_MIN - DL.MIN_LEAD_MIN >= 45


def test_a_frequent_capture_schedule_is_deployed():
    """Reaching ~1,000 settled decisions in a season needs capture through the
    whole day, not one 07:30 run. A dedicated capture agent must run on a short
    interval and invoke the record/settle CLI."""
    import re
    from pathlib import Path

    p = (Path(__file__).resolve().parents[2] / "deploy" /
         "com.barrie.sportspredictor.clubsoccer.capture.plist")
    assert p.exists(), "capture launch agent is missing"
    text = p.read_text()
    m = re.search(r"<key>StartInterval</key>\s*<integer>(\d+)</integer>", text)
    assert m, "capture agent must run on a StartInterval, not once a day"
    assert int(m.group(1)) <= 1800, "capture must run at least every 30 min"
    assert "decision_ledger" in text and "--record" in text and "--settle" in text


def test_backtest_no_longer_reconstructs_by_default():
    """Only the frozen ledger path may exist; hindsight reconstruction is gone."""
    import inspect
    from club_soccer import decision_time_backtest as B
    assert "settled_bets" in inspect.getsource(B._settled_frame)
    assert not hasattr(B, "build_bets_reconstructed")
