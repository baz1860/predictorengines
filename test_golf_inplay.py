"""Tests for the in-play (live-score) golf path: score extraction, the refresh
snapshot, and the engine auto-route from pre-tournament to in-play."""

import csv

import pytest

from golf import engine
from golf.providers import espn as espn_mod
from golf.providers.espn import EspnGolfProvider, EspnEvent


# ──────────────────────────────────────────────
# Synthetic ESPN payload helpers
# ──────────────────────────────────────────────

def _round_line(period, to_par, holes=18):
    return {"period": period, "displayValue": to_par,
            "linescores": [{"period": h, "value": 4} for h in range(1, holes + 1)]}


def _competitor(name, rounds, cut=False):
    c = {"athlete": {"displayName": name, "id": name}, "linescores": rounds}
    if cut:
        c["status"] = {"type": {"name": "STATUS_CUT", "description": "Cut"}}
    return c


def _payload(competitors):
    return {"events": [{"id": "TEST", "competitions": [{"competitors": competitors}]}]}


def _patched_provider(monkeypatch, competitors):
    prov = EspnGolfProvider()
    monkeypatch.setattr(prov, "current_event_payload",
                        lambda *a, **k: _payload(competitors))
    return prov


# ──────────────────────────────────────────────
# Unit: to-par parsing
# ──────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("-7", -7.0), ("E", 0.0), ("+2", 2.0), ("", 0.0), ("--", 0.0), (None, 0.0),
])
def test_to_par_parsing(raw, expected):
    assert espn_mod._to_par(raw) == expected


# ──────────────────────────────────────────────
# Unit: completed_round_scores
# ──────────────────────────────────────────────

def test_completed_round_scores_sums_finished_rounds(monkeypatch):
    comps = [
        _competitor("Leader",   [_round_line(1, "-5"), _round_line(2, "-4")]),
        _competitor("Chaser",   [_round_line(1, "-2"), _round_line(2, "-1")]),
        _competitor("Even",     [_round_line(1, "E"),  _round_line(2, "+1")]),
    ]
    prov = _patched_provider(monkeypatch, comps)
    rows, rounds_done = prov.completed_round_scores("TEST")

    assert rounds_done == 2
    by = {r["name"]: r for r in rows}
    assert by["Leader"]["score"] == -9      # -5 + -4
    assert by["Chaser"]["score"] == -3
    assert by["Even"]["score"] == 1
    assert all(r["made_cut"] == 1 for r in rows)


def test_in_progress_round_not_counted(monkeypatch):
    # Round 2 only 9 holes in → must not be counted or added to the total.
    comps = [
        _competitor("A", [_round_line(1, "-3"), _round_line(2, "-2", holes=9)]),
        _competitor("B", [_round_line(1, "-1"), _round_line(2, "-2", holes=9)]),
    ]
    prov = _patched_provider(monkeypatch, comps)
    rows, rounds_done = prov.completed_round_scores("TEST")
    assert rounds_done == 1
    assert {r["name"]: r["score"] for r in rows} == {"A": -3, "B": -1}


def test_cut_player_excluded(monkeypatch):
    comps = [
        _competitor("Survivor", [_round_line(1, "-4"), _round_line(2, "-3")]),
        _competitor("CutGuy",   [_round_line(1, "+5"), _round_line(2, "+6")], cut=True),
    ]
    prov = _patched_provider(monkeypatch, comps)
    rows, rounds_done = prov.completed_round_scores("TEST")
    by = {r["name"]: r for r in rows}
    assert by["Survivor"]["made_cut"] == 1
    assert by["CutGuy"]["made_cut"] == 0


def test_cut_line_computed_when_feed_omits_it(monkeypatch):
    # ESPN leaves status null even for cut players, so the cut must be derived
    # from 36-hole scores: keep the lowest cut_size and ties. Two 36-hole rounds
    # each (round1 = target score, round2 = level).
    def player(name, score36):
        return _competitor(name, [_round_line(1, f"{score36:+d}" if score36 else "E"),
                                   _round_line(2, "E")])
    comps = [player("A", -5), player("B", -4), player("C", -3),
             player("D", -3), player("E", 1), player("F", 5)]
    prov = _patched_provider(monkeypatch, comps)
    rows, rounds_done = prov.completed_round_scores("TEST", cut_size=3)
    assert rounds_done == 2
    made = {r["name"]: r["made_cut"] for r in rows}
    # top 3 and ties → -3s both survive; +1 and +5 are cut.
    assert made == {"A": 1, "B": 1, "C": 1, "D": 1, "E": 0, "F": 0}


# ──────────────────────────────────────────────
# Stale-board guard (engine)
# ──────────────────────────────────────────────

def test_board_fresh_guard(tmp_path):
    import os, time
    board = tmp_path / "matchups.csv"
    board.write_text("x")
    now = time.time()
    # board written well before the refresh manifest → stale
    os.utime(board, (now - 7200, now - 7200))
    assert engine._board_fresh(board, ref=now) is False
    # board written alongside the refresh → fresh
    os.utime(board, (now - 60, now - 60))
    assert engine._board_fresh(board, ref=now) is True
    # unknown ref (pre-tournament, no manifest) → treated as fresh
    assert engine._board_fresh(board, ref=None) is True
    # missing file → not fresh
    assert engine._board_fresh(tmp_path / "nope.csv", ref=now) is False


# ──────────────────────────────────────────────
# Tournament-only board filter (round groups excluded)
# ──────────────────────────────────────────────

def test_tournament_only_excludes_round_boards(tmp_path):
    from golf import edge as GE
    mp = tmp_path / "matchups.csv"
    mp.write_text(
        "group_id,player_a,player_b,odds_a,odds_b\n"
        "bovada-tmatch:1,Alice,Bob,1.8,2.0\n"
        "bovada-rmatch-r3:2,Cara,Dan,1.9,1.9\n")
    all_m = GE.load_matchup_odds(path=mp)
    tour = GE.load_matchup_odds(path=mp, tournament_only=True)
    assert ("Alice", "Bob") in all_m and ("Cara", "Dan") in all_m
    assert ("Alice", "Bob") in tour and ("Cara", "Dan") not in tour

    tp = tmp_path / "threeballs.csv"
    tp.write_text(
        "group_id,player_a,player_b,player_c,odds_a,odds_b,odds_c\n"
        "bovada-3ball:1,A,B,C,2.5,2.6,2.7\n"
        "bovada-3ball-r3:2,D,E,F,2.5,2.6,2.7\n")
    tour3 = GE.load_threeball_odds(path=tp, tournament_only=True)
    assert ("A", "B", "C") in tour3 and ("D", "E", "F") not in tour3


def test_board_freshness_warns_with_fix_command(tmp_path, monkeypatch):
    import os, time
    from golf import refresh
    monkeypatch.setattr(refresh, "DATA_DIR", tmp_path)
    now = time.time()
    (tmp_path / "odds.csv").write_text("x")
    (tmp_path / "matchups.csv").write_text("x")
    (tmp_path / "threeballs.csv").write_text("x")
    os.utime(tmp_path / "odds.csv", (now - 7200, now - 7200))       # stale
    os.utime(tmp_path / "matchups.csv", (now - 7200, now - 7200))   # stale
    os.utime(tmp_path / "threeballs.csv", (now - 10, now - 10))     # fresh

    checks = refresh._board_freshness_checks(now, rounds_done=2, event_id="401811954")
    flagged = {c.source for c in checks}
    assert flagged == {"freshness.odds.csv", "freshness.matchups.csv"}
    assert all(c.severity == "warning" for c in checks)
    # every warning shows the exact command to bring the board back
    assert all("python3 -m golf.refresh --event 401811954" in c.message for c in checks)

    # pre-tournament: nothing to be stale against
    assert refresh._board_freshness_checks(now, 0, "401811954") == []


def test_card_notes_show_freshness_message():
    from golf import season
    manifest = {"qa": {"errors": [], "warnings": [
        {"source": "freshness.matchups.csv",
         "message": "matchups.csv is older … Bring it back with:  python3 -m golf.refresh --event X"},
        {"source": "bovada", "message": "unrelated"},
    ]}}
    out = season._notes_section(manifest, ["in-play after R2"])
    assert "python3 -m golf.refresh --event X" in out          # command surfaced
    assert "1 other data warning(s)" in out                    # others summarised


def test_write_edge_report_clears_on_empty(tmp_path):
    from golf import edge as GE
    rep = tmp_path / "edge_report.csv"
    # A previous run left recommendations on disk.
    GE.write_edge_report([{"player": "Old Bet", "market": "Win outright",
                           "side": "win", "odds": 5.0, "p_model": 0.3,
                           "p_market": 0.2, "ev_per_unit": 0.5,
                           "stake_gbp": 2.0, "recommended": True}], path=rep)
    assert "Old Bet" in rep.read_text()
    # An empty result must clear the file, not leave the stale bet behind.
    GE.write_edge_report([], path=rep)
    text = rep.read_text()
    assert "Old Bet" not in text
    assert text.strip().splitlines() == [
        "player,market,side,odds,p_model,p_market,ev_per_unit,stake_gbp,recommended"]


# ──────────────────────────────────────────────
# Refresh snapshot writer
# ──────────────────────────────────────────────

def test_refresh_writes_and_clears(monkeypatch, tmp_path):
    from golf import refresh
    monkeypatch.setattr(refresh, "LIVE_SCORES_CSV", tmp_path / "scores_live.csv")
    monkeypatch.setattr(refresh, "LIVE_STATE_JSON", tmp_path / "live_state.json")
    monkeypatch.setattr(refresh, "PREDICTIONS_INPLAY_CSV", tmp_path / "predictions_inplay.csv")
    ev = EspnEvent(event_id="TEST", name="Test Open", start_date="2025-01-01")

    comps = [_competitor("A", [_round_line(1, "-3"), _round_line(2, "-2")]),
             _competitor("B", [_round_line(1, "-1"), _round_line(2, "E")])]
    prov = _patched_provider(monkeypatch, comps)
    rd = refresh._write_live_scores(prov, ev, use_cache=True)
    assert rd == 2
    assert refresh.LIVE_STATE_JSON.exists()
    written = list(csv.DictReader(open(refresh.LIVE_SCORES_CSV)))
    assert {r["name"] for r in written} == {"A", "B"}

    # Pre-tournament (no completed rounds) clears stale artefacts.
    refresh.PREDICTIONS_INPLAY_CSV.write_text("stale")
    pre = _patched_provider(monkeypatch, [_competitor("A", [_round_line(1, "-1", holes=3)])])
    rd2 = refresh._write_live_scores(pre, ev, use_cache=True)
    assert rd2 == 0
    assert not refresh.LIVE_STATE_JSON.exists()
    assert not refresh.PREDICTIONS_INPLAY_CSV.exists()


# ──────────────────────────────────────────────
# Engine auto-route
# ──────────────────────────────────────────────

def _write_scores(path, names, leader_score=-20, rest_score=-1):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "score", "made_cut"])
        w.writeheader()
        for i, nm in enumerate(names):
            w.writerow({"name": nm, "score": leader_score if i == 0 else rest_score,
                        "made_cut": 1})


def test_simulate_autoroutes_to_inplay(tmp_path):
    names = engine._field_names()[:24]
    scores = tmp_path / "scores.csv"
    _write_scores(scores, names)
    params = {"sims": 6000, "seed": 7, "rounds_done": 3, "scores_csv": str(scores)}

    out = engine.cmd_simulate(params)
    assert "in-play" in out["note"].lower()
    assert "score" in [c["key"] for c in out["columns"]]      # live-only column
    # Leader holding a huge lead with one round left must be the clear favourite.
    assert out["rows"][0]["name"] == names[0]
    assert out["rows"][0]["win"] > 0.5
    assert abs(sum(r["win"] for r in out["rows"]) - 1.0) < 0.02


def test_pretournament_flag_forces_pre_event(tmp_path):
    names = engine._field_names()[:24]
    scores = tmp_path / "scores.csv"
    _write_scores(scores, names)

    live = engine.cmd_simulate({"sims": 6000, "seed": 7, "rounds_done": 3,
                                "scores_csv": str(scores)})
    pre = engine.cmd_simulate({"sims": 6000, "seed": 7, "rounds_done": 3,
                               "scores_csv": str(scores), "pretournament": 1})
    assert "in-play" not in pre["note"].lower()
    # The leaderboard leader is far more likely to win in-play than pre-event.
    live_leader = next(r["win"] for r in live["rows"] if r["name"] == names[0])
    pre_leader = next(r["win"] for r in pre["rows"] if r["name"] == names[0])
    assert live_leader > pre_leader


def test_simulate_inplay_command_requires_state():
    with pytest.raises(ValueError):
        engine.cmd_simulate_inplay({"sims": 1000, "pretournament": 1})


# ──────────────────────────────────────────────
# Score-aware matchups / 3-balls
# ──────────────────────────────────────────────

def _three_survivors():
    from golf.model import Player
    return [Player(name="Alice", rating=0.0, sigma=3.0),
            Player(name="Bob", rating=0.0, sigma=3.0),
            Player(name="Cara", rating=0.0, sigma=3.0)]


def test_inplay_matchup_reflects_leaderboard():
    import numpy as np
    from golf import simulate_inplay as S
    # Equal skill, but Alice leads Bob by 5 with one round to play.
    res = S.simulate_inplay(_three_survivors(),
                            {"alice": -8.0, "bob": -3.0, "cara": -3.0},
                            rounds_done=3, n_sims=20000,
                            rng=np.random.default_rng(0),
                            matchups=[("Alice", "Bob")])
    m = res["__matchups__"][("Alice", "Bob")]
    assert abs(sum(m.values()) - 1.0) < 1e-9
    assert m["Alice"] > 0.75          # the lead, not the rating, drives this


def test_inplay_threeball_reflects_leaderboard():
    import numpy as np
    from golf import simulate_inplay as S
    res = S.simulate_inplay(_three_survivors(),
                            {"alice": -8.0, "bob": -3.0, "cara": -3.0},
                            rounds_done=3, n_sims=20000,
                            rng=np.random.default_rng(0),
                            threeballs=[("Alice", "Bob", "Cara")])
    t = res["__threeballs__"][("Alice", "Bob", "Cara")]
    assert abs(sum(t.values()) - 1.0) < 1e-9
    assert t["Alice"] > t["Bob"] and t["Alice"] > t["Cara"]


def test_inplay_drops_group_with_non_survivor():
    import numpy as np
    from golf import simulate_inplay as S
    res = S.simulate_inplay(_three_survivors()[:2],
                            {"alice": -8.0, "bob": -3.0},
                            rounds_done=3, n_sims=2000,
                            rng=np.random.default_rng(1),
                            matchups=[("Alice", "Ghost")])
    assert "__matchups__" not in res   # the bet is already decided — not priced


def test_engine_inplay_results_threads_joints(tmp_path):
    names = engine._field_names()[:6]
    scores = tmp_path / "scores.csv"
    _write_scores(scores, names)
    state = engine._live_state({"rounds_done": 3, "scores_csv": str(scores)})
    rated, _ = engine._rated_field("", False)
    import numpy as np
    results, survivors = engine._inplay_results(
        rated, state, 4000, np.random.default_rng(0),
        matchups=[(names[0], names[1])],
        threeballs=[(names[0], names[1], names[2])])
    assert (names[0], names[1]) in results["__matchups__"]
    assert (names[0], names[1], names[2]) in results["__threeballs__"]
    assert results["__cut_binds__"] is False
