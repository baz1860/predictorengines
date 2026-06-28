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
