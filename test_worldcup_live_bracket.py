"""Regression tests for live World Cup bracket locking."""

import numpy as np

from engines.worldcup.simulate import (
    QF,
    load_group_matches,
    load_live_bracket,
    simulate_once,
)


class FirstTeamModel:
    """Deterministic stand-in for unresolved matches."""

    def sample(self, *_args):
        return 0, 0

    def knockout_winner(self, team1, _team2, _rng):
        return team1


def test_completed_knockout_results_are_locked():
    group_matches = load_group_matches()
    live = load_live_bracket(group_matches)

    assert len(live.slot_team) == 32
    assert len(live.locked_winners) >= 28
    assert live.locked_winners["M74"] == "Paraguay"  # Germany shootout
    assert live.locked_winners["M75"] == "Morocco"   # Netherlands shootout
    assert live.locked_winners["M88"] == "Egypt"     # Australia shootout
    assert live.locked_winners["M96"] == "Switzerland"  # Colombia shootout

    result = simulate_once(
        FirstTeamModel(), group_matches, np.random.default_rng(7), live)
    _groups, _r32, _r16, qf, sf, finalists, champion = result

    # The current semifinalists are the only teams still alive.  The model may
    # choose either unresolved semifinal winner, but never an eliminated team.
    active = {live.locked_winners[match] for match, _, _ in QF}
    assert set(sf) == active
    assert set(finalists) <= active
    assert champion in active
    assert champion not in live.eliminated
