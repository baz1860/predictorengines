#!/usr/bin/env python3
"""Tests for the E1 hierarchical pooled component.

Two obligations. The component sits inside the production fitter behind a
weight of 0.0, so its OFF state must be provably identical to what shipped
before it existed. And its whole claim is that shrinkage fitted from the data
beats the incumbent's hardcoded pseudo-count of 4, so the shrinkage behaviour
itself is tested against cases with a known right answer rather than trusted
because the walk-forward liked it.

Offline: every test builds its own synthetic league.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from club_soccer import model as M

# Real registered competitions. `_primary_league_map` skips anything
# `competitions.get` does not know, so an invented league name would put every
# club in one unnamed bucket and quietly disable the pooling this file tests.
LEAGUE = "Premier League"
STRONG = "Premier League"
WEAK = "Scottish League Two"


def _league(teams: list[str], rounds: int, competition: str,
            seed: int, home_goals: float = 1.4, away_goals: float = 1.1,
            start: str = "2025-01-04", fid: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    d = pd.Timestamp(start)
    order = list(teams)
    for _ in range(rounds):
        for h, a in zip(order[::2], order[1::2]):
            fid += 1
            rows.append(dict(fixture_id=f"{competition}-{fid}", date=d,
                             season=2024, competition=competition,
                             type="league", neutral=0, home=h, away=a,
                             status="FIN",
                             home_goals=int(rng.poisson(home_goals)),
                             away_goals=int(rng.poisson(away_goals)),
                             home_sot=np.nan, away_sot=np.nan, xg_source=""))
        order = order[1:] + order[:1]
        d += pd.Timedelta(days=7)
    return rows


def _frame(**kw) -> pd.DataFrame:
    return pd.DataFrame(_league(**kw))


def _key(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


# ── the promoted state must stay self-consistent ────────────────────────────

def test_promoted_state_is_self_consistent():
    """E1 promoted 2026-08-16. The flag and the weight must agree.

    They are two halves of one switch: `HIERARCHICAL_DEFAULT` decides whether
    the component is FITTED, the weight decides whether it is USED. A True
    flag with weight 0 burns the fit time for nothing; a False flag with
    weight > 0 prices every match off a component that was never fitted, and
    `_lambdas_pooled` would silently serve the incumbent goals lambdas under
    the pooled name. Either mismatch is a bug that produces plausible numbers.
    """
    assert M.HIERARCHICAL_DEFAULT is True
    assert M.DEFAULT_ENSEMBLE_W["pooled"] > 0.0
    assert M.HIERARCHICAL_DEFAULT == (M.DEFAULT_ENSEMBLE_W["pooled"] > 0.0)
    assert sum(M.DEFAULT_ENSEMBLE_W.values()) == pytest.approx(1.0)


def test_promoted_blend_is_the_one_the_evidence_measured():
    """Ship exactly what was validated.

    The A/B scored `HIERARCHICAL_CANDIDATE_W`. If the production blend drifts
    away from it, the evidence file no longer describes the running model and
    the promotion comment in model.py becomes a claim about something else.
    """
    from club_soccer import validate as V
    assert M.DEFAULT_ENSEMBLE_W == V.HIERARCHICAL_CANDIDATE_W


def test_the_flag_controls_fitting_in_both_directions():
    """`hierarchical=False` must still reproduce the pre-promotion fit.

    That is what lets the A/B re-run its incumbent arm, and what the evidence
    file's reproducing command depends on.
    """
    fx = _frame(teams=[f"C{i}" for i in "ABCDEF"], rounds=10,
                competition=LEAGUE, seed=1)

    off = M.fit(fx, hierarchical=False)
    assert off["pooled"] is None
    assert off["hierarchical_active"] is False

    on = M.fit(fx)                       # production default, now promoted
    assert on["pooled"] is not None
    assert on["hierarchical_active"] is True


def test_a_zero_weight_component_contributes_exactly_nothing():
    """The mechanism that made gating safe, kept after promotion.

    This is what let E1 sit in the production ensemble through its whole
    evaluation without moving a price, and it is what any future component
    will rely on. Equality, not tolerance: `0.0 * matrix` is exactly zero and
    `x + 0.0` is exactly x.
    """
    fx = _frame(teams=[f"C{i}" for i in "ABCDEF"], rounds=12,
                competition=LEAGUE, seed=2)
    params = M.fit(fx, hierarchical=True)
    assert params["pooled"] is not None, "fixture should produce a pooled fit"

    zero_weighted = M.predict("CA", "CB", params=params,
                              ensemble_weights={"goals": 0.20, "elo": 0.40,
                                                "xg": 0.40, "pooled": 0.0})
    absent = M.predict("CA", "CB", params=params,
                       ensemble_weights={"goals": 0.20, "elo": 0.40,
                                         "xg": 0.40})
    assert _key(zero_weighted["probs"]) == _key(absent["probs"])


def test_fitting_the_component_does_not_disturb_the_incumbent_model():
    """Turning E1 on adds parameters; it must not change existing ones."""
    fx = _frame(teams=[f"C{i}" for i in "ABCDEF"], rounds=12,
                competition=LEAGUE, seed=3)
    off = M.fit(fx)
    on = M.fit(fx, hierarchical=True)
    for key in ("attack", "defence", "elo", "attack_xg", "defence_xg",
                "global_avg", "home_goal_adv"):
        assert _key(off[key]) == _key(on[key]), f"{key} moved"


def test_lambdas_fall_back_when_the_pooled_block_is_absent():
    """A params file written before E1 must not raise.

    `model_params.json` on disk predates this component, and the app loads it
    directly. The fallback keeps that path alive; weight 0.0 makes it free.
    """
    fx = _frame(teams=[f"C{i}" for i in "ABCDEF"], rounds=8,
                competition=LEAGUE, seed=4)
    params = M.fit(fx, hierarchical=False)          # stands in for an old file
    assert params.get("pooled") is None
    pooled = M._lambdas_pooled(params, "CA", "CB", LEAGUE, False)
    goals = M._lambdas_goals(params, "CA", "CB", LEAGUE, False)
    assert pooled == goals


# ── the shrinkage claim ─────────────────────────────────────────────────────

def _mixed_frame() -> pd.DataFrame:
    """A league of regulars plus one club with almost no history.

    The newcomer wins its handful of matches lopsidedly. An unpooled estimate
    reads that as "best attack in the league"; a pooled one should treat four
    matches as four matches.
    """
    regulars = [f"R{i}" for i in range(6)]
    rows = _league(teams=regulars, rounds=14, competition=LEAGUE,
                   seed=5)
    d = pd.Timestamp("2025-06-01")
    for i in range(2):
        for opp in (regulars[0], regulars[1]):
            rows.append(dict(fixture_id=f"new-{i}-{opp}", date=d, season=2024,
                             competition=LEAGUE, type="league",
                             neutral=0, home="NEWCOMER", away=opp,
                             status="FIN", home_goals=6, away_goals=0,
                             home_sot=np.nan, away_sot=np.nan, xg_source=""))
            d += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_thin_history_club_is_shrunk_toward_its_league():
    """The core E1 claim, on a club whose raw rate is absurd.

    NEWCOMER scored 6 a game across 4 matches. Its pooled attack rating must
    land far below what that rate implies, because 4 matches is not enough
    evidence to believe it — that is what partial pooling is for.
    """
    fx = _mixed_frame()
    params = M.fit(fx, hierarchical=True)
    pooled = params["pooled"]
    assert pooled is not None

    league_mean = pooled["mu_attack"][pooled["team_league"]["NEWCOMER"]]
    newcomer = pooled["attack"]["NEWCOMER"]
    # What an unpooled log-rate estimate would have said.
    raw = np.log(6.0 / pooled["global_avg"])

    assert newcomer > league_mean, "should still read as above average"
    assert newcomer < raw * 0.75, (
        f"pooled attack {newcomer:.3f} is not meaningfully shrunk from the "
        f"raw {raw:.3f}")


def test_data_rich_club_is_barely_shrunk():
    """Pooling must be evidence-weighted, not a blanket haircut.

    A club with a full season of matches should keep almost all of its
    signal — otherwise the component is just flattening everything, which
    is the retired `variance_inflation` mistake in another costume.
    """
    regulars = [f"R{i}" for i in range(6)]
    rows = _league(teams=regulars, rounds=30, competition=LEAGUE,
                   seed=6)
    # R0 is genuinely strong, over many matches.
    for row in rows:
        if row["home"] == "R0":
            row["home_goals"] += 2
        elif row["away"] == "R0":
            row["away_goals"] += 2
    params = M.fit(pd.DataFrame(rows), hierarchical=True)
    pooled = params["pooled"]
    league_mean = pooled["mu_attack"][pooled["team_league"]["R0"]]
    assert pooled["attack"]["R0"] - league_mean > 0.25, (
        "a club with 30 rounds of evidence was shrunk to the league mean")


def test_sigma_is_fitted_not_assumed():
    """The whole point: the pooling strength comes from the data.

    If sigma came back exactly at its initial value, EM never ran and the
    component would be the incumbent's hardcoded shrinkage with extra steps.
    """
    fx = _mixed_frame()
    pooled = M.fit(fx, hierarchical=True)["pooled"]
    for key in ("sigma_attack", "sigma_defence"):
        assert pooled[key] > 0.0
        assert pooled[key] != pytest.approx(M.POOLED_SIGMA_INIT, abs=1e-9)


def test_fit_is_deterministic():
    fx = _mixed_frame()
    assert _key(M.fit(fx, hierarchical=True)["pooled"]) == \
        _key(M.fit(fx, hierarchical=True)["pooled"])


def test_unseen_club_prices_off_its_league_mean():
    """A club with no rating at all is the limiting case of pooling."""
    fx = _mixed_frame()
    params = M.fit(fx, hierarchical=True)
    pooled = params["pooled"]
    pooled["team_league"]["GHOST"] = LEAGUE
    lam_h, _lam_a = M._lambdas_pooled(params, "GHOST", "R0",
                                      LEAGUE, True)
    expected = pooled["global_avg"] * np.exp(
        pooled["mu_attack"][LEAGUE] + pooled["defence"]["R0"])
    assert lam_h == pytest.approx(expected, rel=1e-9)


def test_cross_league_fixture_uses_each_side_own_league():
    """Two leagues of different scoring levels, met in a cup.

    The league means must separate them, which is what lets the component drop
    the hardcoded `(strength - 0.75) * 0.12` competition bump.
    """
    strong = _league(teams=[f"S{i}" for i in range(4)], rounds=14,
                     competition=STRONG, seed=7,
                     home_goals=2.6, away_goals=2.2)
    weak = _league(teams=[f"W{i}" for i in range(4)], rounds=14,
                   competition=WEAK, seed=8,
                   home_goals=0.9, away_goals=0.7, fid=500)
    params = M.fit(pd.DataFrame(strong + weak), hierarchical=True)
    pooled = params["pooled"]
    assert (pooled["mu_attack"][STRONG]
            > pooled["mu_attack"][WEAK]), \
        "league means did not separate two leagues with very different rates"


# ── the A/B harness must not be able to fool itself ─────────────────────────

def test_walk_forward_cache_key_separates_the_arms():
    """Both new options must reach the cache key.

    `walkforward_cache` warns that an option missing from the key serves one
    arm's result for the other — a silent wrong answer. The A/B varies the fit
    flag AND the blend, so both have to be in there.
    """
    from club_soccer import validate as V
    import inspect

    src = inspect.getsource(V.walk_forward)
    key_block = src[src.index("cache_opts = {"):src.index("row_hash =")]
    assert '"hierarchical"' in key_block
    assert '"ensemble_weights"' in key_block
    # And resolved to concrete values, never None — a None key would collide
    # two different models the moment a production default moved.
    assert "M.HIERARCHICAL_DEFAULT if hierarchical is None" in src


def test_candidate_blend_is_a_clean_substitution():
    """The A/B moves weight between the two goals models only.

    If elo or xg moved at the same time, a metric change could not be
    attributed to the pooled component.
    """
    from club_soccer import validate as V

    base, cand = M.DEFAULT_ENSEMBLE_W, V.HIERARCHICAL_CANDIDATE_W
    assert cand["elo"] == base["elo"]
    assert cand["xg"] == base["xg"]
    assert cand["goals"] + cand["pooled"] == pytest.approx(
        base["goals"] + base["pooled"])
    assert sum(cand.values()) == pytest.approx(1.0)
