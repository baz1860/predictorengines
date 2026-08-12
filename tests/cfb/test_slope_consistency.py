"""The validation/backtest Elo→points slope must match production's fit.

Regression test for the FCS-ledger contamination: walk-forward validation,
the ATS backtest, and blend_eval all fit their own slope. Before elo.fit_slope
existed they omitted the FBS mask, so FCS-vs-FCS rows (whose history diffs are
FCS-ledger values anchored at 850) biased the slope ~+2.4% relative to the
model production actually ships.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfb import elo as E


def _synthetic_games():
    """A frame with FBS-vs-FBS, FBS-vs-FCS, and FCS-vs-FCS rows."""
    rows = []
    day = pd.Timestamp("2020-09-05")
    teams_fbs = ["A", "B", "C", "D"]
    teams_fcs = ["x", "y", "z", "w"]
    rng = np.random.default_rng(7)
    for season in (2020, 2021, 2022):
        for week in range(1, 9):
            date = day + pd.Timedelta(days=(season - 2020) * 365 + week * 7)
            f = rng.permutation(teams_fbs)
            s = rng.permutation(teams_fcs)
            rows.append(dict(season=season, week=week, season_type="regular",
                             date=date, neutral=False,
                             home_team=f[0], home_div="fbs",
                             away_team=f[1], away_div="fbs",
                             home_points=int(rng.integers(10, 50)),
                             away_points=int(rng.integers(3, 40))))
            rows.append(dict(season=season, week=week, season_type="regular",
                             date=date, neutral=False,
                             home_team=f[2], home_div="fbs",
                             away_team=s[0], away_div="fcs",
                             home_points=int(rng.integers(20, 60)),
                             away_points=int(rng.integers(0, 30))))
            rows.append(dict(season=season, week=week, season_type="regular",
                             date=date, neutral=False,
                             home_team=s[1], home_div="fcs",
                             away_team=s[2], away_div="fcs",
                             home_points=int(rng.integers(10, 45)),
                             away_points=int(rng.integers(3, 40))))
    g = pd.DataFrame(rows)
    g["home"] = g["home_team"].where(g["home_div"] == "fbs", E.FCS)
    g["away"] = g["away_team"].where(g["away_div"] == "fbs", E.FCS)
    return g


def test_fit_slope_excludes_fcs_only_rows():
    games = _synthetic_games()
    _, history = E.run_elo(games, record_pregame=True)
    mask = (games["season"] < 2022).values
    slope = E.fit_slope(games, history, mask)

    diffs = np.array([h[2] for h in history])
    margins = (games["home_points"] - games["away_points"]).values
    fbs = ((games["home_div"] == "fbs") | (games["away_div"] == "fbs")).values
    m = mask & fbs
    expected = float((diffs[m] * margins[m]).sum() / (diffs[m] ** 2).sum())
    assert slope == pytest.approx(expected)

    # And the contaminated (unmasked) fit is genuinely different here, so this
    # test would catch a regression to the old behaviour.
    contaminated = float((diffs[mask] * margins[mask]).sum()
                         / (diffs[mask] ** 2).sum())
    assert slope != pytest.approx(contaminated)


def test_fit_spread_map_uses_shared_slope():
    games = _synthetic_games()
    _, history = E.run_elo(games, record_pregame=True)
    slope, sigma = E.fit_spread_map(games, history, since=2020)
    assert slope == pytest.approx(
        E.fit_slope(games, history, (games["season"] >= 2020).values))
    assert sigma > 0


def test_walk_forward_slope_matches_production(monkeypatch):
    """validate.walk_forward must fit the identical slope fit_spread_map fits
    on the same window (pre-`since` seasons, champion-ledger rows only)."""
    from cfb import validate as V

    games = _synthetic_games()
    captured = {}
    real_fit_slope = E.fit_slope

    def spy(g, history, mask):
        value = real_fit_slope(g, history, mask)
        captured.setdefault("slopes", []).append(value)
        return value

    monkeypatch.setattr(E, "fit_slope", spy)
    monkeypatch.setattr(E, "season_priors", lambda: (None, {}))
    try:
        V.walk_forward(games, since=2022, quiet=True)
    except (SystemExit, ValueError):
        pass  # tiny synthetic frame may not survive P.fit; the slope ran first
    assert captured.get("slopes"), "walk_forward no longer uses elo.fit_slope"
    _, history = E.run_elo(games, record_pregame=True)
    assert captured["slopes"][0] == pytest.approx(
        real_fit_slope(games, history, (games["season"] < 2022).values))
