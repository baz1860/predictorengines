#!/usr/bin/env python3
"""Club soccer match model.

Models: goals (attack/defence Poisson), elo, and ensemble (default blend).
The engine is intentionally data-file first: API fetchers update fixtures.csv,
while the model can always run from local CSV fallbacks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .competitions import get as comp_get, strength
from . import schema
from .identities import dedupe_fixtures

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
PARAMS = DATA / "model_params.json"
ENSEMBLE_WEIGHTS = DATA / "ensemble_weights.json"
_FIXTURE_CACHE: dict[tuple[str, int, int], pd.DataFrame] = {}
MAX_GOALS = 10
BASE_ELO = 1500.0
HOME_ADV_ELO = 55.0
HALF_LIFE_DAYS = 365.0
DC_RHO = -0.08
RECENT_K = 6        # matches in the shots-on-target recency window
# Per-competition home advantage + Dixon-Coles rho, each Empirical-Bayes shrunk
# toward the global value (home edge and low-score clustering both vary by
# league). K = equivalent matches of global prior; a league needs ~this many
# fixtures before its own number dominates.
HFA_SHRINK_K = 300.0
RHO_SHRINK_K = 400.0
# League-season environment estimates are deliberately more strongly shrunk
# than the existing competition diagnostics. A season has only a few hundred
# matches, and the estimate must survive early-season walk-forward windows.
LEAGUE_ENV_SHRINK_K = 200.0
LEAGUE_HFA_SHRINK_K = 200.0


def _fit_comp_rho(home_goals, away_goals, lam_h: float, lam_a: float) -> float:
    """1-D MLE of the Dixon-Coles low-score rho for one competition, holding the
    competition's mean goals (lam_h, lam_a) fixed. Only the 0-0/1-0/0-1/1-1 cells
    carry rho, so this isolates league-specific low-score clustering."""
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    n00 = int(np.sum((hg == 0) & (ag == 0)))
    n10 = int(np.sum((hg == 1) & (ag == 0)))
    n01 = int(np.sum((hg == 0) & (ag == 1)))
    n11 = int(np.sum((hg == 1) & (ag == 1)))
    best = (-np.inf, DC_RHO)
    for rho in np.arange(-0.20, 0.0001, 0.005):
        t00 = 1.0 - lam_h * lam_a * rho
        t10 = 1.0 + lam_a * rho
        t01 = 1.0 + lam_h * rho
        t11 = 1.0 - rho
        if min(t00, t10, t01, t11) <= 0:
            continue
        ll = (n00 * math.log(t00) + n10 * math.log(t10)
              + n01 * math.log(t01) + n11 * math.log(t11))
        if ll > best[0]:
            best = (ll, float(rho))
    return best[1]
# ensemble blend (chosen by held-out walk-forward search, June 2026):
# goals (actual-goal attack/def), elo, xg (long-run SoT expected goals),
# xgf (xg + recent SoT form). The model-signal sprint retuned this toward Elo:
# time-split checks showed less overfitting than the heavier SoT/form blend.
DEFAULT_ENSEMBLE_W = {"goals": 0.20, "elo": 0.40, "xg": 0.20, "xgf": 0.20, "xpress": 0.0}
ENSEMBLE_W = DEFAULT_ENSEMBLE_W
ENSEMBLE_COMPONENTS = tuple(DEFAULT_ENSEMBLE_W)


def _normalise_weights(weights: dict) -> dict[str, float]:
    vals = {k: max(0.0, float(weights.get(k, 0.0))) for k in ENSEMBLE_COMPONENTS}
    s = sum(vals.values())
    if s <= 0:
        return dict(DEFAULT_ENSEMBLE_W)
    return {k: v / s for k, v in vals.items()}


def load_ensemble_weights(path: Path = ENSEMBLE_WEIGHTS) -> dict[str, float]:
    """Champion ensemble weights, falling back to the validated hardcoded blend."""
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            weights = raw.get("weights", raw)
            if isinstance(weights, dict):
                return _normalise_weights(weights)
        except Exception:
            pass
    return dict(DEFAULT_ENSEMBLE_W)


def load_fixtures(path: Path = FIXTURES) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run club_soccer/fetch.py or add fixtures.csv.")
    stat = path.stat()
    cache_key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _FIXTURE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy(deep=True)
    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    for c in schema.FIXTURE_NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["neutral"] = pd.to_numeric(df.get("neutral", 0), errors="coerce").fillna(0).astype(int)
    clean = dedupe_fixtures(df)
    # Keep one file-stat keyed cache entry for the card and context paths,
    # which otherwise reload/reconcile the same 20k-row file once per
    # upcoming fixture. A changed fetch naturally invalidates it.
    _FIXTURE_CACHE.clear()
    _FIXTURE_CACHE[cache_key] = clean.copy(deep=True)
    return clean


def played(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["home_goals", "away_goals"]).copy()


def upcoming(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["home_goals"].isna() | df["away_goals"].isna()].copy()


def team_names(df: pd.DataFrame | None = None) -> list[str]:
    df = load_fixtures() if df is None else df
    return sorted(set(df["home"].dropna()) | set(df["away"].dropna()))


def _weights(dates: pd.Series, half_life_days: float = HALF_LIFE_DAYS) -> np.ndarray:
    anchor = dates.max()
    age = (anchor - dates).dt.days.to_numpy(dtype=float)
    return np.exp(-math.log(2) * age / half_life_days)


def _season_year(d) -> int:
    d = pd.Timestamp(d)
    return d.year if d.month >= 7 else d.year - 1


def _crossed_july1(prev, curr) -> bool:
    """True iff a July 1 boundary falls strictly between prev and curr."""
    return _season_year(prev) != _season_year(curr)


def _poisson_pmf(lam: float) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    return np.exp(-lam) * np.power(lam, g) / np.array([math.factorial(int(i)) for i in g])


def score_matrix(lam_h: float, lam_a: float, rho: float = DC_RHO) -> np.ndarray:
    lam_h, lam_a = max(0.05, float(lam_h)), max(0.05, float(lam_a))
    M = np.outer(_poisson_pmf(lam_h), _poisson_pmf(lam_a))
    M[0, 0] *= 1 - lam_h * lam_a * rho
    M[1, 0] *= 1 + lam_a * rho
    M[0, 1] *= 1 + lam_h * rho
    M[1, 1] *= 1 - rho
    return M / M.sum()


def probs_from_matrix(M: np.ndarray) -> dict[str, float]:
    total = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
    return {
        "home": float(np.tril(M, -1).sum()),
        "draw": float(np.trace(M)),
        "away": float(np.triu(M, 1).sum()),
        "over25": float(M[total > 2].sum()),
        "under25": float(M[total <= 2].sum()),
        "btts_yes": float(M[1:, 1:].sum()),
        "btts_no": float(1.0 - M[1:, 1:].sum()),
    }


def top_scorelines(M: np.ndarray, n: int = 5) -> list[dict]:
    flat = sorted(((i, j, float(M[i, j])) for i in range(M.shape[0])
                   for j in range(M.shape[1])), key=lambda x: -x[2])
    return [{"score": f"{i}-{j}", "prob": round(p, 4)} for i, j, p in flat[:n]]


def _promo_relegation_priors(df: pd.DataFrame, teams: list[str]) -> dict[str, dict]:
    """team -> {"status": "promoted"|"relegated", "prev_gf": float, "prev_ga": float}
    (previous-season goals-for/against per match, across all competitions),
    for teams whose most recent league match's Competition.tier differs from
    their last league match of the previous season. Point-in-time safe —
    uses only `df`'s own contents (whatever training window the caller
    passed), so it's safe to call from inside a walk-forward fit()."""
    from .competitions import get as comp_get

    league = df[df["type"] == "league"]
    if league.empty:
        return {}

    rate_rows = []
    for r in df.itertuples(index=False):
        rate_rows.append((r.home, r.season, r.home_goals, r.away_goals))
        rate_rows.append((r.away, r.season, r.away_goals, r.home_goals))
    rate_df = pd.DataFrame(rate_rows, columns=["team", "season", "gf", "ga"])
    rates = rate_df.groupby(["team", "season"]).agg(gf=("gf", "mean"), ga=("ga", "mean"))

    last_league_comp: dict[tuple, str] = {}
    for r in league.sort_values("date").itertuples(index=False):
        last_league_comp[(r.home, r.season)] = r.competition
        last_league_comp[(r.away, r.season)] = r.competition

    seasons_by_team: dict[str, set] = {}
    for (t, s) in rates.index:
        seasons_by_team.setdefault(t, set()).add(s)

    out: dict[str, dict] = {}
    for t in teams:
        seasons = seasons_by_team.get(t)
        if not seasons or len(seasons) < 2:
            continue
        last_season = max(seasons)
        prev_season = last_season - 1
        if prev_season not in seasons:
            continue
        comp_last = last_league_comp.get((t, last_season))
        comp_prev = last_league_comp.get((t, prev_season))
        if not comp_last or not comp_prev:
            continue
        c_last, c_prev = comp_get(comp_last), comp_get(comp_prev)
        if c_last is None or c_prev is None or c_last.tier <= 0 or c_prev.tier <= 0:
            continue
        if c_last.tier == c_prev.tier:
            continue
        status = "promoted" if c_last.tier < c_prev.tier else "relegated"
        prev_gf, prev_ga = rates.loc[(t, prev_season)]
        out[t] = {"status": status, "prev_gf": float(prev_gf), "prev_ga": float(prev_ga)}
    return out


def fit(df: pd.DataFrame | None = None, promo_prior: dict | None = None,
        season_regress_rho: float = 0.0, half_life_days: float = HALF_LIFE_DAYS,
        elo_decay_half_life_days: float | None = None,
        league_adjustments: bool = False) -> dict:
    """
    promo_prior: optional {"pi": float, "active": bool} (P4.5). When active,
    a promoted/relegated team's attack/defence shrinkage prior (see the gf/ga
    block below) is seeded from ITS OWN previous-season scoring rate scaled
    by pi (promoted: attack*pi, defence/pi; relegated: symmetric) instead of
    the league-wide global_avg. Defaults to inactive (matches model_params.json
    if not passed).
    season_regress_rho: (P4.6) when a team's consecutive matches straddle a
    July 1 boundary, its Elo regresses toward the mean Elo of the upcoming
    match's competition: elo <- (1-rho)*elo + rho*league_mean. 0.0 (default)
    = incumbent behaviour (no regression).
    half_life_days: recency half-life for the exponential match-weighting
    (P4.6 re-tunes this alongside season_regress_rho). Defaults to the
    incumbent HALF_LIFE_DAYS.
    elo_decay_half_life_days: (P4.6b) plain Elo has no time decay — a rating
    earned in one hot spell three years ago carries forward at full strength
    forever, only eroding through actual match results. When set, a team's
    Elo continuously decays toward BASE_ELO between matches: its distance
    from BASE_ELO halves every elo_decay_half_life_days of elapsed calendar
    time since that team's PREVIOUS match (any competition), applied once
    right before each of its matches. None (default) = incumbent behaviour
    (undecayed, standard sequential Elo).
    league_adjustments: when true, use shrunk league-season scoring-rate and
    home-advantage estimates for league matches. Cups and European matches
    continue to use the existing competition-strength fallback. This is an
    experiment flag so the incumbent production model remains unchanged until
    a walk-forward gate promotes it.
    """
    df = played(load_fixtures() if df is None else df).sort_values("date")
    if df.empty:
        raise ValueError("No played fixtures available to fit the model.")
    teams = sorted(set(df["home"]) | set(df["away"]))
    w = _weights(df["date"], half_life_days)
    avg_home = float(np.average(df["home_goals"], weights=w))
    avg_away = float(np.average(df["away_goals"], weights=w))
    global_avg = max(0.8, (avg_home + avg_away) / 2)

    rows = []
    for r, wt in zip(df.itertuples(index=False), w):
        comp_str = strength(r.competition)
        rows.append((r.home, "for", r.home_goals, wt, comp_str))
        rows.append((r.home, "against", r.away_goals, wt, comp_str))
        rows.append((r.away, "for", r.away_goals, wt, comp_str))
        rows.append((r.away, "against", r.home_goals, wt, comp_str))
    stats = {t: {"gf": 0.0, "ga": 0.0, "wf": 0.0, "wa": 0.0,
                 "xf": 0.0, "xa": 0.0, "wx": 0.0,
                 "xpf": 0.0, "xpa": 0.0, "wxp": 0.0}
             for t in teams}
    for t, typ, goals, wt, comp_str in rows:
        key, wk = ("gf", "wf") if typ == "for" else ("ga", "wa")
        stats[t][key] += float(goals) * wt
        stats[t][wk] += wt

    # Expected-goals signal: use BSD's real team xG wherever available and
    # retain the established SoT conversion as a fallback for older sources.
    has_sot = "home_sot" in df.columns and "away_sot" in df.columns
    if has_sot:
        sot_sum = float(np.average(df["home_sot"].fillna(0) + df["away_sot"].fillna(0), weights=w))
        goal_sum = float(np.average(df["home_goals"] + df["away_goals"], weights=w))
        conv = goal_sum / sot_sum if sot_sum > 0 else 0.0
    else:
        conv = 0.0
    has_real_xg = all(c in df.columns for c in ("home_xg", "away_xg"))
    raw_xg_mask = ((df["home_xg"].notna() & df["away_xg"].notna())
                   if has_real_xg else pd.Series(False, index=df.index))
    # Empty source values are treated as observed for backwards compatibility
    # with pre-v4 fixture files. New fetches explicitly mark BSD xG; `proxy`
    # is reserved for rows deliberately materialised from SoT conversion.
    if "xg_source" in df.columns:
        source = df["xg_source"].fillna("").astype(str).str.strip().str.lower()
        source = source.where(~source.isin({"", "nan", "none"}),
                              np.where(raw_xg_mask, "observed", "none"))
    else:
        source = pd.Series(np.where(raw_xg_mask, "observed", "none"), index=df.index)
    real_xg_mask = raw_xg_mask & source.ne("proxy")
    proxy_xg_mask = (~real_xg_mask) & df["home_sot"].notna() & df["away_sot"].notna() \
        if has_sot else pd.Series(False, index=df.index)
    real_xg_coverage = float(real_xg_mask.mean()) if len(df) else 0.0
    proxy_xg_coverage = float(proxy_xg_mask.mean()) if len(df) else 0.0
    if conv > 0 or bool(real_xg_mask.any()):
        for r, wt in zip(df.itertuples(index=False), w):
            hs, as_ = getattr(r, "home_sot", np.nan), getattr(r, "away_sot", np.nan)
            hxg, axg = getattr(r, "home_xg", np.nan), getattr(r, "away_xg", np.nan)
            xg_src = str(getattr(r, "xg_source", "")).strip().lower()
            if not pd.isna(hxg) and not pd.isna(axg) and xg_src != "proxy":
                xf_h, xf_a = float(hxg), float(axg)
            elif conv > 0 and not pd.isna(hs) and not pd.isna(as_):
                xf_h, xf_a = float(hs) * conv, float(as_) * conv
            else:
                continue
            stats[r.home]["xf"] += xf_h * wt
            stats[r.home]["xa"] += xf_a * wt
            stats[r.away]["xf"] += xf_a * wt
            stats[r.away]["xa"] += xf_h * wt
            stats[r.home]["wx"] += wt
            stats[r.away]["wx"] += wt

    # Shot-pressure xG: local free-data challenger using SoT, non-SoT shots and
    # corners. Coefficients are fit on the training slice only, clipped to
    # plausible non-negative ranges so corner/volume noise cannot dominate.
    xp_coef = {"sot": conv if conv > 0 else 0.30, "non_sot": 0.02, "corner": 0.015}
    has_pressure = all(c in df.columns for c in (
        "home_shots", "away_shots", "home_sot", "away_sot",
        "home_corners", "away_corners"))
    if has_pressure:
        X_rows, y_rows, w_rows = [], [], []
        for r, wt in zip(df.itertuples(index=False), w):
            vals = [getattr(r, c, np.nan) for c in (
                "home_shots", "away_shots", "home_sot", "away_sot",
                "home_corners", "away_corners")]
            if any(pd.isna(v) for v in vals):
                continue
            hn = max(float(r.home_shots) - float(r.home_sot), 0.0)
            an = max(float(r.away_shots) - float(r.away_sot), 0.0)
            X_rows.append([float(r.home_sot), hn, float(r.home_corners)])
            y_rows.append(float(r.home_goals)); w_rows.append(float(wt))
            X_rows.append([float(r.away_sot), an, float(r.away_corners)])
            y_rows.append(float(r.away_goals)); w_rows.append(float(wt))
        if len(X_rows) >= 200:
            X = np.asarray(X_rows, dtype=float)
            yv = np.asarray(y_rows, dtype=float)
            sw = np.sqrt(np.asarray(w_rows, dtype=float))
            try:
                coef, *_ = np.linalg.lstsq(X * sw[:, None], yv * sw, rcond=None)
                xp_coef = {
                    "sot": float(np.clip(coef[0], 0.12, 0.60)),
                    "non_sot": float(np.clip(coef[1], 0.0, 0.08)),
                    "corner": float(np.clip(coef[2], 0.0, 0.08)),
                }
            except np.linalg.LinAlgError:
                pass
        for r, wt in zip(df.itertuples(index=False), w):
            vals = [getattr(r, c, np.nan) for c in (
                "home_shots", "away_shots", "home_sot", "away_sot",
                "home_corners", "away_corners")]
            if any(pd.isna(v) for v in vals):
                continue
            hx = (xp_coef["sot"] * float(r.home_sot)
                  + xp_coef["non_sot"] * max(float(r.home_shots) - float(r.home_sot), 0.0)
                  + xp_coef["corner"] * float(r.home_corners))
            ax = (xp_coef["sot"] * float(r.away_sot)
                  + xp_coef["non_sot"] * max(float(r.away_shots) - float(r.away_sot), 0.0)
                  + xp_coef["corner"] * float(r.away_corners))
            stats[r.home]["xpf"] += hx * wt
            stats[r.home]["xpa"] += ax * wt
            stats[r.away]["xpf"] += ax * wt
            stats[r.away]["xpa"] += hx * wt
            stats[r.home]["wxp"] += wt
            stats[r.away]["wxp"] += wt

    promo_active = bool((promo_prior or {}).get("active", False))
    promo_pi = float((promo_prior or {}).get("pi", 1.0))
    promo_info = _promo_relegation_priors(df, teams) if promo_active else {}

    attack, defence, attack_xg, defence_xg = {}, {}, {}, {}
    attack_xpress, defence_xpress = {}, {}
    base_xf, base_xa = {}, {}
    for t in teams:
        info = promo_info.get(t)
        if info is not None:
            if info["status"] == "promoted":
                gf_prior = info["prev_gf"] * promo_pi
                ga_prior = info["prev_ga"] / promo_pi
            else:  # relegated: symmetric
                gf_prior = info["prev_gf"] / promo_pi
                ga_prior = info["prev_ga"] * promo_pi
        else:
            gf_prior = ga_prior = global_avg
        gf = (stats[t]["gf"] + gf_prior * 4) / (stats[t]["wf"] + 4)
        ga = (stats[t]["ga"] + ga_prior * 4) / (stats[t]["wa"] + 4)
        attack[t] = float(math.log(max(0.25, gf) / global_avg))
        defence[t] = float(math.log(max(0.25, ga) / global_avg))
        xf = (stats[t]["xf"] + global_avg * 4) / (stats[t]["wx"] + 4)
        xa = (stats[t]["xa"] + global_avg * 4) / (stats[t]["wx"] + 4)
        attack_xg[t] = float(math.log(max(0.25, xf) / global_avg))
        defence_xg[t] = float(math.log(max(0.25, xa) / global_avg))
        xpf = (stats[t]["xpf"] + global_avg * 4) / (stats[t]["wxp"] + 4)
        xpa = (stats[t]["xpa"] + global_avg * 4) / (stats[t]["wxp"] + 4)
        attack_xpress[t] = float(math.log(max(0.25, xpf) / global_avg))
        defence_xpress[t] = float(math.log(max(0.25, xpa) / global_avg))
        base_xf[t], base_xa[t] = xf, xa

    # recency form: last RECENT_K matches' xG signal vs the team's season
    # baseline, as a log-ratio attack/defence nudge (the part long-run rates
    # miss). Prefer BSD's observed xG and fall back to the SoT conversion for
    # older sources/competitions where xG is absent.
    recent = {t: [] for t in teams}
    if conv > 0:
        for r in df.sort_values("date").itertuples(index=False):
            hs, as_ = getattr(r, "home_sot", np.nan), getattr(r, "away_sot", np.nan)
            hxg, axg = getattr(r, "home_xg", np.nan), getattr(r, "away_xg", np.nan)
            xg_src = str(getattr(r, "xg_source", "")).strip().lower()
            if not pd.isna(hxg) and not pd.isna(axg) and xg_src != "proxy":
                xf_h, xf_a = float(hxg), float(axg)
            elif not pd.isna(hs) and not pd.isna(as_):
                xf_h, xf_a = float(hs) * conv, float(as_) * conv
            else:
                continue
            recent[r.home].append((xf_h, xf_a))
            recent[r.away].append((xf_a, xf_h))
    fatk, fdef = {}, {}
    for t in teams:
        last = recent[t][-RECENT_K:]
        if len(last) < 3:
            fatk[t] = fdef[t] = 0.0
            continue
        rf = float(np.mean([x[0] for x in last])); ra = float(np.mean([x[1] for x in last]))
        fatk[t] = float(np.clip(math.log(max(0.25, rf) / max(0.25, base_xf[t])), -0.4, 0.4))
        fdef[t] = float(np.clip(math.log(max(0.25, ra) / max(0.25, base_xa[t])), -0.4, 0.4))

    elo = {t: BASE_ELO for t in teams}
    if season_regress_rho > 0:
        comp_teams: dict[str, set] = {}
        for r in df.itertuples(index=False):
            comp_teams.setdefault(r.competition, set()).update((r.home, r.away))
    last_date: dict[str, pd.Timestamp] = {}
    for r in df.itertuples(index=False):
        h, a = r.home, r.away
        if elo_decay_half_life_days:
            for team in (h, a):
                prev = last_date.get(team)
                if prev is not None:
                    gap_days = (r.date - prev).days
                    if gap_days > 0:
                        decay = 0.5 ** (gap_days / elo_decay_half_life_days)
                        elo[team] = BASE_ELO + (elo[team] - BASE_ELO) * decay
        if season_regress_rho > 0:
            for team in (h, a):
                prev = last_date.get(team)
                if prev is not None and _crossed_july1(prev, r.date):
                    members = comp_teams.get(r.competition, set())
                    vals = [elo[t] for t in members if t in elo]
                    league_mean = float(np.mean(vals)) if vals else BASE_ELO
                    elo[team] = (1 - season_regress_rho) * elo[team] + season_regress_rho * league_mean
        last_date[h] = r.date
        last_date[a] = r.date
        adv = 0.0 if int(r.neutral) else HOME_ADV_ELO
        exp_h = 1.0 / (1.0 + 10 ** ((elo[a] - (elo[h] + adv)) / 400.0))
        actual_h = 1.0 if r.home_goals > r.away_goals else (0.5 if r.home_goals == r.away_goals else 0.0)
        margin = abs(float(r.home_goals) - float(r.away_goals))
        comp_k = 18 + 20 * strength(r.competition)
        k = comp_k * (1.0 if margin <= 1 else min(1.75, 1 + margin / 4))
        delta = k * (actual_h - exp_h)
        elo[h] += delta
        elo[a] -= delta

    # ── per-competition home advantage (multiplier vs global) + rho, shrunk ──
    global_hfa = float(max(0.02, avg_home - avg_away))
    comp_adj: dict[str, dict[str, float]] = {}
    for comp, grp in df.groupby("competition"):
        n_c = len(grp)
        if n_c < 30:
            continue
        ah_c = float(grp["home_goals"].mean())
        aa_c = float(grp["away_goals"].mean())
        hfa_c = ah_c - aa_c
        # EB-shrink the league HFA toward the global value, then express as a
        # multiplier so it scales every component's home term consistently.
        hfa_shr = (n_c * hfa_c + HFA_SHRINK_K * global_hfa) / (n_c + HFA_SHRINK_K)
        mult = float(np.clip(hfa_shr / global_hfa, 0.3, 2.5))
        rho_c = _fit_comp_rho(grp["home_goals"], grp["away_goals"],
                              max(0.2, ah_c), max(0.2, aa_c))
        rho_shr = float(np.clip((n_c * rho_c + RHO_SHRINK_K * DC_RHO)
                                / (n_c + RHO_SHRINK_K), -0.20, 0.0))
        comp_adj[str(comp)] = {"hfa_mult": round(mult, 4), "rho": round(rho_shr, 4),
                               "n": int(n_c)}

    # ── hierarchical league-season scoring environment (experimental) ──────
    # The environment is separate from team attack/defence. This prevents a
    # high-scoring league or season from being absorbed into every team's
    # strength, and lets a newly promoted team retain a sensible absolute
    # scoring baseline. Only league rows are included; cup/European fixtures
    # retain the existing static competition-strength treatment.
    league_env: dict[str, dict[str, float]] = {}
    league_env_by_comp: dict[str, dict[str, float]] = {}
    league_hfa: dict[str, dict[str, float]] = {}
    league_hfa_by_comp: dict[str, dict[str, float]] = {}
    if "type" in df.columns:
        league = df[df["type"].astype(str).str.lower().eq("league")].copy()
    else:
        league = df.iloc[0:0].copy()
    league_hfa_prior = global_hfa
    if not league.empty:
        # Align the recency weights to the sorted training frame's original
        # positions. This avoids accidental reindexing after groupby operations.
        weighted = df.copy()
        weighted["_wt"] = w
        weighted["_season"] = weighted["date"].map(_season_year)
        weighted = weighted[weighted["type"].astype(str).str.lower().eq("league")]
        weighted["_goals_per_match"] = weighted["home_goals"] + weighted["away_goals"]
        weighted["_hfa"] = weighted["home_goals"] - weighted["away_goals"]

        league_hfa_prior = float(np.average(weighted["_hfa"], weights=weighted["_wt"]))
        for (comp, season), grp in weighted.groupby(["competition", "_season"]):
            n_eff = float(grp["_wt"].sum())
            if n_eff <= 0:
                continue
            raw_rate = float(np.average(grp["_goals_per_match"], weights=grp["_wt"]) / 2.0)
            rate = (n_eff * raw_rate + LEAGUE_ENV_SHRINK_K * global_avg) / (n_eff + LEAGUE_ENV_SHRINK_K)
            raw_hfa = float(np.average(grp["_hfa"], weights=grp["_wt"]))
            hfa = (n_eff * raw_hfa + LEAGUE_HFA_SHRINK_K * league_hfa_prior) / (n_eff + LEAGUE_HFA_SHRINK_K)
            key = f"{comp}|{int(season)}"
            league_env[key] = {
                "rate": round(float(rate), 6),
                "mult": round(float(np.clip(rate / max(global_avg, 0.1), 0.75, 1.30)), 6),
                "n": int(len(grp)), "n_eff": round(n_eff, 3),
            }
            league_hfa[key] = {
                "hfa": round(float(np.clip(hfa, 0.05, 0.60)), 6),
                "n": int(len(grp)), "n_eff": round(n_eff, 3),
            }

        for comp, grp in weighted.groupby("competition"):
            n_eff = float(grp["_wt"].sum())
            if n_eff <= 0:
                continue
            raw_rate = float(np.average(grp["_goals_per_match"], weights=grp["_wt"]) / 2.0)
            rate = (n_eff * raw_rate + LEAGUE_ENV_SHRINK_K * global_avg) / (n_eff + LEAGUE_ENV_SHRINK_K)
            raw_hfa = float(np.average(grp["_hfa"], weights=grp["_wt"]))
            hfa = (n_eff * raw_hfa + LEAGUE_HFA_SHRINK_K * league_hfa_prior) / (n_eff + LEAGUE_HFA_SHRINK_K)
            league_env_by_comp[str(comp)] = {
                "rate": round(float(rate), 6),
                "mult": round(float(np.clip(rate / max(global_avg, 0.1), 0.75, 1.30)), 6),
                "n": int(len(grp)), "n_eff": round(n_eff, 3),
            }
            league_hfa_by_comp[str(comp)] = {
                "hfa": round(float(np.clip(hfa, 0.05, 0.60)), 6),
                "n": int(len(grp)), "n_eff": round(n_eff, 3),
            }

    params = {"teams": teams, "global_avg": global_avg,
              "home_goal_adv": global_hfa,
              "global_hfa": global_hfa, "comp_adj": comp_adj,
              "comp_adj_active": False,
              "league_adjustments_active": bool(league_adjustments),
              "league_env": league_env,
              "league_env_by_comp": league_env_by_comp,
              "league_hfa": league_hfa,
              "league_hfa_by_comp": league_hfa_by_comp,
              "league_hfa_prior": round(float(league_hfa_prior), 6),
              "attack": attack, "defence": defence,
              "attack_xg": attack_xg, "defence_xg": defence_xg,
              "attack_xpress": attack_xpress, "defence_xpress": defence_xpress,
              "fatk": fatk, "fdef": fdef, "conv": float(conv),
              "xpress_coef": xp_coef,
              "real_xg_coverage": round(real_xg_coverage, 6),
              "proxy_xg_coverage": round(proxy_xg_coverage, 6),
              "xg_source_counts": {str(k): int(v) for k, v in source.value_counts().items()},
              "elo": {k: float(v) for k, v in elo.items()},
              "fitted_matches": int(len(df))}
    return params


def save_params(params: dict, path: Path = PARAMS) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(params, indent=2))


def load_params() -> dict:
    if PARAMS.exists():
        return json.loads(PARAMS.read_text())
    params = fit()
    save_params(params)
    return params


# ── P4.4: fitted competition strength (replaces hand-set constants, gated) ────
COMP_STRENGTH = DATA / "comp_strength.json"

# Cup -> parent top-flight league, for the "0.95 x parent's fitted strength"
# rule. Domestic cups only; European competitions are priced on their own
# hand-set constant (no single "parent league").
_CUP_PARENT_LEAGUE = {
    "FA Cup": "Premier League", "EFL Cup": "Premier League",
    "Scottish Cup": "Scottish Premiership", "Scottish League Cup": "Scottish Premiership",
    "DFB-Pokal": "Bundesliga", "Coppa Italia": "Serie A",
    "Coupe de France": "Ligue 1", "Copa del Rey": "La Liga",
}


def fit_comp_strength(verbose: bool = True, save: bool = True) -> dict:
    """Fitted per-competition strength from the shared Elo scale, replacing
    the hand-set `Competition.strength` constants.

    For each LEAGUE competition: mean end-of-fit Elo of teams that played
    >= 6 matches in it during the last completed season (today's season - 1;
    Elo is shared across competitions in fit(), so cross-league cup/Europe
    matches and promoted teams already propagate information into it).
    Mapped to [0.15, 1.10] via a min-max rescale over league competitions
    only. Domestic cups take 0.95x their parent league's fitted value.

    Written to data/comp_strength.json with "active": false — inactive
    until promoted per plan Sec 12 (competitions.strength() ignores this
    file unless "active" is true).
    """
    from datetime import datetime, timezone
    from .competitions import COMPETITIONS

    df = played(load_fixtures())
    params = fit(df)
    elo = params["elo"]

    today = datetime.now(timezone.utc).date()
    current_season = today.year if today.month >= 7 else today.year - 1
    last_completed = current_season - 1

    league_elo: dict[str, float] = {}
    for comp in COMPETITIONS:
        if comp.kind != "league":
            continue
        sub = df[(df["competition"] == comp.name) & (df["season"] == last_completed)]
        if sub.empty:
            continue
        counts = pd.concat([sub["home"], sub["away"]]).value_counts()
        teams = [t for t, c in counts.items() if c >= 6]
        if not teams:
            continue
        league_elo[comp.name] = float(np.mean([elo.get(t, BASE_ELO) for t in teams]))

    if not league_elo:
        raise ValueError(
            f"No league had a team with >=6 matches in season {last_completed} "
            "— nothing to fit competition strength from.")

    min_e, max_e = min(league_elo.values()), max(league_elo.values())
    span = (max_e - min_e) or 1.0
    strengths: dict[str, float] = {}
    for name, mean_elo in league_elo.items():
        s = 0.15 + 0.85 * (mean_elo - min_e) / span
        strengths[name] = round(float(np.clip(s, 0.15, 1.10)), 4)

    for comp in COMPETITIONS:
        if comp.kind == "cup":
            parent = _CUP_PARENT_LEAGUE.get(comp.name)
            if parent and parent in strengths:
                strengths[comp.name] = round(0.95 * strengths[parent], 4)

    payload = dict(strengths)
    payload["active"] = False
    payload["_fit_season"] = last_completed
    if verbose:
        print(f"Fitted competition strength (season {last_completed} Elo, "
              f">=6-match teams):")
        for name in sorted(strengths):
            hand_set = next((c.strength for c in COMPETITIONS if c.name == name), None)
            print(f"  {name:25s} fitted={strengths[name]:.4f}  hand-set={hand_set}")
    if save:
        COMP_STRENGTH.parent.mkdir(exist_ok=True)
        COMP_STRENGTH.write_text(json.dumps(payload, indent=2))
        if verbose:
            print(f"  saved -> {COMP_STRENGTH.name} (active=false; promote per plan Sec 12)")
    return payload


# ── P4.5: promoted/relegated-team shrinkage-prior multiplier (gated) ──────────
PROMO_PI_GRID = (0.80, 0.85, 0.90, 0.95, 1.00)
PROMO_EVAL_SEASONS = (2023, 2024, 2025, 2026)
PROMO_FIRST_N_ROUNDS = 10


def tune_promo_prior(verbose: bool = True, save: bool = False) -> dict:
    """Grid-search the promotion/relegation shrinkage-prior multiplier pi
    (P4.5), on walk-forward Brier restricted to promoted/relegated teams'
    first PROMO_FIRST_N_ROUNDS league matches of seasons in
    PROMO_EVAL_SEASONS. Report-only: writes model_params.json's
    "promo_prior" key with "active": false regardless of outcome — a code
    change (mirroring comp_adj_active's pattern) is what actually promotes
    a gated term to the default path.
    """
    df_all = played(load_fixtures()).sort_values("date").reset_index(drop=True)
    df_all["_ym"] = df_all["date"].dt.to_period("M")
    months = sorted(df_all["_ym"].unique())

    def _eval(pi: float, active: bool) -> tuple[float, int]:
        brier_sum, n = 0.0, 0
        for ym in months:
            test = df_all[df_all["_ym"] == ym]
            train_cut = test["date"].min()
            train = df_all[df_all["date"] < train_cut]
            if len(train) < 200:
                continue
            try:
                params = fit(train, promo_prior={"pi": pi, "active": active})
            except Exception:
                continue
            seen = set(params["teams"])
            info = _promo_relegation_priors(train, params["teams"]) if active else \
                _promo_relegation_priors(train, list(seen))
            if not info:
                continue
            round_counts: dict[tuple, int] = {}
            for r in train[train["type"] == "league"].itertuples(index=False):
                round_counts[(r.home, r.season)] = round_counts.get((r.home, r.season), 0) + 1
                round_counts[(r.away, r.season)] = round_counts.get((r.away, r.season), 0) + 1
            for r in test.itertuples(index=False):
                if r.type != "league" or r.season not in PROMO_EVAL_SEASONS:
                    continue
                if r.home not in seen or r.away not in seen:
                    continue
                h_elig = r.home in info and round_counts.get((r.home, r.season), 0) < PROMO_FIRST_N_ROUNDS
                a_elig = r.away in info and round_counts.get((r.away, r.season), 0) < PROMO_FIRST_N_ROUNDS
                if not (h_elig or a_elig):
                    continue
                try:
                    pred = predict(r.home, r.away, r.competition, "ensemble",
                                   bool(r.neutral), params)
                except ValueError:
                    continue
                actual = 0 if r.home_goals > r.away_goals else (
                    1 if r.home_goals == r.away_goals else 2)
                p = np.array([pred["probs"]["home"], pred["probs"]["draw"], pred["probs"]["away"]])
                brier_sum += float(np.sum((p - np.eye(3)[actual]) ** 2))
                n += 1
        return (brier_sum / n if n else float("nan")), n

    baseline_brier, baseline_n = _eval(pi=1.0, active=False)
    if verbose:
        print(f"Promo-prior tuning (promoted/relegated teams' first "
              f"{PROMO_FIRST_N_ROUNDS} league rounds, seasons {PROMO_EVAL_SEASONS}):")
        print(f"  baseline (inactive, global_avg prior): n={baseline_n} brier={baseline_brier:.4f}")

    grid_results = {}
    for pi in PROMO_PI_GRID:
        brier, n = _eval(pi=pi, active=True)
        grid_results[pi] = {"brier": brier, "n": n}
        if verbose:
            print(f"  pi={pi:.2f}  n={n}  brier={brier:.4f}")

    valid = {pi: r for pi, r in grid_results.items() if r["n"] > 0}
    best_pi = min(valid, key=lambda k: valid[k]["brier"]) if valid else 1.0
    best_brier = valid[best_pi]["brier"] if valid else float("nan")
    promotes = bool(valid) and baseline_n > 0 and best_brier < baseline_brier

    payload = {"pi": best_pi, "active": False,
               "baseline_brier": round(baseline_brier, 6) if baseline_n else None,
               "best_brier": round(best_brier, 6) if valid else None,
               "would_promote": promotes,
               "grid": {f"{k:.2f}": v for k, v in grid_results.items()}}
    if verbose:
        print(f"  best pi={best_pi} vs baseline: "
              f"{'would PROMOTE' if promotes else 'reject — keep prior OFF'}")
    if save:
        params = load_params()
        params["promo_prior"] = payload
        save_params(params)
        if verbose:
            print(f"  saved diagnostic -> {PARAMS.name}['promo_prior'] (active=false)")
    return payload


# ── P4.6: season-boundary Elo regression + recency half-life re-tune (gated) ──
SEASON_REGRESS_RHO_GRID = (0.0, 0.1, 0.2, 0.3, 0.4)   # 0.0 = incumbent
HALF_LIFE_GRID = (180.0, 270.0, 365.0)                # 365 = incumbent
AUG_OCT_MONTHS = (8, 9, 10)


def tune_season_boundary(verbose: bool = True, save: bool = False) -> dict:
    """Grid-search season_regress_rho x half_life_days (P4.6) on walk-forward
    Brier restricted to August-October fixtures (where a season-boundary
    effect would live). Only refits for Aug/Sep/Oct test months — a given
    month's fit() call only depends on its own train cutoff, so skipping
    months we don't score is exact, not an approximation, and cuts the grid
    search to a fraction of a full walk-forward's fit() calls.

    Report-only: writes model_params.json's "season_regress_rho"/
    "half_life_days" keys with "active": false regardless of outcome — a
    code change to fit()'s defaults is what actually promotes this.
    """
    df_all = played(load_fixtures()).sort_values("date").reset_index(drop=True)
    df_all["_ym"] = df_all["date"].dt.to_period("M")
    months = [ym for ym in sorted(df_all["_ym"].unique()) if ym.month in AUG_OCT_MONTHS]

    def _eval(rho: float, hl: float) -> tuple[float, int]:
        brier_sum, n = 0.0, 0
        for ym in months:
            test = df_all[df_all["_ym"] == ym]
            train = df_all[df_all["date"] < test["date"].min()]
            if len(train) < 200:
                continue
            try:
                params = fit(train, season_regress_rho=rho, half_life_days=hl)
            except Exception:
                continue
            seen = set(params["teams"])
            for r in test.itertuples(index=False):
                if r.home not in seen or r.away not in seen:
                    continue
                try:
                    pred = predict(r.home, r.away, r.competition, "ensemble",
                                   bool(r.neutral), params)
                except ValueError:
                    continue
                actual = 0 if r.home_goals > r.away_goals else (
                    1 if r.home_goals == r.away_goals else 2)
                p = np.array([pred["probs"]["home"], pred["probs"]["draw"], pred["probs"]["away"]])
                brier_sum += float(np.sum((p - np.eye(3)[actual]) ** 2))
                n += 1
        return (brier_sum / n if n else float("nan")), n

    if verbose:
        print(f"Season-boundary tuning (Aug-Oct fixtures only, {len(months)} months):")

    grid: dict[tuple, dict] = {}
    for hl in HALF_LIFE_GRID:
        for rho in SEASON_REGRESS_RHO_GRID:
            brier, n = _eval(rho, hl)
            grid[(rho, hl)] = {"brier": brier, "n": n}
            if verbose:
                print(f"  rho={rho:.1f} half_life={hl:.0f}  n={n}  brier={brier:.4f}")

    incumbent = grid.get((0.0, 365.0))
    valid = {k: v for k, v in grid.items() if v["n"] > 0}
    best_key = min(valid, key=lambda k: valid[k]["brier"]) if valid else (0.0, 365.0)
    best = valid.get(best_key)
    promotes = bool(incumbent and best and best["brier"] < incumbent["brier"]
                    and best_key != (0.0, 365.0))

    payload = {
        "season_regress_rho": {"value": best_key[0], "active": False},
        "half_life_days": {"value": best_key[1], "active": False},
        "incumbent_brier": round(incumbent["brier"], 6) if incumbent else None,
        "best_brier": round(best["brier"], 6) if best else None,
        "would_promote": promotes,
        "grid": {f"rho={k[0]:.1f},hl={k[1]:.0f}": v for k, v in grid.items()},
    }
    if verbose:
        print(f"  best rho={best_key[0]}, half_life={best_key[1]} vs incumbent "
              f"(rho=0, hl=365): {'would PROMOTE' if promotes else 'reject — keep incumbent'}")
    if save:
        params = load_params()
        params["season_regress_rho"] = payload["season_regress_rho"]
        params["half_life_days"] = payload["half_life_days"]
        save_params(params)
        if verbose:
            print(f"  saved diagnostic -> {PARAMS.name} (both active=false)")
    return payload


# ── P4.6b: continuous Elo time-decay (gated) ──────────────────────────────────
# None = incumbent (undecayed). Values are days for the rating's distance from
# BASE_ELO to halve since a team's previous match — 90d is aggressive (a full
# summer break alone nearly wipes the rating), 1095d (3y) is barely-there decay.
ELO_DECAY_GRID = (None, 1095.0, 730.0, 365.0, 180.0, 90.0)


def tune_elo_decay(verbose: bool = True, save: bool = False) -> dict:
    """Grid-search elo_decay_half_life_days (P4.6b) on the FULL walk-forward
    Brier (unlike season_regress_rho, decay can matter at any point in the
    season, not just Aug-Oct, so every month has to be scored).

    Report-only: writes model_params.json's "elo_decay_half_life_days" key
    with "active": false regardless of outcome — a code change to fit()'s
    default is what actually promotes this.
    """
    df_all = played(load_fixtures()).sort_values("date").reset_index(drop=True)
    df_all["_ym"] = df_all["date"].dt.to_period("M")
    months = sorted(df_all["_ym"].unique())

    def _eval(half_life: float | None) -> tuple[float, int]:
        brier_sum, n = 0.0, 0
        for ym in months:
            test = df_all[df_all["_ym"] == ym]
            train = df_all[df_all["date"] < test["date"].min()]
            if len(train) < 200:
                continue
            try:
                params = fit(train, elo_decay_half_life_days=half_life)
            except Exception:
                continue
            seen = set(params["teams"])
            for r in test.itertuples(index=False):
                if r.home not in seen or r.away not in seen:
                    continue
                try:
                    pred = predict(r.home, r.away, r.competition, "ensemble",
                                   bool(r.neutral), params)
                except ValueError:
                    continue
                actual = 0 if r.home_goals > r.away_goals else (
                    1 if r.home_goals == r.away_goals else 2)
                p = np.array([pred["probs"]["home"], pred["probs"]["draw"], pred["probs"]["away"]])
                brier_sum += float(np.sum((p - np.eye(3)[actual]) ** 2))
                n += 1
        return (brier_sum / n if n else float("nan")), n

    if verbose:
        print(f"Elo decay tuning (full walk-forward, {len(months)} months):")

    grid: dict[float | None, dict] = {}
    for hl in ELO_DECAY_GRID:
        brier, n = _eval(hl)
        grid[hl] = {"brier": brier, "n": n}
        if verbose:
            label = "none (incumbent)" if hl is None else f"{hl:.0f}d"
            print(f"  half_life={label:18s}  n={n}  brier={brier:.4f}")

    incumbent = grid.get(None)
    valid = {k: v for k, v in grid.items() if v["n"] > 0}
    best_key = min(valid, key=lambda k: valid[k]["brier"]) if valid else None
    best = valid.get(best_key)
    promotes = bool(incumbent and best and best["brier"] < incumbent["brier"]
                    and best_key is not None)

    payload = {
        "elo_decay_half_life_days": {"value": best_key, "active": False},
        "incumbent_brier": round(incumbent["brier"], 6) if incumbent else None,
        "best_brier": round(best["brier"], 6) if best else None,
        "would_promote": promotes,
        "grid": {("none" if k is None else f"{k:.0f}d"): v for k, v in grid.items()},
    }
    if verbose:
        best_label = "none" if best_key is None else f"{best_key:.0f}d"
        print(f"  best half_life={best_label} vs incumbent (no decay): "
              f"{'would PROMOTE' if promotes else 'reject — keep incumbent'}")
    if save:
        params = load_params()
        params["elo_decay_half_life_days"] = payload["elo_decay_half_life_days"]
        save_params(params)
        if verbose:
            print(f"  saved diagnostic -> {PARAMS.name} (active=false)")
    return payload


def _home_mult(params: dict, competition: str | None) -> float:
    """Per-competition home-advantage multiplier (1.0 = global), shrunk at fit.

    Gated by `comp_adj_active` (default False): walk-forward over ~16.5k
    predictions showed per-competition HFA + rho is neutral-to-slightly-worse on
    held-out Brier (0.61207 global vs 0.61216–0.61234), so the validated global
    constants stay the default. The fitted table is still stored for inspection
    and auto-activates if a future gate flips the flag."""
    if not params.get("comp_adj_active", False):
        return 1.0
    return float(params.get("comp_adj", {}).get(str(competition), {}).get("hfa_mult", 1.0))


def _comp_rho(params: dict, competition: str | None) -> float:
    """Per-competition Dixon-Coles rho (falls back to global DC_RHO). Gated by
    `comp_adj_active` — see `_home_mult`."""
    if not params.get("comp_adj_active", False):
        return DC_RHO
    return float(params.get("comp_adj", {}).get(str(competition), {}).get("rho", DC_RHO))


def _is_league(competition: str | None) -> bool:
    comp = comp_get(competition)
    return bool(comp and comp.kind == "league")


def _season_key(competition: str | None, match_date=None) -> str | None:
    if not competition or match_date is None:
        return None
    try:
        return f"{competition}|{int(_season_year(match_date))}"
    except (TypeError, ValueError, OverflowError):
        return None


def _league_env_mult(params: dict, competition: str | None, match_date=None) -> float:
    """Return a shrunk league-season scoring-environment multiplier.

    The feature is opt-in via the fitted params flag. If a season has no
    observations yet, the competition-level hierarchy is used; if the
    competition is not a registered league, the incumbent multiplier is 1.
    """
    if not params.get("league_adjustments_active", False) or not _is_league(competition):
        return 1.0
    by_season = params.get("league_env", {})
    row = by_season.get(_season_key(competition, match_date)) if match_date is not None else None
    if row is None:
        row = params.get("league_env_by_comp", {}).get(str(competition))
    return float(np.clip((row or {}).get("mult", 1.0), 0.75, 1.30))


def _effective_hfa(params: dict, competition: str | None, match_date=None) -> float:
    """Point-in-time home-goal advantage for the requested competition."""
    base = float(params.get("home_goal_adv", 0.25))
    if not params.get("league_adjustments_active", False) or not _is_league(competition):
        return base * _home_mult(params, competition)
    by_season = params.get("league_hfa", {})
    row = by_season.get(_season_key(competition, match_date)) if match_date is not None else None
    if row is None:
        row = params.get("league_hfa_by_comp", {}).get(str(competition))
    return float(np.clip((row or {}).get("hfa", base), 0.05, 0.60))


def _effective_elo_home_adv(params: dict, competition: str | None,
                            match_date=None) -> float:
    """Map league goal HFA onto Elo's rating-point home advantage."""
    if not params.get("league_adjustments_active", False) or not _is_league(competition):
        return HOME_ADV_ELO * _home_mult(params, competition)
    base_hfa = max(float(params.get("global_hfa", params.get("home_goal_adv", 0.25))), 0.05)
    ratio = _effective_hfa(params, competition, match_date) / base_hfa
    return HOME_ADV_ELO * float(np.clip(ratio, 0.70, 1.30))


def _lambdas_goals(params: dict, home: str, away: str, competition: str | None,
                   neutral: bool, match_date=None) -> tuple[float, float]:
    base = float(params["global_avg"]) * _league_env_mult(params, competition, match_date)
    home_adv = 0.0 if neutral else _effective_hfa(params, competition, match_date) / 2
    comp_adj = 0.0 if (params.get("league_adjustments_active", False) and _is_league(competition)) \
        else (strength(competition) - 0.75) * 0.12
    ah = params["attack"].get(home, 0.0); da = params["defence"].get(away, 0.0)
    aa = params["attack"].get(away, 0.0); dh = params["defence"].get(home, 0.0)
    # defence[t] = log(goals_conceded / global_avg): POSITIVE means a team concedes
    # more (weaker D), so the opponent's expected goals must ADD it. (Was `- da`,
    # which inverted it — cancels in aggregate but mis-ranks individual matches;
    # fixing it improved walk-forward Brier 0.6317 -> 0.6175.)
    return (base * math.exp(ah + da + home_adv + comp_adj),
            base * math.exp(aa + dh - home_adv + comp_adj))


def _lambdas_elo(params: dict, home: str, away: str, neutral: bool,
                 competition: str | None = None, match_date=None) -> tuple[float, float]:
    eh = params["elo"].get(home, BASE_ELO)
    ea = params["elo"].get(away, BASE_ELO)
    home_elo = 0.0 if neutral else _effective_elo_home_adv(params, competition, match_date)
    diff = (eh + home_elo - ea) / 400.0
    total = 2.55 + 0.20 * abs(diff)
    share = 1.0 / (1.0 + math.exp(-1.2 * diff))
    return max(0.15, total * share), max(0.15, total * (1 - share))


def _lambdas_xg(params: dict, home: str, away: str, competition: str | None,
                neutral: bool, form: bool = False, match_date=None) -> tuple[float, float]:
    """SoT-based expected-goals lambdas. With form=True, add the recent-SoT
    attack/defence nudge. Falls back to the goals attack/defence maps if a
    cached params dict predates the xg fields."""
    base = float(params["global_avg"]) * _league_env_mult(params, competition, match_date)
    home_adv = 0.0 if neutral else _effective_hfa(params, competition, match_date) / 2
    comp_adj = 0.0 if (params.get("league_adjustments_active", False) and _is_league(competition)) \
        else (strength(competition) - 0.75) * 0.12
    ax = params.get("attack_xg", params["attack"])
    dx = params.get("defence_xg", params["defence"])
    ah = ax.get(home, 0.0); da = dx.get(away, 0.0)
    aa = ax.get(away, 0.0); dh = dx.get(home, 0.0)
    if form:
        fa, fd = params.get("fatk", {}), params.get("fdef", {})
        ah += fa.get(home, 0.0); da += fd.get(away, 0.0)
        aa += fa.get(away, 0.0); dh += fd.get(home, 0.0)
    return (base * math.exp(ah + da + home_adv + comp_adj),
            base * math.exp(aa + dh - home_adv + comp_adj))


def _lambdas_xpress(params: dict, home: str, away: str, competition: str | None,
                    neutral: bool, match_date=None) -> tuple[float, float]:
    """Shot-pressure lambdas from SoT, non-SoT shots and corners.

    Falls back to the existing SoT-xG maps for cached params that predate the
    shot-pressure fields.
    """
    base = float(params["global_avg"]) * _league_env_mult(params, competition, match_date)
    home_adv = 0.0 if neutral else _effective_hfa(params, competition, match_date) / 2
    comp_adj = 0.0 if (params.get("league_adjustments_active", False) and _is_league(competition)) \
        else (strength(competition) - 0.75) * 0.12
    ax = params.get("attack_xpress", params.get("attack_xg", params["attack"]))
    dx = params.get("defence_xpress", params.get("defence_xg", params["defence"]))
    ah = ax.get(home, 0.0); da = dx.get(away, 0.0)
    aa = ax.get(away, 0.0); dh = dx.get(home, 0.0)
    return (base * math.exp(ah + da + home_adv + comp_adj),
            base * math.exp(aa + dh - home_adv + comp_adj))


def component_matrices(params: dict, home: str, away: str,
                       competition: str | None, neutral: bool,
                       match_date=None) -> dict[str, np.ndarray]:
    rho = _comp_rho(params, competition)
    return {
        "goals": score_matrix(*_lambdas_goals(params, home, away, competition, neutral, match_date), rho),
        "elo": score_matrix(*_lambdas_elo(params, home, away, neutral, competition, match_date), rho),
        "xg": score_matrix(*_lambdas_xg(params, home, away, competition, neutral, match_date=match_date), rho),
        "xgf": score_matrix(*_lambdas_xg(params, home, away, competition, neutral,
                                         form=True, match_date=match_date), rho),
        "xpress": score_matrix(*_lambdas_xpress(params, home, away, competition, neutral, match_date), rho),
    }


def apply_player_adj(M: np.ndarray, player_adj: dict) -> np.ndarray:
    """Re-scale a score matrix using player availability multipliers.

    player_adj format
    -----------------
    {
      "home": {"attack_mult": float, "defense_mult": float},
      "away": {"attack_mult": float, "defense_mult": float},
    }

    attack_mult  < 1.0  → team's scoring rate reduced (key attackers out)
    defense_mult > 1.0  → opponent's scoring rate raised (key defenders out)

    The blended matrix's expected goals are extracted, multipliers applied,
    and a new score_matrix built.  This keeps Dixon-Coles low-score
    corrections intact while correctly propagating availability information
    into both 1X2 and totals/BTTS markets.
    """
    if not player_adj:
        return M
    h_adj = player_adj.get("home") or {}
    a_adj = player_adj.get("away") or {}
    att_h = float(h_adj.get("attack_mult", 1.0))
    def_h = float(h_adj.get("defense_mult", 1.0))
    att_a = float(a_adj.get("attack_mult", 1.0))
    def_a = float(a_adj.get("defense_mult", 1.0))

    # Extract current expected-goals from the blended matrix
    xg_h = float(sum(i * float(M[i, :].sum()) for i in range(M.shape[0])))
    xg_a = float(sum(j * float(M[:, j].sum()) for j in range(M.shape[1])))

    # Home team scores less if their attackers are out; more if away's D is out
    lam_h = max(0.05, xg_h * att_h * def_a)
    # Away team scores less if their attackers are out; more if home's D is out
    lam_a = max(0.05, xg_a * att_a * def_h)

    return score_matrix(lam_h, lam_a)


def apply_context_adj(M: np.ndarray, context_adj: dict) -> np.ndarray:
    """Re-scale a score matrix by a fitted context correction (rest,
    congestion, minutes-load — see context.py). Same mechanism as
    apply_player_adj: extract expected goals from the current matrix,
    multiply by each side's fitted multiplier, rebuild.

    context_adj format
    -------------------
    {"home": {"mult": float}, "away": {"mult": float}}
    """
    if not context_adj:
        return M
    mult_h = float((context_adj.get("home") or {}).get("mult", 1.0))
    mult_a = float((context_adj.get("away") or {}).get("mult", 1.0))
    xg_h = float(sum(i * float(M[i, :].sum()) for i in range(M.shape[0])))
    xg_a = float(sum(j * float(M[:, j].sum()) for j in range(M.shape[1])))
    lam_h = max(0.05, xg_h * mult_h)
    lam_a = max(0.05, xg_a * mult_a)
    return score_matrix(lam_h, lam_a)


def apply_quality_adj(M: np.ndarray, quality_adj: dict) -> np.ndarray:
    """Apply a gated point-in-time player-quality home/away shift."""
    if not quality_adj or not quality_adj.get("active"):
        return M
    shift = float(np.clip(float(quality_adj.get("shift", 0.0)), -0.20, 0.20))
    xg_h = float(sum(i * float(M[i, :].sum()) for i in range(M.shape[0])))
    xg_a = float(sum(j * float(M[:, j].sum()) for j in range(M.shape[1])))
    return score_matrix(max(0.05, xg_h * np.exp(shift)),
                        max(0.05, xg_a * np.exp(-shift)))


def predict(home: str, away: str, competition: str | None = None,
            model: str = "ensemble", neutral: bool = False,
            params: dict | None = None,
            player_adj: dict | None = None,
            context_adj: dict | None = None,
            match_date=None,
            quality_adj: dict | None = None) -> dict:
    """Predict match outcome probabilities.

    Parameters
    ----------
    home, away:    Team names (must exist in params["teams"]).
    competition:   Competition name (used for strength adjustment).
    model:         "ensemble" | "goals" | "elo" | "xg" | "xpress".
    neutral:       True for a neutral venue.
    params:        Pre-loaded model params (default: load from disk).
    player_adj:    Optional player availability multipliers from
                   club_soccer.player_features.PlayerFeatureStore.
                   Format: {"home": {"attack_mult": float, "defense_mult": float},
                             "away": {"attack_mult": float, "defense_mult": float}}
                   Values outside [0.80, 1.25] are silently clamped.
    context_adj:   Optional rest/congestion/minutes-load correction from
                   club_soccer.context (report-only/gated — see context.py).
                   Format: {"home": {"mult": float}, "away": {"mult": float}}
    quality_adj:   Optional point-in-time player-quality correction from
                   club_soccer.player_quality; inactive unless its fixed
                   walk-forward gate passes.
    match_date:     Optional fixture date used by league-season adjustments.
    """
    params = load_params() if params is None else params
    teams = set(params["teams"])
    if home not in teams:
        raise ValueError(f"Unknown team: {home!r}")
    if away not in teams:
        raise ValueError(f"Unknown team: {away!r}")
    if home == away:
        raise ValueError("Pick two different teams.")
    rho = _comp_rho(params, competition)
    if model == "ensemble":
        parts = component_matrices(params, home, away, competition, neutral, match_date)
        weights = load_ensemble_weights()
        M = sum(weights[k] * parts[k] for k in ENSEMBLE_COMPONENTS)
        M = M / M.sum()
    elif model == "goals":
        M = score_matrix(*_lambdas_goals(params, home, away, competition, neutral, match_date), rho)
    elif model == "elo":
        M = score_matrix(*_lambdas_elo(params, home, away, neutral, competition, match_date), rho)
    elif model == "xg":
        M = score_matrix(*_lambdas_xg(params, home, away, competition, neutral,
                                      form=True, match_date=match_date), rho)
    elif model == "xpress":
        M = score_matrix(*_lambdas_xpress(params, home, away, competition, neutral, match_date), rho)
    else:
        raise ValueError("Unknown model: use ensemble, goals, elo, xg, or xpress.")

    # Apply player availability adjustment (if provided)
    player_adj_applied = bool(player_adj)
    applied_player_adj = {}
    if player_adj:
        # Clamp multipliers to [0.80, 1.25] to prevent overcorrection
        for side in ("home", "away"):
            if side in player_adj:
                src = player_adj[side] or {}
                applied_player_adj[side] = {
                    **src,
                    "attack_mult": max(0.80, min(1.25, float(src.get("attack_mult", 1.0)))),
                    "defense_mult": max(0.80, min(1.25, float(src.get("defense_mult", 1.0)))),
                }
        M = apply_player_adj(M, applied_player_adj)

    if quality_adj:
        M = apply_quality_adj(M, quality_adj)

    if context_adj:
        M = apply_context_adj(M, context_adj)

    probs = probs_from_matrix(M)
    xg_h = float(sum(i * M[i, :].sum() for i in range(M.shape[0])))
    xg_a = float(sum(j * M[:, j].sum() for j in range(M.shape[1])))
    out = {"home": home, "away": away, "competition": competition or "",
           "model": model, "xg_home": round(xg_h, 2), "xg_away": round(xg_a, 2),
           "probs": {k: round(v, 4) for k, v in probs.items()},
           "scorelines": top_scorelines(M), "matrix": M}
    if player_adj_applied:
        out["player_adj"] = {
            side: {k: v for k, v in applied_player_adj.get(side, {}).items()
                   if k in ("attack_mult", "defense_mult", "n_missing")}
            for side in ("home", "away")
        }
    if context_adj:
        out["context_adj"] = {
            side: round(float((context_adj.get(side) or {}).get("mult", 1.0)), 4)
            for side in ("home", "away")
        }
    if quality_adj and quality_adj.get("active"):
        out["quality_adj"] = {
            "shift": round(float(quality_adj.get("shift", 0.0)), 4),
            "coverage": round(float(quality_adj.get("coverage", 0.0)), 4),
        }
    return out


def predict_match(home: str, away: str, competition: str | None,
                  match_date: str, model: str = "ensemble",
                  neutral: bool = False, params: dict | None = None,
                  player_adj: dict | None = None, fixture_id=None,
                  apply_context: bool = True,
                  quality_adj: dict | None = None) -> dict:
    """Point-in-time prediction wrapper used by cards and edge pricing.

    This keeps the live paths aligned: if a context coefficient is promoted,
    rest/congestion/motivation/weather/minutes corrections are applied to both
    the displayed prediction and the priced prediction. With inactive context
    coefficients this is exactly the legacy model plus any supplied player
    availability adjustment.
    """
    context_adj = {}
    if apply_context and match_date:
        from . import context as CTX
        comp = CTX.comp_get(competition) if competition else None
        context_adj = CTX.context_adj_for_match(
            home, away, str(match_date),
            is_domestic=bool(comp and comp.kind in ("league", "cup")),
            competition=competition,
            season=_season_year(match_date),
            is_cup=bool(comp and comp.kind == "cup"),
            fixture_id=fixture_id,
        )
    return predict(home, away, competition, model, neutral, params=params,
                   player_adj=player_adj, context_adj=context_adj,
                   match_date=match_date, quality_adj=quality_adj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("home", nargs="?")
    ap.add_argument("away", nargs="?")
    ap.add_argument("--competition", default="")
    ap.add_argument("--model", choices=["ensemble", "goals", "elo", "xg", "xpress"], default="ensemble")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--fit-comp-strength", action="store_true",
                    help="fit competition strength from shared Elo (P4.4); "
                         "writes data/comp_strength.json, inactive by default")
    ap.add_argument("--tune-promo-prior", action="store_true",
                    help="grid-search the promoted/relegated-team shrinkage "
                         "prior pi (P4.5), report-only")
    ap.add_argument("--tune-season-boundary", action="store_true",
                    help="grid-search season_regress_rho x half_life_days "
                         "on Aug-Oct fixtures (P4.6), report-only")
    ap.add_argument("--tune-elo-decay", action="store_true",
                    help="grid-search continuous Elo time-decay half-life "
                         "(P4.6b) on the full walk-forward, report-only")
    ap.add_argument("--write", action="store_true",
                    help="with --tune-promo-prior/--tune-season-boundary/"
                         "--tune-elo-decay, save the diagnostic to "
                         "model_params.json (still inactive)")
    args = ap.parse_args()
    if args.fit:
        params = fit()
        save_params(params)
        print(f"Saved {len(params['teams'])} teams from {params['fitted_matches']} matches -> {PARAMS}")
        return
    if args.tune_season_boundary:
        tune_season_boundary(save=args.write)
        return
    if args.tune_elo_decay:
        tune_elo_decay(save=args.write)
        return
    if args.fit_comp_strength:
        fit_comp_strength()
        return
    if args.tune_promo_prior:
        tune_promo_prior(save=args.write)
        return
    if not args.home or not args.away:
        ap.print_help()
        return
    out = predict(args.home, args.away, args.competition, args.model, args.neutral)
    print(f"{args.home} vs {args.away} ({args.competition or 'club soccer'})")
    print(f"Expected goals: {out['xg_home']:.2f} - {out['xg_away']:.2f}")
    p = out["probs"]
    print(f"Home {p['home']:.1%}  Draw {p['draw']:.1%}  Away {p['away']:.1%}")
    print(f"Over 2.5 {p['over25']:.1%}  BTTS {p['btts_yes']:.1%}")
    for s in out["scorelines"]:
        print(f"  {s['score']} {s['prob']:.1%}")


if __name__ == "__main__":
    main()
