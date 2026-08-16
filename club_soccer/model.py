#!/usr/bin/env python3
"""Club soccer match model.

Models: goals (attack/defence Poisson), elo, and ensemble (default blend).
The engine is intentionally data-file first: API fetchers update fixtures.csv,
while the model can always run from local CSV fallbacks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd

from .competitions import get as comp_get, strength
from . import schema
from . import coverage as _COV
from .identities import dedupe_fixtures

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
PARAMS = DATA / "model_params.json"
_FIXTURE_CACHE: dict[tuple[str, int, int], pd.DataFrame] = {}
MAX_GOALS = 10
_GOAL_GRID = np.arange(MAX_GOALS + 1, dtype=float)
_GOAL_FACTORIALS = np.array(
    [math.factorial(i) for i in range(MAX_GOALS + 1)], dtype=float
)
BASE_ELO = 1500.0
# P4b league seeding: strength points -> Elo points, and the strength value
# that maps to BASE_ELO. Calibrated on the observed Premier League / Scottish
# Premiership Elo gap (~290 pts) across their strength gap (1.00 - 0.58).
ELO_PER_STRENGTH = 690.0
LEAGUE_SEED_ANCHOR = 0.62

# ── PROMOTED 2026-07-22 ───────────────────────────────────────────────────
# League-prior seeding is ON in production. Promotion is recorded here, in the
# code, rather than toggled at runtime — the same discipline as the evidence
# gate and the market blend.
#
# Evidence (data/league_seed_evidence.json), walk-forward, identical folds
# in both arms:
#     2025-07 ->  n=11,261   Brier 0.61799 -> 0.61600   LL 1.02952 -> 1.02675
#     2024-07 ->  n=21,596   Brier 0.61730 -> 0.61521   LL 1.02865 -> 1.02572
#
# Rationale: `elo = {t: BASE_ELO ...}` seeded every club at the pooled mean of
# whatever was loaded — after P3 that is 25 leagues from the Premier League to
# the National League. An under-measured club was therefore rated "average of
# everything", which is how Sturm Graz sat at 1505 before its domestic league
# existed in the dataset.
#
# THIS IS THE SINGLE SOURCE OF TRUTH. validate.walk_forward reads it too, so
# validation always measures the model production actually runs. Flipping one
# without the other would validate a model nobody is using.
LEAGUE_SEED_DEFAULT = True
# Promoted 2026-07-25 after an identical-cutoff walk-forward A/B on 27,363
# deduplicated matches. Opponent adjustment improved 1X2 Brier
# 0.61300 -> 0.61177, log-loss 1.02233 -> 1.02045, OU2.5 Brier
# 0.25188 -> 0.24724, and BTTS Brier 0.25138 -> 0.24812.
OPPONENT_ADJUSTED_XG_DEFAULT = True
HOME_ADV_ELO = 55.0
HALF_LIFE_DAYS = 365.0
DC_RHO = -0.08
RECENT_K = 6        # matches in the shots-on-target recency window
XG_RATING_PRIOR = 8.0
XG_RATING_ITERATIONS = 12


# Ensemble blend (chosen by held-out walk-forward search, June 2026).
# The former xg/xgf entries were two views of the same shot signal. They are
# now one xg component containing their 50/50 matrix mixture. Giving that
# component weight 0.40 is algebraically identical to the old 0.20 + 0.20
# blend, so this removes a fake degree of freedom without changing a price.
DEFAULT_ENSEMBLE_W = {"goals": 0.20, "elo": 0.40, "xg": 0.40}
ENSEMBLE_W = DEFAULT_ENSEMBLE_W
ENSEMBLE_COMPONENTS = tuple(DEFAULT_ENSEMBLE_W)


def _normalise_weights(weights: dict) -> dict[str, float]:
    vals = {k: max(0.0, float(weights.get(k, 0.0))) for k in ENSEMBLE_COMPONENTS}
    s = sum(vals.values())
    if s <= 0:
        return dict(DEFAULT_ENSEMBLE_W)
    return {k: v / s for k, v in vals.items()}


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
    """TRAINING-eligible fixtures: both scores present AND the status is
    neither void nor administrative. Score presence alone is not enough — a
    postponed fixture that (incorrectly) retained its old score must never
    train the model, and an awarded 3-0 is a legal result for SETTLEMENT
    (the adapter grades from raw scores) but not evidence about how either
    team scores goals."""
    out = df.dropna(subset=["home_goals", "away_goals"])
    if "status" in out.columns:
        from .schema import TRAINING_EXCLUDED_STATUSES
        status = out["status"].map(schema.normalize_status)
        out = out[~status.isin(TRAINING_EXCLUDED_STATUSES)]
    return out.copy()


def upcoming(df: pd.DataFrame) -> pd.DataFrame:
    """Schedulable future fixtures: score-less AND not void. A cancelled or
    postponed fixture must not sit in `upcoming` forever (it would keep being
    priced); if a postponed match is rescheduled, the provider sends a fresh
    row with a new status/date."""
    out = df[df["home_goals"].isna() | df["away_goals"].isna()]
    if "status" in out.columns:
        from .schema import VOID_STATUSES, QUARANTINE_STATUSES
        status = out["status"].map(schema.normalize_status)
        # Quarantined rows are excluded too: an unrecognised status might mean
        # "abandoned" for all we know, so it must not be priced or staked.
        out = out[~status.isin(VOID_STATUSES | QUARANTINE_STATUSES)]
    return out.copy()


def team_names(df: pd.DataFrame | None = None) -> list[str]:
    df = load_fixtures() if df is None else df
    return sorted(set(df["home"].dropna()) | set(df["away"].dropna()))


def _weights(dates: pd.Series, half_life_days: float = HALF_LIFE_DAYS) -> np.ndarray:
    anchor = dates.max()
    age = (anchor - dates).dt.days.to_numpy(dtype=float)
    return np.exp(-math.log(2) * age / half_life_days)


def _opponent_adjusted_xg_ratings(
        teams: list[str],
        observations: list[tuple[str, str, float, float, float, float, float]],
        global_avg: float) -> tuple[dict[str, float], dict[str, float]]:
    """Fit attack/defence xG effects while controlling for the opponent.

    Each observation is ``(home, away, home_xg, away_xg, weight,
    home_offset, away_offset)``. Offsets remove the known home and competition
    effects. Alternating updates then estimate each attack against the
    defences it faced, and each defence against the attacks it faced. The
    global prior anchors sparse clubs and resolves the attack/defence scale
    ambiguity without a heavyweight optimiser.
    """
    attack = {team: 0.0 for team in teams}
    defence = {team: 0.0 for team in teams}
    if not observations:
        return attack, defence

    for _ in range(XG_RATING_ITERATIONS):
        attack_sum = {team: 0.0 for team in teams}
        attack_w = {team: 0.0 for team in teams}
        for home, away, hxg, axg, wt, home_offset, away_offset in observations:
            attack_sum[home] += wt * hxg / math.exp(defence[away] + home_offset)
            attack_w[home] += wt
            attack_sum[away] += wt * axg / math.exp(defence[home] + away_offset)
            attack_w[away] += wt
        attack = {
            team: float(np.clip(math.log(max(
                0.25,
                (attack_sum[team] + XG_RATING_PRIOR * global_avg)
                / (attack_w[team] + XG_RATING_PRIOR),
            ) / global_avg), -1.2, 1.2))
            for team in teams
        }

        defence_sum = {team: 0.0 for team in teams}
        defence_w = {team: 0.0 for team in teams}
        for home, away, hxg, axg, wt, home_offset, away_offset in observations:
            defence_sum[away] += wt * hxg / math.exp(attack[home] + home_offset)
            defence_w[away] += wt
            defence_sum[home] += wt * axg / math.exp(attack[away] + away_offset)
            defence_w[home] += wt
        defence = {
            team: float(np.clip(math.log(max(
                0.25,
                (defence_sum[team] + XG_RATING_PRIOR * global_avg)
                / (defence_w[team] + XG_RATING_PRIOR),
            ) / global_avg), -1.2, 1.2))
            for team in teams
        }
    return attack, defence


def _season_year(d) -> int:
    d = pd.Timestamp(d)
    return d.year if d.month >= 7 else d.year - 1


def _poisson_pmf(lam: float) -> np.ndarray:
    return np.exp(-lam) * np.power(lam, _GOAL_GRID) / _GOAL_FACTORIALS


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


def _primary_league_map(df: pd.DataFrame) -> dict[str, str]:
    """Each club's most-played league competition, across the whole frame.

    Used only by the P4b league-seeding path. Deliberately season-agnostic: a
    seed is a starting point for a club with little data, and for such a club
    the extra precision of a per-season league would be spurious.
    """
    counts: dict[tuple[str, str], int] = {}
    for comp, home, away in zip(df["competition"], df["home"], df["away"]):
        c = comp_get(comp)
        if not c or c.kind != "league":
            continue
        for team in (home, away):
            counts[(team, comp)] = counts.get((team, comp), 0) + 1
    best: dict[str, tuple[str, int]] = {}
    for (team, comp), n in counts.items():
        if best.get(team, ("", 0))[1] < n:
            best[team] = (comp, n)
    return {team: comp for team, (comp, _n) in best.items()}


def fit(df: pd.DataFrame | None = None,
        half_life_days: float = HALF_LIFE_DAYS,
        opponent_adjusted_xg: bool | None = None,
        league_seed: bool | None = None,
        coef_as_of: str | None = None,
        row_weights=None) -> dict:
    """Fit the promoted goals, Elo and opponent-adjusted xG model.

    row_weights: optional per-match multiplier used by the E0 bootstrap probe
    (`club_soccer/bootstrap_probe.py`) to resample the training set without
    dropping any club. Accepts an array aligned with `df` as passed, or a
    Series carrying `df`'s index; either way it is realigned to the frame
    AFTER `played()` filtering and date sorting, so a caller cannot silently
    pair weights with the wrong matches.

    `None` is exactly the production fit — no allocation, and the two places
    the multiplier lands (the time-decay weights, and the Elo update size) are
    both identity at 1.0. Do not use this to ship a model: a weighted fit is a
    diagnostic, and nothing under `season.py` may pass it.
    """
    source = load_fixtures() if df is None else df
    df = played(source).sort_values("date")
    if df.empty:
        raise ValueError("No played fixtures available to fit the model.")
    teams = sorted(set(df["home"]) | set(df["away"]))
    w = _weights(df["date"], half_life_days)
    boot = None
    if row_weights is not None:
        # Length is checked before the Series is built: constructing one with
        # a mismatched index raises pandas' own message, which says nothing
        # about row_weights and sends the reader to the wrong place.
        if len(row_weights) != len(source):
            raise ValueError(
                f"row_weights has {len(row_weights)} entries for "
                f"{len(source)} input matches"
            )
        series = (row_weights if isinstance(row_weights, pd.Series)
                  else pd.Series(np.asarray(row_weights, dtype=float),
                                 index=source.index))
        boot = series.reindex(df.index).to_numpy(dtype=float)
        if not np.all(np.isfinite(boot)):
            raise ValueError("row_weights must be finite")
        if np.any(boot < 0.0):
            raise ValueError("row_weights must be non-negative")
        # Time-decay and resampling weight compose multiplicatively: a match
        # drawn twice counts twice at whatever recency it already had.
        w = w * boot
    avg_home = float(np.average(df["home_goals"], weights=w))
    avg_away = float(np.average(df["away_goals"], weights=w))
    global_avg = max(0.8, (avg_home + avg_away) / 2)
    global_hfa = float(max(0.02, avg_home - avg_away))

    rows = []
    for r, wt in zip(df.itertuples(index=False), w):
        comp_str = strength(r.competition)
        rows.append((r.home, "for", r.home_goals, wt, comp_str))
        rows.append((r.home, "against", r.away_goals, wt, comp_str))
        rows.append((r.away, "for", r.away_goals, wt, comp_str))
        rows.append((r.away, "against", r.home_goals, wt, comp_str))
    stats = {t: {"gf": 0.0, "ga": 0.0, "wf": 0.0, "wa": 0.0,
                 "xf": 0.0, "xa": 0.0, "wx": 0.0}
             for t in teams}
    for t, typ, goals, wt, comp_str in rows:
        key, wk = ("gf", "wf") if typ == "for" else ("ga", "wa")
        stats[t][key] += float(goals) * wt
        stats[t][wk] += wt

    # Expected-goals signal: use BSD's real team xG wherever available and
    # retain the established SoT conversion as a fallback for older sources.
    has_sot = "home_sot" in df.columns and "away_sot" in df.columns
    if has_sot:
        # Fit the conversion on the SAME rows in numerator and denominator.
        # Treating missing SoT as zero while retaining those rows' goals
        # inflated the fitted conversion by 33% on the production dataset.
        # Missing shot data is absence of evidence, not zero shots.
        shot_mask = df["home_sot"].notna() & df["away_sot"].notna()
        if bool(shot_mask.any()):
            shot_w = w[shot_mask.to_numpy()]
            sot_sum = float(np.average(
                df.loc[shot_mask, "home_sot"] + df.loc[shot_mask, "away_sot"],
                weights=shot_w,
            ))
            goal_sum = float(np.average(
                df.loc[shot_mask, "home_goals"] + df.loc[shot_mask, "away_goals"],
                weights=shot_w,
            ))
            conv = goal_sum / sot_sum if sot_sum > 0 else 0.0
        else:
            conv = 0.0
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
    xg_observations: list[
        tuple[str, str, float, float, float, float, float]
    ] = []
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
            home_adv = 0.0 if int(getattr(r, "neutral", 0)) else global_hfa / 2
            comp_offset = (strength(r.competition) - 0.75) * 0.12
            xg_observations.append((
                r.home, r.away, xf_h, xf_a, float(wt),
                home_adv + comp_offset, -home_adv + comp_offset,
            ))

    # P4b: seed a team's shrinkage prior from ITS LEAGUE rather than the global
    # pooled mean. The pooled mean spans every fitted competition — after P3
    # that is 25 leagues from the Premier League to the National League — so a
    # club with little data is pulled toward the average of a population it
    # has nothing to do with. This is the mechanism behind the original
    # complaint: Sturm Graz sat at Elo 1505 and attack -0.267, i.e. "average",
    # where average was computed across English League Two among others.
    #
    # PROMOTED — see LEAGUE_SEED_DEFAULT. Pass league_seed=False explicitly to
    # reproduce pre-promotion behaviour (the A/B arms in the evidence file).
    league_seed_active = LEAGUE_SEED_DEFAULT if league_seed is None else bool(league_seed)
    opponent_adjusted_xg = (
        OPPONENT_ADJUSTED_XG_DEFAULT
        if opponent_adjusted_xg is None else bool(opponent_adjusted_xg)
    )
    team_league: dict[str, str] = {}
    league_gf: dict[str, float] = {}
    if league_seed_active:
        team_league = _primary_league_map(df)
        sums: dict[str, list[float]] = {}
        for r, wt in zip(df.itertuples(index=False), w):
            comp = r.competition
            acc = sums.setdefault(comp, [0.0, 0.0, 0.0])
            acc[0] += (float(r.home_goals) + float(r.away_goals)) * wt
            acc[2] += 2.0 * wt
        for comp, (goals, _unused, weight) in sums.items():
            if weight > 0:
                league_gf[comp] = goals / weight

    attack, defence, attack_xg, defence_xg = {}, {}, {}, {}
    base_xf, base_xa = {}, {}
    for t in teams:
        if league_seed_active:
            comp = team_league.get(t)
            gf_prior = ga_prior = league_gf.get(comp, global_avg) if comp else global_avg
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
        base_xf[t], base_xa[t] = xf, xa

    if opponent_adjusted_xg:
        attack_xg, defence_xg = _opponent_adjusted_xg_ratings(
            teams, xg_observations, global_avg
        )
        base_xf = {
            team: global_avg * math.exp(attack_xg[team]) for team in teams
        }
        base_xa = {
            team: global_avg * math.exp(defence_xg[team]) for team in teams
        }

    # recency form: last RECENT_K matches' xG signal vs the team's season
    # baseline, as a log-ratio attack/defence nudge (the part long-run rates
    # miss). Prefer BSD's observed xG and fall back to the SoT conversion for
    # older sources/competitions where xG is absent.
    recent = {t: [] for t in teams}
    if xg_observations:
        for home, away, xf_h, xf_a, _wt, home_offset, away_offset in xg_observations:
            if opponent_adjusted_xg:
                recent[home].append((
                    xf_h / math.exp(defence_xg[away] + home_offset),
                    xf_a / math.exp(attack_xg[away] + away_offset),
                ))
                recent[away].append((
                    xf_a / math.exp(defence_xg[home] + away_offset),
                    xf_h / math.exp(attack_xg[home] + home_offset),
                ))
            else:
                recent[home].append((xf_h, xf_a))
                recent[away].append((xf_a, xf_h))
    fatk, fdef = {}, {}
    for t in teams:
        last = recent[t][-RECENT_K:]
        if len(last) < 3:
            fatk[t] = fdef[t] = 0.0
            continue
        rf = float(np.mean([x[0] for x in last])); ra = float(np.mean([x[1] for x in last]))
        fatk[t] = float(np.clip(math.log(max(0.25, rf) / max(0.25, base_xf[t])), -0.4, 0.4))
        fdef[t] = float(np.clip(math.log(max(0.25, ra) / max(0.25, base_xa[t])), -0.4, 0.4))

    # Elo seeding. The default BASE_ELO start is the pooled-average problem in
    # its purest form: an unmeasured club begins life indistinguishable from a
    # mid-table side of whatever mix of divisions happens to be loaded. With
    # league seeding on, a club starts at its league's coefficient-implied
    # level instead, so the first European match it plays is priced from
    # "typical Austrian Bundesliga club" rather than "typical club anywhere".
    elo = {t: BASE_ELO for t in teams}
    if league_seed_active:
        from .uefa_registry import strength_prior as _sp
        for t in teams:
            comp = comp_get(team_league.get(t) or "")
            if comp is None:
                continue
            # Map strength (0.15-1.10) onto an Elo offset. ELO_PER_STRENGTH is
            # calibrated so the Premier-League-to-Scottish-Premiership gap in
            # strength matches their observed Elo gap; see the retained
            # league_seed_evidence.json artifact.
            # coef_as_of keeps historical folds from seeding on future
            # coefficients (the leak); None = latest snapshot, for live pricing.
            offset = (_sp(comp.country, comp.tier or 1, as_of=coef_as_of)
                      - LEAGUE_SEED_ANCHOR) * ELO_PER_STRENGTH
            elo[t] = BASE_ELO + offset
    # Elo is a sequential learner, so the resampling multiplier cannot ride on
    # the time-decay weights the way it does for goals and xG — those weights
    # never reach this loop. It scales the UPDATE SIZE instead: a match drawn
    # twice moves the ratings twice as far, drawn zero times moves them not at
    # all, which is the sequential analogue of counting it twice or dropping
    # it. Without this the probe would leave the Elo component (40% of the
    # default ensemble) identical across every resample and report a posterior
    # spread that understates the real one.
    boot_elo = repeat(1.0) if boot is None else boot
    for r, bw in zip(df.itertuples(index=False), boot_elo):
        h, a = r.home, r.away
        adv = 0.0 if int(r.neutral) else HOME_ADV_ELO
        exp_h = 1.0 / (1.0 + 10 ** ((elo[a] - (elo[h] + adv)) / 400.0))
        actual_h = 1.0 if r.home_goals > r.away_goals else (0.5 if r.home_goals == r.away_goals else 0.0)
        margin = abs(float(r.home_goals) - float(r.away_goals))
        comp_k = 18 + 20 * strength(r.competition)
        k = comp_k * (1.0 if margin <= 1 else min(1.75, 1 + margin / 4))
        delta = k * (actual_h - exp_h) * bw
        elo[h] += delta
        elo[a] -= delta

    params = {"teams": teams, "global_avg": global_avg,
              "home_goal_adv": global_hfa,
              "global_hfa": global_hfa,
              "opponent_adjusted_xg_active": bool(opponent_adjusted_xg),
              "attack": attack, "defence": defence,
              "attack_xg": attack_xg, "defence_xg": defence_xg,
              "fatk": fatk, "fdef": fdef, "conv": float(conv),
              "real_xg_coverage": round(real_xg_coverage, 6),
              "proxy_xg_coverage": round(proxy_xg_coverage, 6),
              "xg_source_counts": {str(k): int(v) for k, v in source.value_counts().items()},
              "elo": {k: float(v) for k, v in elo.items()},
              "fitted_matches": int(len(df)),
              # Provenance: a stored params file must say which model produced
              # it, so a cached artifact can never be mistaken for the other arm.
              "league_seed_active": bool(league_seed_active),
              # Per-team weight of shots-on-target evidence behind the xg
              # component. Zero means the club has NO shot data at all, so its
              # attack_xg/defence_xg are identically 0 — league-average by
              # construction, not by measurement. 12 competitions (~31% of
              # fitted matches) arrive from fd.co.uk's /new/ files with no shot
              # columns whatsoever, and the ensemble was still giving their xg
              # xg component held 40% of the weight.
              "xg_evidence": {t: round(float(stats[t]["wx"]), 4) for t in teams},
              # P0 coverage instrumentation. Every rating lookup below uses a
              # silent .get(team, default), so a club the fit never saw is
              # scored as exactly average. team_evidence records what the fit
              # actually rests on, per team, so that absence of data becomes
              # visible downstream instead of masquerading as an average team.
              "team_evidence": _COV.build_team_evidence(df)}
    return params


def save_params(params: dict, path: Path | None = None) -> None:
    path = PARAMS if path is None else path
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(params, indent=2))
    tmp.replace(path)


def training_fingerprint(df: pd.DataFrame | None = None) -> str:
    """Hash only inputs that can change a production fit.

    Upcoming fixtures and odds change every day but do not affect ratings, so
    they must not invalidate the model cache. The row hashes include ordering
    because Elo is sequential; the code fingerprint covers model behaviour and
    active coefficient/identity artifacts.
    """
    from . import walkforward_cache as WFC

    frame = played(load_fixtures() if df is None else df)
    frame = frame.sort_values("date").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(WFC.code_fingerprint().encode())
    digest.update(str(LEAGUE_SEED_DEFAULT).encode())
    digest.update(str(OPPONENT_ADJUSTED_XG_DEFAULT).encode())
    digest.update(WFC.row_hashes(frame).tobytes())
    return digest.hexdigest()[:32]


def fit_if_changed(force: bool = False) -> tuple[dict, bool]:
    """Return current params, fitting and atomically saving only when needed."""
    fixtures = load_fixtures()
    fingerprint = training_fingerprint(fixtures)
    if not force and PARAMS.exists():
        try:
            current = json.loads(PARAMS.read_text())
        except Exception:
            current = {}
        if current.get("_training_fingerprint") == fingerprint:
            print(
                f"  model unchanged — reused {current.get('fitted_matches', '?')} "
                "fitted matches"
            )
            return current, False

    params = fit(fixtures)
    params["_training_fingerprint"] = fingerprint
    save_params(params)
    print(
        f"  fitted {params['fitted_matches']} matches across "
        f"{len(params['teams'])} teams"
    )
    return params, True


def load_params() -> dict:
    if PARAMS.exists():
        return json.loads(PARAMS.read_text())
    params = fit()
    save_params(params)
    return params


def _comp_rho(_params: dict, _competition: str | None) -> float:
    return DC_RHO


def _lambdas_goals(params: dict, home: str, away: str, competition: str | None,
                   neutral: bool, match_date=None) -> tuple[float, float]:
    base = float(params["global_avg"])
    home_adv = 0.0 if neutral else float(params.get("home_goal_adv", 0.25)) / 2
    comp_adj = (strength(competition) - 0.75) * 0.12
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
    home_elo = 0.0 if neutral else HOME_ADV_ELO
    diff = (eh + home_elo - ea) / 400.0
    total = 2.55 + 0.20 * abs(diff)
    share = 1.0 / (1.0 + math.exp(-1.2 * diff))
    return max(0.15, total * share), max(0.15, total * (1 - share))


def _lambdas_xg(params: dict, home: str, away: str, competition: str | None,
                neutral: bool, form: bool = False, match_date=None) -> tuple[float, float]:
    """SoT-based expected-goals lambdas. With form=True, add the recent-SoT
    attack/defence nudge. Falls back to the goals attack/defence maps if a
    cached params dict predates the xg fields."""
    base = float(params["global_avg"])
    home_adv = 0.0 if neutral else float(params.get("home_goal_adv", 0.25)) / 2
    comp_adj = (strength(competition) - 0.75) * 0.12
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


def component_matrices(params: dict, home: str, away: str,
                       competition: str | None, neutral: bool,
                       match_date=None, mult_h: float = 1.0,
                       mult_a: float = 1.0) -> dict[str, np.ndarray]:
    """The ensemble component matrices.

    mult_h/mult_a scale each component's lambdas BEFORE the Dixon-Coles matrix
    is built (see _adj_multipliers). Adjustments have to enter here rather than
    on the blended result: the blend is a mixture of DC matrices and is
    not itself one, so rescaling it by its own marginals silently replaces the
    mixture with a plain Poisson. Defaults of 1.0 leave every component exactly
    as it was.
    """
    rho = _comp_rho(params, competition)

    def _m(lams: tuple[float, float]) -> np.ndarray:
        return score_matrix(lams[0] * mult_h, lams[1] * mult_a, rho)

    xg_long = _m(_lambdas_xg(
        params, home, away, competition, neutral, match_date=match_date
    ))
    xg_form = _m(_lambdas_xg(
        params, home, away, competition, neutral, form=True,
        match_date=match_date
    ))
    return {
        "goals": _m(_lambdas_goals(
            params, home, away, competition, neutral, match_date
        )),
        "elo": _m(_lambdas_elo(
            params, home, away, neutral, competition, match_date
        )),
        # Exact collapse of the former 20% long-run + 20% recent-form pair.
        "xg": (xg_long + xg_form) / 2.0,
    }


def _adj_multipliers(player_adj: dict | None = None) -> tuple[float, float]:
    """Collapse availability corrections into one (home, away) pair.

    Player availability terms are multiplicative on a side's expected goals.

    Sign conventions, unchanged from the individual appliers:
      * attack_mult  < 1.0 → that team scores less
      * defense_mult > 1.0 → the OPPONENT scores more
    """
    mult_h = mult_a = 1.0
    if player_adj:
        h = player_adj.get("home") or {}
        a = player_adj.get("away") or {}
        att_h = float(h.get("attack_mult", 1.0))
        def_h = float(h.get("defense_mult", 1.0))
        att_a = float(a.get("attack_mult", 1.0))
        def_a = float(a.get("defense_mult", 1.0))
        mult_h *= att_h * def_a
        mult_a *= att_a * def_h
    return mult_h, mult_a


# Components built from shots on target. Useless — and actively harmful — for
# a club with no shot data, because their attack/defence terms are then
# identically zero and the component returns a flat league-average matrix.
_SHOT_COMPONENTS = ("xg",)
# Weighted matches of shot data at which a club's xg terms earn half their
# nominal ensemble weight. The observed spread is wide — Arsenal 71.8, Sturm
# Graz 7.1 (European matches only), 449 clubs at exactly 0.0 — so this is a
# smooth confidence scale rather than a cliff.
XG_EVIDENCE_K = 8.0


def _weights_for_match(params: dict, weights: dict, home: str, away: str) -> dict:
    """Scale the shot-based components by how much shot data the clubs have.

    Twelve competitions (~31% of fitted matches) come from fd.co.uk's /new/
    files, which carry goals only. For those clubs attack_xg and defence_xg are
    identically zero — league-average by construction, not by measurement — so
    the xg component emits a flat matrix while still holding 40% of the
    default ensemble weight. That dilutes the goals and Elo components, which
    are the ones that actually know something, and it is the mechanism behind
    the post-P3 OU2.5 (+0.0126) and BTTS (+0.0083) regression.

    A match is only as informative as its weaker side, so confidence keys off
    the MINIMUM of the two clubs' evidence. Clubs with full shot histories are
    unaffected; the freed weight is renormalised onto goals and Elo.
    """
    evidence = params.get("xg_evidence")
    if not evidence:
        return weights                      # params predate this; leave as-is
    n = min(float(evidence.get(home, 0.0)), float(evidence.get(away, 0.0)))
    conf = n / (n + XG_EVIDENCE_K)
    if conf > 0.99:
        return weights
    scaled = {k: (v * conf if k in _SHOT_COMPONENTS else v)
              for k, v in weights.items()}
    total = sum(scaled.values())
    if total <= 0:
        # Every surviving weight is zero — fall back rather than divide by
        # zero. A diluted prediction beats no prediction.
        return weights
    return {k: v / total for k, v in scaled.items()}


def predict(home: str, away: str, competition: str | None = None,
            model: str = "ensemble", neutral: bool = False,
            params: dict | None = None,
            player_adj: dict | None = None,
            match_date=None,
            ensemble_weights: dict | None = None) -> dict:
    """Predict match outcome probabilities.

    Parameters
    ----------
    home, away:    Team names (must exist in params["teams"]).
    competition:   Competition name (used for strength adjustment).
    model:         "ensemble" | "goals" | "elo" | "xg".
    neutral:       True for a neutral venue.
    params:        Pre-loaded model params (default: load from disk).
    player_adj:    Optional player availability multipliers from
                   club_soccer.player_features.PlayerFeatureStore.
                   Format: {"home": {"attack_mult": float, "defense_mult": float},
                             "away": {"attack_mult": float, "defense_mult": float}}
                   Values outside [0.80, 1.25] are silently clamped.
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

    # Availability corrections are multiplicative on each side's expected
    # goals and are pushed into the lambdas before a Dixon-Coles matrix is
    # built.
    #
    # They used to be applied to the finished matrix instead, by reading off its
    # marginal means and rebuilding. That is only valid when the matrix is a
    # single DC matrix; the ensemble is a MIXTURE, so the rebuild threw
    # the mixture away and replaced it with a plain Poisson matching only the
    # means. An identity adjustment was therefore not the identity: all-1.0
    # multipliers moved BTTS by +0.0041 and Over 2.5 by +0.0031 on a live
    # fixture, and every match with absence data was priced off a structurally
    # different distribution than one without. The rebuild also dropped `rho`,
    # silently substituting the global DC_RHO for the per-competition value.
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
    mult_h, mult_a = _adj_multipliers(applied_player_adj)

    def _scaled(lams: tuple[float, float]) -> np.ndarray:
        return score_matrix(lams[0] * mult_h, lams[1] * mult_a, rho)

    if model == "ensemble":
        parts = component_matrices(params, home, away, competition, neutral,
                                   match_date, mult_h=mult_h, mult_a=mult_a)
        weights = (_normalise_weights(ensemble_weights)
                   if ensemble_weights is not None
                   else dict(DEFAULT_ENSEMBLE_W))
        weights = _weights_for_match(params, weights, home, away)
        M = sum(weights[k] * parts[k] for k in ENSEMBLE_COMPONENTS)
        M = M / M.sum()
    elif model == "goals":
        M = _scaled(_lambdas_goals(params, home, away, competition, neutral, match_date))
    elif model == "elo":
        M = _scaled(_lambdas_elo(params, home, away, neutral, competition, match_date))
    elif model == "xg":
        M = _scaled(_lambdas_xg(params, home, away, competition, neutral,
                                form=True, match_date=match_date))
    else:
        raise ValueError("Unknown model: use ensemble, goals, elo, or xg.")

    probs = probs_from_matrix(M)
    xg_h = float(sum(i * M[i, :].sum() for i in range(M.shape[0])))
    xg_a = float(sum(j * M[:, j].sum() for j in range(M.shape[1])))
    out = {"home": home, "away": away, "competition": competition or "",
           "model": model, "xg_home": round(xg_h, 2), "xg_away": round(xg_a, 2),
           "probs": {k: round(v, 4) for k, v in probs.items()},
           "scorelines": top_scorelines(M), "matrix": M,
           # P0: report-only evidence tier. Probabilities above are UNCHANGED
           # by this; it exists so a consumer can tell a price built on 300
           # domestic matches from one built on 24 European matches against
           # unrated opposition.
           "coverage": _COV.match_coverage(params, home, away)}
    if player_adj_applied:
        out["player_adj"] = {
            side: {k: v for k, v in applied_player_adj.get(side, {}).items()
                   if k in ("attack_mult", "defense_mult", "n_missing")}
            for side in ("home", "away")
        }
    return out


def predict_match(home: str, away: str, competition: str | None,
                  match_date: str, model: str = "ensemble",
                  neutral: bool = False, params: dict | None = None,
                  player_adj: dict | None = None, fixture_id=None,
                  ensemble_weights: dict | None = None) -> dict:
    """Point-in-time prediction wrapper used by cards and edge pricing."""
    del fixture_id
    return predict(home, away, competition, model, neutral, params=params,
                   player_adj=player_adj,
                   match_date=match_date,
                   ensemble_weights=ensemble_weights)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("home", nargs="?")
    ap.add_argument("away", nargs="?")
    ap.add_argument("--competition", default="")
    ap.add_argument("--model", choices=["ensemble", "goals", "elo", "xg"], default="ensemble")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()
    if args.fit:
        params, _changed = fit_if_changed(force=True)
        print(f"Saved {len(params['teams'])} teams from {params['fitted_matches']} matches -> {PARAMS}")
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
