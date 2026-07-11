#!/usr/bin/env python3
"""Context GLM: rest, congestion, minutes load — fitted, gated.

Direct port of engines/worldcup/context.py's Poisson-IRLS-with-offset
pattern, minus altitude (irrelevant for UK/EU club venues) and travel.

Offset = log(lambda) from the GOALS component only (model._lambdas_goals) —
a deterministic, params-only lambda, not the blended ensemble (the ensemble
already mixes in Elo/xG signal that would double-count against these terms).

Features (per side; diffs clipped):
  rest_diff      own rest days minus opponent's; rest capped at
                 REST_CAP=14, diff clipped to +/-REST_DIFF_CLIP=7 (same
                 constants as the WC module).
  cong14_diff    own matches_14d minus opponent's, clipped +/-4.
  euro_hangover  1 if the side played a Champions/Europa/Conference League
                 match 2-4 days before this match, else 0 — domestic
                 (league/cup) fixtures only.
  xi_load14_diff own xi_load_14d minus opponent's, clipped +/-3. Dropped
                 entirely (not just pruned by |t|) if minutes-cache
                 coverage is below 50% of rows — the player cache is still
                 accumulating history and a near-constant-zero column is
                 numerically unstable to fit.

Fit: Poisson IRLS with offset, on played fixtures since 2022-08, features
computed point-in-time (rest/congestion across ALL competitions, from
feature_store's helpers). Keep |t| >= 2 terms -> data/context_coef_club.json.

Apply: model.predict gains context_adj (dict | None); when provided,
lambda_h *= exp(sum b*f_h), lambda_a *= exp(sum b*f_a), applied to the
blended matrix's extracted lambdas — same mechanism as apply_player_adj.

Usage:
  python3 -m club_soccer.context --fit
  python3 -m club_soccer.context --validate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import model as M
from . import feature_store as FS
from . import motivation as MOT
from .competitions import get as comp_get
from .player_features import PlayerFeatureStore

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
COEF_FILE = DATA / "context_coef_club.json"

REST_CAP = 14
REST_DIFF_CLIP = 7
CONG14_CLIP = 4
XI_LOAD14_CLIP = 3.0
MIN_XI_LOAD_COVERAGE = 0.50
EURO_COMPS = {"Champions League", "Europa League", "Conference League"}
FIT_SINCE = "2022-08-01"
SPLIT = "2025-12-01"   # held-out boundary for --validate


# ── euro-hangover flag (cross-competition, point-in-time) ─────────────────────
def _euro_hangover_flags(played: pd.DataFrame) -> pd.DataFrame:
    """1 if the side played a EURO_COMPS match 2-4 days before THIS match,
    for domestic (league/cup) fixtures only. Single chronological pass;
    per-team state updated after emitting, so a match never sees itself."""
    played = played.sort_values("date").reset_index(drop=True)
    n = len(played)
    hang_h = np.zeros(n, dtype=int)
    hang_a = np.zeros(n, dtype=int)
    recent_euro: dict[str, list] = {}
    for i, r in enumerate(played.itertuples(index=False)):
        d = r.date
        domestic = r.type in ("league", "cup")
        for team, arr in ((r.home, hang_h), (r.away, hang_a)):
            dates = [x for x in recent_euro.get(team, []) if (d - x).days <= 10]
            recent_euro[team] = dates
            if domestic and any(2 <= (d - x).days <= 4 for x in dates):
                arr[i] = 1
        if r.competition in EURO_COMPS:
            for team in (r.home, r.away):
                recent_euro.setdefault(team, []).append(d)
    out = played[["fixture_id"]].copy()
    out["euro_hangover_h"] = hang_h
    out["euro_hangover_a"] = hang_a
    return out


# ── cup tier_gap (P4.7): giant-killing / rotation-by-favourites signal ────────
def _cup_tier_gap(played: pd.DataFrame) -> pd.DataFrame:
    """tier_gap = away league tier - home league tier, for cup-type fixtures
    only (0 for league/europe fixtures, and for cup ties where either side's
    domestic league tier isn't yet known). Each team's tier comes from its
    most recent LEAGUE match strictly before this cup tie."""
    played = played.sort_values("date").reset_index(drop=True)
    n = len(played)
    tier_gap = np.zeros(n)
    last_tier: dict[str, int] = {}
    for i, r in enumerate(played.itertuples(index=False)):
        if r.type == "cup":
            th, ta = last_tier.get(r.home, 0), last_tier.get(r.away, 0)
            if th > 0 and ta > 0:
                tier_gap[i] = float(ta - th)
        elif r.type == "league":
            comp = comp_get(r.competition)
            if comp is not None and comp.tier > 0:
                last_tier[r.home] = comp.tier
                last_tier[r.away] = comp.tier
    out = played[["fixture_id"]].copy()
    out["tier_gap"] = tier_gap
    return out


# ── weather (P5): symmetric — shifts TOTALS, not a side ───────────────────────
def _weather_features(played_all: pd.DataFrame) -> pd.DataFrame:
    """wind_high/precip/temp_cold/temp_hot per fixture from data/weather.csv
    (weather.py). Unlike every other context term, these are NOT diffed by
    side — a windy/wet/cold/hot pitch affects both teams' scoring equally,
    so the same value applies to both the home-row and away-row features."""
    from . import weather as W
    wx = W.load_weather()
    cols = ["wind_high", "precip", "temp_cold", "temp_hot"]
    if wx.empty:
        out = played_all[["fixture_id"]].copy()
        for c in cols:
            out[c] = np.nan
        return out
    feats = wx.apply(lambda r: pd.Series(W.features(r["temp_c"], r["precip_mm"], r["wind_kmh"])),
                     axis=1)
    merged = pd.concat([wx[["fixture_id"]], feats], axis=1)
    out = played_all[["fixture_id"]].merge(merged, on="fixture_id", how="left")
    return out[["fixture_id"] + cols]


# ── dataset construction (mirrors WC's build_dataset) ─────────────────────────
def build_dataset(played_all: pd.DataFrame, mparams: "FS._MonthlyParams",
                  sched: pd.DataFrame, hangover: pd.DataFrame, motiv: pd.DataFrame,
                  tier: pd.DataFrame, weather: pd.DataFrame,
                  mcache: "FS._MinutesCache | None", subset_mask: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """Per-side rows: y=goals, offset=log(goals-component lambda), X=features.

    Returns (y, lam, X, feature_names, xi_load_coverage).
    """
    df = played_all.merge(sched.drop(columns=["date", "home", "away"]),
                          on="fixture_id", how="left")
    df = df.merge(hangover, on="fixture_id", how="left")
    df = df.merge(motiv, on="fixture_id", how="left")
    df = df.merge(tier, on="fixture_id", how="left")
    df = df.merge(weather, on="fixture_id", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    rows = []
    xi_available = xi_total = 0
    for i, r in enumerate(df.itertuples(index=False)):
        if not subset_mask[i]:
            continue
        params = mparams.for_date(r.date)
        teams = set(params["teams"])
        if r.home not in teams or r.away not in teams:
            continue
        lh, la = M._lambdas_goals(params, r.home, r.away, r.competition, bool(r.neutral))

        rh = REST_CAP if pd.isna(r.rest_days_h) else min(r.rest_days_h, REST_CAP)
        ra = REST_CAP if pd.isna(r.rest_days_a) else min(r.rest_days_a, REST_CAP)
        rd = float(np.clip(rh - ra, -REST_DIFF_CLIP, REST_DIFF_CLIP))
        cd = float(np.clip(r.matches_14d_h - r.matches_14d_a, -CONG14_CLIP, CONG14_CLIP))

        xid = 0.0
        xi_total += 1
        if mcache is not None:
            md = pd.Timestamp(r.date).strftime("%Y-%m-%d")
            xi_h = mcache.xi_loads(r.home, md)
            xi_a = mcache.xi_loads(r.away, md)
            if mcache.has_data(r.home, md) or mcache.has_data(r.away, md):
                xi_available += 1
            xid = float(np.clip(xi_h["xi_load_14d"] - xi_a["xi_load_14d"],
                                -XI_LOAD14_CLIP, XI_LOAD14_CLIP))

        ppg = 0.0 if pd.isna(r.ppg_diff) else float(r.ppg_diff)
        fight = 0.0 if pd.isna(r.fight_diff) else float(r.fight_diff)
        dead = 0.0 if pd.isna(r.dead_diff) else float(r.dead_diff)
        tg = 0.0 if pd.isna(r.tier_gap) else float(r.tier_gap)
        wh = 0.0 if pd.isna(r.wind_high) else float(r.wind_high)
        pr = 0.0 if pd.isna(r.precip) else float(r.precip)
        tc = 0.0 if pd.isna(r.temp_cold) else float(r.temp_cold)
        th = 0.0 if pd.isna(r.temp_hot) else float(r.temp_hot)

        # Weather is symmetric — SAME value for both rows (not negated); it
        # shifts total goals, not which side benefits.
        rows.append((r.home_goals, lh, rd, cd, float(r.euro_hangover_h), xid,
                    ppg, fight, dead, tg, wh, pr, tc, th))
        rows.append((r.away_goals, la, -rd, -cd, float(r.euro_hangover_a), -xid,
                    -ppg, -fight, -dead, -tg, wh, pr, tc, th))

    arr = np.array(rows, dtype=float)
    coverage = (xi_available / xi_total) if xi_total else 0.0
    names = ["rest_diff", "cong14_diff", "euro_hangover", "xi_load14_diff",
             "ppg_diff", "fight_diff", "dead_diff", "tier_gap",
             "wind_high", "precip", "temp_cold", "temp_hot"]
    return arr[:, 0], arr[:, 1], arr[:, 2:], names, coverage


def _poisson_fit(y, lam, X):
    """Poisson IRLS with offset=log(lam). Design = [intercept, features].
    Returns coef, SE (same order as columns). Verbatim port of the WC
    context module's fitter."""
    o = np.log(np.clip(lam, 1e-6, None))
    D = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(D.shape[1])
    for _ in range(50):
        mu = np.exp(o + D @ b)
        W = mu
        z = D @ b + (y - mu) / mu
        XtW = D.T * W
        b_new = np.linalg.solve(XtW @ D, XtW @ z)
        if np.max(np.abs(b_new - b)) < 1e-10:
            b = b_new
            break
        b = b_new
    cov = np.linalg.inv((D.T * np.exp(o + D @ b)) @ D)
    se = np.sqrt(np.diag(cov))
    return b, se


def _build_common(since: str, until: str | None):
    fx = M.load_fixtures()
    played_all = M.played(fx).sort_values("date").reset_index(drop=True)
    sched = FS._schedule_features(played_all)
    hangover = _euro_hangover_flags(played_all)
    motiv = MOT.training_features(played_all)
    tier = _cup_tier_gap(played_all)
    weather = _weather_features(played_all)
    mparams = FS._MonthlyParams(played_all)
    store = PlayerFeatureStore().load()
    mcache = FS._MinutesCache(store) if store._player_records() else None

    mask = (played_all["date"] >= pd.Timestamp(since)).to_numpy().copy()
    if until:
        mask &= (played_all["date"] < pd.Timestamp(until)).to_numpy()
    return played_all, mparams, sched, hangover, motiv, tier, weather, mcache, mask


def fit_context(verbose: bool = True, save: bool = True,
                since: str = FIT_SINCE, until: str | None = None) -> dict:
    played_all, mparams, sched, hangover, motiv, tier, weather, mcache, mask = _build_common(since, until)
    y, lam, X, names, coverage = build_dataset(played_all, mparams, sched, hangover,
                                               motiv, tier, weather, mcache, mask)
    if coverage < MIN_XI_LOAD_COVERAGE:
        if verbose:
            print(f"  xi_load14_diff coverage {coverage:.1%} < {MIN_XI_LOAD_COVERAGE:.0%} "
                  f"— dropping that term from the fit")
        keep_idx = [i for i, nm in enumerate(names) if nm != "xi_load14_diff"]
        X = X[:, keep_idx]
        names = [names[i] for i in keep_idx]

    # A constant (zero-variance) column is singular in the IRLS normal
    # equations — e.g. tier_gap has no signal when the data has no PLAYED
    # domestic-cup fixtures yet (only upcoming ones). Drop rather than crash.
    zero_var = [i for i in range(X.shape[1]) if np.std(X[:, i]) < 1e-9]
    if zero_var:
        dropped = [names[i] for i in zero_var]
        if verbose:
            print(f"  dropping zero-variance term(s) (no signal in this data yet): {dropped}")
        keep_idx = [i for i in range(X.shape[1]) if i not in zero_var]
        X = X[:, keep_idx]
        names = [names[i] for i in keep_idx]

    b, se = _poisson_fit(y, lam, X)
    t = b / se
    coef = {}
    all_names = ["intercept"] + names
    if verbose:
        print(f"Poisson context fit ({len(y)} side-observations, "
              f"xi_load coverage {coverage:.1%}):")
        print(f"  {'term':16s}{'coef':>10}{'se':>9}{'t':>8}{'kept':>7}")
    for nm, bi, si, ti in zip(all_names, b, se, t):
        keep = nm != "intercept" and abs(ti) >= 2.0
        if keep:
            coef[nm] = float(bi)
        if verbose:
            print(f"  {nm:16s}{bi:>10.4f}{si:>9.4f}{ti:>8.2f}{('yes' if keep else '—'):>7}")
    if save:
        COEF_FILE.write_text(json.dumps({
            "coef": coef, "n": int(len(y)), "xi_load_coverage": round(coverage, 4),
            "rest_cap": REST_CAP, "rest_diff_clip": REST_DIFF_CLIP,
            "cong14_clip": CONG14_CLIP, "xi_load14_clip": XI_LOAD14_CLIP,
            "active": False,
            "note": "lambda *= exp(sum b_i*feature_i); intercept not applied; "
                    "promote per plan Sec 12 before setting active=true",
        }, indent=2))
        if verbose:
            print(f"  saved kept coefficients -> {COEF_FILE.name}: {coef}")
    return coef


def load_coef() -> dict:
    if COEF_FILE.exists():
        try:
            payload = json.loads(COEF_FILE.read_text())
            if not payload.get("active", False):
                return {}
            return payload.get("coef", {})
        except Exception:
            return {}
    return {}


# ── live application ──────────────────────────────────────────────────────────
def context_features_asof(home: str, away: str, match_date: str, is_domestic: bool,
                          competition: str | None = None, season: int | None = None,
                          is_cup: bool = False, fixture_id=None,
                          store: PlayerFeatureStore | None = None,
                          include_motivation: bool = True) -> dict:
    """Point-in-time context features for a concrete upcoming fixture, using
    only data dated strictly before match_date."""
    fx = M.load_fixtures()
    played = M.played(fx)
    d = pd.Timestamp(match_date)
    before = played[played["date"] < d]

    def _team_matches(team: str) -> pd.DataFrame:
        return before[(before["home"] == team) | (before["away"] == team)]

    def _rest(team: str) -> float:
        tm = _team_matches(team)
        if tm.empty:
            return REST_CAP
        return min((d - tm["date"].max()).days, REST_CAP)

    def _cong14(team: str) -> int:
        tm = _team_matches(team)
        cutoff = d - pd.Timedelta(days=14)
        return int((tm["date"] > cutoff).sum())

    def _hangover(team: str) -> int:
        if not is_domestic:
            return 0
        tm = _team_matches(team)
        euro = tm[tm["competition"].isin(EURO_COMPS)]
        return int(any(2 <= (d - dt).days <= 4 for dt in euro["date"]))

    rh, ra = _rest(home), _rest(away)
    rd = float(np.clip(rh - ra, -REST_DIFF_CLIP, REST_DIFF_CLIP))
    cd = float(np.clip(_cong14(home) - _cong14(away), -CONG14_CLIP, CONG14_CLIP))

    xid = 0.0
    if store is not None and store._player_records():
        from .minutes import build_player_minutes, xi_loads
        asof = str((d - pd.Timedelta(days=1)).date())
        mdf = build_player_minutes(store, asof)
        if not mdf.empty:
            xi_h = xi_loads(mdf, home)["xi_load_14d"]
            xi_a = xi_loads(mdf, away)["xi_load_14d"]
            xid = float(np.clip(xi_h - xi_a, -XI_LOAD14_CLIP, XI_LOAD14_CLIP))

    tg = 0.0
    if is_cup:
        def _last_league_tier(team: str) -> int:
            tm = before[((before["home"] == team) | (before["away"] == team))
                       & (before["type"] == "league")]
            if tm.empty:
                return 0
            last = tm.sort_values("date").iloc[-1]
            c = comp_get(last["competition"])
            return c.tier if c is not None else 0
        th, ta = _last_league_tier(home), _last_league_tier(away)
        if th > 0 and ta > 0:
            tg = float(ta - th)

    wind_high = precip = temp_cold = temp_hot = 0.0
    if fixture_id is not None:
        from . import weather as W
        wx = W.load_weather()
        row = wx[wx["fixture_id"] == fixture_id]
        if not row.empty:
            f = W.features(float(row.iloc[0]["temp_c"]), float(row.iloc[0]["precip_mm"]),
                           float(row.iloc[0]["wind_kmh"]))
            wind_high, precip, temp_cold, temp_hot = (
                f["wind_high"], f["precip"], f["temp_cold"], f["temp_hot"])

    out = {"rest_diff": rd, "cong14_diff": cd,
           "euro_hangover_h": _hangover(home), "euro_hangover_a": _hangover(away),
           "xi_load14_diff": xid, "ppg_diff": 0.0, "fight_diff": 0.0, "dead_diff": 0.0,
           "tier_gap": tg, "wind_high": wind_high, "precip": precip,
           "temp_cold": temp_cold, "temp_hot": temp_hot}
    if include_motivation and is_domestic and competition and season is not None:
        out.update(MOT.live_features(home, away, competition, season, match_date))
    return out


def context_adj_for_match(home: str, away: str, match_date: str, is_domestic: bool,
                          competition: str | None = None, season: int | None = None,
                          is_cup: bool = False, fixture_id=None,
                          store: PlayerFeatureStore | None = None,
                          coef: dict | None = None) -> dict:
    """{"home": {"mult": float}, "away": {"mult": float}} for model.predict's
    context_adj parameter — empty dict (no-op) if no coefficients are active."""
    coef = load_coef() if coef is None else coef
    if not coef:
        return {}
    needs_motivation = any(k in coef for k in ("ppg_diff", "fight_diff", "dead_diff"))
    needs_weather = any(k in coef for k in ("wind_high", "precip", "temp_cold", "temp_hot"))
    f = context_features_asof(
        home, away, match_date, is_domestic, competition, season, is_cup,
        fixture_id if needs_weather else None, store,
        include_motivation=needs_motivation,
    )
    br, bc, be, bx, bp, bf, bd, bt, bw, bpr, btc, bth = (
        coef.get("rest_diff", 0.0), coef.get("cong14_diff", 0.0),
        coef.get("euro_hangover", 0.0), coef.get("xi_load14_diff", 0.0),
        coef.get("ppg_diff", 0.0), coef.get("fight_diff", 0.0), coef.get("dead_diff", 0.0),
        coef.get("tier_gap", 0.0), coef.get("wind_high", 0.0), coef.get("precip", 0.0),
        coef.get("temp_cold", 0.0), coef.get("temp_hot", 0.0))
    weather_term = (bw * f["wind_high"] + bpr * f["precip"]
                    + btc * f["temp_cold"] + bth * f["temp_hot"])  # symmetric, same for both
    mult_h = float(np.exp(br * f["rest_diff"] + bc * f["cong14_diff"]
                          + be * f["euro_hangover_h"] + bx * f["xi_load14_diff"]
                          + bp * f["ppg_diff"] + bf * f["fight_diff"] + bd * f["dead_diff"]
                          + bt * f["tier_gap"] + weather_term))
    mult_a = float(np.exp(br * (-f["rest_diff"]) + bc * (-f["cong14_diff"])
                          + be * f["euro_hangover_a"] + bx * (-f["xi_load14_diff"])
                          + bp * (-f["ppg_diff"]) + bf * (-f["fight_diff"]) + bd * (-f["dead_diff"])
                          + bt * (-f["tier_gap"]) + weather_term))
    return {"home": {"mult": mult_h}, "away": {"mult": mult_a}}


# ── validation (held-out, same layout as the WC module) ───────────────────────
def validate(verbose: bool = True) -> bool:
    """Refit on pre-SPLIT data, evaluate 1X2 log-loss with/without the
    correction on held-out (>= SPLIT) months. No promotion here — this just
    reports; promotion follows plan Sec 12."""
    played_all, mparams, sched, hangover, motiv, tier, weather, mcache, pre_mask = \
        _build_common(FIT_SINCE, SPLIT)
    coef = fit_context(verbose=False, save=False, since=FIT_SINCE, until=SPLIT)

    df = played_all.merge(sched.drop(columns=["date", "home", "away"]),
                          on="fixture_id", how="left")
    df = df.merge(hangover, on="fixture_id", how="left")
    df = df.merge(motiv, on="fixture_id", how="left")
    df = df.merge(tier, on="fixture_id", how="left")
    df = df.merge(weather, on="fixture_id", how="left").sort_values("date").reset_index(drop=True)
    post_mask = (df["date"] >= pd.Timestamp(SPLIT)).to_numpy()

    br, bc, be, bx, bp, bf, bd, bt, bw, bpr, btc, bth = (
        coef.get("rest_diff", 0.0), coef.get("cong14_diff", 0.0),
        coef.get("euro_hangover", 0.0), coef.get("xi_load14_diff", 0.0),
        coef.get("ppg_diff", 0.0), coef.get("fight_diff", 0.0), coef.get("dead_diff", 0.0),
        coef.get("tier_gap", 0.0), coef.get("wind_high", 0.0), coef.get("precip", 0.0),
        coef.get("temp_cold", 0.0), coef.get("temp_hot", 0.0))
    has_weather_coef = any([bw, bpr, btc, bth])

    lb = lc = 0.0
    ens_lb = ens_lc = 0.0
    ens_bb = ens_bc = 0.0
    ou_b = ou_c = 0.0
    n = n_ou = 0
    for i, r in enumerate(df.itertuples(index=False)):
        if not post_mask[i]:
            continue
        params = mparams.for_date(r.date)
        if r.home not in set(params["teams"]) or r.away not in set(params["teams"]):
            continue
        lh, la = M._lambdas_goals(params, r.home, r.away, r.competition, bool(r.neutral))
        rh = REST_CAP if pd.isna(r.rest_days_h) else min(r.rest_days_h, REST_CAP)
        ra = REST_CAP if pd.isna(r.rest_days_a) else min(r.rest_days_a, REST_CAP)
        rd = float(np.clip(rh - ra, -REST_DIFF_CLIP, REST_DIFF_CLIP))
        cd = float(np.clip(r.matches_14d_h - r.matches_14d_a, -CONG14_CLIP, CONG14_CLIP))
        xid = 0.0
        if mcache is not None:
            md = pd.Timestamp(r.date).strftime("%Y-%m-%d")
            xi_h = mcache.xi_loads(r.home, md); xi_a = mcache.xi_loads(r.away, md)
            xid = float(np.clip(xi_h["xi_load_14d"] - xi_a["xi_load_14d"],
                                -XI_LOAD14_CLIP, XI_LOAD14_CLIP))
        ppg = 0.0 if pd.isna(r.ppg_diff) else float(r.ppg_diff)
        fight = 0.0 if pd.isna(r.fight_diff) else float(r.fight_diff)
        dead = 0.0 if pd.isna(r.dead_diff) else float(r.dead_diff)
        tg = 0.0 if pd.isna(r.tier_gap) else float(r.tier_gap)
        wh = 0.0 if pd.isna(r.wind_high) else float(r.wind_high)
        pr = 0.0 if pd.isna(r.precip) else float(r.precip)
        tc = 0.0 if pd.isna(r.temp_cold) else float(r.temp_cold)
        th = 0.0 if pd.isna(r.temp_hot) else float(r.temp_hot)
        weather_term = bw * wh + bpr * pr + btc * tc + bth * th
        mult_h = float(np.exp(br * rd + bc * cd + be * r.euro_hangover_h + bx * xid
                              + bp * ppg + bf * fight + bd * dead + bt * tg + weather_term))
        mult_a = float(np.exp(br * (-rd) + bc * (-cd) + be * r.euro_hangover_a + bx * (-xid)
                              + bp * (-ppg) + bf * (-fight) + bd * (-dead) + bt * (-tg) + weather_term))
        # Evaluate the correction on the production ensemble as well as the
        # historical goals-only diagnostic above. Context coefficients were
        # originally fitted against the goals component, but predict() applies
        # them to the blended score matrix; the ensemble gate must test that
        # actual production path before activation.
        ensemble = M.predict(r.home, r.away, r.competition, "ensemble",
                             bool(r.neutral), params=params)
        ensemble_base = M.probs_from_matrix(ensemble["matrix"])
        ensemble_context = M.probs_from_matrix(
            M.apply_context_adj(
                ensemble["matrix"],
                {"home": {"mult": mult_h}, "away": {"mult": mult_a}},
            )
        )
        y = 0 if r.home_goals > r.away_goals else (1 if r.home_goals == r.away_goals else 2)
        y_over = 1.0 if (r.home_goals + r.away_goals) > 2.5 else 0.0
        has_weather_data = pd.notna(r.wind_high)  # a real (non-default) weather row exists
        for tag, (l1, l2) in (("b", (lh, la)), ("c", (lh * mult_h, la * mult_a))):
            mat = M.score_matrix(l1, l2, M.DC_RHO)
            p = M.probs_from_matrix(mat)
            probs = [p["home"], p["draw"], p["away"]]
            ll = -np.log(max(probs[y], 1e-9))
            if tag == "b":
                lb += ll
            else:
                lc += ll
            if has_weather_coef and has_weather_data:
                brier = (p["over25"] - y_over) ** 2
                if tag == "b":
                    ou_b += brier
                else:
                    ou_c += brier
        base_probs = [ensemble_base["home"], ensemble_base["draw"], ensemble_base["away"]]
        context_probs = [ensemble_context["home"], ensemble_context["draw"], ensemble_context["away"]]
        one = np.eye(3)[y]
        ens_lb += -np.log(max(base_probs[y], 1e-9))
        ens_lc += -np.log(max(context_probs[y], 1e-9))
        ens_bb += float(np.sum((np.asarray(base_probs) - one) ** 2))
        ens_bc += float(np.sum((np.asarray(context_probs) - one) ** 2))
        n += 1
        if has_weather_coef and has_weather_data:
            n_ou += 1

    TOL = 1e-3
    OU_TOL = 0.0005   # plan Sec 12: weather's primary metric is OU2.5 Brier;
                       # 1X2 is allowed to move up to this much
    b_avg = lb / n if n else float("nan")
    c_avg = lc / n if n else float("nan")
    ens_b_avg = ens_lb / n if n else float("nan")
    ens_c_avg = ens_lc / n if n else float("nan")
    ens_bb_avg = ens_bb / n if n else float("nan")
    ens_bc_avg = ens_bc / n if n else float("nan")
    ou_b_avg = ou_b / n_ou if n_ou else float("nan")
    ou_c_avg = ou_c / n_ou if n_ou else float("nan")

    goals_ok = ((ou_c_avg < ou_b_avg) and ((c_avg - b_avg) <= OU_TOL)
                if has_weather_coef and n_ou
                else n > 0 and (c_avg - b_avg) <= TOL)
    ensemble_ok = (n > 0 and ens_c_avg < ens_b_avg and ens_bc_avg < ens_bb_avg)
    ok = bool(goals_ok and ensemble_ok)

    if verbose:
        print(f"\nHeld-out after {SPLIT}. fitted coef (pre-{SPLIT}): {coef}")
        print(f"  n={n}: mean log-loss  base(goals-only) {b_avg:.4f}  "
              f"+context {c_avg:.4f}  (delta {c_avg - b_avg:+.4f})")
        print(f"  n={n}: ensemble Brier {ens_bb_avg:.4f} -> {ens_bc_avg:.4f} "
              f"(delta {ens_bc_avg - ens_bb_avg:+.4f}); "
              f"log-loss {ens_b_avg:.4f} -> {ens_c_avg:.4f} "
              f"(delta {ens_c_avg - ens_b_avg:+.4f})")
        if has_weather_coef:
            print(f"  n_ou={n_ou} (rows with real weather data): OU2.5 Brier "
                  f"base {ou_b_avg:.4f}  +context {ou_c_avg:.4f}  "
                  f"(delta {ou_c_avg - ou_b_avg:+.4f})")
            print(f"  gate goals/OU={goals_ok}, ensemble Brier+log-loss={ensemble_ok}: {ok}")
        else:
            print(f"  gate goals={goals_ok}, ensemble Brier+log-loss={ensemble_ok}: {ok}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Club Soccer context GLM (rest/congestion/minutes-load)")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.fit:
        fit_context()
    elif args.validate:
        validate()
    else:
        print("coef (active only):", load_coef() or "(none active; run --fit, then promote per plan Sec 12)")


if __name__ == "__main__":
    main()
