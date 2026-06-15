#!/usr/bin/env python3
"""Context features for WC2026: rest and altitude (v2 M6).

Builds per-fixture context features and fits a single multiplicative correction to
the model's expected goals (lambda), so a tired or altitude-disadvantaged side is
nudged down. The correction is opt-in (--context in edge.py); the default pipeline
is untouched. Expect small gains.

Features (per side, applied to that side's lambda):
  rest_diff   own rest days minus opponent's (capped), tournaments congest fixtures
  alt_gap     how far ABOVE the team's usual altitude the venue is, in km
              (team's usual altitude = median of its home-match venue altitudes);
              lowland teams at altitude get penalised, altitude teams do not.

Travel (great-circle since last match) was specified but DROPPED: it needs a
city-coordinate dataset we don't have for historical matches, and the plan expects
it insignificant. Only rest/altitude are fit.

Fit: Poisson GLM with the model's log-lambda as a fixed offset, on competitive
(non-friendly) internationals since 2010. Coefficients with |t| < 2 are dropped.
Saved to data/context_coef.json and applied as lambda *= exp(sum b_i * feature_i).

Usage:
  python3 context.py --fit        # fit, report coefficients +/- SE, save json
  python3 context.py --validate   # held-out log-loss on tournament matches
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predictor import (load_matches, compute_elo, fit_goal_model,
                       expected_goals, score_matrix, HOME_ADV, DC_RHO)

HERE = Path(__file__).parent
COEF_FILE = HERE / "data" / "context_coef.json"
REST_CAP = 14          # days; beyond this a team is "fully rested"
REST_DIFF_CLIP = 7     # clip own-minus-opponent rest difference
FIT_SINCE = "2010-01-01"
SPLIT = "2022-01-01"   # held-out boundary for validation
TOURNAMENTS = {"FIFA World Cup", "UEFA Euro", "Copa América",
               "African Cup of Nations", "AFC Asian Cup", "Gold Cup",
               "CONCACAF Championship"}

# Venue altitude in metres. WC2026 venues + the high-altitude football cities that
# carry the historical signal. Everything not listed is treated as lowland (0).
ALT_M = {
    # WC2026 venues
    "Mexico City": 2240, "Zapopan": 1566, "Guadalajara": 1566, "Guadalupe": 540,
    "Monterrey": 540, "Atlanta": 320, "Kansas City": 270, "Denver": 1609,
    "Arlington": 130, "Houston": 12, "Inglewood": 38, "Santa Clara": 8,
    "Seattle": 53, "Vancouver": 0, "Toronto": 76, "East Rutherford": 2,
    "Foxborough": 89, "Philadelphia": 12, "Miami Gardens": 2,
    # historical high-altitude venues (CONMEBOL / CONCACAF / CAF)
    "La Paz": 3640, "El Alto": 4150, "Cochabamba": 2570, "Sucre": 2810,
    "Oruro": 3700, "Quito": 2850, "Ambato": 2577, "Cuenca": 2560,
    "Bogotá": 2640, "Medellín": 1495, "Tunja": 2820, "Pasto": 2527,
    "Cusco": 3400, "Arequipa": 2335, "Juliaca": 3825, "Toluca": 2660,
    "Pachuca": 2400, "Puebla": 2135, "San José": 1170, "Addis Ababa": 2355,
    "Johannesburg": 1753, "Pretoria": 1339, "Bloemfontein": 1395,
    "Asmara": 2325, "Nairobi": 1795, "Sana'a": 2250, "Kabul": 1791,
    "Bishkek": 800, "Almaty": 850,
}


ALT_THRESHOLD_M = 1000   # below this, altitude has no meaningful playing effect

def venue_alt_km(city):
    m = ALT_M.get(str(city), 0)
    return m / 1000.0 if m >= ALT_THRESHOLD_M else 0.0


def _team_home_alt(played):
    """km altitude each team is used to = median of its non-neutral home venues."""
    home = played[~played["neutral"].astype(bool)].copy()
    home["alt"] = home["city"].map(venue_alt_km)
    med = home.groupby("home_team")["alt"].median()
    return med.to_dict()


def _rest_days(played):
    """For each match row, days since each side's previous match (capped)."""
    last = {}
    rest_h = np.empty(len(played))
    rest_a = np.empty(len(played))
    for i, r in enumerate(played[["date", "home_team", "away_team"]]
                          .itertuples(index=False)):
        d, h, a = r
        rest_h[i] = min((d - last[h]).days, REST_CAP) if h in last else REST_CAP
        rest_a[i] = min((d - last[a]).days, REST_CAP) if a in last else REST_CAP
        last[h] = d
        last[a] = d
    return rest_h, rest_a


def build_dataset(played, beta, home_alt, subset=None):
    """Per-side rows: y=goals, offset=log(model lambda), rest_diff, alt_gap."""
    df = played.sort_values("date").reset_index(drop=True)
    rest_h, rest_a = _rest_days(df)
    if subset is not None:
        mask = subset(df).to_numpy()
    else:
        mask = np.ones(len(df), bool)
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        if not mask[i]:
            continue
        adv = 0.0 if r.neutral else HOME_ADV
        lh, la = expected_goals(r.elo_h, r.elo_a, beta, adv)
        valt = venue_alt_km(r.city)
        gap_h = max(0.0, valt - home_alt.get(r.home_team, 0.0))
        gap_a = max(0.0, valt - home_alt.get(r.away_team, 0.0))
        rd = float(np.clip(rest_h[i] - rest_a[i], -REST_DIFF_CLIP, REST_DIFF_CLIP))
        rows.append((r.home_score, lh, rd, gap_h))
        rows.append((r.away_score, la, -rd, gap_a))
    arr = np.array(rows, float)
    return arr[:, 0], arr[:, 1], arr[:, 2:]   # y, lam, X(rest_diff, alt_gap)


def _poisson_fit(y, lam, X):
    """Poisson IRLS with offset=log(lam). Design = [intercept, features].
    Returns coef, SE (same order as columns)."""
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


def fit_context(verbose=True, save=True, fit_mask=None):
    played, _ = load_matches()
    _, played = compute_elo(played)
    beta = fit_goal_model(played)
    home_alt = _team_home_alt(played)
    if fit_mask is None:
        fit_mask = lambda d: (d["date"] >= FIT_SINCE) & (d["tournament"] != "Friendly")
    y, lam, X = build_dataset(played, beta, home_alt, subset=fit_mask)
    b, se = _poisson_fit(y, lam, X)
    names = ["intercept", "rest_diff", "alt_gap"]
    t = b / se
    coef = {}
    if verbose:
        print(f"Poisson context fit ({len(y)} side-observations):")
        print(f"  {'term':10s}{'coef':>10}{'se':>9}{'t':>8}{'kept':>7}")
    for nm, bi, si, ti in zip(names, b, se, t):
        keep = nm != "intercept" and abs(ti) >= 2.0
        if keep:
            coef[nm] = float(bi)
        if verbose:
            print(f"  {nm:10s}{bi:>10.4f}{si:>9.4f}{ti:>8.2f}{('yes' if keep else '—'):>7}")
    if save:
        COEF_FILE.write_text(json.dumps({"coef": coef,
                                         "n": int(len(y)),
                                         "rest_cap": REST_CAP,
                                         "rest_diff_clip": REST_DIFF_CLIP,
                                         "note": "lambda *= exp(sum b_i*feature_i); "
                                                 "intercept not applied"}, indent=2))
        if verbose:
            print(f"  saved kept coefficients -> {COEF_FILE.name}: {coef}")
    return coef


def load_coef():
    if COEF_FILE.exists():
        return json.loads(COEF_FILE.read_text())["coef"]
    return {}


def fixture_features(home, away, date, city, played=None, home_alt=None):
    """(rest_diff, alt_gap_home, alt_gap_away) for a concrete fixture."""
    if played is None:
        played, _ = load_matches()
    if home_alt is None:
        home_alt = _team_home_alt(played)
    date = pd.Timestamp(date)
    prev = played[played["date"] < date]
    def rest(team):
        d = prev[(prev["home_team"] == team) | (prev["away_team"] == team)]["date"]
        return min((date - d.max()).days, REST_CAP) if len(d) else REST_CAP
    rd = float(np.clip(rest(home) - rest(away), -REST_DIFF_CLIP, REST_DIFF_CLIP))
    valt = venue_alt_km(city)
    return (rd, max(0.0, valt - home_alt.get(home, 0.0)),
            max(0.0, valt - home_alt.get(away, 0.0)))


def multipliers(rest_diff, alt_gap_home, alt_gap_away, coef=None):
    """(mult_home, mult_away) lambda multipliers from the fitted coefficients."""
    if coef is None:
        coef = load_coef()
    br, ba = coef.get("rest_diff", 0.0), coef.get("alt_gap", 0.0)
    mh = np.exp(br * rest_diff + ba * alt_gap_home)
    ma = np.exp(br * (-rest_diff) + ba * alt_gap_away)
    return float(mh), float(ma)


def validate(verbose=True):
    """Held-out: fit coef on competitive matches before SPLIT, then compare 1X2
    log-loss with vs without the correction on TOURNAMENT matches after SPLIT."""
    played, _ = load_matches()
    _, played = compute_elo(played)
    beta = fit_goal_model(played)
    home_alt = _team_home_alt(played)
    coef = fit_context(verbose=False, save=False,
                       fit_mask=lambda d: (d["date"] >= FIT_SINCE)
                       & (d["date"] < SPLIT) & (d["tournament"] != "Friendly"))
    df = played.sort_values("date").reset_index(drop=True)
    rest_h, rest_a = _rest_days(df)

    def eval_subset(mask):
        idx = np.where(mask.to_numpy())[0]
        lb = lc = 0.0
        for i in idx:
            r = df.iloc[i]
            adv = 0.0 if r.neutral else HOME_ADV
            lh, la = expected_goals(r.elo_h, r.elo_a, beta, adv)
            valt = venue_alt_km(r.city)
            rd = float(np.clip(rest_h[i] - rest_a[i],
                               -REST_DIFF_CLIP, REST_DIFF_CLIP))
            mh, ma = multipliers(rd, max(0.0, valt - home_alt.get(r.home_team, 0.0)),
                                 max(0.0, valt - home_alt.get(r.away_team, 0.0)), coef)
            y = 0 if r.home_score > r.away_score else (
                1 if r.home_score == r.away_score else 2)
            for tag, (l1, l2) in (("b", (lh, la)), ("c", (lh * mh, la * ma))):
                M = score_matrix(l1, l2, DC_RHO)
                p = [np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()]
                ll = -np.log(max(p[y], 1e-9))
                if tag == "b":
                    lb += ll
                else:
                    lc += ll
        n = len(idx)
        return n, lb / n, lc / n

    post = df["date"] >= SPLIT
    alt_active = post & (df["city"].map(venue_alt_km) > 0) & (df["tournament"] != "Friendly")
    n_t, b_t, c_t = eval_subset(post & df["tournament"].isin(TOURNAMENTS))
    n_a, b_a, c_a = eval_subset(alt_active)
    TOL = 1e-3
    ok = (c_t - b_t <= TOL) and (c_a - b_a <= TOL)
    if verbose:
        print(f"\nHeld-out after {SPLIT}.  fitted coef (pre-{SPLIT}): {coef}")
        print(f"  tournament matches (literal acceptance set), n={n_t}:")
        print(f"    mean log-loss  base {b_t:.4f}  +context {c_t:.4f}  (Δ {c_t-b_t:+.4f})"
              "  — ~flat: post-2022 tournaments are lowland")
        print(f"  altitude-active competitive matches (where the feature applies), "
              f"n={n_a}:")
        print(f"    mean log-loss  base {b_a:.4f}  +context {c_a:.4f}  (Δ {c_a-b_a:+.4f})")
        print(f"  context not worse than baseline (tol {TOL}): {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="WC2026 context features (v2 M6)")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.fit:
        fit_context()
    elif args.validate:
        validate()
    else:
        print("coef:", load_coef() or "(not fitted; run --fit)")


if __name__ == "__main__":
    main()
