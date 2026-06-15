#!/usr/bin/env python3
"""Confederation strength adjustment for World Cup match predictions.

Problem: Elo ratings are accumulated from all international matches.  A
CONCACAF or OFC minnow builds its rating by beating regional opponents that
simply aren't as strong as the UEFA/CONMEBOL teams they'll face at a World
Cup.  The result is systematic over-rating of weaker confederations and
slight under-rating of stronger ones — which translates to inflated underdog
probabilities (e.g. Curaçao ~6% vs Germany instead of ~2%).

Solution: at prediction time, apply a fractional pull toward the WC-field
mean Elo for every participating team:

    adj_i = fraction * (global_mean_elo − conf_mean_elo_i)

Teams from above-average confederations receive a mild boost; below-average
ones are discounted.  The adjustment is applied only to the Elo component of
the blend — Dixon-Coles attack/defense parameters are already disciplined by
the actual match data and need no external correction.

The ``fraction`` hyperparameter is calibrated by backtesting against six
World Cups (2002–2022), with exponential time-weighting (half-life 12 years
≈ 3 WC cycles) so that recent tournaments carry more weight without letting
the most recent one dominate.  The optimal value is saved to
data/conf_adj.json and loaded automatically by predictor.py / edge.py when
the --conf-adj flag is used.

Usage:
  python confederation_adj.py                # show current WC field adjustments
  python confederation_adj.py --backtest     # calibrate & save optimal fraction
  python confederation_adj.py --fraction 0.6 # preview a specific fraction

Integration:
  python predictor.py --conf-adj             # predictions with confederation adj
  python predictor.py "Germany" "Curaçao" --conf-adj
  python edge.py --conf-adj                  # value bets with confederation adj
  python edge.py --conf-adj --squad-adj      # both adjustments together
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CONF_ADJ_FILE = HERE / "data" / "conf_adj.json"

# Backtest time-weighting: half-life of 12 years (3 WC cycles).
# Gives 2022: 1.00, 2018: 0.63, 2014: 0.40, 2010: 0.25, 2006: 0.16, 2002: 0.10
# — old tournaments still inform the estimate but don't swamp recent evidence.
HALFLIFE_YEARS = 12.0
WC_YEARS = [2002, 2006, 2010, 2014, 2018, 2022]
ANCHOR_YEAR = 2026  # current WC year (used for age calculation)

# Fraction search grid: 0.0 = no adjustment, 1.0 = full pull to global mean
FRACTIONS = np.round(np.arange(0.0, 1.025, 0.05), 3)

# Backtest-optimal parameters (calibrated on 2002–2022 WC data, half-life 12 yr):
#   threshold=300, fraction=1.0 → 1.26% log-loss improvement over no adjustment.
# Interpretation: for balanced/moderate matches the Elo model is already
# well-calibrated; the confederation inflation only matters for genuine blowout
# mismatches where one side is ≥300 Elo points above the other.
DEFAULT_THRESHOLD = 300   # Elo gap below which no adjustment is applied
DEFAULT_FRACTION  = 1.0   # when the threshold IS crossed, pull fully to the mean


# ── Confederation membership ──────────────────────────────────────────────────
# Covers all teams that have played in World Cups since 2002 plus 2026
# participants.  Use the exact team names from data/results.csv.

CONF_MAP = {
    # ── UEFA ──────────────────────────────────────────────────────────────────
    "Germany": "UEFA",          "England": "UEFA",
    "France": "UEFA",           "Spain": "UEFA",
    "Portugal": "UEFA",         "Italy": "UEFA",
    "Netherlands": "UEFA",      "Belgium": "UEFA",
    "Croatia": "UEFA",          "Sweden": "UEFA",
    "Denmark": "UEFA",          "Poland": "UEFA",
    "Ukraine": "UEFA",          "Switzerland": "UEFA",
    "Serbia": "UEFA",           "Turkey": "UEFA",
    "Russia": "UEFA",           "Greece": "UEFA",
    "Czech Republic": "UEFA",   "Slovakia": "UEFA",
    "Slovenia": "UEFA",         "Scotland": "UEFA",
    "Wales": "UEFA",            "Austria": "UEFA",
    "Hungary": "UEFA",          "Romania": "UEFA",
    "Norway": "UEFA",           "Finland": "UEFA",
    "Albania": "UEFA",          "North Macedonia": "UEFA",
    "Iceland": "UEFA",          "Montenegro": "UEFA",
    "Georgia": "UEFA",          "Bosnia and Herzegovina": "UEFA",
    "Kosovo": "UEFA",           "Ireland": "UEFA",
    "Republic of Ireland": "UEFA", "Northern Ireland": "UEFA",
    "Azerbaijan": "UEFA",       "Armenia": "UEFA",
    "Moldova": "UEFA",          "Lithuania": "UEFA",
    "Latvia": "UEFA",           "Estonia": "UEFA",
    "Belarus": "UEFA",          "Bulgaria": "UEFA",
    "Cyprus": "UEFA",           "Malta": "UEFA",
    "Liechtenstein": "UEFA",    "Faroe Islands": "UEFA",
    "Gibraltar": "UEFA",        "San Marino": "UEFA",
    "Andorra": "UEFA",          "Luxembourg": "UEFA",
    # Historical names used in the dataset
    "Yugoslavia": "UEFA",       "Serbia and Montenegro": "UEFA",
    "Czechoslovakia": "UEFA",   "Soviet Union": "UEFA",
    "East Germany": "UEFA",     "West Germany": "UEFA",
    "Togo": "CAF",  # (handled below — Togo is CAF, not UEFA; placeholder)

    # ── CONMEBOL ──────────────────────────────────────────────────────────────
    "Brazil": "CONMEBOL",       "Argentina": "CONMEBOL",
    "Colombia": "CONMEBOL",     "Uruguay": "CONMEBOL",
    "Chile": "CONMEBOL",        "Ecuador": "CONMEBOL",
    "Peru": "CONMEBOL",         "Paraguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",    "Bolivia": "CONMEBOL",

    # ── CONCACAF ──────────────────────────────────────────────────────────────
    "Mexico": "CONCACAF",       "United States": "CONCACAF",
    "Canada": "CONCACAF",       "Costa Rica": "CONCACAF",
    "Honduras": "CONCACAF",     "Jamaica": "CONCACAF",
    "Panama": "CONCACAF",       "Trinidad and Tobago": "CONCACAF",
    "El Salvador": "CONCACAF",  "Haiti": "CONCACAF",
    "Curaçao": "CONCACAF",      "Guatemala": "CONCACAF",
    "Cuba": "CONCACAF",         "Suriname": "CONCACAF",
    "Dominican Republic": "CONCACAF", "Barbados": "CONCACAF",
    "Bermuda": "CONCACAF",      "Nicaragua": "CONCACAF",
    "Belize": "CONCACAF",

    # ── CAF ───────────────────────────────────────────────────────────────────
    "Senegal": "CAF",           "Nigeria": "CAF",
    "Ghana": "CAF",             "Cameroon": "CAF",
    "Ivory Coast": "CAF",       "Morocco": "CAF",
    "Tunisia": "CAF",           "Egypt": "CAF",
    "Algeria": "CAF",           "South Africa": "CAF",
    "DR Congo": "CAF",          "Mali": "CAF",
    "Burkina Faso": "CAF",      "Zimbabwe": "CAF",
    "Togo": "CAF",              "Angola": "CAF",
    "Cape Verde": "CAF",        "Equatorial Guinea": "CAF",
    "Benin": "CAF",             "Uganda": "CAF",
    "Kenya": "CAF",             "Tanzania": "CAF",
    "Gabon": "CAF",             "Gambia": "CAF",
    "Ethiopia": "CAF",          "Mozambique": "CAF",
    "Congo": "CAF",             "Sudan": "CAF",
    "Libya": "CAF",             "Guinea": "CAF",
    "Guinea-Bissau": "CAF",     "Mauritania": "CAF",
    "Rwanda": "CAF",            "Zambia": "CAF",
    "Malawi": "CAF",            "Niger": "CAF",
    "Comoros": "CAF",           "Madagascar": "CAF",
    "Sierra Leone": "CAF",      "South Sudan": "CAF",
    "Burundi": "CAF",           "Eritrea": "CAF",
    "Central African Republic": "CAF", "Chad": "CAF",
    "Lesotho": "CAF",           "Eswatini": "CAF",
    "Swaziland": "CAF",         "Namibia": "CAF",
    "Botswana": "CAF",          "Seychelles": "CAF",
    "Mauritius": "CAF",         "Somalia": "CAF",
    "São Tomé and Príncipe": "CAF",

    # ── AFC ───────────────────────────────────────────────────────────────────
    "Japan": "AFC",             "South Korea": "AFC",
    "Iran": "AFC",              "Saudi Arabia": "AFC",
    "Australia": "AFC",         "China": "AFC",
    "Iraq": "AFC",              "Oman": "AFC",
    "Jordan": "AFC",            "Bahrain": "AFC",
    "Qatar": "AFC",             "UAE": "AFC",
    "United Arab Emirates": "AFC", "Uzbekistan": "AFC",
    "Kyrgyzstan": "AFC",        "Syria": "AFC",
    "Palestine": "AFC",         "Tajikistan": "AFC",
    "Vietnam": "AFC",           "Thailand": "AFC",
    "Philippines": "AFC",       "Indonesia": "AFC",
    "Malaysia": "AFC",          "Singapore": "AFC",
    "Hong Kong": "AFC",         "Kuwait": "AFC",
    "Lebanon": "AFC",           "India": "AFC",
    "China PR": "AFC",          "North Korea": "AFC",
    "Myanmar": "AFC",           "Cambodia": "AFC",
    "Nepal": "AFC",             "Maldives": "AFC",
    "Sri Lanka": "AFC",         "Bangladesh": "AFC",
    "Pakistan": "AFC",          "Afghanistan": "AFC",
    "Guam": "AFC",              "Chinese Taipei": "AFC",
    "Turkmenistan": "AFC",      "Kazakhstan": "AFC",
    "Yemen": "AFC",             "Mongolia": "AFC",

    # ── OFC ───────────────────────────────────────────────────────────────────
    "New Zealand": "OFC",       "Tahiti": "OFC",
    "Solomon Islands": "OFC",   "Papua New Guinea": "OFC",
    "Fiji": "OFC",              "Vanuatu": "OFC",
    "New Caledonia": "OFC",     "Cook Islands": "OFC",
    "Samoa": "OFC",             "American Samoa": "OFC",
    "Tonga": "OFC",
}
# Fix the placeholder inserted above
CONF_MAP.pop("Togo", None)
CONF_MAP["Togo"] = "CAF"


# ── Core adjustment logic ─────────────────────────────────────────────────────

def conf_adjustments(elos, wc_teams, fraction):
    """Compute per-team Elo adjustments for a given WC field and fraction.

    Parameters
    ----------
    elos : dict  team -> current Elo rating
    wc_teams : iterable of team names participating in this WC
    fraction : float  0 = no adjustment, 1 = full pull to global mean

    Returns
    -------
    adjs : dict  team -> Elo delta (positive = boost, negative = discount)
    global_mean : float  mean Elo of the WC field
    conf_means : dict  confederation -> mean Elo of its WC participants

    Note: These are the *potential* per-team adjustments.  Whether each
    adjustment is actually applied depends on the match-level Elo gap —
    see apply_match_adj().
    """
    known = {t: elos[t] for t in wc_teams
             if t in elos and CONF_MAP.get(t)}

    conf_elos = defaultdict(list)
    for team, elo in known.items():
        conf_elos[CONF_MAP[team]].append(elo)

    conf_means = {c: float(np.mean(v)) for c, v in conf_elos.items()}
    global_mean = float(np.mean(list(known.values()))) if known else 1500.0

    adjs = {}
    for team in wc_teams:
        conf = CONF_MAP.get(team)
        if conf and conf in conf_means:
            adjs[team] = fraction * (global_mean - conf_means[conf])
        else:
            adjs[team] = 0.0

    return adjs, global_mean, conf_means


def apply_match_adj(elo_h, elo_a, adj_h, adj_a, threshold):
    """Return adjusted Elo pair, gated by the pre-adjustment Elo gap.

    If |elo_h - elo_a| < threshold the adjustments are suppressed entirely:
    the model is well-calibrated for close matches, and applying the
    confederation discount would hurt predictions there.  For genuine
    mismatches (gap >= threshold) the full adjustments are applied.

    Backtest result (2002–2022 WCs, half-life 12 yr):
        threshold=300, fraction=1.0 → 1.26% log-loss improvement.
    """
    if abs(elo_h - elo_a) >= threshold:
        return elo_h + adj_h, elo_a + adj_a
    return elo_h, elo_a


def apply_adjustments(ratings, adjs):
    """Return a copy of *ratings* with per-team *adjs* applied."""
    adj = dict(ratings)
    for team, delta in adjs.items():
        if team in adj:
            adj[team] = adj[team] + delta
    return adj


def load_params():
    """Load calibrated (fraction, threshold) from data/conf_adj.json."""
    if CONF_ADJ_FILE.exists():
        d = json.loads(CONF_ADJ_FILE.read_text())
        return float(d.get("fraction", DEFAULT_FRACTION)), \
               int(d.get("threshold", DEFAULT_THRESHOLD))
    return DEFAULT_FRACTION, DEFAULT_THRESHOLD


def load_optimal_fraction():
    """Backwards-compatible: return fraction only."""
    return load_params()[0]


def save_result(fraction, threshold, meta=None):
    d = {"fraction": round(float(fraction), 4),
         "threshold": int(threshold)}
    if meta:
        d.update(meta)
    CONF_ADJ_FILE.write_text(json.dumps(d, indent=2))


def _wc_teams_2026(played, upcoming):
    """All 48 WC 2026 participants from played + upcoming fixtures."""
    p = played[(played["tournament"] == "FIFA World Cup") &
               (played["date"].dt.year == 2026)]
    u = upcoming[upcoming["tournament"] == "FIFA World Cup"]
    return (set(p["home_team"]) | set(p["away_team"]) |
            set(u["home_team"]) | set(u["away_team"]))


def build_conf_adj_sources(model="blend", fraction=None, threshold=None):
    """Like dixoncoles.build_sources but with threshold-gated confederation adj.

    The Elo component uses a match-aware lambda that applies the confederation
    discount only when the two teams' Elo gap >= threshold (default 300).
    Dixon-Coles attack/defense parameters are unchanged.

    Returns (sources, base_ratings, conf_means, adjs)
    where base_ratings are the UNADJUSTED Elo ratings (adjustments are applied
    per-match inside the lambda, not baked into the ratings dict).
    """
    from predictor import (load_matches, compute_elo, fit_goal_model,
                           expected_goals, HOME_ADV, DC_RHO)

    _frac, _thr = load_params()
    if fraction is None:
        fraction = _frac
    if threshold is None:
        threshold = _thr

    played, upcoming = load_matches()
    ratings, played = compute_elo(played)
    wc_teams = _wc_teams_2026(played, upcoming)

    adjs, global_mean, conf_means = conf_adjustments(ratings, wc_teams, fraction)

    sources = []
    if model in ("elo", "blend"):
        beta = fit_goal_model(played)

        def _elo_fn(t1, t2, h1=0.0, h2=0.0,
                    _r=ratings, _a=adjs, _b=beta,
                    _ha=HOME_ADV, _thr=threshold):
            eh, ea = apply_match_adj(
                _r.get(t1, 1500.0), _r.get(t2, 1500.0),
                _a.get(t1, 0.0),    _a.get(t2, 0.0),
                _thr)
            return expected_goals(eh, ea, _b, (h1 - h2) * _ha)

        sources.append((_elo_fn, DC_RHO))

    if model in ("dc", "blend"):
        from dixoncoles import DCModel
        dc = DCModel.load_or_fit()
        sources.append((dc.lambdas, dc.rho))

    if not sources:
        sys.exit(f"Unknown model {model!r}: use elo, dc, or blend.")

    return sources, ratings, conf_means, adjs


# ── Backtest ──────────────────────────────────────────────────────────────────

def _year_weight(year):
    """Exponential time weight anchored to ANCHOR_YEAR."""
    age = ANCHOR_YEAR - year
    return math.exp(-math.log(2) / HALFLIFE_YEARS * age)


def backtest(played, beta, verbose=True):
    """Sweep fraction over 2002–2022 WCs, return optimal fraction.

    For each WC year:
      • Pre-tournament Elo taken from each team's first WC match (elo_h/elo_a).
      • Confederation adjustment computed from those pre-tournament Elos.
      • Adjustment applied as a constant offset throughout the tournament;
        the point-in-time elo_h/elo_a (which update after each match) remain
        the baseline so mid-tournament form is still captured.
      • Log-loss and Brier score computed across all group-stage + knockout
        matches, then combined with exponential time-weighting.

    Only the Elo model is used here (not the DC blend) to avoid look-ahead
    bias from the DC fit on post-WC data.
    """
    from predictor import expected_goals, score_matrix, HOME_ADV, DC_RHO

    wc_df = played[played["tournament"] == "FIFA World Cup"].copy()
    detail_rows = []

    for year in WC_YEARS:
        yr = wc_df[wc_df["date"].dt.year == year].copy()
        if yr.empty:
            if verbose:
                print(f"  {year}: no data, skipping")
            continue

        wc_teams = set(yr["home_team"]) | set(yr["away_team"])

        # Pre-tournament Elo: first appearance of each team in this WC
        pre_elo = {}
        for row in yr.sort_values("date").itertuples(index=False):
            h, a = row.home_team, row.away_team
            if h not in pre_elo:
                pre_elo[h] = row.elo_h
            if a not in pre_elo:
                pre_elo[a] = row.elo_a

        w_year = _year_weight(year)

        for frac in FRACTIONS:
            adjs, _, _ = conf_adjustments(pre_elo, wc_teams, float(frac))

            ll = brier = 0.0
            n = 0
            for row in yr.itertuples(index=False):
                h, a = row.home_team, row.away_team
                # Point-in-time pre-match Elo + static pre-tournament conf adj
                elo_h = row.elo_h + adjs.get(h, 0.0)
                elo_a = row.elo_a + adjs.get(a, 0.0)
                home_adv = 0.0 if row.neutral else HOME_ADV
                lam1, lam2 = expected_goals(elo_h, elo_a, beta, home_adv)
                M = score_matrix(lam1, lam2, DC_RHO)
                pw = float(np.tril(M, -1).sum())
                pd_ = float(np.trace(M))
                pl = float(np.triu(M, 1).sum())

                hs, as_ = int(row.home_score), int(row.away_score)
                if hs > as_:
                    actual = np.array([1.0, 0.0, 0.0])
                elif hs == as_:
                    actual = np.array([0.0, 1.0, 0.0])
                else:
                    actual = np.array([0.0, 0.0, 1.0])

                probs = np.array([pw, pd_, pl])
                probs_clipped = np.clip(probs, 1e-9, 1.0)
                ll += -float(np.dot(actual, np.log(probs_clipped)))
                brier += float(np.sum((probs - actual) ** 2))
                n += 1

            detail_rows.append({
                "year": year,
                "fraction": float(frac),
                "weight": w_year,
                "log_loss": ll / n if n else np.nan,
                "brier": brier / n if n else np.nan,
                "n": n,
            })

    detail_df = pd.DataFrame(detail_rows)

    # Time-weighted aggregate per fraction
    summary_rows = []
    for frac in FRACTIONS:
        sub = detail_df[np.isclose(detail_df["fraction"], frac)]
        if sub.empty or sub["log_loss"].isna().all():
            continue
        tw = sub["weight"].sum()
        w_ll = float((sub["log_loss"] * sub["weight"]).sum() / tw)
        w_br = float((sub["brier"] * sub["weight"]).sum() / tw)
        summary_rows.append({
            "fraction": float(frac),
            "weighted_log_loss": w_ll,
            "weighted_brier": w_br,
        })

    summary_df = pd.DataFrame(summary_rows)
    best_idx = summary_df["weighted_log_loss"].idxmin()
    optimal = float(summary_df.loc[best_idx, "fraction"])

    if verbose:
        weights_str = "  ".join(
            f"{y}={_year_weight(y):.2f}" for y in WC_YEARS)
        print(f"\n{'─'*64}")
        print(f"Backtest: {len(WC_YEARS)} World Cups  |  "
              f"half-life {HALFLIFE_YEARS:.0f} yr")
        print(f"Weights: {weights_str}")
        print(f"{'─'*64}")
        print(f"\n  {'frac':>6}  {'Wtd log-loss':>14}  {'Wtd Brier':>11}")
        print(f"  {'─'*6}  {'─'*14}  {'─'*11}")
        for _, row in summary_df.iterrows():
            marker = "  ◄ optimal" if abs(row["fraction"] - optimal) < 1e-6 else ""
            print(f"  {row['fraction']:>6.2f}  "
                  f"{row['weighted_log_loss']:>14.5f}  "
                  f"{row['weighted_brier']:>11.5f}{marker}")
        print()

        # Per-year breakdown at optimal fraction
        yr_rows = detail_df[np.isclose(detail_df["fraction"], optimal)]
        print(f"Per-tournament breakdown at fraction={optimal:.2f}:\n")
        print(f"  {'Year':>6}  {'Weight':>8}  {'Matches':>8}  "
              f"{'Log-loss':>10}  {'Brier':>8}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}")
        for _, row in yr_rows.sort_values("year").iterrows():
            print(f"  {int(row['year']):>6}  {row['weight']:>8.3f}  "
                  f"{int(row['n']):>8}  "
                  f"{row['log_loss']:>10.5f}  {row['brier']:>8.5f}")
        print()

    return optimal, summary_df, detail_df


# ── Report ────────────────────────────────────────────────────────────────────

def cmd_report(fraction=None):
    from predictor import load_matches, compute_elo

    if fraction is None:
        fraction = load_optimal_fraction()

    played, upcoming = load_matches()
    ratings, _ = compute_elo(played)
    wc_teams = _wc_teams_2026(played, upcoming)

    if not wc_teams:
        print("No WC 2026 fixtures found in upcoming data.")
        return

    adjs, global_mean, conf_means = conf_adjustments(ratings, wc_teams,
                                                      fraction)

    print(f"\nConfederation adjustments  "
          f"(fraction={fraction:.2f}, WC-field mean Elo={global_mean:.0f})\n")
    print(f"  {'Confederation':15} {'Mean Elo':>10} {'Gap':>8} "
          f"{'Adj':>8} {'Teams':>6}")
    print(f"  {'─'*15} {'─'*10} {'─'*8} {'─'*8} {'─'*6}")
    for conf, mean in sorted(conf_means.items(), key=lambda kv: -kv[1]):
        gap = global_mean - mean
        adj = fraction * gap
        n = sum(1 for t in wc_teams if CONF_MAP.get(t) == conf)
        print(f"  {conf:<15} {mean:>10.0f} {gap:>+8.0f} "
              f"{adj:>+8.0f} {n:>6}")

    print(f"\nPer-team adjustments (|adj| > 3 Elo points):\n")
    print(f"  {'Team':28} {'Conf':10} {'Base Elo':>10} "
          f"{'Adj':>8} {'Adj Elo':>10}")
    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")
    sig = sorted(
        [(t, adjs.get(t, 0.0)) for t in wc_teams if abs(adjs.get(t, 0.0)) > 3],
        key=lambda ta: ta[1])
    for team, adj in sig:
        conf = CONF_MAP.get(team, "?")
        base = ratings.get(team, 0.0)
        print(f"  {team:<28} {conf:<10} {base:>10.0f} "
              f"{adj:>+8.0f} {base + adj:>10.0f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Confederation Elo adjustment for WC predictions")
    ap.add_argument("--backtest", action="store_true",
                    help="calibrate fraction from 2002–2022 WCs and save result")
    ap.add_argument("--fraction", type=float, default=None,
                    help="override fraction (default: load from data/conf_adj.json)")
    args = ap.parse_args()

    if args.backtest:
        from predictor import (load_matches, compute_elo, fit_goal_model,
                               expected_goals as _eg, score_matrix as _sm,
                               HOME_ADV, DC_RHO)
        print("Loading historical match data …")
        played, _ = load_matches()
        ratings, played = compute_elo(played)
        beta = fit_goal_model(played)
        print(f"Fitting goal model on {len(played)} matches …\n")

        # Phase 1: sanity check — universal fraction sweep
        print("Phase 1: universal fraction sweep (no threshold) …")
        backtest(played, beta, verbose=True)

        # Phase 2: joint (fraction × threshold) grid search
        print("\nPhase 2: fraction × threshold grid search …\n")
        thresholds = [0, 150, 200, 250, 300, 350, 400]
        fracs2 = np.round(np.arange(0.0, 1.05, 0.25), 2)
        wc_df = played[played["tournament"] == "FIFA World Cup"].copy()

        best_ll2 = 9e9
        best_thr, best_frac2 = DEFAULT_THRESHOLD, DEFAULT_FRACTION
        grid_rows = []

        for thr in thresholds:
            for frac2 in fracs2:
                rows2 = []
                for year in WC_YEARS:
                    yr2 = wc_df[wc_df["date"].dt.year == year]
                    if yr2.empty:
                        continue
                    wt2 = set(yr2["home_team"]) | set(yr2["away_team"])
                    pe2 = {}
                    for _r in yr2.sort_values("date").itertuples(index=False):
                        if _r.home_team not in pe2:
                            pe2[_r.home_team] = _r.elo_h
                        if _r.away_team not in pe2:
                            pe2[_r.away_team] = _r.elo_a
                    _adjs2, _, _ = conf_adjustments(pe2, wt2, float(frac2))
                    w_y2 = _year_weight(year)
                    ll2 = br2 = n2 = 0
                    for _r in yr2.itertuples(index=False):
                        eh2, ea2 = apply_match_adj(
                            _r.elo_h, _r.elo_a,
                            _adjs2.get(_r.home_team, 0.0),
                            _adjs2.get(_r.away_team, 0.0),
                            thr)
                        ha2 = 0.0 if _r.neutral else HOME_ADV
                        l1, l2 = _eg(eh2, ea2, beta, ha2)
                        M2 = _sm(l1, l2, DC_RHO)
                        pw2 = float(np.tril(M2, -1).sum())
                        pd2 = float(np.trace(M2))
                        pl2 = float(np.triu(M2, 1).sum())
                        hs2 = int(_r.home_score)
                        as2 = int(_r.away_score)
                        act2 = np.array(
                            [1, 0, 0] if hs2 > as2
                            else ([0, 1, 0] if hs2 == as2 else [0, 0, 1]),
                            float)
                        probs2 = np.array([pw2, pd2, pl2])
                        ll2 += -float(np.dot(act2,
                                             np.log(np.clip(probs2, 1e-9, 1.0))))
                        br2 += float(np.sum((probs2 - act2) ** 2))
                        n2 += 1
                    if n2:
                        rows2.append({"w": w_y2,
                                      "ll": ll2 / n2, "br": br2 / n2})
                if rows2:
                    df2 = pd.DataFrame(rows2)
                    tw2 = df2["w"].sum()
                    wll2 = float((df2["ll"] * df2["w"]).sum() / tw2)
                    wbr2 = float((df2["br"] * df2["w"]).sum() / tw2)
                    grid_rows.append({"thr": thr, "frac": frac2,
                                      "wll": wll2, "wbr": wbr2})
                    if wll2 < best_ll2:
                        best_ll2 = wll2
                        best_thr  = thr
                        best_frac2 = float(frac2)

        grid_df = pd.DataFrame(grid_rows).sort_values("wll")
        print(f"  {'thresh':>8}  {'frac':>6}  {'Wtd LL':>12}  {'Wtd Brier':>12}")
        for _, _gr in grid_df.head(12).iterrows():
            marker = (" ◄ optimal"
                      if _gr["thr"] == best_thr
                      and abs(_gr["frac"] - best_frac2) < 1e-6 else "")
            print(f"  {int(_gr['thr']):>8}  {_gr['frac']:>6.2f}  "
                  f"{_gr['wll']:>12.5f}  {_gr['wbr']:>12.5f}{marker}")

        base_ll = float(grid_df.loc[
            (grid_df["thr"] == 0) & (np.isclose(grid_df["frac"], 0.0)),
            "wll"].iloc[0])
        improvement = (base_ll - best_ll2) / base_ll * 100
        print(f"\n  Baseline (no adj): {base_ll:.5f}")
        print(f"  Best: threshold={best_thr}, fraction={best_frac2:.2f}, "
              f"wll={best_ll2:.5f}  ({improvement:.2f}% improvement)\n")

        save_result(best_frac2, best_thr, {
            "halflife_years": HALFLIFE_YEARS,
            "wc_years": WC_YEARS,
            "baseline_wll": round(base_ll, 6),
            "best_wll": round(best_ll2, 6),
            "improvement_pct": round(improvement, 3),
            "note": (
                "Adjustment applied only when pre-match Elo gap >= threshold. "
                "For balanced games the Elo model is well-calibrated without "
                "confederation correction; the bias only matters for genuine "
                "blowout mismatches."),
        })
        print(f"Saved threshold={best_thr}, fraction={best_frac2:.2f} "
              f"→ {CONF_ADJ_FILE.name}\n")

    cmd_report(args.fraction)


if __name__ == "__main__":
    main()
