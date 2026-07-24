#!/usr/bin/env python3
"""P4 — hierarchical competition-strength estimation.

Why the incumbent estimator is not trustworthy
----------------------------------------------
`model.fit_comp_strength` takes each league's mean end-of-fit Elo and min-max
rescales it onto [0.15, 1.10]. Its own output is the argument against it:

    Bundesliga 0.5451   Serie A 0.6235   La Liga 0.5835
    Scottish Premiership 0.6979

It ranks three of Europe's strongest leagues below the Scottish Premiership,
which is why the artifact has sat at `"active": false`. Three causes:

1. **Min-max sets the scale from the two extremes.** One noisy league at either
   end rescales every other league.
2. **No weighting by evidence.** A league connected to the rest of football by
   200 matches is treated exactly like one connected by two.
3. **Mean Elo is only comparable across leagues via inter-league matches.**
   Within a league Elo is close to zero-sum, so a league's absolute level moves
   only when its clubs play outsiders. A league whose clubs rarely do sits near
   BASE_ELO by construction — not because it is average, but because nothing
   has ever measured it.

Measured connectivity, post-P3 (matches against clubs from another league):

    Premier League 222 · Serie A 172 · La Liga 167 · Bundesliga 157
    Eredivisie 95 · Liga Portugal 109 · Austrian Bundesliga 43
    Scottish Championship 2 · Ekstraklasa 2 · League of Ireland 4
    Scottish League One/Two, Liga 3, Parva Liga, Veikkausliiga,
    Russian Premier League: ZERO

Fitting a free parameter per league from that is fitting noise for most of the
table, and pure fabrication for the leagues with no links at all.

The approach here
-----------------
Estimate one strength per league, then shrink it toward the UEFA
country-coefficient prior (uefa_registry, P2) with weight set by how much
inter-league evidence that league actually has:

    strength = (n_eff * observed + K * prior) / (n_eff + K)

Well-connected leagues move to their data. Weakly-connected leagues stay near
an external estimate that is at least defensible. Leagues with zero links get
the prior exactly, which is the honest answer — we have measured nothing.

This is the same Empirical-Bayes idiom `model.py` already uses for per-league
home advantage (LEAGUE_HFA_SHRINK_K).

Both components must live on the SAME scale for the blend to mean anything, so
observed Elo is mapped to the strength scale using the anchors P2 already
established (Premier League 1.00, Scottish Premiership 0.58) rather than a
min-max rescale, which would make the two incomparable.

Known limitation — the two components measure different things
--------------------------------------------------------------
The fitted table makes this visible. For most leagues observed ≈ prior, but for
some the gap is large and one-directional:

    Eredivisie      observed 0.495   prior 0.760
    Liga Portugal   observed 0.527   prior 0.723
    Scottish Prem   observed 0.580   prior 0.580
    Danish Superliga observed 0.551  prior 0.572

The pattern is not noise. A UEFA country coefficient is earned by the three or
four clubs a nation sends to Europe; mean league Elo describes the whole
division. Top-heavy leagues — Eredivisie, Liga Portugal, and to a degree Serie
A and La Liga — therefore score high on coefficient and lower on mean Elo,
while flat leagues (Scotland, Denmark, Sweden, Poland) agree closely.

So the prior is biased upward for top-heavy leagues, and shrinking toward it
inherits that bias. Two consequences, both deliberate:

  * the data-estimated K (10.5) is low, so well-connected leagues largely
    follow their own evidence and the bias mostly affects leagues we have
    little evidence about anyway;
  * this is a further reason the artifact stays gated. A cleaner fix would
    weight the prior by each league's European *participants* rather than the
    league as a whole, which is a P5+ refinement, not a blocker here.

Output stays `"active": false`. Promotion is a separate, deliberate decision.

CLI:
  python3 -m club_soccer.league_strength                 # fit + report
  python3 -m club_soccer.league_strength --tune-k        # grid-search K
  python3 -m club_soccer.league_strength --write
"""
from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import model as M
from . import uefa_registry as R
from .competitions import COMPETITIONS, get as comp_get

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
COMP_STRENGTH = DATA / "comp_strength.json"

# Shrinkage weight, in units of "inter-league matches": a league needs ~K
# cross-league matches before its own data outweighs the prior. NOT hand-set —
# estimated from the data by variance decomposition (see estimate_k), which
# returned 10.5 over 22 leagues. Re-estimate after any large data change.
DEFAULT_K = 10.5

# Anchors shared with uefa_registry.strength_prior so observed and prior land
# on one scale.
ANCHOR_HIGH = ("Premier League", 1.00)
ANCHOR_LOW = ("Scottish Premiership", 0.58)

MIN_MATCHES_IN_LEAGUE = 6      # before a team counts toward its league's mean

# Domestic cups inherit their parent league, discounted: a cup field is a
# mixture of tiers, so it is weaker than its top division.
CUP_PARENT = {
    "FA Cup": "Premier League", "EFL Cup": "Premier League",
    "Scottish Cup": "Scottish Premiership",
    "Scottish League Cup": "Scottish Premiership",
    "DFB-Pokal": "Bundesliga", "Coppa Italia": "Serie A",
    "Coupe de France": "Ligue 1", "Copa del Rey": "La Liga",
}
CUP_DISCOUNT = 0.95


def primary_league(df: pd.DataFrame) -> dict[tuple[str, int], str]:
    """The league a club mainly played in, per season."""
    counts: collections.Counter = collections.Counter()
    for comp, season, home, away in zip(df["competition"], df["season"],
                                        df["home"], df["away"]):
        c = comp_get(comp)
        if not c or c.kind != "league":
            continue
        try:
            year = int(season)
        except (TypeError, ValueError):
            continue
        counts[(home, year, comp)] += 1
        counts[(away, year, comp)] += 1
    best: dict[tuple[str, int], tuple[str, int]] = {}
    for (team, year, comp), n in counts.items():
        if best.get((team, year), ("", 0))[1] < n:
            best[(team, year)] = (comp, n)
    return {k: v[0] for k, v in best.items()}


def connectivity(df: pd.DataFrame) -> dict[str, int]:
    """Inter-league matches per league — the evidence that identifies scale.

    A match counts for league L when one side's primary league is L and the
    opponent's is not. The opponent's league may be unknown (clubs from
    associations we cannot source), which still counts: the match is evidence
    about L's clubs against outsiders, even if we cannot attribute it to a
    specific other league.
    """
    prim = primary_league(df)
    out: collections.Counter = collections.Counter()
    for comp, season, home, away in zip(df["competition"], df["season"],
                                        df["home"], df["away"]):
        try:
            year = int(season)
        except (TypeError, ValueError):
            continue
        lh = prim.get((home, year))
        la = prim.get((away, year))
        if lh and lh != la:
            out[lh] += 1
        if la and la != lh:
            out[la] += 1
    return dict(out)


def observed_strength(df: pd.DataFrame, params: dict | None = None,
                      season: int | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """Per-league mean Elo, and that mean mapped onto the strength scale."""
    params = M.fit(df) if params is None else params
    elo = params["elo"]
    if season is None:
        today = datetime.now(timezone.utc).date()
        current = today.year if today.month >= 7 else today.year - 1
        season = current - 1

    mean_elo: dict[str, float] = {}
    for comp in COMPETITIONS:
        if comp.kind != "league":
            continue
        sub = df[(df["competition"] == comp.name) & (df["season"] == season)]
        if sub.empty:
            continue
        counts = pd.concat([sub["home"], sub["away"]]).value_counts()
        teams = [t for t, n in counts.items() if n >= MIN_MATCHES_IN_LEAGUE]
        if not teams:
            continue
        mean_elo[comp.name] = float(np.mean([elo.get(t, M.BASE_ELO) for t in teams]))

    # Anchored linear map, NOT min-max: the scale must not be hostage to
    # whichever leagues happen to sit at the extremes this season, and it must
    # match the scale uefa_registry.strength_prior uses.
    hi_name, hi_val = ANCHOR_HIGH
    lo_name, lo_val = ANCHOR_LOW
    if hi_name in mean_elo and lo_name in mean_elo and \
            abs(mean_elo[hi_name] - mean_elo[lo_name]) > 1e-6:
        slope = (hi_val - lo_val) / (mean_elo[hi_name] - mean_elo[lo_name])
        intercept = hi_val - slope * mean_elo[hi_name]
    else:
        # Fall back to a fixed Elo-per-strength-point scale rather than
        # silently rescaling to whatever range exists.
        slope, intercept = 1.0 / 700.0, 1.00 - M.BASE_ELO / 700.0

    observed = {name: float(np.clip(slope * e + intercept, 0.10, 1.15))
                for name, e in mean_elo.items()}
    return mean_elo, observed


def prior_strength(comp) -> float:
    """UEFA-coefficient prior for a competition, via uefa_registry."""
    return R.strength_prior(comp.country, comp.tier or 1)


# A default (non-coefficient) prior carries no information, so it gets a much
# smaller shrinkage weight — a league resting on it should follow its own
# observed data rather than be anchored to 0.75. Empirical Bayes: prior weight
# reflects prior confidence, and an uninformative prior has near-zero
# confidence. K_DEFAULT is in the same "equivalent inter-league matches" unit
# as K; 3 lets even a lightly-connected league outvote the placeholder.
K_DEFAULT_PRIOR = 3.0


def _effective_k(comp, k: float) -> float:
    return k if R.prior_is_informative(comp.country) else K_DEFAULT_PRIOR


def fit(df: pd.DataFrame | None = None, k: float = DEFAULT_K,
        params: dict | None = None, season: int | None = None) -> dict:
    """Hierarchical per-competition strength."""
    df = M.played(M.load_fixtures()) if df is None else df
    mean_elo, observed = observed_strength(df, params=params, season=season)
    n_eff = connectivity(df)

    rows: list[dict] = []
    strengths: dict[str, float] = {}
    for comp in COMPETITIONS:
        if comp.kind != "league":
            continue
        prior = prior_strength(comp)
        obs = observed.get(comp.name)
        n = float(n_eff.get(comp.name, 0))
        k_eff = _effective_k(comp, k)
        if obs is None:
            # No current-season data at all: the prior is all we have.
            value, weight = prior, 0.0
        else:
            weight = n / (n + k_eff)
            value = weight * obs + (1.0 - weight) * prior
        strengths[comp.name] = round(float(np.clip(value, 0.10, 1.15)), 4)
        rows.append({
            "competition": comp.name, "country": comp.country, "tier": comp.tier,
            "n_inter_league": int(n), "shrink_weight": round(weight, 3),
            "mean_elo": round(mean_elo.get(comp.name, float("nan")), 1)
            if comp.name in mean_elo else None,
            "observed": None if obs is None else round(obs, 4),
            "prior": round(prior, 4), "fitted": strengths[comp.name],
            "hand_set": comp.strength,
        })

    for comp in COMPETITIONS:
        if comp.kind == "cup":
            parent = CUP_PARENT.get(comp.name)
            if parent and parent in strengths:
                strengths[comp.name] = round(CUP_DISCOUNT * strengths[parent], 4)

    return {"strengths": strengths, "rows": rows, "k": k}


# ── plausibility gate ─────────────────────────────────────────────────────
# Preregistered before looking at any fitted output: the incumbent estimator
# fails these, and any replacement that also fails them is not promotable.
TOP_LEAGUES = ("Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1")


def plausibility(strengths: dict[str, float]) -> tuple[bool, list[str]]:
    """Ordinal sanity checks. Necessary, not sufficient, for promotion."""
    failures: list[str] = []
    scot = strengths.get("Scottish Premiership")
    if scot is not None:
        for name in TOP_LEAGUES:
            value = strengths.get(name)
            if value is not None and value <= scot:
                failures.append(
                    f"{name} ({value:.4f}) <= Scottish Premiership ({scot:.4f})")
    for top, second in (("Premier League", "Championship"),
                        ("Serie A", "Serie B"),
                        ("La Liga", "Segunda División"),
                        ("Bundesliga", "2. Bundesliga"),
                        ("Ligue 1", "Ligue 2")):
        a, b = strengths.get(top), strengths.get(second)
        if a is not None and b is not None and b >= a:
            failures.append(f"{second} ({b:.4f}) >= {top} ({a:.4f})")
    pl = strengths.get("Premier League")
    if pl is not None and pl < 0.85:
        failures.append(f"Premier League ({pl:.4f}) implausibly low")
    return (not failures), failures


def estimate_k(verbose: bool = True) -> dict:
    """Estimate K from the data by variance decomposition (Empirical Bayes).

    K is not a free knob to be grid-searched against a convenient objective. In
    the shrinkage form used here it has a definition:

        K = (within-league sampling variance per match) / (between-league true variance)

    i.e. how noisy one league's estimate is, relative to how much leagues
    genuinely differ. Both terms are measurable.

    Method: split the seasons into two interleaved halves, estimate each
    league's strength independently in each, and read the sampling noise off
    the disagreement between them.

        Var(obs_A - obs_B) = 4 * sigma^2 / n     (each half sees ~n/2 matches)
        tau^2              = Var(obs across leagues) - mean(sigma^2 / n)
        K                  = sigma^2 / tau^2

    Caveat, stated because it biases the answer: alternate-season splitting
    treats genuine year-to-year movement in a league's strength as if it were
    sampling noise, which inflates sigma^2 and so inflates K. That errs toward
    MORE shrinkage — the safe direction for an estimator whose failure mode is
    over-trusting weakly-evidenced leagues.

    An earlier version grid-searched K against "let strong leagues follow their
    data, keep weak leagues near the prior". That objective is monotone in K,
    so it always returned the smallest value on the grid — it was measuring the
    grid's edge, not the data.
    """
    df = M.played(M.load_fixtures())
    seasons = sorted({int(s) for s in df["season"].dropna()})
    if len(seasons) < 4:
        if verbose:
            print("too few seasons to split; falling back to DEFAULT_K")
        return {"k": DEFAULT_K, "method": "fallback (insufficient seasons)"}

    half_a = set(seasons[0::2])
    half_b = set(seasons[1::2])
    df_a = df[df["season"].isin(half_a)]
    df_b = df[df["season"].isin(half_b)]

    def _obs(sub: pd.DataFrame) -> dict[str, float]:
        merged: dict[str, float] = {}
        for season in sorted({int(s) for s in sub["season"].dropna()}):
            try:
                _, obs = observed_strength(sub, params=M.fit(sub), season=season)
            except Exception:
                continue
            for name, value in obs.items():
                merged.setdefault(name, []).append(value) if isinstance(
                    merged.get(name), list) else merged.setdefault(name, [value])
        return {k_: float(np.mean(v)) if isinstance(v, list) else float(v)
                for k_, v in merged.items()}

    obs_a, obs_b = _obs(df_a), _obs(df_b)
    n_eff = connectivity(df)

    shared = [name for name in obs_a
              if name in obs_b and n_eff.get(name, 0) >= 20]
    if len(shared) < 5:
        if verbose:
            print(f"only {len(shared)} leagues estimable in both halves; "
                  "falling back to DEFAULT_K")
        return {"k": DEFAULT_K, "method": "fallback (too few shared leagues)"}

    sigma2_samples = []
    for name in shared:
        d = obs_a[name] - obs_b[name]
        sigma2_samples.append(d * d * n_eff[name] / 4.0)
    sigma2 = float(np.mean(sigma2_samples))

    all_obs = np.array([obs_a[name] for name in shared])
    total_var = float(np.var(all_obs, ddof=1))
    mean_noise = float(np.mean([sigma2 / n_eff[name] for name in shared]))
    tau2 = max(total_var - mean_noise, 1e-6)

    k = float(np.clip(sigma2 / tau2, 1.0, 500.0))
    if verbose:
        print(f"leagues used            : {len(shared)}")
        print(f"sigma^2 (per-match noise): {sigma2:.5f}")
        print(f"tau^2  (between-league)  : {tau2:.5f}")
        print(f"  total between-variance : {total_var:.5f}")
        print(f"  noise component        : {mean_noise:.5f}")
        print(f"\nestimated K = sigma^2 / tau^2 = {k:.1f}")
    return {"k": round(k, 1), "method": "empirical-bayes split-half",
            "sigma2": sigma2, "tau2": tau2, "n_leagues": len(shared),
            "leagues": sorted(shared)}


def sensitivity(grid=(5, 10, 20, 40, 80, 160, 320), verbose: bool = True) -> dict:
    """How the fitted table moves with K — a diagnostic, not a selector."""
    df = M.played(M.load_fixtures())
    params = M.fit(df)
    results = []
    for k in grid:
        out = fit(df, k=float(k), params=params)
        ok, failures = plausibility(out["strengths"])
        weak = [r for r in out["rows"] if r["n_inter_league"] < 10]
        drift = float(np.mean([abs(r["fitted"] - r["prior"]) for r in weak])) if weak else 0.0
        strong = [r for r in out["rows"] if r["n_inter_league"] >= 100]
        follow = float(np.mean([r["shrink_weight"] for r in strong])) if strong else 0.0
        results.append({"k": k, "plausible": ok, "n_failures": len(failures),
                        "weak_league_drift": round(drift, 4),
                        "strong_league_data_weight": round(follow, 3)})
        if verbose:
            flag = "OK " if ok else f"FAIL({len(failures)})"
            print(f"  k={k:>4}  {flag}  weak-league drift {drift:.4f}  "
                  f"strong-league data weight {follow:.3f}")
    return {"grid": results}


def save(result: dict, k: float, verbose: bool = True) -> dict:
    ok, failures = plausibility(result["strengths"])
    payload = dict(result["strengths"])
    payload.update({
        "active": False,
        "_method": "hierarchical shrinkage to UEFA country coefficient",
        "_k": k,
        "_fit_at_utc": datetime.now(timezone.utc).isoformat(),
        "_plausible": ok,
        "_plausibility_failures": failures,
        "_note": ("Report-only. competitions.strength() ignores this file "
                  "unless 'active' is true. Promotion is a separate decision "
                  "and requires the plausibility gate to pass."),
    })
    COMP_STRENGTH.parent.mkdir(exist_ok=True)
    COMP_STRENGTH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if verbose:
        print(f"\nwrote {COMP_STRENGTH.name} (active=false, plausible={ok})")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--estimate-k", action="store_true",
                    help="estimate K from the data (empirical Bayes)")
    ap.add_argument("--sensitivity", action="store_true",
                    help="show how the table moves with K")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()

    if args.estimate_k:
        estimate_k()
        return
    if args.sensitivity:
        sensitivity()
        return

    result = fit(k=args.k, season=args.season)
    rows = sorted(result["rows"], key=lambda r: -r["fitted"])
    print(f"Hierarchical competition strength (K={args.k})\n")
    print(f"{'competition':<26}{'n_int':>6}{'w':>7}{'observed':>10}"
          f"{'prior':>8}{'fitted':>9}{'hand-set':>10}")
    for r in rows:
        obs = "--" if r["observed"] is None else f"{r['observed']:.3f}"
        print(f"  {r['competition']:<24}{r['n_inter_league']:>6}"
              f"{r['shrink_weight']:>7.2f}{obs:>10}{r['prior']:>8.3f}"
              f"{r['fitted']:>9.3f}{r['hand_set']:>10}")

    ok, failures = plausibility(result["strengths"])
    print(f"\nplausibility gate: {'PASS' if ok else 'FAIL'}")
    for f in failures:
        print(f"  - {f}")

    if args.write:
        save(result, args.k)


if __name__ == "__main__":
    main()
