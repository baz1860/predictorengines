#!/usr/bin/env python3
"""Acceptance test for the UEFA league expansion — baseline and gate.

The expansion is judged on one preregistered question: does the model become
properly calibrated on matches involving teams it cannot currently rate?

Test set: completed UEFA club matches where at least one side is below the
full-evidence bar (club_soccer.coverage). As of the P0 baseline that is ~1,094
matches in fixtures.csv, 543 of them with BOTH sides under-evidenced.

Why this is not just `validate.walk_forward`
--------------------------------------------
walk_forward SKIPS any match whose teams were unseen in the training window,
and reports a single pooled metric. It is therefore structurally blind to the
failure this project exists to fix: a team that IS in the training window but
carries only a handful of European matches is scored no differently from a
team with 300 domestic ones, and a team with none is dropped from the metric
entirely. Pooled Brier can improve while cross-league pricing stays broken.

This module keeps the same honest walk-forward discipline (monthly refit, no
look-ahead: context_coef={} and the fixed DEFAULT_ENSEMBLE_W, matching
validate.py) but tags every prediction with the evidence tier computed from
THAT FOLD's params, then reports metrics per tier.

Read the calibration table, not just the Brier score. A model that is
systematically overconfident against under-evidenced teams can post a
respectable Brier while losing money on exactly those fixtures, because Brier
rewards being right on the easy majority. The gate is flatness of the
calibration curve on the thin subset.

CLI:
  python3 -m club_soccer.cross_league_eval --baseline   # measure + store
  python3 -m club_soccer.cross_league_eval              # measure + compare
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import coverage as COV
from . import model as M
from .competitions import get as _get_comp

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BASELINE = DATA / "cross_league_baseline.json"

# Calibration buckets for the predicted probability of the observed outcome.
_BUCKETS = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
            (0.40, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.01)]


def _is_uefa(competition) -> bool:
    comp = _get_comp(competition)
    return bool(comp and comp.kind == "europe")


def _brier(p_vec: list[float], actual: int) -> float:
    """Multiclass Brier over (home, draw, away)."""
    return sum((p - (1.0 if i == actual else 0.0)) ** 2
               for i, p in enumerate(p_vec))


def _logloss(p_vec: list[float], actual: int) -> float:
    return -math.log(max(1e-12, p_vec[actual]))


def evaluate(min_train: int = 200, test_from: str | None = None,
             verbose: bool = True) -> dict:
    """Walk-forward evaluation tagged by evidence tier."""
    df = M.played(M.load_fixtures()).sort_values("date").reset_index(drop=True)
    df["_ym"] = df["date"].dt.to_period("M")
    months = sorted(df["_ym"].unique())
    first = pd.Timestamp(test_from).to_period("M") if test_from else None

    rows: list[dict] = []
    for k, ym in enumerate(months, 1):
        if first is not None and ym < first:
            continue
        test = df[df["_ym"] == ym]
        train = df[df["date"] < test["date"].min()]
        if len(train) < min_train:
            continue
        try:
            params = M.fit(train)
        except Exception:
            continue
        seen = set(params["teams"])
        kept = 0
        for r in test.itertuples(index=False):
            # Unseen teams cannot be priced at all (predict raises), so they
            # are recorded as an explicit 'unpriceable' outcome rather than
            # dropped — the count is part of the result, since dropping them
            # is how the existing validation hid this failure mode.
            if r.home not in seen or r.away not in seen:
                rows.append({"date": str(r.date.date()), "competition": r.competition,
                             "home": r.home, "away": r.away, "tier": "unpriceable",
                             "uefa": _is_uefa(r.competition), "brier": None,
                             "logloss": None, "p_actual": None, "actual": None})
                continue
            try:
                pred = M.predict_match(
                    r.home, r.away, r.competition, str(r.date.date()), "ensemble",
                    bool(r.neutral), params=params,
                    fixture_id=getattr(r, "fixture_id", None),
                    context_coef={},
                    ensemble_weights=dict(M.DEFAULT_ENSEMBLE_W),
                )
            except Exception:
                continue
            actual = 0 if r.home_goals > r.away_goals else (
                1 if r.home_goals == r.away_goals else 2)
            p = pred["probs"]
            vec = [float(p["home"]), float(p["draw"]), float(p["away"])]
            cov = pred.get("coverage") or {}
            h_tier = (cov.get("home") or {}).get("tier", "full")
            a_tier = (cov.get("away") or {}).get("tier", "full")
            # Exactly one weak side => we can isolate how that side was rated.
            # Both-weak matches tell us nothing directional (the errors cancel),
            # which is why they are excluded from the underdog diagnostic.
            p_thin = thin_won = None
            if h_tier != "full" and a_tier == "full":
                p_thin, thin_won = vec[0], float(actual == 0)
            elif a_tier != "full" and h_tier == "full":
                p_thin, thin_won = vec[2], float(actual == 2)
            rows.append({
                "date": str(r.date.date()), "competition": r.competition,
                "home": r.home, "away": r.away,
                "tier": str(cov.get("tier", "unknown")),
                "home_tier": h_tier, "away_tier": a_tier,
                "uefa": _is_uefa(r.competition),
                "brier": _brier(vec, actual), "logloss": _logloss(vec, actual),
                "p_actual": vec[actual], "p_vec": vec, "actual": actual,
                "p_thin": p_thin, "thin_won": thin_won,
            })
            kept += 1
        if verbose:
            print(f"  [{k:>3}/{len(months)}] {ym}  tested {kept}")
    return _summarise(rows)


def _subset_metrics(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("brier") is not None]
    if not scored:
        return {"n": 0, "n_unpriceable": len(rows) - len(scored)}
    n = len(scored)
    return {
        "n": n,
        "n_unpriceable": len(rows) - len(scored),
        "brier": round(sum(r["brier"] for r in scored) / n, 5),
        "logloss": round(sum(r["logloss"] for r in scored) / n, 5),
        "calibration": _calibration(scored),
        "max_abs_calibration_error": _max_cal_error(scored),
    }


def _reliability(rows: list[dict], key_p: str, key_hit: str) -> list[dict]:
    """Standard reliability curve: bucket by predicted probability, compare
    the bucket's mean prediction against the realised frequency."""
    out = []
    for lo, hi in _BUCKETS:
        bucket = [r for r in rows
                  if r.get(key_p) is not None and lo <= r[key_p] < hi]
        if len(bucket) < 15:          # too few to read anything into
            continue
        predicted = sum(r[key_p] for r in bucket) / len(bucket)
        observed = sum(r[key_hit] for r in bucket) / len(bucket)
        out.append({"bucket": f"{lo:.2f}-{hi:.2f}", "n": len(bucket),
                    "predicted": round(predicted, 4),
                    "observed": round(observed, 4),
                    "error": round(observed - predicted, 4)})
    return out


def _calibration(rows: list[dict]) -> list[dict]:
    """Multiclass reliability curve.

    Each match contributes three (predicted, hit) pairs — one per outcome —
    so the curve measures whether "the model says 20%" means "happens 20% of
    the time" across all outcomes. Bucketing only the realised outcome's
    probability would be degenerate (its hit rate is 1 by construction).
    """
    expanded = []
    for r in rows:
        vec = r.get("p_vec")
        if not vec:
            continue
        for i, p in enumerate(vec):
            expanded.append({"p": float(p), "hit": 1.0 if i == r["actual"] else 0.0})
    return _reliability(expanded, "p", "hit")


def _max_cal_error(rows: list[dict]) -> float:
    """Largest absolute reliability gap across populated buckets."""
    curve = _calibration(rows)
    return round(max((abs(b["error"]) for b in curve), default=0.0), 5)


def underdog_reliability(rows: list[dict]) -> dict:
    """THE diagnostic for the Sturm Graz failure.

    Restricted to matches with exactly one under-evidenced side, this asks:
    when the model gave the weakly-rated team probability p of winning, how
    often did it actually win?

    Overall Brier cannot answer this and in fact points the wrong way — thin
    fixtures are disproportionately mismatches, which are easy to call, so
    their pooled Brier looks GOOD while the model is systematically writing
    off teams it simply has not measured. If `observed` materially exceeds
    `predicted` in the low buckets, under-evidenced teams are being
    under-rated, which is exactly the Sturm-Graz-at-18% pattern.
    """
    subset = [r for r in rows if r.get("p_thin") is not None]
    if not subset:
        return {"n": 0}
    predicted = sum(r["p_thin"] for r in subset) / len(subset)
    observed = sum(r["thin_won"] for r in subset) / len(subset)
    return {
        "n": len(subset),
        "mean_predicted_win_rate": round(predicted, 4),
        "observed_win_rate": round(observed, 4),
        # Positive => the model under-rates under-evidenced teams.
        "bias": round(observed - predicted, 4),
        "curve": _reliability(subset, "p_thin", "thin_won"),
    }


# Competitions that existed before the P3 expansion. The promotion baseline in
# validation_baseline.json was measured on these alone, so a pooled Brier
# comparison across the expansion is meaningless — the match population
# changed, and the new leagues are intrinsically harder (ten of them arrive via
# fd.co.uk's /new/ files with no shot data at all, so the xg and xpress
# ensemble components have nothing to work with). The regression question that
# actually matters is narrower: did adding leagues damage the leagues we
# already modelled?
PRE_EXPANSION_COMPETITIONS = {
    "Premier League", "Championship", "League One", "League Two",
    "FA Cup", "EFL Cup", "Scottish Premiership", "Scottish Championship",
    "Scottish League One", "Scottish League Two", "Scottish Cup",
    "Scottish League Cup", "Bundesliga", "DFB-Pokal", "Serie A",
    "Coppa Italia", "Ligue 1", "Coupe de France", "La Liga", "Copa del Rey",
    "Champions League", "Europa League", "Conference League", "UEFA Super Cup",
}


def _summarise(rows: list[dict]) -> dict:
    uefa = [r for r in rows if r["uefa"]]
    thin_tiers = {"thin", "defaulted", "unpriceable"}
    pre = [r for r in rows if r["competition"] in PRE_EXPANSION_COMPETITIONS]
    new = [r for r in rows if r["competition"] not in PRE_EXPANSION_COMPETITIONS]
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_predictions": len(rows),
        "overall": _subset_metrics(rows),
        "by_tier": {t: _subset_metrics([r for r in rows if r["tier"] == t])
                    for t in sorted({r["tier"] for r in rows})},
        # THE headline subset: UEFA matches with at least one weak side.
        "uefa_cross_league": _subset_metrics(
            [r for r in uefa if r["tier"] in thin_tiers]),
        "uefa_full_evidence": _subset_metrics(
            [r for r in uefa if r["tier"] == "full"]),
        "domestic_full_evidence": _subset_metrics(
            [r for r in rows if not r["uefa"] and r["tier"] == "full"]),
        # Apples-to-apples regression check across the expansion.
        "pre_expansion_leagues": _subset_metrics(pre),
        "expansion_leagues": _subset_metrics(new),
        # The decisive diagnostic — see underdog_reliability().
        "underdog_reliability_all": underdog_reliability(rows),
        "underdog_reliability_uefa": underdog_reliability(uefa),
    }
    return result


def _print(label: str, res: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"predictions: {res['n_predictions']}")
    for key in ("overall", "pre_expansion_leagues", "expansion_leagues",
                "uefa_cross_league", "uefa_full_evidence",
                "domestic_full_evidence"):
        m = res.get(key) or {}
        if not m.get("n"):
            print(f"  {key:<24} n=0")
            continue
        print(f"  {key:<24} n={m['n']:<6} brier={m['brier']:<8} "
              f"logloss={m['logloss']:<8} cal_err={m['max_abs_calibration_error']}")
    print("  by tier:")
    for tier, m in (res.get("by_tier") or {}).items():
        if m.get("n"):
            print(f"    {tier:<12} n={m['n']:<6} brier={m['brier']:<8} "
                  f"logloss={m['logloss']}")
        else:
            print(f"    {tier:<12} n=0 (unpriceable={m.get('n_unpriceable', 0)})")

    for key in ("underdog_reliability_all", "underdog_reliability_uefa"):
        u = res.get(key) or {}
        if not u.get("n"):
            continue
        print(f"\n  {key}  (n={u['n']})")
        print(f"    model gave under-evidenced side : "
              f"{u['mean_predicted_win_rate']:.1%} win rate")
        print(f"    they actually won              : "
              f"{u['observed_win_rate']:.1%}")
        print(f"    bias (+ => model under-rates them): {u['bias']:+.4f}")
        if u.get("curve"):
            print(f"    {'bucket':<12}{'n':>6}{'predicted':>11}{'observed':>10}{'error':>9}")
            for b in u["curve"]:
                print(f"    {b['bucket']:<12}{b['n']:>6}{b['predicted']:>11.3f}"
                      f"{b['observed']:>10.3f}{b['error']:>+9.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", action="store_true",
                    help="store this run as the pre-change baseline")
    ap.add_argument("--test-from", default=None,
                    help="only evaluate months from this date (e.g. 2023-07-01)")
    ap.add_argument("--min-train", type=int, default=200)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    res = evaluate(min_train=args.min_train, test_from=args.test_from,
                   verbose=not args.quiet)
    _print("current", res)

    if args.baseline:
        if BASELINE.exists():
            print(f"\nREFUSING to overwrite {BASELINE.name} — the baseline must be "
                  "measured once, before changes. Delete it deliberately if you "
                  "really mean to re-baseline.")
            return
        DATA.mkdir(exist_ok=True)
        BASELINE.write_text(json.dumps(res, indent=2))
        print(f"\nwrote baseline -> {BASELINE}")
        return

    if BASELINE.exists():
        base = json.loads(BASELINE.read_text())
        _print("baseline", base)
        print("\n=== delta (negative = better) ===")
        for key in ("overall", "uefa_cross_league", "uefa_full_evidence",
                    "domestic_full_evidence"):
            b, c = base.get(key) or {}, res.get(key) or {}
            if b.get("n") and c.get("n"):
                print(f"  {key:<24} brier {c['brier'] - b['brier']:+.5f}  "
                      f"logloss {c['logloss'] - b['logloss']:+.5f}")
    else:
        print(f"\nno baseline stored — run with --baseline first.")


if __name__ == "__main__":
    main()
