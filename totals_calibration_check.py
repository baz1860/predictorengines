#!/usr/bin/env python3
"""Totals calibration check + fit: does the goal model's P(over 2.5) match reality?

Leak-free, point-in-time. For WC2018 and WC2022 we fit the SAME elo+Poisson /
Dixon-Coles blend the suite ships (trained strictly before kickoff), score every
match's totals distribution exactly as edge.market_probs does (two-engine mean
score matrix), and compare model vs realised on totals.

Findings it surfaces:
  - mean model expected total goals     vs  mean realised total goals
  - mean model P(over 2.5)              vs  realised over-2.5 rate
  - totals log-loss / Brier of the model on the over/under outcome

It then isolates the lever:
  [A] global rho sweep      — low-score correction (expected: ~no effect on 2.5)
  [B] global lambda x-sweep — overall scoring level (expected: the real lever)

Finally it FITS a global lambda multiplier by max-likelihood on the realised
over/under outcomes and GATES it leave-one-tournament-out (must beat lam=1.0 on
every held-out tournament). It also reports the 1X2 log-loss under the multiplier
so we can confirm the totals fix does not disturb the shipped 1X2 prices. If the
gate passes it writes data/totals_calibration.json (consumed by edge.py).

Run:
  python3 totals_calibration_check.py          # report + sweep only
  python3 totals_calibration_check.py --fit     # also fit, gate, write json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from engines.worldcup.predictor import (load_matches, compute_elo, fit_goal_model,
                       expected_goals, score_matrix, DC_RHO)
from engines.worldcup.dixoncoles import fit_dc

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CALIB_FILE = DATA / "totals_calibration.json"
_NAME_MAP = {"USA": "United States", "Korea Republic": "South Korea"}
EPS = 1e-9

TOURNAMENTS = {
    "WC2018": {"odds": "wc2018_odds.csv", "cutoff": "2018-06-14"},
    "WC2022": {"odds": "wc2022_odds.csv", "cutoff": "2022-11-20"},
}


def _matrices(le, ld, rho_e, rho_d, lam_mult):
    return [score_matrix(le[0] * lam_mult, le[1] * lam_mult, rho_e),
            score_matrix(ld[0] * lam_mult, ld[1] * lam_mult, rho_d)]


def _p_over(Ms, line_idx=2):
    M = np.mean(Ms, axis=0)
    n = M.shape[0]
    tot = np.add.outer(np.arange(n), np.arange(n))
    return 1.0 - M[tot <= line_idx].sum()


def _exp_total(Ms):
    M = np.mean(Ms, axis=0)
    n = M.shape[0]
    tot = np.add.outer(np.arange(n), np.arange(n))
    return float((M * tot).sum())


def _p_1x2(Ms):
    M = np.mean(Ms, axis=0)
    return (float(np.tril(M, -1).sum()), float(np.trace(M)),
            float(np.triu(M, 1).sum()))


def build(cutoff, odds_file):
    """Per-match leak-free engine outputs + realised result, for one tournament."""
    played, _ = load_matches()
    _, played = compute_elo(played)
    train = played[played["date"] < cutoff]
    beta = fit_goal_model(train)
    dc = fit_dc(train, anchor=cutoff, verbose=False)
    ratings, _ = compute_elo(train)

    res = played.copy()
    res["key"] = list(zip(res["date"].dt.strftime("%Y-%m-%d"),
                          res["home_team"], res["away_team"]))
    score_by_key = {k: (hs, as_) for k, hs, as_ in
                    zip(res["key"], res["home_score"], res["away_score"])}

    odds = pd.read_csv(DATA / odds_file)
    rows = []
    for r in odds.itertuples(index=False):
        home = _NAME_MAP.get(r.home, r.home)
        away = _NAME_MAP.get(r.away, r.away)
        if home not in ratings or home not in dc.att or away not in dc.att:
            continue
        sc = score_by_key.get((r.date, home, away))
        if sc is None or pd.isna(sc[0]) or pd.isna(sc[1]):
            continue
        hs, as_ = sc
        rows.append({
            "le": expected_goals(ratings[home], ratings[away], beta, 0.0),
            "ld": dc.lambdas(home, away),
            "rho_e": DC_RHO, "rho_d": dc.rho,
            "over_actual": 1.0 if (hs + as_) >= 3 else 0.0,
            "total_actual": float(hs + as_),
            "res_idx": 0 if hs > as_ else (1 if hs == as_ else 2),
        })
    return rows


def score(rows, rho_override=None, lam_mult=1.0):
    p_over, exp_tot, over_y, tot_y = [], [], [], []
    ll_1x2 = 0.0
    for d in rows:
        re_ = d["rho_e"] if rho_override is None else rho_override
        rd_ = d["rho_d"] if rho_override is None else rho_override
        Ms = _matrices(d["le"], d["ld"], re_, rd_, lam_mult)
        po = _p_over(Ms)
        p_over.append(po); exp_tot.append(_exp_total(Ms))
        over_y.append(d["over_actual"]); tot_y.append(d["total_actual"])
        p1x2 = _p_1x2(Ms)
        ll_1x2 += -np.log(max(p1x2[d["res_idx"]], EPS))
    p = np.array(p_over); y = np.array(over_y)
    ll = -np.mean(y * np.log(np.clip(p, EPS, 1)) +
                  (1 - y) * np.log(np.clip(1 - p, EPS, 1)))
    return {
        "n": len(rows), "model_E_total": float(np.mean(exp_tot)),
        "actual_total": float(np.mean(tot_y)), "model_over": float(p.mean()),
        "actual_over": float(y.mean()), "ou_logloss": float(ll),
        "brier": float(np.mean((p - y) ** 2)),
        "x12_logloss": float(ll_1x2 / len(rows)),
    }


def line(label, s):
    print(f"{label:24s} n={s['n']:3d} | model E[total]={s['model_E_total']:.2f} "
          f"actual={s['actual_total']:.2f} | model P(over)={s['model_over']:.3f} "
          f"actual={s['actual_over']:.3f} | OU logloss={s['ou_logloss']:.4f} "
          f"1X2 logloss={s['x12_logloss']:.4f}")


def fit_lambda(pooled_rows, grid=None):
    """MLE global lambda multiplier on the realised over/under outcome."""
    grid = grid if grid is not None else np.round(np.arange(0.90, 1.31, 0.01), 2)
    best, best_ll = 1.0, np.inf
    for m in grid:
        ll = score(pooled_rows, lam_mult=float(m))["ou_logloss"]
        if ll < best_ll:
            best_ll, best = ll, float(m)
    return best, best_ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true",
                    help="fit + gate the lambda multiplier and write json")
    args = ap.parse_args()

    data = {n: build(c["cutoff"], c["odds"]) for n, c in TOURNAMENTS.items()}
    pooled = [r for rows in data.values() for r in rows]

    print("=" * 112)
    print("TOTALS CALIBRATION — leak-free, point-in-time "
          f"(current model: rho_elo={DC_RHO}, rho_dc=fitted, lam_mult=1.0)")
    print("=" * 112)
    for n, rows in data.items():
        line(n, score(rows))
    line("POOLED", score(pooled))

    print("\n[A] rho sweep (scoring level held at model default):")
    for rho in [-0.20, -0.10, 0.0, 0.05]:
        line(f"  rho={rho:+.2f}", score(pooled, rho_override=rho))
    print("\n[B] lambda multiplier sweep (rho at model default):")
    for m in [0.95, 1.00, 1.05, 1.10, 1.15, 1.20]:
        line(f"  lam x{m:.2f}", score(pooled, lam_mult=m))

    if not args.fit:
        print("\n(report only; pass --fit to gate + write json)")
        return

    print("\n" + "=" * 112)
    print("FIT + GATE lambda multiplier (MLE on realised O/U; "
          "leave-one-tournament-out must beat lam=1.0 everywhere)")
    print("=" * 112)
    m_star, ll_star = fit_lambda(pooled)
    base_ll = score(pooled, lam_mult=1.0)["ou_logloss"]
    print(f"pooled MLE multiplier m* = {m_star:.2f} "
          f"(pooled OU logloss {base_ll:.4f} -> {ll_star:.4f})")

    gate_ok = True
    for held in TOURNAMENTS:
        train_rows = [r for n, rows in data.items() if n != held for r in rows]
        test_rows = data[held]
        m_tr, _ = fit_lambda(train_rows)
        base = score(test_rows, lam_mult=1.0)["ou_logloss"]
        tuned = score(test_rows, lam_mult=m_tr)["ou_logloss"]
        ok = tuned < base
        gate_ok &= ok
        print(f"  LOTO hold-out {held}: fit-on-rest m={m_tr:.2f} | "
              f"held-out OU logloss {base:.4f} -> {tuned:.4f}  "
              f"{'PASS' if ok else 'FAIL'}")

    s_base = score(pooled, lam_mult=1.0)
    s_tuned = score(pooled, lam_mult=m_star)
    print(f"  1X2 impact (pooled): logloss {s_base['x12_logloss']:.4f} -> "
          f"{s_tuned['x12_logloss']:.4f} "
          f"({'unharmed' if s_tuned['x12_logloss'] <= s_base['x12_logloss'] + 5e-3 else 'WORSE'})")

    if gate_ok:
        out = {
            "lambda_mult": round(m_star, 3),
            "applies_to": ["over25", "under25", "btts_yes", "btts_no"],
            "n": len(pooled),
            "ou_logloss_base": round(base_ll, 4),
            "ou_logloss_tuned": round(ll_star, 4),
            "gate": "leave-one-tournament-out, beats lam=1.0 on every held-out cup",
            "source": "WC2018+WC2022 realised over/under, leak-free point-in-time",
            "note": ("rho sweep is flat at the 2.5 line; the bias is the overall "
                     "scoring level. Applied to totals/BTTS only; 1X2 left as-is."),
        }
        CALIB_FILE.write_text(json.dumps(out, indent=2))
        print(f"\nGATE PASSED -> wrote {CALIB_FILE.relative_to(ROOT)}")
    else:
        print("\nGATE FAILED -> no json written; model stays uncorrected.")


if __name__ == "__main__":
    main()
