"""
golf/validate.py  –  Walk-forward backtest + regression gate (the yardstick).

For each completed tournament (after a minimum history), refit the model on
rounds STRICTLY before that event, simulate the field, and score the predicted
win / top-5 / top-10 / top-20 / make-cut probabilities against what actually
happened. No look-ahead. Mirrors club_soccer/validate.py + root validate.py.

Metrics per market: Brier, log-loss, and a reliability table; plus a skill
score vs the base-rate baseline (1 − Brier/Brier_base). Win is scored both as
per-player favorite calibration and as event-level surprise −log p(winner).

Outputs:
  data/validation_predictions.csv   (feeds calibrate.py)
  data/validation_baseline.json     (Brier baseline for --gate)

Usage:
  python validate.py [--since 2023-06-01] [--sims 20000] [--gate]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import model
import simulate as gsim

DATA_DIR = Path(__file__).parent / "data"
PRED_CSV = DATA_DIR / "validation_predictions.csv"
BASELINE_JSON = DATA_DIR / "validation_baseline.json"

MARKETS = ["win", "top5", "top10", "top20", "cut"]
TOPN = {"top5": 5, "top10": 10, "top20": 20}
GATE_TOL = 0.004          # allowed Brier regression on the headline metric
MIN_TRAIN_ROUNDS = 4000   # don't evaluate until the model has enough history
EPS = 1e-12


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p: np.ndarray, y: np.ndarray, bins=10) -> list[tuple]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    idx = np.digitize(p, edges[1:-1])
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append((round(float(p[m].mean()), 3), round(float(y[m].mean()), 3), int(m.sum())))
    return out


def _actuals(event: pd.DataFrame) -> dict[str, dict]:
    """Per-player actual outcomes for one tournament (recompute finish from
    72-hole totals so ties are handled, independent of ESPN's order field)."""
    g = event.groupby("player")
    total = g["score_to_par"].sum()
    made = g["made_cut"].max()
    nrounds = g["round"].count()
    # rank only players who completed the tournament; missed-cut → no top-N
    finishers = total[made == 1]
    rank = finishers.rank(method="min")
    out = {}
    for player in total.index:
        mc = int(made.loc[player])
        r = int(rank.loc[player]) if (mc == 1 and player in rank.index) else 999
        out[player] = {
            "made_cut": mc,
            "win": int(r == 1),
            "top5": int(r <= 5),
            "top10": int(r <= 10),
            "top20": int(r <= 20),
            "finish": r,
        }
    return out


# ─────────────────────────────────────────────
# Walk-forward loop
# ─────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, since: str, sims: int,
                 seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    events = (df[["tournament_id", "date", "course", "is_major"]]
              .drop_duplicates("tournament_id")
              .sort_values("date"))
    rng = np.random.default_rng(seed)
    rows = []
    since_ts = pd.Timestamp(since)

    for ev in events.itertuples():
        start = pd.Timestamp(ev.date)
        if start < since_ts:
            continue
        prior = df[df["date"] < start]
        if len(prior) < MIN_TRAIN_ROUNDS:
            continue
        event_rounds = df[df["tournament_id"] == ev.tournament_id]
        field = sorted(event_rounds["player"].unique())
        if len(field) < 30:
            continue
        try:
            params = model.fit(df, asof=start)
        except ValueError:
            continue
        rated = model.predict_field(field, params, course=str(ev.course),
                                    is_major=bool(ev.is_major))
        res = gsim.simulate_tournament(rated, n_sims=sims, cut_rule=65, rng=rng)
        actual = _actuals(event_rounds)
        for p in rated:
            a = actual.get(p.name)
            if a is None:
                continue
            r = res[p.name]
            rows.append({
                "tournament_id": ev.tournament_id, "date": str(start.date()),
                "is_major": int(bool(ev.is_major)), "player": p.name,
                "p_win": r["win"], "p_top5": r["top5"], "p_top10": r["top10"],
                "p_top20": r["top20"], "p_cut": r["made_cut"],
                "y_win": a["win"], "y_top5": a["top5"], "y_top10": a["top10"],
                "y_top20": a["top20"], "y_cut": a["made_cut"],
            })
        if verbose:
            print(f"  {str(start.date())}  {ev.tournament_id}  "
                  f"{len(field):>3} players  (train={len(prior):,})")
    return pd.DataFrame(rows)


def summarize(pred: pd.DataFrame) -> dict:
    report = {}
    for mkt in MARKETS:
        col = "cut" if mkt == "cut" else mkt
        p = pred[f"p_{col}"].values
        y = pred[f"y_{col}"].values.astype(float)
        base = float(y.mean())
        b = brier(p, y)
        b_base = brier(np.full_like(y, base), y)
        report[mkt] = {
            "n": int(len(y)), "base_rate": round(base, 4),
            "brier": round(b, 5), "brier_base": round(b_base, 5),
            "skill": round(1 - b / b_base, 4) if b_base > 0 else 0.0,
            "logloss": round(logloss(p, y), 5),
            "reliability": reliability(p, y),
        }
    # event-level win surprise: −log p(actual winner)
    surprises, base_surprises = [], []
    for _tid, g in pred.groupby("tournament_id"):
        winners = g[g["y_win"] == 1]
        if winners.empty:
            continue
        pw = float(np.clip(winners["p_win"].mean(), EPS, 1))
        surprises.append(-np.log(pw))
        base_surprises.append(-np.log(1.0 / len(g)))
    report["win_event"] = {
        "events": len(surprises),
        "mean_surprise": round(float(np.mean(surprises)), 4) if surprises else None,
        "uniform_surprise": round(float(np.mean(base_surprises)), 4) if base_surprises else None,
    }
    # headline gate metric: mean skill across the lower-variance markets
    report["headline_brier"] = round(
        float(np.mean([report[m]["brier"] for m in ("top10", "top20", "cut")])), 5)
    return report


def print_report(rep: dict) -> None:
    print(f"\n{'Market':<8}{'N':>7}{'base':>8}{'Brier':>9}{'vs base':>9}"
          f"{'skill':>8}{'logloss':>9}")
    print("-" * 58)
    for mkt in MARKETS:
        r = rep[mkt]
        print(f"{mkt:<8}{r['n']:>7}{r['base_rate']:>8.3f}{r['brier']:>9.4f}"
              f"{r['brier_base']:>9.4f}{r['skill']:>8.1%}{r['logloss']:>9.4f}")
    we = rep["win_event"]
    if we["mean_surprise"] is not None:
        print(f"\nWinner surprise −log p: model {we['mean_surprise']:.3f}  vs "
              f"uniform {we['uniform_surprise']:.3f}  over {we['events']} events "
              f"(lower = better)")
    print(f"\nHeadline Brier (top10+top20+cut): {rep['headline_brier']:.5f}")
    # show reliability for make-cut (the cleanest signal)
    print("\nMake-cut reliability  (pred → actual, n):")
    for pp, yy, nn in rep["cut"]["reliability"]:
        bar = "█" * int(yy * 30)
        print(f"  {pp:>5.2f} → {yy:>5.2f}  {bar}  ({nn})")


def main():
    ap = argparse.ArgumentParser(description="Walk-forward golf backtest + gate")
    ap.add_argument("--since", default="2023-06-01",
                    help="Evaluate events on/after this date (default %(default)s)")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", action="store_true",
                    help="Exit non-zero if headline Brier regresses vs baseline")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    df = model.load_rounds_df()
    print(f"Walk-forward from {args.since}  ({args.sims:,} sims/event)…")
    pred = walk_forward(df, since=args.since, sims=args.sims, seed=args.seed,
                        verbose=not args.quiet)
    if pred.empty:
        print("No evaluable events — seed more history.")
        sys.exit(1)
    pred.to_csv(PRED_CSV, index=False)
    print(f"\n{len(pred):,} player-event predictions → {PRED_CSV}")

    rep = summarize(pred)
    print_report(rep)

    head = rep["headline_brier"]
    if BASELINE_JSON.exists():
        baseline = json.loads(BASELINE_JSON.read_text())
        prev = baseline.get("headline_brier", head)
        delta = head - prev
        print(f"\nBaseline headline Brier {prev:.5f}  →  now {head:.5f}  "
              f"(Δ {delta:+.5f}, tol {GATE_TOL})")
        if args.gate and delta > GATE_TOL:
            print("GATE FAIL: model regressed beyond tolerance.")
            sys.exit(2)
        if delta < -GATE_TOL:  # improvement → adopt new baseline
            BASELINE_JSON.write_text(json.dumps(
                {"headline_brier": head, "gate_tol": GATE_TOL,
                 "asof": pred["date"].max()}, indent=1))
            print("Improved — baseline updated.")
    else:
        BASELINE_JSON.write_text(json.dumps(
            {"headline_brier": head, "gate_tol": GATE_TOL,
             "asof": pred["date"].max()}, indent=1))
        print(f"\nBaseline written → {BASELINE_JSON}")


if __name__ == "__main__":
    main()
