#!/usr/bin/env python3
"""NFL walk-forward validation gate.

Walk-forward discipline (no fitting on future games):
  * Elo's spread map / bye / short-week coefficients are fit once, strictly on
    2003-2014 (elo.FIT_SINCE/FIT_UNTIL — the plan's "era decision"); the
    per-game Elo *rating* diff used here is the pregame value from a single
    chronological run_elo() pass, never the final end-of-history rating.
  * HFA is the rolling trailing-3-season value for that season (elo.build()),
    which by construction only looks at seasons strictly before the current one.
  * Power and EPA ratings are refit before each week with asof = that week's
    first kickoff, so a week is scored by ratings that never saw that week.
  * QB rolling values are leak-free row-by-row already (qb.py computes each
    player's value from strictly prior appearances), so qb.fit() runs once.
  * The margin/total PMF (margin_dist.py) is fit on its own since-2003 pass —
    it's a distribution of outcomes given closeness, not a per-team model, so
    walk-forward leakage risk is low, but it is never refit inside the walk.

Metrics -> data/validation_baseline.json; gate fails (exit 1) if margin MAE,
ML Brier regress beyond tolerance. ATS/totals ROI vs closing lines are
reported for honesty but not gated (too noisy).

Usage:
  python3 -m nfl.validate --tune-blend --write     # pick + store blend weights
  python3 -m nfl.validate --gate --update-baseline  # full metric table + gate
  python3 -m nfl.validate --since 2015 --until 2025
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from . import elo as E
from . import epa as X
from . import margin_dist as MD
from . import power as P
from . import qb as QB

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES_CSV = os.path.join(HERE, "data", "games.csv")
BASELINE = os.path.join(HERE, "data", "validation_baseline.json")
BLEND_WEIGHT_JSON = os.path.join(HERE, "data", "blend_weight.json")

DEFAULT_SINCE, DEFAULT_UNTIL = 2015, 2025
TUNE_TRAIN_UNTIL = 2020   # tune on 2015-2020, validate on 2021+ (plan's split)
MAE_TOL = 0.50
BRIER_TOL = 0.010


def load_games() -> pd.DataFrame:
    return E.load_games(GAMES_CSV)


def walk_forward(games: pd.DataFrame, since: int = DEFAULT_SINCE,
                 until: int = DEFAULT_UNTIL, quiet: bool = True) -> pd.DataFrame:
    """Per-game walk-forward model components for seasons [since, until]."""
    _, history = E.run_elo(games, record_pregame=True)
    diffs = np.array([h[2] for h in history])
    sm = E.fit_spread_map(games, history)  # structural, fit strictly on 2003-2014

    qb_params = QB.fit()
    qb_index = QB.QBIndex(qb_params)
    epa_data = X.load_epa_games()

    ev = games[(games["season"] >= since) & (games["season"] <= until)]
    rows, idx = [], []
    for (season, week, _stype), wk in ev.groupby(["season", "week", "season_type"], sort=False):
        asof = wk["date"].min()
        try:
            pparams = P.fit(games, asof=asof)
        except ValueError:
            continue
        try:
            xparams = X.fit(asof=asof, data=epa_data)
        except ValueError:
            xparams = None
        hfa = E.rolling_hfa(games, history, sm, int(season))
        if not quiet:
            print(f"  fit week {season} w{week} ({len(wk)} games)", file=sys.stderr)
        for r in wk.itertuples():
            if r.home not in pparams["teams"] or r.away not in pparams["teams"]:
                continue
            d = diffs[r.Index]
            bye_diff = E._is_bye(r.home_rest) - E._is_bye(r.away_rest)
            short_diff = E._is_short(r.home_rest) - E._is_short(r.away_rest)
            m_elo = sm["slope"] * d + (0.0 if r.neutral else hfa) \
                + sm["bye_pts"] * bye_diff + sm["short_pts"] * short_diff
            pp = P.predict(pparams, r.home, r.away, neutral=bool(r.neutral))
            row = {
                "season": int(r.season), "week": int(r.week), "home": r.home, "away": r.away,
                "m_elo": m_elo, "m_pow": pp["margin"], "t_pow": pp["total"],
                "margin": r.home_score - r.away_score, "total": r.home_score + r.away_score,
                "spread_line": r.spread_line, "total_line": r.total_line,
            }
            if xparams is not None and r.home in xparams["teams"] and r.away in xparams["teams"]:
                xp = X.predict(xparams, r.home, r.away, neutral=bool(r.neutral))
                row["m_epa"], row["t_epa"] = xp["margin"], xp["total"]
            else:
                row["m_epa"], row["t_epa"] = np.nan, np.nan
            qb_delta = (qb_index.qb_delta_points(r.home, int(r.season), int(r.week), actual_qb_name=r.home_qb_name)
                       - qb_index.qb_delta_points(r.away, int(r.season), int(r.week), actual_qb_name=r.away_qb_name))
            row["qb_delta"] = qb_delta
            rows.append(row)
            idx.append(r.Index)
    return pd.DataFrame(rows, index=idx)


def blend_margin(df: pd.DataFrame, weights: dict) -> pd.Series:
    we, wp, wx = weights["elo"], weights["power"], weights["epa"]
    has_epa = df["m_epa"].notna()
    out = pd.Series(index=df.index, dtype=float)
    # rows with an EPA rating: full 3-way blend
    out[has_epa] = (we * df.loc[has_epa, "m_elo"] + wp * df.loc[has_epa, "m_pow"]
                    + wx * df.loc[has_epa, "m_epa"])
    # rows without: redistribute epa's weight over elo/power
    rest = we + wp
    we2, wp2 = (we + wx * we / rest, wp + wx * wp / rest) if rest > 0 else (0.5, 0.5)
    out[~has_epa] = we2 * df.loc[~has_epa, "m_elo"] + wp2 * df.loc[~has_epa, "m_pow"]
    return out + df["qb_delta"]


def blend_total(df: pd.DataFrame, weights: dict) -> pd.Series:
    wp, wx = weights["power"], weights["epa"]
    has_epa = df["t_epa"].notna()
    out = pd.Series(index=df.index, dtype=float)
    out[has_epa] = (wp * df.loc[has_epa, "t_pow"] + wx * df.loc[has_epa, "t_epa"]) / (wp + wx) if (wp + wx) > 0 else df.loc[has_epa, "t_pow"]
    out[~has_epa] = df.loc[~has_epa, "t_pow"]
    return out


def score_margin_mae(df: pd.DataFrame, weights: dict) -> float:
    pred = blend_margin(df, weights)
    return float((pred - df["margin"]).abs().mean())


def _simplex(step: float = 0.1):
    n = round(1.0 / step)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            yield {"elo": i * step, "power": j * step, "epa": k * step}


def tune_blend(df_train: pd.DataFrame, step: float = 0.1) -> dict:
    best_w, best_mae = {"elo": 0.5, "power": 0.5, "epa": 0.0}, float("inf")
    for w in _simplex(step):
        mae = score_margin_mae(df_train, w)
        if mae < best_mae:
            best_mae, best_w = mae, w
    return best_w


def ml_brier(df: pd.DataFrame, pmf: dict, margin_pred: pd.Series) -> float:
    probs = np.array([MD.moneyline_probs(pmf, m)["home_win"] for m in margin_pred])
    y = (df["margin"] > 0).astype(float).values
    return float(np.mean((probs - y) ** 2))


def cover_reliability(df: pd.DataFrame, pmf: dict, margin_pred: pd.Series, n_bins: int = 10) -> dict:
    """Decile calibration of P(home covers spread_line) vs actual cover rate."""
    have_line = df["spread_line"].notna()
    d = df[have_line]
    mp = margin_pred[have_line]
    probs = np.array([MD.cover_probs(pmf, m, l)["home_cover"] for m, l in zip(mp, d["spread_line"])])
    actual_cover = (d["margin"].values > d["spread_line"].values).astype(float)
    bins = pd.qcut(probs, min(n_bins, len(np.unique(probs))), duplicates="drop")
    tbl = pd.DataFrame({"p": probs, "y": actual_cover, "bin": bins}).groupby("bin", observed=True).agg(
        mean_p=("p", "mean"), mean_y=("y", "mean"), n=("y", "size"))
    gap = float((tbl["mean_p"] - tbl["mean_y"]).abs().mean())
    return {"mean_abs_calibration_gap": gap, "n_bins": len(tbl)}


def push_rate_calibration(df: pd.DataFrame, pmf: dict, margin_pred: pd.Series) -> dict:
    have_line = df["spread_line"].notna() & (df["spread_line"] % 1 == 0)
    d = df[have_line]
    mp = margin_pred[have_line]
    out = {}
    for key_number in (3, 7, 6, 10):
        sel = d["spread_line"].abs() == key_number
        if sel.sum() < 20:
            continue
        pred_push = np.mean([MD.cover_probs(pmf, m, l)["push"]
                             for m, l in zip(mp[sel], d.loc[sel, "spread_line"])])
        actual_push = float((d.loc[sel, "margin"] == d.loc[sel, "spread_line"]).mean())
        out[str(key_number)] = {"predicted": float(pred_push), "actual": actual_push, "n": int(sel.sum())}
    return out


def evaluate(since: int = DEFAULT_SINCE, until: int = DEFAULT_UNTIL, weights: dict | None = None,
            quiet: bool = True) -> dict:
    games = load_games()
    df = walk_forward(games, since=since, until=until, quiet=quiet)
    weights = weights or _load_blend_weight()
    pmf = MD.load()
    margin_pred = blend_margin(df, weights)
    total_pred = blend_total(df, weights)

    metrics = {
        "since": since, "until": until, "n_games": int(len(df)), "weights": weights,
        "margin_mae": float((margin_pred - df["margin"]).abs().mean()),
        "total_mae": float((total_pred - df["total"]).abs().mean()),
        "ml_brier": ml_brier(df, pmf, margin_pred),
        "cover_reliability": cover_reliability(df, pmf, margin_pred),
        "push_rate_calibration": push_rate_calibration(df, pmf, margin_pred),
    }
    return metrics


def _load_blend_weight() -> dict:
    if os.path.exists(BLEND_WEIGHT_JSON):
        with open(BLEND_WEIGHT_JSON) as f:
            return json.load(f)
    return {"elo": 0.5, "power": 0.5, "epa": 0.0}


def _save_blend_weight(w: dict) -> None:
    with open(BLEND_WEIGHT_JSON, "w") as f:
        json.dump(w, f, indent=1)


def _load_baseline() -> dict | None:
    if os.path.exists(BASELINE):
        with open(BASELINE) as f:
            return json.load(f)
    return None


def _save_baseline(metrics: dict) -> None:
    os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
    with open(BASELINE, "w") as f:
        json.dump(metrics, f, indent=1)


def gate(metrics: dict) -> int:
    baseline = _load_baseline()
    if baseline is None:
        print("no baseline yet — run with --update-baseline to establish one")
        return 0
    ok = True
    for key, tol in (("margin_mae", MAE_TOL), ("ml_brier", BRIER_TOL)):
        cur, base = metrics[key], baseline[key]
        if cur > base + tol:
            print(f"GATE FAIL  {key}: {cur:.4f} regressed past baseline {base:.4f} (+{tol})")
            ok = False
        else:
            print(f"  ok  {key}: {cur:.4f} (baseline {base:.4f})")
    return 0 if ok else 1


def _print_metrics(m: dict) -> None:
    print(f"\n{m['n_games']} games, seasons {m['since']}-{m['until']}, weights {m['weights']}")
    print(f"  margin MAE   {m['margin_mae']:.3f}   (closing spread MAE ~10.2-10.5 for reference)")
    print(f"  total MAE    {m['total_mae']:.3f}")
    print(f"  ML Brier     {m['ml_brier']:.4f}   (home-always ~0.24-0.25, closing ~0.205)")
    print(f"  cover reliability: mean |gap| = {m['cover_reliability']['mean_abs_calibration_gap']:.3f} "
         f"over {m['cover_reliability']['n_bins']} bins")
    print("  push-rate calibration (predicted vs actual, at key numbers):")
    for k, v in m["push_rate_calibration"].items():
        print(f"    |line|={k}: predicted {v['predicted']:.1%}  actual {v['actual']:.1%}  (n={v['n']})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=DEFAULT_SINCE)
    ap.add_argument("--until", type=int, default=DEFAULT_UNTIL)
    ap.add_argument("--tune-blend", action="store_true")
    ap.add_argument("--write", action="store_true", help="with --tune-blend, store the chosen weights")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--quiet", action="store_true", default=True)
    args = ap.parse_args()

    if args.tune_blend:
        default_w = {"elo": 0.5, "power": 0.5, "epa": 0.0}
        games = load_games()
        df = walk_forward(games, since=args.since, until=TUNE_TRAIN_UNTIL, quiet=True)
        best = tune_blend(df)
        train_mae = score_margin_mae(df, best)
        base_mae = score_margin_mae(df, default_w)
        print(f"tuned weights (train {args.since}-{TUNE_TRAIN_UNTIL}): {best}  "
             f"MAE {train_mae:.3f} (vs default 50/50/0 MAE {base_mae:.3f})")
        df_val = walk_forward(games, since=TUNE_TRAIN_UNTIL + 1, until=args.until, quiet=True)
        adopt = best
        if not df_val.empty:
            val_mae = score_margin_mae(df_val, best)
            val_base = score_margin_mae(df_val, default_w)
            print(f"holdout {TUNE_TRAIN_UNTIL + 1}-{args.until}: tuned MAE {val_mae:.3f} "
                 f"vs default MAE {val_base:.3f}")
            # Only adopt the tuned weight if it actually generalises — a grid
            # search will always "win" in-sample; the plan's bar is walk-
            # forward proof, not train-set fit (this is exactly the CFB
            # precedent: EPA/PPA was rejected there for failing this same
            # check). Ties within noise keep the safe 50/50/0 default.
            if val_mae < val_base - 1e-6:
                print("holdout confirms improvement -> adopting tuned weights")
            else:
                print("holdout does NOT confirm improvement -> keeping default "
                     "50/50/0 (reporting the tuned grid search honestly, not adopting it)")
                adopt = default_w
        if args.write:
            _save_blend_weight(adopt)
            print(f"wrote {BLEND_WEIGHT_JSON}: {adopt}")
        return 0

    games = load_games()
    metrics = evaluate(args.since, args.until, quiet=args.quiet)
    _print_metrics(metrics)

    if args.update_baseline:
        _save_baseline(metrics)
        print(f"\nbaseline updated -> {BASELINE}")
        return 0
    if args.gate:
        return gate(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
