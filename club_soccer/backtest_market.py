#!/usr/bin/env python3
"""Market-anchored backtest — the honest scoreboard.

Joins walk-forward predictions (validate.walk_forward, which already carries
1X2/OU2.5 model probabilities and labels) to data/market_history.csv
(fetch_fdcouk.py) and reports, per competition and overall:

  * n matched rows
  * model 1X2 log-loss vs de-vigged Pinnacle CLOSING log-loss on the same rows
    (the market benchmark to beat or approach)
  * simulated betting vs closing prices at edge thresholds {2%, 4%, 6%}
    (model prob - de-vigged closing prob): flat 1-unit ROI, quarter-Kelly ROI,
    bet count, split by market (1X2 / OU2.5)
  * CLV proxy: for bets simulated at the Bet365 PRE-MATCH price, CLV =
    de-vigged Pinnacle closing prob of the side - de-vigged B365 prob of the
    side. Mean and fraction-positive, rows lacking either book excluded
    (count reported).

This is a diagnostic, not a gate — no promote/reject verdict. Runs offline
once market_history.csv exists; prints "no market_history.csv" and returns
an empty report otherwise (never raises out of a pipeline).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import edge as E
from . import validate as V

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MARKET_HISTORY = DATA / "market_history.csv"
REPORT_JSON = DATA / "backtest_market.json"

EDGE_THRESHOLDS = (0.02, 0.04, 0.06)
KELLY_FRACTION = 0.25


def _devig3(oh, od, oa):
    inv = np.array([1.0 / oh, 1.0 / od, 1.0 / oa])
    p = inv / inv.sum()
    return p[0], p[1], p[2]


def _joined_rows() -> pd.DataFrame:
    rows, _ = V.walk_forward(verbose=False)
    if not rows:
        return pd.DataFrame()
    pred = pd.DataFrame(rows)
    if not MARKET_HISTORY.exists():
        return pd.DataFrame()
    mh = pd.read_csv(MARKET_HISTORY)
    if mh.empty:
        return pd.DataFrame()
    j = pred.merge(mh, left_on=["date", "home", "away"],
                   right_on=["match_date", "home", "away"], how="inner",
                   suffixes=("", "_mkt"))
    if j.empty:
        return j
    has_psc = j[["psc_h", "psc_d", "psc_a"]].notna().all(axis=1)
    ph, pd_, pa = np.full(len(j), np.nan), np.full(len(j), np.nan), np.full(len(j), np.nan)
    idx = j.index[has_psc]
    for i in idx:
        r = j.loc[i]
        h, d, a = _devig3(float(r["psc_h"]), float(r["psc_d"]), float(r["psc_a"]))
        ph[j.index.get_loc(i)], pd_[j.index.get_loc(i)], pa[j.index.get_loc(i)] = h, d, a
    j["p_close_h"], j["p_close_d"], j["p_close_a"] = ph, pd_, pa

    has_b365 = j[["b365_h", "b365_d", "b365_a"]].notna().all(axis=1)
    bh, bd, ba = np.full(len(j), np.nan), np.full(len(j), np.nan), np.full(len(j), np.nan)
    for i in j.index[has_b365]:
        r = j.loc[i]
        h, d, a = _devig3(float(r["b365_h"]), float(r["b365_d"]), float(r["b365_a"]))
        bh[j.index.get_loc(i)], bd[j.index.get_loc(i)], ba[j.index.get_loc(i)] = h, d, a
    j["p_b365_h"], j["p_b365_d"], j["p_b365_a"] = bh, bd, ba

    has_ou = j[["b365_over25", "b365_under25"]].notna().all(axis=1)
    p_close_over = np.where(has_ou,
                            1.0 / j["b365_over25"] / (1.0 / j["b365_over25"] + 1.0 / j["b365_under25"]),
                            np.nan)
    j["p_close_over25"] = p_close_over
    j["p_close_under25"] = 1.0 - p_close_over
    return j


def _log_loss(p_home, p_draw, p_away, actual) -> float:
    p = np.stack([p_home, p_draw, p_away], axis=1)
    idx = actual.astype(int).to_numpy()
    chosen = p[np.arange(len(p)), idx]
    return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())


def _sim_market(j: pd.DataFrame, market: str, sides: list[str],
                p_model_cols: dict[str, str], p_close_cols: dict[str, str],
                odds_close_cols: dict[str, str], p_b365_cols: dict[str, str] | None,
                odds_b365_cols: dict[str, str] | None, win_fn) -> dict:
    """Simulate flat + quarter-Kelly betting on one market across all
    edge thresholds, bet at the closing price. When p_b365_cols/odds_b365_cols
    are given AND distinct from the closing price source, also compute the
    CLV proxy for the same bet set (bets priced at the B365 pre-match price).
    For OU2.5, market_history.csv has no separate closing-totals feed (P1.3),
    so the "closing" price literally IS the B365 price — CLV would be a
    fabricated exact-zero, not a real finding, so it's reported as n/a there
    (caller passes p_b365_cols=None for that case)."""
    has_clv = p_b365_cols is not None and odds_b365_cols is not None
    out = {}
    for thr in EDGE_THRESHOLDS:
        n_bets = 0
        flat_profit = 0.0
        kelly_staked = 0.0
        kelly_profit = 0.0
        clv_vals: list[float] = []
        clv_excluded = 0
        for side in sides:
            p_model = j[p_model_cols[side]].to_numpy(float)
            p_close = j[p_close_cols[side]].to_numpy(float)
            odds_close = j[odds_close_cols[side]].to_numpy(float)
            edge = p_model - p_close
            take = np.isfinite(edge) & (edge >= thr) & np.isfinite(odds_close) & (odds_close > 1.0)
            if not take.any():
                continue
            won = win_fn(j, side)
            for i in np.nonzero(take)[0]:
                n_bets += 1
                o = float(odds_close[i])
                w = bool(won.iloc[i])
                flat_profit += (o - 1.0) if w else -1.0
                kf = KELLY_FRACTION * E.kelly(float(p_model[i]), o)
                kelly_staked += kf
                kelly_profit += kf * (o - 1.0) if w else -kf
                if has_clv:
                    p_b365 = j[p_b365_cols[side]].to_numpy(float)[i]
                    o_b365 = j[odds_b365_cols[side]].to_numpy(float)[i]
                    if np.isfinite(p_b365) and np.isfinite(o_b365):
                        clv_vals.append(float(p_close[i] - p_b365))
                    else:
                        clv_excluded += 1
        out[f"{thr:.0%}"] = {
            "n_bets": n_bets,
            "flat_roi": round(flat_profit / n_bets, 4) if n_bets else None,
            "kelly_roi": round(kelly_profit / kelly_staked, 4) if kelly_staked > 0 else None,
            "clv_mean": (round(float(np.mean(clv_vals)), 4) if has_clv and clv_vals else None),
            "clv_frac_positive": (round(float(np.mean([v > 0 for v in clv_vals])), 4)
                                  if has_clv and clv_vals else None),
            "clv_n": len(clv_vals) if has_clv else None,
            "clv_excluded_missing_book": clv_excluded if has_clv else None,
        }
    return out


def run(verbose: bool = True) -> dict:
    j = _joined_rows()
    if j.empty:
        msg = ("no market_history.csv rows to backtest — run "
               "`python3 -m club_soccer.fetch_fdcouk` first")
        if verbose:
            print(f"backtest_market: {msg}")
        return {"n_matched": 0, "note": msg}

    per_comp = {}
    for comp, grp in j.groupby("competition"):
        per_comp[comp] = int(len(grp))

    model_ll = _log_loss(j["p_home"], j["p_draw"], j["p_away"], j["actual"])
    valid_close = j[["p_close_h", "p_close_d", "p_close_a"]].notna().all(axis=1)
    market_ll = (_log_loss(j.loc[valid_close, "p_close_h"], j.loc[valid_close, "p_close_d"],
                           j.loc[valid_close, "p_close_a"], j.loc[valid_close, "actual"])
                if valid_close.any() else None)

    sim_1x2 = _sim_market(
        j, "1x2", ["home", "draw", "away"],
        p_model_cols={"home": "p_home", "draw": "p_draw", "away": "p_away"},
        p_close_cols={"home": "p_close_h", "draw": "p_close_d", "away": "p_close_a"},
        odds_close_cols={"home": "psc_h", "draw": "psc_d", "away": "psc_a"},
        p_b365_cols={"home": "p_b365_h", "draw": "p_b365_d", "away": "p_b365_a"},
        odds_b365_cols={"home": "b365_h", "draw": "b365_d", "away": "b365_a"},
        win_fn=lambda jj, side: jj["actual"] == {"home": 0, "draw": 1, "away": 2}[side],
    )
    j["_p_under25"] = 1.0 - j["p_over25"]  # model prob for "under", since the
                                            # row only carries p_over25 natively
    sim_ou25 = _sim_market(
        j, "total", ["over", "under"],
        p_model_cols={"over": "p_over25", "under": "_p_under25"},
        p_close_cols={"over": "p_close_over25", "under": "p_close_under25"},
        odds_close_cols={"over": "b365_over25", "under": "b365_under25"},
        p_b365_cols=None, odds_b365_cols=None,  # no separate closing-totals feed — see docstring
        win_fn=lambda jj, side: (jj["total_goals"] > 2.5) if side == "over" else (jj["total_goals"] <= 2.5),
    )

    report = {
        "n_matched": int(len(j)),
        "n_by_competition": per_comp,
        "model_log_loss_1x2": round(model_ll, 4),
        "market_log_loss_1x2_devigged_pinnacle_closing": (
            round(market_ll, 4) if market_ll is not None else None),
        "market_log_loss_n": int(valid_close.sum()),
        "simulated_betting": {"1x2": sim_1x2, "total_over_under_2_5": sim_ou25},
    }

    if verbose:
        print(f"Market-anchored backtest: {report['n_matched']} matched rows")
        for comp, n in sorted(per_comp.items()):
            print(f"  {comp:25s} n={n}")
        print(f"\n  Model 1X2 log-loss:  {report['model_log_loss_1x2']:.4f}")
        if market_ll is not None:
            print(f"  Market (Pinnacle closing, devigged) log-loss: {market_ll:.4f} "
                  f"(n={report['market_log_loss_n']})")
        else:
            print("  Market log-loss: n/a (no Pinnacle closing odds matched)")
        for market_name, sim in report["simulated_betting"].items():
            print(f"\n  {market_name}:")
            for thr, s in sim.items():
                print(f"    edge>={thr}: n_bets={s['n_bets']} "
                      f"flat_roi={s['flat_roi']} kelly_roi={s['kelly_roi']} "
                      f"clv_mean={s['clv_mean']} clv_frac_pos={s['clv_frac_positive']} "
                      f"(clv n={s['clv_n']}, excluded={s['clv_excluded_missing_book']})")

    DATA.mkdir(exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    run(verbose=True)


if __name__ == "__main__":
    main()
