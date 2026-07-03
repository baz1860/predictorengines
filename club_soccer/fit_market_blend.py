#!/usr/bin/env python3
"""P6.3 — fit the model<->market logit blend weight for club soccer.

Joins walk-forward predictions (validate.walk_forward, extended with
p_over25 in P1.4) to data/market_history.csv's Bet365 PRE-MATCH odds
(the "current" odds available at bet time — closing odds are excluded,
they're a teacher/benchmark, never a feature per the plan's guardrail).

Time-series CV over season splits (train seasons < S, test == S, S in
{2024, 2025}): grid-search w on train, evaluate held-out log-loss on test,
for BOTH the 1X2 and OU2.5 markets separately. A market's blend is only
written as production-ready (`app/market_blend.DEFAULT_BLEND_ON`-eligible)
if it beats BOTH pure-model and pure-market held-out log-loss on EVERY
split — otherwise it's recorded with the honest numbers and left off.

This is a PRICING-layer weight (used at edge/pricing time only) — never fed
back into model.fit(). Stored in data/market_blend_suite.json under
"club_soccer" (1X2) and "club_soccer_ou25" (totals), which
app/market_blend.py already reads generically.
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

from app.market_blend import blend_probs, blend_two
from . import validate as V

HERE = Path(__file__).resolve().parent
MARKET_HISTORY = HERE / "data" / "market_history.csv"
WEIGHTS_FILE = ROOT / "data" / "market_blend_suite.json"
REPORT_FILE = HERE / "data" / "market_blend_report.json"

W_GRID = np.linspace(0.0, 1.0, 21)   # w = weight on the MODEL (1.0 = pure model)
SPLITS = (2024, 2025)


def _season_of(date_str: str) -> int:
    d = pd.Timestamp(date_str)
    return d.year if d.month >= 7 else d.year - 1


def _devig3(oh, od, oa):
    inv = np.array([1.0 / oh, 1.0 / od, 1.0 / oa])
    p = inv / inv.sum()
    return p[0], p[1], p[2]


def _joined() -> pd.DataFrame:
    rows, _ = V.walk_forward(verbose=False)
    if not rows or not MARKET_HISTORY.exists():
        return pd.DataFrame()
    pred = pd.DataFrame(rows)
    mh = pd.read_csv(MARKET_HISTORY)
    if mh.empty:
        return pd.DataFrame()
    j = pred.merge(mh, left_on=["date", "home", "away"],
                   right_on=["match_date", "home", "away"], how="inner")
    if j.empty:
        return j
    has_1x2 = j[["b365_h", "b365_d", "b365_a"]].notna().all(axis=1)
    ph, pd_, pa = np.full(len(j), np.nan), np.full(len(j), np.nan), np.full(len(j), np.nan)
    for i in j.index[has_1x2]:
        r = j.loc[i]
        h, d, a = _devig3(float(r.b365_h), float(r.b365_d), float(r.b365_a))
        pos = j.index.get_loc(i)
        ph[pos], pd_[pos], pa[pos] = h, d, a
    j["p_market_h"], j["p_market_d"], j["p_market_a"] = ph, pd_, pa

    has_ou = j[["b365_over25", "b365_under25"]].notna().all(axis=1)
    p_over = np.where(has_ou,
                      1.0 / j["b365_over25"] / (1.0 / j["b365_over25"] + 1.0 / j["b365_under25"]),
                      np.nan)
    j["p_market_over25"] = p_over
    j["season"] = j["date"].map(_season_of)
    return j


def _ll_1x2(rows: pd.DataFrame, w: float) -> float:
    ll, n = 0.0, 0
    for r in rows.itertuples(index=False):
        p = blend_probs([r.p_home, r.p_draw, r.p_away],
                        [r.p_market_h, r.p_market_d, r.p_market_a], w)
        ll += -np.log(max(p[int(r.actual)], 1e-12))
        n += 1
    return ll / n if n else float("nan")


def _ll_ou25(rows: pd.DataFrame, w: float) -> float:
    ll, n = 0.0, 0
    for r in rows.itertuples(index=False):
        p_over = blend_two(r.p_over25, r.p_market_over25, w)
        y = 1 if r.total_goals > 2.5 else 0
        p = p_over if y == 1 else (1.0 - p_over)
        ll += -np.log(max(p, 1e-12))
        n += 1
    return ll / n if n else float("nan")


def _fit_one_market(df: pd.DataFrame, ll_fn, market_cols: list[str], label: str) -> dict:
    rows = df.dropna(subset=market_cols)
    splits_out = []
    beats_both_every_split = True
    any_split = False
    for S in SPLITS:
        train = rows[rows["season"] < S]
        test = rows[rows["season"] == S]
        if len(train) < 200 or len(test) < 50:
            continue
        any_split = True
        best_w, best_ll = 1.0, float("inf")
        for w in W_GRID:
            ll = ll_fn(train, float(w))
            if ll < best_ll:
                best_w, best_ll = float(w), ll
        blend_ll = float(ll_fn(test, best_w))
        model_ll = float(ll_fn(test, 1.0))
        market_ll = float(ll_fn(test, 0.0))
        beats_both = bool(blend_ll < model_ll and blend_ll < market_ll)
        beats_both_every_split = beats_both_every_split and beats_both
        splits_out.append({"split": S, "train_n": int(len(train)), "test_n": int(len(test)),
                           "w": round(best_w, 3), "blend_logloss": round(blend_ll, 4),
                           "model_logloss": round(model_ll, 4), "market_logloss": round(market_ll, 4),
                           "beats_both": beats_both})
    # Production weight: refit on ALL available data once CV has validated (or not) the approach.
    final_w, final_ll = 1.0, float("inf")
    for w in W_GRID:
        ll = ll_fn(rows, float(w))
        if ll < final_ll:
            final_w, final_ll = float(w), ll
    promotes = any_split and beats_both_every_split
    return {"market": label, "n": int(len(rows)), "final_w": round(final_w, 3),
           "splits": splits_out, "would_promote": promotes}


def fit_market_blend(verbose: bool = True, write: bool = False) -> dict:
    df = _joined()
    if df.empty:
        msg = "no matched rows (need market_history.csv + walk-forward) — run fetch_fdcouk.py first"
        if verbose:
            print(f"fit_market_blend: {msg}")
        return {"note": msg}

    r_1x2 = _fit_one_market(df, _ll_1x2, ["p_market_h", "p_market_d", "p_market_a"], "1x2")
    r_ou25 = _fit_one_market(df, _ll_ou25, ["p_market_over25"], "ou25")

    if verbose:
        for r in (r_1x2, r_ou25):
            print(f"\n{r['market']} market blend (n={r['n']}):")
            for s in r["splits"]:
                print(f"  split {s['split']}: w={s['w']} blend={s['blend_logloss']:.4f} "
                      f"model={s['model_logloss']:.4f} market={s['market_logloss']:.4f} "
                      f"beats_both={s['beats_both']}")
            print(f"  production w (refit on all data): {r['final_w']}  "
                  f"-> {'PROMOTE' if r['would_promote'] else 'reject — keep OFF'}")

    payload = {"1x2": r_1x2, "ou25": r_ou25}
    HERE.joinpath("data").mkdir(exist_ok=True)
    REPORT_FILE.write_text(json.dumps(payload, indent=2))

    if write:
        weights = {}
        if WEIGHTS_FILE.exists():
            try:
                weights = json.loads(WEIGHTS_FILE.read_text())
            except Exception:
                weights = {}
        weights["club_soccer"] = r_1x2["final_w"]
        weights["club_soccer_ou25"] = r_ou25["final_w"]
        WEIGHTS_FILE.parent.mkdir(exist_ok=True)
        WEIGHTS_FILE.write_text(json.dumps(weights, indent=2))
        if verbose:
            print(f"\nWrote weights -> {WEIGHTS_FILE}")
            print("Note: app/market_blend.DEFAULT_BLEND_ON does NOT include "
                  "club_soccer — promotion (adding it there) is a deliberate "
                  "code change per plan Sec 12, not automatic from this script.")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write fitted weights to data/market_blend_suite.json "
                         "(still requires a code change to DEFAULT_BLEND_ON to activate)")
    args = ap.parse_args()
    fit_market_blend(write=args.write)


if __name__ == "__main__":
    main()
