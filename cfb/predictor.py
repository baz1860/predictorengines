#!/usr/bin/env python3
"""Blended FBS match predictor: Elo (elo.py) + offense/defense power ratings
(power.py), averaged 50/50. Predicts win probability, spread, and total.

Usage:
  python3 predictor.py "Ohio State" "Michigan"             # team 1 at home
  python3 predictor.py "Georgia" "Texas" --neutral
  python3 predictor.py ... --model elo|power|blend         # default blend
  python3 predictor.py --backtest [--since 2023]           # walk-forward eval
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

import elo as E
import power as P
import epa as X

# V3 M6: the elo/power blend weight is tunable. `w_elo` is the weight on Elo for
# the win-prob and margin blend (power always supplies the total). It defaults to
# 0.50 (the V2 50/50 blend) so default behaviour is unchanged. A validated weight
# can be opted into by writing cfb/data/blend_weight.json {"w_elo": <float>} via
# `python3 validate.py --tune-blend --write` — see V3_NOTES.md (M6).
#
# Model-sprint note: EPA/PPA is available as an explicit challenger (`epa` or
# `blend3`) but gets zero default weight until the walk-forward gate proves lift.
DEFAULT_W_ELO = 0.50
DEFAULT_W_EPA = 0.00
_BLEND_WEIGHT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "data", "blend_weight.json")


def load_blend_weight() -> float:
    """Stored elo blend weight, or DEFAULT_W_ELO (0.5) when not opted into."""
    try:
        if os.path.exists(_BLEND_WEIGHT_FILE):
            w = float(json.load(open(_BLEND_WEIGHT_FILE))["w_elo"])
            return min(max(w, 0.0), 1.0)
    except Exception:
        pass
    return DEFAULT_W_ELO


def _normalise_weights(weights: dict) -> dict:
    vals = {
        "elo": max(0.0, float(weights.get("elo", 0.0))),
        "power": max(0.0, float(weights.get("power", 0.0))),
        "epa": max(0.0, float(weights.get("epa", 0.0))),
    }
    s = sum(vals.values())
    if s <= 0:
        return {"elo": DEFAULT_W_ELO, "power": 1.0 - DEFAULT_W_ELO, "epa": 0.0}
    return {k: v / s for k, v in vals.items()}


def load_blend_weights() -> dict:
    """Stored model-stack weights, backwards-compatible with {"w_elo": x}.

    Existing installs only write `w_elo`; in that case EPA remains off and the
    power weight is `1 - w_elo`. A future gate may write `{weights:{...}}`.
    """
    try:
        if os.path.exists(_BLEND_WEIGHT_FILE):
            raw = json.load(open(_BLEND_WEIGHT_FILE))
            if isinstance(raw.get("weights"), dict):
                return _normalise_weights(raw["weights"])
            w_elo = min(max(float(raw.get("w_elo", DEFAULT_W_ELO)), 0.0), 1.0)
            w_epa = min(max(float(raw.get("w_epa", DEFAULT_W_EPA)), 0.0), 1.0)
            return _normalise_weights({
                "elo": w_elo,
                "power": max(0.0, 1.0 - w_elo - w_epa),
                "epa": w_epa,
            })
    except Exception:
        pass
    return {"elo": DEFAULT_W_ELO, "power": 1.0 - DEFAULT_W_ELO, "epa": DEFAULT_W_EPA}


def blend_predict(eparams, pparams, t1, t2, neutral=False, model="blend",
                  w_elo=None, xparams=None, weights=None):
    games, ratings, slope, sigma_e = eparams
    pe = E.predict(ratings, slope, sigma_e, t1, t2, neutral)
    pp = P.predict(pparams, t1, t2, neutral)
    px = None
    if model in ("epa", "blend3") or (weights and weights.get("epa", 0.0) > 0):
        if xparams is None:
            xparams = X.load_params()
        px = X.predict(xparams, t1, t2, neutral)
    if model == "elo":
        return {"p1": pe["p1"], "margin": pe["margin"], "total": pp["total"]}
    if model == "power":
        return {"p1": pp["p1"], "margin": pp["margin"], "total": pp["total"]}
    if model == "epa":
        return {"p1": px["p1"], "margin": px["margin"], "total": px["total"]}
    if model == "blend3":
        w = _normalise_weights(weights or load_blend_weights())
        total_w = w["power"] + w["epa"]
        total = pp["total"] if total_w <= 0 else (
            w["power"] * pp["total"] + w["epa"] * px["total"]) / total_w
        return {"p1": w["elo"] * pe["p1"] + w["power"] * pp["p1"] + w["epa"] * px["p1"],
                "margin": w["elo"] * pe["margin"] + w["power"] * pp["margin"] + w["epa"] * px["margin"],
                "total": total}
    w = load_blend_weight() if w_elo is None else float(w_elo)
    return {"p1": w * pe["p1"] + (1.0 - w) * pp["p1"],
            "margin": w * pe["margin"] + (1.0 - w) * pp["margin"],
            "total": pp["total"]}


def backtest(since=2023):
    games = E.load_games()
    carry, offs = E.season_priors()
    _, history = E.run_elo(games, record_pregame=True, carry=carry, prior_offsets=offs)
    diffs = np.array([h[2] for h in history])
    # spread map fitted only on pre-`since` data (no leakage)
    pre = games["season"] < since
    m_all = (games["home_points"] - games["away_points"]).values
    x, y = diffs[pre.values], m_all[pre.values]
    slope = float((x * y).sum() / (x * x).sum())

    ev = games[(games["season"] >= since) & (games["home"] != E.FCS) & (games["away"] != E.FCS)]
    print(f"Backtest: {len(ev)} FBS-vs-FBS games, seasons {since}-{int(games['season'].max())}")
    print("Power ratings refit before each week (walk-forward)...")

    rows = []
    pparams = None
    for (season, week, stype), wk in ev.groupby(["season", "week", "season_type"], sort=False):
        asof = wk["date"].min()
        try:
            pparams = P.fit(games, asof=asof)
        except ValueError:
            continue
        for r in wk.itertuples():
            if r.home not in pparams["teams"] or r.away not in pparams["teams"]:
                continue
            d = diffs[r.Index]
            p_elo = E.win_prob(d)
            m_elo = slope * d
            pp = P.predict(pparams, r.home, r.away, neutral=bool(r.neutral))
            actual_m = r.home_points - r.away_points
            actual_t = r.home_points + r.away_points
            rows.append({
                "p_elo": p_elo, "p_pow": pp["p1"], "m_elo": m_elo, "m_pow": pp["margin"],
                "t_pow": pp["total"], "actual_m": actual_m, "actual_t": actual_t,
            })
    df = pd.DataFrame(rows)
    df["p_blend"] = 0.5 * (df["p_elo"] + df["p_pow"])
    df["m_blend"] = 0.5 * (df["m_elo"] + df["m_pow"])
    res = (df["actual_m"] > 0).astype(float)

    print(f"\n{'model':<14s}{'accuracy':>9s}{'Brier':>8s}{'margin MAE':>12s}{'total MAE':>11s}")
    for name, pcol, mcol in [("Elo", "p_elo", "m_elo"), ("Power", "p_pow", "m_pow"),
                             ("50/50 blend", "p_blend", "m_blend")]:
        acc = ((df[pcol] > 0.5) == (res > 0.5)).mean()
        brier = ((df[pcol] - res) ** 2).mean()
        mae = (df[mcol] - df["actual_m"]).abs().mean()
        tmae = (df["t_pow"] - df["actual_t"]).abs().mean() if "pow" in mcol or "blend" in mcol else float("nan")
        t = f"{tmae:>11.2f}" if not math.isnan(tmae) else f"{'-':>11s}"
        print(f"{name:<14s}{acc:>9.1%}{brier:>8.4f}{mae:>12.2f}{t}")
    print(f"\n(binary Brier: 0.25 = coin flip, lower is better; "
          f"favourite-picks-all baseline acc = {max(res.mean(), 1 - res.mean()):.1%} for home side)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", nargs="*")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--model", choices=["elo", "power", "epa", "blend", "blend3"], default="blend")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--since", type=int, default=2023)
    args = ap.parse_args()

    if args.backtest:
        backtest(args.since)
        return
    if len(args.teams) != 2:
        raise SystemExit(__doc__)
    t1, t2 = args.teams
    eparams = E.build()
    pparams = P.load_params()
    xparams = X.load_params() if args.model in ("epa", "blend3") else None
    out = blend_predict(eparams, pparams, t1, t2, args.neutral, args.model,
                        xparams=xparams)
    venue = "neutral site" if args.neutral else f"{t1} at home"
    print(f"{t1} vs {t2} ({venue}, model={args.model})")
    print(f"  P({t1} win) = {out['p1']:.1%}   P({t2} win) = {1 - out['p1']:.1%}")
    print(f"  Spread: {t1} {-out['margin']:+.1f}   Total: {out['total']:.1f}")


if __name__ == "__main__":
    main()
