#!/usr/bin/env python3
"""Empirical discrete margin/total PMFs — the key-number core of the NFL engine.

Do NOT use a normal distribution anywhere a bet is priced. NFL margins pile up
on key numbers (3, 7, 6, 10, 14, 4) because scoring is discrete (TD+XP=7,
FG=3, safety=2, etc). A normal approximation smears that mass away and misses
almost all of the value in half-point line placement around 3 and 7. This
module builds P(home_margin = k | closing spread ~= s) and P(total = t |
closing total ~= T) directly from history, via kernel-smoothed empirical
histograms, so the discreteness (and its dependence on how close the game is
expected to be) comes straight from data.

Sign convention (matches nflverse `spread_line`): positive spread_line means
the HOME team is favored by that many points; home_margin = home_score -
away_score. So a home line of +3.0 means "home favored by 3": home covers if
margin > 3, pushes if margin == 3, loses the spread if margin < 3.

Central statistical assumption (stated here, verified in validate.py): the
distribution of margins conditioned on a *model-predicted* spread of value s
equals the distribution conditioned on a *market* closing spread of the same
value s. This holds to the extent the model is well-calibrated in the mean.
If validate.py's cover-probability reliability check finds this is violated,
widen the PMF via `inflation_k` (convolution with a small discrete kernel) —
see `widen()` below. Default inflation_k = 1.0 (no widening).

Usage:
  python3 -m nfl.margin_dist --fit             # fit + write data/margin_pmf.json
  python3 -m nfl.margin_dist --selftest        # sanity-anchor checks (Phase 2 gate)
  python3 -m nfl.margin_dist --show -3         # print margin PMF near s=-3
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES_CSV = os.path.join(HERE, "data", "games.csv")
PMF_JSON = os.path.join(HERE, "data", "margin_pmf.json")

FIRST_SEASON = 2003
MARGIN_SUPPORT = list(range(-60, 61))
TOTAL_SUPPORT = list(range(0, 101))
SPREAD_GRID = [round(-20.0 + 0.5 * i, 1) for i in range(81)]     # -20.0 .. +20.0
TOTAL_GRID = list(range(28, 62))                                  # 28 .. 61
DEFAULT_BANDWIDTH = 3.0
DEFAULT_SHRINKAGE = 0.20
CANDIDATE_BANDWIDTHS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def _tricube(u: np.ndarray) -> np.ndarray:
    u = np.clip(np.abs(u), 0.0, 1.0)
    return (1.0 - u ** 3) ** 3


def _weighted_pmf(values: np.ndarray, weights: np.ndarray, support: list[int]) -> np.ndarray:
    support = np.asarray(support)
    counts = np.zeros(len(support), dtype=float)
    # values already clipped to support range by caller
    idx = values - support[0]
    np.add.at(counts, idx, weights)
    total = counts.sum()
    if total <= 0:
        return np.full(len(support), 1.0 / len(support))
    return counts / total


def _unconditional_pmf(values: np.ndarray, support: list[int]) -> np.ndarray:
    return _weighted_pmf(values, np.ones(len(values)), support)


def _fit_one_axis(line: np.ndarray, outcome: np.ndarray, grid: list[float],
                  support: list[int], bandwidth: float, shrinkage: float) -> dict:
    lo, hi = support[0], support[-1]
    outcome = np.clip(np.round(outcome).astype(int), lo, hi)
    uncond = _unconditional_pmf(outcome, support)

    # Residual distribution: how far actual outcomes land from THEIR OWN
    # closing line, pooled across every historical line value. Its own
    # near-zero spikes already encode the average (across all common lines)
    # key-number push/discreteness effect. Shifting *this* by round(s0) and
    # blending it in stabilises sparse buckets (e.g. huge favourites, where
    # few historical games share a similar spread) WITHOUT transplanting a
    # single key number's spike (e.g. margin==3) to an unrelated query point
    # — key numbers are anchored at their absolute values (3, 7, 10, 14...),
    # not at (spread + key number), which is why the shift must act on
    # residuals, not on the raw unconditional PMF.
    residual_support = list(range(2 * lo, 2 * hi + 1))
    resid = np.clip(np.round(outcome - np.round(line)).astype(int), residual_support[0], residual_support[-1])
    resid_pmf = _unconditional_pmf(resid, residual_support)

    pmfs = []
    for s0 in grid:
        w = _tricube((line - s0) / bandwidth)
        cond = _weighted_pmf(outcome, w, support) if w.sum() > 1e-9 else uncond
        shift = int(round(s0))
        shifted = np.zeros(len(support))
        for i, k in enumerate(support):
            j = k - shift  # residual value that would land on outcome k
            if residual_support[0] <= j <= residual_support[-1]:
                shifted[i] = resid_pmf[j - residual_support[0]]
        if shifted.sum() > 0:
            shifted = shifted / shifted.sum()
        else:
            shifted = uncond
        blended = (1.0 - shrinkage) * cond + shrinkage * shifted
        blended = blended / blended.sum()
        pmfs.append(blended.tolist())
    return {"grid": grid, "support": support, "pmf": pmfs, "unconditional": uncond.tolist(),
            "bandwidth": bandwidth, "shrinkage": shrinkage}


def _loglik(line: np.ndarray, outcome: np.ndarray, grid: list[float], support: list[int],
            bandwidth: float, shrinkage: float, train_mask: np.ndarray, test_mask: np.ndarray) -> float:
    """CV log-likelihood: fit on train_mask, score held-out games by nearest-grid PMF."""
    axis = _fit_one_axis(line[train_mask], outcome[train_mask], grid, support, bandwidth, shrinkage)
    pmf = np.array(axis["pmf"])
    lo = support[0]
    grid_arr = np.array(grid)
    ll = 0.0
    n = 0
    for s0, y in zip(line[test_mask], outcome[test_mask]):
        gi = int(np.argmin(np.abs(grid_arr - s0)))
        yk = int(np.clip(round(y), support[0], support[-1])) - lo
        p = max(pmf[gi, yk], 1e-6)
        ll += np.log(p)
        n += 1
    return ll / max(n, 1)


def _tune_bandwidth(line: np.ndarray, outcome: np.ndarray, grid: list[float], support: list[int],
                    shrinkage: float, seed: int = 0) -> float:
    """Pick bandwidth by 5-fold CV log-likelihood over CANDIDATE_BANDWIDTHS."""
    rng = np.random.default_rng(seed)
    n = len(line)
    folds = rng.integers(0, 5, size=n)
    best_bw, best_ll = DEFAULT_BANDWIDTH, -np.inf
    for bw in CANDIDATE_BANDWIDTHS:
        lls = []
        for f in range(5):
            test_mask = folds == f
            train_mask = ~test_mask
            if test_mask.sum() < 20 or train_mask.sum() < 200:
                continue
            lls.append(_loglik(line, outcome, grid, support, bw, shrinkage, train_mask, test_mask))
        if lls and np.mean(lls) > best_ll:
            best_ll = np.mean(lls)
            best_bw = bw
    return best_bw


def fit(since: int = FIRST_SEASON, tune: bool = True) -> dict:
    g = pd.read_csv(GAMES_CSV)
    g = g[(g["season"] >= since) & g["spread_line"].notna() & g["total_line"].notna()].copy()
    margin = (g["home_score"] - g["away_score"]).values.astype(float)
    total = (g["home_score"] + g["away_score"]).values.astype(float)
    spread_line = g["spread_line"].values.astype(float)
    total_line = g["total_line"].values.astype(float)

    bw_margin = _tune_bandwidth(spread_line, margin, SPREAD_GRID, MARGIN_SUPPORT,
                                DEFAULT_SHRINKAGE) if tune else DEFAULT_BANDWIDTH
    margin_axis = _fit_one_axis(spread_line, margin, SPREAD_GRID, MARGIN_SUPPORT,
                                bw_margin, DEFAULT_SHRINKAGE)

    bw_total = _tune_bandwidth(total_line, total, TOTAL_GRID, TOTAL_SUPPORT,
                               DEFAULT_SHRINKAGE) if tune else DEFAULT_BANDWIDTH
    total_axis = _fit_one_axis(total_line, total, TOTAL_GRID, TOTAL_SUPPORT,
                               bw_total, DEFAULT_SHRINKAGE)

    return {
        "since": since, "n_games": int(len(g)), "inflation_k": 1.0,
        "margin": margin_axis, "total": total_axis,
    }


def load(path: str = PMF_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def _interp_pmf(axis: dict, s: float) -> np.ndarray:
    """Linearly interpolate the PMF between the two nearest grid points, then
    renormalise (interpolation of two simplex points stays in the simplex, so
    this is only needed for float safety)."""
    grid = np.array(axis["grid"])
    pmf = np.array(axis["pmf"])
    s = float(np.clip(s, grid[0], grid[-1]))
    if s <= grid[0]:
        p = pmf[0]
    elif s >= grid[-1]:
        p = pmf[-1]
    else:
        i = int(np.searchsorted(grid, s, side="right") - 1)
        i = min(max(i, 0), len(grid) - 2)
        s0, s1 = grid[i], grid[i + 1]
        t = (s - s0) / (s1 - s0) if s1 != s0 else 0.0
        p = (1 - t) * pmf[i] + t * pmf[i + 1]
    p = np.clip(p, 0.0, None)
    return p / p.sum()


def widen(p: np.ndarray, inflation_k: float) -> np.ndarray:
    """Convolve a PMF with a small discrete triangular kernel to inflate
    variance by roughly `inflation_k` (1.0 = no-op). Used if validate.py finds
    model-conditioned reliability is worse than market-conditioned reliability
    (the model is noisier than the market it borrows the PMF shape from)."""
    if inflation_k <= 1.0:
        return p
    width = max(1, int(round((inflation_k - 1.0) * 4)))
    kernel = np.array([width + 1 - abs(k) for k in range(-width, width + 1)], dtype=float)
    kernel = kernel / kernel.sum()
    out = np.convolve(p, kernel, mode="same")
    return out / out.sum()


def margin_pmf_at(pmf_data: dict, s: float) -> tuple[np.ndarray, list[int]]:
    axis = pmf_data["margin"]
    p = _interp_pmf(axis, s)
    p = widen(p, pmf_data.get("inflation_k", 1.0))
    return p, axis["support"]


def total_pmf_at(pmf_data: dict, t: float) -> tuple[np.ndarray, list[int]]:
    axis = pmf_data["total"]
    p = _interp_pmf(axis, t)
    p = widen(p, pmf_data.get("inflation_k", 1.0))
    return p, axis["support"]


# ── pricing helpers ────────────────────────────────────────────────────────
def _cdf_gt(p: np.ndarray, support: list[int], x: float) -> float:
    """P(outcome > x)."""
    support = np.asarray(support)
    return float(p[support > x].sum())


def _pmf_eq(p: np.ndarray, support: list[int], x: int) -> float:
    support = np.asarray(support)
    if x < support[0] or x > support[-1]:
        return 0.0
    return float(p[support == x][0]) if (support == x).any() else 0.0


def cover_probs(pmf_data: dict, predicted_margin: float, line: float) -> dict:
    """Home-side pricing for a spread `line` (home-perspective, positive =
    home favored, matching spread_line). Returns home-cover / push / away-cover."""
    p, support = margin_pmf_at(pmf_data, predicted_margin)
    push = _pmf_eq(p, support, int(round(line))) if float(line).is_integer() else 0.0
    home_cover = _cdf_gt(p, support, line)
    away_cover = 1.0 - home_cover - push
    return {"home_cover": home_cover, "push": push, "away_cover": max(away_cover, 0.0)}


def moneyline_probs(pmf_data: dict, predicted_margin: float) -> dict:
    p, support = margin_pmf_at(pmf_data, predicted_margin)
    tie = _pmf_eq(p, support, 0)
    home_win = _cdf_gt(p, support, 0)
    away_win = 1.0 - home_win - tie
    return {"home_win": home_win, "tie": tie, "away_win": max(away_win, 0.0)}


def total_probs(pmf_data: dict, predicted_total: float, line: float) -> dict:
    p, support = total_pmf_at(pmf_data, predicted_total)
    push = _pmf_eq(p, support, int(round(line))) if float(line).is_integer() else 0.0
    over = _cdf_gt(p, support, line)
    under = 1.0 - over - push
    return {"over": over, "push": push, "under": max(under, 0.0)}


# ── Phase 2 sanity anchors ───────────────────────────────────────────────────
def selftest(pmf_data: dict | None = None) -> tuple[int, int]:
    pmf_data = pmf_data or load()
    ok, bad = 0, 0

    def check(name, cond, detail=""):
        nonlocal ok, bad
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            bad += 1
            print(f"  FAIL  {name}  {detail}")

    uncond = np.array(pmf_data["margin"]["unconditional"])
    support = pmf_data["margin"]["support"]

    def p_abs(k):
        return _pmf_eq(uncond, support, k) + _pmf_eq(uncond, support, -k)

    check("unconditional P(|margin|=3) in [12,18]%", 0.12 <= p_abs(3) <= 0.18, f"{p_abs(3):.3f}")
    check("unconditional P(|margin|=7) in [7,12]%", 0.07 <= p_abs(7) <= 0.12, f"{p_abs(7):.3f}")
    p3, p4, p6, p7, p10, p14 = p_abs(3), p_abs(4), p_abs(6), p_abs(7), p_abs(10), p_abs(14)
    p5 = p_abs(5)
    check("key numbers (3,6,7,10,14) exceed neighbouring non-key (5)",
          min(p3, p6, p7, p10, p14) > p5, f"3={p3:.3f} 6={p6:.3f} 7={p7:.3f} 10={p10:.3f} 14={p14:.3f} 5={p5:.3f}")
    check("P(margin=0) < 0.5%", _pmf_eq(uncond, support, 0) < 0.005,
          f"{_pmf_eq(uncond, support, 0):.4f}")

    # s=+3.0: home favored by 3 (nflverse spread_line convention: positive =
    # home favorite). The mass sitting exactly on margin=3 is the classic
    # "key number 3" effect: covering the cheaper half-point line (+2.5, i.e.
    # margin >= 3) is far more likely than covering the pricier one (+3.5,
    # i.e. margin >= 4) — the gap between them IS P(margin == 3).
    p, sup = margin_pmf_at(pmf_data, 3.0)
    cover_p25 = _cdf_gt(p, sup, 2.5)
    cover_p35 = _cdf_gt(p, sup, 3.5)
    gap = cover_p25 - cover_p35
    check("half-point gap at s=+3 (cover +2.5 vs +3.5) in [0.07, 0.13]",
          0.07 <= gap <= 0.13, f"gap={gap:.3f}")
    push3 = _pmf_eq(p, sup, 3)
    check("push rate at s=+3, line=+3 in [6,12]%", 0.06 <= push3 <= 0.12, f"{push3:.3f}")

    return ok, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--since", type=int, default=FIRST_SEASON)
    ap.add_argument("--no-tune", action="store_true", help="skip bandwidth CV (use default 3.0)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", type=float, default=None, help="print margin PMF near this spread")
    args = ap.parse_args()

    if args.fit:
        data = fit(since=args.since, tune=not args.no_tune)
        os.makedirs(os.path.dirname(PMF_JSON), exist_ok=True)
        with open(PMF_JSON, "w") as f:
            json.dump(data, f)
        print(f"fitted margin PMF ({data['n_games']} games since {data['since']}): "
              f"bandwidth={data['margin']['bandwidth']}, total bandwidth={data['total']['bandwidth']}")
        return 0

    if args.selftest:
        ok, bad = selftest()
        print(f"\n{ok} passed, {bad} failed")
        return 1 if bad else 0

    if args.show is not None:
        data = load()
        p, sup = margin_pmf_at(data, args.show)
        print(f"margin PMF near predicted spread {args.show:+.1f}:")
        for k, prob in zip(sup, p):
            if prob >= 0.01:
                print(f"  {k:+4d}  {prob:6.2%}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
