"""Held-out validation for push-aware CFB market probability challengers.

The champion uses the existing continuous normal approximation. The challenger
uses integer empirical residuals fitted only on the selection seasons, which
gives integer spread/total lines explicit push mass. Platt calibration is also
fit only on selection rows. All model choice and calibration parameters are
then frozen before the 2025 holdout is scored.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset_fingerprint as DF
from . import elo as E
from . import validate as V
from .ats_backtest import SPREADS_CSV
from .predictor import load_blend_weight
from .totals_backtest import TOTALS_CSV

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "data" / "market_validation_2025.json"
DECIMAL_MINUS_110 = 1.0 + 100.0 / 110.0


def _phi(value):
    values = np.asarray(value, dtype=float)
    return np.array([0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
                     for x in values])


def residual_pmf(actual: pd.Series, predicted: pd.Series) -> dict[int, float]:
    """Integer residual PMF around the rounded forecast, fitted pre-holdout."""
    residual = (actual.astype(int) - np.rint(predicted).astype(int)).astype(int)
    counts = residual.value_counts().sort_index()
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("cannot fit an empty residual distribution")
    return {int(value): float(count / total) for value, count in counts.items()}


def discrete_three_way(predicted: pd.Series, line: pd.Series,
                       pmf: dict[int, float]) -> np.ndarray:
    """Return P(primary side win, push, loss) on the integer score lattice.

    For spreads the primary side is home and ``line`` is the home handicap.
    For totals the primary side is over and callers pass ``-total_line`` so the
    comparison remains ``projected + line``.
    """
    centers = np.rint(np.asarray(predicted, dtype=float)).astype(int)
    lines = np.asarray(line, dtype=float)
    out = np.zeros((len(centers), 3), dtype=float)
    for residual, mass in pmf.items():
        value = centers + int(residual) + lines
        out[:, 0] += mass * (value > 0)
        out[:, 1] += mass * np.isclose(value, 0.0, atol=1e-9)
        out[:, 2] += mass * (value < 0)
    return out


def normal_three_way(predicted: pd.Series, line: pd.Series,
                     sigma: pd.Series) -> np.ndarray:
    z = (-np.asarray(line, dtype=float) - np.asarray(predicted, dtype=float)) \
        / np.asarray(sigma, dtype=float)
    p_win = 1.0 - _phi(z)
    return np.column_stack((p_win, np.zeros(len(p_win)), 1.0 - p_win))


def fit_platt(probability: np.ndarray, outcome: np.ndarray) -> dict:
    """Fit intercept/slope on logit probability using damped Newton steps."""
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(outcome, dtype=float)
    x = np.column_stack((np.ones(len(p)), np.log(p / (1.0 - p))))
    beta = np.array([0.0, 1.0])
    ridge = np.diag([1e-8, 1e-8])
    for _ in range(100):
        eta = np.clip(x @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(mu * (1.0 - mu), 1e-8, None)
        hessian = x.T @ (weights[:, None] * x) + ridge
        step = np.linalg.solve(hessian, x.T @ (y - mu))
        beta += step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return {"intercept": float(beta[0]), "slope": float(beta[1]),
            "n": int(len(y))}


def apply_platt(probability: np.ndarray, params: dict) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1.0 - p))
    eta = np.clip(params["intercept"] + params["slope"] * logit, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-eta))


def calibrate_three_way(probabilities: np.ndarray, params: dict) -> np.ndarray:
    """Calibrate conditional win probability while preserving push mass."""
    out = np.asarray(probabilities, dtype=float).copy()
    nonpush = np.clip(out[:, 0] + out[:, 2], 1e-12, 1.0)
    conditional = out[:, 0] / nonpush
    calibrated = apply_platt(conditional, params)
    out[:, 0] = calibrated * nonpush
    out[:, 2] = (1.0 - calibrated) * nonpush
    return out


def _actual_three_way(value: pd.Series) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    return np.column_stack((values > 0, np.isclose(values, 0.0), values < 0)).astype(float)


def _reliability(probability: np.ndarray, outcome: np.ndarray) -> tuple[float, list[dict]]:
    bins = np.linspace(0.0, 1.0, 11)
    rows, ece = [], 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = ((probability >= lower) &
                (probability <= upper if upper == 1.0 else probability < upper))
        if not mask.any():
            continue
        predicted = float(probability[mask].mean())
        observed = float(outcome[mask].mean())
        n = int(mask.sum())
        ece += n / len(probability) * abs(predicted - observed)
        rows.append({"lower": round(float(lower), 1), "upper": round(float(upper), 1),
                     "n": n, "predicted": round(predicted, 4),
                     "observed": round(observed, 4)})
    return float(ece), rows


def _roi(probabilities: np.ndarray, actual: np.ndarray, weeks: pd.Series,
         min_ev: float = 0.03) -> dict:
    profit = DECIMAL_MINUS_110 - 1.0
    ev_primary = probabilities[:, 0] * profit - probabilities[:, 2]
    ev_other = probabilities[:, 2] * profit - probabilities[:, 0]
    primary = ev_primary >= ev_other
    edge = np.maximum(ev_primary, ev_other)
    mask = edge >= min_ev
    if not mask.any():
        return {"min_ev": min_ev, "n": 0}
    chosen_win = np.where(primary, actual[:, 0], actual[:, 2]).astype(bool)
    push = actual[:, 1].astype(bool)
    pnl = np.where(push, 0.0, np.where(chosen_win, profit, -1.0))[mask]
    selected_weeks = np.asarray(weeks)[mask]
    weekly = [pnl[selected_weeks == week] for week in np.unique(selected_weeks)]
    rng = np.random.default_rng(20260802)
    samples = []
    for _ in range(2000):
        picked = rng.integers(0, len(weekly), len(weekly))
        samples.append(float(np.concatenate([weekly[i] for i in picked]).mean()))
    selected_win = chosen_win[mask]
    selected_push = push[mask]
    return {
        "odds_assumption": "-110 both sides",
        "min_ev": min_ev, "n": int(mask.sum()),
        "won": int((selected_win & ~selected_push).sum()),
        "lost": int((~selected_win & ~selected_push).sum()),
        "push": int(selected_push.sum()),
        "roi": round(float(pnl.mean()), 4),
        "roi_ci_95": [round(float(np.quantile(samples, 0.025)), 4),
                      round(float(np.quantile(samples, 0.975)), 4)],
    }


def score(name: str, probabilities: np.ndarray, actual: np.ndarray,
          weeks: pd.Series) -> dict:
    brier = float(np.square(probabilities - actual).sum(axis=1).mean())
    nonpush = actual[:, 1] == 0
    conditional = probabilities[nonpush, 0] / np.clip(
        probabilities[nonpush, 0] + probabilities[nonpush, 2], 1e-12, 1.0)
    outcome = actual[nonpush, 0]
    ece, buckets = _reliability(conditional, outcome)
    calibration = fit_platt(conditional, outcome)
    return {
        "model": name, "n": int(len(actual)),
        "pushes": int(actual[:, 1].sum()),
        "multiclass_brier": round(brier, 5),
        "ece": round(ece, 5),
        "holdout_calibration": {
            "intercept": round(calibration["intercept"], 5),
            "slope": round(calibration["slope"], 5),
            "n": calibration["n"],
        },
        "reliability": buckets,
        "strategy": _roi(probabilities, actual, weeks),
    }


def _market_frame(frame: pd.DataFrame, market: str, weight: float) -> pd.DataFrame:
    if market == "spread":
        lines = pd.read_csv(SPREADS_CSV)
        out = frame.merge(lines, on=["season", "week", "home_team", "away_team"])
        out["predicted"] = weight * out["m_elo"] + (1.0 - weight) * out["m_pow"]
        out["line"] = out["home_line"]
        out["actual_value"] = out["margin"] + out["home_line"]
        out["residual_actual"] = out["margin"]
        out["sigma"] = out["sigma_margin"]
    else:
        lines = pd.read_csv(TOTALS_CSV)
        out = frame.merge(lines, on=["season", "week", "home_team", "away_team"])
        out["predicted"] = out["t_pow"]
        out["line"] = -out["total_line"]
        out["actual_value"] = out["total"] - out["total_line"]
        out["residual_actual"] = out["total"]
        out["sigma"] = out["sigma_total"]
    return out


def evaluate(selection_since: int = 2023, selection_until: int = 2024,
             holdout_season: int = 2025) -> dict:
    weight = load_blend_weight()
    # Totals line history has a 2020-24 gap. Start in 2018 so its calibration
    # can use the latest two genuinely pre-holdout line seasons (2018-19), while
    # residual distributions for both markets still use 2023-24 forecasts.
    frame = V.walk_forward(E.load_games(), min(selection_since, 2018),
                           quiet=True, w_elo=weight)
    report = {
        "selection_window": f"{selection_since}-{selection_until}",
        "holdout_season": holdout_season,
        "locked_w_elo": weight,
        "data_fingerprint": DF.compact_snapshot(),
        "markets": {},
    }
    for market in ("spread", "total"):
        data = _market_frame(frame, market, weight)
        residual_selection = frame[
            frame["season"].between(selection_since, selection_until)].copy()
        if market == "spread":
            residual_selection["predicted"] = (
                weight * residual_selection["m_elo"]
                + (1.0 - weight) * residual_selection["m_pow"])
            residual_selection["residual_actual"] = residual_selection["margin"]
        else:
            residual_selection["predicted"] = residual_selection["t_pow"]
            residual_selection["residual_actual"] = residual_selection["total"]

        calibration_pool = data[data["season"] <= selection_until].copy()
        available = sorted(int(value) for value in calibration_pool["season"].unique())
        if market == "spread":
            calibration_seasons = [value for value in available
                                   if value >= selection_since]
        else:
            calibration_seasons = available[-2:]
        selection = calibration_pool[
            calibration_pool["season"].isin(calibration_seasons)].copy()
        holdout = data[data["season"] == holdout_season].copy()
        if residual_selection.empty or selection.empty or holdout.empty:
            raise ValueError(f"empty {market} selection or holdout frame")

        pmf = residual_pmf(residual_selection["residual_actual"],
                           residual_selection["predicted"])
        normal_sel = normal_three_way(selection["predicted"], selection["line"],
                                      selection["sigma"])
        discrete_sel = discrete_three_way(selection["predicted"], selection["line"], pmf)
        actual_sel = _actual_three_way(selection["actual_value"])
        nonpush = actual_sel[:, 1] == 0
        normal_cal = fit_platt(normal_sel[nonpush, 0], actual_sel[nonpush, 0])
        discrete_nonpush = discrete_sel[nonpush, 0] / np.clip(
            discrete_sel[nonpush, 0] + discrete_sel[nonpush, 2], 1e-12, 1.0)
        discrete_cal = fit_platt(discrete_nonpush, actual_sel[nonpush, 0])

        normal = normal_three_way(holdout["predicted"], holdout["line"], holdout["sigma"])
        discrete = discrete_three_way(holdout["predicted"], holdout["line"], pmf)
        candidates = {
            "champion_normal": normal,
            "calibrated_normal": calibrate_three_way(normal, normal_cal),
            "discrete": discrete,
            "calibrated_discrete": calibrate_three_way(discrete, discrete_cal),
        }
        actual = _actual_three_way(holdout["actual_value"])
        scores = {name: score(name, probs, actual, holdout["week"])
                  for name, probs in candidates.items()}
        champion = scores["champion_normal"]
        calibrated_normal_score = scores["calibrated_normal"]
        discrete_score = scores["discrete"]
        challenger = scores["calibrated_discrete"]
        calibration_gate = (
            calibrated_normal_score["multiclass_brier"] < champion["multiclass_brier"]
            and calibrated_normal_score["ece"] <= champion["ece"]
        )
        discrete_gate = (
            discrete_score["multiclass_brier"] < champion["multiclass_brier"]
            and discrete_score["ece"] <= champion["ece"]
        )
        combined_gate = (
            challenger["multiclass_brier"]
            < calibrated_normal_score["multiclass_brier"]
            and challenger["ece"] <= calibrated_normal_score["ece"]
        )
        strategy = challenger["strategy"]
        betting_gate = bool(
            calibration_gate and discrete_gate and combined_gate
            and strategy.get("n", 0) >= 100
            and strategy.get("roi_ci_95", [-1.0])[0] > 0.0
        )
        report["markets"][market] = {
            "residual_selection_window": f"{selection_since}-{selection_until}",
            "residual_selection_rows": int(len(residual_selection)),
            "calibration_selection_seasons": calibration_seasons,
            "calibration_selection_rows": int(len(selection)),
            "holdout_rows": int(len(holdout)),
            "residual_pmf": {str(key): round(value, 8) for key, value in pmf.items()},
            "residual_summary": {
                "support": [min(pmf), max(pmf)],
                "mean": round(sum(key * value for key, value in pmf.items()), 4),
                "sd": round(math.sqrt(sum(
                    ((key - sum(k * v for k, v in pmf.items())) ** 2) * value
                    for key, value in pmf.items())), 4),
            },
            "selection_calibration": {
                "normal": {key: round(value, 6) if isinstance(value, float) else value
                           for key, value in normal_cal.items()},
                "discrete": {key: round(value, 6) if isinstance(value, float) else value
                             for key, value in discrete_cal.items()},
            },
            "scores": scores,
            "calibration_gate": calibration_gate,
            "discrete_gate": discrete_gate,
            "combined_incremental_gate": combined_gate,
            "promote_to_betting": betting_gate,
        }
    report["promote_runtime"] = all(
        market["promote_to_betting"] for market in report["markets"].values())
    report["verdict"] = (
        "promote calibrated discrete runtime" if report["promote_runtime"] else
        "retain champion runtime; challenger evidence does not clear held-out betting gate"
    )
    return report


def write_artifact(report: dict, path: str | Path = ARTIFACT) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def _print(report: dict) -> None:
    print(f"CFB market challengers · selection {report['selection_window']} · "
          f"holdout {report['holdout_season']} · w_elo={report['locked_w_elo']:.2f}")
    for market, result in report["markets"].items():
        print(f"\n{market.upper()} · {result['holdout_rows']} holdout rows")
        print(f"{'model':<24s}{'Brier':>9s}{'ECE':>9s}{'bets':>8s}{'ROI':>9s}  ROI 95% CI")
        for score_row in result["scores"].values():
            strategy = score_row["strategy"]
            if strategy.get("n", 0):
                ci = strategy["roi_ci_95"]
                roi_text = f"{strategy['roi']:+9.1%}  [{ci[0]:+.1%}, {ci[1]:+.1%}]"
            else:
                roi_text = f"{'—':>9s}  [no qualifying bets]"
            print(f"{score_row['model']:<24s}{score_row['multiclass_brier']:>9.5f}"
                  f"{score_row['ece']:>9.5f}{strategy.get('n', 0):>8d}{roi_text}")
        print(f"  calibration: {'PASS' if result['calibration_gate'] else 'FAIL'}"
              f" · discrete: {'PASS' if result['discrete_gate'] else 'FAIL'}"
              f" · incremental combination: "
              f"{'PASS' if result['combined_incremental_gate'] else 'FAIL'}"
              f" · betting: {'PASS' if result['promote_to_betting'] else 'FAIL'}")
    print(f"\nverdict: {report['verdict']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = evaluate()
    _print(report)
    if args.write:
        print(f"artifact -> {write_artifact(report)}")


if __name__ == "__main__":
    main()
