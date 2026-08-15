#!/usr/bin/env python3
"""Model-versus-market diagnostics with no production promotion authority.

This report answers whether published probabilities add information to the
market. It deliberately uses the median snapshot history only as a diagnostic
benchmark. Gate CLV comes from the separate raw per-book closing ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .market_settlement import devig, power_devig

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNTIME = Path(os.environ.get("CLUB_SOCCER_RUNTIME_DIR", str(DATA)))
FORECASTS = RUNTIME / "forecast_ledger.csv"
RESULTS = RUNTIME / "forecast_settlements.csv"
ODDS = DATA / "odds_history_club.csv"
REPORT = RUNTIME / "market_diagnostics.json"

SIDES_1X2 = ("home", "draw", "away")
SIDES_TOTAL = ("over", "under")
BLEND_GRID = np.linspace(0.0, 1.0, 21)
MIN_CHALLENGER_TRAIN = 100
MIN_CHALLENGER_TEST = 50
MIN_PROMOTION_WEEKS = 8


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""


def _number(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _published_cohort(kind: str = "latest_pre_kickoff") -> pd.DataFrame:
    if not FORECASTS.exists() or not RESULTS.exists():
        return pd.DataFrame()
    forecasts = pd.read_csv(FORECASTS, low_memory=False)
    results = pd.read_csv(RESULTS, low_memory=False)
    if forecasts.empty or results.empty:
        return pd.DataFrame()
    forecasts = forecasts[forecasts["primary_eligible"].astype(str).isin({"1", "1.0", "True", "true"})]
    forecasts = forecasts.sort_values("forecast_ts")
    if kind == "first_published":
        forecasts = forecasts.groupby("fixture_identity", as_index=False).first()
    elif kind == "latest_pre_kickoff":
        forecasts = forecasts.groupby("fixture_identity", as_index=False).last()
    elif kind == "t24":
        forecasts = _number(forecasts, ["lead_hours"])
        eligible = forecasts[forecasts["lead_hours"] >= 24].copy()
        if eligible.empty:
            return eligible
        idx = eligible.groupby("fixture_identity")["lead_hours"].idxmin()
        forecasts = eligible.loc[idx]
    else:
        raise ValueError(f"unknown cohort {kind!r}")
    merged = forecasts.merge(
        results[["fixture_identity", "actual_1x2", "over25_actual"]],
        on="fixture_identity", how="inner", validate="one_to_one",
    )
    return _number(merged, [
        "p_home", "p_draw", "p_away", "p_over25", "p_under25",
    ])


def _complete_snapshot_rows(odds: pd.DataFrame, sides: tuple[str, ...]) -> pd.DataFrame:
    """Latest complete median-odds snapshot per fixture, strictly pre-kickoff."""
    if odds.empty:
        return pd.DataFrame()
    keys = ["match_date", "competition", "home", "away", "snapshot_time"]
    pivot = odds.pivot_table(index=keys, columns="side", values="odds_median", aggfunc="last")
    if not set(sides).issubset(pivot.columns):
        return pd.DataFrame()
    pivot = pivot.dropna(subset=list(sides))
    pivot = pivot.rename(columns={side: f"odds_{side}" for side in sides}).reset_index()
    pivot["snapshot_time"] = pd.to_datetime(pivot["snapshot_time"], utc=True, errors="coerce")
    return pivot.dropna(subset=["snapshot_time"])


def _join_market(forecasts: pd.DataFrame, market: str) -> pd.DataFrame:
    if forecasts.empty or not ODDS.exists():
        return pd.DataFrame()
    sides = SIDES_1X2 if market == "1x2" else SIDES_TOTAL
    odds = pd.read_csv(ODDS, low_memory=False)
    odds = odds[odds["market"] == market].copy()
    odds = _number(odds, ["odds_median"])
    snapshots = _complete_snapshot_rows(odds, sides)
    if snapshots.empty:
        return snapshots
    left = forecasts.copy()
    left["match_date"] = left["match_date"].astype(str).str[:10]
    left["kickoff_dt"] = pd.to_datetime(left["kickoff_utc"], utc=True, errors="coerce")
    left = left.dropna(subset=["kickoff_dt"])
    keys = ["match_date", "competition", "home", "away"]
    joined = left.merge(snapshots, on=keys, how="inner")
    joined = joined[joined["snapshot_time"] < joined["kickoff_dt"]]
    if joined.empty:
        return joined
    idx = joined.groupby("fixture_identity")["snapshot_time"].idxmax()
    joined = joined.loc[idx].copy().sort_values("match_date")
    probability_rows = []
    for row in joined.itertuples(index=False):
        raw = {side: float(getattr(row, f"odds_{side}")) for side in sides}
        proportional = devig(raw)
        power, exponent = power_devig(raw)
        probability_rows.append({
            **{f"p_market_{side}": power.get(side) for side in sides},
            **{f"p_market_prop_{side}": proportional.get(side) for side in sides},
            "power_k": exponent,
            "market_overround": sum(1.0 / value for value in raw.values()) - 1.0,
        })
    return pd.concat([joined.reset_index(drop=True), pd.DataFrame(probability_rows)], axis=1)


def _paired_summary(frame: pd.DataFrame, market: str,
                    market_prefix: str = "p_market") -> dict:
    if frame.empty:
        return {"n": 0, "model_log_loss": None, "market_log_loss": None,
                "paired_delta_model_minus_market": None, "paired_delta_se": None,
                "paired_delta_t": None, "model_brier": None, "market_brier": None}
    if market == "1x2":
        actual = frame["actual_1x2"].astype(str).to_numpy()
        model = frame[[f"p_{s}" for s in SIDES_1X2]].to_numpy(float)
        market_p = frame[[f"{market_prefix}_{s}" for s in SIDES_1X2]].to_numpy(float)
        lookup = {side: i for i, side in enumerate(SIDES_1X2)}
        y_idx = np.asarray([lookup[value] for value in actual])
        y = np.eye(3)[y_idx]
    else:
        y_over = pd.to_numeric(frame["over25_actual"], errors="coerce").to_numpy(int)
        model = frame[["p_over25", "p_under25"]].to_numpy(float)
        market_p = frame[[f"{market_prefix}_over", f"{market_prefix}_under"]].to_numpy(float)
        y_idx = np.where(y_over == 1, 0, 1)
        y = np.eye(2)[y_idx]
    rows = np.arange(len(frame))
    model_ll = -np.log(np.clip(model[rows, y_idx], 1e-12, 1.0))
    market_ll = -np.log(np.clip(market_p[rows, y_idx], 1e-12, 1.0))
    delta = model_ll - market_ll
    se = float(delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else None
    return {
        "n": int(len(frame)),
        "model_log_loss": round(float(model_ll.mean()), 6),
        "market_log_loss": round(float(market_ll.mean()), 6),
        "paired_delta_model_minus_market": round(float(delta.mean()), 6),
        "paired_delta_se": round(se, 6) if se is not None else None,
        "paired_delta_t": round(float(delta.mean() / se), 3) if se else None,
        "model_brier": round(float(np.mean(np.sum((model - y) ** 2, axis=1))), 6),
        "market_brier": round(float(np.mean(np.sum((market_p - y) ** 2, axis=1))), 6),
    }


def _blend_curve(frame: pd.DataFrame, market: str) -> list[dict]:
    if frame.empty:
        return []
    if market == "1x2":
        model = frame[[f"p_{s}" for s in SIDES_1X2]].to_numpy(float)
        market_p = frame[[f"p_market_{s}" for s in SIDES_1X2]].to_numpy(float)
        lookup = {side: i for i, side in enumerate(SIDES_1X2)}
        y_idx = np.asarray([lookup[v] for v in frame["actual_1x2"].astype(str)])
    else:
        model = frame[["p_over25", "p_under25"]].to_numpy(float)
        market_p = frame[["p_market_over", "p_market_under"]].to_numpy(float)
        y_idx = np.where(pd.to_numeric(frame["over25_actual"]).to_numpy() == 1, 0, 1)
    rows = np.arange(len(frame))
    out = []
    for weight in BLEND_GRID:
        blended = weight * model + (1.0 - weight) * market_p
        blended /= blended.sum(axis=1, keepdims=True)
        loss = -np.log(np.clip(blended[rows, y_idx], 1e-12, 1.0)).mean()
        out.append({"model_weight": round(float(weight), 2), "log_loss": round(float(loss), 6)})
    return out


def _calibration_by_side_and_price(frame: pd.DataFrame, market: str) -> list[dict]:
    if frame.empty:
        return []
    sides = SIDES_1X2 if market == "1x2" else SIDES_TOTAL
    price_bins = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, math.inf)]
    rows = []
    for side in sides:
        actual = ((frame["actual_1x2"].astype(str) == side).astype(int)
                  if market == "1x2" else
                  ((pd.to_numeric(frame["over25_actual"]) == 1).astype(int)
                   if side == "over" else
                   (pd.to_numeric(frame["over25_actual"]) == 0).astype(int)))
        for lo, hi in price_bins:
            price = pd.to_numeric(frame[f"odds_{side}"], errors="coerce")
            mask = (price >= lo) & (price < hi)
            if not mask.any():
                continue
            rows.append({
                "side": side,
                "odds_band": f"{lo:g}-{hi:g}" if math.isfinite(hi) else f"{lo:g}+",
                "n": int(mask.sum()),
                "model_probability": round(float(frame.loc[mask, f"p_{side if market == '1x2' else side + '25'}"].mean()), 5),
                "market_probability": round(float(frame.loc[mask, f"p_market_{side}"].mean()), 5),
                "actual_rate": round(float(actual[mask].mean()), 5),
            })
    return rows


def _selection_diagnostics(edge_floor: float = 0.02) -> dict:
    """Calibration/ROI of bets the current strategy actually selects."""
    from . import decision_ledger as DL
    from .strategy_contract import STRATEGY_VERSION

    rows = DL.settled_bets()
    if not rows:
        return {"strategy_version": STRATEGY_VERSION, "edge_floor": edge_floor,
                "n": 0, "markets": {}}
    frame = pd.DataFrame(rows)
    eligible = frame["strategy_eligible"].astype(str).str.lower().isin(
        {"", "1", "1.0", "true", "yes"}
    )
    frame = frame[
        eligible
        & ~frame["identity_excluded"].fillna(False).astype(bool)
        & (frame["strategy_version"].astype(str) == STRATEGY_VERSION)
    ].copy()
    frame = _number(frame, [
        "edge", "p_model", "odds_executed", "won", "legacy_clv", "clv",
        "raw_price_clv",
    ])
    frame = frame[frame["edge"] >= edge_floor]

    def summarize(group: pd.DataFrame, label: str) -> dict:
        if group.empty:
            return {"band": label, "n": 0}
        profit = np.where(group["won"] == 1, group["odds_executed"] - 1.0, -1.0)
        return {
            "band": label,
            "n": int(len(group)),
            "predicted_rate": round(float(group["p_model"].mean()), 5),
            "actual_rate": round(float(group["won"].mean()), 5),
            "flat_roi": round(float(profit.mean()), 5),
            "mean_odds": round(float(group["odds_executed"].mean()), 4),
            "legacy_clv_mean": (round(float(group["legacy_clv"].dropna().mean()), 5)
                                if group["legacy_clv"].notna().any() else None),
            "fair_clv_v2_mean": (round(float(group["clv"].dropna().mean()), 5)
                                 if group["clv"].notna().any() else None),
            "raw_same_book_clv_mean": (
                round(float(group["raw_price_clv"].dropna().mean()), 5)
                if group["raw_price_clv"].notna().any() else None
            ),
        }

    markets = {}
    for market, group in frame.groupby("market"):
        prob_rows = []
        for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, 1.01)):
            selected = group[(group["p_model"] >= lo) & (group["p_model"] < hi)]
            if not selected.empty:
                prob_rows.append(summarize(selected, f"{lo:.1f}-{min(hi, 1):.1f}"))
        price_rows = []
        for lo, hi in ((1, 2), (2, 3), (3, 5), (5, 8), (8, math.inf)):
            selected = group[(group["odds_executed"] >= lo)
                             & (group["odds_executed"] < hi)]
            if not selected.empty:
                price_rows.append(summarize(
                    selected, f"{lo:g}-{hi:g}" if math.isfinite(hi) else f"{lo:g}+"
                ))
        legacy = group.dropna(subset=["legacy_clv", "odds_executed"])
        raw = group.dropna(subset=["raw_price_clv", "odds_executed"])
        markets[str(market)] = {
            "overall": summarize(group, "all"),
            "by_model_probability": prob_rows,
            "by_executed_odds": price_rows,
            "legacy_clv_odds_correlation": (
                round(float(legacy["legacy_clv"].corr(legacy["odds_executed"])), 5)
                if len(legacy) >= 3 else None
            ),
            "raw_clv_odds_correlation": (
                round(float(raw["raw_price_clv"].corr(raw["odds_executed"])), 5)
                if len(raw) >= 3 else None
            ),
        }
    return {"strategy_version": STRATEGY_VERSION, "edge_floor": edge_floor,
            "n": int(len(frame)), "markets": markets}


def _challenger_split(frame: pd.DataFrame, market: str) -> dict:
    """One chronological holdout; exploratory and incapable of promotion."""
    if frame.empty:
        return {"status": "insufficient_data", "promotion_authority": False}
    ordered = frame.sort_values(["match_date", "fixture_identity"])
    cut = max(MIN_CHALLENGER_TRAIN, int(len(ordered) * 0.7))
    train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    weeks = pd.to_datetime(ordered["match_date"]).dt.to_period("W").nunique()
    if len(train) < MIN_CHALLENGER_TRAIN or len(test) < MIN_CHALLENGER_TEST:
        return {"status": "insufficient_data", "n": int(len(ordered)),
                "train_n": int(len(train)), "test_n": int(len(test)),
                "independent_weeks": int(weeks), "promotion_authority": False}
    train_curve = _blend_curve(train, market)
    best = min(train_curve, key=lambda row: row["log_loss"])
    test_curve = {row["model_weight"]: row for row in _blend_curve(test, market)}
    selected = test_curve[best["model_weight"]]
    market_only, model_only = test_curve[0.0], test_curve[1.0]
    return {
        "status": ("exploratory_insufficient_temporal_span"
                   if weeks < MIN_PROMOTION_WEEKS else "exploratory_holdout"),
        "train_n": int(len(train)), "test_n": int(len(test)),
        "independent_weeks": int(weeks),
        "selected_model_weight": best["model_weight"],
        "test_log_loss": selected["log_loss"],
        "test_market_log_loss": market_only["log_loss"],
        "test_model_log_loss": model_only["log_loss"],
        "beats_both_on_holdout": bool(
            selected["log_loss"] < market_only["log_loss"]
            and selected["log_loss"] < model_only["log_loss"]
        ),
        "promotion_authority": False,
    }


def _market_report(forecasts: pd.DataFrame, market: str) -> dict:
    frame = _join_market(forecasts, market)
    curve = _blend_curve(frame, market)
    best = min(curve, key=lambda row: row["log_loss"]) if curve else None
    return {
        "comparison": _paired_summary(frame, market),
        "proportional_devig_comparison": _paired_summary(
            frame, market, market_prefix="p_market_prop"
        ),
        "blend_curve": curve,
        "best_in_sample_model_weight": best["model_weight"] if best else None,
        "calibration_by_side_and_price": _calibration_by_side_and_price(frame, market),
        "challenger": _challenger_split(frame, market),
    }


def run(*, write: bool = False) -> dict:
    cohorts = {}
    for name in ("first_published", "t24", "latest_pre_kickoff"):
        published = _published_cohort(name)
        cohorts[name] = {
            "n_settled_forecasts": int(len(published)),
            "1x2": _market_report(published, "1x2"),
            "ou25": _market_report(published, "total25"),
        }
    report = {
        "schema_version": "market_diagnostics_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_only_no_promotion_authority",
        "market_source": "latest_complete_pre_kickoff_median_snapshot",
        "devig_method": "power",
        "cohorts": cohorts,
        "betting_selection_diagnostics": _selection_diagnostics(),
        "provenance": {
            "forecast_ledger_sha256": _sha(FORECASTS),
            "forecast_settlements_sha256": _sha(RESULTS),
            "odds_history_sha256": _sha(ODDS),
            "decision_ledger_sha256": _sha(RUNTIME / "decision_ledger.csv"),
            "settlement_ledger_sha256": _sha(RUNTIME / "settlement_ledger.csv"),
            "code_sha256": _sha(Path(__file__)),
        },
        "limitations": [
            "Median snapshot odds are a diagnostic benchmark, not executable per-book CLV.",
            "Short live history makes calibration bands and chronological challengers exploratory.",
            "No result in this artifact can activate staking or promote a model.",
        ],
    }
    if write:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        tmp = REPORT.with_name(f"{REPORT.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
        tmp.replace(REPORT)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="atomically write data/market_diagnostics.json")
    args = parser.parse_args()
    report = run(write=args.write)
    for cohort, payload in report["cohorts"].items():
        print(f"{cohort} (settled={payload['n_settled_forecasts']}):")
        for market in ("1x2", "ou25"):
            comparison = payload[market]["comparison"]
            print(
                f"  {market}: n={comparison['n']} model={comparison['model_log_loss']} "
                f"market={comparison['market_log_loss']} "
                f"delta={comparison['paired_delta_model_minus_market']}"
            )


if __name__ == "__main__":
    main()
