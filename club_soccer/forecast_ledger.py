#!/usr/bin/env python3
"""Append-only forward forecasts and outcome scoring for the Club Soccer card.

This ledger answers a different question from ``decision_ledger``:

* forecast_ledger: are the probabilities printed on the card well calibrated?
* decision_ledger: did a quote-backed betting strategy beat executable prices?

Every card run freezes the complete supported forecast universe before the
renderer applies its narrower presentation policy. This preserves shadow data
for competitions that are not currently surfaced. Results are appended
separately once official, so forecasts are never rewritten with hindsight.
Repeated daily forecasts for one fixture are all retained; reports select one
independent row per fixture for first-published, T-24 and latest-pre-kickoff
views.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import model as M
from .identities import match_identity
from .schema import OFFICIAL_RESULT_STATUSES, normalize_status

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNTIME = Path(os.environ.get("CLUB_SOCCER_RUNTIME_DIR", str(DATA)))
FORECASTS = RUNTIME / "forecast_ledger.csv"
RESULTS = RUNTIME / "forecast_settlements.csv"
PERFORMANCE = RUNTIME / "forecast_performance.json"

FORECAST_FIELDS = [
    "forecast_id", "forecast_ts", "run_id", "run_mode", "primary_eligible",
    "fixture_identity", "fixture_id", "kickoff_utc", "match_date", "lead_hours",
    "season", "competition", "competition_id", "country", "type",
    "home_club_id", "away_club_id", "home", "away", "neutral",
    "model", "model_hash", "code_hash", "resolver_version",
    "training_fingerprint", "p_home", "p_draw", "p_away",
    "p_over25", "p_under25", "p_btts_yes", "p_btts_no",
    "xg_home", "xg_away", "evidence_tier", "evidence_ok",
    "n_matches_home", "n_matches_away", "lineup_confidence",
    "home_attack_mult", "home_defense_mult", "away_attack_mult",
    "away_defense_mult", "n_missing_home", "n_missing_away",
]

RESULT_FIELDS = [
    "fixture_identity", "settled_ts", "result_fixture_id", "match_date",
    "competition", "home", "away", "status", "result_scope",
    "home_goals", "away_goals", "reg_home_goals", "reg_away_goals",
    "actual_1x2", "over25_actual", "btts_actual", "shootout_winner",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""


def _code_hash() -> str:
    digest = hashlib.sha256()
    for name in (
        "forecast_ledger.py", "season.py", "model.py", "availability.py",
        "player_features.py", "calibrate.py", "coverage.py",
    ):
        path = HERE / name
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()[:16]


def _resolver_version() -> str:
    # Reuse the betting ledger's definition so identity cohorts are comparable.
    from .decision_ledger import resolver_version

    return resolver_version()


def _present(value) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return not (isinstance(value, str) and not value.strip())


def _text(value) -> str:
    return str(value) if _present(value) else ""


def _number(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _lead_hours(kickoff_utc, forecast_ts: datetime) -> float | None:
    if not _present(kickoff_utc):
        return None
    kickoff = pd.to_datetime(kickoff_utc, utc=True, errors="coerce")
    if pd.isna(kickoff):
        return None
    return float((kickoff.to_pydatetime() - forecast_ts).total_seconds() / 3600.0)


def _validate_probs(probs: dict) -> None:
    keys = ("home", "draw", "away", "over25", "under25", "btts_yes", "btts_no")
    for key in keys:
        value = probs.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"forecast probability {key} is not finite: {value!r}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"forecast probability {key} outside [0,1]: {value!r}")
    if abs(sum(float(probs[k]) for k in ("home", "draw", "away")) - 1.0) > 1e-5:
        raise ValueError("1X2 forecast probabilities do not sum to 1")
    if abs(float(probs["over25"]) + float(probs["under25"]) - 1.0) > 1e-5:
        raise ValueError("OU2.5 forecast probabilities do not sum to 1")
    if abs(float(probs["btts_yes"]) + float(probs["btts_no"]) - 1.0) > 1e-5:
        raise ValueError("BTTS forecast probabilities do not sum to 1")


def build_forecasts(
    today_ts: pd.Timestamp,
    player_adj_map: dict | None,
    calib_maps: dict | None = None,
    *,
    run_id: str,
    run_mode: str,
    primary_eligible: bool,
    horizon_days: int = 7,
    forecast_ts: datetime | None = None,
    fixtures: pd.DataFrame | None = None,
) -> list[dict]:
    """Build the full shadow + surfaced forecast universe for this horizon."""
    stamp = forecast_ts or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        raise ValueError("forecast_ts must be timezone-aware")
    stamp = stamp.astimezone(timezone.utc)
    fx = M.load_fixtures() if fixtures is None else fixtures.copy()
    up = M.upcoming(fx)
    start = pd.Timestamp(today_ts).normalize()
    if start.tz is not None:
        start = start.tz_localize(None)
    horizon = start + pd.Timedelta(days=horizon_days)
    dates = pd.to_datetime(up["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    up = up[(dates >= start) & (dates <= horizon)].copy()
    up["date"] = dates[(dates >= start) & (dates <= horizon)]
    up = up.sort_values(["date", "competition", "home"])

    params = M.load_params()
    teams = set(params["teams"])
    model_hash = _sha(DATA / "model_params.json")
    code_hash = _code_hash()
    resolver = _resolver_version()
    training_fp = str(params.get("_training_fingerprint") or "")
    rows: list[dict] = []

    for fixture in up.to_dict("records"):
        home, away = str(fixture.get("home") or ""), str(fixture.get("away") or "")
        competition = str(fixture.get("competition") or "")
        if home not in teams or away not in teams:
            continue
        lead = _lead_hours(fixture.get("kickoff_utc"), stamp)
        # A date-only fixture remains recordable, but a known kickoff already in
        # the past must never be republished or frozen as a pre-match forecast.
        if lead is not None and lead <= 0:
            continue
        player_adj = None
        if player_adj_map:
            player_adj = player_adj_map.get((home.lower(), away.lower(), competition))
        match_date = pd.Timestamp(fixture["date"]).strftime("%Y-%m-%d")
        pred = M.predict_match(
            home, away, competition, match_date, "ensemble",
            bool(_number(fixture.get("neutral"))), params=params,
            player_adj=player_adj, fixture_id=fixture.get("fixture_id"),
        )
        if calib_maps is not None:
            from .calibrate import apply as apply_calibration

            ph, pd_, pa = apply_calibration(
                pred["probs"]["home"], pred["probs"]["draw"],
                pred["probs"]["away"], calib_maps,
            )
            pred["probs"].update({"home": ph, "draw": pd_, "away": pa})
        probs = dict(pred["probs"])
        probs.setdefault("under25", 1.0 - float(probs["over25"]))
        probs.setdefault("btts_no", 1.0 - float(probs["btts_yes"]))
        # model.predict returns display-rounded probabilities (normally four
        # decimals), so H/D/A can total 0.9999 or 1.0001. Normalize the frozen
        # row into a proper distribution before it becomes scoring evidence.
        one_total = sum(float(probs[k]) for k in ("home", "draw", "away"))
        if one_total > 0:
            for key in ("home", "draw", "away"):
                probs[key] = float(probs[key]) / one_total
        for yes, no in (("over25", "under25"), ("btts_yes", "btts_no")):
            total = float(probs[yes]) + float(probs[no])
            if total > 0:
                probs[yes], probs[no] = float(probs[yes]) / total, float(probs[no]) / total
        _validate_probs(probs)

        coverage = pred.get("coverage") or {}
        home_cov, away_cov = coverage.get("home") or {}, coverage.get("away") or {}
        home_adj = (player_adj or {}).get("home") or {}
        away_adj = (player_adj or {}).get("away") or {}
        lineup_conf = _number((player_adj or {}).get("lineup_confidence"), 1.0)
        identity = match_identity(fixture)
        forecast_id = hashlib.sha256(
            f"{run_id}|{identity}|production_card".encode()
        ).hexdigest()[:24]
        row = {
            "forecast_id": forecast_id,
            "forecast_ts": stamp.isoformat(),
            "run_id": run_id,
            "run_mode": run_mode,
            "primary_eligible": int(bool(primary_eligible)),
            "fixture_identity": identity,
            "fixture_id": _text(fixture.get("fixture_id")),
            "kickoff_utc": _text(fixture.get("kickoff_utc")),
            "match_date": match_date,
            "lead_hours": round(lead, 3) if lead is not None else "",
            "season": _text(fixture.get("season")),
            "competition": competition,
            "competition_id": _text(fixture.get("competition_id")),
            "country": _text(fixture.get("country")),
            "type": _text(fixture.get("type")),
            "home_club_id": _text(fixture.get("home_club_id")),
            "away_club_id": _text(fixture.get("away_club_id")),
            "home": home,
            "away": away,
            "neutral": int(bool(_number(fixture.get("neutral")))),
            "model": "ensemble",
            "model_hash": model_hash,
            "code_hash": code_hash,
            "resolver_version": resolver,
            "training_fingerprint": training_fp,
            "p_home": round(float(probs["home"]), 8),
            "p_draw": round(float(probs["draw"]), 8),
            "p_away": round(float(probs["away"]), 8),
            "p_over25": round(float(probs["over25"]), 8),
            "p_under25": round(float(probs["under25"]), 8),
            "p_btts_yes": round(float(probs["btts_yes"]), 8),
            "p_btts_no": round(float(probs["btts_no"]), 8),
            "xg_home": round(_number(pred.get("xg_home")), 8),
            "xg_away": round(_number(pred.get("xg_away")), 8),
            "evidence_tier": str(coverage.get("tier") or "unknown"),
            "evidence_ok": int(bool(coverage.get("reliable", False))),
            "n_matches_home": int(_number(home_cov.get("n"))),
            "n_matches_away": int(_number(away_cov.get("n"))),
            "lineup_confidence": round(lineup_conf, 4),
            "home_attack_mult": round(_number(home_adj.get("attack_mult"), 1.0), 4),
            "home_defense_mult": round(_number(home_adj.get("defense_mult"), 1.0), 4),
            "away_attack_mult": round(_number(away_adj.get("attack_mult"), 1.0), 4),
            "away_defense_mult": round(_number(away_adj.get("defense_mult"), 1.0), 4),
            "n_missing_home": int(_number(home_adj.get("n_missing"))),
            "n_missing_away": int(_number(away_adj.get("n_missing"))),
        }
        rows.append(row)
    return rows


def _append_unique(path: Path, fields: list[str], rows: list[dict], key: str) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lock the target itself. Every writer follows this helper, so reading the
    # existing IDs and appending new rows is one critical section.
    with path.open("a+", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        existing = {row.get(key, "") for row in csv.DictReader(fh)}
        seen = {str(value) for value in existing}
        fresh = []
        for row in rows:
            value = str(row.get(key, ""))
            if not value or value in seen:
                continue
            fresh.append(row)
            seen.add(value)
        if not fresh:
            return 0
        fh.seek(0, os.SEEK_END)
        writer = csv.DictWriter(fh, fieldnames=fields)
        if fh.tell() == 0:
            writer.writeheader()
        for row in fresh:
            writer.writerow({field: row.get(field, "") for field in fields})
        fh.flush()
        os.fsync(fh.fileno())
        return len(fresh)


def append_forecasts(rows: list[dict]) -> int:
    return _append_unique(FORECASTS, FORECAST_FIELDS, rows, "forecast_id")


def _regulation_score(row: dict) -> tuple[float, float]:
    # *_goals_ft is the explicit regulation/full-time score where a provider
    # supplied extra-time detail. Legacy and normal league rows use home_goals.
    hft, aft = row.get("home_goals_ft"), row.get("away_goals_ft")
    if _present(hft) and _present(aft):
        return float(hft), float(aft)
    return float(row["home_goals"]), float(row["away_goals"])


def _match_finished_fixture(forecast: dict, finished: list[dict]) -> dict | None:
    exact = [row for row in finished if match_identity(row) == forecast["fixture_identity"]]
    if exact:
        return exact[0]
    # Provider IDs often change when the richest duplicate wins reconciliation.
    # Fall back to canonical names/competition with a one-day timezone tolerance.
    try:
        forecast_date = pd.Timestamp(forecast["match_date"]).date()
    except (TypeError, ValueError):
        return None
    candidates = []
    for row in finished:
        if (str(row.get("competition") or "") != str(forecast.get("competition") or "")
                or str(row.get("home") or "") != str(forecast.get("home") or "")
                or str(row.get("away") or "") != str(forecast.get("away") or "")):
            continue
        try:
            diff = abs((pd.Timestamp(row.get("date")).date() - forecast_date).days)
        except (TypeError, ValueError):
            continue
        if diff <= 1:
            candidates.append((diff, row))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if candidates else None


def settle(fixtures: pd.DataFrame | None = None, verbose: bool = True) -> int:
    """Append official outcomes for every forecast fixture that has finished."""
    if not FORECASTS.exists():
        return 0
    with FORECASTS.open(encoding="utf-8") as fh:
        forecasts = list(csv.DictReader(fh))
    if not forecasts:
        return 0
    settled_ids: set[str] = set()
    if RESULTS.exists():
        with RESULTS.open(encoding="utf-8") as fh:
            settled_ids = {row["fixture_identity"] for row in csv.DictReader(fh)}
    fx = M.load_fixtures() if fixtures is None else fixtures.copy()
    if not {"home_goals", "away_goals"}.issubset(fx.columns):
        return 0
    fx = fx.dropna(subset=["home_goals", "away_goals"])
    if "status" in fx.columns:
        fx = fx[fx["status"].map(normalize_status).isin(OFFICIAL_RESULT_STATUSES)]
    finished = fx.to_dict("records")
    now = datetime.now(timezone.utc).isoformat()
    out = []
    # One result row per frozen fixture identity, regardless of how many daily
    # forecast snapshots exist for that fixture.
    first_by_identity = {}
    for forecast in forecasts:
        first_by_identity.setdefault(forecast["fixture_identity"], forecast)
    for identity, forecast in first_by_identity.items():
        if identity in settled_ids:
            continue
        result = _match_finished_fixture(forecast, finished)
        if result is None:
            continue
        hg, ag = float(result["home_goals"]), float(result["away_goals"])
        rhg, rag = _regulation_score(result)
        out.append({
            "fixture_identity": identity,
            "settled_ts": now,
            "result_fixture_id": _text(result.get("fixture_id")),
            "match_date": pd.Timestamp(result.get("date")).strftime("%Y-%m-%d"),
            "competition": _text(result.get("competition")),
            "home": _text(result.get("home")),
            "away": _text(result.get("away")),
            "status": normalize_status(result.get("status")),
            "result_scope": _text(result.get("result_scope")),
            "home_goals": hg,
            "away_goals": ag,
            "reg_home_goals": rhg,
            "reg_away_goals": rag,
            "actual_1x2": "home" if rhg > rag else "away" if rag > rhg else "draw",
            "over25_actual": int(rhg + rag > 2.5),
            "btts_actual": int(rhg > 0 and rag > 0),
            "shootout_winner": _text(result.get("shootout_winner")),
        })
    added = _append_unique(RESULTS, RESULT_FIELDS, out, "fixture_identity")
    if verbose:
        print(f"  forecast_ledger: settled {added} fixture(s)")
    return added


def _metric_rows(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    probs = frame[["p_home", "p_draw", "p_away"]].astype(float)
    sides = ["home", "draw", "away"]
    actual_idx = frame["actual_1x2"].map({side: i for i, side in enumerate(sides)}).astype(int)
    y = pd.get_dummies(actual_idx).reindex(columns=range(3), fill_value=0).to_numpy(float)
    p = probs.to_numpy(float)
    chosen = p[range(len(frame)), actual_idx.to_numpy()]
    over_y = frame["over25_actual"].astype(float).to_numpy()
    btts_y = frame["btts_actual"].astype(float).to_numpy()
    over_p = frame["p_over25"].astype(float).to_numpy()
    btts_p = frame["p_btts_yes"].astype(float).to_numpy()
    eps = 1e-12
    top = p.max(axis=1)
    hit = p.argmax(axis=1) == actual_idx.to_numpy()
    return {
        "n": int(len(frame)),
        "accuracy_1x2": round(float(hit.mean()), 6),
        "brier_1x2": round(float(((p - y) ** 2).sum(axis=1).mean()), 6),
        "log_loss_1x2": round(float((-pd.Series(chosen).clip(eps, 1).map(math.log)).mean()), 6),
        "brier_ou25": round(float(((over_p - over_y) ** 2).mean()), 6),
        "log_loss_ou25": round(float((-(over_y * pd.Series(over_p).clip(eps, 1-eps).map(math.log)
            + (1-over_y) * pd.Series(1-over_p).clip(eps, 1-eps).map(math.log))).mean()), 6),
        "brier_btts": round(float(((btts_p - btts_y) ** 2).mean()), 6),
        "log_loss_btts": round(float((-(btts_y * pd.Series(btts_p).clip(eps, 1-eps).map(math.log)
            + (1-btts_y) * pd.Series(1-btts_p).clip(eps, 1-eps).map(math.log))).mean()), 6),
        "mean_top_probability": round(float(top.mean()), 6),
        "top_choice_hit_rate": round(float(hit.mean()), 6),
    }


def _select_cohort(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.sort_values("forecast_ts")
    if kind == "first_published":
        return ordered.groupby("fixture_identity", as_index=False).first()
    if kind == "latest_pre_kickoff":
        return ordered.groupby("fixture_identity", as_index=False).last()
    if kind == "t24":
        eligible = ordered[pd.to_numeric(ordered["lead_hours"], errors="coerce") >= 24].copy()
        if eligible.empty:
            return eligible
        eligible["lead_hours"] = pd.to_numeric(eligible["lead_hours"], errors="coerce")
        idx = eligible.groupby("fixture_identity")["lead_hours"].idxmin()
        return eligible.loc[idx].sort_values("forecast_ts")
    raise ValueError(f"unknown forecast cohort {kind!r}")


def _calibration(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    p = frame[["p_home", "p_draw", "p_away"]].astype(float).to_numpy()
    actual = frame["actual_1x2"].map({"home": 0, "draw": 1, "away": 2}).to_numpy()
    top, pick = p.max(axis=1), p.argmax(axis=1)
    hit = pick == actual
    bins = [(0.0, .40), (.40, .45), (.45, .50), (.50, .55), (.55, .60),
            (.60, .65), (.65, .70), (.70, .80), (.80, 1.000001)]
    out = []
    for lo, hi in bins:
        mask = (top >= lo) & (top < hi)
        if not mask.any():
            continue
        out.append({
            "bucket": f"{lo:.0%}-{min(hi, 1):.0%}",
            "n": int(mask.sum()),
            "mean_probability": round(float(top[mask].mean()), 6),
            "observed_hit_rate": round(float(hit[mask].mean()), 6),
        })
    return out


def performance_report(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    base = {
        "generated_at_utc": now.isoformat(),
        "forecast_rows": 0,
        "forecast_fixtures": 0,
        "settled_fixtures": 0,
        "cohorts": {},
        "rolling_t24": {},
        "by_competition_t24": {},
        "by_evidence_t24": {},
        "by_fixture_type_t24": {},
        "by_lead_bucket": {},
        "calibration_t24": [],
    }
    if not FORECASTS.exists():
        return base
    forecasts = pd.read_csv(FORECASTS, low_memory=False)
    base.update({
        "forecast_rows": int(len(forecasts)),
        "forecast_fixtures": int(forecasts["fixture_identity"].nunique()),
    })
    if not RESULTS.exists():
        return base
    results = pd.read_csv(RESULTS, low_memory=False)
    base["settled_fixtures"] = int(results["fixture_identity"].nunique())
    joined = forecasts.merge(results, on="fixture_identity", how="inner", suffixes=("", "_result"))
    if joined.empty:
        return base
    joined["forecast_ts"] = pd.to_datetime(joined["forecast_ts"], utc=True, errors="coerce")
    joined["result_date"] = pd.to_datetime(joined["match_date_result"], errors="coerce")
    production = joined[joined["primary_eligible"].astype(str).str.lower().isin({"1", "true", "yes"})]
    if production.empty:
        production = joined.iloc[0:0]
    cohorts = {kind: _select_cohort(production, kind)
               for kind in ("first_published", "t24", "latest_pre_kickoff")}
    base["cohorts"] = {kind: _metric_rows(frame) for kind, frame in cohorts.items()}
    t24 = cohorts["t24"]
    for label, days in (("last_7_days", 7), ("last_30_days", 30), ("last_365_days", 365),
                        ("lifetime", None)):
        subset = t24 if days is None else t24[t24["result_date"] >= pd.Timestamp(now.date()) - pd.Timedelta(days=days)]
        base["rolling_t24"][label] = _metric_rows(subset)
    if not t24.empty:
        for competition, group in t24.groupby("competition"):
            base["by_competition_t24"][str(competition)] = _metric_rows(group)
        for tier, group in t24.groupby("evidence_tier"):
            base["by_evidence_t24"][str(tier)] = _metric_rows(group)
        for fixture_type, group in t24.groupby("type"):
            base["by_fixture_type_t24"][str(fixture_type)] = _metric_rows(group)
        base["calibration_t24"] = _calibration(t24)
        latest_code = production.sort_values("forecast_ts")["code_hash"].iloc[-1]
        base["current_code_hash"] = str(latest_code)
        base["current_code_t24"] = _metric_rows(t24[t24["code_hash"] == latest_code])
    # Horizon diagnostics retain every lead-time regime, but still select at
    # most one row per fixture inside a bucket so repeated manual/daily runs do
    # not manufacture sample size.
    lead = production.copy()
    lead["lead_hours_num"] = pd.to_numeric(lead["lead_hours"], errors="coerce")
    for label, lo, hi in (("0_12h", 0, 12), ("12_36h", 12, 36),
                          ("36_72h", 36, 72), ("72_168h", 72, 168)):
        bucket = lead[(lead["lead_hours_num"] >= lo) & (lead["lead_hours_num"] < hi)]
        if not bucket.empty:
            idx = bucket.groupby("fixture_identity")["lead_hours_num"].idxmin()
            bucket = bucket.loc[idx]
        base["by_lead_bucket"][label] = _metric_rows(bucket)
    return base


def write_performance_report(now: datetime | None = None) -> dict:
    report = performance_report(now=now)
    PERFORMANCE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PERFORMANCE.with_name(f"{PERFORMANCE.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
    tmp.replace(PERFORMANCE)
    return report


def status() -> dict:
    forecasts = pd.read_csv(FORECASTS, low_memory=False) if FORECASTS.exists() else pd.DataFrame()
    results = pd.read_csv(RESULTS, low_memory=False) if RESULTS.exists() else pd.DataFrame()
    return {
        "forecast_rows": int(len(forecasts)),
        "forecast_fixtures": int(forecasts["fixture_identity"].nunique()) if not forecasts.empty else 0,
        "settled_fixtures": int(results["fixture_identity"].nunique()) if not results.empty else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settle", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.settle:
        settle()
    if args.report:
        report = write_performance_report()
        print(json.dumps(report, indent=2))
    if args.status or not (args.settle or args.report):
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main()
