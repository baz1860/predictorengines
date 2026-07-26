"""Prospective golf odds, decision, CLV, and ROI evidence.

This module deliberately does not backfill betting decisions from outcomes.
Odds are captured when providers refresh, model decisions are captured when the
edge screen runs, and both are later joined to the authoritative rounds history
by stable event ID. Until the prospective sample is large enough, the report
stays in ``collecting`` state and cannot enable blending or staking.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import statistics
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from . import market, model
from .io_utils import atomic_write_csv, atomic_write_text

DATA_DIR = Path(__file__).parent / "data"
ODDS_HISTORY = DATA_DIR / "odds_history.csv"
DECISION_HISTORY = DATA_DIR / "decision_history.csv"
SETTLED_DECISIONS = DATA_DIR / "settled_decisions.csv"
ECONOMIC_REPORT = DATA_DIR / "economic_report.json"

ODDS_COLS = [
    "snapshot_id", "observed_at", "provider_updated_at",
    "event_id", "event_name", "event_start_date", "phase",
    "book", "source", "market", "round_no", "group_id",
    "selection_id", "selection_name", "decimal_odds", "implied_prob",
    "fair_prob", "fair_method", "board_complete", "settlement_rule",
]
DECISION_COLS = [
    "decision_id", "observed_at", "event_id", "event_name", "phase",
    "player", "market", "side", "odds", "p_model", "p_market", "p_final",
    "ev_per_unit", "push_prob", "thin_sample", "paper_bet", "recommended",
    "stake_gbp", "model_asof", "model_fingerprint",
    "calibration_fingerprint", "sim_fingerprint",
]
SETTLEMENT_COLS = DECISION_COLS + [
    "settled", "result_credit", "unit_return", "unit_profit",
    "closing_fair_prob", "clv", "closing_quality",
]

MIN_SETTLED_BETS = 300
MIN_SETTLED_EVENTS = 30
MIN_CLOSE_COVERAGE = 0.75
PAPER_MIN_EV = 0.02


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _normalized_time(value: str, fallback: str = "") -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc).isoformat(timespec="seconds")
    except ValueError:
        return fallback


def _fold(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.casefold().split())


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _truthy(value) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _hash_payload(*parts) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _file_fingerprint(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _archive_incompatible_schema(path: Path, columns: list[str]) -> Path | None:
    """Preserve an older ledger instead of coercing it into a new schema."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
    except (OSError, csv.Error):
        header = []
    if header == columns:
        return None
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = path.with_name(
        f"{path.stem}.legacy-{stamp}-{_hash_payload(*header)[:8]}{path.suffix}"
    )
    path.replace(archive)
    return archive


def _append_unique(
    path: Path,
    columns: list[str],
    new_rows: Iterable[Mapping],
    key: str,
) -> int:
    """Atomically append rows while making a retried capture idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _archive_incompatible_schema(path, columns)
        existing = _read_rows(path)
        seen = {str(row.get(key) or "") for row in existing}
        accepted = []
        for raw in new_rows:
            row = {column: raw.get(column, "") for column in columns}
            value = str(row.get(key) or "")
            if not value or value in seen:
                continue
            seen.add(value)
            accepted.append(row)
        if accepted:
            atomic_write_csv(
                path, columns, [*existing, *accepted], extrasaction="ignore"
            )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return len(accepted)


def _quote_dict(quote) -> dict:
    return quote.as_dict() if hasattr(quote, "as_dict") else dict(quote)


def _fair_board(rows: list[dict]) -> tuple[dict[str, float], str, bool]:
    market_name = str(rows[0].get("market") or "")
    odds = [float(row["decimal_odds"]) for row in rows]
    names = [str(row["selection_name"]) for row in rows]
    implied_sum = sum(1.0 / value for value in odds)
    if market_name == "win":
        complete = len(rows) >= 20 and implied_sum >= 1.10
        fair = market.devig_outright(dict(zip(names, odds)), complete=complete)
        method = "power_complete" if complete else "raw_implied_partial"
        return fair, method, complete
    expected = {
        "tournament_matchup": 2,
        "round_matchup": 2,
        "2ball": 2,
        "3ball": 3,
    }.get(market_name)
    if expected and len(rows) == expected:
        probs = market.devig(odds, method="multiplicative")
        return dict(zip(names, probs)), "multiplicative_complete", True
    return (
        {name: 1.0 / value for name, value in zip(names, odds)},
        "raw_implied",
        False,
    )


def record_odds_snapshot(
    quotes: Iterable,
    *,
    event_id: str,
    event_name: str,
    event_start_date: str = "",
    phase: str = "pre_event",
    observed_at: str | None = None,
    path: Path | None = None,
) -> int:
    """Persist one provider-refresh observation with stable event provenance."""
    if not str(event_id).strip():
        return 0
    observed_at = observed_at or _utc_now()
    normalized = []
    seen = set()
    for quote in quotes:
        raw = _quote_dict(quote)
        name = str(raw.get("player_name") or raw.get("name") or "").strip()
        market_name = str(raw.get("market") or "").strip()
        odds = _safe_float(raw.get("decimal_odds") or raw.get("odds"))
        if not name or not market_name or odds is None or odds <= 1.0:
            continue
        row = {
            "provider_updated_at": _normalized_time(
                str(raw.get("timestamp") or ""), observed_at
            ),
            "event_id": str(event_id),
            "event_name": str(event_name),
            "event_start_date": str(event_start_date or "")[:10],
            "phase": str(phase),
            "book": str(raw.get("book") or "unknown"),
            "source": str(raw.get("source") or "unknown"),
            "market": market_name,
            "round_no": _safe_int(raw.get("round_no")) or "",
            "group_id": str(raw.get("group_id") or ""),
            "selection_id": str(
                raw.get("player_id") or raw.get("source_player_id")
                or "name:" + _fold(name)
            ),
            "selection_name": name,
            "decimal_odds": float(odds),
            "settlement_rule": str(raw.get("settlement_rule") or ""),
        }
        dedupe = (
            row["book"], row["source"], row["market"], row["round_no"],
            row["group_id"], row["selection_id"], row["decimal_odds"],
        )
        if dedupe not in seen:
            seen.add(dedupe)
            normalized.append(row)
    if not normalized:
        return 0

    grouped: dict[tuple, list[dict]] = {}
    for row in normalized:
        group_key = (
            row["book"], row["source"], row["market"], row["round_no"],
            row["group_id"],
        )
        grouped.setdefault(group_key, []).append(row)
    output = []
    for group_key, rows in grouped.items():
        fair, method, complete = _fair_board(rows)
        snapshot_id = _hash_payload(
            event_id, phase, *group_key,
            *sorted(
                (
                    row["selection_id"],
                    f"{row['decimal_odds']:.6f}",
                    row["provider_updated_at"],
                )
                for row in rows
            ),
        )
        for row in rows:
            fp = float(fair.get(row["selection_name"], 1.0 / row["decimal_odds"]))
            output.append({
                "snapshot_id": _hash_payload(snapshot_id, row["selection_id"]),
                "observed_at": observed_at,
                **row,
                "implied_prob": round(1.0 / row["decimal_odds"], 8),
                "fair_prob": round(fp, 8),
                "fair_method": method,
                "board_complete": int(complete),
            })
    return _append_unique(path or ODDS_HISTORY, ODDS_COLS, output, "snapshot_id")


def record_decisions(
    rows: Iterable[Mapping],
    *,
    event_id: str,
    event_name: str,
    phase: str,
    observed_at: str | None = None,
    path: Path | None = None,
) -> int:
    """Append the exact model decisions visible at pricing time."""
    if not str(event_id).strip():
        return 0
    observed_at = observed_at or _utc_now()
    params = model.load_params() or {}
    model_asof = str(params.get("asof") or "")
    model_fp = _file_fingerprint(model.PARAMS_JSON)
    calibration_fp = _file_fingerprint(DATA_DIR / "calibration.json")
    sim_fp = _file_fingerprint(DATA_DIR / "sim_config.json")
    output = []
    for raw in rows:
        player = str(raw.get("player") or "").strip()
        side = str(raw.get("side") or "").strip()
        odds = _safe_float(raw.get("odds"))
        if not player or not side or odds is None or odds <= 1.0:
            continue
        thin_raw = raw.get("thin_sample")
        thin = (
            thin_raw if isinstance(thin_raw, bool)
            else _truthy(thin_raw)
        )
        ev = float(_safe_float(raw.get("ev_per_unit"), 0.0) or 0.0)
        decision_id = _hash_payload(
            observed_at, event_id, phase, player, side, f"{odds:.6f}", model_fp
        )
        output.append({
            "decision_id": decision_id,
            "observed_at": observed_at,
            "event_id": event_id,
            "event_name": event_name,
            "phase": phase,
            "player": player,
            "market": str(raw.get("market") or ""),
            "side": side,
            "odds": odds,
            "p_model": _safe_float(raw.get("p_model"), 0.0),
            "p_market": _safe_float(raw.get("p_market"), ""),
            "p_final": _safe_float(raw.get("p_final"), 0.0),
            "ev_per_unit": ev,
            "push_prob": _safe_float(raw.get("push_prob"), 0.0),
            "thin_sample": int(thin),
            "paper_bet": int(ev >= PAPER_MIN_EV and not thin),
            "recommended": int(
                raw.get("recommended")
                if isinstance(raw.get("recommended"), bool)
                else _truthy(raw.get("recommended"))
            ),
            "stake_gbp": _safe_float(raw.get("stake_gbp"), 0.0),
            "model_asof": model_asof,
            "model_fingerprint": model_fp,
            "calibration_fingerprint": calibration_fp,
            "sim_fingerprint": sim_fp,
        })
    return _append_unique(
        path or DECISION_HISTORY, DECISION_COLS, output, "decision_id"
    )


def _event_outcomes(rounds: pd.DataFrame) -> dict[str, dict]:
    events: dict[str, dict] = {}
    for event_id, event in rounds.groupby("tournament_id", sort=False):
        total_rounds = int(
            pd.to_numeric(event.get("total_rounds"), errors="coerce")
            .dropna().max()
            if "total_rounds" in event else 4
        )
        if int(event["round"].max()) < total_rounds:
            continue
        grouped = event.groupby("player")
        totals = grouped["score_to_par"].sum()
        counts = grouped["round"].count()
        made_cut = grouped["made_cut"].max()
        official_finish = grouped["finish"].min()
        finishers = totals[(counts == total_rounds) & (made_cut == 1)]
        rank = finishers.rank(method="min")
        score_groups = finishers.groupby(finishers).groups

        def top_credit(player: str, limit: int) -> float:
            if player not in rank.index:
                return 0.0
            tied = score_groups[finishers.loc[player]]
            start = int(rank.loc[player])
            return max(0, min(len(tied), limit - start + 1)) / len(tied)

        players = {}
        for player in totals.index:
            player_rounds = event[event["player"] == player].sort_values("round")
            completed = int(counts.loc[player]) == total_rounds and int(made_cut.loc[player]) == 1
            r36 = float(player_rounds[player_rounds["round"] <= 2]["score_to_par"].sum())
            settlement_score = (
                (0, float(totals.loc[player]))
                if completed else (1, r36)
            )
            players[_fold(player)] = {
                "name": player,
                "win": float(official_finish.loc[player] == 1),
                "top5": top_credit(player, 5),
                "top10": top_credit(player, 10),
                "top20": top_credit(player, 20),
                "cut": float(made_cut.loc[player] == 1),
                "settlement_score": settlement_score,
            }
        events[str(event_id)] = {"players": players, "complete": True}
    return events


def _decision_market(side: str) -> str:
    prefix = str(side or "").split(":", 1)[0]
    return {
        "matchup": "tournament_matchup",
        "3ball": "3ball",
        "cut": "make_cut",
    }.get(prefix, prefix)


def _settle_one(decision: Mapping, outcomes: dict) -> tuple[float, float] | None:
    players = outcomes.get("players") or {}
    player = players.get(_fold(decision.get("player", "")))
    if player is None:
        return None
    side = str(decision.get("side") or "")
    odds = float(decision["odds"])
    prefix, _, detail = side.partition(":")
    if prefix in {"win", "top5", "top10", "top20", "cut"}:
        credit = float(player[prefix])
        return credit, credit * odds
    names = [name for name in detail.split("|") if name]
    if prefix == "matchup" and len(names) >= 2:
        selected = players.get(_fold(names[0]))
        opponent = players.get(_fold(names[1]))
        if selected is None or opponent is None:
            return None
        a, b = selected["settlement_score"], opponent["settlement_score"]
        if a == b:
            return 0.0, 1.0
        credit = float(a < b)
        return credit, credit * odds
    if prefix == "3ball" and len(names) >= 3:
        group = [players.get(_fold(name)) for name in names[:3]]
        if any(row is None for row in group):
            return None
        scores = [row["settlement_score"] for row in group]
        best = min(scores)
        tied = [index for index, score in enumerate(scores) if score == best]
        credit = 1.0 / len(tied) if 0 in tied else 0.0
        return credit, credit * odds
    return None


def _matching_group(
    odds_rows: list[dict],
    event_id: str,
    market_name: str,
    names: list[str],
) -> set[str]:
    wanted = {_fold(name) for name in names}
    groups: dict[str, set[str]] = {}
    for row in odds_rows:
        if row.get("event_id") != event_id or row.get("market") != market_name:
            continue
        group_id = str(row.get("group_id") or "")
        groups.setdefault(group_id, set()).add(_fold(row.get("selection_name", "")))
    return {group_id for group_id, members in groups.items() if wanted <= members}


def _closing_fair(
    decision: Mapping,
    odds_rows: list[dict],
) -> tuple[float | None, str]:
    if decision.get("phase") != "pre_event":
        return None, "in_play_not_comparable"
    event_id = str(decision.get("event_id") or "")
    side = str(decision.get("side") or "")
    prefix, _, detail = side.partition(":")
    market_name = _decision_market(side)
    selected_name = str(decision.get("player") or "")
    decision_time = _normalized_time(str(decision.get("observed_at") or ""))
    allowed_groups: set[str] | None = None
    if prefix in {"matchup", "3ball"}:
        names = [name for name in detail.split("|") if name]
        allowed_groups = _matching_group(odds_rows, event_id, market_name, names)
        if not allowed_groups:
            return None, "group_not_found"
    candidates = []
    for row in odds_rows:
        if (
            row.get("event_id") != event_id
            or row.get("phase") != "pre_event"
            or row.get("market") != market_name
            or _fold(row.get("selection_name", "")) != _fold(selected_name)
        ):
            continue
        if allowed_groups is not None and row.get("group_id", "") not in allowed_groups:
            continue
        fp = _safe_float(row.get("fair_prob"))
        quote_time = _normalized_time(
            str(row.get("provider_updated_at") or ""),
            _normalized_time(str(row.get("observed_at") or "")),
        )
        # A stale snapshot from before the decision is not a closing line.
        if fp and fp > 0 and quote_time >= decision_time:
            row = {**row, "_quote_time": quote_time}
            candidates.append(row)
    if not candidates:
        return None, "missing_close"
    latest_by_source: dict[tuple[str, str, str], dict] = {}
    for row in candidates:
        key = (
            str(row.get("book") or ""),
            str(row.get("source") or ""),
            str(row.get("group_id") or ""),
        )
        if row.get("_quote_time", "") >= latest_by_source.get(key, {}).get("_quote_time", ""):
            latest_by_source[key] = row
    probs = [float(row["fair_prob"]) for row in latest_by_source.values()]
    quality = (
        "complete_consensus"
        if all(_truthy(row.get("board_complete")) for row in latest_by_source.values())
        else "raw_or_partial_consensus"
    )
    return float(statistics.median(probs)), quality


def settle_decisions(
    decisions: list[dict],
    odds_rows: list[dict],
    rounds: pd.DataFrame,
) -> list[dict]:
    outcomes = _event_outcomes(rounds)
    settled = []
    for decision in decisions:
        event = outcomes.get(str(decision.get("event_id") or ""))
        result = _settle_one(decision, event) if event else None
        row = {column: decision.get(column, "") for column in DECISION_COLS}
        if result is None:
            settled.append({
                **row,
                "settled": 0,
                "result_credit": "",
                "unit_return": "",
                "unit_profit": "",
                "closing_fair_prob": "",
                "clv": "",
                "closing_quality": "",
            })
            continue
        credit, unit_return = result
        close, quality = _closing_fair(decision, odds_rows)
        odds = float(decision["odds"])
        settled.append({
            **row,
            "settled": 1,
            "result_credit": round(credit, 6),
            "unit_return": round(unit_return, 6),
            "unit_profit": round(unit_return - 1.0, 6),
            "closing_fair_prob": round(close, 8) if close else "",
            "clv": round(odds * close - 1.0, 6) if close else "",
            "closing_quality": quality,
        })
    return settled


def _bootstrap_roi(rows: list[dict], seed: int = 20260726) -> list[float] | None:
    by_event: dict[str, list[float]] = {}
    for row in rows:
        by_event.setdefault(str(row["event_id"]), []).append(float(row["unit_profit"]))
    if len(by_event) < 5:
        return None
    events = sorted(by_event)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(2000):
        draw = rng.choice(events, size=len(events), replace=True)
        profits = [profit for event_id in draw for profit in by_event[event_id]]
        samples.append(float(np.mean(profits)))
    return [
        round(float(np.quantile(samples, 0.025)), 5),
        round(float(np.quantile(samples, 0.975)), 5),
    ]


def economic_report(
    *,
    odds_path: Path | None = None,
    decisions_path: Path | None = None,
    rounds_path: Path | None = None,
    settlements_path: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    odds_rows = _read_rows(odds_path or ODDS_HISTORY)
    decisions = _read_rows(decisions_path or DECISION_HISTORY)
    rounds_path = rounds_path or model.ROUNDS_CSV
    rounds = model.load_rounds_df(rounds_path) if rounds_path.exists() else pd.DataFrame()
    settlements = (
        settle_decisions(decisions, odds_rows, rounds)
        if decisions and not rounds.empty else []
    )
    if settlements:
        atomic_write_csv(
            settlements_path or SETTLED_DECISIONS,
            SETTLEMENT_COLS,
            settlements,
            extrasaction="ignore",
        )
    paper_candidates = [
        row for row in settlements
        if _truthy(row.get("settled"))
        and _truthy(row.get("paper_bet"))
        and row.get("phase") == "pre_event"
    ]
    # One prospective entry per event/side. Re-running the same price screen is
    # another observation, not another independent bet.
    paper_by_bet: dict[tuple[str, str], dict] = {}
    for row in sorted(paper_candidates, key=lambda item: item.get("observed_at", "")):
        paper_by_bet.setdefault((row["event_id"], row["side"]), row)
    paper = list(paper_by_bet.values())
    with_close = [row for row in paper if row.get("clv") not in ("", None)]
    events = {row["event_id"] for row in paper}
    roi = float(np.mean([float(row["unit_profit"]) for row in paper])) if paper else None
    mean_clv = float(np.mean([float(row["clv"]) for row in with_close])) if with_close else None
    coverage = len(with_close) / len(paper) if paper else 0.0
    roi_ci = _bootstrap_roi(paper)
    enough_sample = len(paper) >= MIN_SETTLED_BETS and len(events) >= MIN_SETTLED_EVENTS
    evidence_pass = bool(
        enough_sample
        and coverage >= MIN_CLOSE_COVERAGE
        and mean_clv is not None and mean_clv > 0
        and roi_ci is not None and roi_ci[0] > 0
    )
    report = {
        "generated_at": _utc_now(),
        "status": "eligible_for_review" if evidence_pass else "collecting",
        "automatic_activation": False,
        "coverage": {
            "odds_rows": len(odds_rows),
            "odds_events": len({row.get("event_id", "") for row in odds_rows}),
            "decision_rows": len(decisions),
            "settled_decisions": sum(_truthy(row.get("settled")) for row in settlements),
            "paper_bets": len(paper),
            "paper_events": len(events),
            "closing_line_rows": len(with_close),
            "closing_line_coverage": round(coverage, 4),
        },
        "paper_ev_2pct": {
            "flat_stake_roi": round(roi, 5) if roi is not None else None,
            "event_block_bootstrap_95pct": roi_ci,
            "mean_clv": round(mean_clv, 5) if mean_clv is not None else None,
        },
        "readiness": {
            "minimum_settled_bets": MIN_SETTLED_BETS,
            "minimum_settled_events": MIN_SETTLED_EVENTS,
            "minimum_close_coverage": MIN_CLOSE_COVERAGE,
            "paper_minimum_ev": PAPER_MIN_EV,
            "enough_sample": enough_sample,
            "positive_clv": bool(mean_clv is not None and mean_clv > 0),
            "positive_roi_lower_bound": bool(roi_ci is not None and roi_ci[0] > 0),
            "evidence_pass": evidence_pass,
            "note": (
                "Passing this gate makes blending/staking eligible for human "
                "review; it never changes production settings automatically."
            ),
        },
    }
    atomic_write_text(
        report_path or ECONOMIC_REPORT,
        json.dumps(report, indent=2) + "\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prospective golf odds/decision CLV and ROI report"
    )
    parser.add_argument("--report", action="store_true", help="build settlement and evidence report")
    args = parser.parse_args()
    report = economic_report()
    coverage = report["coverage"]
    print(
        f"Golf economic evidence: {report['status']} · "
        f"{coverage['odds_rows']} odds · {coverage['decision_rows']} decisions · "
        f"{coverage['paper_bets']} settled paper bets across "
        f"{coverage['paper_events']} events"
    )
    if not report["readiness"]["evidence_pass"]:
        print("  blending and automatic staking remain disabled")


if __name__ == "__main__":
    main()
