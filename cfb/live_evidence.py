"""Append-only CFB live quote and paper-signal evidence.

Quote history retains exact provider/book/event/market outcomes. Paper signals
lock the first qualifying best edge per CFBD event and market. Reports compare
that entry with the latest observed same-book quote, but label it as closing
evidence only after kickoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import edge as CE
from . import engine as CENGINE

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
QUOTE_HISTORY = DATA / "live_quote_history.csv"
SIGNAL_HISTORY = DATA / "paper_signal_history.csv"
STATUS_JSON = DATA / "live_evidence_status.json"
REPORT_JSON = DATA / "live_evidence_report.json"

QUOTE_COLS = [
    "captured_at", "date", "home", "away", "event_id", "cfbd_game_id",
    "commence_time", "bookmaker", "quote_time", "market", "side", "line",
    "odds", "p_implied", "quote_eligible", "identity_version",
]
QUOTE_KEY = [
    "event_id", "bookmaker", "quote_time", "market", "side", "line", "odds",
]
SIGNAL_COLS = [
    "signal_id", "captured_at", "date", "home", "away", "event_id",
    "provider_event_id", "cfbd_game_id", "commence_time", "market", "side",
    "line", "bookmaker", "entry_quote_time", "entry_odds", "p_model",
    "p_book", "edge", "ev_per_unit", "policy_status", "model_snapshot",
    "prior_mode", "team_restricted", "stake", "runtime_eligible",
]
SIGNAL_KEY = ["event_id", "market"]


def _utc_now(value=None) -> pd.Timestamp:
    stamp = pd.Timestamp(value or datetime.now(timezone.utc))
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _atomic_frame(path: str | Path, frame: pd.DataFrame) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".csv", dir=dest.parent)
    try:
        os.close(fd)
        frame.to_csv(tmp, index=False)
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def _atomic_json(path: str | Path, payload: object) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def _read(path: str | Path, columns: list[str]) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(source)
    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
    return frame[columns]


def _key_values(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    return frame[keys].fillna("").astype(str).agg("\x1f".join, axis=1)


def append_unique(path: str | Path, rows: pd.DataFrame,
                  columns: list[str], keys: list[str]) -> tuple[int, int]:
    """Atomically append novel rows; return (new rows, total rows)."""
    current = _read(path, columns)
    incoming = rows.reindex(columns=columns).copy()
    if incoming.empty:
        return 0, len(current)
    incoming = incoming.loc[~_key_values(incoming, keys).duplicated()].copy()
    existing = set(_key_values(current, keys)) if not current.empty else set()
    novel = incoming.loc[~_key_values(incoming, keys).isin(existing)].copy()
    if novel.empty:
        return 0, len(current)
    output = pd.concat([current, novel], ignore_index=True)
    _atomic_frame(path, output)
    return len(novel), len(output)


def capture_quotes(now=None, odds_path: str | Path | None = None,
                   history_path: str | Path = QUOTE_HISTORY) -> dict:
    source = Path(odds_path or CE.ODDS_CSV)
    odds = pd.read_csv(source)
    dates = pd.to_datetime(odds["date"], errors="coerce").dropna()
    seasons = {int(value.year if value.month != 1 else value.year - 1)
               for value in dates}
    if len(seasons) != 1:
        raise ValueError(f"CFB quote snapshot spans ambiguous seasons: {sorted(seasons)}")
    prepared = CE.prepare_odds(odds, seasons, now=now)
    accepted = prepared[
        prepared["fixture_matched"] & prepared["pair_complete"]].copy()
    captured_at = _utc_now(now).isoformat()
    accepted["captured_at"] = captured_at
    accepted = accepted.rename(columns={"source": "provider"})
    new_rows, total_rows = append_unique(
        history_path, accepted, QUOTE_COLS, QUOTE_KEY)
    return {
        "captured_at": captured_at,
        "accepted_rows": int(len(accepted)),
        "new_quote_rows": int(new_rows),
        "total_quote_rows": int(total_rows),
        "rejected_rows": int(len(prepared) - len(accepted)),
    }


def _signal_id(event_id: str, market: str) -> str:
    return hashlib.sha256(f"{event_id}|{market}".encode()).hexdigest()[:20]


def _commence_lookup(odds_path: str | Path | None = None) -> dict[tuple, str]:
    odds = pd.read_csv(Path(odds_path or CE.ODDS_CSV))
    lookup = {}
    for row in odds.itertuples():
        key = (str(row.event_id), str(row.bookmaker), str(row.market), str(row.side),
               "" if pd.isna(row.line) else str(float(row.line)))
        lookup[key] = str(row.commence_time)
    return lookup


def capture_signals(now=None, odds_path: str | Path | None = None,
                    history_path: str | Path = SIGNAL_HISTORY,
                    result: dict | None = None) -> dict:
    result = result or CENGINE.cmd_edge({"bankroll": 100.0, "model": "blend"})
    state = result.get("model_state") or {}
    restricted = set(state.get("restricted_teams") or ())
    qualifying = [
        row for row in result.get("rows", [])
        if float(row.get("edge", 0.0)) >= CE.MIN_EDGE
        and row.get("market_status") in {"diagnostic", "paper"}
    ]
    best = {}
    for row in qualifying:
        key = (str(row.get("event_id")), str(row.get("market")))
        if key not in best or float(row["edge"]) > float(best[key]["edge"]):
            best[key] = row
    commence = _commence_lookup(odds_path)
    captured_at = _utc_now(now).isoformat()
    records = []
    for row in best.values():
        line = row.get("line", "")
        line_key = "" if line in (None, "") else str(float(line))
        lookup_key = (
            str(row.get("provider_event_id")), str(row.get("bookmaker")),
            str(row.get("market")), str(row.get("side")), line_key,
        )
        event_id = str(row.get("event_id"))
        records.append({
            "signal_id": _signal_id(event_id, str(row.get("market"))),
            "captured_at": captured_at,
            "date": row.get("date"), "home": row.get("home"),
            "away": row.get("away"), "event_id": event_id,
            "provider_event_id": row.get("provider_event_id"),
            "cfbd_game_id": row.get("cfbd_game_id"),
            "commence_time": commence.get(lookup_key, ""),
            "market": row.get("market"), "side": row.get("side"),
            "line": line, "bookmaker": row.get("bookmaker"),
            "entry_quote_time": row.get("quote_time"),
            "entry_odds": row.get("odds"), "p_model": row.get("p_model"),
            "p_book": row.get("p_book"), "edge": row.get("edge"),
            "ev_per_unit": row.get("ev_per_unit"),
            "policy_status": row.get("market_status"),
            "model_snapshot": state.get("snapshot_hash"),
            "prior_mode": state.get("prior_mode"),
            "team_restricted": bool(
                row.get("home") in restricted or row.get("away") in restricted),
            "stake": 0.0, "runtime_eligible": False,
        })
    frame = pd.DataFrame(records, columns=SIGNAL_COLS)
    new_rows, total_rows = append_unique(
        history_path, frame, SIGNAL_COLS, SIGNAL_KEY)
    return {
        "captured_at": captured_at,
        "qualifying_candidates": len(qualifying),
        "locked_candidates": len(best),
        "new_paper_signals": int(new_rows),
        "total_paper_signals": int(total_rows),
    }


def line_clv(market: str, side: str, entry_line, latest_line) -> float | None:
    if pd.isna(entry_line) or pd.isna(latest_line):
        return None
    entry, latest = float(entry_line), float(latest_line)
    if market == "spread":
        return entry - latest
    if market == "total" and side == "over":
        return latest - entry
    if market == "total" and side == "under":
        return entry - latest
    return None


def build_report(now=None, quote_path: str | Path = QUOTE_HISTORY,
                 signal_path: str | Path = SIGNAL_HISTORY) -> dict:
    quotes = _read(quote_path, QUOTE_COLS)
    signals = _read(signal_path, SIGNAL_COLS)
    current_time = _utc_now(now)
    rows = []
    for signal in signals.itertuples(index=False):
        matched = quotes[
            (quotes["event_id"].astype(str) == str(signal.provider_event_id))
            & (quotes["bookmaker"].astype(str) == str(signal.bookmaker))
            & (quotes["market"].astype(str) == str(signal.market))
            & (quotes["side"].astype(str) == str(signal.side))
        ].copy()
        kickoff = pd.to_datetime(signal.commence_time, errors="coerce", utc=True)
        matched["observed_at"] = pd.to_datetime(
            matched["quote_time"], errors="coerce", utc=True)
        if not pd.isna(kickoff):
            matched = matched[matched["observed_at"] <= kickoff]
        matched = matched.dropna(subset=["observed_at"]).sort_values("observed_at")
        latest = matched.iloc[-1] if not matched.empty else None
        latest_odds = None if latest is None else float(latest["odds"])
        entry_odds = float(signal.entry_odds)
        odds_clv = (entry_odds / latest_odds - 1.0
                    if signal.market == "ml" and latest_odds else None)
        latest_line = None if latest is None or pd.isna(latest["line"]) else float(latest["line"])
        rows.append({
            "signal_id": signal.signal_id,
            "event_id": signal.event_id,
            "market": signal.market, "side": signal.side,
            "bookmaker": signal.bookmaker,
            "entry_odds": entry_odds, "latest_odds": latest_odds,
            "entry_line": None if pd.isna(signal.line) else float(signal.line),
            "latest_line": latest_line,
            "odds_clv": odds_clv,
            "line_clv_points": line_clv(
                signal.market, signal.side, signal.line, latest_line),
            "latest_quote_time": None if latest is None else str(latest["quote_time"]),
            "kickoff": None if pd.isna(kickoff) else kickoff.isoformat(),
            "is_closing_evidence": bool(not pd.isna(kickoff) and current_time >= kickoff),
        })
    completed = [row for row in rows if row["is_closing_evidence"]]
    return {
        "schema_version": 1,
        "generated_at": current_time.isoformat(),
        "signals": len(rows),
        "signals_with_latest_quote": sum(row["latest_odds"] is not None for row in rows),
        "closing_evidence_rows": len(completed),
        "label": "closing evidence" if completed else "latest observed movement; not closing evidence",
        "rows": rows,
    }


def health(quote_path: str | Path = QUOTE_HISTORY,
           signal_path: str | Path = SIGNAL_HISTORY,
           status_path: str | Path = STATUS_JSON) -> dict:
    issues = []
    try:
        status = json.loads(Path(status_path).read_text())
    except (OSError, json.JSONDecodeError):
        status = {}
        issues.append("live evidence status is missing or unreadable")
    if status.get("status") != "success":
        issues.append(f"live evidence status is {status.get('status', 'missing')}")
    quotes = _read(quote_path, QUOTE_COLS)
    signals = _read(signal_path, SIGNAL_COLS)
    if quotes.empty:
        issues.append("live quote history is empty")
    elif _key_values(quotes, QUOTE_KEY).duplicated().any():
        issues.append("live quote history contains duplicate quote keys")
    if signals.empty:
        issues.append("paper signal history is empty")
    else:
        if _key_values(signals, SIGNAL_KEY).duplicated().any():
            issues.append("paper signal history contains duplicate event/market keys")
        stakes = pd.to_numeric(signals["stake"], errors="coerce").fillna(0.0)
        if (stakes != 0.0).any():
            issues.append("paper signal history contains a non-zero stake")
        runtime = signals["runtime_eligible"].astype(str).str.lower().isin(
            {"true", "1", "yes"})
        if runtime.any():
            issues.append("paper signal history contains a runtime-eligible row")
    return {
        "passed": not issues, "issues": issues,
        "quote_rows": int(len(quotes)), "signal_rows": int(len(signals)),
    }


def capture(now=None) -> dict:
    try:
        quote = capture_quotes(now=now)
        signal = capture_signals(now=now)
        report = build_report(now=now)
        payload = {
            "status": "success", "updated_at": _utc_now(now).isoformat(),
            "quote_capture": quote, "signal_capture": signal,
            "report_summary": {key: report[key] for key in (
                "signals", "signals_with_latest_quote", "closing_evidence_rows", "label")},
        }
        _atomic_json(REPORT_JSON, report)
        _atomic_json(STATUS_JSON, payload)
        return payload
    except Exception as exc:
        _atomic_json(STATUS_JSON, {
            "status": "failure", "updated_at": _utc_now(now).isoformat(),
            "error": str(exc),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if not args.capture and not args.report:
        parser.error("choose --capture and/or --report")
    if args.capture:
        print(json.dumps(capture(), indent=2))
    elif args.report:
        report = build_report()
        _atomic_json(REPORT_JSON, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
