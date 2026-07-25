"""Offline integrity checks for the authoritative golf history and model fit.

This module never fetches or writes data. It is safe to run in CI, from cron,
or before a model refresh:

    python3 -m golf.integrity
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

from .providers.legacy import ROUNDS_COLUMNS, ROUNDS_CSV

DATA_DIR = Path(__file__).parent / "data"
MODEL_PARAMS = DATA_DIR / "model_params.json"


def check_rounds(path: Path = ROUNDS_CSV) -> tuple[dict, list[str]]:
    errors: list[str] = []
    if not path.exists():
        return {}, [f"missing authoritative history: {path}"]

    seen: set[tuple[str, str, str, str]] = set()
    seen_names: set[tuple[str, str, str, str]] = set()
    event_meta: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    max_date = ""
    row_count = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in ROUNDS_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            return {}, [f"rounds.csv missing columns: {', '.join(missing)}"]
        for line_no, row in enumerate(reader, 2):
            row_count += 1
            event_id = str(row.get("tournament_id") or "").strip()
            tour = str(row.get("tour") or "").strip()
            player = str(row.get("player") or "").strip()
            player_id = str(row.get("dg_id") or "").strip() or player.casefold()
            round_no = str(row.get("round") or "").strip()
            key = (tour, event_id, player_id, round_no)
            if key in seen:
                errors.append(f"line {line_no}: duplicate round key {key}")
            seen.add(key)
            name_key = (tour, event_id, player.casefold(), round_no)
            if name_key in seen_names:
                errors.append(
                    f"line {line_no}: ambiguous duplicate display-name round {name_key}"
                )
            seen_names.add(name_key)

            if not event_id or not tour or not player:
                errors.append(f"line {line_no}: blank event, tour, or player identity")
            try:
                parsed_round = int(round_no)
                total_rounds = int(row["total_rounds"])
                if parsed_round < 1 or parsed_round > total_rounds:
                    errors.append(
                        f"line {line_no}: round {parsed_round} outside 1..{total_rounds}"
                    )
                score = float(row["score_to_par"])
                if not -15.0 <= score <= 30.0:
                    errors.append(
                        f"line {line_no}: impossible round score {score:g}"
                    )
                if int(row["made_cut"]) not in (0, 1):
                    raise ValueError("made_cut")
                if int(row["no_cut"]) not in (0, 1):
                    raise ValueError("no_cut")
                if int(row["multi_course"]) not in (0, 1):
                    raise ValueError("multi_course")
                if int(row["field_size"]) <= 0:
                    raise ValueError("field_size")
                cut_rule = int(row["cut_rule"])
                if not 1 <= cut_rule <= 70:
                    raise ValueError("cut_rule")
                realized = int(row["realized_cut_count"])
                if realized < 0 or realized > int(row["field_size"]):
                    raise ValueError("realized_cut_count")
            except (TypeError, ValueError):
                errors.append(f"line {line_no}: invalid numeric round fields")

            date = str(row.get("date") or "")[:10]
            try:
                dt.date.fromisoformat(date)
                max_date = max(max_date, date)
            except ValueError:
                errors.append(f"line {line_no}: invalid ISO date {date!r}")

            meta = event_meta[(tour, event_id)]
            for column in (
                "event_name",
                "course_id",
                "course_name",
                "course_par",
                "course_yards",
                "cut_round",
                "cut_rule",
                "realized_cut_count",
                "total_rounds",
                "no_cut",
                "multi_course",
            ):
                meta[column].add(str(row.get(column) or ""))

    for event_key, values in event_meta.items():
        inconsistent = [column for column, observed in values.items() if len(observed) > 1]
        if inconsistent:
            errors.append(
                f"event {event_key} has inconsistent metadata: {', '.join(inconsistent)}"
            )

    return {
        "rows": row_count,
        "events": len(event_meta),
        "max_date": max_date,
    }, errors


def check_model_params(path: Path = MODEL_PARAMS, *, history_max_date: str = "") -> list[str]:
    if not path.exists():
        return [f"missing fitted model: {path}"]
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return [f"invalid model_params.json: {exc}"]
    errors = []
    if not payload.get("players"):
        errors.append("model_params.json has no fitted players")
    if not payload.get("fitted_rounds"):
        errors.append("model_params.json has no fitted rounds")
    for key in ("birdie_rate_field", "bogey_rate_field", "double_bogey_rate_field"):
        if not payload.get(key):
            errors.append(f"model_params.json missing fitted scoring-shape field {key}")
    context = payload.get("course_context") or {}
    if not all(context.get(key) for key in ("par_mean", "par_sd", "yards_mean", "yards_sd")):
        errors.append("model_params.json missing measured course context")
    profiled = sum(
        bool(row.get("course_profile") or row.get("par_profile"))
        for row in (payload.get("players") or {}).values()
    )
    if profiled == 0:
        errors.append("model_params.json has no fitted player course profiles")
    asof = str(payload.get("asof") or "")[:10]
    if history_max_date:
        try:
            if dt.date.fromisoformat(asof) < dt.date.fromisoformat(history_max_date):
                errors.append(
                    f"model asof {asof} is older than history {history_max_date}"
                )
        except ValueError:
            errors.append(f"model has invalid asof date {asof or '(missing)'}")
    return errors


def run(
    rounds_path: Path = ROUNDS_CSV,
    model_path: Path = MODEL_PARAMS,
) -> tuple[dict, list[str]]:
    stats, errors = check_rounds(rounds_path)
    errors.extend(
        check_model_params(model_path, history_max_date=str(stats.get("max_date") or ""))
    )
    return stats, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check golf history/model integrity offline")
    parser.add_argument("--rounds", type=Path, default=ROUNDS_CSV)
    parser.add_argument("--model", type=Path, default=MODEL_PARAMS)
    parser.add_argument(
        "--rounds-only",
        action="store_true",
        help="validate authoritative history without requiring a post-refresh fit",
    )
    args = parser.parse_args()
    if args.rounds_only:
        stats, errors = check_rounds(args.rounds)
    else:
        stats, errors = run(args.rounds, args.model)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    print(
        "PASS: "
        f"{stats['rows']:,} rounds · {stats['events']:,} events · "
        f"{'history' if args.rounds_only else 'history/model'} through "
        f"{stats['max_date']}"
    )


if __name__ == "__main__":
    main()
