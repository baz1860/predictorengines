#!/usr/bin/env python3
"""Data health checks for the Club Soccer engine.

Run standalone (`python3 -m club_soccer.health`) or call `run_checks()` from
season.py / update.sh. Exit code is 1 only when a hard check fails (future-
dated finished rows, duplicate fixture_ids) — everything else is reported,
never fatal, per the offline-first / never-raise-out-of-a-pipeline rule.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import model as M

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"

# Off-season months (Jun/Jul): a stale "days since last result" is expected,
# not a warning sign, since most tracked leagues are on their summer break.
_OFF_SEASON_MONTHS = {6, 7}
_FINISHED_STATUSES = {"FT", "FIN", "AET", "PEN"}


def run_checks() -> dict:
    """Compute and print the club soccer data health report.

    Returns a dict with every metric plus "ok": bool (True iff both hard
    checks pass).
    """
    today = datetime.now(timezone.utc).date()
    report: dict = {"checked_at": str(today)}

    if not FIXTURES.exists():
        report.update({
            "future_ft_rows": None, "duplicate_fixture_ids": None,
            "days_since_last_result": None, "upcoming_count": None,
            "stats_coverage": None, "ok": False,
            "error": f"{FIXTURES} not found",
        })
        return report

    df = pd.read_csv(FIXTURES)

    finished_mask = df["status"].astype(str).str.upper().isin(_FINISHED_STATUSES)
    future_ft = df[finished_mask & (df["date"].astype(str) > str(today))]
    report["future_ft_rows"] = int(len(future_ft))

    report["duplicate_fixture_ids"] = int(df["fixture_id"].duplicated().sum())

    played = df.dropna(subset=["home_goals", "away_goals"])
    if not played.empty:
        last_result_date = pd.to_datetime(played["date"]).max().date()
        days_since = (today - last_result_date).days
    else:
        days_since = None
    report["days_since_last_result"] = days_since

    fx = M.load_fixtures()
    report["upcoming_count"] = int(len(M.upcoming(fx)))

    if not played.empty:
        sot_present = played["home_sot"].notna() & played["away_sot"].notna()
        report["stats_coverage"] = round(float(sot_present.mean()), 4)
    else:
        report["stats_coverage"] = None

    report["ok"] = report["future_ft_rows"] == 0 and report["duplicate_fixture_ids"] == 0

    print(f"Club Soccer health check ({today}):")
    status = "PASS" if report["future_ft_rows"] == 0 else "FAIL"
    print(f"  [{status}] future_ft_rows = {report['future_ft_rows']} (must be 0)")
    status = "PASS" if report["duplicate_fixture_ids"] == 0 else "FAIL"
    print(f"  [{status}] duplicate_fixture_ids = {report['duplicate_fixture_ids']} (must be 0)")

    if days_since is None:
        print("  [WARN] days_since_last_result: no played rows found")
    else:
        level = "INFO" if today.month in _OFF_SEASON_MONTHS else (
            "WARN" if days_since > 7 else "INFO")
        print(f"  [{level}] days_since_last_result = {days_since}")

    print(f"  [INFO] upcoming_count = {report['upcoming_count']}")
    cov = report["stats_coverage"]
    print(f"  [INFO] stats_coverage (SoT present) = "
          f"{'n/a' if cov is None else f'{cov:.1%}'}")

    return report


def main() -> None:
    report = run_checks()
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
