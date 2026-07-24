#!/usr/bin/env python3
"""Data health checks for the Club Soccer engine.

Run standalone (`python3 -m club_soccer.health`) or call `run_checks()` from
season.py / update.sh. Exit code is 1 only when a hard check fails (future-
dated finished rows, duplicate fixture_ids) — everything else is reported,
never fatal, per the offline-first / never-raise-out-of-a-pipeline rule.
"""
from __future__ import annotations

import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import model as M
from .identities import (conflicting_score_identity_count,
                         duplicate_identity_count)

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
            "duplicate_match_identities": None, "conflicting_score_identities": None,
            "days_since_last_result": None, "upcoming_count": None,
            "stats_coverage": None, "xg_coverage": None,
            "coverage_by_type": {}, "cup_coverage": {}, "xg_source_counts": {},
            "player_cache": {}, "ok": False,
            "error": f"{FIXTURES} not found",
        })
        return report

    df = pd.read_csv(FIXTURES, low_memory=False)

    finished_mask = df["status"].astype(str).str.upper().isin(_FINISHED_STATUSES)
    future_ft = df[finished_mask & (df["date"].astype(str) > str(today))]
    report["future_ft_rows"] = int(len(future_ft))

    report["duplicate_fixture_ids"] = int(df["fixture_id"].duplicated().sum())

    # Void statuses (postponed/cancelled/abandoned/...) must never retain a
    # result: a POS row with a score keeps training the model on a result
    # that never stood. Hard check.
    from .schema import (RESULT_COLUMNS, VOID_STATUSES, QUARANTINE_STATUSES,
                         normalize_status)
    status_norm = df["status"].map(normalize_status)
    void_mask = status_norm.isin(VOID_STATUSES)
    # Unmapped provider statuses are inert (never trained, priced or settled)
    # but they are invisible work: surface the count and the raw strings.
    quarantined = status_norm.isin(QUARANTINE_STATUSES)
    report["quarantined_status_rows"] = int(quarantined.sum())
    if quarantined.any() and "status_raw" in df.columns:
        report["quarantined_status_values"] = sorted(
            set(df.loc[quarantined, "status_raw"].dropna().astype(str)))[:20]
    result_cols = [c for c in RESULT_COLUMNS if c in df.columns]
    has_result = df[result_cols].notna().any(axis=1) if result_cols else False
    report["void_with_results"] = int((void_mask & has_result).sum())
    report["duplicate_match_identities"] = duplicate_identity_count(df)
    report["conflicting_score_identities"] = conflicting_score_identity_count(df)

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
        if "home_xg" in played.columns and "away_xg" in played.columns:
            xg_present = played["home_xg"].notna() & played["away_xg"].notna()
            report["xg_coverage"] = round(float(xg_present.mean()), 4)
        else:
            report["xg_coverage"] = 0.0

        coverage_by_type = {}
        type_series = (played["type"] if "type" in played.columns
                       else pd.Series("unknown", index=played.index))
        for kind, grp in played.groupby(type_series, dropna=False):
            sot = grp["home_sot"].notna() & grp["away_sot"].notna()
            xg = ((grp["home_xg"].notna() & grp["away_xg"].notna())
                  if "home_xg" in grp.columns and "away_xg" in grp.columns
                  else pd.Series(False, index=grp.index))
            coverage_by_type[str(kind)] = {
                "played": int(len(grp)),
                "sot": round(float(sot.mean()), 4),
                "xg": round(float(xg.mean()), 4),
            }
        report["coverage_by_type"] = coverage_by_type
        if "xg_source" in played.columns:
            source = (played["xg_source"].fillna("missing").astype(str).str.strip()
                      .replace({"": "missing", "nan": "missing"}))
            report["xg_source_counts"] = {str(k): int(v) for k, v in source.value_counts().items()}
        else:
            report["xg_source_counts"] = {"missing": int(len(played))}

        cups = played[played["type"] == "cup"] if "type" in played.columns else played.iloc[0:0]
        if not cups.empty:
            cup_cov = {}
            for comp, grp in cups.groupby("competition"):
                sot = grp["home_sot"].notna() & grp["away_sot"].notna()
                xg = ((grp["home_xg"].notna() & grp["away_xg"].notna())
                      if "home_xg" in grp.columns and "away_xg" in grp.columns
                      else pd.Series(False, index=grp.index))
                shootouts = (grp["shootout_winner"].notna()
                             & grp["shootout_winner"].astype(str).ne("")) \
                    if "shootout_winner" in grp.columns else pd.Series(False, index=grp.index)
                cup_cov[str(comp)] = {
                    "played": int(len(grp)),
                    "sot": round(float(sot.mean()), 4),
                    "xg": round(float(xg.mean()), 4),
                    "shootouts_recorded": int(shootouts.sum()),
                }
            report["cup_coverage"] = cup_cov
    else:
        report["stats_coverage"] = None
        report["xg_coverage"] = None
        report["coverage_by_type"] = {}
        report["cup_coverage"] = {}
        report["xg_source_counts"] = {}

    cache = DATA / "player_stats_cache.json"
    player_report = {"exists": cache.exists(), "schema": None, "players": 0,
                     "players_with_apps": 0, "apps": 0,
                     "oldest_app_date": None, "latest_app_date": None}
    if cache.exists():
        try:
            raw = json.loads(cache.read_text())
            records = [v for k, v in raw.items() if k != "v" and isinstance(v, dict)]
            app_dates = [str(a.get("date")) for v in records for a in (v.get("apps") or [])
                         if a.get("date")]
            player_report.update({
                "schema": raw.get("v"),
                "players": len(records),
                "players_with_apps": sum(bool(v.get("apps")) for v in records),
                "apps": sum(len(v.get("apps") or []) for v in records),
                "oldest_app_date": min(app_dates) if app_dates else None,
                "latest_app_date": max(app_dates) if app_dates else None,
            })
        except Exception as exc:
            player_report["error"] = str(exc)
    report["player_cache"] = player_report

    report["ok"] = (report["future_ft_rows"] == 0
                    and report["duplicate_fixture_ids"] == 0
                    and report["duplicate_match_identities"] == 0
                    and report["conflicting_score_identities"] == 0
                    and report.get("void_with_results", 0) == 0)

    print(f"Club Soccer health check ({today}):")
    status = "PASS" if report["future_ft_rows"] == 0 else "FAIL"
    print(f"  [{status}] future_ft_rows = {report['future_ft_rows']} (must be 0)")
    status = "PASS" if report.get("void_with_results", 0) == 0 else "FAIL"
    print(f"  [{status}] void_with_results = {report.get('void_with_results')} (must be 0)")
    status = "PASS" if report["duplicate_fixture_ids"] == 0 else "FAIL"
    print(f"  [{status}] duplicate_fixture_ids = {report['duplicate_fixture_ids']} (must be 0)")
    status = "PASS" if report["duplicate_match_identities"] == 0 else "FAIL"
    print(f"  [{status}] duplicate_match_identities = {report['duplicate_match_identities']} (must be 0)")
    status = "PASS" if report["conflicting_score_identities"] == 0 else "FAIL"
    print(f"  [{status}] conflicting_score_identities = {report['conflicting_score_identities']} (must be 0)")

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
    xg_cov = report.get("xg_coverage")
    print(f"  [INFO] xg_coverage (both sides present) = "
          f"{'n/a' if xg_cov is None else f'{xg_cov:.1%}'}")
    print(f"  [INFO] xg_sources = {report.get('xg_source_counts', {})}")
    for kind, values in report.get("coverage_by_type", {}).items():
        print(f"  [INFO] {kind}: played={values['played']} SoT={values['sot']:.1%} "
              f"xG={values['xg']:.1%}")
    if report.get("cup_coverage"):
        for comp, values in report["cup_coverage"].items():
            print(f"  [INFO] cup {comp}: played={values['played']} "
                  f"SoT={values['sot']:.1%} xG={values['xg']:.1%} "
                  f"shootouts={values['shootouts_recorded']}")
    pc = report["player_cache"]
    print(f"  [INFO] player_cache: schema={pc.get('schema')} players={pc.get('players', 0)} "
          f"apps={pc.get('apps', 0)} latest={pc.get('latest_app_date') or 'n/a'}")

    # Per-league staleness. A single global "days since last result" hides the
    # failure that matters after the P3 expansion: one league silently ceasing
    # to update while the other 40 keep flowing, so the aggregate looks fine
    # and that league is priced off frozen ratings. Off-season gaps are normal,
    # so only a league with NO upcoming fixtures and a long gap is flagged.
    try:
        from . import seed_fdcouk_leagues as SFL
        rows = SFL.staleness()
        stale = [r for r in rows if r["warn"]]
        report["league_staleness"] = rows
        report["stale_leagues"] = [r["competition"] for r in stale]
        print(f"  [INFO] leagues tracked = {len(rows)}; "
              f"stale in-season (no upcoming, >21d) = {len(stale)}")

        # Authoritative check for the BSD-less leagues: is the SOURCE ahead of
        # us? The season heuristic above is a cheap proxy that still flags
        # pre-season gaps; this compares against the actual fd.co.uk file and
        # only warns when there really are results we failed to ingest.
        # Networked, so best-effort — the offline INFO above always prints.
        try:
            behind = [r for r in SFL.refresh_health() if r.get("behind")]
            report["leagues_behind_source"] = [r["competition"] for r in behind]
            if behind:
                print(f"  [WARN] {len(behind)} league(s) are BEHIND their source "
                      "— the fd.co.uk refresh is not ingesting available results:")
                for r in behind[:8]:
                    print(f"         {r['competition']} — ours {r.get('our_latest')} "
                          f"vs source {r.get('source_latest')}")
            else:
                print("  [INFO] all BSD-less leagues are level with their source "
                      "(no genuine staleness)")
        except Exception as exc:
            print(f"  [INFO] source-freshness check skipped ({exc})")
    except Exception as exc:
        print(f"  [INFO] league staleness unavailable ({exc})")

    return report


def main() -> None:
    report = run_checks()
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
