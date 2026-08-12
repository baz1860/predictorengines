#!/usr/bin/env python3
"""Offline preflight: report missing data files and missing API keys (V3 M2).

Pure local check — never makes a network call. Tells you, per engine, which key
inputs and fitted-model files are present and how stale they are, and which API
keys are configured (masked). Use it before a refresh/predict session, or as a
quick "is this checkout ready?" smoke test.

Usage:
  python3 preflight.py            # human-readable table
  python3 preflight.py --json     # machine-readable
Exit code is 0 always (missing data must not block offline operation); read the
report to decide what to refresh.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# Per-engine key inputs. (label, relative path, required?) — required files that
# are missing show as ✗; optional ones as ·.
ENGINE_FILES: dict[str, list[tuple[str, str, bool]]] = {
    "worldcup": [
        ("results", "data/results.csv", True),
        ("goal model", "data/dc_params.json", True),
        ("calibration", "data/calibration.json", False),
        ("market blend", "data/market_blend.json", False),
    ],
    "club_soccer": [
        ("fixtures", "club_soccer/data/fixtures.csv", True),
        ("model params", "club_soccer/data/model_params.json", True),
        ("calibration", "club_soccer/data/calibration.json", False),
        ("odds", "club_soccer/data/odds.csv", False),
    ],
    "cfb": [
        ("games", "cfb/data/games.csv", True),
        ("power params", "cfb/data/power_params.json", True),
        ("upcoming", "cfb/data/upcoming.csv", False),
        ("odds", "cfb/odds.csv", False),
    ],
    "golf": [
        ("rounds", "golf/data/rounds.csv", True),
        ("model params", "golf/data/model_params.json", True),
        ("free db", "golf/data/golf.db", False),
        ("free manifest", "golf/data/free_source_manifest.json", False),
        ("PGA stats", "golf/data/pgatour_stats.csv", False),
        ("field", "golf/data/field.csv", False),
        ("odds", "golf/data/odds.csv", False),
        ("3-balls", "golf/data/threeballs.csv", False),
    ],
    "tennis": [
        ("matches", "tennis/data/matches.csv", True),
        ("ATP model", "tennis/data/atp_model_params.json", True),
        ("WTA model", "tennis/data/wta_model_params.json", True),
        ("draw", "tennis/data/draw.csv", False),
        ("odds", "tennis/data/odds.csv", False),
    ],
    "nhl": [
        ("team stats", "nhl/data/team_stats.csv", True),
        ("fixtures", "nhl/data/fixtures.csv", False),
        ("results", "nhl/data/results.csv", False),
        ("odds", "nhl/data/odds.csv", False),
    ],
}

# Which API keys each engine can use (for the masked-key report).
ENGINE_KEYS: dict[str, list[str]] = {
    "worldcup": ["the-odds-api"],
    "club_soccer": ["api-football", "the-odds-api"],
    "cfb": ["collegefootballdata", "the-odds-api"],
    "golf": ["the-odds-api"],
    "tennis": [],
    "nhl": [],
}


def _age(path: Path) -> str:
    secs = time.time() - path.stat().st_mtime
    days = secs / 86400
    if days >= 1:
        return f"{days:.0f}d"
    hours = secs / 3600
    return f"{hours:.0f}h" if hours >= 1 else f"{secs/60:.0f}m"


def build_report() -> dict:
    try:
        from app import settings_store
        keys_set = settings_store.public_view().get("odds_api_keys_set", {})
    except Exception:
        keys_set = {}

    engines = {}
    for eid, files in ENGINE_FILES.items():
        files = list(files)
        if eid == "cfb":
            season = date.today().year if date.today().month >= 2 else date.today().year - 1
            files += [
                ("season schedule", f"cfb/data/schedule_{season}.json", True),
                ("talent priors", f"cfb/data/cfbd/talent_{season}.json", False),
                ("returning priors", f"cfb/data/cfbd/returning_{season}.json", False),
            ]
        file_rows = []
        missing_required = 0
        for label, rel, required in files:
            p = ROOT / rel
            exists = p.exists()
            if required and not exists:
                missing_required += 1
            file_rows.append({
                "label": label, "path": rel, "exists": exists,
                "required": required,
                "age": _age(p) if exists else None,
            })
        key_rows = []
        for k in ENGINE_KEYS.get(eid, []):
            configured = bool(keys_set.get(k))
            if eid == "cfb" and not configured:
                try:
                    from api_keys import get_key
                    env = "CFBD_API_KEY" if k == "collegefootballdata" else "THE_ODDS_API_KEY"
                    configured = bool(get_key(k, env=env))
                except Exception:
                    configured = bool(os.environ.get(
                        "CFBD_API_KEY" if k == "collegefootballdata" else "THE_ODDS_API_KEY"))
            key_rows.append({"source": k, "set": configured})
        issues = []
        model_state = None
        run_status = None
        if eid == "cfb":
            schedule_path = ROOT / f"cfb/data/schedule_{season}.json"
            try:
                schedule = json.loads(schedule_path.read_text())
                if not isinstance(schedule, list) or len(schedule) < 20:
                    issues.append(f"season schedule has no usable {season} coverage")
                else:
                    wrong = sum(g.get("season") != season for g in schedule)
                    ids = [str(g.get("id") or "").strip() for g in schedule]
                    if wrong:
                        issues.append(f"season schedule contains {wrong} non-{season} row(s)")
                    if any(not game_id for game_id in ids) or len(ids) != len(set(ids)):
                        issues.append("season schedule has missing or duplicate event IDs")
                    starts = pd.to_datetime(
                        [g.get("startDate") for g in schedule], errors="coerce", utc=True)
                    if starts.isna().any():
                        issues.append("season schedule has invalid kickoff timestamps")
            except Exception:
                issues.append(f"season schedule is unreadable for {season}")

            for label in ("talent priors", "returning priors"):
                row = next((r for r in file_rows if r["label"] == label), None)
                if not row or not row["exists"]:
                    issues.append(f"{label} missing for {season}")
                    continue
                try:
                    payload = json.loads((ROOT / row["path"]).read_text())
                    if not isinstance(payload, list) or len(payload) < 20:
                        issues.append(f"{label} has no usable {season} coverage")
                except Exception:
                    issues.append(f"{label} is unreadable for {season}")
            try:
                params = json.loads((ROOT / "cfb/data/power_params.json").read_text())
                teams = params.get("teams") if isinstance(params, dict) else None
                asof = pd.Timestamp(params.get("asof"))
                completed = pd.read_csv(ROOT / "cfb/data/games.csv", usecols=["date"])
                games_through = pd.to_datetime(completed["date"], errors="coerce").max()
                if not isinstance(teams, dict) or len(teams) < 100:
                    issues.append("power params have inadequate team coverage")
                if pd.isna(asof) or pd.isna(games_through):
                    issues.append("power params or completed-game dates are invalid")
                elif asof < games_through:
                    issues.append(
                        f"power params are stale ({asof.date()} before {games_through.date()})")
                elif asof > games_through + pd.Timedelta(days=2):
                    issues.append("power params as-of date is inconsistent with completed games")
            except Exception:
                issues.append("power params could not be semantically validated")
            try:
                from cfb import elo as cfb_elo
                _, model_state = cfb_elo.build_as_of(season, as_of=date.today())
                if model_state.get("model_season") != season:
                    issues.append("CFB snapshot does not match the target season")
                if model_state.get("source_max_season", season) > season:
                    issues.append("CFB snapshot contains future-season results")
            except Exception as exc:
                issues.append(f"CFB snapshot build failed: {exc}")
            missing_keys = [r["source"] for r in key_rows if not r["set"]]
            if missing_keys:
                issues.append("missing API key(s): " + ", ".join(missing_keys))
            try:
                from app.provenance import validate_odds_file
                odds_errors = validate_odds_file("cfb")
                if odds_errors:
                    issues.append(f"odds schema/provenance has {len(odds_errors)} issue(s)")
                else:
                    from cfb.edge import prepare_odds
                    odds = pd.read_csv(ROOT / "cfb/odds.csv")
                    gated = prepare_odds(odds, {season})
                    if gated.empty or not gated["quote_eligible"].any():
                        issues.append("odds contain no fresh, matched, paired executable quotes")
            except Exception:
                issues.append("odds schema/provenance could not be checked")
            status_path = ROOT / "cfb/data/update_status.json"
            if status_path.exists():
                try:
                    run_status = json.loads(status_path.read_text())
                except Exception:
                    issues.append("CFB update status is unreadable")
        engines[eid] = {
            "files": file_rows,
            "keys": key_rows,
            "ready": missing_required == 0 and not issues,
            "diagnostic_ready": missing_required == 0,
            "missing_required": missing_required,
            "issues": issues,
            **({"model_state": model_state, "run_status": run_status}
               if eid == "cfb" else {}),
        }
    return {"engines": engines, "international_data": _international_health()}


def _international_health() -> dict:
    """Data-integrity summary for the international module.

    Surfaced in preflight because these failures are invisible in a file-existence
    check: results.csv can be present, recent and internally corrupt. Duplicate
    fixtures and unclassified teams both produce confident, wrong predictions.
    """
    try:
        from international import gate as G
        from international.odds import OddsStore
        from international.store import FixtureStore
        from international import venues as V
    except Exception as exc:                                   # pragma: no cover
        return {"available": False, "error": str(exc)}

    try:
        failures = G.run(strict=False)
        fixtures = FixtureStore().load()
        no_tz = 0
        if not fixtures.empty:
            no_tz = int((fixtures.venue_tz.isna()
                         | (fixtures.venue_tz.astype(str).str.strip() == "")).sum())
        return {
            "available": True,
            "gate_pass": not failures,
            "failures": failures,
            "fixtures": len(fixtures),
            "fixtures_without_timezone": no_tz,
            "venues": V.coverage(),
            "odds": OddsStore().coverage(),
        }
    except Exception as exc:                                   # pragma: no cover
        return {"available": False, "error": str(exc)}


def _print_international(report: dict) -> None:
    intl = report.get("international_data") or {}
    if not intl.get("available"):
        if intl:
            print(f"\ninternational_data  [unavailable: {intl.get('error', '?')}]")
        return
    flag = "healthy" if intl["gate_pass"] else "GATE FAILING"
    print(f"\ninternational_data  [{flag}]")
    print(f"  fixtures in store        {intl['fixtures']}"
          f"  ({intl['fixtures_without_timezone']} without a venue timezone)")
    v, o = intl["venues"], intl["odds"]
    print(f"  venues                   {v['venues']} "
          f"({v['with_timezone']} with a timezone)")
    print(f"  odds snapshots           {o['snapshots']} across {o['fixtures']} "
          f"fixture(s); {o['priced_fixtures']} ever priced")
    for f in intl.get("failures", []):
        print(f"  ! {f.splitlines()[0]}")


def _print(report: dict) -> None:
    for eid, e in report["engines"].items():
        if e["ready"]:
            flag = "ready"
        elif e.get("diagnostic_ready"):
            flag = "diagnostic only"
        else:
            flag = f"MISSING {e['missing_required']} required"
        print(f"\n{eid}  [{flag}]")
        for f in e["files"]:
            mark = "✓" if f["exists"] else ("✗" if f["required"] else "·")
            age = f"  ({f['age']})" if f["age"] else ""
            print(f"  {mark} {f['label']:<14} {f['path']}{age}")
        for k in e["keys"]:
            print(f"  {'✓' if k['set'] else '·'} key: {k['source']}"
                  f"{'' if k['set'] else ' (not set)'}")
        for issue in e.get("issues", []):
            print(f"  ⚠ {issue}")
    _print_international(report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--engine", choices=sorted(ENGINE_FILES))
    ap.add_argument("--require-ready", action="store_true",
                    help="exit non-zero unless selected engines are betting-ready")
    ap.add_argument("--require-diagnostic", action="store_true",
                    help="exit non-zero unless selected engines have core diagnostic inputs")
    ap.add_argument("--require-data-healthy", action="store_true",
                    help="exit non-zero if the international data gate is failing")
    args = ap.parse_args()
    report = build_report()
    selected = report["engines"]
    if args.engine:
        selected = {args.engine: selected[args.engine]}
    # Carry international_data through: an earlier version rebuilt `view` with only
    # the engines key, so the section was computed and then silently dropped.
    view = {"engines": selected,
            "international_data": report.get("international_data", {})}
    if args.json:
        print(json.dumps(view, indent=2))
    else:
        _print(view)
    if args.require_ready and any(not e["ready"] for e in selected.values()):
        return 1
    if args.require_diagnostic and any(not e.get("diagnostic_ready", e["ready"])
                                       for e in selected.values()):
        return 1
    if args.require_data_healthy:
        intl = view["international_data"]
        if not intl.get("available") or not intl.get("gate_pass"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
