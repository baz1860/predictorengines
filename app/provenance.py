"""Data provenance & refresh hygiene for every engine (V3 M7).

Three offline, network-free capabilities shared by all engines:

1. **Manifests** — `data_manifest.json` per engine recording, for each key input,
   its path, `fetched_at` (file mtime), row count, schema version and a source
   label. Written by the update scripts after a refresh.
2. **Freshness** — staleness warnings derived purely from file mtimes, surfaced
   in each engine's schema so the UI can flag stale fixtures / odds / field /
   model params. Advisory only — never blocks offline operation.
3. **Manual-odds schema checks** — `validate_odds_file()` returns actionable
   errors that name the **row number, column and expected value** for a malformed
   manual odds CSV.

Runnable: `python3 -m app.provenance --write [--engine X]`
          `python3 -m app.provenance --freshness`
          `python3 -m app.provenance --check-odds <engine>`
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
MANIFEST_NAME = "data_manifest.json"

# Per-role staleness thresholds (days). Advisory — purely for UI warnings.
_STALE_DAYS = {
    "results": 3, "fixtures": 3, "games": 3, "rounds": 3,
    "field": 2, "odds": 1, "model": 30,
    "results_live": 3 / 24,
    "fixtures_live": 3 / 24,
    "odds_live": 1 / 24,
    "lineups": 15 / 1440,
    "availability": 12 / 24,
    "stats": 3 / 24,
}

# Each engine's key inputs: (key, repo-relative path, role, source-label).
ENGINE_INPUTS: dict[str, list[tuple[str, str, str, str]]] = {
    "worldcup": [
        ("results", "data/results.csv", "results_live", "martj42/international_results + local"),
        ("odds", "data/odds_live.csv", "odds_live", "The Odds API / manual (snapshot written by edge.py)"),
        ("live_fixtures", "data/worldcup/fixtures_live.csv", "fixtures_live", "Bzzoiro Sports Data fixtures"),
        ("availability", "data/worldcup/player_availability.csv", "availability", "Bzzoiro Sports Data injuries/suspensions + manual"),
        ("lineups", "data/worldcup/lineups.csv", "lineups", "Bzzoiro Sports Data lineups"),
        ("match_stats", "data/worldcup/match_stats.csv", "stats", "Bzzoiro Sports Data match statistics"),
        ("market_snapshots", "data/worldcup/market_snapshots.csv", "odds_live", "The Odds API normalized market snapshots"),
        ("model", "data/dc_params.json", "model", "dixoncoles.py --fit"),
        ("squads", "data/squad_ratings.csv", "model", "squads.py"),
    ],
    "club_soccer": [
        ("fixtures", "club_soccer/data/fixtures.csv", "fixtures", "fetch.py / football-data"),
        ("odds", "club_soccer/data/odds.csv", "odds", "API-Football / The Odds API / manual"),
        ("model", "club_soccer/data/model_params.json", "model", "model.py --fit"),
    ],
    "cfb": [
        ("games", "cfb/data/games.csv", "games", "fetch_data.py / CFBD"),
        ("odds", "cfb/odds.csv", "odds", "manual"),
        ("model", "cfb/data/power_params.json", "model", "power.py --fit"),
    ],
    "golf": [
        ("field", "golf/data/field.csv", "field", "ESPN/golf.refresh"),
        ("rounds", "golf/data/rounds.csv", "rounds", "ESPN scoreboard / fetch.py --accumulate"),
        ("free_db", "golf/data/golf.db", "stats", "SQLite free-source cache"),
        ("free_manifest", "golf/data/free_source_manifest.json", "stats", "golf.refresh provider QA"),
        ("pga_stats", "golf/data/pgatour_stats.csv", "stats", "PGA Tour public stats pages"),
        ("odds", "golf/data/odds.csv", "odds", "manual / The Odds API majors"),
        ("threeballs", "golf/data/threeballs.csv", "odds", "manual pasted 3-ball boards"),
        ("model", "golf/data/model_params.json", "model", "model.py --fit"),
    ],
    "tennis": [
        ("matches", "tennis/data/matches.csv", "results", "fetch.py --seed / --accumulate"),
        ("draw", "tennis/data/draw.csv", "fixtures", "fetch.py --draw-template / manual"),
        ("odds", "tennis/data/odds.csv", "odds", "manual"),
        ("model_atp", "tennis/data/atp_model_params.json", "model", "model.py --fit --tour atp"),
        ("model_wta", "tennis/data/wta_model_params.json", "model", "model.py --fit --tour wta"),
    ],
}

# Where each engine's manifest lives (co-located with its data dir).
MANIFEST_DIRS = {"worldcup": "data", "club_soccer": "club_soccer/data",
                 "cfb": "cfb/data", "golf": "golf/data", "tennis": "tennis/data"}

ODDS_FILES = {"worldcup": "odds.csv", "club_soccer": "club_soccer/data/odds.csv",
              "cfb": "cfb/odds.csv", "golf": "golf/data/odds.csv",
              "tennis": "tennis/data/odds.csv"}


# ── helpers ───────────────────────────────────────────────────────────────────
def _mtime_iso(p: Path) -> str | None:
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")


def _age_days(p: Path) -> float | None:
    if not p.exists():
        return None
    age = datetime.now(timezone.utc).timestamp() - p.stat().st_mtime
    return round(age / 86400.0, 2)


def _row_count(p: Path) -> int | None:
    if not p.exists() or p.suffix.lower() != ".csv":
        return None
    try:
        with p.open(newline="") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)  # minus header
    except Exception:
        return None


def _fmt_age_days(days: float | None) -> str:
    if days is None:
        return "unknown age"
    if days < 1:
        return f"{days * 24:.1f}h old"
    return f"{days:.1f}d old"


def _fmt_limit_days(days: float) -> str:
    if days < 1:
        return f"{days * 24:.1f}h"
    return f"{days:.0f}d"


# ── manifests ─────────────────────────────────────────────────────────────────
def build_manifest(engine: str) -> dict:
    inputs = ENGINE_INPUTS.get(engine, [])
    entries = {}
    for key, rel, role, source in inputs:
        p = ROOT / rel
        entries[key] = {
            "path": rel, "role": role, "source": source,
            "exists": p.exists(), "fetched_at": _mtime_iso(p),
            "rows": _row_count(p), "schema_version": SCHEMA_VERSION,
        }
    return {"engine": engine, "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "inputs": entries}


def manifest_path(engine: str) -> Path:
    return ROOT / MANIFEST_DIRS.get(engine, "data") / MANIFEST_NAME


def write_manifest(engine: str) -> Path:
    m = build_manifest(engine)
    path = manifest_path(engine)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2))
    return path


def read_manifest(engine: str) -> dict | None:
    path = manifest_path(engine)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ── freshness (offline, mtime-based) ──────────────────────────────────────────
def freshness(engine: str) -> list[dict]:
    """Per-input staleness, computed from file mtimes only (no network, no
    manifest needed). status ∈ {ok, stale, missing}."""
    out = []
    for key, rel, role, _src in ENGINE_INPUTS.get(engine, []):
        p = ROOT / rel
        if not p.exists():
            out.append({"key": key, "role": role, "status": "missing",
                        "age_days": None, "message": f"{key}: file missing ({rel})"})
            continue
        age = _age_days(p)
        limit = _STALE_DAYS.get(role, 7)
        status = "stale" if (age is not None and age > limit) else "ok"
        msg = (f"{key}: {_fmt_age_days(age)} (> {_fmt_limit_days(limit)})"
               if status == "stale" else f"{key}: {_fmt_age_days(age)}")
        out.append({"key": key, "role": role, "status": status,
                    "age_days": age, "message": msg})
    return out


def freshness_warnings(engine: str) -> list[str]:
    """Concise warning strings for the UI — only stale/missing inputs. Never
    raises; returns [] if anything goes wrong so it can't break a schema call."""
    try:
        return [f["message"] for f in freshness(engine)
                if f["status"] in ("stale", "missing")]
    except Exception:
        return []


# ── manual-odds schema checks ─────────────────────────────────────────────────
_LONG_SIDES = {
    "club_soccer": {"1x2": {"home", "draw", "away"}, "total": {"over", "under"},
                    "btts": {"yes", "no"}},
    "cfb": {"ml": {"home", "away"}, "spread": {"home", "away"},
            "total": {"over", "under"}},
}
_LONG_COLUMNS = {
    "club_soccer": ["date", "competition", "home", "away", "market", "side", "line", "odds"],
    "cfb": ["date", "home", "away", "neutral", "market", "side", "line", "odds"],
}
_WIDE_ODDS_COLS = {
    "worldcup": ["odds_home", "odds_draw", "odds_away", "odds_over25",
                 "odds_under25", "odds_btts_yes", "odds_btts_no"],
    "golf": ["odds_win", "odds_top5", "odds_top10", "odds_top20", "odds_cut", "odds_nocut"],
}
_WIDE_KEY_COLS = {"worldcup": ["date", "home", "away"], "golf": ["name"]}


def _err(row, column, value, expected):
    return {"row": row, "column": column, "value": value, "expected": expected,
            "message": f"row {row}, column '{column}': {value!r} — expected {expected}"}


def _check_odds_value(v) -> bool:
    """Blank is allowed (unfilled template); otherwise a decimal > 1.0."""
    if v is None or str(v).strip() == "":
        return True
    try:
        return float(v) > 1.0
    except (TypeError, ValueError):
        return False


def validate_odds_file(engine: str, path: str | Path | None = None) -> list[dict]:
    """Return a list of actionable error dicts (row/column/value/expected) for a
    manual odds CSV. Empty list = valid (or nothing to check). `row` is the
    1-based data row (header is row 0)."""
    p = Path(path) if path else (ROOT / ODDS_FILES.get(engine, ""))
    if not p or not p.exists():
        return []
    try:
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows = list(reader)
    except Exception as e:
        return [{"row": 0, "column": "", "value": str(e), "expected": "a readable CSV",
                 "message": f"could not read {p.name}: {e}"}]

    errors: list[dict] = []
    if engine in _LONG_COLUMNS:
        for col in _LONG_COLUMNS[engine]:
            if col not in header:
                errors.append(_err(0, col, "<missing>", "this column in the header"))
        sides = _LONG_SIDES[engine]
        for i, r in enumerate(rows, start=1):
            market = (r.get("market") or "").strip().lower()
            side = (r.get("side") or "").strip().lower()
            if market and market not in sides:
                errors.append(_err(i, "market", market, f"one of {sorted(sides)}"))
            elif market and side and side not in sides[market]:
                errors.append(_err(i, "side", side,
                                   f"one of {sorted(sides[market])} for market '{market}'"))
            if not _check_odds_value(r.get("odds")):
                errors.append(_err(i, "odds", r.get("odds"), "blank or a decimal > 1.0"))
            line = (r.get("line") or "").strip()
            if line:
                try:
                    float(line)
                except ValueError:
                    errors.append(_err(i, "line", line, "blank or a number"))
            if engine == "cfb":
                neu = (r.get("neutral") or "").strip()
                if neu and neu not in ("0", "1"):
                    errors.append(_err(i, "neutral", neu, "0 or 1"))
                if market in ("spread", "total") and not line and _check_odds_value(r.get("odds")) \
                        and str(r.get("odds") or "").strip():
                    errors.append(_err(i, "line", "<blank>",
                                       f"a number ('{market}' needs a line)"))
    elif engine in _WIDE_ODDS_COLS:
        for col in _WIDE_KEY_COLS[engine]:
            if col not in header:
                errors.append(_err(0, col, "<missing>", "this column in the header"))
        priced_cols = [c for c in _WIDE_ODDS_COLS[engine] if c in header]
        for i, r in enumerate(rows, start=1):
            for col in priced_cols:
                if not _check_odds_value(r.get(col)):
                    errors.append(_err(i, col, r.get(col), "blank or a decimal > 1.0"))
    return errors


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Data provenance & refresh hygiene (V3 M7)")
    ap.add_argument("--engine", help="limit to one engine slug (default: all)")
    ap.add_argument("--write", action="store_true", help="write data_manifest.json per engine")
    ap.add_argument("--freshness", action="store_true", help="print freshness warnings")
    ap.add_argument("--check-odds", metavar="ENGINE", help="validate an engine's manual odds file")
    args = ap.parse_args()

    engines = [args.engine] if args.engine else list(ENGINE_INPUTS)

    if args.check_odds:
        errs = validate_odds_file(args.check_odds)
        if not errs:
            print(f"{args.check_odds}: odds file OK (or nothing to check).")
        else:
            print(f"{args.check_odds}: {len(errs)} issue(s):")
            for e in errs:
                print("  " + e["message"])
        return

    if args.write:
        for eng in engines:
            path = write_manifest(eng)
            print(f"wrote {path.relative_to(ROOT)}")

    if args.freshness or not args.write:
        for eng in engines:
            warns = freshness_warnings(eng)
            if warns:
                print(f"{eng}:")
                for w in warns:
                    print(f"  ⚠ {w}")
            else:
                print(f"{eng}: all inputs fresh.")


if __name__ == "__main__":
    main()
