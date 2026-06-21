"""Free-source weekly refresh for the PGA golf engine.

This command is intentionally conservative: it gathers and caches free data,
exports the existing CSV contract, and reports provider QA warnings. It does not
force a bet or hide missing market data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import provider_qa as qa
from . import store
from .providers.espn import EspnGolfProvider
from .providers.odds_manual import ManualOddsProvider, THREEBALLS_RAW, write_threeballs_csv
from .providers.odds_theoddsapi import MAJOR_SPORT_KEYS, TheOddsApiGolfProvider
from .providers.pgatour_stats import PgaTourStatsProvider, write_stats_csv
from .providers.weather import OpenMeteoProvider

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh free-source PGA golf data")
    ap.add_argument("--season", type=int, default=None, help="PGA season/year")
    ap.add_argument("--event", default="", help="ESPN event/tournament id; default current leaderboard")
    ap.add_argument("--stats", action="store_true", help="pull public PGA Tour stat pages")
    ap.add_argument("--weather", action="store_true", help="pull Open-Meteo forecast when course coordinates exist")
    ap.add_argument("--odds-api-sport", default="",
                    help="optional The Odds API golf sport key for major outrights")
    ap.add_argument("--manual-raw", default=str(THREEBALLS_RAW),
                    help="manual 3-ball raw paste path")
    ap.add_argument("--round", type=int, default=1, dest="round_no",
                    help="round number for manual 3-ball raw paste")
    ap.add_argument("--fit", action="store_true", help="refit model_params.json after refresh")
    ap.add_argument("--use-cache", action="store_true",
                    help="prefer cached provider payloads when fresh")
    args = ap.parse_args()

    manifest = run_refresh(
        season=args.season,
        event=args.event,
        stats=args.stats,
        weather=args.weather,
        odds_api_sport=args.odds_api_sport,
        manual_raw=args.manual_raw,
        round_no=args.round_no,
        fit=args.fit,
        use_cache=args.use_cache,
    )
    _print_summary(manifest, Path(manifest["manifest_path"]))
    if manifest["qa"]["errors"]:
        sys.exit(2)


def run_refresh(
    *,
    season: int | None = None,
    event: str = "",
    stats: bool = False,
    weather: bool = False,
    odds_api_sport: str = "",
    manual_raw: str = str(THREEBALLS_RAW),
    round_no: int = 1,
    fit: bool = False,
    use_cache: bool = False,
) -> dict:
    """Run the free-source refresh and return the manifest payload.

    The CLI and the desktop app use this same function so provider behavior,
    cache writes, QA checks, and CSV exports stay identical across surfaces.
    """
    event_id = event
    db = store.init_db()
    checks: list[qa.SourceCheck] = []
    provider_rows = {}

    rounds_rows = store.import_rounds_csv()
    provider_rows["rounds_csv"] = rounds_rows

    espn = EspnGolfProvider()
    event = None
    field_rows = []
    try:
        event = espn.current_event(event_id or None, use_cache=use_cache)
        if event:
            with store.connect() as con:
                store.upsert_events(con, [event.as_store_row()])
        field_rows = espn.field(event.event_id if event else (event_id or None), use_cache=use_cache)
        checks.extend(espn.qa_checks(field_rows))
        provider_rows["espn_field"] = len(field_rows)
        if event and field_rows:
            with store.connect() as con:
                store.upsert_field(con, event.event_id, [r.as_store_row() for r in field_rows])
            store.export_field_csv(event.event_id)
    except Exception as exc:  # noqa: BLE001
        checks.append(qa.SourceCheck("espn.field", False, "error", str(exc), 0))
        provider_rows["espn_field"] = 0

    stats_written = None
    if stats:
        stats = PgaTourStatsProvider()
        stat_rows = []
        stat_payload = stats.fetch_default_stats(season=season, use_cache=use_cache)
        for stat_name, rows in stat_payload.items():
            stat_rows.extend(rows)
            checks.extend(stats.qa_checks(rows, label=f"pgatour_stats.{stat_name}"))
        provider_rows["pgatour_stats"] = len(stat_rows)
        if stat_rows:
            stats_written = write_stats_csv(stat_rows)
            with store.connect() as con:
                store.upsert_stat_rows(con, [r.as_dict() for r in stat_rows])

    weather_summary = {}
    if weather and event:
        weather_provider = OpenMeteoProvider()
        locs = weather_provider.load_course_locations()
        loc = locs.get(_course_key(event.course_name))
        if loc is None:
            checks.append(qa.SourceCheck(
                "open_meteo",
                False,
                "warning",
                f"no coordinates for course '{event.course_name}' in golf/data/course_locations.csv",
            ))
        else:
            try:
                payload = weather_provider.forecast(loc, event.start_date, use_cache=use_cache)
                weather_summary = weather_provider.summarize_wave(payload)
                provider_rows["open_meteo"] = weather_summary.get("hours", 0)
            except Exception as exc:  # noqa: BLE001
                checks.append(qa.SourceCheck("open_meteo", False, "warning", str(exc)))

    manual = ManualOddsProvider()
    quotes = []
    if event:
        quotes.extend(manual.load_outrights(event_id=event.event_id))
        quotes.extend(manual.load_matchups(event_id=event.event_id))
        quotes.extend(manual.load_threeballs(event_id=event.event_id, round_no=round_no))
    raw_path = Path(manual_raw)
    if raw_path.exists():
        raw_quotes = manual.parse_threeball_text(
            raw_path.read_text(errors="replace"),
            event_id=event.event_id if event else "",
            round_no=round_no,
            book="manual_paste",
        )
        if raw_quotes:
            quotes.extend(raw_quotes)
            write_threeballs_csv(raw_quotes)
    if quotes:
        checks.extend(manual.qa_checks(quotes))
        with store.connect() as con:
            store.upsert_odds_quotes(con, [q.as_dict() for q in quotes])
    provider_rows["manual_odds"] = len(quotes)

    if odds_api_sport:
        odds_api = TheOddsApiGolfProvider()
        odds_quotes = odds_api.fetch_outrights(odds_api_sport, event_id=event.event_id if event else "")
        if odds_quotes:
            with store.connect() as con:
                store.upsert_odds_quotes(con, [q.as_dict() for q in odds_quotes])
        provider_rows["the_odds_api"] = len(odds_quotes)
    elif event:
        inferred = _infer_major_sport(event.name)
        if inferred:
            checks.append(qa.SourceCheck(
                "the_odds_api",
                True,
                "info",
                f"major outright sport key available: {inferred}; pass --odds-api-sport to fetch",
            ))

    if fit:
        from . import model

        params = model.fit(model.load_rounds_df())
        model.save_params(params)
        provider_rows["model_fit_rounds"] = params.get("fitted_rounds", 0)

    summary = qa.summarize(checks)
    manifest = {
        "event": event.as_store_row() if event else None,
        "database": str(db),
        "field_csv": str(store.FIELD_CSV),
        "stats_csv": str(stats_written) if stats_written else "",
        "provider_rows": provider_rows,
        "weather": weather_summary,
        "qa": summary,
        "source_priority": [
            "local verified cache",
            "ESPN/golfastR-style event data",
            "PGA Tour public stats pages",
            "Open-Meteo weather",
            "The Odds API major outrights",
            "manual odds boards",
        ],
    }
    path = store.write_manifest(manifest)
    manifest["manifest_path"] = str(path)
    return manifest


def _print_summary(manifest: dict, path: Path) -> None:
    event = manifest.get("event") or {}
    print("Free-source golf refresh")
    if event:
        print(f"  Event: {event.get('name')} ({event.get('event_id')})")
    print(f"  DB: {manifest['database']}")
    for key, rows in manifest["provider_rows"].items():
        print(f"  {key}: {rows}")
    warnings = manifest["qa"].get("warnings") or []
    errors = manifest["qa"].get("errors") or []
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w['source']}: {w['message']}")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e['source']}: {e['message']}")
    print(f"\nManifest -> {path}")


def _infer_major_sport(event_name: str) -> str:
    n = str(event_name or "").lower()
    if "masters" in n:
        return MAJOR_SPORT_KEYS["masters"]
    if "pga championship" in n:
        return MAJOR_SPORT_KEYS["pga_championship"]
    if "u.s. open" in n or "us open" in n:
        return MAJOR_SPORT_KEYS["us_open"]
    if "open championship" in n or n.startswith("the open"):
        return MAJOR_SPORT_KEYS["the_open"]
    return ""


def _course_key(name: str) -> str:
    return " ".join(str(name or "").lower().split())


if __name__ == "__main__":
    main()
