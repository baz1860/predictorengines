"""Free-source weekly refresh for the PGA golf engine.

This command is intentionally conservative: it gathers and caches free data,
exports the existing CSV contract, and reports provider QA warnings. It does not
force a bet or hide missing market data.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

from . import provider_qa as qa
from . import store
from .providers.bovada import BovadaGolfProvider, export_csvs as bovada_export_csvs
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
    bovada: bool = True,
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

    # Live in-tournament scores → between-rounds snapshot the engine auto-routes
    # to once a round is complete. Best-effort: a parser/network failure degrades
    # to a QA warning and leaves the engine on its pre-tournament projection.
    rounds_done = 0
    if event:
        try:
            rounds_done = _write_live_scores(espn, event, use_cache=use_cache)
            provider_rows["live_scores_round"] = rounds_done
            checks.append(qa.SourceCheck(
                "espn.live_scores", True, "info",
                (f"between-rounds snapshot after round {rounds_done}"
                 if rounds_done else "pre-tournament — no completed rounds yet"),
                rounds_done))
        except Exception as exc:  # noqa: BLE001 — never let scoring break refresh
            checks.append(qa.SourceCheck("espn.live_scores", False, "warning", str(exc), 0))

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

    # Bovada: standard free, keyless source for this week's outright / matchup /
    # 2-3-ball boards. Writes the odds.csv / matchups.csv / threeballs.csv
    # contract that the pricer and the manual loaders below read. Best-effort:
    # the endpoint is unofficial and geo-sensitive, so any failure degrades to a
    # QA warning with the previous CSVs (or a manual paste) left intact.
    bovada_rows = 0
    if bovada and event:
        try:
            provider = BovadaGolfProvider()
            coupon = provider.fetch_coupon(use_cache=use_cache)
            b_quotes = provider.event_quotes(
                coupon, event.name, event_id=event.event_id, round_no=round_no)
            if b_quotes:
                written = bovada_export_csvs(b_quotes)
                checks.extend(provider.qa_checks(b_quotes))
                with store.connect() as con:
                    store.upsert_odds_quotes(con, [q.as_dict() for q in b_quotes])
                bovada_rows = len(b_quotes)
                checks.append(qa.SourceCheck(
                    "bovada", True, "info",
                    "wrote " + ", ".join(f"{k} ({v})" for k, v in written.items())
                    if written else "no priced markets exported", bovada_rows))
            else:
                checks.append(qa.SourceCheck(
                    "bovada", True, "warning",
                    f"no Bovada markets matched event '{event.name}'", 0))
        except Exception as exc:  # noqa: BLE001 — never let a flaky book break refresh
            checks.append(qa.SourceCheck("bovada", False, "warning", str(exc), 0))
    provider_rows["bovada"] = bovada_rows

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
    quotes = _dedupe_quotes(quotes)
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
        "rounds_done": rounds_done,
        "weather": weather_summary,
        "qa": summary,
        "source_priority": [
            "local verified cache",
            "ESPN/golfastR-style event data",
            "PGA Tour public stats pages",
            "Open-Meteo weather",
            "Bovada weekly outright/matchup/2-3-ball boards",
            "The Odds API major outrights",
            "manual odds boards (override)",
        ],
    }
    path = store.write_manifest(manifest)
    manifest["manifest_path"] = str(path)
    return manifest


LIVE_SCORES_CSV = DATA_DIR / "scores_live.csv"
LIVE_STATE_JSON = DATA_DIR / "live_state.json"
PREDICTIONS_INPLAY_CSV = DATA_DIR / "predictions_inplay.csv"


def _write_live_scores(espn, event, *, use_cache: bool = False) -> int:
    """Write the between-rounds scores snapshot and live_state.json.

    Returns the number of completed rounds (0 = pre-tournament). When no round
    is complete we clear any stale in-play artefacts so the engine falls back to
    its pre-tournament projection instead of last week's leaderboard.
    """
    rows, rounds_done = espn.completed_round_scores(event.event_id, use_cache=use_cache)

    if rounds_done < 1 or not rows:
        for stale in (LIVE_SCORES_CSV, LIVE_STATE_JSON, PREDICTIONS_INPLAY_CSV):
            stale.unlink(missing_ok=True)
        return 0

    with open(LIVE_SCORES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "score", "made_cut"])
        w.writeheader()
        for r in rows:
            w.writerow({"name": r["name"], "score": r["score"], "made_cut": r["made_cut"]})

    survivors = sum(1 for r in rows if r["made_cut"])
    LIVE_STATE_JSON.write_text(json.dumps({
        "event_id": event.event_id,
        "event_name": event.name,
        "rounds_done": rounds_done,
        "scores_csv": LIVE_SCORES_CSV.name,
        "survivors": survivors,
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    return rounds_done


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


def _dedupe_quotes(quotes: list) -> list:
    out = []
    seen = set()
    for q in quotes:
        d = q.as_dict() if hasattr(q, "as_dict") else dict(q)
        key = (
            d.get("event_id", ""),
            d.get("market", ""),
            d.get("round_no", ""),
            d.get("group_id", ""),
            d.get("player_name", ""),
            d.get("decimal_odds", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


if __name__ == "__main__":
    main()
