"""Command-line front door for the horse-racing engine."""
from __future__ import annotations

import argparse
import sys
from datetime import date

from .edge import price_race
from .model import artifact_path_for, fit, predict_race, save_artifact
from .schema import DataError, init_templates
from .validate import main as validate_main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m horse_racing")
    sub = ap.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create blank canonical CSV templates")
    init.add_argument("--data-dir")
    init.add_argument("--overwrite", action="store_true")
    fit_p = sub.add_parser("fit", help="fit the race-level probability model")
    fit_p.add_argument("--data-dir")
    fit_p.add_argument("--min-races", type=int, default=30)
    pred = sub.add_parser("predict", help="price one saved race")
    pred.add_argument("race_id")
    pred.add_argument("--data-dir")
    edge = sub.add_parser("edge", help="compare one race with cutoff win odds")
    edge.add_argument("race_id")
    edge.add_argument("--data-dir")
    edge.add_argument("--source")
    edge.add_argument("--min-edge", type=float, default=0.03)
    val = sub.add_parser("validate", help="chronological walk-forward validation")
    val.add_argument("--data-dir")
    val.add_argument("--min-train", type=int, default=30)
    val.add_argument("--test-size", type=int, default=15)
    val.add_argument("--half-life-scale", type=float, default=1.0)
    val.add_argument("--lockbox-frac", type=float, default=0.0)
    val.add_argument("--gate", action="store_true")
    val.add_argument("--write-baseline", action="store_true")
    exp = sub.add_parser("experiments",
                         help="run the V2 feature-family ladder with promotion gates")
    exp.add_argument("--data-dir", required=True)
    exp.add_argument("--min-train", type=int, default=30)
    exp.add_argument("--test-size", type=int, default=15)
    exp.add_argument("--lockbox-frac", type=float, default=0.0)
    exp.add_argument("--skip-hl-search", action="store_true")
    fetch = sub.add_parser("fetch", help="ingest The Racing API historical data")
    fetch.add_argument("--start", required=True, help="first date, YYYY-MM-DD")
    fetch.add_argument("--end", required=True, help="last date, YYYY-MM-DD")
    fetch.add_argument("--data-dir", required=True,
                       help="dedicated provider dataset directory")
    fetch.add_argument("--bookmaker", default="Betfair Exchange",
                       help="price-history source to retain")
    rp = sub.add_parser("fetch-rpscrape", help="scrape free retrospective RP results")
    rp.add_argument("--start", required=True, help="first date, YYYY-MM-DD")
    rp.add_argument("--end", required=True, help="last date, YYYY-MM-DD")
    rp.add_argument("--region", choices=("gb", "ire", "both"), default="both")
    rp.add_argument("--data-dir", required=True)
    rp.add_argument("--checkout", default=".providers/rpscrape")
    rp.add_argument("--clean", action="store_true", help="discard a resumable prior scrape")
    bf = sub.add_parser("ingest-betfair", help="join a Betfair Historical BASIC tar to races")
    bf.add_argument("--archive", required=True, help="downloaded Betfair data.tar")
    bf.add_argument("--data-dir", required=True, help="existing canonical race dataset")
    bf.add_argument("--cutoff-minutes", type=int, default=15)
    bf.add_argument("--match-tolerance-minutes", type=int, default=20)
    bf.add_argument("--max-component-staleness-seconds", type=int, default=3600,
                    help="quarantine boards containing an older component LTP")
    args = ap.parse_args(argv)
    try:
        if args.command == "init":
            paths = init_templates(args.data_dir, args.overwrite)
            print("\n".join(f"wrote {p}" for p in paths) if paths else "templates already exist")
        elif args.command == "fit":
            artifact = fit(data_dir=args.data_dir, min_races=args.min_races)
            path = save_artifact(artifact, artifact_path_for(args.data_dir))
            print(f"wrote {path} · {artifact['n_races']} races · "
                  f"temperature {artifact['temperature']:.3f}")
        elif args.command == "predict":
            print(predict_race(args.race_id, data_dir=args.data_dir).to_string(index=False))
        elif args.command == "edge":
            print(price_race(args.race_id, data_dir=args.data_dir, source=args.source,
                             min_edge=args.min_edge).to_string(index=False))
        elif args.command == "validate":
            forwarded = ["--min-train", str(args.min_train),
                         "--test-size", str(args.test_size),
                         "--half-life-scale", str(args.half_life_scale),
                         "--lockbox-frac", str(args.lockbox_frac)]
            if args.data_dir:
                forwarded += ["--data-dir", args.data_dir]
            if args.gate:
                forwarded.append("--gate")
            if args.write_baseline:
                forwarded.append("--write-baseline")
            return validate_main(forwarded)
        elif args.command == "experiments":
            from .experiments import main as experiments_main
            forwarded = ["--data-dir", args.data_dir,
                         "--min-train", str(args.min_train),
                         "--test-size", str(args.test_size),
                         "--lockbox-frac", str(args.lockbox_frac)]
            if args.skip_hl_search:
                forwarded.append("--skip-hl-search")
            return experiments_main(forwarded)
        elif args.command == "fetch":
            from .providers.the_racing_api import Client, ingest
            start = date.fromisoformat(args.start)
            end = date.fromisoformat(args.end)
            if end < start:
                raise ValueError("--end must not precede --start")
            manifest = ingest(Client(), start, end, args.data_dir, args.bookmaker)
            rows = manifest["rows"]
            print(f"ingested {rows['races']} races, {rows['runners']} runners, "
                  f"{rows['results']} results and {rows['odds']} odds changes")
            print(f"validation_grade={manifest['validation_grade']} · "
                  f"odds_executable={manifest['odds_executable']}")
        elif args.command == "fetch-rpscrape":
            from .providers.rpscrape import scrape_and_ingest
            start = date.fromisoformat(args.start)
            end = date.fromisoformat(args.end)
            if end < start:
                raise ValueError("--end must not precede --start")
            manifest = scrape_and_ingest(start, end, args.region, args.data_dir,
                                         args.checkout, args.clean)
            rows = manifest["rows"]
            print(f"ingested {rows['races']} races and {rows['runners']} runners · "
                  f"validation_grade={manifest['validation_grade']}")
        elif args.command == "ingest-betfair":
            from .providers.betfair_historical import ingest_basic_archive
            manifest = ingest_basic_archive(
                args.archive, args.data_dir, args.cutoff_minutes,
                args.match_tolerance_minutes, args.max_component_staleness_seconds)
            stats = manifest["stats"]
            completeness = "complete" if manifest["archive"]["complete"] else "TRUNCATED"
            print(f"matched {stats['matched_markets']} markets; wrote "
                  f"{stats['written_boards']} cutoff boards · archive={completeness}")
            print("odds_executable=false · BASIC has no available-size ladder")
    except (DataError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
