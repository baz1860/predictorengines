"""Command-line front door for the horse-racing engine."""
from __future__ import annotations

import argparse
import sys

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
    val.add_argument("--gate", action="store_true")
    val.add_argument("--write-baseline", action="store_true")
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
                         "--test-size", str(args.test_size)]
            if args.data_dir:
                forwarded += ["--data-dir", args.data_dir]
            if args.gate:
                forwarded.append("--gate")
            if args.write_baseline:
                forwarded.append("--write-baseline")
            return validate_main(forwarded)
    except (DataError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
