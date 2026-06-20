#!/usr/bin/env bash
# tennis/update.sh — daily refresh: accumulate results → refit ATP + WTA →
# walk-forward validate (gate) → recalibrate. Offline-safe: the accumulate step
# degrades to cached CSVs and the pipeline still finishes. Sim / edge run on
# demand from the app (they need a loaded draw / odds).
#
# Usage: bash tennis/update.sh [--since 2023-01-01]
set -uo pipefail
# Run from the repo root so the tennis package resolves (package-relative
# imports; invoke modules with `python3 -m tennis.X`).
cd "$(dirname "$0")/.."

SINCE="2023-01-01"
while [[ $# -gt 0 ]]; do
  case $1 in
    --since) SINCE="$2"; shift 2 ;;
    *) shift ;;
  esac
done

echo "════════════════════════════════════════════"
echo "  Tennis engine update  $(date '+%Y-%m-%d %H:%M')"
echo "════════════════════════════════════════════"

echo ""; echo "── 1/5 Accumulate latest matches → matches.csv ──"
python3 -m tennis.fetch --accumulate || echo "  accumulate skipped (offline)"

echo ""; echo "── 2/5 Refit ATP model ──"
python3 -m tennis.model --fit --tour atp --top 10 || echo "  ATP fit skipped"

echo ""; echo "── 3/5 Refit WTA model ──"
python3 -m tennis.model --fit --tour wta --top 10 || echo "  WTA fit skipped"

echo ""; echo "── 4/5 Walk-forward validate (gate) ──"
python3 -m tennis.validate --since "$SINCE" --gate --quiet \
  || echo "  validation gate warning (model may have regressed)"

echo ""; echo "── 5/5 Refit calibration ──"
python3 -m tennis.calibrate --fit || echo "  calibration skipped"

# Record data provenance (offline, never blocks).
python3 -m app.provenance --engine tennis --write || echo "  manifest skipped"

echo ""; echo "Done. Load a draw/odds then price on demand:"
echo "  python3 -m tennis.fetch --draw-template   # then fill tennis/data/draw.csv"
echo "  python3 -m tennis.fetch --odds-template   # then fill tennis/data/odds.csv"
echo "  python3 -c \"from tennis.engine import cmd_simulate; print(cmd_simulate({'tour':'atp','sims':50000})['note'])\""
