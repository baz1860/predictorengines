#!/usr/bin/env bash
# golf/update.sh  –  v2 daily refresh: accumulate results → refit → validate
# → recalibrate → refresh current field/odds. Offline-safe: every networked
# step degrades to cached CSVs and the pipeline still finishes.
#
# Usage: bash update.sh [--course COURSE] [--major]
#   env: DG_API_KEY, ODDS_API_KEY (optional)

set -uo pipefail
# Run from the repo root so the golf package resolves (modules use package-relative
# imports since Phase 4; invoke them with `python3 -m golf.X`).
cd "$(dirname "$0")/.."

COURSE=""; MAJOR_FLAG=""
DG_KEY="${DG_API_KEY:-}"; ODDS_KEY="${ODDS_API_KEY:-}"
while [[ $# -gt 0 ]]; do
  case $1 in
    --course) COURSE="$2"; shift 2 ;;
    --major)  MAJOR_FLAG="--major"; shift ;;
    *) shift ;;
  esac
done

echo "════════════════════════════════════════════"
echo "  Golf engine v2 update  $(date '+%Y-%m-%d %H:%M')"
echo "════════════════════════════════════════════"

echo ""; echo "── 1/5 Accumulate latest results → rounds.csv ──"
python3 -m golf.fetch --accumulate || echo "  accumulate skipped (offline)"

echo ""; echo "── 2/5 Refresh current field + odds ──"
FETCH_ARGS="--espn"
[ -n "$DG_KEY" ]   && FETCH_ARGS="$FETCH_ARGS --dg-key $DG_KEY"
[ -n "$ODDS_KEY" ] && FETCH_ARGS="$FETCH_ARGS --odds-key $ODDS_KEY"
python3 -m golf.fetch $FETCH_ARGS || echo "  field/odds refresh skipped (offline)"

echo ""; echo "── 3/5 Refit skill + variance model ──"
python3 -m golf.model --fit --top 10 || echo "  fit skipped"

echo ""; echo "── 4/5 Walk-forward validate (gate) ──"
python3 -m golf.validate --since 2024-06-01 --sims 8000 --gate --quiet \
  || echo "  validation gate warning (model may have regressed)"

echo ""; echo "── 5/5 Refit calibration ──"
python3 -m golf.calibrate --fit || echo "  calibration skipped"

# Record data provenance (offline, never blocks) — run from the repo root.
python3 -m app.provenance --engine golf --write || echo "  manifest skipped"

echo ""; echo "Done. Sim + edge run on demand from the app (they need live odds),"
echo "or standalone:  python3 -m golf.simulate --sims 50000 ${COURSE:+--course \"$COURSE\"} $MAJOR_FLAG"
echo "                python3 -m golf.edge --min-edge 1.0"
echo ""
echo "Round matchups: paste this week's tee groups into data/threeballs_r{N}_raw.txt"
echo "  (2 Ball headers for twosome/no-cut events, 3 Ball for full-field), then:"
echo "                python3 -m golf.season --round 1"
echo "  Names are checked against field.csv; a stale board is skipped, not priced."
