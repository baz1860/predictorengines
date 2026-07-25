#!/usr/bin/env bash
# golf/update.sh  –  v2 daily refresh: accumulate results → refit → validate
# → recalibrate → refresh current field/odds. Offline-safe: every networked
# step degrades to cached CSVs and the pipeline still finishes.
#
# Usage: bash update.sh [--course COURSE] [--major]
#   env: THE_ODDS_API_KEY (optional; majors only)

set -euo pipefail
# Run from the repo root so the golf package resolves (modules use package-relative
# imports since Phase 4; invoke them with `python3 -m golf.X`).
cd "$(dirname "$0")/.."

COURSE=""; MAJOR_FLAG=""
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
# Tours to keep current. PGA uses the keyed provider (DataGolf) when available;
# LIV + DP World Tour (eur) come from ESPN — the only free source that carries
# them — so their players stay fitted from real rounds instead of manual priors.
# Override with e.g. GOLF_TOURS=pga to revert to PGA-only.
GOLF_TOURS="${GOLF_TOURS:-pga,liv,eur}"
python3 -m golf.fetch --accumulate --tours "$GOLF_TOURS" || echo "  accumulate skipped (offline)"

echo ""; echo "── 2/5 Refresh current field + odds ──"
# refresh.py is the one authoritative current-event writer. The legacy fetch
# writer omitted event/course/cut/tee metadata and could silently strip field.csv.
python3 -m golf.refresh || {
  echo "  current-event refresh failed; refusing to price mixed/stale inputs" >&2
  exit 2
}

echo ""; echo "── 3/5 Walk-forward validate (blocking gate) ──"
python3 -m golf.integrity --rounds-only
if ! python3 -m golf.validate --since 2024-06-01 --sims 8000 --gate --quiet; then
  echo "  validation gate failed; model and calibration were not replaced" >&2
  exit 2
fi

echo ""; echo "── 4/5 Refit skill + variance model ──"
python3 -m golf.model --fit --top 10
python3 -m golf.integrity

echo ""; echo "── 5/5 Refit calibration ──"
python3 -m golf.calibrate --fit

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
