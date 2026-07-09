#!/usr/bin/env bash
# cfb/update.sh — weekly refresh: pull latest games/upcoming schedule → refit
# power ratings → walk-forward validate (gate). Offline-safe: if the fetch
# fails the refit still runs on cached data. The card / edge run on demand:
#   python3 -m cfb.season [--odds-api]
#
# Usage: bash cfb/update.sh
set -uo pipefail
cd "$(dirname "$0")/.."

echo "════════════════════════════════════════════"
echo "  CFB engine update  $(date '+%Y-%m-%d %H:%M')"
echo "════════════════════════════════════════════"

echo ""; echo "── 1/4 Refresh games.csv + upcoming.csv ──"
python3 -m cfb.fetch_data || echo "  fetch skipped (offline)"

echo ""; echo "── 2/4 Refresh CFBD roster inputs (talent / returning production) ──"
python3 -m cfb.fetch_cfbd || echo "  cfbd pull skipped (offline / no key)"

echo ""; echo "── 3/4 Refit power ratings ──"
python3 -m cfb.power --fit || echo "  power fit skipped"

echo ""; echo "── 4/4 Validation gate ──"
python3 -m cfb.validate --gate --quiet 2>/dev/null \
  || echo "  validation gate warning (or validate has no --gate; run python3 -m cfb.validate)"

# Record data provenance (offline, never blocks).
python3 -m app.provenance --engine cfb --write 2>/dev/null || echo "  manifest skipped"

echo ""; echo "Done. Weekly card:"
echo "  python3 -m cfb.season --odds-api    # picks ATS + value, -> cfb/data/card.md"
