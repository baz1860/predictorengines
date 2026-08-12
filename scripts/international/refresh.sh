#!/usr/bin/env bash
# International module refresh. Designed for cron.
#
# Cadence rationale (plan §9): international football happens in ~6 windows a year,
# roughly 60 active days. Polling hard for 365 days wastes provider allowance on
# nothing; polling only during windows misses fixture announcements, which arrive
# weeks ahead. So fixtures refresh daily and cheaply, while odds polling is
# self-throttling — fetch_odds.py only touches fixtures within 14 days of kickoff.
#
# Suggested crontab:
#   0  6 * * *  /path/to/scripts/international/refresh.sh fixtures
#   0 * * * *   /path/to/scripts/international/refresh.sh odds     # hourly is safe
#   0  7 * * 1  /path/to/scripts/international/refresh.sh weekly
#   0  5 1 * *  /path/to/scripts/international/refresh.sh venues   # monthly is ample
#
# Exit codes: non-zero if the data gate fails. Blocking is the point — an earlier
# version of update.sh printed gate failures and carried on, and a betting card was
# produced from a table nobody had checked.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:-fixtures}"
LOG_DIR="data/international/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

gate() {
  log "data gate"
  if ! python3 -m international.gate --quiet; then
    log "GATE FAILED — see errors above. Refresh aborted."
    return 1
  fi
}

case "$MODE" in
  fixtures)
    log "refreshing venues (cheap; new venues appear with new fixtures)"
    python3 -m scripts.international.fetch_fixtures --venues || log "venue refresh skipped"
    log "refreshing fixtures"
    python3 -m scripts.international.fetch_fixtures --write --caf
    gate
    ;;

  odds)
    # Self-throttling by cadence: each fixture has a minimum interval keyed to how
    # far off kickoff is (48h at discovery range, 30 min inside the last 6 hours),
    # so this can safely run every hour. Measured BSD behaviour: prices appear
    # ~60h before kickoff and are universal inside 24h.
    log "polling odds (cadence-based)"
    python3 -m scripts.international.fetch_odds --write --max-requests 250
    ;;

  venues)
    log "rebuilding home-venue profiles (geocode + elevation, cached)"
    python3 -m scripts.international.build_home_venues --geocode --build
    ;;

  weekly)
    log "full refresh + provider comparison + coverage"
    python3 -m scripts.international.fetch_fixtures --venues
    python3 -m scripts.international.fetch_fixtures --write --caf
    python3 -m scripts.international.compare_providers \
      --json data/international/provider_comparison.json || log "comparison skipped"
    python3 -m international.coverage | head -5
    python3 -m scripts.international.fetch_odds --report
    gate
    log "strict gate (report only — pending adjudications are expected)"
    python3 -m international.gate --strict || true
    ;;

  *)
    echo "usage: $0 [fixtures|odds|venues|weekly]" >&2
    exit 2
    ;;
esac

log "done ($MODE)"
