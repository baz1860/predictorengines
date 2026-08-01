#!/usr/bin/env bash
# cfb/update.sh — weekly refresh: pull latest games/upcoming schedule → refit
# power ratings → walk-forward validate (gate). Network refreshes may fall back
# to validated cached inputs, but fit/gate failures are fatal and propagate a
# non-zero exit code. The card / edge run on demand:
#   python3 -m cfb.season [--odds-api]
#
# Usage: bash cfb/update.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CURRENT_STEP="start"
python3 -m cfb.run_status --status running --step "$CURRENT_STEP"

on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    python3 -m cfb.run_status --status failure --step "$CURRENT_STEP" \
      --message "required update step exited ${rc}" || true
  fi
  exit "$rc"
}
trap on_exit EXIT

optional_step() {
  local label="$1"
  shift
  if ! "$@"; then
    echo "  WARNING: ${label} failed; retaining validated cached input"
    return 0
  fi
}

echo "════════════════════════════════════════════"
echo "  CFB engine update  $(date '+%Y-%m-%d %H:%M')"
echo "════════════════════════════════════════════"

echo ""; echo "── 1/4 Refresh games.csv + upcoming.csv ──"
CURRENT_STEP="refresh_games"
optional_step "games/schedule refresh" python3 -m cfb.fetch_data

echo ""; echo "── 2/4 Refresh CFBD roster inputs (talent / returning production) ──"
CURRENT_STEP="refresh_priors"
optional_step "CFBD roster refresh" python3 -m cfb.fetch_cfbd

# Cached core inputs must still be usable after any optional network failure.
CURRENT_STEP="diagnostic_preflight"
python3 preflight.py --engine cfb --require-diagnostic

echo ""; echo "── 3/4 Refit power ratings ──"
CURRENT_STEP="fit_power"
python3 -m cfb.power --fit

echo ""; echo "── 4/4 Validation gate ──"
CURRENT_STEP="validation_gate"
python3 -m cfb.validate --gate --quiet

# Record data provenance (offline, never blocks).
optional_step "provenance manifest" python3 -m app.provenance --engine cfb --write

echo ""; echo "── Betting readiness ──"
if python3 preflight.py --engine cfb --require-ready; then
  echo "  betting-ready inputs present"
else
  echo "  diagnostic update complete; betting remains disabled until readiness passes"
fi

CURRENT_STEP="complete"
python3 -m cfb.run_status --status success --step "$CURRENT_STEP"
trap - EXIT

echo ""; echo "Update complete. Weekly card:"
echo "  python3 -m cfb.season --odds-api    # picks ATS + value, -> cfb/data/card.md"
