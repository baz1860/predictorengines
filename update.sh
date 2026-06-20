#!/usr/bin/env bash
# Daily World Cup model update: refresh results, refit, re-simulate.
# Usage:
#   ./update.sh [n_sims]      # default full daily flow, same behavior as before
#   ./update.sh morning       # results, live feeds, predictions, edge, manifest
#   ./update.sh prekickoff    # lineups/availability/odds, squad ratings, edge
#   ./update.sh postmatch     # results/stats, settlement, CLV, validation
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-default}"
if [[ "$MODE" == "morning" || "$MODE" == "prekickoff" || "$MODE" == "postmatch" ]]; then
  shift || true
else
  MODE="default"
fi
SIMS="${1:-20000}"

refresh_results() {
  echo "== Refreshing match data =="
  TMP=$(mktemp -d)
  git clone --quiet --depth 1 https://github.com/martj42/international_results "$TMP/intres"
  # Merge (not blind-copy): upstream is authoritative for played scores, but keep
  # local-only fixtures (future NA rows the predictor needs) and any manual scores
  # the feed hasn't published yet. See merge_results.py.
  python3 merge_results.py "$TMP/intres/results.csv" data/results.csv \
    || { echo "   merge failed — falling back to upstream copy"; cp "$TMP/intres/results.csv" data/results.csv; }
  rm -rf "$TMP"
  echo "   results.csv now $(wc -l < data/results.csv) rows"
}

refresh_live() {
  local mode="$1"
  echo "== Live World Cup data ($mode) =="
  python3 scripts/worldcup/live_data.py --mode "$mode" || echo "   live data skipped"
}

refit_models() {
  echo "== Refitting Dixon-Coles =="
  python3 -m engines.worldcup.dixoncoles --fit | tail -1
  echo "== Refreshing squad availability ratings =="
  python3 -m engines.worldcup.squads | tail -1 || echo "   squad ratings skipped"
}

write_predictions() {
  echo "== Match predictions =="
  python3 -m engines.worldcup.predictor --worldcup | tail -1
}

run_edge() {
  echo "== Edge report (live odds via The Odds API, or filled odds.csv) =="
  python3 -m engines.worldcup.edge || echo "   edge report skipped (no odds available)"
}

write_manifest() {
  echo "== Data manifest (provenance) =="
  python3 -m app.provenance --engine worldcup --write || echo "   manifest skipped"
}

run_dashboard_summary() {
  echo "== Dashboard =="
  python3 scripts/worldcup/report.py || echo "   dashboard skipped"
  echo "== Daily suite summary =="
  python3 daily_summary.py || echo "   summary skipped"
}

if [[ "$MODE" == "morning" ]]; then
  refresh_results
  refresh_live morning
  refit_models
  write_predictions
  run_edge
  write_manifest
  run_dashboard_summary
  echo "Done: morning World Cup refresh complete."
  exit 0
fi

if [[ "$MODE" == "prekickoff" ]]; then
  refresh_live prekickoff
  echo "== Refreshing squad availability ratings =="
  python3 -m engines.worldcup.squads | tail -1 || echo "   squad ratings skipped"
  run_edge
  write_manifest
  echo "Done: pre-kickoff World Cup refresh complete."
  exit 0
fi

if [[ "$MODE" == "postmatch" ]]; then
  refresh_results
  refresh_live postmatch
  echo "== Settling open bets =="
  python3 -m core.bankroll --settle
  echo "== CLV snapshot (closing-line value for open bets) =="
  python3 -m core.clv --snapshot || echo "   CLV snapshot skipped (no network / no open bets)"
  echo "== Validation gate =="
  python3 -m engines.worldcup.validate --quiet --gate \
    || echo "   ##### VALIDATION GATE FAILED — blend Brier regressed vs baseline; review before betting #####"
  write_manifest
  run_dashboard_summary
  echo "Done: post-match World Cup refresh complete."
  exit 0
fi

refresh_results

echo "== Settling open bets =="
python3 -m core.bankroll --settle

refit_models

write_predictions

echo "== 4/4 Tournament simulation ($SIMS runs) =="
python3 -m engines.worldcup.simulate -n "$SIMS"

run_edge

echo "== Refreshing Betting Tracker.xlsx =="
python3 refresh_tracker.py || true

echo "== CLV snapshot (closing-line value for open bets) =="
# Needs The Odds API; degrades gracefully offline. Never blocks the update.
python3 -m core.clv --snapshot || echo "   CLV snapshot skipped (no network / no open bets)"

echo "== Validation gate =="
# Warn loudly on regression but NEVER block the daily update (|| guard).
python3 -m engines.worldcup.validate --quiet --gate \
  || echo "   ##### VALIDATION GATE FAILED — blend Brier regressed vs baseline; review before betting #####"

write_manifest
run_dashboard_summary

echo "Done: predictions_worldcup_2026.csv, tournament_odds.csv, bet_queue.csv, dashboard.html refreshed."
