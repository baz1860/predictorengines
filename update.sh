#!/usr/bin/env bash
# Daily World Cup model update: refresh results, refit, re-simulate.
# Usage: ./update.sh [n_sims]   (default 20000)
set -euo pipefail
cd "$(dirname "$0")"
SIMS="${1:-20000}"

echo "== 1/4 Refreshing match data =="
TMP=$(mktemp -d)
git clone --quiet --depth 1 https://github.com/martj42/international_results "$TMP/intres"
cp "$TMP/intres/results.csv" data/results.csv
rm -rf "$TMP"
echo "   results.csv updated ($(wc -l < data/results.csv) rows)"

echo "== Settling open bets =="
python3 bankroll.py --settle

echo "== 2/4 Refitting Dixon-Coles =="
python3 dixoncoles.py --fit | tail -1

echo "== Refreshing squad availability ratings =="
python3 squads.py | tail -1 || echo "   squad ratings skipped"

echo "== 3/4 Match predictions =="
python3 predictor.py --worldcup | tail -1

echo "== 4/4 Tournament simulation ($SIMS runs) =="
python3 simulate.py -n "$SIMS"

echo "== Edge report (live odds via The Odds API, or filled odds.csv) =="
python3 edge.py || echo "   edge report skipped (no odds available)"

echo "== Refreshing Betting Tracker.xlsx =="
python3 refresh_tracker.py || true

echo "== CLV snapshot (closing-line value for open bets) =="
# Needs The Odds API; degrades gracefully offline. Never blocks the update.
python3 clv.py --snapshot || echo "   CLV snapshot skipped (no network / no open bets)"

echo "== Validation gate =="
# Warn loudly on regression but NEVER block the daily update (|| guard).
python3 validate.py --quiet --gate \
  || echo "   ##### VALIDATION GATE FAILED — blend Brier regressed vs baseline; review before betting #####"

echo "== Dashboard =="
python3 report.py || echo "   dashboard skipped"

echo "Done: predictions_worldcup_2026.csv, tournament_odds.csv, bet_queue.csv, dashboard.html refreshed."
