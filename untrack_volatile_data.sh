#!/usr/bin/env bash
# Syncthing migration: stop git from tracking the volatile club_soccer working
# data, so it is replicated peer-to-peer instead and never re-committed.
#
# Run this AFTER commit_plan.sh has finished (so the data churn is already
# committed and the tree is clean). It appends the ignore rules, drops the
# files from the index with `git rm --cached` (they STAY on disk), and commits
# the untracking. Idempotent and portable (bash 3.2 / zsh). Run from repo root:
#   bash untrack_volatile_data.sh
set -eu
cd "$(dirname "$0")"
rm -f .git/index.lock 2>/dev/null || true

MARKER="# >>> Syncthing-owned club_soccer working data >>>"

# 1) Append the ignore block once.
if grep -qF "$MARKER" .gitignore 2>/dev/null; then
  echo ".gitignore already has the block — skipping append"
else
  cat >> .gitignore <<'EOF'

# >>> Syncthing-owned club_soccer working data >>>
# These change on essentially every pipeline run, or are written continuously
# by the capture agent, so they are replicated via Syncthing (mini -> laptop),
# NOT versioned in git — that keeps history clean and stops a stale committed
# copy overwriting fresh state on another machine. Reference/evidence data
# (club_alias_map, club_registry, uefa_coefficients*, *_evidence.json, the
# cross-league/promotion baselines, identity_verdicts) stays tracked.
club_soccer/data/decision_ledger.csv
club_soccer/data/settlement_ledger.csv
club_soccer/data/decision_time_ledger.csv
club_soccer/data/backtest_market.json
club_soccer/data/fixtures.csv
club_soccer/data/market_history.csv
club_soccer/data/squads_club.csv
club_soccer/data/model_params.json
club_soccer/data/comp_strength.json
club_soccer/data/ensemble_weights.json
club_soccer/data/last_run.json
club_soccer/data/season.lock
club_soccer/data/bet_card.md
club_soccer/data/forward_predictions_club.csv
club_soccer/data/validation_latest.json
# <<< Syncthing-owned club_soccer working data <<<
EOF
  echo "appended ignore block to .gitignore"
fi

# 2) Drop them from the index (files remain on disk). --ignore-unmatch: fine if
#    a file was never tracked (some are already untracked).
for f in \
  club_soccer/data/decision_ledger.csv \
  club_soccer/data/settlement_ledger.csv \
  club_soccer/data/decision_time_ledger.csv \
  club_soccer/data/backtest_market.json \
  club_soccer/data/fixtures.csv \
  club_soccer/data/market_history.csv \
  club_soccer/data/squads_club.csv \
  club_soccer/data/model_params.json \
  club_soccer/data/comp_strength.json \
  club_soccer/data/ensemble_weights.json \
  club_soccer/data/last_run.json \
  club_soccer/data/season.lock \
  club_soccer/data/bet_card.md \
  club_soccer/data/forward_predictions_club.csv \
  club_soccer/data/validation_latest.json ; do
  git rm --cached --quiet --ignore-unmatch "$f" || true
done

git add .gitignore

if git diff --cached --quiet; then
  echo "nothing to untrack — already migrated"
else
  echo "--- staged ($(git diff --cached --name-only | wc -l | tr -d ' ') files) ---"
  git diff --cached --name-only
  git commit -m "chore: untrack volatile club_soccer working data (Syncthing-owned)

The ledgers, fitted params, fixtures/market history, gate artifact and per-run
operational files change every run and are now replicated via Syncthing, not
git. Reference/evidence data stays tracked. Files remain on disk; only their
index entries are removed."
fi
echo "Done. The files stay on disk; git just no longer tracks them."
echo "Review: git show --stat HEAD    then: git push origin main"
