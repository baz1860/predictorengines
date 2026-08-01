#!/usr/bin/env bash
# Scoped commit plan for the Soccer Prediction repo.
#
# Portable (works on macOS bash 3.2 — no `mapfile`) and idempotent: each commit
# only fires if something is actually staged, so it is safe to re-run. If you
# already ran commit 1, re-running this just completes commits 2 and 3.
#
# Run from the repo root:  bash commit_plan.sh
set -eu
cd "$(dirname "$0")"

# A stale lock can block git; safe to clear (no git process is running).
rm -f .git/index.lock 2>/dev/null || true

# Commit only if there is something staged; never fail on an empty commit.
commit_if_staged() {
  if git diff --cached --quiet; then
    echo "   (nothing to stage — skipping)"
  else
    echo "--- staged ($(git diff --cached --name-only | wc -l | tr -d ' ') files) ---"
    git diff --cached --name-only
    git commit -m "$1"
  fi
}

echo "############################################################"
echo "# COMMIT 1 (primary) — club_soccer subsystem + adversarial fixes"
echo "############################################################"

git add club_soccer/*.py club_soccer/*.md club_soccer/*.sh
git add scripts/club_soccer conftest.py app/engines/club_soccer.py .gitignore
git add deploy/com.barrie.sportspredictor.clubsoccer.capture.plist
git status --porcelain | grep '^??' | awk '{print $2}' \
  | grep '^test_.*\.py$' | grep -v '^test_blend_gate.py$' \
  | while read -r f; do git add "$f"; done
git add test_v5.py test_club_soccer.py test_m3.py test_m4.py
git add \
  club_soccer/data/club_alias_map.json \
  club_soccer/data/club_registry.json \
  club_soccer/data/uefa_coefficients.json \
  club_soccer/data/uefa_coefficients_history.json \
  club_soccer/data/league_seed_evidence.json \
  club_soccer/data/cross_league_baseline.json \
  club_soccer/data/cross_league_baseline_p0_preidentity.json \
  club_soccer/data/cross_league_p3.json \
  club_soccer/data/promotion_baseline.json \
  club_soccer/data/identity_verdicts.json \
  club_soccer/data/identity_review.csv \
  club_soccer/data/variance_inflation_evidence.json \
  club_soccer/data/xg_gating_evidence.json 2>/dev/null || true

commit_if_staged "club_soccer: decision-ledger evidence system + adversarial-review fixes

Brings the club_soccer decision-ledger / evidence-gate / identity / UEFA
league-expansion subsystem under version control (it had accumulated
uncommitted), together with this session's adversarial-review remediations:

Blockers
- settle on stable match identity, not the dedup-discarded fixture_id
- fix the 1X2 log-loss key so market log-loss is no longer always None
- benchmark edge against a selection-independent consensus, not the
  best-priced book's own de-vig
- Wilson bound on the CLV-scored count with coverage + per-league gating

Findings 5-15
- registry same-club confirmation requires equal canonical identity
- require kelly_roi_lb95 > 0 (block-bootstrap), closing the Kelly epsilon hole
- one version cohort per gate artifact; hash the decision/settlement ledgers
- settlement requires an official result; training excludes live/scheduled
- association country modelled separately from league membership (cross-border)
- association-name normalization (Turkiye/Turkey, Czechia/Czech Republic)
- point-in-time snapshot dating (no future-snapshot leak; anchor guard)
- mtime-aware walk-forward cache fingerprint
- frequent capture launch agent so the ledger can reach a season of volume
- centralized season derivation across all seed writers
- NaN->null, a 'no evidence' gate reason, and hermetic V5 tests"

echo
echo "############################################################"
echo "# COMMIT 2 (secondary) — other engines' in-progress code"
echo "############################################################"

git status --porcelain | awk '{print $2}' | grep -E '\.(py|toml)$' \
  | while read -r f; do git add "$f"; done

commit_if_staged "engines: snapshot in-progress work (golf, cfb, nfl, worldcup, tennis)

Non-club_soccer code changes that had accumulated uncommitted in the tree.
Data/params refresh is committed separately."

echo
echo "############################################################"
echo "# COMMIT 3 (tertiary) — refreshed generated data / fitted params"
echo "############################################################"

git status --porcelain | awk '{print $2}' | grep -E '\.(csv|json|jsonl)$' \
  | grep -vE '(^|/)(last_run\.json|season\.lock|decision_time_ledger\.csv|forward_predictions_club\.csv|forecast_ledger\.csv|forecast_settlements\.csv|forecast_performance\.json|validation_latest\.json|run_history\.jsonl)$' \
  | while read -r f; do git add "$f"; done

commit_if_staged "data: refresh fitted params and regenerated datasets

Tracked data/param updates (model params, coefficients, fixtures, market
history, per-engine datasets). Volatile operational state (ledgers, locks,
run status, generated cards) is intentionally left untracked."

echo
echo "############################################################"
echo "# Left UNCOMMITTED on purpose (Syncthing / scratch):"
echo "#   ledgers, season.lock, last_run.json, bet_card.md,"
echo "#   forward_predictions_club.csv, forecast_ledger.csv,"
echo "#   forecast_settlements.csv, forecast_performance.json, validation_latest.json,"
echo "#   _score_run.txt, springbreak.zip, codex_*.md review prompts"
echo "############################################################"
git status --short
echo
echo "Done. Review 'git log --stat -3', then: git push origin main"
