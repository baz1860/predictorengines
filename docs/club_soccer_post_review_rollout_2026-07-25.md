# Club Soccer post-review rollout plan

Date: 2026-07-25  
Scope: package the adversarial-review fixes cleanly, smoke-test the live
workflow, install reliable decision capture, and verify forward evidence begins
accumulating.

## Current state

- Club Soccer suite: **419 passed**.
- Repository check: **615 passed**.
- Fixed-window promotion gate: **PASS**, 27,181 matched evaluation rows.
- Offline data health: **PASS**, including both identity directions:
  one display name per club ID and one club ID per model display name.
- `main` is six commits ahead of `origin/main`; nothing has been pushed.
- Commit `79bd05d` already contains the final Club Soccer remediation, but it
  also contains unrelated NHL and Tennis changes. It is therefore not the
  focused commit originally intended.
- The worktree was clean before this plan was created. At the final check this
  plan and an unrelated concurrent `plans/tennis_feed_operations_plan.md` were
  untracked. The Tennis plan must not be staged with Club Soccer.

## Phase 1 — decide whether to split the mixed commit

This is the only history-rewriting step. Do not run it without explicit user
approval. The commit is currently local, so it can still be split safely after
creating a recovery branch.

### User intervention required

Choose one:

1. **Recommended:** split only `79bd05d` into a focused Club Soccer commit and
   leave the unrelated NHL/Tennis changes unstaged for their owners.
2. Keep `79bd05d` intact, commit this plan separately, and proceed directly to
   Phase 3.

Commands for option 2:

```bash
git add -- docs/club_soccer_post_review_rollout_2026-07-25.md
git commit -m "docs(club-soccer): add post-review rollout plan"
```

### Commands for the recommended split

```bash
cd "/Users/lucky/AI Models/Soccer Prediction"

# Confirm the expected starting point.
git status --short
git rev-parse HEAD
# Expected HEAD: 79bd05d863a27a2f2305a27902be077d98a61d3e

# Create a recoverable pointer before rewriting the local commit.
git branch codex/safety-before-club-soccer-split-2026-07-25 79bd05d

# Uncommit only 79bd05d. Files remain in the working tree.
git reset --mixed 7a72b8080373d42745344d3fe1117027c2bc4ccf

# Stage only the Club Soccer remediation and this rollout plan.
git add -- \
  docs/club_soccer_post_review_rollout_2026-07-25.md \
  club_soccer/README.md \
  club_soccer/club_identity.py \
  club_soccer/com.sportspredictor.clubsoccer.plist \
  club_soccer/data/club_alias_map.json \
  club_soccer/data/opponent_adjusted_xg_evidence.json \
  club_soccer/data/promotion_baseline.json \
  club_soccer/data/validation_baseline.json \
  club_soccer/experiments.json \
  club_soccer/fetch.py \
  club_soccer/health.py \
  club_soccer/seed_footballdata.py \
  club_soccer/seed_real.py \
  club_soccer/validate.py \
  club_soccer/walkforward_cache.py \
  tests/club_soccer/test_club_registry.py \
  tests/club_soccer/test_decision_ledger.py \
  tests/club_soccer/test_status_normalization.py \
  tests/club_soccer/test_xg_gating.py

# Verify no NHL, Tennis, Golf, or shared-app file entered the staged set.
git diff --cached --name-status
git diff --cached --check

python3 -m pytest tests/club_soccer -q
python3 -m club_soccer.validate --gate
python3 -m club_soccer.health --offline

git commit -m "fix(club-soccer): close adversarial review findings"
```

Expected after the focused commit:

- `git show --name-only --format= HEAD` lists only `club_soccer/`,
  `tests/club_soccer/`, and this plan.
- NHL/Tennis changes remain visible in `git status --short`; do not discard or
  commit them as part of Club Soccer.

Recovery if the split is wrong and no new work has been added:

```bash
git reset --hard codex/safety-before-club-soccer-split-2026-07-25
```

That recovery command is destructive to post-split working-tree edits. Inspect
`git status --short` first.

## Phase 2 — final static and regression checks

Run after either keeping or splitting the commit:

```bash
cd "/Users/lucky/AI Models/Soccer Prediction"

git diff --check
python3 -m compileall -q club_soccer app/engines/club_soccer.py
bash -n deploy/install_launchagents.sh
plutil -lint \
  deploy/com.barrie.sportspredictor.clubsoccer.season.plist \
  deploy/com.barrie.sportspredictor.clubsoccer.monitor.plist \
  deploy/com.barrie.sportspredictor.clubsoccer.capture.plist

python3 -m pytest tests/club_soccer -q
python3 run_checks.py
```

Acceptance:

- 419 Club Soccer tests pass.
- The aggregate repository check passes.
- All three plists lint successfully.
- `git diff --check` emits no output.

## Phase 3 — cached production smoke test

This exercises card generation without changing remote state or depending on a
provider response.

```bash
cd "/Users/lucky/AI Models/Soccer Prediction"

python3 -m club_soccer.season --no-network --fast
python3 -m club_soccer.health --offline
python3 -m club_soccer.validate --gate
python3 -m club_soccer.decision_ledger --status

sed -n '1,220p' club_soccer/data/card.md
```

Inspect the card for:

- plausible upcoming fixtures and probabilities;
- no live or void fixture presented as settled;
- stakes suppressed while the evidence gate is closed;
- no stale text referring to a global-AND market gate;
- no duplicate club identities.

Stop if the card is empty despite `upcoming_count > 0`, the promotion gate
fails, or health reports an identity/result conflict.

## Phase 4 — normal network refresh

This changes local source and generated artifacts and requires working provider
credentials. Run it only after the cached smoke test passes.

### User intervention required

If the BSD key is already configured through the project's normal key store:

```bash
cd "/Users/lucky/AI Models/Soccer Prediction"
python3 -m club_soccer.season
```

If a temporary environment variable is required, avoid placing the key in shell
history:

```bash
cd "/Users/lucky/AI Models/Soccer Prediction"
read -s "BSD_API_KEY?BSD API key: "
export BSD_API_KEY
echo
python3 -m club_soccer.season
unset BSD_API_KEY
```

Then verify:

```bash
python3 -m club_soccer.health --offline
python3 -m club_soccer.validate --gate
python3 -m club_soccer.decision_ledger --status
```

Acceptance:

- the refresh completes without invoking the fixture shrink guard;
- health remains green;
- the fixed-window promotion gate remains green;
- generated card dates and freshness fields reflect the refresh.

## Phase 5 — install and verify decision capture

The evidence ledger only grows reliably if the capture agent runs every 15
minutes. Installing LaunchAgents changes the user LaunchAgents directory and
the live `launchctl` state.

### User intervention required

```bash
cd "/Users/lucky/AI Models/Soccer Prediction"
bash deploy/install_launchagents.sh

launchctl print \
  "gui/$(id -u)/com.barrie.sportspredictor.clubsoccer.capture" | head -40
tail -n 100 "$HOME/Library/Logs/club_soccer_capture.log"
```

The installer also installs the season and monitor agents. Do not copy the
deleted legacy `club_soccer/com.sportspredictor.clubsoccer.plist`.

To stop only capture if it misbehaves:

```bash
launchctl bootout \
  "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.barrie.sportspredictor.clubsoccer.capture.plist"
```

## Phase 6 — verify forward evidence accumulation

Run after at least one fixture has entered the 60–120 minute pre-kickoff window
while odds were available:

```bash
cd "/Users/lucky/AI Models/Soccer Prediction"

python3 -m club_soccer.decision_ledger --status
tail -n 10 club_soccer/data/decision_ledger.csv
python3 -m club_soccer.decision_time_backtest --report
python3 -m club_soccer.evidence_gate
```

`club_soccer.evidence_gate` is expected to exit with status 1 while staking is
closed; its printed reasons, rather than that expected exit code, are what
should be inspected here.

Acceptance:

- new decisions carry one named executable bookmaker and quote;
- `decision_lead_min` is between 60 and 120;
- `resolver_version`, `model_hash`, and `code_hash` are populated;
- normal model refits do not remove older rows from the current strategy
  cohort;
- staking remains closed until the market and its individual league both meet
  every evidence threshold.

No decision rows after one day is not automatically a failure: there may have
been no fixture with complete odds inside the capture window. Repeated eligible
fixtures with no rows require investigation of the capture log and BSD odds
response.

## Phase 7 — publish

Pushing is an external state change. Do it only after the commit-splitting
decision, smoke test, and live capture verification are complete.

### User intervention required

Inspect the six local commits first:

```bash
cd "/Users/lucky/AI Models/Soccer Prediction"
git log --oneline origin/main..main
git status --short
```

If the complete local history is intended for `main`:

```bash
git push origin main
```

If any of the six commits are not ready, do not push `main`; create a review
branch or clean the local history first.

## Completion criteria

The rollout is complete when:

1. the Club Soccer remediation is intentionally packaged;
2. cached and live season runs complete cleanly;
3. health and promotion gates pass after the live refresh;
4. the capture LaunchAgent is loaded and its log is clean;
5. at least one eligible fixture produces immutable decision-ledger rows; and
6. the chosen commit history has been reviewed before any push.
