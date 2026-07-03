# PR: golf — auto-detect current event + fix live in-play conditioning

**Commit:** `1d77e0f` (tested: 31/31 golf tests pass)
**Suggested branch:** `feat/golf-live-inplay-autodetect`
**Base branch:** `club-soccer-engine-upgrade`
> The work builds on golf code (tee_times, edge, model, refresh scaffolding…) that
> currently lives only on `club-soccer-engine-upgrade`, not on `main`, so the PR
> must target that branch (or be merged after it lands on main).

I could not push or open the PR from this session — the workspace has no GitHub
credentials and no `gh` CLI. The commit is exported two ways below; apply either,
then push and open the PR from your machine.

## Option A — apply the patch
```bash
cd /path/to/predictorengines
git checkout club-soccer-engine-upgrade
git checkout -b feat/golf-live-inplay-autodetect
git am "golf-live-inplay-autodetect.patch"
git push -u origin feat/golf-live-inplay-autodetect
```

## Option B — pull from the bundle
```bash
cd /path/to/predictorengines
git checkout -b feat/golf-live-inplay-autodetect club-soccer-engine-upgrade
git pull "golf-live-inplay-autodetect.bundle" feat/golf-live-inplay-autodetect
git push -u origin feat/golf-live-inplay-autodetect
```

Then open the PR:
https://github.com/baz1860/predictorengines/pull/new/feat/golf-live-inplay-autodetect

## PR title
`golf: auto-detect current event + fix live in-play conditioning`

## PR description
The golf engine was always emitting the pre-tournament projection because the live
in-play path was silently broken and never wired into the season front door. This
makes `python -m golf.season` ascertain the current PGA event and its completed
rounds on its own and condition predictions accordingly.

- **providers/espn.py** — define `_to_par` / `_is_out` (were undefined, crashing
  `completed_round_scores` on every refresh — the root cause of the fallback to
  pre-tournament). Add an event resolver so a name / past event id maps to its
  dated scoreboard (ESPN ignores the `event` query param).
- **refresh.py** — mount-safe clearing of stale live artefacts.
- **season.py** — defer to the engine's native auto-route (`cmd_simulate` /
  `cmd_edge` condition on `live_state.json`); mirror the in-play board into the
  canonical `predictions.csv`; persist the full edge board (outrights + in-play
  matchups/3-balls) to `edge_report.csv` (was never written, so the card read a
  stale file). Removed the redundant parallel in-play path.
- **engine.py** — relax the round-group guard to drop only groups naming an
  unmatched player instead of the whole board (was refusing over one bookmaker
  spelling diff); add a thin-sample staking guard so unmodelled players
  (default-skill fallback) are never staked.
- **round_pricer.py** — never stake thin-sample players (kept for context, £0).
- **simulate_inplay.py** — vectorise the ranking loop (~20x faster; the full run
  no longer times out).

**Tests:** 31 golf tests pass, including the 17 in-play tests that were red before.

**Files changed:** `golf/engine.py`, `golf/providers/espn.py`, `golf/refresh.py`,
`golf/round_pricer.py`, `golf/season.py`, `golf/simulate_inplay.py`.
