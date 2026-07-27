# Tennis Feed Operations and Ranking Enrichment Plan

**Status:** Ready for implementation  
**Created:** 2026-07-25  
**Scope:** Free ATP results/rankings, unattended daily operation, health monitoring,
provenance, and safe card refresh  
**Current baseline:** ESPN ATP results are integrated and `tennis/update.sh` passes
end to end through 2026-07-25.

---

## 1. Objective

Turn the working ESPN ATP integration into an unattended, observable process that:

1. refreshes results every day without duplicate rows;
2. detects a dead, stale, malformed, or partially updated feed;
3. records enough state to diagnose a failed scheduled run;
4. improves new ATP rows with point-in-time ranking snapshots without look-ahead;
5. refreshes active draws and produces probability cards safely; and
6. never presents stale/manual odds as current betting opportunities.

This phase does **not** add a paid API. ESPN remains the current ATP results and
ranking source; TML remains historical ATP data already cached locally.

---

## 2. Non-negotiable invariants

- `tennis/data/matches.csv` remains the canonical completed-match store.
- A failed fetch must not truncate, replace, or partially rewrite a good store.
- The same update run can be repeated and add zero duplicate matches.
- Results ingestion and card generation remain separate failure domains.
  A card/draw failure must not invalidate a successful results/model refresh.
- Feed freshness is based on source timestamps and match dates, not only file
  modification times.
- Ranking data used for a match must satisfy:

  ```text
  ranking_effective_at <= match_date
  ```

- Missing ranking data remains `9999`; future ranking snapshots must never be
  applied retrospectively to historical validation rows.
- No bet is recommended unless its odds snapshot is explicitly current.
- All health gates fail closed for scheduled production runs and emit actionable
  state explaining the failing condition.

---

## 3. Target architecture

```text
ESPN rankings ──> rankings.py ──> rankings.csv ─────────┐
                                                       │ point-in-time join
ESPN scoreboard ─> ESPNResultsProvider ─> MatchRecord ─┼─> matches.csv
                                                       │
TML cache ───────> historical ATP rows ────────────────┘
                                      │
                                      v
                              health.py --gate
                                      │
                            fit → validate → calibrate
                                      │
                         core run state + provenance

ESPN active draws ─> season.py ─> draw.csv ─> probability cards
manual/current odds ────────────────────────> priced betting cards
```

The scheduled process is split into:

- **Core update:** rankings, results, health gate, model fit, validation,
  calibration, provenance.
- **Card refresh:** active draws and probability cards. This runs only after a
  successful core update and reports its status separately.
- **Monitor:** checks that the scheduled core run completed and that the source
  remains fresh.

---

## 4. Files to add or change

| File | Change |
|---|---|
| `tennis/rankings.py` | New ESPN rankings snapshot fetcher and point-in-time lookup. |
| `tennis/health.py` | New feed/data health report and fail-closed CLI gate. |
| `tennis/monitor.py` | New scheduled-run monitor and local notification hook. |
| `tennis/providers.py` | Preserve ESPN athlete/match IDs and enrich new records from eligible ranking snapshots. |
| `tennis/fetch.py` | Add `--rankings` and health-aware accumulation reporting. |
| `tennis/update.sh` | Add run-state recording, rankings refresh, and health gate. |
| `tennis/card_refresh.sh` | New separate draw/card refresh wrapper. |
| `app/provenance.py` | Register ranking snapshots, ESPN cache, feed health, and run state. |
| `deploy/*.plist` | Add tennis core, card, and monitor LaunchAgents. |
| `deploy/install_launchagents.sh` | Install, validate, bootstrap, and print tennis agents. |
| `tennis/README.md` | Document operations, status files, recovery, and manual odds rule. |
| `test_tennis_rankings.py` | Ranking parsing, snapshots, ID matching, and no-look-ahead tests. |
| `test_tennis_feed_health.py` | Fresh/stale/idle/malformed/cache-fallback health cases. |
| `test_tennis_scheduler.py` | Run-state, lock, exit propagation, and monitor tests. |

New generated data:

```text
tennis/data/rankings.csv
tennis/data/feed_health.json
tennis/data/last_update.json
tennis/data/last_card_refresh.json
```

Raw ESPN result/ranking payloads remain under `tennis/data/api_cache/`.

---

## 5. Phase 1 — Ranking snapshots without look-ahead

### 5.1 Source

Use ESPN's free, no-key ATP ranking JSON:

```text
https://site.api.espn.com/apis/site/v2/sports/tennis/atp/rankings
```

The endpoint currently returns:

- the top 150;
- ESPN athlete ID;
- display name;
- current and previous rank;
- points; and
- `rankings[0].update`, the effective source timestamp.

The effective timestamp—not fetch time—is the snapshot date used by joins and
freshness checks.

### 5.2 Snapshot schema

Create `tennis/data/rankings.csv`:

```text
effective_date,tour,player_id,player,rank,previous_rank,points,source,fetched_at
```

Uniqueness key:

```text
(effective_date, tour, player_id)
```

Append a snapshot only when its effective date/player rows are not already
stored. Re-fetching the same ESPN ranking release must be idempotent.

### 5.3 Match identity schema

Extend `MatchRecord` and `MATCH_COLUMNS` with optional trailing fields:

```text
winner_id,loser_id,source,source_match_id
```

Rules:

- ESPN rows populate athlete IDs, `source=espn`, and competition ID.
- Historical rows leave IDs blank and retain their existing values.
- Schema migration rewrites atomically and preserves all current rows.
- Existing model code continues ignoring the optional metadata columns.

### 5.4 Ranking join

Implement:

```python
ranking_for(player_id, player_name, match_date, tour) -> int
```

Lookup order:

1. exact `(tour, player_id)` match;
2. folded-name match only when it resolves to exactly one player ID;
3. otherwise `9999`.

Select the newest snapshot where `effective_date <= match_date`. Do not use
fetch time as ranking time.

The first snapshot can enrich matches on or after its effective date. Earlier
2026 ESPN rows remain unknown unless a legitimate historical ranking archive
is added later.

### 5.5 Update ordering

The daily core run performs:

```text
refresh ranking snapshot
→ accumulate completed results
→ point-in-time rank enrichment
→ health gate
```

If the ranking endpoint fails but a valid snapshot is available, results still
ingest with a warning. If no eligible snapshot exists, affected ranks remain
`9999`; ingestion must not fail merely because ranking coverage is incomplete.

### 5.6 Acceptance criteria

- Repeated ranking refresh adds no duplicate snapshot.
- Every assigned rank has `effective_date <= match_date`.
- ESPN athlete ID is preferred over names.
- Ambiguous folded names do not receive a rank.
- New ESPN rank coverage is reported overall and for main-draw R16-or-later
  matches.
- Validation explicitly tests that adding a future snapshot cannot change an
  earlier walk-forward row.

---

## 6. Phase 2 — Feed health and data-quality gate

### 6.1 Health output

`python3 -m tennis.health --write` creates:

```json
{
  "status": "healthy",
  "checked_at": "2026-07-25T08:30:00+01:00",
  "results": {
    "latest_match_date": "2026-07-25",
    "stored_rows": 11685,
    "duplicate_identities": 0
  },
  "espn": {
    "cache_fetched_at": "...",
    "cache_age_hours": 1.2,
    "events": 4,
    "completed_matches": 2627,
    "active_events": 4
  },
  "rankings": {
    "effective_date": "2026-07-16",
    "age_days": 9,
    "players": 150
  },
  "checks": [],
  "warnings": [],
  "failures": []
}
```

Status values:

```text
healthy | warning | failed | idle
```

`idle` means no ATP event is active or due to produce results; it is not a
failure.

### 6.2 Required checks

1. ESPN cache parses and has an `events` list.
2. Cache `fetched_at` is present and not in the future.
3. Every stored match has a valid ISO date, tour, players, and surface.
4. No exact provider match key is duplicated.
5. No cross-source ESPN/TML overlap remains.
6. Latest stored ATP date does not move backwards.
7. Active-event state is derived from ESPN competition timestamps/statuses.
8. Ranking snapshot is parseable and its effective date is valid.
9. The update run wrote a terminal state: `succeeded` or `failed`.

### 6.3 Freshness policy

Initial thresholds:

| Condition | Warning | Failure |
|---|---:|---:|
| ESPN cache age while events are active | > 12 hours | > 26 hours |
| Latest ATP result age while events are active | > 2 days | > 3 days |
| ESPN ranking effective age | > 10 days | > 17 days |
| A run remains `running` | > 90 minutes | > 2 hours |

Thresholds are constants with tests. They should be adjusted only after at
least 30 days of observed scheduled-run timings.

Important exceptions:

- No active events: stale-result age becomes `idle`, not failed.
- Weather delay/no completed match: fresh cache plus active/in-progress schedule
  produces a warning before a failure.
- A temporary endpoint failure with a cache younger than 26 hours produces a
  warning and permits the model refresh.

### 6.4 CLI behavior

```bash
python3 -m tennis.health --write         # report only
python3 -m tennis.health --gate --write  # exit 2 on failed
```

Exit codes:

```text
0 = healthy or idle
1 = warning
2 = failed
```

`tennis/update.sh` treats exit 2 as fatal. A warning is recorded and printed but
does not stop a results/model refresh.

### 6.5 Acceptance criteria

- A malformed first-run response writes nothing to `matches.csv` and fails.
- A temporary network failure with a recent valid cache warns but continues.
- An old cache during an active event fails.
- An off-season/no-event day returns `idle`.
- Synthetic duplicate and backwards-date fixtures fail.
- Health JSON is written atomically even when the gate fails.

---

## 7. Phase 3 — Scheduled run state and locking

### 7.1 Run ledger

`tennis/data/last_update.json` records:

```json
{
  "run_id": "20260725T071500Z-12345",
  "state": "succeeded",
  "started_at": "...",
  "finished_at": "...",
  "stage": "complete",
  "exit_code": 0,
  "added": {"atp": 0, "wta": 12},
  "latest_dates": {"atp": "2026-07-25", "wta": "2026-07-25"},
  "validation_brier": 0.2295
}
```

The state file is updated atomically:

```text
running → succeeded
running → failed
```

The failure state includes the last stage and exit code. A process killed before
it can write `failed` remains `running`; the separate monitor detects that state
after the timeout.

### 7.2 Single-run lock

Use an atomic lock directory:

```text
tennis/data/.update.lock/
```

The wrapper stores PID and start time inside it. A second invocation exits
without running. A lock is stale only when:

- its PID is not alive; and
- its recorded age exceeds the maximum run duration.

Do not delete a live lock.

### 7.3 Core update stages

Refactor `tennis/update.sh` to preserve exact nonzero exits and record stages:

```text
1. start run ledger and acquire lock
2. refresh ATP ranking snapshot
3. accumulate ATP/WTA results
4. write provenance
5. run feed/data health gate
6. refit ATP model
7. refit WTA model
8. run walk-forward validation gate
9. refit calibration
10. write final provenance and succeeded state
11. release lock
```

No required stage uses `|| true` or otherwise swallows failure.

### 7.4 Acceptance criteria

- Every exit path writes a terminal state unless the process is forcibly killed.
- Concurrent invocation does not start a second fit.
- A forced stage failure records its stage and original exit code.
- SIGTERM releases the lock and records failure.
- SIGKILL leaves a detectable stale `running` record/lock.

---

## 8. Phase 4 — macOS LaunchAgent automation

Use the existing repository convention under `deploy/`.

Add:

```text
deploy/com.barrie.sportspredictor.tennis.update.plist
deploy/com.barrie.sportspredictor.tennis.card.plist
deploy/com.barrie.sportspredictor.tennis.monitor.plist
```

### 8.1 Proposed schedule

All times are local macOS time:

| Agent | Time | Purpose |
|---|---:|---|
| Tennis core update | 08:15 daily | Capture the prior global tennis day, then fit/validate. |
| Tennis card refresh | 08:50 daily | Refresh draws/cards after a successful core run. |
| Tennis monitor | 09:15 and 10:30 daily | Catch ordinary failure and killed/stuck runs. |

The second monitor pass is intentional: it detects a run that died shortly
after starting but had not exceeded the first timeout at 09:15.

### 8.2 LaunchAgent requirements

- Absolute repository and Python paths.
- Explicit `WorkingDirectory`.
- Explicit minimal `PATH`.
- `ProcessType=Background`.
- stdout/stderr under `~/Library/Logs/`.
- No secrets embedded in plist files.
- `plutil -lint` before bootstrap.
- `launchctl bootout` old definitions before `bootstrap`.

Extend `deploy/install_launchagents.sh` to install and print the tennis agents.
Installation remains an explicit user action; tests validate plist syntax but
do not load agents.

### 8.3 Monitor behavior

`python3 -m tennis.monitor` reads `last_update.json` and `feed_health.json`.

On failure:

1. print one actionable message to the monitor log;
2. exit nonzero;
3. attempt a local macOS notification with `/usr/bin/osascript`;
4. include the failing stage, last successful update, and log path.

Notification failure must not hide the monitor's own nonzero exit.

### 8.4 Acceptance criteria

- All plist files pass `plutil -lint`.
- A manual `launchctl kickstart` completes successfully.
- The update agent cannot overlap itself.
- Logs show stage boundaries and final state.
- A forced failed run is reported by the monitor.

---

## 9. Phase 5 — Draw and card refresh

### 9.1 Separate wrapper

Create `tennis/card_refresh.sh`:

```text
1. verify the latest core update state is succeeded and younger than 26 hours;
2. refresh ATP and WTA active draws with `tennis.season --tour both`;
3. write `last_card_refresh.json`;
4. preserve separate card/draw failure status.
```

The card wrapper does not rerun model fitting.

### 9.2 Odds safety

Free ESPN data supplies results, rankings, and draws—not reliable bookmaker
prices. Therefore:

- probability-only cards may always be generated from a fresh model/draw;
- a betting recommendation requires an odds row with a recorded capture time;
- stale or timestamp-less odds are displayed as informational only and receive
  no stake.

Extend `odds.csv` with an optional trailing field:

```text
captured_at
```

Initial policy:

```text
captured_at age <= 12 hours → may price/stake
captured_at missing or > 12 hours → probability only, no bet
```

Manual odds remain supported. Automated paid odds are outside this plan.

### 9.3 Acceptance criteria

- A failed core update prevents a card from claiming to be current.
- A draw failure does not alter the last good model/results.
- Missing/stale odds cannot produce a recommended stake.
- Current odds still flow through the existing de-vig, blend, and portfolio
  controls.

---

## 10. Provenance and operator visibility

Add these tennis manifest entries:

| Key | Role | Source |
|---|---|---|
| `results_atp` | results | ESPN current ATP + TML history |
| `espn_atp_cache` | source cache | ESPN scoreboard |
| `rankings_atp` | rankings | ESPN ATP rankings |
| `feed_health` | health | `tennis.health` |
| `last_update` | run state | `tennis/update.sh` |
| `draw` | fixtures | ESPN scoreboard |
| `last_card_refresh` | run state | `tennis/card_refresh.sh` |

The app schema should surface:

```text
latest ATP result date
ranking effective date
last successful core update
last successful card refresh
health status and warnings
```

Do not show a generic green status derived solely from file mtime.

---

## 11. Test strategy

### Unit tests

- ESPN ranking payload normalization.
- Snapshot idempotency.
- Athlete-ID and folded-name matching.
- Ambiguous-name rejection.
- Strict no-look-ahead ranking selection.
- Health state transitions and threshold boundaries.
- Atomic state writes.
- Same-source rematches preserved; cross-source overlap removed.
- Stale odds cannot produce stakes.

### Integration tests

Use local fixtures and redirected temporary data paths:

```bash
python3 -m pytest -q \
  test_tennis_espn_provider.py \
  test_tennis_rankings.py \
  test_tennis_feed_health.py \
  test_tennis_scheduler.py \
  test_tennis_adversarial_fixes.py \
  test_tennis_contract.py
```

Required live smoke checks:

```bash
python3 -m tennis.rankings --refresh --tour atp
python3 -m tennis.fetch --accumulate --tours atp
python3 -m tennis.health --gate --write
bash tennis/update.sh
bash tennis/card_refresh.sh
```

Run accumulation twice and require the second run to add zero rows.

### Model regression check

Ranking enrichment must not bypass `tennis.validate --gate`. Record:

- headline Brier;
- Elo control delta;
- rank-logistic control delta;
- calibration OOS Brier;
- rank coverage overall and by round.

The phase is accepted only if the existing gate passes. If ranking enrichment
does not improve or preserve validation, retain snapshots for provenance but do
not wire them into the model.

---

## 12. Rollout order

1. Add ranking snapshots and tests.
2. Add optional match metadata columns and point-in-time enrichment.
3. Add health report in report-only mode.
4. Run report-only health for seven daily updates and inspect false warnings.
5. Enable the fail-closed health gate.
6. Add run ledger and locking.
7. Add and manually test LaunchAgents.
8. Run the scheduler in shadow mode for seven days.
9. Enable local failure notifications.
10. Add the separate card refresh and odds-timestamp gate.

Do not combine all phases into one unobserved scheduler rollout.

---

## 13. Rollback and recovery

Before schema migration:

```text
copy matches.csv to a timestamped local backup
validate row count and checksum
```

Recovery rules:

- Ranking failure: keep ingesting results with `9999`; do not fabricate ranks.
- ESPN failure with fresh cache: warn and use cache.
- ESPN failure with stale/no cache: stop before fitting and preserve last good
  models.
- Validation failure: retain newly ingested raw data, but do not mark the core
  run successful or refresh production cards.
- Scheduler problem: `launchctl bootout` the tennis agents and run
  `bash tennis/update.sh` manually.
- Bad ESPN batch: filter using `source=espn` and `source_match_id`, restore the
  pre-migration backup if required, then rerun validation.

No recovery step deletes the whole data directory or raw source cache.

---

## 14. Definition of done

This plan is complete when:

- the daily core run executes unattended for seven consecutive days;
- every run has a terminal state and health report;
- stale/dead/malformed feed fixtures fail closed;
- temporary network failure uses a recent cache safely;
- ranking assignments are point-in-time correct and ID-backed;
- repeat accumulation is idempotent;
- validation and calibration gates pass;
- the monitor detects a forced failed and a forced killed run;
- active draws refresh separately from the core model update;
- stale/manual odds cannot create a recommended bet; and
- the README contains exact run, inspect, failure, and recovery commands.

