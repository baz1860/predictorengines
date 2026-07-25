# Codex Prompt: Round-6 Review — Verify Round-5 Items 1–6, Pre-flight the Approved Builds

Copy everything below the line into Codex.

---

Round 6. Your round-5 "implement now" items 1–6 have been executed; items 7–10 (the four approved builds) are deliberately NOT started — they are dedicated projects and your work plans for them will be executed next. Verify 1–6 adversarially, then pre-flight the build plans against the now-changed codebase.

Standing rules: harsh, `file:line` + code + impact + fix, do not modify files.

## Part 1 — Verify round-5 remediation

### 1. Pytest discovery restored (`pyproject.toml`: `testpaths = ["."]` with an honest comment; the historical bare-import collision no longer reproduces — 187 tests collect)
Attack: run the full flat suite. Expected state: everything passes EXCEPT `test_m3.py::test_fitted_w`, which is deliberately left red — the stored WC blend artifact shows `logloss_blend == logloss_market_only` (0.9705 = 0.9705, w=0.163), i.e. the World Cup market blend demonstrates no edge over the market. That is a true finding requiring a refit-or-demote decision in the WC engine, and weakening the test to hide it would recreate the false-green pattern. Confirm the failure is exactly this and propose the WC-engine fix (full-precision metrics in the artifact, and a demotion path if the blend genuinely ties). Note: `test_worldcup_live_bracket` PASSED in the fixing environment — your round-5 run had it failing on Germany–Paraguay mapping. Re-run it; if it fails for you again, the failure is data/state-dependent (possibly bracket state mutated by a prior season run) — diagnose which file drives the divergence.

### 2. Evidence schema round 3 (`_finite` + integer-valued `n_bets`; null/non-dict threshold rows, non-dict `thresholds`, non-dict `simulated_betting` all fail as reasons instead of crashing `evaluate()`; `generated_at_utc` must be tz-aware AND +00:00 — `+05:30` now rejected)
Attack with fresh artifacts: `n_bets` as `True` (bool is rejected by `_finite` — confirm), numeric strings, `Decimal`-style values, deeply nested nulls elsewhere, duplicate JSON keys. Confirm `evaluate()` can no longer raise on ANY malformed JSON structure (fuzz it). Provenance hashes remain absent by design until decision_time_v2 — confirm still documented.

### 3. Leak test round 2 (counter traps assert ZERO loader calls AND exactly n==12 folds; negative control retained; legacy `tune_ensemble --write` is now permanently report-only — the PROMOTE path cannot write `ensemble_weights.json` or move the baseline; comp-strength artifact asserted inactive with the per-fold plumbing gap documented)
Attack: could a partial leak still hide (loader called but not raising — e.g. a future refactor makes traps non-raising)? Is n==12 stable across pandas versions (deterministic rng, but grouping/monthly boundaries)? Verify no OTHER writer can touch `ensemble_weights.json` or `validation_baseline.json`. The calibration and comp-strength per-fold plumbing is still the PredictionConfig work in build 3 — confirm the interim inactive-assertions suffice.

### 4. Quote provenance round 2 (`FETCH_TIME_ONLY_MAX_AGE_MINUTES = 2` enforced per-row for live sources; provider-timestamped quotes keep 6h; kickoff backfill EXECUTED — 2,361 fixtures gained `kickoff_utc` from cached BSD events, zero >1-day date disagreements; future rows gain kickoff on the next networked fetch since `fetch.py` now writes it)
Attack: verify the backfill correctness (sample rows against cached events). Confirm the 2,582 future rows will actually be covered by the next networked run (the fetch window logic). BSD quotes are all `fetch_time_only` — with the 2-minute rule, a normal `season.py` run (fetch → price within seconds) still works, but a slow pipeline step between fetch and pricing could now legitimately expire quotes: check the gap between `fetch_bsd_odds` and `rows_from_odds` in `season.run` for any slow intervening step, and whether the expiry produces a clear pricing_note.

### 5. Central write boundary (`fetch.write_fixtures()` — status normalization + void clearing + atomic tmp/replace; ALL eight direct `to_csv(FIXTURES)` sites across `fetch.py`, `fetch_fdorg.py`, and the three seeders now route through it; AWD excluded from training via `TRAINING_EXCLUDED_STATUSES` while remaining an official result for settlement; openfootball January–June season assignment fixed)
Attack: any remaining writer (grep for `to_csv` against the fixtures path in every module including app/ and scripts/); does `write_fixtures`'s status truncation (`[:3]`) corrupt any legitimate status value in current data? Does the settlement path (`grade_open_bets`) definitely bypass `played()` so AWD results still settle? Existing openfootball-seeded rows retain the OLD wrong season values — quantify how many and whether a one-off season repair is needed.

### 6. Monitor round 2 (running-state marker written atomically before work with 2h dead-run deadline; future-dated `finished_at_utc` beyond 5-min skew fails; naive timestamps fail; `state: finished` recorded)
Attack: kill -9 semantics — running marker then nothing: verified red after 2h, but is 2h right for a run that includes a full refit? Marker overwrite ordering: does `_write_running_marker` racing a concurrent run corrupt status? The plist remains a placeholder and the monitor remains unscheduled — this is deployment work the owner must do on the Mac itself; provide the exact plist contents and install commands for both jobs (season 07:30, monitor 09:00) as a copy-paste block for the owner.

## Part 2 — Pre-flight the four approved builds

Your round-5 work plans (identity registry → decision-time odds store → hierarchical ensemble → availability shadow) were written against the pre-round-6 codebase. Re-validate each plan against the current code: file paths, row counts (fixtures now carry `kickoff_utc` for 2,361 rows; one CAN row was repaired), function signatures (`write_fixtures` now exists — the registry's `canonical_store.upsert_fixtures` should subsume it), and the report-only tuner (build 3's promoter is now the ONLY legal writer — confirm the plan's guard matches). Produce the final build-1 (identity registry) execution spec as the next session's work order: exact modules, migration dry-run commands, review-queue workflow for the 42 collision groups, and acceptance checks with current row counts.

## Part 3 — Still open (track, no fix attempted)

Sharp-book anchor; per-window context/calibration plumbing (PredictionConfig, build 3); fuzzy matching (build 1 dependency); cross-league strength; congestion vs scheduled fixtures; rho reset on adjustment rebuilds; GATE_TOL; overdispersion; provider fixture-ID namespacing (build 1); corners all-or-nothing; hyperparameter provenance; `pull_absences` RefreshResult (build 4); preview_bets exposure parity; WC blend refit-or-demote (new, from item 1).

## Output format

Verdicts 1–6 (VERIFIED / INCOMPLETE / WRONG / REGRESSION with reproductions), severity-ranked findings, operator-delusion list, the build-1 execution spec, and a max-10 plan.

Run before concluding: `python3 -m pytest -q` (expect 187 collected; only test_m3 red, and possibly the state-dependent worldcup bracket — diagnose), `python3 -m pytest test_club_soccer.py -q` (24 passed), `python3 test_club_soccer.py`, `python3 -m compileall club_soccer`, `python3 -m club_soccer.evidence_gate` (exit 1), `python3 -m club_soccer.season --no-network --fast` (exit 0), `python3 -m club_soccer.monitor` (exit 0), plus reproductions. Report all outcomes. Do not modify files.
