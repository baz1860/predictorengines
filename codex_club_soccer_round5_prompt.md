# Codex Prompt: Round-5 Review — Verify Round-4 Remediation

Copy everything below the line into Codex.

---

Round 5 of the adversarial review cycle on `club_soccer/`. Your round-4 "implement now" list (items 1–7) has been executed, plus one item from your Critical #2 that wasn't on the list: `ensemble_weights.json` has been deactivated (`active: false` + reason) and `load_ensemble_weights` now requires an explicit `active: true`, so production and validation are the same predictor (verified: identical probabilities on the same fixture, live path vs explicit-args path). Verify everything adversarially; your previous rounds prove first attempts are usually incomplete.

Standing rules: harsh, `file:line` + code + impact + fix for every finding, do not modify files.

## Part 1 — Verify round-4 remediation

### 1. Gate numerics (`evidence_gate.py`: `_finite()` helper — real number, `math.isfinite`, bounded range, bool rejected; applied to lead time, n_bets, ROIs, CLV, fractions, log-losses; naive `generated_at_utc` rejected; gate-tightening TODOs with your exact formulas preregistered in the module)
Attack with new crafted artifacts: values as strings ("0.05"), bools, nested nulls, `n_bets` float like 1000.5, boundary values (exactly 0, exactly the bound), `generated_at_utc` with unusual but tz-aware formats (+05:30 offsets — should those count as "UTC provenance"?). Provenance hashes are still absent by design until decision_time_v2 exists — confirm the TODO block covers them.

### 2. Runtime invariant + adapter gating (`apply_evidence_gate` raises `RuntimeError` — verified firing under `python3 -O`; adapter calls the gate unconditionally before `_mark_recommended` and AGAIN before building recording candidates; recording filter requires no suppression + nonzero stake + future date)
Attack: can `bankroll_store.place_bets` still be reached with club rows from any other caller? Does the double gate call have side effects (double printing, sorting)? Is the `Sneaky`-dict class of bypass (objects resisting mutation) actually representative of any real failure mode, or is the invariant now sufficient?

### 3. Leak test (`test 20` rebuilt: synthetic 84-fixture league, real `walk_forward` run with `CTX.load_coef` and `M.load_ensemble_weights` monkeypatched to raise; leak signal is n==0 because walk_forward swallows per-prediction exceptions; negative control asserts the traps fire on the default path)
Attack: is n>0 a sufficient signal — could a partial leak (some folds tripping, some not) still pass? Could `walk_forward` be refactored to catch the AssertionError differently? Are there OTHER artifact loaders reachable from the prediction path not covered by the two traps (calibration is applied outside walk_forward — confirm; competition strength JSON — confirm still inactive and design the trap for it)?

### 4. Kickoff plumbing (BSD odds take kickoff from `event_date_utc(ev)` with legacy-fixture fallback; The Odds API rows carry `commence_time`; `validate_quotes` now REQUIRES a parseable future `kickoff_utc` for `source="live"` — missing kickoff is non-stakeable; manual keeps the age-gated date-only fallback; tz-aware NaT fallback bug found and fixed in verification)
Attack: `event_date_utc` output format vs `pd.to_datetime(..., utc=True, format="mixed")` — any BSD date shape that coerces to NaT and silently kills valid quotes? The 2,582 legacy future fixtures still lack `kickoff_utc` in fixtures.csv (only newly-fetched rows gain it) — quantify how fast the hole closes with normal operation and whether a backfill fetch is worth it. `fetch_time_only` quotes remain stakeable when fresh — the round-4 "within two minutes" rule is NOT implemented; assess the risk given BSD is currently the only live source.

### 5. Status semantics (AWD removed from `VOID_STATUSES`, added to new `OFFICIAL_RESULT_STATUSES`; `upcoming()` excludes void statuses; `_clear_void_results()` applied on the empty-existing merge path; health gained a hard `void_with_results` check — which immediately caught and led to repair of a real poisoned row: CAN Nantes–Toulouse 2026-05-17 with retained HT scores/shots/xG/possession)
Attack: seeders still bypass `_clear_void_results` (only the fetch merge and health catch it now) — is health-as-backstop sufficient or does each seeder need the call? `played()` includes AWD rows with scores — should awarded 3-0s train the goals model or only settle bets (your round-4 note)? Does anything break when a POS fixture is later rescheduled (new date, same fixture_id)?

### 6. Durable status (`run()` wraps `_run_steps()` in try/except/finally; `last_run.json` written atomically on success, required-failure, health hard-fail, and crash — verified: health hard-fail now writes a red status; `crashed` field; absences/squads/odds-snapshot age fields; new `python3 -m club_soccer.monitor` with 26h freshness, red-status detection, stale-input warnings, `CLUB_SOCCER_NOTIFY_CMD` hook)
Attack: SIGKILL/power-loss leaves the previous last_run.json — is staleness detection sufficient? Monitor isn't scheduled anywhere — propose the launchd/cron wiring including the still-placeholder plist. Is `absences_age_days` from file mtime subject to the same `touch` critique as the old odds gate?

### 7. False-green sweep + capped preview (root `conftest.py` autouse fixture fails any test whose module `FAIL` counter or `_fails` list grew — generic across all root suites; card table gained a "Stake (capped)" column computed via the same pure `app.portfolio.apply_caps` recording uses, with a footnote when caps bite)
Attack: run the FULL root test suite (`python3 -m pytest -q` from repo root, expect some suites to newly fail — those are pre-existing hidden failures the conftest just exposed; enumerate them, they are real bugs to triage, DO NOT "fix" them by weakening the conftest). Preview caps use bankroll=100 and no open-ledger exposure while recording uses live bankroll and priors — quantify how far apart preview and recording can drift and whether that gap matters while the evidence gate keeps stakes at zero.

### 8. Ensemble weight deactivation (`load_ensemble_weights` requires `active: true`; artifact deactivated with dated reason; test asserts flag-less artifacts are ignored; production == validation verified to 1e-12)
Attack: anything else reading `ensemble_weights.json` directly (validate.py's weight-selection section writes it — does promotion now set `active`? If promotion writes `active: true` without the nested holdout your memo 3 requires, the deactivation is one auto-promotion away from being undone — check and propose the guard).

## Part 2 — Still open

Same list as round 4 part 2, minus what's now fixed. Re-verify: sharp-book anchor, per-window context/ensemble refit design, fuzzy matching, cross-league strength, congestion vs scheduled fixtures, rho reset, ensemble promotion holdout + GATE_TOL, overdispersion, provider fixture-ID namespacing, seeder overwrite, openfootball season, corners all-or-nothing, hyperparameter provenance, `pull_absences` failure-vs-quiet-day ambiguity, the placeholder plist.

## Part 3 — Owner decisions: ALL FOUR APPROVED as recommended

The owner has approved memo 1 (full canonical identity registry), memo 2 (T−60m decision-time backtest + per-book snapshot collection), memo 3 (hierarchical ensemble weights), and memo 4 (prospective availability shadow logging). These are no longer "awaiting decision" — convert each memo into an implementation-ready work plan: exact file-by-file changes, new modules and their interfaces, data migrations with row-count checks and rollback commands, and a build order that sequences them (identity registry first, since the backtest and shadow logging both depend on canonical IDs). These plans will be executed in subsequent rounds — make them concrete enough to implement without further design work.

## Output format

Verdicts 1–8 first (VERIFIED / INCOMPLETE / WRONG / REGRESSION with reproductions), then severity-ranked findings, updated operator-delusion list, and a max-10 plan split "implement now" vs "awaiting owner decision".

Run before concluding: `python3 -m pytest test_club_soccer.py -q` (expect 24 passed), `python3 -m pytest -q` from repo root (expect newly-exposed failures in other suites — report them), `python3 test_club_soccer.py`, `python3 -m compileall club_soccer`, `python3 -m club_soccer.evidence_gate` (exit 1, methodology-first), `python3 -m club_soccer.season --no-network --fast` (exit 0), `python3 -m club_soccer.monitor` (exit 0 after the season run), plus your own reproductions. Report all outcomes. Do not modify files.
