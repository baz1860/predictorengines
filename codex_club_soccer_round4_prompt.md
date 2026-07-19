# Codex Prompt: Round-4 Review — Verify Round-3 Remediation + Finalize Designs

Copy everything below the line into Codex.

---

Round 4 of the adversarial review cycle on `club_soccer/`. Your round-3 "implement now" list has been executed. Verify each item adversarially — the previous rounds proved these fixes are usually incomplete on first attempt — then re-check everything still open, and produce final implementable designs for the owner-decision items.

Standing rules: be harsh, no praise, every finding needs `file:line` + code + impact + fix, do not modify files.

## Part 1 — Verify the round-3 remediation

### A. Gate runs last (`edge.py`: `apply_evidence_gate()` extracted; called after the market blend; hard `assert` that a closed gate and nonzero stake never coexist; app adapter re-applies the gate after its experimental blend)
Attack: any remaining code path that rewrites `kelly_stake`/`stake_gbp` after the final gate call (in `edge.py`, the adapter, `bankroll_store`, report writers)? Is the assert reachable in production or optimized out under `python -O`? Should the invariant be an exception even without asserts? Does the adapter's `record` path call the gate function itself, or only rely on `suppressed_reason` filtering?

### B. Strict evidence schema (`evidence_gate.py`: methodology-first validation — `backtest_version == "decision_time_v2"`, selection/execution/CLV-reference fields, `decision_lead_minutes >= 60`, `generated_at_utc` freshness replacing mtime, exact thresholds {2%,4%,6%}, min bets at EVERY threshold, CLV required non-null)
Attack with crafted artifacts: correct methodology strings but absurd values elsewhere; thresholds as {"2%","4%","6%"} with empty rows; `generated_at_utc` with weird timezone formats; extra unknown fields (currently NOT rejected — is that a real risk?); data/model/code-hash provenance still absent — how much does that matter before the decision-time backtest exists? Your round-3 statistical criticisms (no confidence intervals, no per-league minimums, no Wilson bound) are still open by design — confirm they're documented as gate-tightening TODOs and propose the exact formulas to use when `decision_time_v2` artifacts exist.

### C. Validation/production alignment (context artifact deactivated with `deactivated_reason`; `predict()` gained explicit `ensemble_weights`; `walk_forward` passes `context_coef={}` + `DEFAULT_ENSEMBLE_W`; new test 20 pins both leak-free paths by monkeypatching the loaders to raise)
Attack: with context now inactive, validation and production should be the SAME predictor — verify by comparing a live-path and validation-path prediction for an identical fixture. Any remaining `load_*()` reads inside the walk-forward paths (calibration? league adjustments? competition strength still `active: false`?). Is `DEFAULT_ENSEMBLE_W` itself contaminated (was it hand-tuned on this data historically)? Does test 20 actually fail if someone removes `context_coef={}` from `validate.py` (the source-inspection check) — try it mentally against refactorings that would evade a string match. Design the per-window context + ensemble-weight refit concretely enough to implement (you estimated ~18s/fold for context; propose the caching).

### D. Timestamp hardening (future-dated `quoted_at_utc` rejected with 5-min skew tolerance; `format="mixed"` date parsing; The Odds API uses provider `last_update` with `quote_time_source` provenance; BSD labeled `fetch_time_only`; `kickoff_utc` added to schema + BSD ingestion + odds frames; `validate_quotes` drops already-kicked-off fixtures when `kickoff_utc` present)
Attack: legacy fixtures have no `kickoff_utc` — what fraction of current upcoming fixtures carry it, and does the date-only fallback still leave the same-day hole for them? Should `fetch_time_only` quotes be barred from any future strong gate (currently only labeled)? Backfill design for cached BSD event payloads. Does `snapshot_odds.py` need the same columns to ever support the decision-time backtest (you found it stores only 120 median rows — specify the new snapshot schema).

### E. Void fixtures (`schema.VOID_STATUSES`/`RESULT_COLUMNS`; `_merge_fixture_rows` clears results on void transitions including newly-appended rows; `model.played()` excludes void statuses)
Attack: the status set {POS, CAN, ABD, SUS, INT, AWD} — is AWD (awarded) right to void? An awarded match HAS an official result. Check real BSD status strings for walkovers/awards. `upcoming()` still returns score-less fixtures — does a cancelled (CAN) fixture now show as upcoming forever? Do seeders/`seed_real.py` paths bypass `_merge_fixture_rows`? Is there stale poisoned data ALREADY in fixtures.csv (void status + retained scores) needing a one-time repair, and does `health.py` check for it?

### F. Fail-closed completion (BSD fetch `required=True` on networked runs; `_FAILED_REQUIRED.clear()` at `run()` entry; durable `data/last_run.json` with ok/failures/backed-count)
Attack: is `last_run.json` written on the `sys.exit` paths (health hard-fail at the top of `run()`)? Who reads it — propose the concrete freshness monitor (launchd plist fix included; you found the shipped plist has a placeholder WorkingDirectory). Absences/squads staleness: still silently used when refresh fails?

## Part 2 — Still open (no fix attempted; re-verify and keep the pressure on)

Identity registry (42 collision groups — the round-3 design awaits owner approval; refine it into a migration script spec with row counts and rollback), decision-time backtest redesign (blocked on odds snapshot schema — specify it), sharp-book anchor for edges, per-window context/ensemble refit (from C), portfolio caps in card/CLI previews (`bankroll_store` caps exist at recording only), false-green `check()` in the other root test suites, fuzzy player/team matching, cross-league strength, future congestion vs scheduled fixtures, rho reset on adjustment rebuilds, ensemble promotion holdout + ±0.01 gate, overdispersion, provider fixture-ID namespacing, seeder overwrite, openfootball season derivation, corners all-or-nothing, hyperparameter provenance (365d half-life etc.), per-league ensemble heterogeneity.

## Part 3 — Owner-decision packages

For each of the four owner decisions (identity migration, decision-time backtest parameters, ensemble weight structure, player-availability prospective evaluation), produce a one-page decision memo: options, recommendation, cost, risk, and the exact acceptance criteria that would let the related gate open. These will be read by the owner, not implemented by you.

## Output format

Fix verdicts A–F first (VERIFIED / INCOMPLETE / WRONG / REGRESSION with reproductions), then severity-ranked findings, then the updated operator-delusion list, then the four decision memos, then a max-10 remediation plan split into "implement now" vs "awaiting owner decision".

Run before concluding: `python3 -m pytest test_club_soccer.py -q` (expect 24 passed), `python3 test_club_soccer.py`, `python3 -m compileall club_soccer`, `python3 -m club_soccer.evidence_gate` (expect exit 1, failing on methodology first), `python3 -m club_soccer.season --no-network --fast` (expect exit 0, card + last_run.json written, no backed bets), plus your own reproductions. Report all outcomes. Do not modify files.
