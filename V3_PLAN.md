# Sports Predictor Suite - V3 Plan

Cross-engine implementation plan for the app adapters and the World Cup, Club
Soccer, CFB, and Golf engines. V3 is about making the suite safer, more
consistent, and more measurable before adding new modelling power.

## 0. Current State

The desktop app exposes four engines through `app/engines/`:

| Engine | App capabilities | Core model state | V3 priority |
|---|---:|---|---|
| World Cup | predict, simulate, edge | Most mature V2 path: validation gate, calibration, market blend, CLV, context, squad adjustments, portfolio caps | Preserve quality, extract reusable patterns, reduce direct-import collision risk |
| Club Soccer | predict, edge | Ensemble of goals/Elo/SoT xG/form, walk-forward validation, 1X2 calibration, API/manual odds | Bring market blend, CLV, portfolio discipline, stricter validation |
| CFB | predict, edge | Elo + points-power blend, EPA evaluation exists, spread/totals backtests exist | Fix contract drift, add validation gate/calibration, use model choice correctly, add EPA/market blend safely |
| Golf | predict, simulate, edge | Fitted player skill/sigma/form/course model, Monte Carlo, validation gate, portfolio-aware edge | Improve reproducibility/performance, event-aware settlement, common edge contract |

Important existing strengths:

- App adapters are capability-driven; adding features should not require new API
  routes.
- Subprocess runners isolate flat module names for Club Soccer, CFB, and Golf.
- API keys are stored outside `data/app_settings.json` and masked in settings
  responses.
- The World Cup engine has the clearest model-governance pattern: opt-in
  analytical changes, gates before default flips, and dated implementation notes.

Known risks to address early:

- Edge/recording rows are shaped differently per engine, which makes staking,
  settlement, and UI behavior fragile.
- CFB app edge currently ignores the requested `model` parameter in the runner
  path and duplicates logic from `cfb/edge.py`.
- Golf settlement grades against the latest completed event rather than an
  explicit event identifier, so stale/open bets can be settled against the wrong
  tournament.
- API request bodies are loosely typed (`dict`), which is convenient but gives
  little validation or input-size control.
- Subprocess errors are intentionally shortened, but the runner output contract
  is not strict enough to detect non-finite JSON or noisy stdout systematically.

## 1. Non-Negotiable V3 Guardrails

1. No model change ships without an engine-specific validation gate passing.
2. New modelling features default OFF until they improve a held-out metric or are
   pure risk controls.
3. No raw API keys in logs, API responses, reports, screenshots, or exception
   messages.
4. No shell execution with user-controlled command strings. Fixed runner paths
   and argv lists only.
5. No auto-placement of real money. Recording remains a local ledger action.
6. Ledger migrations must be append/backward-compatible and backed up before any
   destructive rewrite.
7. Every stochastic path must accept a seed and be reproducible in validation.
8. Every edge row must include enough metadata to settle it later without
   guessing.

## 2. Milestones

Execute in order. Each milestone ends with a dated `V3_NOTES.md` entry containing
files changed, metrics, and any rejected approaches.

### M1 - Common Engine Contract and Contract Tests

Goal: make all engines speak one stable app contract before changing modelling.

Build:

- Add `app/engines/contracts.py` with lightweight helpers for:
  - prediction result validation;
  - table/column validation;
  - edge row normalization;
  - finite-number JSON validation;
  - stable market identifiers.
- Define a canonical edge row shape:
  `event_id`, `match_date`, `home`, `away`, `market`, `side`, `line`, `bet`,
  `odds`, `p_model`, `p_book` or `p_market`, `edge`, `ev_per_unit`,
  `kelly_frac`, `stake_gbp`, `source`, `model`, `recommended`.
- Update adapters/runners to emit this shape while preserving current UI columns.
- Add `test_engines_contract.py` that calls every registered adapter:
  `/schema`, `predict` where possible, `edge` from manual odds where possible,
  and `simulate` for World Cup/Golf with tiny seeded sims.
- Fix CFB runner model selection so `model=elo|power|blend` is honored in edge.

Acceptance:

- `python3 test_engines_contract.py` passes.
- Existing tests still pass: `test_m2.py` through `test_m7.py` and
  `test_club_soccer.py`.
- CFB edge output changes only when a non-default model is requested.
- No engine emits NaN/Inf or unserializable objects.

### M2 - Security and Reliability Hardening

Goal: reduce attack surface without changing model outputs.

Build:

- Replace loose `dict` API request validation with bounded Pydantic models:
  `engine` must be a registered slug, params must stay under a conservative
  serialized size, and numeric fields get min/max clamps at the API boundary.
- Add a `safe_runner_env()` helper that passes only required environment keys to
  subprocesses. Preserve API-key access only where needed.
- Harden `run_engine()`:
  - reject unknown runner commands before subprocess launch;
  - parse strict final-line JSON with finite-value validation;
  - include sanitized stderr snippets that redact known key values and
    key-looking tokens;
  - keep timeouts engine-specific.
- Update `api_keys.save_keys()` to write with owner-only permissions where the
  OS supports it.
- Add network-fetch wrappers per engine that include timeout, provider label,
  and redacted errors.
- Add a preflight check/report command for missing data files and missing keys.

Acceptance:

- App settings still never return raw keys.
- A synthetic runner error containing a fake API key is redacted in the API error.
- Path traversal or unknown engine ids return 404/422, never filesystem errors.
- Prediction and edge numeric outputs are unchanged for default local/manual
  runs.

### M3 - Validation Gates Across All Engines

Goal: make "do not degrade modelling" enforceable everywhere.

Build:

- Add `validate_all.py` that runs each engine's validation in quiet/gate mode:
  - World Cup: existing `validate.py --quiet --gate`;
  - Club Soccer: `club_soccer/validate.py --gate`;
  - Golf: `golf/validate.py --quiet --gate --sims <small default>`;
  - CFB: new `cfb/validate.py --gate`.
- Create `cfb/validate.py` by consolidating the useful pieces from
  `predictor.py --backtest`, `blend_eval.py`, `ats_backtest.py`, and
  `totals_backtest.py`.
- Store CFB baseline in `cfb/data/validation_baseline.json` with separate metrics
  for moneyline Brier, margin MAE, total MAE, ATS ROI by threshold, and totals
  ROI by threshold.
- Add a suite summary JSON at `data/validation_suite.json`.

Acceptance:

- `python3 validate_all.py --gate` exits non-zero on any regression and prints a
  compact per-engine table.
- CFB validation is walk-forward and does not fit on future games.
- No baseline is silently loosened. Baseline updates require an explicit
  `--update-baseline`.

### M4 - Shared Bankroll, Portfolio, and Settlement Semantics

Goal: one safer staking/settlement path for all engines.

Build:

- Move portfolio sizing concepts into `app/portfolio.py` or a shared root module:
  daily cap, single-event cap, correlated exposure cap, drawdown brake, minimum
  stake.
- Make each adapter return `recommended` and `stake_gbp`; recording should not
  re-pick candidates with ad hoc filters in the adapter.
- Add event identity:
  - World Cup/Club/CFB: stable fixture key from date, home, away, competition
    when available;
  - Golf: tournament id plus market participant key.
- Extend `suite_ledger.csv` backward-compatibly with optional columns:
  `event_id`, `market`, `line`, `source`, `model`, `closing_odds`.
- Fix Golf settlement to grade only against the matching tournament/event, not
  simply the latest event.
- Add settlement dry-run mode that reports what would settle without writing.

Acceptance:

- Old suite ledger loads unchanged.
- A mixed ledger with all four engines settles only matching events.
- Golf stale outright bets remain open when the latest event is not their event.
- Recording the same open bet twice remains deduped.

### M5 - Market Blend and CLV for Every Priced Engine

Goal: make betting evaluation less fake-edge prone across the suite.

Build:

- Generalize market blending:
  - World Cup keeps existing 1X2 logit blend;
  - Club Soccer adds 1X2 market blend first, then evaluates totals/BTTS after
    calibration support is understood;
  - CFB adds spread/total market anchoring by blending model margin/total toward
    closing/current market lines with fitted weights;
  - Golf uses existing calibrated/market-blended paths but writes the same
    metadata as other engines.
- Generalize CLV snapshots:
  - one suite-level `clv_suite.py`;
  - provider adapters for The Odds API, API-Football, DataGolf/manual snapshots;
  - closing proxy matched by `event_id` and market key.
- Add `closing_odds` and `clv_pct` to reports where available.

Acceptance:

- Held-out log-loss or line-error improves versus pure model and pure market for
  any default-candidate market blend.
- If a market blend does not improve, it remains available only behind an
  experimental flag and is not used for recommendations.
- Empty/no-network CLV history produces a clear no-data report, not a crash.

### M6 - Engine-Specific Modelling Upgrades

Goal: improve each engine only where validation says it helps.

World Cup:

- Preserve V2 as the baseline.
- Add richer calibration only if enough post-V2 data exists: separate calibration
  for totals/BTTS or stage-specific knockout calibration.
- Keep squad/context adjustments opt-in until two consecutive validation reports
  support a default flip.

Club Soccer:

- Tune ensemble weights with nested walk-forward validation instead of fixed
  constants.
- Add league/competition-level home advantage and goal environment parameters.
- Add team-news/injury hooks only as metadata until a validation set exists.
- Add market blend and CLV before increasing stake aggressiveness.

CFB:

- Promote EPA into the main blend only if the untouched validation season beats
  current blend on Brier and margin MAE.
- Calibrate moneyline probabilities.
- Calibrate spread/total residual sigma by season/week and market type.
- Add preseason priors/returning-production checks around season boundaries.

Golf:

- Vectorize or chunk simulations to improve speed without changing seeded
  outputs beyond Monte Carlo tolerance.
- Add event-specific cut/no-cut metadata so cut and placement markets are never
  priced when structurally invalid.
- Validate course-fit and recent-form weights with time-split tournaments before
  changing defaults.
- Add dead-heat/place-rule metadata for books where available.

Acceptance:

- Every engine-specific change has a before/after validation table.
- Any worse or inconclusive feature is documented and left off by default.
- Existing default CLI/app behavior is unchanged unless a milestone explicitly
  promotes a gated improvement.

### M7 - Data Provenance and Refresh Hygiene

Goal: know what data produced a prediction.

Build:

- Add `data_manifest.json` per engine with source, fetched_at, row counts, and
  schema version for key inputs.
- Add data freshness warnings to engine schemas/info so the UI can show stale
  fixtures, stale odds, stale field, or stale model params.
- Make update scripts write manifests after refresh.
- Add schema checks for manual odds files with actionable errors.

Acceptance:

- Each engine can report freshness without making network calls.
- Manual odds mistakes identify row number, column, and expected values.
- Data freshness warnings do not block offline operation.

### M8 - App UX for Power Users

Goal: expose V3 controls without making the app noisier.

Build:

- Add a compact "model audit" panel per engine: last validation status, model
  params age, data freshness, active flags.
- Add edge filters in the UI: min edge, min EV, source, market, recommended only.
- Add dry-run/record split for Edge so users can preview recommendations before
  writing the ledger.
- Add "write odds template" result with absolute path and row count.
- Add CSV export for any result table.

Acceptance:

- No API keys or secret paths appear in UI exports.
- Existing predict/simulate/edge workflows remain one-click.
- Mobile and desktop layouts do not overlap text or controls.

### M9 - Release and Ops

Goal: make V3 maintainable after implementation.

Build:

- Add `V3_NOTES.md` implementation log.
- Update README files per engine.
- Add a single `python3 run_checks.py` or documented command list:
  contract tests, fast unit tests, validation gates where feasible.
- Add daily update summary that includes validation status, CLV status, stale
  data warnings, and recommendations count.

Acceptance:

- A fresh local run can reproduce the app and checks from README instructions.
- V3 default recommendations are backed by passing gates.
- Any unavailable network provider degrades with a clear local action.

## 3. Suggested Execution Order

```
M1 contract tests
  -> M2 security hardening
  -> M3 validation gates
  -> M4 bankroll/settlement
  -> M5 market blend + CLV
  -> M6 modelling upgrades
  -> M7 provenance
  -> M8 UX
  -> M9 release docs
```

If time-boxed, do M1-M4 first. Those reduce modelling and money-management risk
without requiring a bet on any new algorithm.

## 4. First Implementation Checklist

- Create `V3_NOTES.md`.
- Implement `test_engines_contract.py` before changing engine behavior.
- Fix CFB model-selection drift in the runner.
- Add redaction tests for subprocess/API errors.
- Run:
  - `python3 test_club_soccer.py`
  - `python3 test_m2.py`
  - `python3 test_m3.py`
  - `python3 test_m4.py`
  - `python3 test_m5.py`
  - `python3 test_m6.py`
  - `python3 test_m7.py`
  - `python3 validate_all.py --gate` once M3 exists

## 5. Definition of Done for V3

V3 is done when all engines share one app/edge/ledger contract, every engine has
a regression gate, settlement is event-safe, keys and subprocesses are hardened,
and any new modelling default has demonstrated held-out improvement without
weakening the security or bankroll safeguards.
