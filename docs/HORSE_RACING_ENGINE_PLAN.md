# Horse Racing Predictor Engine Plan

Date: 2026-07-14
Status: proposed; no production model or provider selected

## Outcome

Build a local-first, auditable horse-racing engine that reconstructs what was
known at a fixed pre-race cutoff, produces calibrated race-level win
probabilities and fair odds, compares them with executable prices, and records
paper-traded decisions for later settlement and CLV analysis.

The first release is successful when its data replay and probability forecasts
are trustworthy. Profitability is a research result, not an acceptance shortcut.

## Recommended V1 scope

Unless the Gate 0 data audit shows a better starting segment:

- UK and Ireland flat racing only.
- Turf and all-weather retained as separate conditions within the model.
- Pre-race win market only.
- One declared prediction cutoff, initially 15 minutes before scheduled off.
- Decimal odds; one odds-source contract at first.
- Paper trading only.
- No jumps, in-play, each-way, place, forecast, tricast, or laying in V1.

The scope is intentionally narrow because jumps require materially different
form/variance treatment, while each-way and place products add field-dependent
terms, deductions, dead heats, and settlement complexity before the win model is
proven.

## Non-negotiable design rules

1. Every feature has an `asof_timestamp` and is reproducible at the prediction
   cutoff.
2. Raw provider responses are immutable; corrections create new ingested
   versions rather than rewriting history invisibly.
3. Horse, trainer, jockey, course, race, and runner identities use stable IDs.
   Names are labels and matching inputs, never primary keys.
4. A race is modelled jointly. Runner win probabilities must be finite, non-
   negative, and sum to one after withdrawals are applied.
5. Pure-model, market, and blended probabilities are stored separately.
6. Starting price or post-cutoff market data is never a pure-model feature.
7. A challenger remains report-only until it clears an untouched walk-forward
   gate. Failed experiments remain documented and default-off.
8. Backtests execute the production path: cutoff, field state, available odds,
   commission/terms, withdrawal handling, staking, and settlement.
9. Missing or stale data causes an explicit degraded/no-prediction state; it is
   never silently replaced by future or market information.
10. Given the same snapshot and model artifact, prediction output is
    deterministic apart from explicitly seeded simulation.

## Product contract to freeze at Gate 0

The following decisions must be written into a versioned configuration before
historical model work begins:

- Included jurisdictions, race codes, classes, surfaces, and age restrictions.
- Scheduled-off versus actual-off semantics.
- Prediction cutoff and acceptable source latency.
- Declaration/non-runner policy at that cutoff.
- Odds source, price type, availability definition, and commission/terms.
- Whether official ratings and third-party speed figures are available as-of or
  only retrospectively.
- Minimum history and data-completeness thresholds for issuing a price.
- Result treatment for dead heats, disqualifications, voids, abandoned races,
  walkovers, and corrections.
- Primary evaluation metric and promotion margin.
- Paper-trading exposure limits; no real-money enablement is in this plan.

## Architecture

```text
provider payloads
      |
      v
immutable raw archive -----> ingestion manifest / checksums
      |
      v
canonical race + entity store
      |
      v
as-of historical snapshot builder
      |
      +----> feature store ----> fit ----> model artifact
      |                              
      +----> replay race ----> predict ----> calibrate
                                         |
odds snapshot ---------------------------+----> price/edge report
                                                  |
results ----------------------------------------> settlement + CLV
```

### Proposed package layout

```text
horse_racing/
├── README.md                 # front door and operational contract
├── config.py                 # versioned scope/cutoff/model configuration
├── schema.py                 # canonical records and validation
├── identities.py             # stable IDs, aliases, unresolved-match quarantine
├── ingest/
│   ├── base.py               # provider interfaces
│   ├── racecards.py
│   ├── results.py
│   └── odds.py
├── snapshots.py              # immutable as-of race reconstruction
├── features.py               # leakage-safe feature calculation
├── ratings.py                # dynamic horse/trainer/jockey state
├── model.py                  # race-level probability models
├── calibrate.py              # held-out calibration only
├── simulate.py               # seeded complete-order simulation, post-V1-ready
├── market.py                 # de-vig and optional held-out market blend
├── edge.py                   # fair price, EV, eligibility, recommendations
├── portfolio.py              # fractional Kelly and exposure caps; default-off
├── settle.py                 # result/withdrawal/dead-heat grading
├── validate.py               # walk-forward evaluation and promotion gate
├── quality.py                # freshness, completeness, identity/data checks
├── provenance.py             # artifact and source manifests
├── engine.py                 # app command boundary
├── season.py                 # daily CLI orchestrator
└── data/
    ├── raw/                  # immutable, source/timestamp partitioned
    ├── quarantine/           # unresolved identities or invalid records
    ├── curated/              # canonical local tables
    ├── snapshots/            # replayable decision-time snapshots
    ├── artifacts/            # fitted models/calibrators/manifests
    └── reports/              # validation, predictions, edge, paper ledger

app/engines/horse_racing.py   # suite adapter, added only after standalone gate
```

CSV may be used for small human-edited inputs, but historical race/runner/odds
tables should use a typed local store such as Parquet plus DuckDB, or SQLite if
the data licence/provider shape makes that simpler. The choice is made at Gate 0
after measuring volume and update patterns; the canonical schema is independent
of storage.

## Canonical data contract

### Race

Required fields:

```text
race_id, source_race_id, jurisdiction, course_id, meeting_date,
scheduled_off_utc, actual_off_utc, race_number, code, surface, going,
distance_metres, race_class, race_type, age_band, sex_restriction,
handicap_flag, prize_money, declared_field_size, source_updated_at,
ingested_at, record_version
```

### Runner declaration/snapshot

Required fields:

```text
race_id, runner_id, horse_id, trainer_id, jockey_id, saddlecloth,
draw, age, sex, weight_carried_kg, claim_kg, official_rating,
headgear, declared_status, non_runner_status, source_updated_at,
ingested_at, record_version
```

### Historical run/result

Required fields:

```text
race_id, runner_id, finish_position, official_position,
completion_status, beaten_distance, starting_price, result_status,
result_published_at, result_updated_at, record_version
```

Raw timing, sectional, speed-figure, comments, and pace fields are optional
extensions. Their publication timestamps and definitions must be known before
they can enter an as-of feature set.

### Odds snapshot

Required fields:

```text
race_id, runner_id, market_id, source, price_type, decimal_odds,
available_size, captured_at, source_updated_at, field_version,
market_status
```

### Prediction snapshot

Required fields:

```text
prediction_id, race_id, runner_id, prediction_cutoff,
data_snapshot_id, field_version, feature_schema_version, model_version,
p_model_raw, p_model_calibrated, p_market, p_blended, fair_odds,
data_quality_status, generated_at
```

The adapter must supply an explicit provider race ID or deterministic race key;
the shared two-team `fixture_key()` is not valid for racing.

## Data ingestion and replay design

### Raw archive

- Partition by provider, endpoint, retrieval date, and retrieval timestamp.
- Store payload checksum, request parameters excluding secrets, HTTP/source
  status, parser version, and licence/source metadata.
- Make ingestion idempotent by source object/version.
- Never log provider keys or authenticated URLs.

### Entity resolution

- Prefer provider IDs and maintain a cross-provider identity map.
- Normalize display names only to generate match candidates.
- Resolve ambiguous matches using supporting attributes and reject uncertainty.
- Quarantine unresolved records; never merge on name alone.
- Add regression fixtures for suffixes, punctuation, renames, duplicate names,
  transferred trainers, and cross-jurisdiction IDs.

### As-of replay

`build_snapshot(race_id, cutoff)` must select the latest version of each record
whose source publication/update time was available by the cutoff. It records the
exact input record versions in `data_snapshot_id`.

Replay tests must cover:

- Late declarations and withdrawals.
- Going changes.
- Jockey/trainer changes.
- Rescheduled or delayed races.
- Provider result/rating corrections.
- Same-day trainer/jockey statistics.
- Odds snapshots before and after the cutoff.
- A race replayed after later horse runs have occurred.

## Feature plan

Features are calculated from prior completed races, never copied from a current
retrospective profile.

### V0 features

- Dynamic, time-decayed horse ability and uncertainty.
- Recent starts, days since last run, age, carried weight, draw, and field size.
- Relative official rating available at the cutoff.
- Distance, going, surface, course, and race-class compatibility.
- Time-decayed trainer and jockey effects with hierarchical shrinkage.
- Course/distance and surface history with sample counts and shrinkage.
- Field-relative versions of important runner features.
- Explicit missingness and history-depth indicators.

### Later challengers

- Pace/running-style interactions and likely pace map.
- Sectional timing and speed figures, only with historical as-of availability.
- Trainer/horse and jockey/horse interaction effects.
- Draw interactions by course, distance, surface, going, and field size.
- Travel, rest pattern, stable form, equipment changes, and weather.
- Text-derived race comments only after a separately audited NLP pipeline; no
  LLM-generated numeric probability enters the production model.

Each feature group requires an ablation report. Sparse interaction effects are
shrunk or excluded rather than accepted because their in-sample fit is strong.

## Model ladder

### Baseline 0: uniform field

`p_i = 1 / field_size`. This verifies scoring and probability invariants.

### Baseline 1: ratings-only softmax

A transparent official/dynamic-rating score converted to race-level
probabilities. This establishes the value of history without complex features.

### Baseline 2: de-vigged market

An odds-only benchmark at the same decision cutoff. It is a benchmark, not a
pure-model input.

### Champion candidate: hierarchical race-level utility model

Use a regularized conditional-logit/Plackett-Luce-style winner model:

```text
utility_i = dynamic_horse_ability
          + recent_form
          + trainer_effect
          + jockey_effect
          + condition_suitability
          + race/field interactions

p_i = exp(utility_i) / sum_j(exp(utility_j))
```

Horse, trainer, jockey, and sparse condition effects are partially pooled.
Ability is time-varying and carries uncertainty for lightly raced horses.

This is the preferred first champion because it respects race competition,
provides coherent probabilities, and remains diagnosable.

### Challenger: gradient-boosted residual utility

After the statistical champion is stable, fit a boosted-tree challenger to
runner utility/residual structure using the same walk-forward folds. Convert
scores to race-level probabilities and calibrate on a disjoint calibration
period. It is promoted only if gains survive ablation, temporal stability, and
segment tests.

### Future finishing-order simulation

Place and exotic markets require coherent complete-order draws, potentially
using Plackett-Luce sampling or latent-performance simulation with race-level
common shocks. Win probabilities must reconcile with simulated first-place
frequency, and nested market probabilities must be enforced.

## Calibration and market use

- Fit calibration only on predictions from data not used to train the base
  model.
- Compare temperature scaling, beta/logistic calibration, and carefully
  regularized isotonic calibration; choose through future folds.
- Report calibration intercept/slope, reliability buckets, expected calibration
  error, Brier, and log loss.
- Check calibration by probability/odds band, field size, class, surface,
  jurisdiction, course, going, age band, and data completeness.
- Learn any market blend out-of-sample, preferably in log-odds space.
- Keep pure-model output available for audit and information-value measurement.
- Closing odds are an evaluation/CLV target, never retroactively treated as a
  decision-time executable price.

## Validation protocol

### Splitting

- Use expanding-window chronological walk-forward evaluation.
- Group all runners in a race in the same fold.
- Prevent same-meeting or same-day aggregates from seeing later races.
- Reserve the most recent substantial period as a final untouched test set.
- Tune hyperparameters and calibration only inside the development period.
- Freeze the entire pipeline before opening the final test set.

### Primary metrics

- Multinomial/race-level log loss: primary probability metric.
- Brier score and Brier skill versus declared baselines.
- Calibration intercept, slope, and reliability error.
- Rank probability score or suitable finishing/ranking metric when order models
  are introduced.

### Secondary decision metrics

- Closing-line value.
- EV and ROI using prices captured at the actual cutoff.
- Bet count, turnover, maximum drawdown, and confidence intervals.
- Sensitivity to commission, slippage, missing prices, and stricter edge filters.

Accuracy, winner strike rate, and raw ROI are never sufficient promotion metrics.

### Segment and stress tests

- Favourite, mid-price, and longshot bands.
- Small/medium/large fields.
- Turf versus all-weather.
- Going, class, distance, course, age, and jurisdiction.
- Debutants and low-history runners.
- Missing official rating or connection history.
- High non-runner races and post-declaration field changes.
- One season removed, one course removed, and provider-correction replay.
- Odds latency and price-unavailability stress.

### Promotion gate

A challenger can become default only when:

1. Schema, identity, replay, leakage, and probability-invariant tests pass.
2. Its final configuration was chosen without the untouched test period.
3. It improves the primary held-out metric by a predeclared margin or provides a
   clearly justified calibration/robustness gain without material regression.
4. Improvement is not concentrated in one time period, course, trainer, odds
   band, or small group of races.
5. Bootstrap confidence intervals and fold results do not indicate a fragile
   result.
6. Production-path replay yields the same predictions as offline evaluation.
7. Artifact provenance is complete and the previous champion remains available
   for rollback.

The numeric promotion margin is set after baseline variance is measured, before
challenger results are inspected.

## Pricing, eligibility, and paper portfolio

For decimal odds `o` and selected probability `p`:

```text
fair_odds = 1 / p
ev_per_unit = p * o - 1
```

Any exchange commission or product-specific term must be included in expected
return and settlement. Recommendations require:

- A valid, available price captured at/before the decision timestamp.
- Matching field version and no unresolved withdrawal.
- Passing freshness and data-completeness checks.
- Edge above a configured uncertainty/transaction-cost buffer.
- Passing per-race, meeting, day, runner, and correlated-exposure caps.

Fractional Kelly is optional and default-off until probability calibration and
paper-trading stability are demonstrated. The pure prediction engine does not
depend on staking.

## Settlement and audit ledger

The V1 ledger records the offered price, odds timestamp, field version,
probability source, model/data versions, eligibility decision, stake if any, and
later result. Settlement logic must explicitly test:

- Non-runners and market voiding.
- Withdrawals after prediction but before bet placement.
- Dead heats.
- Disqualifications and amended results.
- Abandoned/void races and walkovers.
- Duplicate settlement attempts.
- Result corrections after initial settlement.

Settlement is idempotent, event-safe, and reversible through a correction audit
entry rather than destructive history edits.

## Repository integration

The standalone engine must pass its own gates before app registration.

1. Implement `horse_racing.engine.COMMANDS` for `schema`, `refresh`, `predict`,
   `simulate` when available, and `edge`.
2. Add `app/engines/horse_racing.py` using the existing `EngineAdapter` pattern.
3. Return table-style race predictions and explicitly set each row's `event_id`
   to the stable race ID.
4. Extend shared market normalization only if required; do not overload two-team
   home/away semantics internally. At the adapter boundary, map the horse label
   into the existing canonical participant fields without losing runner/race IDs.
5. Add registry wiring in `app/engines/__init__.py` only after contract tests pass.
6. Add horse validation to `validate_all.py`, preflight/provenance, dashboard, and
   daily-card orchestration only after the standalone CLI is deterministic.
7. Keep all horse model/data imports package-qualified to avoid the repository's
   existing flat-module collision problem.

## Test strategy

### Unit tests

- Schema types, units, timezone conversion, and required timestamps.
- Stable ID mapping and ambiguous identity rejection.
- As-of record selection and later-correction exclusion.
- Rolling-feature cutoff correctness.
- Probability sum/range/finite invariants.
- De-vig, fair odds, EV, commission, and edge eligibility.
- Withdrawal, dead-heat, void, and correction settlement.
- Seeded simulation reproducibility.

### Integration tests

- Raw fixture -> canonical store -> as-of snapshot -> features -> prediction.
- Historical race replay after later data has been ingested.
- Odds/field version mismatch rejection.
- Prediction -> ledger -> result -> idempotent settlement.
- Adapter contract and strict JSON safety.

### Golden tests

Maintain a small set of hand-audited races spanning ordinary, missing-data,
withdrawal, dead-heat, delayed, and corrected-result scenarios. Golden fixtures
store source timestamps and expected record-version selection, not merely final
probabilities.

## Milestones and exit gates

### M0 — Charter and data feasibility

Deliver:

- Frozen V1 scope and cutoff contract.
- Provider/licence/retention assessment.
- Sample racecard, result, odds, and historical-form payloads.
- Coverage, latency, correction, and stable-ID report.
- Storage decision and canonical schema review.

Exit gate: enough legal, timestamped data exists to reconstruct historical
decision-time snapshots. If not, stop or change scope before model work.

### M1 — Ingestion and identity foundation

Deliver:

- Immutable raw archive and ingestion manifest.
- Canonical race/runner/result/odds tables.
- Cross-provider identity map and quarantine flow.
- Data-quality and freshness report.

Exit gate: repeated ingestion is idempotent; ambiguity is rejected; sampled
races reconcile to source records.

### M2 — Historical replay and feature store

Deliver:

- Versioned as-of snapshot builder.
- Leakage-safe rolling features.
- Golden replay suite and contamination audit.
- Training dataset manifest.

Exit gate: audited historical races reproduce only cutoff-available information,
including after later runs and corrections have been ingested.

### M3 — Baselines and validation harness

Deliver:

- Uniform, ratings, and de-vigged-market baselines.
- Walk-forward folds and untouched test reservation.
- Metric, calibration, segment, and bootstrap reports.
- Predeclared promotion policy based on measured baseline variance.

Exit gate: metrics are reproducible and intentionally broken/leaky challengers
are caught by tests.

### M4 — Hierarchical race model

Deliver:

- Dynamic ratings and race-level utility model.
- Held-out calibration.
- Feature ablations, uncertainty, and failure-mode report.
- Versioned champion artifact with rollback.

Exit gate: champion clears the promotion gate without relying on ROI.

### M5 — Odds, edge, and settlement

Deliver:

- Timestamped odds ingest and de-vig.
- Pure/market/blended price comparison.
- Recommendation eligibility and default-off portfolio layer.
- Paper ledger, settlement, CLV, and correction audit.

Exit gate: exact historical production replay passes under conservative costs
and operational edge cases.

### M6 — Shadow operation

Deliver:

- Scheduled refresh/predict/settle flow.
- No-bet/degraded-state handling.
- Freshness, missingness, drift, calibration, CLV, and operational dashboards.
- Incident and rollback runbook.

Exit gate: sustained shadow operation has no unresolved material data or
settlement defects. Forecast quality is consistent with backtest expectations.

### M7 — Suite integration

Deliver:

- App adapter and contract tests.
- Shared provenance, ledger, validation gate, dashboard, and daily-card wiring.
- User-facing race table with probabilities, fair odds, quality status, and model
  provenance.

Exit gate: the full repository check suite and horse gate pass; integration does
not alter existing engines.

## Model artifact manifest

Every model/calibrator/blend artifact records:

```text
artifact_id, created_at, git_commit, provider/source versions,
raw and curated data checksums, training date range, data max timestamp,
race/runner counts, scope filters, cutoff policy, feature schema version,
model class and hyperparameters, random seeds, fold definitions,
calibration method and period, market-blend method and period,
validation report ID, environment/dependency versions
```

## Operational monitoring

Monitor and alert on:

- Source failures, latency, stale racecards, missing odds, and schema drift.
- Identity quarantine count and new/unseen entity rates.
- Non-runner frequency and field-version mismatches.
- Feature missingness and distribution drift.
- Probability entropy, favourite/longshot distribution, and calibration drift.
- Pure-model versus market disagreement.
- CLV, recommendation count, turnover, settlement lag, and correction rate.
- Model artifact, data snapshot, and code-version mismatch.

## Principal risks and controls

| Risk | Control |
|---|---|
| Retrospective data leakage | Versioned source timestamps, as-of snapshots, golden replay and contamination tests |
| Wrong cross-provider identity | Stable IDs, evidence-based mapping, quarantine and audited overrides |
| Market leakage | Separate pure/market/blended features and artifacts; cutoff enforcement |
| Sparse-feature overfit | Hierarchical shrinkage, ablation, time folds and segment stability |
| Non-runner/field mismatch | Field version in odds and prediction records; mandatory repricing/rejection |
| Lucky ROI | Probability metrics, CLV, confidence intervals and untouched test period |
| Provider corrections | Immutable versions, manifests, correction replay and reversible settlement |
| Operational price fantasy | Executable timestamped prices, availability/latency stress and no-bet states |
| LLM-introduced errors | LLMs write/review code only; deterministic tests and evaluation decide promotion |
| Scope expansion | Gate 0 contract; new markets/codes require a new validation track |

## Definition of done for V1

V1 is complete when a single command can, for an eligible meeting:

1. Refresh racecards, declarations, history, and odds with provenance.
2. Reject stale, ambiguous, incomplete, or mismatched data.
3. Build the declared as-of race snapshot.
4. Produce calibrated win probabilities summing to one and fair odds.
5. Compare them with the exact cutoff odds without contaminating the pure model.
6. Write an immutable prediction/decision ledger.
7. Later settle results idempotently and calculate CLV.
8. Reproduce the same output from archived inputs and artifacts.
9. Pass unit, replay, contract, walk-forward, and production-path gates.

No real-money deployment is implied by completion of this engineering plan.

