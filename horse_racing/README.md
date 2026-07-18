# Horse Racing Predictor

Provider-neutral, pre-race win-probability engine for UK and Irish flat racing.
It uses only records available at a declared cutoff, builds time-decayed horse,
trainer, jockey and condition history, and fits a regularized race-level
conditional-logit model. Probabilities are temperature-calibrated coherently so
each race still sums to one.

Two provider paths are implemented:

- `rpscrape` is the selected free research source. A pinned external checkout
  supplies retrospective GB/IE results with Racing Post race, course, horse,
  trainer and jockey IDs. Its unlicensed scraper code is not copied into this
  repository.
- [The Racing API](https://www.theracingapi.com/) remains the supported paid API
  option for authenticated racecards, results and price histories.

Free retrospective acquisition:

```bash
python3 -m horse_racing fetch-rpscrape --start 2026-06-01 --end 2026-07-13 \
  --region both --data-dir horse_racing/data/rpscrape
python3 -m horse_racing validate --data-dir horse_racing/data/rpscrape \
  --min-train 500 --test-size 100
```

The rpscrape checkout is pinned to the reviewed commit and carries a local,
hash-verified bounded-retry safety patch. Raw CSVs are archived by checksum.
Retrospective declarations and inferred result-availability timestamps make the
dataset `research_only`; its starting prices are stored separately and never
treated as cutoff odds or model features.

Paid API acquisition:

Set credentials without putting them in a command or repository, then ingest
into a dedicated directory:

```bash
export THE_RACING_API_USERNAME='...'
export THE_RACING_API_PASSWORD='...'
python3 -m horse_racing fetch --start 2025-07-01 --end 2026-06-30 \
  --data-dir horse_racing/data/theracingapi
python3 -m horse_racing validate --data-dir horse_racing/data/theracingapi \
  --min-train 500 --test-size 100
```

The provider does not expose executable available size, so imported odds are a
market-validation baseline only and cannot generate recommendations. Historical
racecards also lack field-level publication timestamps. The adapter records its
conservative cutoff-pinning and next-day result-availability assumptions in
`provider_manifest.json`; validation on such a backfill is research-grade, not
point-in-time production proof. Live accumulation can remove that limitation.

## Quick start

Create or confirm the four templates:

```bash
python3 -m horse_racing init
```

Populate:

- `data/races.csv` — one canonical row per race.
- `data/runners.csv` — versioned runner declarations with source timestamps.
- `data/results.csv` — complete result versions with publication/update times.
- `data/odds.csv` — timestamped, source-specific win prices.

Fit and validate:

```bash
python3 -m horse_racing fit
python3 -m horse_racing validate --min-train 30 --test-size 15
```

Join a downloaded Betfair Historical BASIC archive to an overlapping canonical
race dataset without extracting the tar:

```bash
python3 -m horse_racing ingest-betfair \
  --archive horse_racing/data1.tar \
  --data-dir horse_racing/data/rpscrape
```

The join requires an exact jurisdiction/date/course/runner-field identity match.
Ambiguous markets, delayed snapshots and boards with a component LTP older than
one hour are quarantined. Repeated calls add disjoint archives, upsert overlapping
races and retain one checksum-keyed manifest per archive; re-ingesting the same
archive is idempotent. A later rpscrape refresh preserves odds for canonical
race/runner pairs that still exist. BASIC last-traded prices are a point-in-time
analytical benchmark only: they have no available-size ladder and can never make
an edge row recommendation-eligible. Use
`--max-component-staleness-seconds` to set a different pre-registered research
threshold.

Price a race and compare it with the last coherent board no later than cutoff:

```bash
python3 -m horse_racing predict RACE_ID
python3 -m horse_racing edge RACE_ID
```

The desktop app registers the engine automatically. Its Predict tab uses a
single race selector; the Edge tab reads `data/odds.csv`.

## Safety contract

- `scheduled_off_utc`, source timestamps, result timestamps, and odds timestamps
  must be timezone-aware ISO-8601 values.
- Jurisdiction is canonical `GB` or `IE`; code is `flat`; accepted surface
  aliases normalize to `turf` or `all_weather`. Other scopes fail closed.
- Race metadata and declarations later than the prediction cutoff are rejected
  or excluded.
- Result versions become historical features only at `result_updated_at` (or
  `result_published_at` for the first version).
- Result corrections are full versions; they trigger a chronological state
  rebuild rather than mutating prior state in place.
- Active runners are matched by stable `runner_id`/`horse_id`, never names.
- Starting/closing price is not a pure-model feature.
- Odds from a different or incomplete field version cannot be recommended.
- The latest state for every runner must still be open at cutoff; suspension or
  closure never falls back to an older quote. Positive `available_size` is
  required for every runner before a board is recommendation-eligible.
- V1 shows analytical edges with zero stakes. Shared-ledger recording remains
  disabled until racing dead-heat settlement can be represented exactly.

## Model (V2)

The V2 feature engine replaces the single-half-life history statistics with an
explicit multi-horizon state layer (30/90/365-day decayed form with effective
sample sizes) and adds layered feature families, each independently switchable
for ablation: `core` (V1 parity), `form_multi` (multi-horizon form, top-3 and
top-half rates, DNF rate, trainer/jockey short-term form, trainer-jockey pair
effects), `class_struct` (class moves, rating and weight changes, handicap
interaction), `suitability` (going affinity, course-distance affinity,
continuous log-distance similarity), `draw_hier` (course x surface x
distance-band draw slopes with hierarchical shrinkage toward surface level and
zero), and `weight_rating` (weight/age x distance interactions).

Every feature is documented in `horse_racing/registry.py` with source, event
timestamp, cutoff availability, lookback, missingness policy, leakage risk and
allowed scope; `verify_registry()` fails the offline tests if the registry and
the live schema disagree. Features the plan wants but the canonical schema
cannot prove available at prediction time (beaten distance, headgear, sex
restriction, ...) are registered as blocked.

Run the feature-family experiment ladder with promotion gates (paired
day-clustered log-loss deltas, slice-regression and calibration tolerances)
plus a decay half-life search:

```bash
python3 -m horse_racing experiments --data-dir horse_racing/data/rpscrape \
  --min-train 500 --test-size 100
```

Reports land in `<data-dir>/experiments_report.{json,md}`. Gate thresholds are
provisional until the multi-season backfill is in place. Walk-forward
validation additionally reports per-slice metrics (surface, going, handicap,
field size, distance band, month, course), supports `--lockbox-frac` for an
untouched final holdout, and uses day-clustered bootstrap confidence intervals.

For each active runner, the feature builder calculates race-relative official
rating, weight, draw, age, recency, time-decayed horse form/win history,
trainer/jockey win effects, and horse surface/distance/course history.

The fitted utility is:

```text
utility_i = X_i beta
p_i = exp(utility_i / T) / sum_j exp(utility_j / T)
```

`beta` is fitted by conditional race likelihood with L2 regularization. `T` is
learned on a later chronological calibration slice. Walk-forward validation
refits on prior races only and reports race-level log loss and Brier score versus
uniform and official-rating baselines.

The implementation plan and release gates are in
`docs/HORSE_RACING_ENGINE_PLAN.md`.

Artifacts record whether the horse-racing tree was dirty when fitted and are
refused at load time if their code hash differs from the running implementation.
