# Horse Racing Predictor

Provider-neutral, pre-race win-probability engine for UK and Irish flat racing.
It uses only records available at a declared cutoff, builds time-decayed horse,
trainer, jockey and condition history, and fits a regularized race-level
conditional-logit model. Probabilities are temperature-calibrated coherently so
each race still sums to one.

The engine is usable from canonical CSV inputs today. It deliberately does not
scrape or assume a provider: source licensing, stable IDs, timestamps, and
historical version availability must be verified before a provider adapter is
trusted.

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

## Model

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
