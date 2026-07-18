# Horse Racing Predictor V2 Plan

## Objective

Build a predictor that improves over the current racewise conditional-logit baseline without introducing leakage, overfitting to sparse recent data, or relying on unsupported exotic modeling before the data foundation is stable.

The current baseline is useful: it already beats uniform and official-only probabilities on the validation set, but it still trails the market materially on matched races. That means the next gains should come from better state estimation and richer race context, not from a blind switch to a more complex model.

## What The Research Says

The literature and the current implementation point to the same direction:

- Race prediction works best when modeled as a within-race probability problem, not as a global classification problem.
- The strongest features are usually recent form, opponent strength, time since last run, distance/surface/going fit, weight, draw, class moves, and jockey/trainer context.
- Dynamic ability matters. Horse strength is not static, and time decay should be learned rather than hard-coded.
- Evaluation must be chronological and out-of-sample, with proper scoring rules such as log loss as the primary metric.
- A market blend can help, but only after the pure model is calibrated on strictly out-of-fold predictions.

## Current Diagnosis

The current system is a good skeleton, but the feature engine is still too coarse in a few places:

- Several history windows are hard-coded.
- Suitability features are currently mostly low-dimensional and relative-only.
- Class, going, and draw effects are present, but not yet modeled with enough interaction structure or shrinkage.
- The model is still relying on a small validation slice, which makes nonlinear methods premature.
- The current field size and date coverage are too limited for aggressive model expansion.

## Plan

### 1. Expand the data foundation first

Use a larger chronological backfill before changing the model class.

- Keep a strict point-in-time rule for every feature.
- Build a feature eligibility registry that records:
  - source
  - event timestamp
  - availability at prediction time
  - lookback window
  - missingness policy
  - leakage risk
  - whether the feature is allowed in the pure model, market blend, or both
- Backfill enough history to support stable estimates for:
  - horse ability
  - trainer form
  - jockey form
  - course-distance suitability
  - draw bias
  - going preference
  - class transitions

Practical target:

- Use one warm-up period for state initialization.
- Score on several later chronological seasons or blocks.
- Do not move to nonlinear model families until the backfill is large enough that each major slice has meaningful support.

### 2. Replace hard-coded history with a state engine

Convert the current history feature logic into an explicit state layer.

The state engine should maintain, per entity:

- horse ability
- horse recent form
- trainer form
- jockey form
- trainer-jockey pair effects
- course affinity
- distance affinity
- going affinity
- recency and inactivity
- uncertainty / effective sample size

Recommended design:

- Use multiple decay horizons instead of a single fixed half-life.
- Estimate form over 30, 90, and 365 day windows.
- Treat older evidence as weaker unless there is strong support.
- Return both a point estimate and a confidence measure for each state variable.

### 3. Upgrade the feature set in controlled layers

Add features in a sequence that isolates gains.

#### A. Outcome and form features

Add richer history transforms:

- finish percentile
- beaten distance
- top-3 rate
- top-half rate
- did-not-finish / pulled-up flags
- weighted recent form

Do not rely only on winner/non-winner outcomes. Use field-relative residuals where possible.

#### B. Class and race structure features

Expose fields already available in raw data that are currently underused:

- race class
- rating band
- age band
- sex restriction
- race type / handicap structure
- declared field size
- going changes
- headgear and headgear changes

Derived features to add:

- class rise/drop from prior runs
- handicap adjustment indicators
- weight-for-age residuals
- first-time or changed headgear
- class strength of prior opposition

#### C. Distance, going, and course features

Replace coarse buckets with structured versions:

- continuous distance similarity
- log or spline distance ratios
- course-distance interaction
- going-surface interaction
- course-going interaction
- course-specific draw bias by distance band and field size

#### D. Draw and pace-adjacent structure

Draw should be hierarchical, not global.

- Start with course × distance × surface × field-size shrinkage.
- Add raw draw only after normalizing by race structure.
- Keep a conservative prior unless data support a strong bias.

#### E. Weight and rating features

Weight and official rating should be treated as contextual signals, not isolated causes.

- weight relative to field
- weight change from previous run
- official rating relative to field
- rating change where available
- interaction with handicap status, age, and distance

### 4. Preserve a strong conditional-logit pure model

The first upgraded production candidate should still be a racewise conditional-logit style model.

Why:

- it matches the race structure
- it produces probabilities directly
- it is easy to calibrate
- it is easier to debug than a tree or neural model

What to improve:

- learn decay half-lives instead of hard-coding them
- add hierarchical shrinkage for sparse interactions
- allow a small set of nonlinear transforms or splines for continuous variables
- compare both standardized relative features and some absolute/context features where they matter

This should be the main V2 pure-model candidate.

### 5. Add a dynamic ability challenger

Build a second candidate that explicitly models latent horse ability over time.

Use it to test:

- winner-only likelihood
- top-k or partial ranking likelihood
- full ranking likelihood where supported

This is useful because horse ability drifts over time and repeated observations are not exchangeable.

### 6. Only then test a nonlinear grouped-race model

If the backfill is large enough, add a tree-based grouped-race challenger.

Preferred shape:

- grouped softmax / query-softmax style objective
- race as the group
- strict chronological validation
- calibrated probabilities after training

Do not use a generic ranking objective as the first probabilistic challenger if the goal is calibrated win probabilities.

### 7. Keep the market separate, then consider a blend

Do not inject market information into the pure model.

Instead:

- train the pure model on historical racing data only
- train the market blend only on out-of-fold pure probabilities plus market probabilities
- learn the blend on chronological folds only

If the pure model can close part of the market gap, the blend can become a secondary improvement layer rather than a crutch.

### 8. Standardize validation before adding complexity

Use a walk-forward evaluation scheme.

Recommended evaluation rules:

- one warm-up segment for state initialization
- expanding or rolling chronological folds
- a final untouched lockbox
- day-clustered bootstrap or grouped confidence intervals
- primary metric: log loss
- secondary metrics: Brier score and calibration
- diagnostic only: winner accuracy

Report slices separately for:

- surface
- going
- handicap vs non-handicap
- field size
- course
- distance band
- season

### 9. Define promotion gates before running experiments

Set explicit thresholds before comparing variants.

Suggested gates:

- feature-family ablation must improve log loss by a meaningful margin with confidence
- a full replacement must improve the untouched lockbox, not just an in-fold slice
- no candidate should worsen calibration materially
- no major subgroup should regress beyond a small tolerance

The exact thresholds should be recalculated after the backfill is in place.

## Recommended Experiment Order

1. Freeze the current model as the baseline.
2. Add multi-horizon state features.
3. Add beaten-distance and richer form features.
4. Add class, age-band, sex-restriction, and headgear features.
5. Upgrade distance, going, and course interaction modeling.
6. Add hierarchical draw modeling.
7. Add weight and rating interactions.
8. Train a dynamic-ability challenger.
9. Train a grouped-race nonlinear challenger.
10. Compare the best pure candidates as an ensemble only if they are genuinely independent.
11. Fit a market blend only after the pure model is stable and calibrated.

## Defer For Now

Do not spend effort on these before the core system is stable:

- end-to-end neural models
- transformer-style architectures
- wagering optimization
- complex ensemble stacking without clean ablations
- unsupported source signals without point-in-time provenance
- any feature that cannot be proven available at prediction time

## Decision

The most suitable immediate direction is:

1. strengthen the feature/state engine
2. keep the pure model as a racewise probabilistic model
3. use a dynamic ability challenger next
4. only then test a grouped-race nonlinear model
5. leave market blending as a later calibration layer

That is the most defensible path if the goal is “right first time.”
