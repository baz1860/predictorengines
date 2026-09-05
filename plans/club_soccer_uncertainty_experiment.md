# Club Soccer — Model Uncertainty & Variance-Aware Staking

Date: 2026-08-15
Status: **scoped, not started**
Relates to: `plans/club_soccer_engine_plan.md` (§12 promotion gate governs everything here)

---

## 0. Ground rules

Inherited from the engine plan §0 unchanged. Two of them bind hard here and are
called out because they shape the design:

1. **§0.1 — "numpy + pandas + stdlib only. No new pip packages."**
   PyMC and Stan both violate this. Nothing in this document requires them:
   E0 is pure numpy resampling, and E1 is specified as MAP + Laplace, which is
   numpy plus the `scipy.optimize` already vendored in `tennis/model.py` and
   `golf/model.py`. **If E1 promotes and a full sampler later looks worth it,
   that is a separate ground-rule exception the owner must grant explicitly.**
   Do not quietly add a dependency mid-experiment.

2. **§0.5 — point-in-time.** Any posterior used to price a match must be fitted
   only on matches dated strictly before it. `validate.walk_forward` already
   enforces this by refitting once per calendar month on prior data only, so a
   new fitted component inherits the guarantee for free — provided it is fitted
   *inside* `model.fit()` and not loaded from a whole-history artifact. The
   `context_coefficients` experiment was retired for exactly this failure
   ("full-history coefficients could not be evaluated without leakage"). Do not
   repeat it.

3. **§0.4 — gated OFF first.** Every flag below defaults `False` until its gate
   passes. Follow the `OPPONENT_ADJUSTED_XG_DEFAULT` / `LEAGUE_SEED_DEFAULT`
   pattern in `club_soccer/model.py`: the promoted default is a module constant
   with the evidence numbers in a comment above it, read by both production and
   `validate.walk_forward`, so validation always measures what production runs.

---

## 1. Hypothesis

Stated so it can be killed:

> **H1 (accuracy).** Team strengths fitted with partial pooling toward a league
> mean produce better-calibrated 1X2 probabilities than the current independent
> per-team estimates, most visibly for clubs with thin match history.

> **H2 (staking).** The model's *own* uncertainty about `p_model` varies enough
> match to match that sizing stakes against it beats the flat quarter-Kelly in
> `club_soccer/edge.py:377`.

H1 and H2 are separately falsifiable and are gated separately. Bundling them
would mean a failure tells you nothing about which half failed.

## 2. Why this is plausible in *this* codebase

Not a generic argument for Bayesian methods — three specific things already in
the engine are hand-rolled approximations of what a pooled model does properly:

- `XG_EVIDENCE_K = 8.0` (`model.py:698`) smoothly down-weights the `xg`
  component for clubs with little shot data. That is shrinkage-by-hand, applied
  to one component, with a constant picked by eye.
- `LEAGUE_SEED_DEFAULT` seeds each club's Elo from its league strength instead
  of the pooled mean, promoted 2026-07-22 on a real improvement (Brier
  0.61799 → 0.61600). That is a **prior**, and it worked. Partial pooling is
  the same idea applied continuously rather than once at initialisation.
- `XG_RATING_PRIOR = 8.0` with `XG_RATING_ITERATIONS = 12` is a fixed-point
  ratings loop with a prior weight — a shrinkage estimator without a variance.

So the engine has already found, empirically and three times over, that
shrinking under-measured clubs toward a sensible prior helps. H1 is the
generalisation of a result this repo has already banked.

## 3. Prior negative result this must respect

`experiments.json` records **`variance_inflation`** — *"rejected by A/B;
implementation deleted 2026-07-25"*. Naively widening the predictive
distribution did **not** work.

This is not the same proposal (that one inflated variance uniformly; this one
varies it per match by how much data stands behind each club), but it is close
enough that the burden of proof is high. **E0 exists specifically to check
whether the per-match variation is real before anything is built.** If posterior
spread turns out to be near-constant across matches, this collapses into
`variance_inflation` and should be abandoned on those grounds.

---

## 4. Sequencing

Cheapest falsification first. Each stage can kill the ones after it.

| | Experiment | Tests | Cost | Kills |
|---|---|---|---|---|
| **E0** | Bootstrap spread probe | H2 precondition | ~2 days | E2 |
| **E1** | Hierarchical pooled component | H1 | ~1 week | — |
| **E2** | Posterior-variance Kelly | H2 | ~4 days | — |

E1 is worth running on its own merits even if E0 kills E2 — better point
estimates are valuable independent of staking.

---

## 5. E0 — bootstrap spread probe

**Purpose.** Establish, before building anything, whether `p_model` uncertainty
actually varies match to match. No new model, no new dependency, nothing
promoted.

**Status: BUILT** — `club_soccer/bootstrap_probe.py`, tests in
`tests/club_soccer/test_bootstrap_probe.py`, registered as a candidate.
Three details settled differently during implementation than this section
first specified; each is recorded below with its reason.

**Method.** Random-weight (Bayesian) bootstrap, B=120, monthly refit,
point-in-time folds. Each resample draws one Exponential(1) weight per training
match, normalised to mean 1, and passes it to `model.fit(row_weights=)`. Per
match, record the across-resample SD of `p_home`, `p_draw`, `p_away`.

> **B reduced from 200 to 120.** An SD estimated from B resamples is itself
> noisy, and that noise alone inflates read-out 1's ratio — so the probe now
> computes the floor explicitly (`_dispersion_noise_floor`): the p90/p10 the
> read-out would show if every match had identical true spread, from the
> distribution of `sqrt(chi2_{B-1}/(B-1))`. That floor is 1.14 at B=200 and
> 1.18 at B=120, both far below the 1.5 threshold, and a pass must now clear
> the floor as well as the threshold. Read-out 2's power comes from the ~27k
> matches in the window rather than from B. So 200 buys a marginally lower
> floor for two-thirds more compute, and 120 is the better trade.

> **Changed from "resample fixtures with replacement".** A multinomial resample
> can drop every match a thin-history club played, which removes it from
> `params["teams"]` and makes it unpredictable. Those clubs are the population
> this probe exists to measure, so dropping them biases the result toward
> well-measured clubs and *understates* dispersion — a false negative in the
> one direction that would wrongly kill the proposal. Exponential weights are
> proportional to a Dirichlet(1,…,1) draw, so this is still a bootstrap; every
> club survives every resample with only its influence varying. Mean-1
> normalisation preserves effective sample size, which matters because
> `XG_RATING_PRIOR` is measured against accumulated weight.

The `row_weights` hook lands in two places, because the components are fitted
by different machinery: it multiplies the time-decay weights (goals, xG) and
scales the Elo update size (`delta = k * (actual - exp) * bw`). Elo is a
sequential loop that never sees the decay weights, so without the second hook
the probe would leave 40% of the default ensemble identical across every
resample and report a spread that understates the real one. `row_weights=None`
is exactly the production fit — asserted as equality, not tolerance, since both
sites multiply by 1.0.

Window: the `2024-07-01` → `2026-07-01` already pinned in
`promotion_baseline.json`, so the population matches every other club-soccer
measurement.

**Read-out.** Three numbers decide it:

1. **Spread dispersion.** Ratio of 90th to 10th percentile of per-match
   posterior SD, which must clear both 1.5 and the resample-count noise floor
   above. If it does not, uncertainty is effectively constant, H2 is dead, and
   `KELLY_FRACTION = 0.25` is already doing the right thing.
2. **Signal.** Spearman correlation between posterior SD and **excess** Brier
   error — realised Brier minus the Brier the prediction itself implies
   (`sum_k p_k(1 - p_k)`). Requires rho > 0.05 at p < 0.01.

   > **Changed from "realised squared calibration error".** The raw version is
   > confounded: a near-uniform prediction carries a high expected Brier no
   > matter how certain the model is of it, and posterior SD is also largest at
   > mid-range probabilities, so the two correlate through predictive entropy
   > alone with parameter uncertainty contributing nothing. Subtracting the
   > implied Brier leaves only the error the point prediction did not already
   > account for, which is the quantity H2 actually rests on. Both numbers are
   > reported; only the excess one decides.

3. **Thin-club check.** Posterior SD for matches involving low-history clubs vs
   well-measured ones, split at the median training match count. Informational,
   confirming the effect sits where theory says — not a kill criterion.

   > **Changed from the `XG_EVIDENCE_K` split.** That constant measures *shot
   > data* coverage specifically; total match count is the more direct measure
   > of how much evidence stands behind a club's rating.

**Kill criterion.** Fail (1) *or* (2) → stop. Flip the registry entry to
`retired` with the numbers, per the honest-negative pattern. Do not build E2.

### 5.1 Result — COMPLETE, full pinned window

`data/bootstrap_spread_evidence.json`, window `2024-07-01` → `2026-07-01`,
24 months, B = 120, 2,904 fits in 34 min.

**n = 26,969 — identical to `promotion_baseline.json`'s pinned row count**, so
the probe describes exactly the population the promotion gate is measured on,
not an approximation of it.

| read-out | measured | bar | |
|---|---|---|---|
| 1. dispersion (p90/p10 of posterior SD) | **1.68** | ≥ 1.50, and > 1.18 noise floor | pass |
| 2. Spearman(SD, excess Brier) | **+0.0683**, p = 3.0e-29 | > 0.05 at p < 0.01 | pass |
| 3. thin-club SD ratio | 1.11 | informational | — |

Posterior SD quartiles: p10 0.02592, p50 0.03183, p90 0.04347. Raw (confounded)
correlation +0.0752 against the +0.0683 excess figure — the entropy confound was
inflating it by about 10%, and the signal survives its removal.

**Verdict: proceed.** Per-match uncertainty is genuinely uneven, and it predicts
error the point prediction does not already imply. That is the precondition E2
needed, and it is what distinguishes this from the retired `variance_inflation`.

**Robustness.** An 8-month interim run (n = 7,700) gave rho +0.0680, thin-club
ratio 1.11, dispersion 1.76 — against +0.0683, 1.11, 1.68 on the full 24 months.
Two of the three read-outs are stable to three decimal places across a 3.5x
change in sample; dispersion drifted down slightly as expected, since the
interim window's off-season months carry fewer matches. The result is not an
artefact of which months were measured.

**The signal is real but small.** rho = 0.068 at p = 3e-29 is statistically
overwhelming and practically modest. A correlation that size means posterior
variance carries genuine information about where the model errs — not that
variance-aware staking will move ROI much. E0 was built to answer "is there
anything here at all", and the answer is yes. It does not promise E2 will clear
its own bar. Expect a small effect, and hold E2 to the log-growth gate in §7
rather than relaxing it on the strength of this.

**Artifacts.** `data/bootstrap_spread_evidence.json`. Report-only; a test
asserts the module writes nothing else.

```bash
python3 -m club_soccer.bootstrap_probe --write-evidence
python3 -m club_soccer.bootstrap_probe --n-boot 40 --months 6   # quick look
```

> **Changed from `validate --bootstrap-probe`.** `validate.py` owns the
> promotion gate and is already ~850 lines; a report-only research diagnostic
> that promotes nothing belongs beside `decision_time_backtest.py` and
> `market_diagnostics.py` as its own module, which is the established
> convention for this kind of tool.

---

## 6. E1 — hierarchical pooled component

**Status: PROMOTED 2026-08-16.** `HIERARCHICAL_DEFAULT = True`, blend
`{goals 0.0, elo 0.40, xg 0.40, pooled 0.20}`. Post-promotion full-window
validation: accuracy 49.0% → 49.3%, Brier 0.6118 → 0.6110, log-loss
1.0204 → 1.0194, `--gate` PASS. Numbers and caveats in §6.1; the changelog
entry is in `docs/model_improvements_changelog.md`.

**Change surface.** One new ensemble component, `pooled`, alongside
`goals` / `elo` / `xg`.

- `model.py`: add `_lambdas_pooled()` and a `"pooled"` entry in
  `component_matrices()`. Fitted inside `fit()` — never loaded from a
  whole-history file (§0.2).
- `DEFAULT_ENSEMBLE_W` gains `"pooled"` at weight `0.0`. **Weight 0.0 is the
  OFF state** and is exactly price-preserving — verified against the two places
  weights are touched: `_weights_for_match` (`model.py:701`) scales only
  `_SHOT_COMPONENTS` and then renormalises, so a zero weight stays zero through
  `v / total`; and the blend at `model.py:802`,
  `sum(weights[k] * parts[k] for k in ENSEMBLE_COMPONENTS)`, gains a term that
  is identically zero. The card must be byte-identical with the flag off; assert
  it rather than assuming it.
- `HIERARCHICAL_DEFAULT = False` module constant, read by both `predict()` and
  `walk_forward()`.

**Estimator.** Hierarchical Poisson on attack/defence with a per-competition
mean and a shared variance, fitted by MAP with a Laplace approximation for the
covariance. Uses `scipy.optimize.minimize(method="L-BFGS-B", jac=True)` — the
same call already in `tennis/model.py:374`, so the pattern is in-house. No
sampler, no new dependency.

The pooling variance is the one genuinely new parameter and it must be fitted,
not chosen — fit it on training folds only.

**Walk-forward cache.** `walkforward_cache.py` carries the warning *"EVERY fit
option must appear here. An option absent from the cache key means the cache
serves results produced under different settings — a silent wrong answer, not a
slow one."* Add `hierarchical` to `cache_opts` in `walk_forward()` **before**
running any arm, and resolve it to a concrete bool (not `None`) exactly as
`league_seed` is, or the two arms will collide on one cache entry.

**Gate.** §12 items 1–4, unchanged, against `promotion_baseline.json`:

| metric | baseline | tolerance |
|---|---|---|
| `brier` | 0.611777 | 0.0010 |
| `log_loss` | 1.020405 | 0.0015 |
| `brier_ou25` | 0.245201 | 0.0010 |
| `brier_btts` | 0.246944 | 0.0010 |

n = 26,969, window `2024-07-01` → `2026-07-01`, `evaluation_hash`
`2c9396c63ee078449c6f`. Plus §12.3 time-split robustness on `2025-01-01`,
`2025-07-01`, `2025-12-01` — win ≥ 2 of 3, worst regression ≤ 0.0015 Brier.

**Harness.** Copy `validate.opponent_xg_ab()` verbatim as the template. It
already does the thing that matters: it asserts both arms produced identical
evaluation populations (`_evaluation_hash` equality) and raises otherwise. It
also records `code_hash`, `fixture_data_hash` and the reproducing command in the
evidence payload. Match all of it.

**Artifacts.** `data/hierarchical_evidence.json`.

```bash
python3 -m club_soccer.validate --hierarchical-ab \
  --test-from 2024-07-01 --test-to 2026-07-01 --write-evidence
python3 -m club_soccer.validate --gate
```

**On promotion.** Set `HIERARCHICAL_DEFAULT = True` with the evidence numbers in
the comment above it, tune `DEFAULT_ENSEMBLE_W` off zero via the existing
held-out search, refresh `promotion_baseline.json` with `--update-baseline`,
append to `docs/model_improvements_changelog.md`.

### 6.1 Result — gate passed, promoted

Two steps of the promotion recipe above turned out to reference tooling that no
longer exists, and were resolved as follows.

*"Tune `DEFAULT_ENSEMBLE_W` off zero via the existing held-out search."*
`tune_ensemble` was deleted when `ensemble_weight_tuner` was retired on
2026-07-25 — its own nested holdout never promoted it. Rather than rebuild a
retired tuner, the promoted blend is exactly the one the A/B measured. Searching
a blend on the same window the gate scores would be fitting the gate, and
shipping precisely what was validated is the honest option.

*"Refresh `promotion_baseline.json` with `--update-baseline`."* No such flag
exists; `validate.py` never writes that file by design, and the audited
`promote_baseline.py` owns it. Run against the promoted model it **refuses** —
`gate already passes against the current baseline; nothing to re-baseline`.
Its doctrine admits two cases, a changed evaluation population and (with
`--force`) a metric regression, and has no case for ratcheting after a
validated improvement. The baseline was therefore left pinned to the pre-E1
metrics rather than hand-edited around the module that owns it.

**Consequence, worth closing deliberately:** the gate now references metrics the
production model beats by 0.00076, so it tolerates roughly 0.0018 Brier of drift
(E1's gain plus the 0.0010 tolerance) before failing. Options are to teach
`promote_baseline.py` a third, audited case for post-improvement ratcheting, or
to accept the slack. Not decided here.

`data/hierarchical_evidence.json`. Window `2024-07-01` → `2026-07-01`,
n = 26,969, `evaluation_hash` `2c9396c63ee078449c6f` — **identical to
`promotion_baseline.json`**, so both arms priced exactly the pinned population.

Candidate blend `{goals 0.0, elo 0.40, xg 0.40, pooled 0.20}`: a clean
substitution of one goals model for the other, with `elo` and `xg` untouched so
a metric move can only be attributed to the pooled component.

| metric | incumbent | candidate | delta | |
|---|---|---|---|---|
| 1X2 Brier | 0.611777 | 0.611016 | **−0.000762** | §12.1 pass |
| log-loss | 1.020406 | 1.019376 | **−0.001029** | §12.2 pass |
| OU2.5 Brier | 0.245202 | 0.245479 | +0.000277 | within 0.0010 tolerance |
| BTTS Brier | 0.246943 | 0.246875 | −0.000069 | — |

§12.3 time splits — **3 of 3 won**, and the worst split delta is itself a win:

| split | n | incumbent | candidate | delta |
|---|---|---|---|---|
| 2025-01-01 | 21,186 | 0.612197 | 0.611595 | −0.000603 |
| 2025-07-01 | 15,131 | 0.612516 | 0.611750 | −0.000766 |
| 2025-12-01 | 9,190 | 0.612698 | 0.611849 | −0.000849 |

§12.4: `--gate` passes unchanged, because the component carries weight 0.0 —
that check has to be re-run *after* the weight moves, not before.

**What the fitted parameters say.** On the full training set the EM lands at
`sigma_attack = 0.194` and `sigma_defence = 0.150`. Attack genuinely varies
more between clubs than defence does, and the incumbent's single pseudo-count
of 4 applied the same shrinkage to both — it had no way to express the
difference. Fitted `hfa = 0.108` in log space, i.e. a home side scores ~11%
above its neutral rate.

**Honest reading of the size.** −0.00076 Brier is a real improvement measured on
27k matches and repeated across all three time splits, which is what makes it
credible rather than a lucky window. It is also small: roughly a tenth of the
gate's own 0.0010 tolerance band, and about a third of what opponent-adjusted xG
delivered (−0.00123). The direction is consistent everywhere it was measured;
the magnitude is a fraction of a percent of Brier.

**One cost to weigh.** OU2.5 Brier moves the wrong way (+0.000277). It stays
well inside the gate tolerance, and §12.1's primary metric is 1X2, but the
totals market is priced off the same matrix — so a promotion trades a little
totals accuracy for a little more 1X2 accuracy. Worth a deliberate decision
rather than an automatic one.

**Also true:** fitting the component roughly doubles fit time (1.28s → 2.14s on
the full training set), which matters for the 24-fold walk-forward and the A/B,
not for a daily card.

---

## 7. E2 — posterior-variance-aware Kelly

**Status: BUILT, GATED OFF, verdict UNDECIDABLE.** See §7.2. Not a pass and not
a failure — the evidence base cannot currently distinguish this rule from the
flat fraction, and the reason is a hard constraint of the domain rather than
anything about the rule.

**Precondition.** E0 passed. Gated behind E1 only if E1's posterior is the
variance source; if E0's bootstrap spread is used directly, E2 can run standalone.

**Change surface.** `club_soccer/edge.py:377`, currently:

```python
kfrac = KELLY_FRACTION * kelly(p_model, o) * lineup_confidence
```

The proposal replaces the flat `KELLY_FRACTION` with a per-match shrinkage
derived from posterior variance — Kelly integrated over the posterior rather
than evaluated at its mean. Keep `lineup_confidence` untouched; it haircuts a
different, known unknown and is not in scope.

`POSTERIOR_KELLY_DEFAULT = False`. With the flag off, the arithmetic must reduce
**exactly** to `0.25 * kelly(...)`, so a stake-for-stake diff on the current
card is empty. Assert this in a test.

### 7.1 The measurement problem — read before designing the gate

§12.5 asks for simulated ROI vs closing at the 4% threshold. **That measurement
is underpowered for this change and must not be the promotion signal.**

`settlement_ledger.csv` holds 2,100 settled rows. Bet-level ROI has a standard
deviation on the order of 100%+ at typical odds, so the standard error on mean
ROI is roughly 100%/√2100 ≈ **2.2 percentage points**. A staking refinement that
genuinely adds ~1pp of ROI is invisible at that sample size. Promoting on it
would be promoting on noise — and with only ~1,346 rows in `closing_ledger.csv`
the picture is thinner still.

**Primary metric instead: expected log growth**, Kelly's actual objective,
evaluated against devigged closing prices on `closing_market_ledger_v2.csv`
(7,937 market rows — the largest closing-price surface in the engine). For each
forecast with a closing price, compute expected log bankroll growth under each
staking rule using the model's probability, and compare totals. This measures
the quantity Kelly is trying to maximise, on ~4x the sample, without waiting for
settlement.

**Secondary, confirmatory only:** realised ROI at the 2%/4%/6% thresholds from
`decision_time_backtest.py` (`THRESHOLDS`, line 67). Reported in the evidence
payload and required not to *visibly* regress, but explicitly **not** the
promotion signal, and the payload must say so in a `note` field so a future
reader doesn't mistake a noisy positive for evidence.

**Gate for E2.**

1. Expected log growth vs closing strictly improves.
2. §12 items 1–4 unchanged — a staking change must not move model metrics at
   all; if `brier` moves, something leaked into `predict()` and the experiment
   is invalid, not passing.
3. Realised ROI at 4% does not visibly regress (confirmatory).
4. Per-league view from `_threshold_metrics_by_league` shows no league where
   stakes grow while its own CLV is negative. The existing comment there warns
   that a pooled pass can mask leagues with no closing feed — the same trap
   applies here.

**Artifacts.** `data/posterior_kelly_evidence.json`.

```bash
python3 -m club_soccer.decision_time_backtest --posterior-kelly-ab --write-evidence
```

---

## 8. Registry entries

Add to `club_soccer/experiments.json` at kickoff, `status: "candidate"`,
`expires_on` = start + 60 days per `policy_days`. `health.py:258` fails the run
on any candidate past expiry, so these are self-cleaning — three entries:
`bootstrap_spread_probe`, `hierarchical_pooling`, `posterior_kelly`.

Record the originating `git_sha` at kickoff, as every existing entry does.

## 9. Tests to add

Under `tests/club_soccer/`, offline-runnable, no network:

- `test_hierarchical_pooling.py` — OFF state is exactly price-preserving
  (component weight 0.0 leaves the blended matrix bit-identical); pooled
  estimates shrink a synthetic thin-data club toward its league mean and leave a
  data-rich club nearly unmoved; fit is deterministic under a fixed seed.
- `test_posterior_kelly.py` — flag OFF reproduces `0.25 * kelly(p, o)` exactly;
  higher posterior variance never increases a stake; stake stays in `[0, 1]` at
  degenerate inputs (p→0, p→1, odds→1.0).
- Extend `test_walkforward_cache.py` — the new fit option appears in the cache
  key and two arms cannot collide.
- Extend `test_gate_reporting.py` — E2's evidence payload carries its
  "ROI is confirmatory, not primary" note.

Full run before any promotion:

```bash
python3 run_checks.py
```

## 10. Decision tree

```
E0 spread probe
├── dispersion < 1.5 or no error-correlation
│     → H2 dead. Record negative. Consider E1 alone on H1 merits.
└── pass
      ├── E1 hierarchical → §12 gate
      │     ├── pass → promote, re-pin baseline, changelog
      │     └── fail → retire with numbers; E2 may still run off bootstrap variance
      └── E2 posterior Kelly → log-growth gate
            ├── pass → promote staking rule
            └── fail → retire. Quarter-Kelly stands, now with evidence behind it.
```

## 11. Effort and honest expectations

Roughly two and a half weeks of focused work for all three stages, with E0's two
days being the ones that decide whether the rest is worth starting.

Six of the nine experiments in `experiments.json` are retired. That is a healthy
rate for genuine research and the realistic prior here: **most likely outcome is
E1 promotes on H1 and E2 retires on H2.** Both results are worth having. A
retired E2 converts `KELLY_FRACTION = 0.25` from an unexamined default into a
measured choice, which is itself a real gain for a number that scales every
stake the engine places.

### 7.2 Result — undecidable, and why that is the finding

`data/posterior_kelly_evidence.json`, via
`python3 -m club_soccer.posterior_kelly_ab`.

Replay of **1,244 eligible decisions** (288 fixtures, 10 days, posterior SD
available for 100% of them), scored by expected log growth against the devigged
closing price, with posterior SDs refit point-in-time at each decision's own
`train_cutoff`.

| | value |
|---|---|
| log-growth delta at matched stake (**primary**) | **+0.02217** |
| 95% day-block CI | **[−0.01735, +0.07107]** |
| independent blocks | 10 |
| raw unmatched delta | +0.04771 *(confounded)* |
| realised ROI | −0.1360 → −0.1290 *(confirmatory only)* |
| candidate stake before matching | 97.3% of incumbent |

**Verdict: undecidable.** The direction is favourable and the interval spans
zero. On this sample the rule cannot be told apart from a flat quarter-Kelly.

#### Why the plan's own power estimate was optimistic

§7.1 proposed scoring against `closing_market_ledger_v2.csv` on the grounds
that it is "the largest closing-price surface in the engine". That was the
wrong population. Those are closing *snapshots*; a staking rule can only be
scored on *decisions* — a placed bet with an executable price and a model
probability — and `decision_time_backtest.py` is explicit about why that set
cannot be extended backwards:

> A decision-time quote cannot be reconstructed after the fact; it had to be
> recorded before kick-off. So this backtest can only ever cover fixtures that
> were snapshotted while upcoming, and it ACCUMULATES forward.

So the ceiling is not 7,937 rows but the 1,244 decisions across **10 days** that
have accrued since the ledger started on 2026-07-26. Switching the metric from
ROI to log growth removed the binomial noise exactly as intended — that part of
§7.1 held — but it cannot manufacture independent blocks, and blocks are what
the interval is built from.

#### The constraint sits upstream of E2

The engine's own `evidence_gate` still reports the staking gate **CLOSED**:

- 183 bets at 1X2 @2% against `MIN_BETS = 1000`, required at every threshold
- **1** independent block against `MIN_INDEPENDENT_BLOCKS = 8`
- flat ROI −0.087 at 1X2 @2%, not finite-positive

E2 refines how a strategy sizes stakes that the strategy has not yet earned the
right to place. Its evidence requirement cannot reasonably be lighter than the
base strategy's, which is why `_verdict` reads `MIN_INDEPENDENT_BLOCKS`
directly rather than inventing its own bar.

One number in that report is worth watching: **CLV is positive (+0.038 at
1X2 @4%) while flat ROI is negative (−0.091)** on n=140. Beating the close
while losing money is exactly what a real edge looks like before it has enough
samples to show up in results — or what noise looks like. More decision-time
evidence separates those two, and it is the same evidence E2 needs.

#### What was learned that survives regardless

- Expected log growth vs the close is the right metric for a staking change and
  is now implemented; it will be well-powered before ROI is.
- Kelly at the posterior mean is *already* the log-growth optimum — expected log
  growth is linear in `p`, so parameter uncertainty alone does not change the
  optimal fraction. The case for a variance-aware stake rests entirely on
  **selection bias**: a bet is placed when `edge > threshold`, so among noisy
  estimates the selected ones are optimistically biased, and that bias grows
  with variance. This is a sharper motivation than §7 originally carried, and it
  is why the rule shrinks toward the executing book and is clamped to never
  increase a stake.
- The clamp makes the rule one-sided, which means any A/B on a negative-ROI book
  must match total stake or it rewards betting less for the wrong reason.

#### To resolve it

Re-run as evidence accrues; the registry entry expires 2026-10-15, which forces
the question back. Nothing needs rebuilding:

```bash
python3 -m club_soccer.posterior_kelly_ab --write-evidence
```
