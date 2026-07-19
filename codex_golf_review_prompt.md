# Codex Prompt: Adversarial Review of `golf/`

Copy everything below the line into Codex from the repository root.

---

You are performing an independent, adversarial code, data, statistical-model, and betting-system review of the `golf/` module (roughly 10,000 lines of Python plus tests). Your job is to find defects, invalid assumptions, leakage, and operational paths that can lose real money or silently manufacture confidence. Do not summarize the module and do not praise it. Assume ordinary happy paths work; attack the boundaries where plausible-looking output is wrong.

This is a review, not an implementation task. Do not modify production code or data. You may create temporary files outside the repository for experiments. Treat all existing uncommitted changes as user-owned and leave them untouched.

## Read first

Read, in this order:

1. `AGENTS.md`
2. `golf/README.md`
3. Every Python file under `golf/`, including `golf/providers/`
4. `test_golf_app_wrapper.py`, `test_golf_config.py`, `test_golf_free_sources.py`, `test_golf_inplay.py`, and `test_golf_weekly_report.py`
5. The golf paths in the app/engine contract and any root-level tests that exercise golf indirectly
6. The schemas and metadata—not bulk player records—in `golf/data/model_params.json`, `sim_config.json`, `calibration.json`, `validation_baseline.json`, `data_manifest.json`, and `free_source_manifest.json`

Do not assume the README's performance claims or architecture description are true. Verify them against code and artifacts.

## System context

This engine prices PGA Tour events and majors for outright win, top-5/10/20, make-cut, head-to-head, 2-ball, and 3-ball markets, including in-play prices. Real money may be staked from its output.

The main paths are:

- `season.py` / `engine.py`: orchestration and app-facing commands
- `providers/`, `fetch.py`, `refresh.py`, `store.py`, `provider_qa.py`: external data, caching, CSV/SQLite persistence, and freshness checks
- `model.py`: time-decayed ridge skill, tournament-round difficulty, player variance, form, course, public-stat, global-prior, weather, and major adjustments
- `simulate.py`: pre-tournament correlated fat-tailed four-round Monte Carlo, cut logic, placements, matchups, and a separate win draw
- `simulate_inplay.py`: score-conditioned remaining-round simulation
- `calibrate.py`, `market.py`, `edge.py`, `portfolio.py`: probability calibration, de-vigging, market blending, EV, Kelly staking, and exposure controls
- `round_pricer.py`, `tee_times.py`, `providers/odds_manual.py`: manually pasted 2-ball/3-ball boards and player/event matching
- `validate.py`: walk-forward evaluation, hyperparameter tuning, simulation-shape selection, feature ablations, and the regression gate

## Review standard

A valid finding must include:

- severity: Critical, High, Medium, or Low;
- exact `file:line` location(s);
- the smallest relevant code excerpt or a precise description of the expression/path;
- a concrete failure scenario with realistic input/state;
- the observed or logically demonstrated wrong result;
- why existing tests or guards do not catch it;
- the smallest safe remediation and a regression test that would fail before the fix.

Do not file speculative findings merely because a technique is simplistic. Trace the executable path and prove the consequence. For statistical concerns, distinguish a demonstrated implementation bug from a methodological risk requiring new evidence.

## Part 1 — Data integrity and operational correctness

Attack every boundary between providers, persistence, and pricing.

1. **Event identity and staleness**
   - Can current-event data be mixed with a prior event that shares players, course names, dates, or co-sanctioned fields?
   - Trace event IDs/names through ESPN, refresh, SQLite, `field.csv`, live-state files, odds files, pasted round boards, predictions, and reports.
   - Test missing event tags, blank tags, renamed events, timezone boundaries, cached responses, touched/copied files, partial refreshes, and a refresh that fails after overwriting only some artifacts.
   - Check whether mtime-based freshness can mistake old content for a fresh price or reject a legitimately fresh board.

2. **Player identity**
   - Audit all independent normalization/alias implementations. Look for accent, apostrophe, suffix, hyphen, initials, transliteration, duplicate-name, and same-surname collisions.
   - Determine whether unresolved players are rejected, silently assigned a default/20th-percentile skill, blended with an unrelated prior, omitted, or duplicated.
   - Verify matching is symmetric across model parameters, fields, leaderboards, tee sheets, public stats, odds, matchups, and settlement.

3. **Provider and cache failure semantics**
   - Inspect broad exception handlers, fallback providers, cache use, retry behavior, response validation, partial payloads, rate limits, HTML/JSON schema drift, and source-quality thresholds.
   - Find cases where a failed source leaves a successful-looking manifest or stale artifact that downstream code trusts.
   - Check atomicity: interruption during CSV/JSON/SQLite writes must not leave a plausible but truncated state.
   - Verify SQLite and CSV paths cannot diverge and cause different CLI/app results.

4. **Golf result semantics**
   - Test withdrawals before and during an event, disqualifications, did-not-start, incomplete rounds, playoffs, tied positions, MDF/secondary cuts, cut top-N plus ties, no-cut events, limited fields, 54-hole/shortened events, team events, alternate formats, and suspended/postponed rounds.
   - Verify ESPN status inference does not label an active, cut, WD, or DQ player incorrectly.
   - Check that missing rounds are not interpreted as exceptionally good scores, ordinary missed cuts, or zero variance.

5. **Manual odds ingestion**
   - Fuzz pasted 2-ball/3-ball input: extra headers, wrapped lines, American/fractional odds, duplicate players, incomplete groups, ties offered/not offered, each-way text, Unicode whitespace, reordered selections, and a mixture of rounds/events.
   - Check whether group probabilities sum correctly and whether malformed groups are refused rather than partially priced.

6. **Reproducibility and configuration**
   - Identify nondeterminism from RNG use, iteration order, clock/date calls, caches, or mutable data files.
   - Verify the CLI, app wrapper, weekly report, and standalone scripts use the same defaults for simulations, cut rule, seed, course, major flag, bankroll, Kelly multiplier, calibration, and market blend.
   - Flag duplicate/legacy paths whose formulas or defaults have drifted.

## Part 2 — Statistical model validity

Treat every reported validation gain as untrusted until its information set and evaluation protocol are proven.

1. **Temporal leakage**
   - Trace `asof` filtering inside `model.fit()` and every feature it calls. Prove that skill, per-player sigma, tournament-round difficulty, form, course history, public-stat priors, OWGR/global priors, course architecture, weather, and aliases contain only information available before the predicted event.
   - Check whether current files such as `pgatour_stats.csv`, `global_player_priors.csv`, course features, or weather are reused unchanged while backtesting historical events.
   - Look for leakage through tournament dates assigned per round, completed-event fields, final field membership, cut status, or later corrections to historical records.

2. **Walk-forward validity**
   - Audit `validate.walk_forward()` end to end. Confirm model fitting receives only prior rows in effect, not merely in appearance.
   - Check minimum-history selection, excluded small fields, missing players, survivorship, event duplication, course/event grouping, actual-position construction, ties, and shortened/no-cut events.
   - Determine whether the evaluated population matches the population the live system bets.
   - Verify baseline probabilities are legitimate out-of-sample comparators; a global outcome mean computed over the evaluation set is not a deployable forecast.

3. **Hyperparameter and feature-selection overfitting**
   - Inventory every tuned or hand-set number in `model.py`, `simulate.py`, `market.py`, `edge.py`, and `portfolio.py`.
   - Examine repeated reuse of the post-2025 split/full history for model-config screening, win-correlation selection, tail/correlation selection, calibration, ablations, market-blend weights, regression baselines, and README claims.
   - Quantify multiple-comparison and winner's-curse risk. Check the suspicious case where candidate selection, promotion, and headline reporting touch the same events.
   - Note any tuning code bugs, duplicated assignments, fallback-to-full-window behavior, mutable baselines, or promotion gates that allow evaluation data to become training data.

4. **Skill model specification**
   - Verify signs and identifiability in `score_to_par = mu + tournament_round_difficulty - player_skill + error`.
   - Check ridge weighting, time decay, centering constraints, field-strength comparability, disconnected player/event graphs, weak-field events, debutants, promoted/default priors, and numerical conditioning.
   - Determine whether tournament-round fixed effects erase useful signal or create biased skill estimates when fields are non-random.
   - Verify form and course effects are residualized without using the same observations twice or shrinking toward a biased population.
   - Check whether public-stat/global-prior scale alignment is causal, stable, and fit only on historical overlap.

5. **Variance and score distribution**
   - Audit per-player sigma estimation, empirical-Bayes shrinkage, minimum sample behavior, outlier handling, major multipliers, Student-t scaling, weather shifts, and correlation construction.
   - Confirm marginal and joint variances match the comments mathematically.
   - Test extreme sigma, one-player fields, identical players, huge fields, zero/NaN inputs, and nearly deterministic favourites.
   - Assess whether one common round correlation and tail parameter create unrealistic dependence across players or rounds, especially because shared course/weather shocks and individual hot-week effects are different latent processes.

## Part 3 — Simulation and settlement correctness

Use property tests or small deterministic reproductions, not visual inspection alone.

1. For every simulated field, verify finite probabilities in `[0,1]`, win mass, placement counts under ties, nesting (`win <= top5 <= top10 <= top20 <= cut` where applicable), matchup/3-ball mass including ties, and permutation invariance.
2. Determine exactly how tied winners and tied placements are counted. Compare model probability/EV to bookmaker dead-heat settlement; a full-win indicator on a tied finish is not economically equivalent to full payout.
3. Trace cut application and ranking of missed-cut players. Check matchups where both miss the cut, one withdraws, or players complete unequal holes.
4. Verify the separate win draw cannot make displayed win probabilities inconsistent with the displayed top-5 probabilities before or after calibration.
5. Check whether calibration's post-hoc nesting guard distorts marginal calibration or hides an internally inconsistent simulator.
6. Compare pre-tournament and in-play engines at the round-zero boundary. They should agree within Monte Carlo error under identical assumptions.
7. Test in-play state after every round and mid-round: holes completed, `score_thru`, remaining-round variance, cut survivors, completed/partial scores, weather waves, and stale live leaderboards.
8. Determine whether fixed seeds shared across candidate models create useful common random numbers or accidental correlations/bias, and whether live runs without seeds are operationally reproducible.

## Part 4 — Calibration, market pricing, edge, and bankroll

1. **Calibration leakage and sample adequacy**
   - Determine whether `calibration.json` is fitted on the same `validation_predictions.csv` later used to claim calibrated performance or set thresholds.
   - Inspect isotonic fitting for small samples, duplicate x values, endpoints, clipping, interpolation, regime drift, and random-fold diagnostics that ignore event/time clustering.
   - Require event-level temporal cross-fitting for any claimed out-of-sample calibration gain.

2. **De-vigging and market blend**
   - Verify power de-vig on complete and partial outright boards, single-sided top-N/cut margin assumptions, 2-ball/3-ball books, and explicit draw/tie selections.
   - Check `OUTRIGHT_MARGIN` and `LINE_MARGIN` assumptions under missing runners or a nonrepresentative pasted subset.
   - Identify circularity: the model is blended toward the same price against which edge and EV are measured. Establish which probability and which odds are used at each step.
   - Prove market-blend weights are learned without look-ahead and are appropriate by market, event type, odds band, and timestamp.

3. **EV and settlement**
   - Recompute representative bets by hand from raw decimal odds through de-vig, calibration, blend, EV, Kelly, caps, and final stake.
   - Check each-way/place terms, dead heats, pushes/ties, voids, withdrawals, rule-4 deductions, and bookmaker-specific settlement assumptions.
   - Verify the code does not confuse book odds, fair odds, and probability when calculating EV or Kelly.

4. **Portfolio risk**
   - `portfolio.py` describes simultaneous/correlation-aware Kelly. Verify whether it actually optimizes a joint return distribution or merely caps summed independent Kelly stakes.
   - Test nested bets on one golfer, mutually exclusive outrights, multiple selections in one 2-/3-ball, matchup cycles, weather-wave concentration, and many golfers exposed to the same event.
   - Evaluate the 10% per-player and 40% weekly caps, 20% single-bet cap elsewhere, drawdown brake, minimum stake, and bankroll/peak inputs for consistency and empirical support.
   - Look for stake rounding that breaches caps or places economically meaningless bets.

5. **CLV and performance reporting**
   - Verify timestamp/event keys, append behavior, duplicates, timezone handling, and definition of "closing" in `odds_history.csv`.
   - Check whether the final observed snapshot is truly pre-start and whether unavailable/changed markets create survivorship bias.
   - Separate simulated Brier performance from executable betting ROI. Demand realistic odds availability, limits, price movement, dead-heat rules, and transaction timing before accepting profitability claims.

## Part 5 — Security, robustness, and tests

1. Inspect external requests, API-key handling, raw response persistence, path handling, CSV/Excel formula injection risk, SQLite use, and unsafe parsing.
2. Look for denial-of-service or memory blowups from simulation size, field size, malformed payload depth, enormous pasted boards, and repeated provider retries.
3. Map every test assertion to the production behavior it protects. Identify tests that only check shape, existence, or non-crashing while allowing wrong probabilities, wrong event data, or wrong stakes.
4. List the ten highest-risk untested paths. Prioritize cross-module integration and invariant/property tests over one-off happy-path mocks.
5. Run the full golf test set and relevant contract/security tests. At minimum:

   ```bash
   pytest -q \
     test_golf_app_wrapper.py \
     test_golf_config.py \
     test_golf_free_sources.py \
     test_golf_inplay.py \
     test_golf_weekly_report.py
   ```

   Also run relevant root-level engine-contract, provenance, and security tests you discover. Report exact commands and results. Use quick static checks and targeted temporary reproductions where useful.

## Mandatory adversarial scenarios

Explicitly test or trace all of these and state whether each survives:

1. ESPN refresh succeeds for the field but fails for the leaderboard/weather/odds.
2. Yesterday's odds file is copied or touched after today's refresh.
3. Consecutive events share 80% of the field and the board has a blank/wrong event tag.
4. Two distinct golfers normalize to the same folded name.
5. A golfer withdraws after one hole; another is DQ after making the cut.
6. A limited-field no-cut event is run with the default cut rule.
7. An event is shortened to 54 holes after pricing.
8. Three players tie for first and a top-5 market has a boundary tie.
9. Both players in a matchup miss the cut with unequal 36-hole scores.
10. A 3-ball board omits its tie price or contains only two valid selections.
11. A historical walk-forward event uses today's public stats, OWGR/global priors, course file, or weather configuration.
12. Calibration and simulation hyperparameters are selected and reported on the same post-2025 events.
13. Pre-tournament and in-play simulation are compared at zero holes completed.
14. One player receives win probability greater than top-5 probability after the separate win draw, calibration, and market blend.
15. Twenty individually positive-EV bets are all correlated to the same golfer or weather wave.

## Required output

Produce these sections only:

### 1. Findings

Rank findings Critical → High → Medium → Low. Do not dilute severe findings with style nits. If there are no findings at a severity, omit that subsection.

### 2. Most likely ways the system is fooling its operator

Give the top five mechanisms by which reported validation, calibration, CLV, or apparent edge could be optimistic. Tie each to exact code/data evidence.

### 3. Adversarial scenario matrix

For each of the 15 mandatory scenarios: `survives`, `fails`, or `not proven`, with evidence and test command/reproduction.

### 4. Test and tooling results

List exact commands, pass/fail counts, failures, and any limitations. Do not claim a path was tested if you only read it.

### 5. Prioritized remediation

Maximum ten items, ordered by expected reduction in financial/model risk. Separate must-fix correctness defects from experiments needed to validate methodology.

### 6. Residual uncertainty

State what cannot be established from the repository alone, such as historical point-in-time feature snapshots, bookmaker settlement terms, true bet timestamps, and executable prices.

No praise, no filler, no generic best practices, and no restatement of the README. A long list of weak possibilities is worse than a short list of demonstrated defects. When you allege a statistical flaw, show the information leak, invalid comparison, mathematical contradiction, or targeted empirical reproduction.
