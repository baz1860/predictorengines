# CFB Week 0 Readiness Plan

**Created:** 2026-08-01  
**Target:** Week 0, beginning 2026-08-29  
**Current release posture:** **Red for real-money recommendations; amber for paper trading**

## Objective

Make the CFB module operationally trustworthy for Week 0: every prediction must
use an explicit 2026 preseason state, every market quote must identify a real
fixture and executable bookmaker price, and the product must not label or size a
bet unless that market has passed its evidence and risk gates.

The historical model can remain the forecasting baseline. The critical path is
correctness, data integrity, recommendation policy, and production rehearsal—not
adding more model features.

## Current evidence

- The frozen nested validation gate passes on 2,394 FBS-vs-FBS games: Brier
  0.18814, margin MAE 12.783, and total MAE 13.023.
- The current runtime blend is a documented and frozen 55% Elo / 45% power,
  selected on 2023–24 before evaluation on the untouched 2025 holdout.
- Every production entry point now builds an explicit 2026 snapshot. The current
  Week 0 state is `regression_only`, and that state disables recommendation and
  staking.
- `talent_2026.json` and `returning_2026.json` are empty. Live CFBD checks on
  2026-08-01 also returned zero rows for both endpoints.
- `power_params.json` is fitted as of 2026-01-21 and has no roster adjustment.
- The local 2026 FBS schedule is substantially usable: all 888 FBS-involved event
  IDs still exist in the current CFBD response. Five kickoff times have changed.
- The legacy `odds.csv` snapshot has 242 schema/provenance issues and no quote is
  eligible. Production now requires fresh, exact-event, same-book paired quotes.
- Current ATS replay does not support real-money recommendations:
  - 2025 all lined games: 392-393-16, 49.9% cover, -4.6% ROI.
  - 2025 at a 3-point disagreement: 225-233-7, -6.1% ROI.
  - 2023-2025 overall: -5.8% ROI; at 3 points: -6.5% ROI.
- The Week 0 rehearsal card contains eight games, zero eligible bets, and £0 total
  stake. Its manifest binds the exact card to its inputs and risk policy.
- Production-path tests now cover rollover, prior coverage, identity, schedule
  fallback, quote integrity, atomic publication, update failure propagation,
  settlement, generated documentation, and card-manifest verification.

The bullets above are the review baseline. The implementation tracker below is
the current source of truth for remediation progress.

## Implementation tracker — 2026-08-02

| ID | State | Implementation evidence / remaining work |
|---|---|---|
| CFB-01 | **Implemented** | All production entry points use `build_as_of`; the Week 0 smoke run reports season 2026, decision date, prior mode, and snapshot hash. |
| CFB-02 | **Implemented** | FBS and FCS ledgers retain separate anchors; transition and exactly-once rollover tests pass. |
| CFB-03 | **Safety fallback implemented; source blocked** | Coverage is measured explicitly. Current coverage is 0/138, so the model is `regression_only` and all staking/recording is disabled. A usable external 2026 talent/returning source is still required for `full_prior`. |
| CFB-04 | **Implemented** | Required fit/preflight/gate failures propagate non-zero; an atomic JSON run status records running, failure, or success. Fault-injection coverage passes. |
| CFB-05 | **Implemented** | CFB preflight semantically validates schedule season/IDs/kickoffs, power alignment with completed games, the built target-season snapshot, prior coverage, provider keys, odds provenance/freshness, and run status, with separate diagnostic and betting readiness exits. |
| CFB-06 | **Implemented** | Quote use requires CFBD event identity or an exact legacy date/home/away match. The known UNLV/Memphis date mismatch is rejected. |
| CFB-07 | **Implemented** | Odds API snapshots retain event, book, kickoff, and quote time; only complete same-book/two-sided markets are de-vigged. Legacy files remain diagnostic-only. |
| CFB-08 | **Implemented for source/model publishers** | Games, schedule, priors, odds, and power parameters are staged, validated, and atomically replaced. Empty/malformed-response retention tests pass. |
| CFB-09 | **Implemented** | Policy is ML=`diagnostic`, spread=`diagnostic`, total=`paper`; none is currently recordable. |
| CFB-10 | **Implemented** | Card and app recommendation displays call the same suite cap preview used by recording; failure disables the affected stakes. |
| CFB-11 | **Implemented as rejected challenger** | Integer empirical residual PMFs fitted on 2023–24 now produce explicit win/push/loss probabilities. On 2025, discrete probabilities improve Brier/ECE slightly, but their betting ROI intervals cross zero; production remains on the champion. |
| CFB-12 | **Implemented as rejected challenger** | Platt intercept/slope and reliability buckets are fitted strictly pre-holdout. Calibration sharply reduces ECE, but spread’s calibrated-discrete combination fails the incremental ECE gate and neither market clears the held-out betting gate. |
| CFB-13 | **Implemented** | Validation inputs now have source/decision-time metadata, row and season counts, SHA-256 fingerprints, and an atomic manifest. The gate rejected the pre-fingerprint baseline until an explicit reviewed rebaseline. |
| CFB-14 | **Implemented** | Blend selection uses 1,587 games from 2023–24 and locks `w_elo=0.55` before scoring the untouched 807-game 2025 holdout. Week-block confidence intervals and season metrics are frozen in the nested-validation artifact. |
| CFB-15 | **Implemented for Week 0 and Week 1** | CFBD team IDs are canonical; prefix matching was removed. The live through-2026-09-06 audit resolves all 183 current Odds API spellings through reviewed aliases, in-window unknowns block odds publication, and snapshots record the identity-registry hash. |
| CFB-16 | **Implemented** | Settlement requires one team-verified CFBD event ID result, supports postponements, and fails closed for duplicates, reversed sites, or missing scores. The legacy fallback is date-bounded and unambiguous. |
| CFB-17 | **Implemented** | The README evidence section is generated from frozen nested/market artifacts, writes atomically, and has a test/CLI drift check. |

Verification at this checkpoint:

- Repository-wide offline suite: **685 passed**.
- Focused CFB production-safety module: **32 passed**.
- Production smoke card: 2026 `regression_only`, 0/138 complete priors,
  0 recommended bets, 0 staked, stale/legacy quotes diagnostic-only.
- Historical CFB gate remains green for forecast accuracy; the negative ATS
  evidence is unchanged and does not authorize recommendations.
- Nested 2025 holdout at the locked 55/45 blend: ATS ≥3 points was 217-239-7,
  **-9.0% ROI** with 95% CI [-17.6%, -0.4%]. Totals were 231-186-1,
  **+5.7% ROI** with 95% CI [-0.0%, +11.9%], so totals remain paper-only.
- Push-aware/calibration challenger: spread calibrated-discrete Brier 0.51933
  versus normal champion 0.55554, but ECE 0.02642 is worse than calibrated
  normal 0.02522 and its betting ROI interval is [-8.4%, +34.1%]. Totals
  calibrated-discrete Brier is 0.50506, but calibration leaves zero bets at the
  3% EV threshold. Neither challenger is promoted.
- First clean operational rehearsal day: all six controls passed; the 2026
  regression-only golden card contains eight games and £0 stake. Two additional
  clean rehearsal days remain required.

## Principles and scope

1. **Fail closed.** Stale or ambiguous state may still produce a diagnostic
   forecast, but it must not produce a recommended or recordable bet.
2. **One state builder.** Card, app, edge finder, single-game prediction, and win
   totals must consume the same season-aware model snapshot.
3. **Point-in-time evidence.** A betting backtest must use the quote available at
   the declared decision time; closing-line evaluation is not evidence for an
   opener strategy.
4. **Separate forecast quality from bet eligibility.** Passing Brier/MAE regression
   checks does not authorize recommendations in an unprofitable market.
5. **Preserve bookmaker provenance.** A synthetic consensus may be a benchmark,
   but a stake requires a named, timestamped, executable quote.

## Findings mapped to fixes

| ID | Priority | Flaw | Impact | Projected fix | Acceptance evidence | Effort |
|---|---|---|---|---|---|---:|
| CFB-01 | P0 | Weekly predictions stop at end-of-2025 Elo; no explicit 2026 rollover | Week 0 margins and win probabilities use the wrong season state | Implement `build_as_of(season, as_of)` and advance each ledger exactly once before prediction | Tests show 2026 regression is applied once, 2025 replays are unchanged, and every Week 0 prediction reports `model_season=2026` | 2-3 days |
| CFB-02 | P0 | FBS and individual FCS teams use different anchors, but `win_totals.py` rolls every rating toward 1500 | Cross-division win totals and transition-team prices can be corrupted | Preserve ledger/division metadata; regress FBS around 1500 and FCS around 850; define transition/new-team policy | Synthetic FBS, FCS, and transition-team tests; card and win totals share identical pregame ratings | 1-2 days |
| CFB-03 | P0 | 2026 talent and returning-production inputs are empty; power is roster-blind | Entire blend is last-season form at the point where roster churn is largest | Add explicit prior-coverage state and a fallback decision: acquire a point-in-time 2026 source, or run regression-only with wider uncertainty and betting disabled | Readiness report lists coverage by team and feature; missing required priors cannot produce `recommended=true` | 2-4 days, source-dependent |
| CFB-04 | P0 | `update.sh` swallows fetch, fit, and gate failures and exits successfully | Automation can publish a stale/partial card after a failed refresh | Use required/optional step semantics, non-zero exit codes, and a final machine-readable run manifest | Fault-injection tests prove required failures stop the run; last good artifacts remain intact | 1 day |
| CFB-05 | P0 | Preflight checks existence, not semantic readiness, and lists no CFB keys | Empty priors, old model state, or missing provider access can appear ready | Check schedule season, model as-of/season, prior coverage, quote age, API-key availability, and last successful run | `preflight --json` returns `ready=false` with actionable reasons for each injected defect | 1-2 days |
| CFB-06 | P0 | Odds attach by home/away only; `load_market(slate)` ignores date and event ID | A quote can be applied to the wrong fixture | Carry provider event ID and commence time end to end; require exact event mapping with a bounded kickoff-time tolerance | The current UNLV/Memphis date mismatch is rejected or explicitly reconciled; no unmatched quote is silently used | 2 days |
| CFB-07 | P0 | API aggregation discards bookmaker and quote timestamp and creates synthetic modal-line/median-price quotes | Displayed odds may not be executable; cross-book de-vig is incoherent | Store bookmaker-level snapshots; de-vig paired sides from the same book; retain consensus only as a benchmark | Every recommended row has bookmaker, retrieved time, event ID, paired-market status, and executable line/price | 2-3 days |
| CFB-08 | P0 | Odds refreshes and data refreshes overwrite live artifacts before full validation | Empty/schema-drift responses can destroy a usable prior snapshot | Write to temporary files, validate schema/coverage/freshness, then atomically replace; retain last-good snapshot | Empty and malformed provider fixtures leave last-good files untouched and return failure | 1-2 days |
| CFB-09 | P0 | ATS recommendations use a universal 3% edge threshold despite negative current ROI | The UI can promote unvalidated bets | Add a per-market recommendation policy; set ATS and ML to paper-only until a decision-time gate passes | ATS/ML rows may be shown as leans but cannot be recorded as recommended bets | 1 day |
| CFB-10 | P0 | Card stake display bypasses portfolio caps and treats correlated bets independently | Proposed exposure can exceed event, engine, and daily limits | Route displayed and recorded stakes through the same portfolio-sizing function; group ML/spread correlation by event | Golden card respects 15% event, 25% engine, 40% daily caps and labels every capped stake | 1-2 days |
| CFB-11 | P1 | Continuous normal cover model assigns no push mass and ignores football key numbers | Integer spread/total probabilities and EV are mispriced | Build a walk-forward empirical/discrete residual distribution by market; include win/push/loss probabilities in EV | Reliability and ROI tables compare champion normal vs discrete challenger on held-out seasons | 2-4 days |
| CFB-12 | P1 | Runtime cover probabilities use current in-window power sigmas, not held-out blend/total calibration | Large displayed edges can be overconfident | Fit calibration and uncertainty point-in-time; report calibration slope/intercept and reliability buckets | Held-out calibration metrics and maximum-edge sanity checks pass per market | 2-3 days |
| CFB-13 | P1 | Historical line population changed during refresh without baseline invalidation | ROI and README claims became stale while the accuracy gate stayed green | Version line datasets with source, hash, row counts, season counts, and decision-time definition; bind baselines to the fingerprint | Gate refuses comparisons when the line fingerprint differs until an explicit reviewed rebaseline | 1-2 days |
| CFB-14 | P1 | Blend weight and some priors are selected on data also represented in reported validation | Improvement claims are partially selection-biased | Use nested season splits: choose on earlier seasons, lock, then report untouched holdout seasons | A frozen-config 2025 holdout report plus season-by-season intervals is reproducible | 2 days |
| CFB-15 | P1 | Team identity uses prefix matching and prediction errors can be silently skipped | Coverage loss or wrong-team mappings may be hidden | Add canonical team IDs and an alias/review registry; make unmatched/ambiguous identities blocking and visible | 100% reviewed Week 0 provider mappings; zero silent skips | 1-2 days |
| CFB-16 | P1 | Settlement finds the first matching future home/away game rather than the recorded event | Postponements or repeat matchups can settle against the wrong result | Settle by provider/CFBD event ID, with a tightly checked legacy fallback | Tests cover postponed, rescheduled, repeated, and reversed-site fixtures | 1 day |
| CFB-17 | P2 | README describes 50/50 runtime blend, old line coverage, and outdated ATS results | Operators may make decisions from stale claims | Generate the metric section from frozen validation output and document live configuration | README values match checked artifacts and CI detects drift | 1 day |

## Four-week delivery plan

### Week 1: Season correctness and hard safety stop

**Goal:** no code path can price Week 0 with an implicit 2025 state.

1. Implement CFB-01 and CFB-02: the shared, division-aware `build_as_of` model
   snapshot.
2. Route `season.py`, `engine.py`, `edge.py`, `predictor.py`, and `win_totals.py`
   through it.
3. Implement the readiness portion of CFB-03. Record whether predictions are
   `full_prior`, `regression_only`, or `in_season`.
4. Disable `recommended` and staking for any regression-only preseason snapshot.
5. Implement CFB-04 and the essential portions of CFB-05 so failures stop the
   update before any card is published.
6. Add focused unit/integration tests for season rollover, FCS anchoring, new teams,
   empty priors, and failure propagation.

**Week 1 exit criteria**

- Every Week 0 output reports model season, as-of time, prior mode, and data hash.
- The old raw end-of-2025 Week 0 card cannot be produced accidentally.
- A failed fetch, fit, readiness check, or validation gate exits non-zero and does
  not overwrite the last-good card.

### Week 2: Fixture, quote, and bankroll integrity

**Goal:** every priced market is tied to a real event and real bookmaker quote.

1. Implement CFB-06 through CFB-08: event IDs, bookmaker-level snapshots,
   same-book pairing, quote timestamps, atomic writes, and coverage checks.
2. Implement CFB-15 aliases/identity review for all Week 0 and Week 1 teams.
3. Implement CFB-16 event-ID settlement.
4. Implement CFB-10 so the card previews the same capped stakes the suite would
   record.
5. Add quote-age limits by mode: configurable for exploratory preseason markets,
   strict on game day. Never infer freshness from file mtime alone.

**Week 2 exit criteria**

- Every recommended candidate has event ID, kickoff, book, retrieved time, line,
  price, and paired-side provenance.
- No quote is used because names merely happen to match.
- Empty, partial, or malformed provider responses cannot replace last-good data.
- Card exposure respects all portfolio caps.

### Week 3: Betting evidence and recommendation policy

**Goal:** stop translating forecast disagreement directly into an unsupported bet.

1. Implement CFB-09 immediately: ATS and ML remain paper-only.
2. Implement CFB-13 and freeze versioned historical spread/total inputs.
3. Build two explicit backtests:
   - **Closing benchmark:** model quality and closing-line comparison.
   - **Decision-time/opener strategy:** quote available at the stated decision time,
     with subsequent CLV to close.
4. Implement CFB-14 nested season evaluation and confidence intervals.
5. Prototype CFB-11 and CFB-12 as challengers. Promote only if they improve held-out
   calibration without worsening market-level results.
6. Define market-specific statuses and thresholds:
   `disabled`, `diagnostic`, `paper`, or `eligible`.

**Week 3 exit criteria**

- No market is recordable merely because model edge exceeds 3%.
- Results identify dataset fingerprint, seasons, bet count, W-L-P, ROI, confidence
  interval, calibration, and decision time.
- Totals may remain paper-trade eligible; ATS/ML remain diagnostic unless untouched
  holdout evidence supports promotion.

### Week 4: Production rehearsal and go/no-go

**Goal:** prove repeatable operation before the first slate.

1. Run daily refresh/readiness/card dry-runs from a clean environment.
2. Verify the current CFBD schedule and review every kickoff-time change.
3. Produce a golden Week 0 card and independently recalculate a representative
   sample of moneyline, spread, total, de-vig, edge, and capped-stake rows.
4. Exercise provider outage, empty response, malformed row, stale quote, changed
   kickoff, unknown team, postponed game, and validation-regression scenarios.
5. Add alerts for failed runs, stale data, coverage loss, schedule drift, prior-mode
   downgrade, and quote rejection.
6. Complete CFB-17 documentation and an operator runbook.

**Week 4 exit criteria**

- Three consecutive clean rehearsals with no unexplained skips or manual file edits.
- 100% of quoted Week 0 fixtures are matched by event ID and reviewed team identity.
- The golden card is reproducible from its manifest and contains no uncapped stakes.
- A go/no-go review explicitly approves each market independently.

## Go/no-go scorecard

| Area | Go criterion | Current |
|---|---|---|
| Season state | All Week 0 predictions use a declared 2026 snapshot; rollover applied once | **Control complete** |
| Preseason priors | Adequate 2026 coverage, or explicit regression-only downgrade with betting disabled | Control complete; data remains **no-go** at 0/138 |
| Schedule | Current event IDs, kickoff drift reviewed, no unknown Week 0/1 teams | **Control complete through 2026-09-06 (183/183 live provider names)** |
| Odds | Named book, timestamp, exact event, same-book pairing, within freshness limit | **No-go** |
| Refresh | Required failures return non-zero; atomic last-good preservation | Control complete; rehearsals pending |
| ATS/ML evidence | Untouched decision-time holdout supports policy threshold | **No-go** |
| Totals evidence | Frozen backtest plus live paper CLV tracking | Partial; paper only |
| Staking | Display and recording share event/engine/day caps | **Control complete** |
| Settlement | Event-ID based and tested for reschedules/repeats | **Control complete** |
| Operations | Three clean rehearsal days and actionable machine-readable failures | **1/3 clean days; runbook and status controls complete** |

## Resourcing and sequencing

Expected critical-path effort is approximately **15-20 engineering days**, excluding
procurement or manual construction of an external 2026 roster/portal dataset. One
engineer can complete the safety and integrity path in four weeks if model-feature
work is deferred. A second contributor is most useful for independent validation,
fixture/identity review, and the Week 4 golden-card audit.

Work that may proceed in parallel after Week 1:

- Odds/event plumbing and atomic refreshes.
- Historical line fingerprinting and decision-time backtest construction.
- Operator runbook and fault-injection fixtures.

## Explicitly deferred

These are valuable but should not displace release blockers:

- EPA/PPA promotion into win probability.
- Weather and tempo model features.
- Venue-specific home-field advantage.
- CFP or conference simulation.
- Automated depth-chart scraping.

A manual QB/depth-chart status file may be added as a **report-only scenario** if
time permits. It should not alter recordable probabilities until its adjustment is
fitted and evaluated point-in-time.

## Deliverables

1. Season-aware model snapshot and readiness report.
2. Atomic CFB update workflow with non-zero failure behavior.
3. Bookmaker-level odds snapshot schema and event-ID mapping.
4. Versioned closing and decision-time validation datasets.
5. Per-market evidence/policy configuration.
6. Capped, reproducible golden Week 0 card.
7. CFB production-path test suite and operator runbook.
8. Updated README generated from frozen validation outputs.
