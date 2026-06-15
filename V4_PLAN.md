# Sports Predictor Suite - V4 Plan

V4 assumes V3 is fully implemented: every engine has a common contract,
security hardening, event-safe settlement, validation gates, market-blend/CLV
tracking, provenance, and clean app workflows. V4 is the modelling-power release.
Its purpose is to improve accuracy and line-making quality by modelling the same
inputs a serious bookmaker would consider before posting, moving, or limiting a
market.

> **Status (2026-06-15): not started.** V4 depends on V3, and V3 is itself not
> yet implemented — `app/engines/contracts.py`, `app/portfolio.py`,
> `cfb/validate.py`, the `event_id`/`market`/`source` ledger columns, and the
> per-engine data manifests referenced below do not exist yet. Do not begin V4
> until the V3 milestones that create them land. The World Cup engine already has
> V3-style validation (`validate.py --gate`), calibration, CLV (`clv.py`) and
> portfolio caps, so for WC much of M1–M2 is hardening rather than greenfield; the
> Club Soccer, CFB, and Golf engines start much closer to zero.

**Concrete V3 prerequisites (build first, in this order):**

- common engine contract + canonical edge row (V3 M1) — V4 feature rows and reason
  codes extend that shape;
- `event_id` and the extended `suite_ledger.csv` columns (V3 M4) — M1's feature
  store and M2's line history both key off `event_id`;
- per-engine validation gates incl. a new `cfb/validate.py` and `validate_all.py`
  (V3 M3) — every V4 acceptance ("beats V3 on a held-out metric") needs these to
  exist to mean anything;
- data manifests / provenance (V3 M7) — M1's `asof`/`fetched_at`/`schema_version`
  reuse that machinery rather than reinventing it.

**UI note:** the explainability and model-audit surfaces in M2/M7 should extend
the existing web app — the dashboard/history/fixtures screens added in the GUI
upgrade (see `GUI_UPGRADE_PLAN.md`) — not a new UI, until V6 replaces the front
end natively.

## 0. V4 Theme

Build a bookmaker-grade line engine for every sport.

V3 makes the suite safer and measurable. V4 should make it smarter. The suite
should stop thinking only in terms of "who is better?" and start producing fair
lines from a richer view of:

- underlying team/player strength;
- injuries, absences, rotations, depth, and replacement value;
- matchup-specific strengths and weaknesses;
- rest, travel, weather, venue, altitude, surface, and course fit;
- schedule congestion, motivation, tournament state, relegation/title pressure,
  and lookahead spots;
- public bias, liquidity, sharp action, and line movement;
- cross-market consistency between moneyline, spread/handicap, totals, props,
  outrights, placements, and matchups.

The architectural shift is to maintain two related models:

1. **Fundamental model** - what should happen on the field/course.
2. **Market model** - how prices move, when the market is informative, and where
   bias or stale pricing can appear.

Recommendations should come only from disagreements where the fundamental model,
market model, calibration, uncertainty, and bankroll discipline agree.

## 1. V4 Guardrails

1. No V4 modelling feature ships by default unless it beats V3 on a held-out
   validation metric and does not worsen CLV materially.
2. Every feature must be point-in-time: no future injury status, closing line,
   lineup, result, or post-event statistic can leak into a historical prediction.
3. Closing lines are a teacher and benchmark, not an oracle. The suite should
   learn from them without blindly copying them.
4. Feature importance must be inspectable. Every recommendation should be able to
   explain the main reasons the fair line moved.
5. More model power must not weaken V3 security, contract, settlement, or
   bankroll safeguards.
6. Features with thin data remain opt-in or report-only until sample size catches
   up.

## 2. Milestones

### M1 - Point-In-Time Feature Store

Goal: create the substrate for honest bookmaker-grade modelling.

Build per-engine feature stores with historical snapshots for:

- team/player ratings at prediction time;
- available lineup/injury/team-news information known at prediction time;
- opening, current, and closing odds with timestamps;
- venue, rest, travel, weather, altitude, surface/course context;
- competition state, schedule congestion, and motivation flags;
- final result and settlement-relevant score.

Acceptance:

- Every feature row has `asof`, `event_id`, `source`, `fetched_at`, and
  `schema_version`.
- Walk-forward validation can rebuild historical feature matrices without
  reading future rows.
- Leakage tests intentionally inject future-only columns and confirm they are
  rejected.

### M2 - Closing-Line Teacher and Market Movement Model

Goal: learn when and how the market improves, and when it can still be beaten.

Build:

- open/current/close line history for each priced market;
- line-movement features: open-to-current, current-to-close, steam moves,
  reversal moves, stale-book divergence, and bookmaker dispersion;
- market confidence by sport, competition, market, time-to-start, and liquidity;
- fit market-blend weights by segment rather than one global value;
- CLV as a first-class validation target.

Acceptance:

- Each engine reports prediction metrics versus closing line, not only versus
  final result.
- Market blend improves held-out log-loss or line error versus pure model and
  pure market in any segment where it becomes default.
- The suite can identify "do not bet" cases where the model edge is likely just
  market information the model has not absorbed.

### M3 - Player-Level Availability and Replacement Value

Goal: price absences like a bookmaker, not as a flat team-strength nudge.

World Cup / Club Soccer:

- expected lineups and lineup confidence;
- player value by position and likely minutes;
- goalkeeper-specific value;
- attacking, defensive, set-piece, and transition contribution splits;
- rotation likelihood from fixture congestion;
- bench/depth replacement value.

CFB:

- QB status, QB quality, and backup drop-off;
- offensive line and defensive front availability;
- position-group injury aggregation;
- transfer portal and returning production;
- coordinator/system continuity;
- depth-adjusted preseason priors.

Golf:

- withdrawal/injury signals;
- round-by-round fatigue or volatility indicators;
- strokes-gained category availability and recency;
- tee-time/weather draw exposure.

Acceptance:

- Absence effects are learned or calibrated against historical examples, not
  hand-tuned only.
- The model reports uncertainty when lineup confidence is low.
- Availability features improve held-out accuracy or CLV before becoming default.

### M4 - Matchup-Specific Models

Goal: price styles and tactical fit, not only average strength.

World Cup / Club Soccer:

- attack style versus defensive weakness;
- pressing/transition exposure;
- set-piece strength and set-piece concessions;
- goalkeeper shot-stopping and cross-claiming where data exists;
- finishing versus chance-creation separation;
- referee/card/penalty tendencies where data is robust.

CFB:

- offensive EPA by play type versus defensive EPA allowed;
- rush/pass split and explosiveness;
- tempo and plays-per-game;
- red-zone efficiency;
- special teams;
- weather impact on passing, kicking, and totals.

Golf:

- course archetype model: distance, accuracy, approach, scrambling, putting
  surface, rough severity, and wind exposure;
- player-course skill translation;
- major/non-major pressure or variance shifts;
- correlated matchup and placement outcomes.

Acceptance:

- Matchup features are measured in held-out tests against a simpler strength-only
  baseline.
- Any tactical feature with weak or unstable signal remains report-only.
- Totals/BTTS/spread improvements are evaluated separately from winner accuracy.

### M5 - Stronger Probabilistic Models

Goal: replace fixed blends where richer models clearly win.

Candidate methods:

- hierarchical Bayesian latent-strength models;
- regularized gradient-boosted or generalized additive models for sport-specific
  residuals;
- dynamic state-space ratings with uncertainty intervals;
- ensemble stacking with out-of-fold predictions only;
- full score/finish distributions rather than only point estimates.

Engine targets:

- World Cup: richer Dixon-Coles/xG hybrid with stage and team-style adjustments.
- Club Soccer: xG-based team strength, league-specific parameters, promoted-team
  priors, and schedule congestion.
- CFB: joint win/spread/total model using EPA, tempo, weather, QB, and market
  priors.
- Golf: player-specific skill/variance, course-fit, weather-wave simulation, and
  correlated finish distributions.

Acceptance:

- Any stronger model beats V3/V4 baseline on held-out metrics after accounting for
  calibration.
- Model uncertainty is surfaced to staking.
- More complex models must fail closed: if features are missing, they fall back
  to the validated simpler model.

### M6 - Cross-Market Consistency

Goal: make fair prices reconcile the way a bookmaker's board must reconcile.

Build:

- soccer score-distribution backbone producing 1X2, Asian handicap, totals,
  BTTS, correct score, and outrights consistently;
- CFB score-distribution backbone producing moneyline, spread, total, team total,
  and derivative markets consistently;
- Golf tournament simulation backbone producing outright, top-N, make-cut,
  matchup, three-ball, and dead-heat-aware placement prices consistently;
- consistency checks that flag impossible or contradictory prices.

Acceptance:

- No recommendation can come from an incoherent market view.
- Cross-market diagnostics show which market is stale or inconsistent.
- Backtests compare isolated-market pricing versus coherent-board pricing.

### M7 - Uncertainty-Aware Staking

Goal: size bets by edge quality, not just edge size.

Build:

- confidence intervals for fair odds;
- model-risk haircut by engine, market, data freshness, and feature coverage;
- CLV history for similar spots;
- liquidity/source reliability haircut;
- reason-code-driven stake adjustments;
- "pass" recommendations when uncertainty overwhelms expected value.

Acceptance:

- Stake sizing is lower for thin data, stale odds, high lineup uncertainty, or
  historically poor CLV segments.
- A recommendation includes fair odds, confidence range, market odds, CLV
  context, and main reason codes.
- Simulated bankroll backtests improve drawdown-adjusted returns versus V3/V4
  staking.

### M8 - Engine-Specific V4 Deliverables

World Cup:

- lineup/availability model;
- tactical matchup layer;
- knockout state and motivation;
- better totals/BTTS calibration;
- live tournament updating from confirmed results and lineups.

Club Soccer:

- xG-first model where data exists;
- league-specific home advantage and goal environment;
- promoted/relegated team priors;
- rest, travel, rotation, and congestion;
- market movement by competition.

CFB:

- EPA-first core model;
- QB/depth chart layer;
- weather and tempo totals model;
- conference, transfer portal, and returning-production priors;
- closing spread and totals calibration.

Golf:

- course archetype model;
- weather-wave simulation;
- strokes-gained category weighting;
- dead-heat/place market handling;
- matchup and three-ball correlation refinement.

Acceptance:

- Every engine has a V4 validation report comparing V3 baseline, V4 fundamentals,
  V4 market model, and final V4 recommendation model.
- Any engine not clearing the default gate remains on V3 defaults with V4
  features available only for analysis.

## 3. Suggested Execution Order

```
M1 point-in-time feature store
  -> M2 closing-line teacher
  -> M3 player availability/replacement value
  -> M4 matchup-specific modelling
  -> M5 stronger probabilistic models
  -> M6 cross-market consistency
  -> M7 uncertainty-aware staking
  -> M8 engine-specific release gates
```

If time-boxed, do M1-M3 first. A leak-free feature store, closing-line teacher,
and replacement-value layer are the highest expected accuracy gains.

## 4. Definition of Done for V4

V4 is done when the suite can produce coherent fair lines from point-in-time
fundamental and market information, explain why those lines differ from the
bookmaker, quantify uncertainty, and demonstrate held-out improvement over V3 in
accuracy, calibration, and CLV without weakening V3's security or bankroll
controls.
