# Sports Predictor Suite - V5 Planning Draft

V5 assumes V4 is fully implemented: the suite has bookmaker-grade pre-match
modelling, point-in-time features, market movement learning, player-level
replacement value, matchup models, cross-market consistency, and
uncertainty-aware staking. V5 should move from "strong pre-match line engine" to
"adaptive intelligence system" that learns continuously, supports live markets,
and actively manages portfolio risk across time.

This is an initial planning draft. It should be revisited after V4 validation
reports show which engines and markets actually improved.

> **Status (2026-06-15): planning draft, not started.** V5 assumes V4 — and
> therefore V3 — are complete; neither is yet. The foundations V5 builds on do not
> exist in the codebase today: there is no model registry / versioned artifacts,
> no feature snapshots, no recommendation records (the ledger stores placed bets
> only, not recommendations + model/feature versions), no suite-level
> `clv_suite.py` (only the World Cup `clv.py`), and no `event_id`-keyed data.
> Treat this as a direction document, not an executable backlog, until V4 ships.

**Prerequisites before any V5 milestone:**

- V4 point-in-time feature store + uncertainty-aware staking (V4 M1, M7) — V5's
  model registry and drift detection consume those feature/version records;
- V3 common contract, `event_id`, and extended ledger (V3 M1, M4) — recommendation
  audit and portfolio exposure need stable event/market identity;
- V5 M1 itself must create the `model_versions` / `recommendations` tables; today
  they don't exist, so "every recommendation records model + feature version"
  cannot be satisfied by the current ledger alone.

**UI note:** P5 (Scenario / What-If Line Lab) and P7 (human-review analytics)
should build on the shipped web app — the History and Dashboard screens, sortable
tables, and SVG chart kit added in `GUI_UPGRADE_PLAN.md` — rather than a fresh UI,
at least until the V6 native rewrite.

## 0. V5 Theme

Build a continuously learning betting research and execution-support platform.

V4 answers: "What is the right fair line before the event?"

V5 should answer:

- how should the fair line change as new information arrives?
- when is a price stale enough to act?
- how much confidence should the system place in its own recent performance?
- how should the portfolio adapt to correlated exposure across sports, markets,
  and time?
- which modelling ideas deserve more data collection or research effort?

V5 is not about auto-betting. It is about decision intelligence, live updating,
model governance, and capital allocation.

## 1. V5 Guardrails

1. No automatic real-money execution.
2. Live/in-play outputs are advisory unless explicitly promoted after separate
   live-market validation.
3. Continuous learning must be auditable and reproducible. Every model version,
   feature snapshot, and recommendation must be reconstructable.
4. The system must detect when it is out of distribution and reduce confidence.
5. Portfolio optimization must respect hard risk limits before expected return.
6. V5 cannot relax V3/V4 security, key handling, settlement, or validation gates.

## 2. Candidate V5 Pillars

### P1 - Live and In-Play Updating

Goal: update fair lines during an event using state, time, and live data.

Build:

- event-state models:
  - soccer: score, minute, red cards, substitutions, shot/xG flow if available;
  - CFB: score, clock, possession, down/distance, field position, timeouts;
  - golf: live leaderboard, holes remaining, wave/weather changes, cut line;
- live fair-price calculators derived from the same pre-match model;
- latency-aware stale-price detection;
- live validation using archived state/odds snapshots.

Acceptance:

- Live models are validated separately from pre-match models.
- In-play recommendations include latency, data freshness, and confidence.
- Missing or delayed live state causes a pass, not a stale recommendation.

### P2 - Adaptive Model Governance

Goal: let the suite learn from recent performance without overreacting.

Build:

- model registry with versioned artifacts, feature schemas, and validation
  reports;
- champion/challenger deployment per engine and market;
- drift detection for features, calibration, market behavior, and CLV;
- automatic downgrade to previous champion when drift or regression is detected;
- scheduled recalibration with holdout protection.

Acceptance:

- Every recommendation records model version and feature version.
- A challenger cannot become champion without passing gates.
- Drift alerts explain what changed and which markets are affected.

### P3 - Portfolio Optimization Across Sports

Goal: allocate bankroll like a portfolio, not a list of independent bets.

Build:

- correlation model across teams, outrights, match markets, placements, and
  same-event derivatives;
- expected utility optimization with drawdown constraints;
- exposure by sport, league, team/player, market type, time window, and source;
- scenario stress tests: bad injury read, market shock, correlated underdog day,
  weather miss, model outage;
- stake suggestions with marginal contribution to risk.

Acceptance:

- Portfolio optimizer never exceeds hard caps.
- Recommendations show marginal EV and marginal risk contribution.
- Backtests compare independent Kelly, V4 uncertainty Kelly, and V5 portfolio
  allocation on drawdown-adjusted return.

### P4 - Automated Research Loop

Goal: make the system better at deciding what to test next.

Build:

- research backlog generated from model errors, CLV misses, and drift reports;
- feature ablation reports per engine and market;
- experiment runner with standardized train/validation/test splits;
- model cards for every promoted feature/model;
- "graveyard" for rejected ideas with evidence, so bad ideas do not keep
  returning.

Acceptance:

- Every new modelling proposal has a measurable hypothesis.
- Experiment outputs are comparable across engines.
- Rejected features are documented with metrics and sample-size caveats.

### P5 - Scenario and What-If Line Lab

Goal: let the user stress-test lines like a bookmaker trading desk.

Build:

- what-if controls for injuries, lineups, weather, venue, market movement, and
  event state;
- fair-line sensitivity analysis;
- alternate-book comparison;
- explainable deltas: "QB downgrade moved spread 4.1 points", "red card moved
  live 1X2 by 18 percentage points";
- scenario export for notes/review.

Acceptance:

- Scenario changes never overwrite production features.
- What-if outputs are clearly marked as synthetic.
- Sensitivity calculations are deterministic and reproducible.

### P6 - Data Acquisition and Quality Expansion

Goal: expand modelling power by improving data breadth and reliability.

Possible sources to evaluate:

- soccer xG, lineups, player minutes, cards, substitutions, referee data;
- CFB play-by-play, depth charts, injury reports, weather, recruiting/transfer
  data;
- golf DataGolf-style strokes-gained, tee times, weather, course setup, live
  scoring;
- odds screen style multi-book market data.

Acceptance:

- New data source onboarding includes provenance, schema validation, refresh
  health, cost/rate-limit notes, and fallback behavior.
- Paid or fragile sources are isolated behind provider interfaces.
- Source failures do not break offline operation.

### P7 - Decision Review and Human Feedback

Goal: capture the user's judgement without contaminating model validation.

Build:

- review states: accepted, rejected, watched, manually adjusted;
- reason tags for human overrides;
- post-event review comparing model recommendation, user action, closing line,
  and result;
- separate human-feedback analytics from model-training data unless explicitly
  promoted.

Acceptance:

- Human decisions are auditable but do not leak into model validation by default.
- The suite can report whether human overrides improve or worsen CLV/P&L.
- Feedback tags help prioritize research without rewriting history.

## 3. Suggested V5 Milestones

### M1 - Model Registry and Recommendation Audit Trail

Create versioned model artifacts, feature snapshots, and recommendation records.
This is the foundation for continuous learning and live updating.

### M2 - Drift Detection and Champion/Challenger

Add drift reports and safe model promotion/demotion. V5 should know when a model
has stopped behaving like its validation sample.

### M3 - Portfolio Optimizer

Replace bet-by-bet sizing with cross-sport risk allocation while preserving hard
risk caps.

### M4 - Live/In-Play Prototype

Start with one narrow, high-data market:

- golf live outright/top-N from leaderboard state, or
- CFB live win probability from score/clock/possession, or
- soccer live 1X2 from score/minute/red-card state.

Promote only after archived live snapshots validate.

### M5 - Scenario Line Lab

Expose what-if line movement and sensitivity analysis in the app.

### M6 - Automated Research Loop

Standardize experiments, ablations, model cards, and rejected-feature logs.

### M7 - Data Source Expansion

Add one high-value data source per engine, but only behind provider interfaces
with provenance and offline fallbacks.

### M8 - Human Review Analytics

Track whether user decisions improve or worsen model recommendations without
letting subjective feedback contaminate validation.

## 4. Open Questions For V5

- Which sport should get live modelling first?
- Is the primary V5 objective accuracy, CLV, drawdown-adjusted ROI, or workflow
  speed?
- What paid data sources are acceptable, if any?
- Should V5 remain purely local/offline-first, or can it support optional hosted
  storage for larger odds/live-state histories?
- How much complexity is worth adding before the betting volume justifies it?

## 5. Definition of Done For V5

V5 is done when the suite can track model versions and drift, adapt safely to new
information, support at least one validated live/in-play market, allocate risk
across the full portfolio, and turn post-event results into a structured research
loop without compromising reproducibility, security, or human control.
