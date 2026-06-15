# Golf Prediction Engine (v2)

PGA Tour + majors betting engine. A **fitted** strokes-gained + variance model
learned from round-by-round history, Monte-Carlo simulated, **calibrated** and
**market-anchored**, validated by a walk-forward backtest, and settled into the
shared suite ledger — the same `fetch → fit → predict → edge → validate →
calibrate` backbone as the `cfb/` and `club_soccer/` engines.

## Architecture

```
golf/
├── providers.py        # Data abstraction: EspnProvider (free) + DataGolfProvider
│                       #   (drop-in upgrade) behind one RoundsProvider interface
├── fetch.py            # --accumulate/--seed → rounds.csv; --espn field; odds
├── model.py            # fit(): time-decayed ridge skill + per-player σ + form +
│                       #   course fit → model_params.json;  predict_field()
├── simulate.py         # 4-round Monte Carlo with cut; joint-sim matchups/3-balls
├── market.py           # power de-vig, log-odds market blend, CLV tracking
├── calibrate.py        # isotonic per-market maps (fit + apply, nesting guard)
├── edge.py             # price_all(): calibrated + blended EV across all markets
├── portfolio.py        # simultaneous-Kelly: per-player + total caps, drawdown brake
├── validate.py         # walk-forward backtest + regression gate (the yardstick)
├── update.sh           # daily: accumulate → fit → validate --gate → recalibrate
└── data/
    ├── rounds.csv             # SOURCE OF TRUTH: one row per player per round
    ├── model_params.json      # fitted skill/σ/form/course params
    ├── validation_predictions.csv, validation_baseline.json
    ├── calibration.json, market_blend.json, odds_history.csv (CLV)
    ├── odds.csv               # outright/place/cut board (name, odds_win, …)
    ├── matchups.csv           # player_a, player_b, odds_a, odds_b
    ├── threeballs.csv         # player_a/b/c, odds_a/b/c
    └── predictions.csv, edge_report.csv      # outputs
```

## Model

Each round is decomposed by time-decayed, ridge-shrunk least squares:

```
score_to_par[player, tournament, round] = mu + difficulty[t,r] − skill[player] + ε
ε ~ Normal(0, σ[player])
```

- **skill** — strokes-gained vs field (higher = better). Ridge shrinks
  low-sample players to the mean; the per-tournament `difficulty` term
  field-strength-adjusts so weak fields and majors are comparable.
- **σ (fitted, per player)** — round-to-round variance from the fit residuals,
  Empirical-Bayes shrunk toward the field σ (~2.85). Drives longshot/outright
  value; majors get a fitted σ bump.
- **form** — short-window (≈6-week) residual nudge; **course fit** — shrunk
  per-(player, course) residual, applied when the course is known.

`predict_field()` turns these into per-player `rating` + `σ` that `simulate.py`
consumes directly. Win / top-N / make-cut come from the simulated finishes;
matchups & 3-balls come from the **same** draws, so they are internally
consistent (missed-cut players ranked behind survivors by 36-hole score).

## Data: free now, DataGolf later

`providers.py` hides the source behind `RoundsProvider`. **EspnProvider** seeds
years of round history from one scoreboard call per season — no key needed.
A DataGolf key (set in the app or `DG_API_KEY`) makes `get_provider()` return
**DataGolfProvider** for richer SG categories and history; nothing else changes.

```bash
python3 fetch.py --seed 2022 2023 2024 2025 2026   # backfill rounds.csv
python3 fetch.py --accumulate                       # append new results (daily)
```

## Calibration, market, staking

- **Calibration** (`calibrate.py`) — isotonic maps per market fitted on the
  walk-forward predictions correct the Monte-Carlo's systematic miscalibration
  (e.g. make-cut pred 0.35 → actual ~0.50), with a nesting guard so
  win ≤ T5 ≤ T10 ≤ T20 ≤ cut.
- **Market** (`market.py`) — power de-vig for complete outright boards
  (favourite-longshot correction), per-line margin for place lines, log-odds
  blend toward the market (sharp longshots lean to market, cut/matchups to
  model), and CLV tracking to `odds_history.csv`.
- **Portfolio** (`portfolio.py`) — simultaneous-Kelly with a per-player
  correlation cap (nested win/T-N/cut/matchup exposure), a total weekly cap, and
  a drawdown brake.

## Quick start

```bash
python3 fetch.py --seed 2022 2023 2024 2025 2026   # one-time history seed
python3 model.py --fit                              # fit → model_params.json
python3 validate.py --since 2024-06-01 --sims 8000  # backtest + set baseline
python3 calibrate.py --fit                          # fit calibration maps
python3 fetch.py --espn                             # current field → field.csv
# add data/odds.csv (+ optional matchups.csv / threeballs.csv), then:
python3 simulate.py --sims 50000
python3 edge.py --min-edge 1.0
bash update.sh                                      # daily refresh (all of the above)
```

## App integration

`GolfAdapter` (capabilities `simulate` · `edge` · `predict`) drives the engine
via `golf_runner.py`. The **Predict** tab gives head-to-head matchup
probabilities; **Edge** prices every market (calibrated + market-blended,
portfolio-staked) and records the recommended bets into the shared
`suite_ledger.csv`. Bets **auto-settle**: `grade_open_bets()` grades win / top-N
/ cut / matchup / 3-ball against the latest completed event in `rounds.csv`.

## Backtest (walk-forward, 2024-06 → 2026-06)

Positive skill on every market vs base rate; make-cut and top-20 ≈ +9–10% Brier
skill, and the model beats a uniform field at picking winners. See
`validate.py` output and `validation_predictions.csv`.
