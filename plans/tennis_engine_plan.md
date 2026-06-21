# Tennis Prediction Engine — Build Plan

ATP + WTA match outcome predictor. Follows the same `fetch → fit → predict → edge → validate → calibrate` backbone as the golf and soccer engines, wired into the shared suite ledger and the existing app adapter contract.

---

## Architecture

```
tennis/
├── providers.py        # Data abstraction: SackmannProvider (free) + OddsProvider
│                       #   behind a common MatchProvider interface
├── fetch.py            # --seed / --accumulate → matches.csv; --rankings → rankings.csv
│                       #   --draw → draw.csv (upcoming fixtures); --odds → odds.csv
├── model.py            # fit(): surface-split Bradley-Terry skill + H2H nudge +
│                       #   recent-form decay → model_params.json; predict_match()
├── simulate.py         # Set/game-level Monte Carlo for match & tournament draw
├── market.py           # power de-vig, log-odds market blend, CLV tracking
├── calibrate.py        # isotonic per-market calibration maps (nesting guard)
├── edge.py             # price_all(): match/set/games markets, calibrated + blended
├── portfolio.py        # simultaneous-Kelly with per-player correlation cap
├── validate.py         # walk-forward backtest + regression gate
├── engine.py           # command API: schema / predict / simulate / edge
└── data/
    ├── matches.csv            # SOURCE OF TRUTH: one row per completed match
    ├── rankings.csv           # weekly ATP/WTA rankings snapshots
    ├── draw.csv               # upcoming fixtures (player_a, player_b, surface, round)
    ├── model_params.json      # fitted skill / σ / surface / form params
    ├── odds.csv               # match winner odds (player_a, player_b, odds_a, odds_b)
    ├── calibration.json, market_blend.json, odds_history.csv (CLV)
    ├── validation_predictions.csv, validation_baseline.json
    └── predictions.csv, edge_report.csv      # outputs
```

---

## Data

### Source: Jeff Sackmann's public repositories (no API key needed)

- **ATP**: `https://github.com/JeffSackmann/tennis_atp` — `atp_matches_YYYY.csv` files going back to 1968.
- **WTA**: `https://github.com/JeffSackmann/tennis_wta` — `wta_matches_YYYY.csv` files.

Each file has ~50 columns per match including: `tourney_date`, `surface`, `round`, `winner_name`, `loser_name`, `winner_rank`, `loser_rank`, `score`, `best_of`.

`fetch.py --seed 2019 2020 2021 2022 2023 2024 2025` downloads and appends to `matches.csv`. `fetch.py --accumulate` appends new completed results. `fetch.py --draw` pulls upcoming fixtures for the current tournament.

### Canonical `matches.csv` schema

```
date, tourney_id, tourney_name, surface, round, best_of,
winner, loser, winner_rank, loser_rank,
winner_sets, loser_sets, score
```

`surface` is normalised to one of: `hard`, `clay`, `grass`, `carpet`.

---

## Model

### Core: surface-split Bradley-Terry

Each player has a **base skill** and **surface offsets** (clay/grass/carpet relative to hard). Match probability is:

```
logit P(A beats B) = skill_A − skill_B
                   + surface_offset_A[s] − surface_offset_B[s]
                   + form_nudge_A − form_nudge_B
                   + h2h_weight × h2h_log_odds(A, B, s)
```

Parameters are fitted by **penalised logistic regression** (scikit-learn `LogisticRegression(C=..., solver='lbfgs')`) on the binary match outcome matrix, with a **time-decay weight** (half-life ≈ 52 weeks) so recent matches count more.

**Ridge shrinkage** (the `C` hyperparameter) pulls low-sample players toward the mean.

#### Additional features

| Feature | Implementation |
|---|---|
| **Surface offset** | Per-(player, surface) intercept, ridge-shrunk. Players with < 20 surface matches carry reduced weight. |
| **Recent form** | 8-week rolling win-rate residual vs expected — a momentum nudge, not a separate model. |
| **H2H** | Log-odds of historical head-to-head record on the same surface. Shrunk heavily (default weight 0.05) to avoid overfitting small samples. |
| **Ranking (cold start)** | For players with < 10 matches in the dataset, initialise skill from `log(rank) × −0.12` and let the fitter update from there. |
| **Tour flag** | ATP and WTA share the same code but are fitted on separate datasets → separate `model_params.json` files (`atp_model_params.json`, `wta_model_params.json`). |

`fit()` saves `model_params.json` with keys: `skills` (dict[name→float]), `surface_offsets` (dict[name→dict[surface→float]]), `form` (dict[name→float]), `meta` (fit date, n_matches, hyperparams).

`predict_match(player_a, player_b, surface, tour)` returns `{"p_a": float, "p_b": float}`.

---

## Simulation (`simulate.py`)

Two levels:

### 1. Match simulation (set/game level)

Given `p_point_a` — the probability player A wins any given point on serve — the exact set and match probability is computed via a **recursive Markov chain** (not Monte Carlo) for standard scoring:

```
P(win game | p_serve) → P(win set | p_game_serve, p_game_return) → P(win match | p_set, best_of)
```

`p_point_a` is derived from the player's skill rating:

```
p_point_a_serve  = base_serve_rate + skill_delta_adjustment
p_point_a_return = 1 − p_point_b_serve
```

The Markov chain gives an **exact** match win probability without simulation; the Monte Carlo layer is only needed for tournament draw simulation.

### 2. Tournament simulation

Given a `draw.csv` (the full bracket), simulate N tournaments by:
1. For each match in the draw, draw a winner probabilistically using `predict_match()`.
2. Advance winners through the bracket.
3. Accumulate win / finalist / SF / QF reach frequencies.

This gives per-player `win`, `final`, `sf`, `qf` probabilities for outright markets.

Grand Slams use best-of-5 for men; all WTA matches and ATP non-Slam matches use best-of-3. The `draw.csv` `best_of` column controls this per-fixture.

---

## Markets

### Match markets
- **Match winner** — `p_a` / `p_b` from model.
- **Set handicap** (−1.5 / +1.5 sets) — derived from the Markov chain set distribution.
- **Total games** (over/under) — expected total games from the Markov chain.
- **First set winner** — computed from the set-level Markov chain independently.

### Tournament markets (outright)
- **Win tournament**, **Reach final**, **Reach SF**, **Reach QF** — from the draw simulation.
- Calibration and market blend apply identically to the golf engine.

---

## Market (`market.py`)

Inherits logic from the golf `market.py`:

- **Power de-vig**: for complete two-way or three-way boards.
- **Log-odds market blend**: blend model probability toward the sharp market price. Match markets get a heavier market weight than outright markets (books are sharper on singles than futures).
- **CLV tracking**: record model price vs closing line to `odds_history.csv`. Used to audit model sharpness over time.

---

## Calibration (`calibrate.py`)

Same isotonic regression approach as golf, fit separately per market:

- `match_winner` — main binary calibration.
- `set_hcp` — −1.5 / +1.5 set handicap.
- `total_games` — over/under.
- `win_tournament`, `reach_final`, `reach_sf`, `reach_qf` — outright markets.

Nesting guard: ensures `win ≤ final ≤ sf ≤ qf` after calibration.

---

## Edge & Portfolio (`edge.py`, `portfolio.py`)

`price_all()` loops over every row in `odds.csv` / `draw.csv`, prices it via the calibrated + blended model, computes EV per unit, and returns an edge report with columns:

```
player, opponent, surface, market, odds, p_model, p_market, ev_per_unit, stake_gbp, recommended
```

`portfolio.py` applies simultaneous-Kelly with:
- Per-player cap (don't over-expose to one player across multiple markets).
- Total event cap.
- Drawdown brake (reduce stakes if the bank is below the peak by > 15%).

---

## Validation (`validate.py`)

Walk-forward backtest from a configurable start date (default 2023-01-01):

1. For each week, fit the model on all matches **before** that week.
2. Predict all matches **that week**.
3. Accumulate predictions vs outcomes.

Metrics reported per market:
- **Brier skill score** vs naive base rate.
- **Log-loss** vs naive.
- **Calibration curve** (decile reliability).
- **Backtest ROI** at flat stakes and at Kelly stakes (informational only — not tuned on this).

The regression gate (`--gate`) compares against a stored `validation_baseline.json`. The daily `update.sh` will not proceed to edge pricing if Brier skill score regresses by > 5%.

---

## App Integration (`engine.py`)

Same command API pattern as golf:

```python
COMMANDS = {
    "schema":   cmd_schema,    # player list, surfaces, markets, tour selector
    "predict":  cmd_predict,   # head-to-head P(A beats B), per-surface breakdown
    "simulate": cmd_simulate,  # full draw simulation → outright probabilities
    "edge":     cmd_edge,      # calibrated + blended EV across all loaded odds
}
```

A `TennisAdapter` registers in `app/engines/` under id `tennis` with capabilities `predict · simulate · edge`. Settlement grades open bets against completed matches in `matches.csv` automatically.

---

## Implementation Phases

### Phase 1 — Data pipeline
1. `providers.py`: `SackmannProvider` pulls ATP and WTA CSV files from GitHub. Normalises columns to the canonical schema. Handles redirects and file naming differences across years.
2. `fetch.py --seed`: downloads, normalises, and writes `matches.csv` for both tours. `--accumulate`: appends new weeks. `--draw`: parses upcoming draw from a JSON source or manual CSV. `--odds`: loads bookmaker odds into `odds.csv`.
3. Verify `matches.csv` row counts (expect ~100k ATP + ~80k WTA rows from 2000 onward).

### Phase 2 — Model
1. `model.py fit()`: penalised logistic regression with time decay, surface offsets, form, H2H. Validate that fitted skills correlate with historical rankings (Spearman ρ > 0.7 expected).
2. `model.py predict_match()`: returns `p_a`, confidence interval, feature breakdown.
3. Unit-test: Djokovic vs Alcaraz on clay should give Alcaraz a meaningful edge; Djokovic vs Alcaraz on grass should be closer to even.

### Phase 3 — Simulation
1. `simulate.py`: Markov chain for exact match win probability from point-level parameters. Verify against known analytical results (e.g., at `p_serve = 0.64`, best-of-3 win probability ≈ 0.70 for the stronger player).
2. Tournament draw simulation: bracket logic for 128/64/32 draw sizes. Support Grand Slam (5 sets, 128 draw) and ATP 500/250 (3 sets, 32/48 draw) formats.

### Phase 4 — Market, calibrate, edge
1. `market.py`: adapt from golf. Key difference — tennis match odds are two-way, so power de-vig is simpler.
2. `calibrate.py`: fit isotonic maps from walk-forward predictions. Expect match winner calibration to be close out-of-the-box; outright markets will need more correction.
3. `edge.py`: price all markets, output edge report.
4. `portfolio.py`: adapt simultaneous-Kelly from golf. Tennis has correlated exposure (same player in multiple rounds), so the per-player cap is important.

### Phase 5 — Validation & baseline
1. `validate.py --since 2023-01-01 --gate`: walk-forward backtest. Set baseline. Expected match winner Brier skill score: 8–14% above base rate.
2. `update.sh`: daily script — accumulate → fit → validate --gate → recalibrate → simulate (if draw loaded) → edge (if odds loaded).

### Phase 6 — App wiring
1. `TennisAdapter` in `app/engines/tennis.py`. Capabilities: `predict`, `simulate`, `edge`.
2. Settlement: `grade_open_bets()` grades match winner, set handicap, total games, and outright bets against `matches.csv`.
3. Register under id `tennis` in `app/engines/__init__.py`.
4. Add `test_tennis_contract.py` to verify the adapter contract (schema, predict, edge shapes, settlement).

---

## Quick Start (once built)

```bash
# One-time seed
python3 tennis/fetch.py --seed 2020 2021 2022 2023 2024 2025  # ATP + WTA

# Fit and validate
python3 tennis/model.py --fit --tour atp
python3 tennis/model.py --fit --tour wta
python3 tennis/validate.py --since 2023-01-01 --gate
python3 tennis/calibrate.py --fit

# Per-tournament workflow (e.g. Wimbledon)
python3 tennis/fetch.py --draw --tour atp       # load the bracket
python3 tennis/fetch.py --odds --tour atp        # load current book prices
python3 tennis/simulate.py --sims 100000 --tour atp
python3 tennis/edge.py --min-edge 2.0 --tour atp

# Daily refresh
bash tennis/update.sh
```

---

## Key Design Decisions & Rationale

**Bradley-Terry over Elo**: BT is fitted globally in one pass (not sequentially), handles surface offsets cleanly as additional covariates, and integrates naturally with ridge shrinkage. Elo would require sequential updates and surface-specific variants run separately.

**Markov chain over point-level Monte Carlo**: Exact for a single match, far faster, and the only stochastic layer needed is the tournament bracket. This keeps `predict` sub-millisecond.

**Separate ATP/WTA fits**: Women's and men's tennis have different surface dynamics (clay dominance is less pronounced in WTA), different ranking structures, and different tour depths. Sharing a fit would require tour-as-covariate and would muddy the calibration maps.

**Shared market/calibrate/portfolio modules**: The golf versions are sport-agnostic enough to import directly or adapt with minimal changes. This keeps the tennis engine thin.

**Free data first**: Sackmann's repositories provide 20+ years of match history with no key needed — more than enough to fit a robust model. A paid feed (Tennis Abstract API, or Sportradar) can be added later behind the `MatchProvider` interface without touching the model or app layers.
