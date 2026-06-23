# Tennis Prediction Engine (ATP + WTA)

Surface-split Bradley-Terry match-outcome model fitted on Jeff Sackmann's free
match archives, with an exact Markov-chain match simulator for set/game
sub-markets and a bracket Monte-Carlo for outright markets. Follows the golf
engine's `fetch → fit → predict → simulate → edge` backbone and registers into
the shared app contract under id `tennis`.

See [`plans/tennis_engine_plan.md`](../plans/tennis_engine_plan.md) for the full design.

## Quick start

```bash
# 1. Seed match history (ATP + WTA, no API key needed)
python3 -m tennis.fetch --seed 2019 2020 2021 2022 2023 2024 2025

# 2. Fit the models (separate ATP / WTA fits)
python3 -m tennis.model --fit --tour atp
python3 -m tennis.model --fit --tour wta

# 3. Per-tournament: load the draw and book prices, then price
python3 -m tennis.fetch --draw-template    # fill in tennis/data/draw.csv
python3 -m tennis.fetch --odds-template    # fill in tennis/data/odds.csv

# Daily refresh (accumulate new results → refit both tours)
bash tennis/update.sh
```

Prediction, simulation, and edge are also driven from the app (Predict /
Simulate / Edge tabs) once the models are fitted.

## Layout

| File | Role |
|---|---|
| `providers.py` | `SackmannProvider` → normalised `matches.csv`; store I/O |
| `fetch.py` | `--seed` / `--accumulate`; `--draw-template` / `--odds-template` |
| `model.py` | surface-split Bradley-Terry fit (ridge logistic, time-decay) + `predict_match` |
| `simulate.py` | Markov chain (game/set/match, tiebreak) + draw / bracket Monte-Carlo |
| `market.py` | two-way & power de-vig, log-odds market blend, CLV tracking |
| `validate.py` | walk-forward backtest (match + outright markets) + regression gate |
| `calibrate.py` | per-market isotonic calibration maps (outright nesting guard) |
| `portfolio.py` | simultaneous-Kelly staking (per-player + total caps, drawdown brake) |
| `engine.py` | command API: `schema` / `predict` / `simulate` / `edge` |
| `data/` | `matches.csv` (source of truth), `*_model_params.json`, `draw.csv`, `odds.csv`, `calibration.json`, `market_blend.json`, `odds_history.csv`, `validation_baseline.json` |

The adapter lives in [`app/engines/tennis.py`](../app/engines/tennis.py);
contract + behaviour tests in [`test_tennis_contract.py`](../test_tennis_contract.py).

## Model

```
logit P(A beats B) = skill_A − skill_B
                   + surface_offset_A[s] − surface_offset_B[s]
                   + form_weight · (form_A − form_B)
                   + h2h_weight · h2h_log_odds(A, B, s)
```

Fitted by penalised (ridge) logistic regression over a sparse design with
time-decay sample weights (≈52-week half-life), solved with scipy L-BFGS — no
scikit-learn dependency. Low-sample players regress toward a rank-based prior
(`skill ≈ −0.12·log(rank)`); surface offsets are kept only above a minimum
sample and shrunk harder. ATP and WTA are fitted separately.

The Markov chain gives **exact** game/set/match probabilities from point-on-serve
rates; `point_edge_for_target` inverts it so set/game sub-markets stay
consistent with the Bradley-Terry match probability. The only stochastic layer
is the tournament bracket.

### Matchup serve base (games markets)

The Markov inversion needs a *serve level* — the average share of points the
server wins, which sets the total-games regime (two big servers hold more → more
games). Previously this was a single constant (`BASE_SERVE = 0.64`) for every
match. `fit()` now also estimates each player's **serve-points-won** and
**return-points-won** rates from the Sackmann/TML point columns
(`w_sv_pts/w_sv_won/...`, EB-shrunk to the surface-specific tour average), and
`model.serve_base()` combines them into a matchup-specific base
(`spw_i − (rpw_j − avg_rpw)`, averaged over both servers; e.g. Isner–Opelka on
hard ≈ 0.72 vs De Minaur–Medvedev ≈ 0.60). The headline match probability stays
pinned to the Bradley-Terry estimate — the base only reshapes the games markets.
Walk-forward (ATP, 2025) the matchup base lifts the **games-per-set** correlation
with actual from **+0.11 to +0.18** with no change to match-winner / set-handicap
/ first-set (those are mathematically independent of the base). When serve stats
are missing (e.g. the WTA MatchCharting feed, or a low-sample player) it falls
back to `BASE_SERVE`, so WTA behaviour is unchanged.

### Total-games level calibration (`games_cal`)

The idealised Markov games model over-predicts the *total* by a stable ~9%
(server-hold and set-count idealisations). `fit()` estimates a multiplicative
correction `games_cal = Σactual / Σpredicted` on the training matches (walk-forward
safe; ATP ≈ 0.90, WTA = 1.0 when no scored point data is available) and
`match_markets(..., games_cal=...)` applies it. Held-out (train < 2025, test 2025)
this takes expected-total-games **bias from +2.5 to 0.0 games** and **MAE from
6.13 to 5.49** (serve base + calibration vs the old fixed base), making the
over/under market priceable.

## Validation & calibration

```bash
python3 -m tennis.validate --since 2023-01-01 --gate              # match markets
python3 -m tennis.validate --since 2023-01-01 --outright --sims 20000  # + outright
python3 -m tennis.calibrate --fit                                 # isotonic maps + OOS
```

`validate.py` refits the model on matches strictly before each retrain date
(default cadence 28 days, no look-ahead), orients every match neutrally (players
sorted by folded name so the calibration set isn't biased toward p≈1), and
scores **match_winner**, **set_hcp** (covers −1.5 sets) and **first_set** against
completed matches. With `--outright` it additionally **reconstructs each
tournament's bracket** from its completed matches (linking each round's winners
back to the matches they won) and Monte-Carlos it to score **win / final / sf /
qf** reach probabilities — round-robin and irregular events are skipped. It
reports Brier / Brier-skill / log-loss and a reliability table, writes
`validation_predictions.csv` (feeds calibration) and a `validation_baseline.json`
for the `--gate` check (headline = match-winner Brier, tolerance 0.005).

`calibrate.py` fits an isotonic map per market from those walk-forward
predictions and reports an honest grouped K-fold (by tournament) out-of-sample
Brier improvement. Match-winner is typically already well-calibrated; the
set/first-set and outright markets gain more. Outright markets carry a nesting
guard (`win ≤ final ≤ sf ≤ qf`). Predict and edge apply calibration by default
(`calibrated: false` to disable).

`market.py` de-vigs book prices (exact two-way for match markets; power de-vig
for many-runner outright boards), blends the model toward the market in log-odds
space (`market_blend.json` weights; `market_blend: false` to disable), and logs
CLV to `odds_history.csv`. `edge` applies the blend, then `portfolio.py` sizes
stakes with simultaneous-Kelly: a per-player cap (one player's correlated
exposure across rounds/markets), a total-card cap, and a drawdown brake.

## Status vs. the plan

Built end-to-end: data pipeline (Phase 1), model (Phase 2), simulation
(Phase 3), market de-vig/blend + CLV, isotonic calibration, simultaneous-Kelly
portfolio, walk-forward validation/gate for **both match and outright markets**
(Phase 4–5), app wiring + settlement (Phase 6).

Remaining polish (follow-ups): a **total_games** over/under market needs a book
line to calibrate against; market-blend weights are sensible defaults, not yet
tuned by a sweep; CLV is logged but not yet surfaced in the suite ledger; and
the app `edge` currently prices match-winner markets (set/first-set/outright
pricing is available in the engine but not yet exposed as separate edge rows).
