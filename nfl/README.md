# NFL Prediction Engine (ATS-focused)

Sibling of the CFB engine, built for picks against the spread. Same
`fetch -> fit -> predict -> edge -> validate` backbone, same bankroll/ledger
conventions. The design difference from CFB: **no normal margin approximation
anywhere a bet is priced.** NFL margins are discrete and pile up on key
numbers (3, 7, 6, 10, 14); cover, push, total, and moneyline probabilities all
come from an empirical discrete margin/total PMF fitted on real history
(`margin_dist.py`).

**Honesty clause.** NFL closing lines are the sharpest lines in sports. Over
seasons 2015-2025 (3,028 lined games), the closing `spread_line` itself has a
MAE of **9.81 points** against the actual margin. This engine's blended
margin prediction comes in at **10.29 MAE** — worse than the market, as
expected for a from-scratch model with no proprietary data. The job here
isn't to beat closers blind; it's to price soft/opening lines correctly and
quantify key-number/half-point value. Every number in this README is from an
actual walk-forward run of this codebase, not an aspiration.

## Data

`fetch_data.py` pulls three free nflverse sources (no API key):

- `data/games.csv` — every game 1999-> with closing spread/total lines,
  moneylines, rest days, div game, roof/surface/weather, QB starters, coaches
  (nflverse/nfldata `games.csv`). Split into completed (`games.csv`) and
  scheduled (`upcoming.csv`), same convention as `cfb/`.
- `data/epa_team_week.csv` — per-team-week offense EPA/play splits
  (nflverse-data `stats_team` release, `stats_team_week_{season}.csv`).
  Defense-allowed splits are derived from the opponent's offensive row in the
  same `game_id` — no raw play-by-play needed.
- `data/qb_week.csv` — per-QB-week EPA/dropback splits (nflverse-data
  `player_stats` release, `stats_player_week_{season}.csv`, filtered to QBs).

Known data gap: the `player_stats` release is missing 2002, 2019, and 2025
entirely (a gap in nflverse's own published assets, not a bug here) — the QB
rolling-value table just carries forward the last known value across those
seasons for any player active across the gap.

Team identity: `team_names.py` folds relocated franchises (OAK->LV, SD->LAC,
STL->LA) onto their current abbreviation so ratings carry a continuous
history, then maps to a stable full name (e.g. "Kansas City Chiefs").

## Models

Three rating models, blended (validated weight: **elo 0.5 / power 0.5 / epa
0.0** — see below):

1. **Elo** (`elo.py`) — 538-style MOV-scaled K=20. The Elo *update* is pure
   team-strength (no HFA/rest baked in); HFA, rest effects, and the
   elo-diff-to-points spread map are fit separately by OLS on 2003-2014 and
   applied at prediction time. HFA is additionally re-estimated on a rolling
   trailing 3-season window so it can decline over time without a hardcoded
   constant (structural/2003-2014 fit: 2.78 pts; current rolling estimate:
   2.36 pts — consistent with the well-known decline in NFL home-field
   advantage).
2. **Power ratings** (`power.py`) — offense/defense ridge regression on
   points, adapted from `cfb/power.py` with a tighter decay (0.8-season
   half-life, 2.5-season window — NFL rosters churn faster than FBS and a
   32-team/17-game league needs a shorter memory).
3. **EPA ratings** (`epa.py`) — same ridge structure on team-week EPA/play
   (pass weighted 1.6x rush), calibrated to points. **Evaluated and NOT
   adopted into the default blend**: a walk-forward grid search over 2015-2020
   picks a small EPA weight (elo 0.4/power 0.5/epa 0.1), but it does not
   confirm on the 2021-2025 holdout (margin MAE 10.354 vs 10.336 for the
   plain 50/50 elo/power default) — the same outcome CFB found for its PPA
   model. `validate.py --tune-blend` reports this honestly each time it runs
   and only adopts a tuned weight when the holdout actually confirms it.
4. **QB adjustment** (`qb.py`) — rolling EPA/dropback, decayed by *cumulative
   dropbacks* (half-life 250 dropbacks) rather than calendar time, shrunk to
   replacement level under 50 career dropbacks. Sniff test on the current
   rolling table: top starters land around +7 to +8 pts vs. league average,
   replacement-level backups around -6 to -8 — same order of magnitude as
   the plan's +4/-4 assumption, a bit wider in practice.

## Margin/total PMF (`margin_dist.py`) — the key-number core

Built from every game 2003-> with a closing line: a kernel-smoothed empirical
histogram of margin conditioned on the closing spread (tri-cube kernel,
bandwidth chosen by 5-fold CV log-likelihood), blended 80/20 with a
*residual*-based prior (the distribution of margin-minus-its-own-line, pooled
across all historical lines and then shifted to the query spread) to
stabilise sparse buckets for big favourites. The residual-shift construction
matters: shifting the raw (non-residual) unconditional PMF instead would
transplant a fixed key-number spike (e.g. the mass at margin=3) to whatever
spread is being queried — that was an actual bug caught during
validation and fixed before this shipped.

Sanity anchors (all pass, `python3 test_nfl_pmf.py`):
unconditional P(|margin|=3) and P(|margin|=7) fall in the expected 12-18%/
7-12% bands; key numbers (3,6,7,10,14) all exceed neighbouring non-key
values; P(margin=0) < 0.5%; at a 3-point home favourite, the gap between
covering +2.5 and +3.5 (i.e. the mass sitting exactly on 3) is 7-13%.

Known soft spot: the fitted PMF's push-rate calibration slightly
*under*-predicts actual push rates at some key numbers when queried at the
*model's* (noisier) predicted margin rather than the market's precise line
— e.g. at |line|=10 the PMF predicts a 3.7% push rate against an actual
7.5% over 2015-2025. Documented, not hidden; a fitted convolution widening
(`margin_dist.widen`, `inflation_k`) is wired up as a lever if this needs
tightening later.

## Validation (`validate.py`, `test_nfl_pmf.py`, `ats_backtest.py`, `totals_backtest.py`)

Walk-forward from 2015 week 1 (3,028 lined games through 2025): power/EPA
refit before each week, Elo updated game-by-game with pregame ratings only,
QB values from strictly prior appearances, spread map/HFA/rest/PMF fit no
later than 2014.

| Metric | This engine | Reference |
|---|---|---|
| Margin MAE | **10.29** | closing spread MAE 9.81; plan target <= 11.3 |
| Total MAE | **10.78** | closing total MAE 10.44 |
| ML Brier | **0.2245** | home-always ~0.24-0.25; closing ~0.205; plan target <= 0.215 (not quite cleared) |
| Cover-prob reliability | mean abs calibration gap 0.092 over 10 deciles | — |
| ATS vs closers (`ats_backtest.py`) | 49-52% cover across disagreement thresholds, ROI mostly negative to flat, occasional +1% at 6+ pt disagreement on a 350-bet sample (not proven) | break-even 52.4% at -110 |
| Totals vs closers (`totals_backtest.py`) | 48-51% at small disagreement, **degrading to 41% at 6+ pt disagreement** (-20.5% ROI) — large gaps are model error, same finding as CFB | break-even 52.4% |

`python3 -m nfl.validate --gate` checks margin MAE and ML Brier against
`data/validation_baseline.json` (tolerance 0.5 pts / 0.010 Brier); ROI is
reported, not gated (too noisy). Both backtests show the same pattern CFB
found: betting into large model-vs-market disagreements does NOT perform
better, and totals get worse as the gap grows — a sign the disagreement is
model error, not insight the market missed. Report this honestly rather than
cherry-picking a threshold that looks good in one run.

## Edge finder (`edge.py`, `bankroll.py`)

```bash
python3 -m nfl.edge --template          # writes odds.csv from upcoming games
# fill in blank lines/decimal odds (both sides where possible), then:
python3 -m nfl.edge                     # edge report -> edge_report.csv, logs bets >= 3% edge
python3 -m nfl.edge --no-bet --half-points
python3 -m nfl.bankroll --settle        # grade vs games.csv once results land
```

**Sign convention**: `odds.csv` uses the *traditional* bookmaker spread
convention (negative = home favorite, matching a sportsbook screen and
`cfb/odds.csv`) — internally the engine (and nflverse's own `spread_line`)
uses the opposite convention (positive = home favorite). The conversion
happens once, at the CSV boundary (`edge._internal_spread_line`); don't mix
the two when reading code across files.

Kelly staking is on the **win/push/loss trinomial**, not the textbook binary
formula — pushes at 3 and 7 are common enough to move the optimal stake.
`--half-points` prints the fair price of the adjacent half-point lines for
every spread bet, straight from the PMF — the practical payoff of the
key-number work (buying/selling a half point across 3 or 7 is worth
materially more than across most other numbers).

## Quick start

```bash
python3 -m nfl.fetch_data
python3 -m nfl.margin_dist --fit
python3 -m nfl.elo --fit && python3 -m nfl.power --fit && python3 -m nfl.epa --fit && python3 -m nfl.qb --fit
python3 -m nfl.validate --tune-blend --write
python3 -m nfl.validate --gate --update-baseline
python3 -m nfl.predictor "Eagles" "Cowboys"
python3 -m nfl.edge --template && python3 -m nfl.edge --min-edge 3
python3 -m nfl.bankroll --settle
```

## Non-goals (v1)

Live/in-game pricing, teasers/parlays (the PMF makes teaser EV a natural v2
— pricing 6-pt teasers through 3 and 7 is the classic key-number
application), playoff simulator, weather model (roof/temp/wind columns exist
in `data/games.csv`, left for v2), player props.
