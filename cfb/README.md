# College Football (FBS) Prediction Engine

Sibling of the World Cup engine, adapted for CFB: no draws, point-based scoring, spread-centric betting markets. Predicts win probability, point spread, and total for any FBS matchup.

## How it works

Two models, blended 50/50 by default (the blend beats either alone out-of-sample):

1. **Elo** (`elo.py`) — margin-of-victory-scaled K over ~19,800 FBS games (2001–present), home-field advantage (~62 Elo ≈ 2.5 pts), all non-FBS opponents pooled into one self-calibrating 'FCS' pseudo-team. Spread mapping (Elo points per point of margin) and margin sigma fitted from data. Between seasons, ratings regress 35% to the mean and get a **preseason prior** from the 247 talent composite and returning production (`priors.py`, data via CFBD API into `data/cfbd/`; coefficients tuned on weeks 1–4 of 2016–2024 with `priors.py --tune`). This cut the model's early-season deficit to the closing line from 1.8 to 1.1 points; if `data/cfbd/` is absent everything falls back to plain regression.
2. **Offense/defense power ratings** (`power.py`) — the Dixon-Coles analogue. Per-team offense and defense ratings in points, fitted by weighted ridge regression (exponential time decay, 1.5-season half-life, 4-season window), with fitted home-field advantage and L2 shrinkage. Predicts expected points per side, hence margin *and* total. Separates how teams are strong — e.g. 2025 Ohio State: +8 offense but +14 defense, invisible to a single Elo number.

Win probabilities for spreads/totals come from a normal margin distribution with fitted sigma (~16 pts for margin, similar for totals).

```bash
python3 power.py --fit                    # refit, save data/power_params.json
python3 power.py "Ohio State" "Michigan"  # power-only prediction
python3 power.py --ratings                # offense/defense table
```

After refreshing `data/games.csv`, rerun `power.py --fit`.

## Usage

```bash
python3 predictor.py "Ohio State" "Michigan"        # team 1 at home
python3 predictor.py "Georgia" "Texas" --neutral
python3 predictor.py ... --model elo|power|blend    # default blend
python3 predictor.py --backtest [--since 2023]      # walk-forward evaluation
python3 elo.py --ratings                            # top 30 Elo
```

Team names as in `data/games.csv` (e.g. "Ohio State", "Ole Miss", "UTSA").

### EPA model (experimental, not in the default blend)

`epa.py` fits the same opponent-adjusted ridge structure on per-game PPA/play (CFBD `/ppa/games` into `data/cfbd/`) instead of points, calibrated back to points. **Tested and rejected for the blend** (`blend_eval.py`): game-level aggregate EPA underperformed the points model on 2023–24 selection and 2025 validation alike (margin MAE 13.6 vs 13.4; every EPA blend worse than elo+power's 12.61). Game-level PPA averages keep garbage time and turnover noise without the situational filtering that makes SP+/FPI work — making this competitive would need play-level data with garbage-time filters, a much bigger data lift. Kept for reference and ratings tables.

## Projected win totals

```bash
python3 win_totals.py   # -> projected_win_totals_2026.csv
```

Needs `data/schedule_<year>.json` (CFBD `/games`) plus that year's returning/talent files in `data/cfbd/`. Applies preseason carryover + priors to end-of-last-season Elo, blends with power ratings (which carry no roster adjustment — caveat), and computes each team's exact win distribution (Poisson-binomial). Columns include expected wins, quartiles, P(over) at the nearest half-line, and P(6+ wins) for bowl eligibility. FBS newcomers start at the standard new-team rating. Note the model compresses elites toward the mean relative to market win totals — check `nearest_line` vs your book's actual line before reading too much into `p_over_line`.

## Edge finder (moneyline, spreads, totals)

```bash
python3 edge.py --template   # writes odds.csv (upcoming week's fixtures once the season schedule is in data/upcoming.csv)
# fill in lines and decimal odds from your bookmaker, then:
python3 edge.py              # edge report, EV, quarter-Kelly stakes -> edge_report.csv
python3 edge.py --no-bet     # report only, don't log to ledger
```

Enter **both sides of each market** where possible — the vig is then removed exactly; with one side only, a 4.5% overround is assumed. Spread/total cover probabilities use the fitted normal margin model. `odds_sample.csv` shows the format with illustrative odds. Caveats: integer lines can push (the normal approximation slightly misprices these); edges under ~3% are model noise; closing lines at sharp books are hard to beat.

### Bankroll tracking

Same conventions as the soccer engine: live bankroll in `data/bankroll.json` (starts £100), `edge.py` auto-logs recommendations with edge ≥ 3% (best per market per game) to `data/ledger.csv`.

```bash
python3 bankroll.py --settle   # settle open bets against games.csv results
python3 bankroll.py            # status: bankroll, open bets, P&L
python3 bankroll.py --reset 100
```

Settlement handles moneyline, spread (with pushes), and totals from final scores.

## Performance (walk-forward, 2,398 FBS-vs-FBS games, 2023–2025)

Power ratings refit before each week; Elo updated game by game; spread map fitted on pre-2023 data only.

| Model | Accuracy | Brier (binary) | Margin MAE | Total MAE |
|---|---|---|---|---|
| Elo | 70.1% | 0.1895 | 13.10 | — |
| Power | 69.0% | 0.1977 | 13.38 | 13.10 |
| **50/50 blend** | **70.8%** | **0.1885** | **12.79** | 13.10 |

With preseason priors (weeks 1–4 of 2023–24 overlap the prior-tuning window; 2025 is fully out-of-sample). 2025 alone: blend 71.3% accuracy, Brier 0.1868, margin MAE 12.61 — versus 70.7%/0.1903/12.81 before priors. ATS performance did *not* improve (51.2% cover in 2025, was 52.1%): better predictions converged the model toward information the market already priced.

Coin-flip Brier = 0.25; picking the home side every time = 58.6%. For reference, Vegas closing spreads run ~12.0–12.5 MAE, so the model is competitive but the market is still sharper — treat the edge finder accordingly.

### Against the spread (real closing lines)

`ats_backtest.py` bets the blend against consensus closing spreads (median across ~8 books, from the data mirror, 2006–2019 only) whenever model and market disagree by ≥ N points, walk-forward:

```bash
python3 ats_backtest.py                  # 2015-2019
python3 ats_backtest.py --since 2010 --until 2019
```

Result on 2,886 lined games 2015–2019: **47–48% cover, ROI −7% to −12% at closing juice** (break-even = 52.4% at −110). Performance *worsens* as the model/market gap grows — large disagreements are model error, not market inefficiency. Closing spread MAE 12.4 vs model 13.6 on the same games.

2025 season (807 lined games, consensus spreads via CFBD API, −110 juice assumed — CFBD carries no spread odds; import with `import_cfbd_lines.py`): **412-379-16 ATS, 52.1% cover, −0.6% ROI** — essentially break-even, and no betting threshold is statistically distinguishable from a coin flip. Model margin MAE 12.8 vs closing 11.8.

Conclusion: do not bet this model blind against closing spreads; its edge-finder output is only plausibly useful against soft openers, stale lines, or as one input among several.

### Totals (real closing O/U)

`totals_backtest.py` does the same for over/unders (`data/closing_totals.csv`: mirror 2006–2019 with juice, CFBD 2025 at assumed −110). The power model carries a recent-season intercept recalibration (`total_bias`, fitted on the trailing 365 days, walk-forward safe) because scoring drifts faster than the 4-year window adapts.

This is the engine's most competitive market: model totals MAE 12.82 vs market 12.52 in 2025 (a 0.3-pt gap, vs 0.8 on spreads). Results: 2015–2019 **negative** (~49.6% win, −5% ROI); 2025 **positive at every threshold** (54.8% win, +4.6% ROI at ≥3 pts, 408 bets) and robust to bias-correction method. Caveat: one good season on ~800 games is not statistically distinguishable from break-even (≈1 SD), and the older era says otherwise. Status: paper-trade totals through 2026 with CLV tracking before staking real money.

## Data

`data/games.csv` (completed games) and `data/upcoming.csv` (future schedule, in season) from the [sportsdataverse/cfbfastR-data](https://github.com/sportsdataverse/cfbfastR-data) GitHub mirror of CollegeFootballData.com, updated daily in season:

```bash
python3 fetch_data.py   # refresh both, then: python3 power.py --fit
```

Seasons 2001–present, FBS games only (FBS vs FCS included, FCS side pooled). No API key needed.

## Ideas for v2

- Season simulator: conference championships + CFP bracket Monte Carlo → `cfp_odds.csv` (analogue of `simulate.py`)
- Backtest the edge finder against historical closing lines (`betting/` in the same data repo, 2006–present)
- Discrete scoring model for totals/teasers (points come in 3s and 7s — normal approximation misprices key numbers)
- Extra features: returning production, talent composites, QB changes, rest/travel
- Calibration check and shrinkage toward market lines

## V3 tooling

- **Walk-forward gate** — `python3 validate.py --gate` (leak-free; metrics in
  `data/validation_baseline.json`).
- **Tunable elo/power blend weight** — `python3 validate.py --tune-blend` prints
  a before/after table (ml_brier / margin_mae per weight). The default is the V2
  50/50 blend; opt into the validated weight with `--tune-blend --write` (writes
  `data/blend_weight.json`), then `--gate --update-baseline`. See `V3_NOTES.md`.
- **Experimental market blend** in the app Edge tab (default OFF) anchors the
  model toward the de-vigged book; not used for recommendations until validated.
- **Provenance** — `python3 -m app.provenance --check-odds cfb` validates a
  manual `odds.csv`; freshness shows in the app's model-audit panel.
