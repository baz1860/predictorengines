# College Football (FBS) Prediction Engine

Sibling of the World Cup engine, adapted for CFB: no draws, point-based scoring, spread-centric betting markets. Predicts win probability, point spread, and total for any FBS matchup.

## How it works

Two models, blended **55% Elo / 45% power** by a frozen nested-season selection:

1. **Elo** (`elo.py`) — margin-of-victory-scaled K over completed games, home-field advantage (~62 Elo), and separate FBS/FCS ledgers. Between seasons FBS ratings regress 30% toward 1500 and FCS ratings toward 850. Target-season talent and returning-production priors are applied only when coverage is adequate; otherwise the snapshot is explicitly `regression_only` and betting is disabled.
2. **Offense/defense power ratings** (`power.py`) — the Dixon-Coles analogue. Per-team offense and defense ratings in points, fitted by weighted ridge regression (exponential time decay, 1.5-season half-life, 4-season window), with fitted home-field advantage and L2 shrinkage. Predicts expected points per side, hence margin *and* total. Separates how teams are strong — e.g. 2025 Ohio State: +8 offense but +14 defense, invisible to a single Elo number.

Win probabilities for spreads/totals come from a normal margin distribution with fitted sigma (~16 pts for margin, similar for totals).

**Totals shrinkage.** Out of sample the additive points model over-disperses the *total* — extreme projected totals regress toward the league mean more than the in-window fit implies. `power.predict` shrinks the total toward the window mean by `TOTAL_SHRINK` (0.80), leaving margin and win probability untouched (`pts1`/`pts2` are recomputed to keep the same margin). Validated by leave-one-season-out over 2019–2025 (≈5,100 FBS games): training folds consistently select k≈0.80 and pooled held-out total MAE improves 13.103 → 13.086; the per-engine gate shows total MAE 13.05 → 12.86 with moneyline Brier and margin MAE unchanged. (Blending the EPA total into the power total was tested over the same window and *rejected* — held-out total MAE preferred 100% power.) Set `total_shrink = 1.0` in the params to disable.

```bash
python3 power.py --fit                    # refit, save data/power_params.json
python3 power.py "Ohio State" "Michigan"  # power-only prediction
python3 power.py --ratings                # offense/defense table
```

After refreshing `data/games.csv`, rerun `power.py --fit`.

## Usage

### The front door (weekly card — same pattern as tennis/golf)

`season.py` is the one entry point for a normal week; everything else below is
plumbing it drives. It prices the upcoming week's FBS slate with the blend,
writes `cfb/data/card.md` with the model's straight-up pick, spread, and total
for every game, the **ATS pick** against each market spread with cover
probability, a total lean, and a value-bets table (edge ≥ 3%, quarter-Kelly).
Each publish also writes `cfb/data/card_manifest.json`, binding the exact card
hash to its model state, configuration, policy, identity registry, odds snapshot,
and frozen validation fingerprints. Operators should follow
[`RUNBOOK.md`](RUNBOOK.md), not edit generated card artifacts.

```bash
bash cfb/update.sh                    # weekly refresh: data + CFBD roster inputs + power refit + gate
python3 -m cfb.fetch_cfbd [year]      # just the CFBD pulls (talent, returning production, schedule)
python3 -m cfb.season --odds-api      # pull NCAAF ml/spread/total lines, price the card
python3 -m cfb.season                 # reprice with whatever is in cfb/odds.csv
python3 -m cfb.season --days 3        # narrower slate window
python3 -m cfb.season --min-edge 0.05 --model elo|power|blend
python3 -m cfb.rehearsal                  # offline Week 0 safety rehearsal
python3 -m cfb.live_evidence --report     # latest paper-signal movement / closing evidence
```

An Odds API card run automatically appends exact bookmaker quotes to
`data/live_quote_history.csv` and locks the first qualifying zero-stake paper
signal per event/market in `data/paper_signal_history.csv`. Repeated unchanged
captures deduplicate. Movement is not labelled closing evidence until kickoff.

Lines come from The Odds API (key `the-odds-api` in `data/api_keys.json`, US
regions) as bookmaker-level, timestamped quotes or manually via
`python3 -m cfb.edge --template` + filling `cfb/odds.csv`. Without lines the
card still shows model picks for every matchup — just no ATS pick or edges.
The card covers every game with an FBS side, including FBS-vs-FCS (FCS teams
carry their own Elo; the power side substitutes the pooled FCS entity).
Preseason, when `data/upcoming.csv` is empty, the slate falls back to
`data/schedule_<year>.json`. All modules run package-qualified from the repo
root (`python3 -m cfb.X`).

### Single-game plumbing

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

Enter **both sides from the same bookmaker**; incomplete pairs are rejected. Spread/total probabilities use the fitted normal champion. A discrete push-aware challenger was evaluated and not promoted. Edges under ~3% are model noise, and no market is recordable unless its policy is `eligible`.

### Bankroll tracking

The application uses the suite-level pooled bankroll and ledger. Displayed CFB stakes and recorded stakes pass through the same event, engine, daily, drawdown, and available-funds caps.

```bash
python3 bankroll.py --settle   # settle open bets against games.csv results
python3 bankroll.py            # status: bankroll, open bets, P&L
python3 bankroll.py --reset 100
```

Settlement handles moneyline, spread (with pushes), and totals from final scores.

<!-- CFB_METRICS_START -->
## Frozen validation evidence

Runtime blend: **60% Elo / 40% power**, selected on 2023-2024 (1,587 games) before the untouched 2025 holdout (807 games).

| Window | Games | ML Brier | Accuracy | Margin MAE | Total MAE |
|---|---:|---:|---:|---:|---:|
| Selection 2023-2024 | 1,587 | 0.18858 | 70.4% | 12.872 | 13.128 |
| Holdout 2025 | 807 | 0.18650 | 70.5% | 12.610 | 12.817 |
| Combined | 2,394 | 0.18788 | 70.4% | 12.784 | 13.023 |

The frozen 2025 closing-line benchmark at a three-point disagreement:

| Market | W-L-P | Bets | ROI | Week-block 95% CI | Policy |
|---|---:|---:|---:|---:|---|
| Spread | 228-233-7 | 468 | -5.5% | [-15.0%, +3.5%] | diagnostic |
| Total | 231-186-1 | 418 | +5.7% | [-0.0%, +11.9%] | paper |

The push-aware calibrated-discrete challenger remains unpromoted. Spread holdout Brier/ECE are 0.51941/0.03432; totals are 0.50548/0.03256. Neither market cleared the held-out betting gate.

The reconstructed recruiting/transfer preseason-prior challenger also remains unpromoted. On 2025 Weeks 1–4 it improved Brier from 0.19880 to 0.18984 (241 games), but its historical inputs are not archived point-in-time snapshots. The transition-team challenger selected 1450 Elo but has only 8 holdout games versus a 30-game gate. Both remain blocked.

Validation line fingerprint: `f8ae47c25a53a006`. Regenerate with `python3 -m cfb.generate_docs --write`; CI-style drift check: `python3 -m cfb.generate_docs --check`.
<!-- CFB_METRICS_END -->

## Data

`data/games.csv` (completed games) and `data/upcoming.csv` (future schedule, in season) from the [sportsdataverse/cfbfastR-data](https://github.com/sportsdataverse/cfbfastR-data) GitHub mirror of CollegeFootballData.com, updated daily in season:

```bash
python3 fetch_data.py   # refresh both, then: python3 power.py --fit
```

Seasons 2001–present, FBS games only (FBS vs FCS included, FCS side pooled). No API key needed.

## Ideas for v2

- Season simulator: conference championships + CFP bracket Monte Carlo → `cfp_odds.csv` (analogue of `simulate.py`)
- Backtest the edge finder against historical closing lines (`betting/` in the same data repo, 2006–present)
- Extra features: returning production, talent composites, QB changes, rest/travel

## V3 tooling

- **Walk-forward gate** — `python3 -m cfb.validate --gate` (leak-free; metrics in
  `data/validation_baseline.json`).
- **Frozen elo/power blend** — the 55/45 runtime weight was selected on 2023–24
  and evaluated on the untouched 2025 holdout; see
  `data/nested_validation_2025.json`.
- **Preseason-prior challenger** — `python3 -m cfb.prior_challenger --fetch --write`
  refreshes the compact recruiting/portal inputs and freezes a nested holdout
  report. It is evidence-only and cannot alter runtime configuration.
- **Experimental market blend** in the app Edge tab (default OFF) anchors the
  model toward the de-vigged book; not used for recommendations until validated.
- **Provenance** — `python3 -m app.provenance --check-odds cfb` validates a
  manual `odds.csv`; freshness shows in the app's model-audit panel.
