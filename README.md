# World Cup 2026 Prediction Engine

Predicts match outcomes (win/draw/loss probabilities and likely scorelines) using an Elo rating system feeding a Poisson goal model with a Dixon-Coles draw adjustment.

## How it works

Two match models, blended 50/50 by default (the blend beats either alone out-of-sample):

1. **Elo + Poisson** (`predictor.py`) — Elo ratings over ~49,000 international matches (1872–present), K scaled by tournament importance and goal margin; expected goals from a Poisson regression on Elo difference; fixed Dixon-Coles draw adjustment (rho = -0.10).
2. **Dixon-Coles attack/defense** (`dixoncoles.py`) — per-team attack and defense parameters fitted by weighted maximum likelihood (exponential time-decay, 2.5-year half-life, 12-year window), with fitted home advantage and low-score correlation rho, plus L2 shrinkage so sparse-data teams stay sane. Separates *how* teams are strong: e.g. Morocco rates mid-pack on attack but elite on defense — invisible to a single Elo number.

```bash
python3 dixoncoles.py --fit               # refit and save data/dc_params.json
python3 dixoncoles.py "Brazil" "Morocco"  # DC-only match prediction
python3 dixoncoles.py --ratings           # attack/defense table
python3 dixoncoles.py --backtest          # compare Elo vs DC vs blend
```

`simulate.py` and `edge.py` accept `--model elo|dc|blend` (default blend). After refreshing `data/results.csv`, run `dixoncoles.py --fit` to update DC parameters.

## Usage

```bash
python3 predictor.py "Brazil" "Morocco"             # one match, neutral venue
python3 predictor.py "Mexico" "South Africa" --home # team 1 has home advantage
python3 predictor.py --worldcup   # all unplayed WC 2026 fixtures -> predictions_worldcup_2026.csv
python3 predictor.py --backtest   # walk-forward evaluation on matches since 2024
python3 predictor.py --ratings    # top 30 current Elo ratings
```

Team names must match the dataset (e.g. "United States", "South Korea", "Ivory Coast").

## API keys

Live odds/injury fetchers read API keys from `data/api_keys.json` (ignored by
git). Copy `data/api_keys.example.json` to `data/api_keys.json` and fill in the
providers you use:

```json
{
  "the-odds-api": "your_key",
  "api-football": "your_key",
  "datagolf": "your_key"
}
```

Explicit CLI flags still win (`--api-key`, `--dg-key`, `--odds-key`), and
environment variables still work (`THE_ODDS_API_KEY`, `API_FOOTBALL_KEY`,
`DG_API_KEY`).

## Tournament simulator

```bash
python3 simulate.py            # 10,000 Monte Carlo runs
python3 simulate.py -n 50000   # more precision (~5s)
```

Simulates the group stage (already-played matches count as fixed results — refresh `data/results.csv` mid-tournament and re-run), applies FIFA tiebreakers (points, GD, GF, head-to-head, lots), ranks the eight best third-placed teams, and plays the official Round-of-32 bracket through the final. Knockout draws go to extra time (1/3 intensity) then 50/50 penalties; hosts (US/Mexico/Canada) keep home advantage. Third-place teams are assigned to bracket slots from FIFA's exact **Annex C** table (`data/annexc_thirds.json`, all 495 scenarios) when present, falling back to constraint matching otherwise. Output: `tournament_odds.csv` — per-team probabilities of winning the group, reaching each round, and lifting the trophy.

## Squad availability adjustment (squads.py / injuries.py)

The results-based models can't see team news. `squads.py` quantifies it: each squad (data/squads.csv — FIFA's official lists, with EA-derived proxy squads for 12 federations whose lists weren't parseable; see the `source` column) is matched to EA FC 26 player ratings (data/ea_players.csv), and squad power is a starter-weighted mean (best XI full, ranks 12–18 half). Squad *quality* is already priced into Elo/DC ratings, so predictions are adjusted only by the gap between full-strength and currently-available power, converted to Elo points via a cross-team calibration (~25 Elo per rating point) and **split into attack/defence by the absent players' positions** (v2 M5). Teams with fewer than 15 EA-matched players (mostly smaller Asian/African federations) get no adjustment — the model falls back to unadjusted output.

```bash
python3 squads.py                 # refresh data/squad_ratings.csv
python3 squads.py --report        # power table + listed absences
python3 squads.py --match "Canada" "Bosnia and Herzegovina" --home \
                  --without "Alphonso Davies"        # what-if, not persisted
python3 injuries.py --api-key KEY # pull live WC injury list (api-football.com,
                                  # free key; run locally, writes absences_api.csv)
python3 edge.py --squad-adj       # edge report with availability adjustments
```

Confirmed absences live in `data/absences.csv` (manual, one `team,player,note` row each) and `data/absences_api.csv` (rewritten by injuries.py; the manual file wins on duplicates). The adjustment is **opt-in** (`--squad-adj` on edge.py) — compare adjusted and unadjusted edges while the approach accumulates evidence.

## Edge finder (betting odds comparison)

```bash
python3 edge.py --template     # writes odds.csv with all upcoming fixtures
# fill in decimal odds from your bookmaker, then:
python3 edge.py                # edge report, EV, quarter-Kelly stakes
python3 edge.py --api-key KEY  # or pull live median odds from the-odds-api.com
python3 edge.py --bankroll 500 # convert Kelly fractions to currency stakes
```

For each outcome it removes the bookmaker's vig (overround), compares the implied probability to the model's, and reports edge, expected value per unit, and a quarter-Kelly stake in £. Output: `edge_report.csv`. `odds_sample.csv` shows the format with illustrative (not real) odds. Edges under ~3% are within model noise; closing lines at sharp books are hard to beat consistently.

### Bankroll tracking

Stakes are sized from a live bankroll (started £100, stored in `data/bankroll.json`). `edge.py` auto-records its recommendations (edge ≥ 3%, best outcome per match) in `data/ledger.csv`; pass `--no-bet` to skip. After results come in:

```bash
python3 bankroll.py --settle   # settle open bets, compound the bankroll
python3 bankroll.py            # status: bankroll, open bets, P&L
python3 bankroll.py --reset 100
```

`update.sh` settles automatically each run, so stakes always reflect the current bankroll. Knockout settlement is on the **90-minute** score (v2 M4): group-stage bets settle exactly from the dataset, and for any knockout that went to extra time, add a `data/ko_overrides.csv` row (`date,home,away,score90`) so the 1X2/O-U/BTTS markets settle correctly.

## Performance (out-of-sample, ~2,540 matches since Jan 2024)

| Model | Accuracy | Brier (lower = better) |
|---|---|---|
| Elo + Poisson | 60.2% | 0.5070 |
| Dixon-Coles | 59.8% | 0.5089 |
| **50/50 blend** | **60.4%** | **0.5038** |

Random chance Brier = 0.667. Both models fitted only on pre-2024 data for this test.

## Data

`data/results.csv` from [martj42/international_results](https://github.com/martj42/international_results), which includes the 2026 World Cup fixtures. Re-download to refresh ratings as group-stage results come in:

```bash
git clone --depth 1 https://github.com/martj42/international_results /tmp/intres
cp /tmp/intres/results.csv data/results.csv
```

## v2 (new)

All v2 additions are **opt-in flags** (except portfolio staking, which is a default
safety layer); the default behaviour of every command is unchanged. See
`V2_NOTES.md` for fitted parameters and acceptance numbers, and `V2_PLAN.md` for
the design. Quick reference:

**Validation harness — `validate.py`** (the yardstick everything is measured against)
```bash
python3 validate.py               # walk-forward accuracy/Brier/log-loss + reliability
python3 validate.py --gate        # CI gate: non-zero exit if blend Brier regressed >0.002
python3 validate.py --calibrate   # fit isotonic calibration -> data/calibration.json
```

**Probability calibration (M2)** — isotonic per-outcome, fit by `validate.py --calibrate`:
```bash
python3 edge.py --calibrated      # apply calibration to the model's 1X2
```

**Market anchoring + CLV (M3)**
```bash
python3 market_blend.py --fit     # fit model/market weight w on WC2022 -> data/market_blend.json
python3 edge.py --market-blend    # anchor model 1X2 toward the de-vigged market (logit blend)
python3 clv.py --snapshot         # record current odds for open bets (The Odds API)
python3 clv.py --report           # closing-line-value per settled bet, rolling mean CLV
```

**Knockout correctness (M4)** — 1X2/O-U/BTTS settle on the **90-minute** score.
`data/results.csv` records the after-extra-time score, so for any knockout that
went to extra time add a row to `data/ko_overrides.csv` (`date,home,away,score90`)
and `bankroll.py --settle` uses it. The exact FIFA **Annex C** third-place table
(`data/annexc_thirds.json`, 495 scenarios) is used by `simulate.py` when present.

**Squad layer v2 (M5)** — `--squad-adj` now splits an absence into attack vs
defence by position (a missing forward lowers the team's own goals; a missing
defender raises the opponent's) and weights players by likely minutes. All 48
squads use official lists.

**Context features (M6)**
```bash
python3 context.py --fit          # fit rest/altitude lambda correction -> data/context_coef.json
python3 edge.py --context         # apply rest/altitude correction per fixture
```
Only the altitude effect proved significant (a sea-level side at Mexico City scores
~27% fewer goals); rest and travel were dropped as insignificant.

**Portfolio staking (M7, default on)** — `edge.py` sizes same-day bets jointly:
single-match cap 10%, daily cap 25%, correlated-exposure cap 15% (shared-team,
incl. open outrights), and a drawdown brake that halves Kelly below 70% of the
bankroll's running peak (tracked in `data/bankroll.json`). Disable with
`--no-portfolio`.

**Ops (M8)**
```bash
python3 report.py                 # -> dashboard.html (offline: bankroll, CLV, calibration, queue, title movers)
```
`edge.py` writes `bet_queue.csv` (the day's reviewable candidates with active
adjustments). `update.sh` now also runs `clv.py --snapshot`, `validate.py --gate`
(warns, never blocks), and `report.py`.

Flags combine, applied in this order: `--calibrated` → `--market-blend` →
`--context` (and `--squad-adj` / `--conf-adj` on the ratings). Example:
```bash
python3 edge.py --squad-adj --calibrated --market-blend --context
```
