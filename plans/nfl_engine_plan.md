# NFL Prediction Engine — Build Plan (ATS-focused)

Sibling of the CFB engine, built for picks against the spread. Same `fetch → fit → predict → edge → validate` backbone, same suite contracts (`contracts/protocol.py`), same bankroll/ledger conventions. The central design difference from CFB: **no normal margin approximation**. NFL margins are discrete and pile up on key numbers (3, 7, 6, 10, 14), and half-point line placement around those numbers is where ATS pricing lives. Cover, push, total, and moneyline probabilities all come from an **empirical discrete margin PMF**.

Honesty clause (learned from `cfb/ats_backtest.py`): NFL closing lines are the sharpest in sports (closing spread MAE ≈ 10.2–10.5 vs actual margin; a good public model lands ≈ 10.8–11.3). The engine's job is *not* to beat closers blind — it is to price openers/soft lines correctly, exploit key-number line placement, quantify half-point value, and track CLV. The validation gate enforces this framing.

---

## Architecture

```
nfl/
├── fetch_data.py       # games (with closing lines) + pbp EPA aggregates + QB weekly stats
├── elo.py              # MOV-scaled Elo, rest/bye adjustments, preseason regression
├── power.py            # offense/defense ridge points model (adapt cfb/power.py)
├── epa.py              # opponent-adjusted team EPA/play ratings → points (blend CANDIDATE)
├── qb.py               # rolling QB value (EPA/dropback), starter deltas, overrides
├── margin_dist.py      # empirical discrete PMFs: margin | predicted spread, total | predicted total
├── predictor.py        # blend ratings → predicted margin/total → full distributions; CLI
├── edge.py             # odds.csv → cover/push/ML/total probs → edge_report.csv + ledger
├── ats_backtest.py     # walk-forward vs real closing spreads (2015+)
├── totals_backtest.py  # same for totals
├── validate.py         # walk-forward gate → data/validation_baseline.json
├── bankroll.py         # reuse cfb/bankroll.py pattern (settle vs games.csv)
├── engine.py           # command API: schema / predict / edge
└── data/
    ├── games.csv               # SOURCE OF TRUTH: one row per game, 1999→, incl. closing lines
    ├── epa_team_week.csv       # per-team-week EPA splits from pbp
    ├── qb_week.csv             # per-QB-week dropbacks + EPA
    ├── qb_overrides.csv        # manual game-day starter overrides (team, week, qb_name)
    ├── elo_params.json, power_params.json, epa_params.json, qb_values.json
    ├── margin_pmf.json         # fitted conditional PMFs (margin + total)
    ├── blend_weight.json       # tuned via validate.py --tune-blend
    ├── odds.csv                # manual/API odds entry (both sides where possible)
    ├── ledger.csv, bankroll.json, odds_history.csv (CLV)
    └── validation_baseline.json, edge_report.csv
```

---

## Data (free, no API key)

**Primary source: nflverse.** One file covers games, results, AND historical closing lines:

- **Games + lines**: `https://github.com/nflverse/nfldata/raw/master/data/games.csv` — every game 1999→, with `spread_line` (closing, home-perspective), `total_line`, `home_moneyline`/`away_moneyline`, `home_rest`/`away_rest`, `div_game`, `roof`, `surface`, `temp`, `wind`, `home_qb_name`/`away_qb_name`, coaches, scores. This single file powers fitting, backtesting against real closers, QB-starter history, and settlement. (Verify column names at implementation time; nflverse occasionally renames.)
- **Play-by-play (for EPA)**: nflverse-data releases, per-season parquet/csv: `https://github.com/nflverse/nflverse-data/releases/tag/pbp` (files `play_by_play_YYYY.parquet`). `fetch_data.py` aggregates to `epa_team_week.csv` (team-week offense/defense EPA/play, pass/rush splits, dropback counts) and `qb_week.csv` (passer_id, dropbacks, EPA/dropback). Aggregate at fetch time and store the small CSVs — don't keep raw pbp in the repo.

Era decision (settled): **fit structural parameters on 2003–2014, walk-forward test 2015→**. The margin PMF uses 2003→ (needs volume); team ratings use rolling windows so era barely matters; anything tuned must be tuned pre-2015 or walk-forward.

---

## Models

Three rating models produce a predicted home margin; power/EPA also produce a total. Blend weights are **tuned by walk-forward, not assumed** — the CFB lesson (EPA rejected there; NFL pbp EPA is much cleaner and is expected to earn a positive weight, but it must prove it via `--tune-blend`).

### 1. Elo (`elo.py`) — 538-style

- Update: `K = 20`, MOV multiplier `ln(|margin|+1) × 2.2 / (0.001·|elo_diff| + 2.2)`.
- HFA: fitted per rolling 3-season window (NFL HFA has declined from ~2.6 to ~1.5 pts; do NOT hardcode a constant). Store as points, convert via Elo scale.
- Rest: +1.0 pt off a bye, −0.7 pt short week (Thu after Sun) — initialise with these, then fit on 2003–2014 (`home_rest`/`away_rest` are in games.csv).
- Preseason: regress 1/3 to 1505 mean.
- Spread map: Elo diff / 25 = points (fit exactly on 2003–2014).

### 2. Offense/defense power ratings (`power.py`)

Direct adaptation of `cfb/power.py`: per-team offense/defense in points via weighted ridge, exponential time decay **half-life ≈ 0.8 seasons, window 2.5 seasons** (NFL rosters churn faster than CFB; 32 teams × 17 games needs a tighter window — expose both as params and fit). Fitted HFA. Produces margin AND total. Re-evaluate CFB's `TOTAL_SHRINK` (fit k on 2003–2014; expect ~0.85–0.95).

### 3. EPA ratings (`epa.py`)

Same ridge structure fitted on team-week offense/defense EPA/play (pass and rush separately, pass weighted ~1.6× — passing is more predictive), opponent-adjusted, converted to points via a fitted scalar (league plays/game × EPA-to-points). Candidate for the blend; keep if walk-forward margin MAE improves, drop to reference-only otherwise.

### 4. QB adjustment (`qb.py`)

- **QB value** = shrunk rolling EPA/dropback: exponentially decayed over trailing dropbacks (half-life ≈ 250 dropbacks), shrunk toward league-average backup level for low samples. Convert to points/game: `qb_pts = (qb_epa_pd − league_avg_epa_pd) × 36` (≈ dropbacks/game; fit the scalar on 2003–2014).
- **Application**: each team carries a rolling "expected starter". When the actual/announced starter differs, adjust that team's predicted margin by `qb_pts(actual) − qb_pts(expected)`. Backtests read actual starters from `home_qb_name`/`away_qb_name` in games.csv (leak-free: value computed from prior weeks only). Live picks read `data/qb_overrides.csv` (columns: `season, week, team, qb_name`) for game-day news.
- **Assumption**: unlisted/rookie QBs with < 50 career dropbacks get value = league backup level (≈ −4 pts vs average starter) until data accrues.

### Blend

`predicted_margin = Σ wᵢ · marginᵢ + qb_delta_home − qb_delta_away`, weights from `validate.py --tune-blend` (grid over elo/power/epa simplex, walk-forward margin MAE on 2015–2020, validated on 2021+). Start at elo/power 50/50, epa 0. Predicted total from power (or power/epa blend, same procedure).

---

## Margin distribution (`margin_dist.py`) — the key-number core

**Do not use a normal distribution anywhere a bet is priced.**

### Construction

1. Take all games 2003→ with a closing `spread_line`. For each, record `(spread_line, home_margin)`.
2. Build the conditional PMF `P(home_margin = k | spread ≈ s)` on integer support k ∈ [−60, 60]:
   - Bucket by spread with **kernel smoothing**: weight each historical game by a tri-cube kernel on `|spread_line − s|` with bandwidth ≈ 3 pts (tune by likelihood CV on pre-2015 data).
   - Blend each conditional PMF with the unconditional shifted PMF (shrinkage ~20%) to stabilise sparse buckets (big favourites).
   - Evaluate on a grid of s ∈ {−20.0, −19.5, …, +20.0}; store in `margin_pmf.json`; interpolate between grid points at query time.
3. **Pricing**: given the model's predicted margin m, look up the PMF at s = m and compute exactly:
   - `P(cover line L) = P(margin > L)`, `P(push) = P(margin = L)` for integer L,
   - `P(home ML) = P(margin > 0)`, `P(tie) = P(margin = 0)` (≈ 0.2–0.4%; graded as ML push),
   - EV including three-way spread outcomes (win/push/loss) — pushes matter at 3 and 7.

**Central statistical assumption (state in code comments, verify in validation):** the distribution of margins around a *model-predicted* spread equals the distribution around a *market* spread of the same value. This is approximately true when the model is well-calibrated in the mean; it is verified by the cover-probability reliability check in `validate.py`. If model-conditioned reliability is poor, widen the PMF by convolving with a small discrete kernel (fitted residual inflation) — the model is noisier than the market it borrows the shape from.

### Sanity anchors (unit tests)

- Unconditional: `P(|margin| = 3) ≈ 14–16%`, `P(|margin| = 7) ≈ 9–10%`, elevated mass at 6, 10, 14, 4; `P(margin = 0)` < 0.5%.
- At s = −3 (home favourite by 3): `P(cover −2.5) − P(cover −3.5) ≈ 0.09–0.11` — the mass sitting on 3. A normal(σ=13) gives ~0.03; this gap is the whole point of the engine.
- Pushes at −3 with line −3: ≈ 9–10%.

### Totals

Identical machinery: `P(total_points = t | predicted_total ≈ T)`, kernel-smoothed on `total_line`, key totals (37, 41, 44, 47, 51) emerge from data — no hand-coding. Used for over/under cover and push probabilities.

---

## Edge & staking (`edge.py`, `bankroll.py`)

- `edge.py --template` writes `odds.csv` from upcoming games (from games.csv future rows): `game_id, date, home, away, market, side, line, odds_decimal`. Enter both sides where possible → exact de-vig; single side assumes −110 (4.55% overround).
- Prices spread/total/ML per the PMF; emits canonical edge rows (`CANONICAL_EDGE_FIELDS` in `contracts/protocol.py`) → `edge_report.csv`; quarter-Kelly (push-adjusted: Kelly on the win/push/loss trinomial), best edge per market per game, ≥ 3% edge auto-logs to `data/ledger.csv`.
- **Half-point value report**: for each spread bet, also print the fair price of the adjacent half-point lines (±0.5) so buying points across 3/7 can be judged. Cheap to produce from the PMF, and it's the practical payoff of the key-number work.
- Bankroll: copy `cfb/bankroll.py` conventions (starts £100, `--settle` grades vs games.csv scores; spread pushes and ML ties handled).
- CLV: log line at bet time; `core/clv.py` + `clv_suite.py --snapshot` work unchanged once ledger rows are canonical.

---

## Validation (`validate.py`, backtests)

Walk-forward weekly from 2015 week 1: refit power/EPA before each week, Elo updated game-by-game, QB values from prior weeks only, blend weight and PMF fitted on pre-2015 (PMF may accumulate walk-forward — it's a distribution of *outcomes given closeness*, low leak risk, but keep it strictly past-only anyway).

Metrics and expectations (baseline → `data/validation_baseline.json`, gate = no metric regresses > 5%):

| Metric | Target | Reference |
|---|---|---|
| Margin MAE | ≤ 11.3 | Closing line ≈ 10.2–10.5 |
| ML Brier | ≤ 0.215 | Home-team-always ≈ 0.24; closing ML ≈ 0.205 |
| Cover-prob reliability | Decile calibration within noise | Validates the PMF assumption |
| Push-rate calibration | Predicted vs actual push % at 3, 7, 6, 10 | Validates key-number mass |
| ATS vs closers (`ats_backtest.py`) | Report honestly; expect 49–52% | Break-even 52.4% at −110 |
| Totals vs closers | Report; CFB found totals softer | Break-even 52.4% |

`ats_backtest.py` mirrors the CFB one: bet blend vs `spread_line` when disagreement ≥ N pts, sweep N, report cover %, ROI at −110, and whether performance decays as disagreement grows (in CFB it did — large gaps were model error; expect the same, and say so in the README).

---

## App integration (`engine.py`, adapter)

Same pattern as NHL/CFB: `COMMANDS = {schema, predict, edge}` (no tournament simulate in v1; a playoff simulator is v2). `NFLAdapter` in `app/engines/nfl.py`, id `nfl`, capabilities `predict · edge`; settlement via bankroll grading against games.csv. Add `test_nfl_contract.py` (clone `test_nhl_contract.py` shape: schema, predict shape, edge canonical fields, settlement incl. a push case and an ML tie case). Register in `app/engines/__init__.py`; add provenance manifest entries for `games.csv` freshness.

---

## Implementation phases (each ends with passing checks)

### Phase 1 — Data
1. `fetch_data.py`: download nfldata games.csv → `nfl/data/games.csv` (normalise team abbreviations to full names consistently, incl. relocations: OAK→LV, SD→LAC, STL→LA); pbp → `epa_team_week.csv`, `qb_week.csv` for 2003→.
2. **Checks**: ≥ 6,000 games 2003→; `spread_line` non-null ≥ 99% from 2003; spot-check 3 known finals; mean home margin ≈ +2.2 declining by era.

### Phase 2 — Margin PMF (build this BEFORE the ratings — it's testable standalone against market spreads)
1. `margin_dist.py --fit` from games.csv; grid PMFs for margin and total.
2. **Checks**: the sanity anchors above; CV log-likelihood beats normal(σ fitted) by a clear margin; plot-free text histogram at s = −3 shows the spike at 3.

### Phase 3 — Ratings
1. `elo.py --fit` (spread map, HFA, rest on 2003–2014), `power.py --fit`, `epa.py --fit`, `qb.py --fit`.
2. **Checks**: Elo top-5 in any season passes the sniff test; power margin MAE (walk-forward 2015–2018 sample) ≤ 11.8; QB values: elite QBs ≈ +4 to +6 pts, replacement ≈ −4; predicted margins correlate with closing spreads r ≥ 0.85.

### Phase 4 — Blend + predictor
1. `validate.py --tune-blend` over the simplex; `predictor.py "Chiefs" "Bills"` prints margin, total, distribution summary, cover probs at the market line, key-number table.
2. **Checks**: blend beats best single model on 2015–2020 margin MAE; 2021+ holdout confirms; QB-override changes prediction by the expected delta.

### Phase 5 — Edge + bankroll
1. `edge.py`, `bankroll.py`, half-point value report, ledger/CLV wiring.
2. **Checks**: hand-computed EV matches for one spread (with push), one total, one ML; both-sides de-vig sums to 1; quarter-Kelly trinomial matches closed form.

### Phase 6 — Backtests + gate
1. `ats_backtest.py`, `totals_backtest.py`, `validate.py --gate --update-baseline`.
2. **Checks**: full metric table produced; README written with honest numbers (CFB README is the tone template); gate passes reproducibly.

### Phase 7 — App wiring
1. `NFLAdapter`, `test_nfl_contract.py`, registration, provenance.
2. **Checks**: `python3 run_checks.py` green; engine appears in `/api/engines`; a bet settles end-to-end including a push.

---

## Quick start (once built)

```bash
python3 -m nfl.fetch_data                      # games + EPA + QB aggregates
python3 -m nfl.margin_dist --fit
python3 -m nfl.elo --fit && python3 -m nfl.power --fit && python3 -m nfl.epa --fit && python3 -m nfl.qb --fit
python3 -m nfl.validate --tune-blend --write
python3 -m nfl.validate --gate --update-baseline
python3 -m nfl.predictor "Eagles" "Cowboys"
python3 -m nfl.edge --template                 # fill odds.csv, then:
python3 -m nfl.edge --min-edge 3
python3 -m nfl.bankroll --settle
```

---

## Stated statistical assumptions

1. Margin PMF conditioned on model prediction ≈ PMF conditioned on market spread of equal value (verified by reliability; widened by fitted convolution if not).
2. QB effects are additive in points and transfer across teams; unknown QBs = league backup (−4 pts) until 50 dropbacks.
3. HFA and rest effects are additive and slowly varying (rolling refit, no constants).
4. Missing juice = −110 both sides.
5. Games are independent (no same-game or cross-game correlation in staking, v1).
6. The 2003→ margin shape is stationary enough for the PMF after conditioning on the spread (key-number mass has been stable; the extra-point rule change of 2015 shifted 7↔6/8 mass slightly — the kernel absorbs it, but check the push-rate calibration split pre/post 2015).

## Explicit non-goals (v1)

Live/in-game pricing, teasers/parlays (though the PMF makes teaser EV a natural v2 — pricing 6-pt teasers through 3 and 7 is the classic key-number application), playoff simulator, weather model (roof/temp/wind columns exist in games.csv; leave as a v2 feature for totals), player props.
