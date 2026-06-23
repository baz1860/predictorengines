# Golf Prediction Engine

A PGA Tour + majors betting engine. It does the same four things the World Cup
engine does, just for a sport with a new tournament every week:

1. **pulls the season's tournament list** (ESPN schedule),
2. **gets this week's field**,
3. **prices it with a fitted model** (strokes-gained + variance, Monte-Carlo
   simulated, calibrated and market-anchored), and
4. **prints the best bets for the tournament — round by round.**

## Use it

One command. It refreshes this week's field, runs the model, and writes a
round-by-round best-bets card:

```bash
python3 -m golf.season
```

That writes [`data/card.md`](data/card.md) — the only file you normally read. It
lists:

- **Tournament card** — outright, top-5/10/20, make-cut and matchup bets the
  model backs (staked, +EV, calibrated and market-blended). Sides it prices but
  doesn't back are left off, so the page is signal not noise.
- **Round N matchups** — that round's group bets. Signature/no-cut events play in
  twosomes (2-balls); full-field events play in threesomes (3-balls). The pricer
  handles either — the section title reflects whichever you pasted.
- **Field forecast** — top-10 win / top-N / make-cut for context.

Other things you can do:

```bash
python3 -m golf.season --schedule        # the season's tournament list
python3 -m golf.season --round 2         # also price round 2's matchups (2-/3-balls)
python3 -m golf.season --no-refresh      # reprice from cached data (no network)
python3 -m golf.season --stats --fit     # refresh stat pages + refit, then price
```

### Round-by-round matchups (2-balls / 3-balls)

Group boards aren't on a free feed, so you paste them in. Drop a bookmaker board
into `data/threeballs_r{N}_raw.txt`, then:

```bash
python3 -m golf.season --round 1         # parses the paste and prices that round
```

The paste is a flat list of header + player + odds lines. A `2 Ball` or `3 Ball`
header opens a group; each following name is paired with the price on the next
line. Use **2-ball** headers for twosome events (signature/no-cut) and **3-ball**
for full-field events — the pricer detects group size from the paste:

```
3 Ball (Round 1) - Rai / Morikawa / Day      2 Ball (Round 1) - Rai / McNealy
AARON RAI                                     AARON RAI
2.75                                          1.90
COLLIN MORIKAWA                               MAVERICK MCNEALY
2.38                                          1.90
JASON DAY
3.50
```

Prices land in `data/round_edges.csv` and the card's "Round N matchups" section.

> **Stale-board guard.** Before pricing, every player on the board is checked
> against the current `data/field.csv`. If names don't match (the usual cause is
> last week's board left in place), the round is skipped with a loud note rather
> than pricing the wrong event — so always re-paste the current week's groups.

Outright / place / matchup prices go in `data/odds.csv` and `data/matchups.csv`.

### First-time setup

Once, to build the data the model learns from:

```bash
python3 -m golf.fetch --seed 2022 2023 2024 2025 2026   # backfill history
python3 -m golf.refresh --stats --fit                   # fit the model
```

After that, `python3 -m golf.season` is all you run week to week.

## In the app

The **Predict / Simulate / Edge** tabs drive the same engine (head-to-head
matchups, full-field projection, and staked edges into the shared
`suite_ledger.csv`, which auto-settle against results). `golf.season` is the
command-line equivalent that hands you the whole week in one page.

---

## Under the hood

`golf.season` is a thin orchestrator. The modelling it drives is unchanged and is
where the quality lives:

```
golf/
├── season.py       # THE front door: schedule → field → model → card
├── providers/      # ESPN schedule/field/leaderboard, PGA stats, weather, odds
├── fetch.py        # --seed / --accumulate → rounds.csv (history)
├── refresh.py      # free-source weekly refresh → field.csv + SQLite cache
├── model.py        # fit(): time-decayed ridge skill + per-player σ + form +
│                   #   course fit → model_params.json;  predict_field()
├── simulate.py     # 4-round Monte Carlo with cut; joint matchups / 3-balls
├── round_pricer.py # single-round group pricing (2-/3-balls; driven by season.py)
├── market.py       # power de-vig, log-odds market blend, CLV tracking
├── calibrate.py    # isotonic per-market maps (win ≤ T5 ≤ … ≤ cut guard)
├── edge.py         # calibrated + blended EV across all markets
├── portfolio.py    # simultaneous-Kelly: per-player + total caps, drawdown brake
├── validate.py     # walk-forward backtest + regression gate (the yardstick)
├── weekly_report.py# longer narrative report (season.py is the lean version)
└── data/
    ├── rounds.csv          # SOURCE OF TRUTH: one row per player per round
    ├── model_params.json   # fitted skill/σ/form/course params
    ├── field.csv           # current field (written by refresh)
    ├── card.md             # ← the output you read
    ├── calibration.json, market_blend.json, odds_history.csv (CLV)
    ├── odds.csv, matchups.csv, threeballs.csv   # book prices you provide
    └── predictions.csv, edge_report.csv, round_edges.csv  # raw tables
```

### The model

Each round is decomposed by time-decayed, ridge-shrunk least squares:

```
score_to_par[player, tournament, round] = mu + difficulty[t,r] − skill[player] + ε
ε ~ Normal(0, σ[player])
```

- **skill** — strokes-gained vs field; ridge shrinks low-sample players toward
  the mean, and a per-tournament `difficulty` term field-strength-adjusts so weak
  fields and majors are comparable.
- **σ (fitted, per player)** — round-to-round variance from fit residuals,
  Empirical-Bayes shrunk toward the field σ (~2.85); drives longshot value.
- **form** — short-window residual nudge; **course fit** — shrunk
  per-(player, course) residual when the course is known.

`predict_field()` turns these into per-player `rating` + `σ`; `simulate.py` draws
four correlated, fat-tailed rounds (`data/sim_config.json`: `round_corr`,
`tail_df`) so win / top-N / make-cut and the matchup/3-ball markets all come from
the **same** draws and stay internally consistent.

### Calibration, market, staking

- **calibrate.py** — isotonic maps per market correct the simulator's systematic
  miscalibration, with a nesting guard (win ≤ T5 ≤ T10 ≤ T20 ≤ cut).
- **market.py** — power de-vig for outright boards, per-line place margins, a
  log-odds blend toward the market, and CLV tracking.
- **portfolio.py** — simultaneous-Kelly with a per-player correlation cap, a
  total weekly cap, and a drawdown brake.

### Validating the model

```bash
python3 -m golf.validate --since 2024-06-01 --sims 8000   # walk-forward + gate
```

Walk-forward (139 events, 2023-06 → 2026-06) shows positive Brier skill on every
market; make-cut and top-20 ≈ +9–10%. `validate.py` is the regression gate the
daily `update.sh` runs before trusting a refit.
