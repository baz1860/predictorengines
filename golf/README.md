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

> **Stale-board guard.** Boards are tagged with the event they were captured
> for (an `event` column in `threeballs.csv` / `matchups.csv`, written
> automatically by refresh), and a board whose tag doesn't match the current
> `data/field.csv` event is refused rather than priced — name overlap alone
> can't tell consecutive events apart when fields overlap (co-sanctioned
> weeks). A paste older than the current event week is likewise ignored.
> Player names are still checked against the field as a second line of
> defence. If you hand-edit a board CSV, set its `event` column to the value
> in `field.csv`.

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
├── fetch.py        # --rebuild / --seed / --accumulate → rounds.csv (history)
├── refresh.py      # free-source weekly refresh → field.csv + live SQLite cache
├── model.py        # fit(): source-ID skill + per-player σ + form +
│                   #   measured course profiles → model_params.json
├── simulate.py     # 4-round Monte Carlo with cut; joint matchups / 3-balls
├── round_pricer.py # single-round group pricing (2-/3-balls; driven by season.py)
├── market.py       # complete-board de-vig + raw implied-price tracking
├── calibrate.py    # isotonic per-market maps (win ≤ T5 ≤ … ≤ cut guard)
├── edge.py         # raw model and final EV probabilities across all markets
├── portfolio.py    # opt-in capped Kelly; automatic staking disabled by default
├── validate.py     # walk-forward backtest + regression gate (the yardstick)
├── weekly_report.py# longer narrative report (season.py is the lean version)
└── data/
    ├── rounds.csv          # SOURCE OF TRUTH: one row per player per round;
    │                       # real venue/rules/tee time from ESPN per-event data
    ├── golf.db             # rebuildable LIVE cache only: current event/field,
    │                       # odds, public-stat snapshots, provider runs
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
- **form** — short-window residual nudge. Sparse exact-venue effects are disabled;
  EB-shrunk par-3/4/5 performance and within-player par/yardage sensitivities use
  only single-course events and provide a capped general-course adjustment.
- **scoring shape inputs** — EB-shrunk birdie, bogey, and double-bogey
  frequencies from cached hole scorecards are available for an asymmetric round
  shape, but the production mix remains zero after validation rejected it.

`predict_field()` turns these into per-player `rating` + `σ`; `simulate.py` draws
integer-stroke rounds. Correlation and player-specific scoring tails are disabled
because the 2026-07-26 pre-holdout retune did not clear its promotion gate. The
general-course adjustment remains at its conservative 0.50 weight for the same
reason. Win, top-N, make-cut,
matchup and 2-/3-ball probabilities all come from that one joint distribution,
so market nesting is true by construction.

### Calibration, market, staking

- **calibrate.py** — a make-cut isotonic map is retained only when it improves a
  temporal holdout; coherent win/place probabilities remain uncalibrated.
- **market.py** — power de-vig only for complete mutually exclusive boards.
  One-sided/partial prices remain raw implied probabilities; guessed margins and
  market blending are disabled.
- **portfolio.py** — stake sizing is opt-in (`--kelly`); the default is zero until
  timestamped offered-price history supports an economic backtest.

### Validating the model

```bash
python3 -m golf.integrity                                    # offline data/model checks
python3 -m golf.validate --since 2024-06-01 --sims 8000      # walk-forward + gate
python3 -m golf.validate --tune-shape --since 2024-06-01 \
  --selection-split 2025-07-01 --holdout 2026-01-01 \
  --sims 8000 --write                                        # sealed-holdout retune
```

`validate.py` excludes current-only public stats, global priors, weather, and
exact-course effects from historical folds because point-in-time/economic
evidence does not exist. No-cut events are excluded from cut-market metrics. The gate
runs before the live refit. Use `--rebaseline` only after
reviewing an intentional model or ground-truth change; ordinary runs never move
the frozen baseline.

The offline integrity command also runs every Monday in GitHub Actions and on
changes to the golf module. It checks round-key uniqueness, event metadata
consistency, numeric/date validity, and that the fitted model is not older than
the authoritative CSV.

To repair or reproduce the full free history without reading existing rows:

```bash
python3 -m golf.fetch --rebuild 2022 2023 2024 2025 2026 \
  --tours pga,liv,eur
```

The rebuild is atomic, uses the per-event ESPN leaderboard for statuses, course,
tee times and tournament rules, and is byte-deterministic from the same payloads.
`cut_rule` is the rule known before play; `realized_cut_count` is stored
separately and is never fed to simulation.
