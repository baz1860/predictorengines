# NHL Prediction Engine

Local-first NHL engine for match win probability, projected goals, puck-line,
totals, edge finding, gated staking, and settlement through the shared app
ledger.

## Model

The first version uses `nhl/data/team_stats.csv` as a baseline. It converts each
team's goals, shots, power play, penalty kill, save percentage, and points share
into attack, defence-allowed, and form ratings. Match pricing uses expected
regulation goals, independent Poisson score distributions, and an overtime split
for moneyline probabilities.

Totals are priced against final score. If regulation is tied, the distribution
adds one deciding OT/SO goal before grading totals.

Markets supported:

- `ml`: home/away moneyline
- `spread`: NHL puck line, typically home `-1.5` / away `+1.5`
- `total`: over/under goals

The included CSVs are runnable seed data, not an official historical database.
Replace `team_stats.csv`, `fixtures.csv`, `results.csv`, and `odds.csv` with
your preferred NHL data source before treating outputs as live analysis.

## Staking Gate

NHL staking is disabled by default through `nhl/data/validation_gate.json`.
The engine can show analytical edges, but every `recommended` flag and
`stake_gbp` is forced to zero unless the gate explicitly has both
`status: "PASS"` and `staking_enabled: true`.

The current forecast validation is walk-forward and uses fixed NHL-style league
priors plus prior same-season results. It beat trivial moneyline baselines on
`nhl/data/results_2025_26.csv`, but the staking gate remains failed because
there is no timestamped historical sportsbook-odds dataset proving positive
out-of-sample ROI.

## Historical Odds

Decision-grade ROI backtests use `nhl/data/odds_history.csv`, one quoted side per
row:

```csv
event_id,game_date,start_time_utc,captured_at_utc,bookmaker,market,side,line,decimal_odds,source
2025020001,2025-10-07,2025-10-07T21:00:00Z,2025-10-07T15:00:00Z,pinnacle,ml,home,,1.91,provider
```

The loader rejects unknown `event_id`s, quotes captured at or after game start,
duplicate sides, missing complementary sides, invalid markets/sides, non-numeric
lines, and decimal odds `<= 1.0`.

Fetch provider snapshots into the strict schema:

Free path, OddsPapi. Historical odds start in January 2026, so this cannot
backfill the whole 2025-26 season:

```bash
ODDSPAPI_KEY=... python3 -m nhl.oddspapi \
  --results nhl/data/results_2025_26.csv \
  --out nhl/data/odds_history.csv \
  --bookmakers pinnacle,bet365,draftkings \
  --markets ml,total,spread \
  --date-from 2026-01-01 \
  --date-to 2026-04-18
```

Paid/full-snapshot path, The Odds API:

```bash
THE_ODDS_API_KEY=... python3 -m nhl.the_odds_api \
  --results nhl/data/results_2025_26.csv \
  --out nhl/data/odds_history.csv \
  --regions us \
  --markets h2h,spreads,totals \
  --snapshot-time 15:00:00Z \
  --date-from 2025-10-07 \
  --date-to 2025-10-14
```

Use `--append` to add more snapshot windows after the first run. The adapter
maps provider event IDs back to local NHL `game_id`s by normalized home/away
teams and nearby game date. Ambiguous provider events are reported and not
written.

OddsPapi historical odds are a per-outcome change log rather than a synchronized
snapshot. The adapter reconstructs complete pre-game two-sided quotes by carrying
forward the latest active price for each side and only writing complete market
pairs.

## CLI

```bash
python3 -m nhl.predictor "Toronto Maple Leafs" "Boston Bruins"
python3 -m nhl.edge --template
python3 -m nhl.edge --model blend --bankroll 250
python3 -m nhl.backtest --results nhl/data/results.csv --model blend
python3 -m nhl.backtest --results nhl/data/results_2025_26.csv --odds-history nhl/data/odds_history.csv
python3 -m nhl.oddspapi --results nhl/data/results_2025_26.csv --date-from 2026-01-01 --date-to 2026-01-07 --max-fixtures 5
python3 -m nhl.the_odds_api --results nhl/data/results_2025_26.csv --date-from 2025-10-07 --date-to 2025-10-14
```

In the desktop app, NHL is discovered automatically through `app/engines/nhl.py`
and supports the Predict and Edge tabs.
