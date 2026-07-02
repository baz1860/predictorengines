# NHL Prediction Engine

Local-first NHL engine for match win probability, projected goals, puck-line,
totals, edge finding, staking, and settlement through the shared app ledger.

## Model

The first version uses `nhl/data/team_stats.csv` as a baseline. It converts each
team's goals, shots, power play, penalty kill, save percentage, and points share
into attack, defence-allowed, and form ratings. Match pricing uses expected
regulation goals, independent Poisson score distributions, and an overtime split
for moneyline probabilities.

Markets supported:

- `ml`: home/away moneyline
- `spread`: NHL puck line, typically home `-1.5` / away `+1.5`
- `total`: over/under goals

The included CSVs are runnable seed data, not an official historical database.
Replace `team_stats.csv`, `fixtures.csv`, `results.csv`, and `odds.csv` with
your preferred NHL data source before treating outputs as live analysis.

## CLI

```bash
python3 -m nhl.predictor "Toronto Maple Leafs" "Boston Bruins"
python3 -m nhl.edge --template
python3 -m nhl.edge --model blend --bankroll 250
python3 -m nhl.backtest --results nhl/data/results.csv --model blend
```

In the desktop app, NHL is discovered automatically through `app/engines/nhl.py`
and supports the Predict and Edge tabs.
