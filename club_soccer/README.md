# Club Soccer Engine

Predicts club football matches across the requested UK, European top-flight,
European club, and primary domestic cup competitions.

## Data

`data/fixtures.csv` is the source of truth. It can be maintained manually or
refreshed from API-Football:

```bash
python3 club_soccer/fetch.py --season 2025 --current
```

API keys are read from `data/api_keys.json` using the `api-football` key, or from
`API_FOOTBALL_KEY`, or from an explicit `--api-key`.

## Model

```bash
python3 club_soccer/model.py --fit
python3 club_soccer/model.py "Arsenal" "Chelsea" --competition "Premier League"
```

Models:

- `ensemble` - default blend of goals, Elo, and shot-form proxies.
- `goals` - attack/defence Poisson.
- `elo` - club Elo translated to expected goals.

### Per-competition home advantage + rho (fitted, default OFF)

`fit()` also estimates a per-competition home-advantage multiplier and Dixon-Coles
rho (Empirical-Bayes shrunk toward the global value) and stores them under
`comp_adj`. Applying them is gated by `comp_adj_active` (default `False`): a
walk-forward over ≈16.5k predictions found per-competition HFA + rho neutral-to-
slightly-worse on held-out Brier (0.61207 global vs 0.61216–0.61234), so the
validated global constants (`HOME_ADV_ELO`, `DC_RHO`) remain the default. The
fitted table is kept for inspection and auto-activates if a future gate sets the
flag. With the flag off, predictions are identical to the previous global model.

## Edge

```bash
python3 club_soccer/edge.py --template
python3 club_soccer/edge.py
python3 club_soccer/edge.py --api-odds
```

Markets: 1X2, over/under 2.5, and BTTS.

## Validation

```bash
python3 club_soccer/validate.py
python3 club_soccer/validate.py --gate
```

The first validation run writes `data/validation_baseline.json`; later gate runs
compare Brier score against that baseline.
