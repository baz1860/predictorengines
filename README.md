# Sports Predictor

A local-first sports prediction and betting-analysis suite for World Cup 2026,
club soccer, college football, and PGA golf.

The project combines model-driven predictions, market/odds comparison,
bankroll tracking, validation gates, and a desktop-style web UI. It is designed
for research, paper trading, and model evaluation rather than blind wagering.

> Betting markets are noisy and sharp. Treat edge output as analytical
> information, not financial advice.

## What Is Included

- **Desktop app**: FastAPI backend + PyWebView/macOS launcher + dependency-free
  HTML/CSS/JS frontend in `app/`.
- **World Cup 2026 engine**: Elo/Poisson and Dixon-Coles blend, tournament
  simulation, squad/context adjustments, edge finding, CLV, and bankroll tools.
- **Club soccer engine**: goals/Elo/shot-form ensemble for domestic and European
  club competitions.
- **College football engine**: Elo + power-rating blend for win probability,
  spreads, totals, and win-total projections.
- **Golf engine**: PGA/majors round-history model, Monte Carlo tournament
  simulation, calibrated/market-anchored edge pricing, matchup and placement
  markets.
- **Shared suite layer**: engine contracts, suite ledger, pooled bankroll,
  portfolio caps, settlement, dashboard payloads, validation gates, and
  security checks.

## Quick Start

Clone the repository and install the app dependencies:

```bash
git clone https://github.com/baz1860/predictorengines.git
cd predictorengines
python3 -m pip install -r app/requirements.txt
```

Run the desktop app:

```bash
python3 -m app.main
```

Or run the backend for browser development:

```bash
uvicorn app.server:app --port 8765
```

Then open `http://127.0.0.1:8765`.

On macOS, the included `Sports Predictor.app` launcher can start the app without
opening a terminal, provided it remains inside the project folder and your
Python environment has the dependencies installed.

## Repository Layout

```text
.
├── app/                  # Desktop/web app, API, adapters, shared suite logic
│   ├── engines/          # Engine adapters and isolated subprocess runners
│   └── web/              # Frontend shell, charts, styles
├── club_soccer/          # Club soccer model, fetchers, calibration, edge
├── cfb/                  # College football models, validation, edge, data
├── golf/                 # Golf model, providers, simulation, edge, validation
├── data/                 # World Cup data, suite ledger, shared model artifacts
├── predictor.py          # World Cup match prediction CLI
├── simulate.py           # World Cup tournament simulation
├── edge.py               # World Cup odds/edge CLI
├── validate_all.py       # Cross-engine validation gate
├── preflight.py          # Offline readiness/key/data check
└── *_PLAN.md / *_NOTES.md
```

Engine-specific details live in:

- [app/README.md](app/README.md)
- [club_soccer/README.md](club_soccer/README.md)
- [cfb/README.md](cfb/README.md)
- [golf/README.md](golf/README.md)
- [docs/archive/V3_PLAN.md](docs/archive/V3_PLAN.md) and [docs/archive/V3_NOTES.md](docs/archive/V3_NOTES.md)

## Desktop App

The app is capability-driven. Each engine adapter declares what it supports
(`predict`, `simulate`, `edge`), and the UI renders the relevant tabs and forms
from that schema.

Main app routes:

- `/api/engines` - registered engines and schemas
- `/api/predict` - engine-specific predictions
- `/api/simulate` - tournament/event simulations
- `/api/edge` - odds comparison and recommended stakes
- `/api/bankroll` - shared bankroll status, settle, reset
- `/api/dashboard`, `/api/history`, `/api/fixtures`, `/api/outrights`

## CLI Examples

World Cup match prediction:

```bash
python3 predictor.py "Brazil" "Morocco"
python3 predictor.py "Mexico" "South Africa" --home
python3 predictor.py --worldcup
```

World Cup tournament simulation:

```bash
python3 simulate.py
python3 simulate.py -n 50000
```

World Cup edge report:

```bash
python3 edge.py --template
# fill odds.csv with decimal odds, then:
python3 edge.py --calibrated --market-blend --context
```

Club soccer:

```bash
python3 club_soccer/model.py "Arsenal" "Chelsea" --competition "Premier League"
python3 club_soccer/edge.py --template
python3 club_soccer/validate.py --gate
```

College football:

```bash
python3 -m cfb.predictor "Ohio State" "Michigan"
python3 -m cfb.predictor "Georgia" "Texas" --neutral --model blend
python3 -m cfb.validate --quiet --gate
```

Golf:

```bash
python3 golf/model.py --fit
python3 golf/simulate.py --sims 50000
python3 golf/edge.py --min-edge 1.0
```

## API Keys

Live odds and injury/data fetchers can read keys from `data/api_keys.json`, which
is ignored by Git. Start from the example file:

```bash
cp data/api_keys.example.json data/api_keys.json
```

Expected shape:

```json
{
  "the-odds-api": "your_key",
  "api-football": "your_key",
  "datagolf": "your_key"
}
```

Explicit CLI flags and environment variables can also be used:

- `THE_ODDS_API_KEY`
- `API_FOOTBALL_KEY`
- `DG_API_KEY`

Do not commit real API keys. GitHub secret scanning is enabled on public repos
and will block pushes containing provider-shaped tokens.

## Validation And Tests

One command runs every fast offline test suite (contract, security, bankroll,
per-milestone, and the V3 suites) and, with `--gates`, the per-engine validation
gates:

```bash
python3 run_checks.py              # all fast suites (~13s)
python3 run_checks.py --gates      # also run validation gates (~1 min)
```

Individual fast checks and the full cross-engine gate also run standalone:

```bash
python3 test_engines_contract.py
python3 validate_all.py --gate --sims 4000
```

Preflight readiness report:

```bash
python3 preflight.py            # offline data/key check
python3 preflight.py --json
```

The validation system is intentionally conservative: model changes should pass
engine-specific gates before being treated as defaults.

## Suite Tooling (V3)

Offline operations helpers shared across engines:

```bash
python3 daily_summary.py             # gates, freshness, recommendations, CLV, bankroll
python3 clv_suite.py --snapshot      # record current odds for open bets
python3 clv_suite.py --report        # closing-line value per settled bet
python3 -m app.provenance --freshness     # per-engine data-staleness warnings
python3 -m app.provenance --check-odds cfb  # validate a manual odds file
python3 -m cfb.validate --tune-blend       # CFB elo/power blend-weight table
```

Market blending is generalised in `app/market_blend.py` (the World Cup keeps its
own 1X2 blend; Club Soccer and CFB expose an experimental, default-OFF blend).
Per-engine data manifests and the model-audit panel are surfaced in the app.

## Data And Generated Files

The repository includes several local datasets and fitted model artifacts so the
engines can run offline. Generated reports, local app settings, local API keys,
Python caches, backup ledgers, and launcher logs are ignored by `.gitignore`.

Important data sources include:

- World Cup/international results: `data/results.csv`
- CFB games and lines: `cfb/data/`
- Golf round history: `golf/data/rounds.csv`
- Club soccer fixtures/model artifacts: `club_soccer/data/`

Refresh scripts exist per engine (`update.sh`, `golf/update.sh`,
`club_soccer/update.sh`) but may require local API keys depending on the source.

## Current Direction

The current product direction is the local web/desktop app, not a native Swift
rewrite. V3 is complete (see [docs/archive/V3_NOTES.md](docs/archive/V3_NOTES.md)):

- **M1–M4** — shared engine contracts, request/subprocess hardening, validation
  gates, pooled bankroll/ledger with event-safe settlement.
- **M5** — generalised market blending + suite-level CLV (`clv_suite.py`).
- **M6** — gated, validated modelling upgrades (CFB tunable blend weight; other
  engines measured and left at validated defaults).
- **M7** — data-provenance manifests, freshness warnings, manual-odds checks.
- **M8** — power-user UX: model-audit panel, edge filters, dry-run/record split,
  CSV export.
- **M9** — `run_checks.py`, `daily_summary.py`, and refreshed docs.

Near-term refactor targets:

- split backend routes by responsibility;
- factor shared adapter edge/recording helpers;
- separate bankroll state, ledger IO, and settlement;
- modularize `app/web/app.js`.

## License

No license is currently declared. Until a license is added, all rights are
reserved by the repository owner.
