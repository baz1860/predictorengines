# Tennis Prediction Engine (ATP + WTA)

A tennis betting engine. It does the same four things the World Cup engine does,
on a week-by-week, draw-by-draw basis:

1. **pulls the week's tournament list** (ESPN ATP/WTA scoreboards),
2. **gets the draw** for a tournament,
3. **prices every match with a fitted model** (surface-split Bradley–Terry, with
   an exact Markov-chain set/games simulator and a bracket Monte-Carlo), and
4. **prints the best bets — round by round** (R128 → … → final).

## Use it

See what's on this week, then price one tournament's draw:

```bash
python3 -m tennis.season --schedule                 # live ATP tournaments + draws
python3 -m tennis.season --schedule --tour wta

python3 -m tennis.season                            # price the current ATP draw
python3 -m tennis.season --tour wta --event Berlin  # pick a tournament by name
```

That pulls the draw from ESPN, saves it to `data/draw.csv`, and writes
[`data/card.md`](data/card.md) — the only file you normally read. For every match
in every round it shows the model's pick and win probability; where you've added
book prices it also shows the de-vigged market, the edge, and a staked bet
(**bold** = backed).

```bash
python3 -m tennis.season --no-fetch                 # reprice the saved draw offline
python3 -m tennis.season --min-edge 2               # only count ≥2% edges as bets
```

### Adding prices

Match probabilities come for free; **bets need book odds**, which are yours to
provide. Write a skeleton and fill it in:

```bash
python3 -m tennis.fetch --odds-template             # → data/odds.csv
```

`odds.csv` columns: `tour, surface, best_of, player_a, player_b, odds_a, odds_b`.
Any match in the draw whose names match a row gets priced, blended toward the
market, and staked with fractional Kelly. Rerun `python3 -m tennis.season` and
the backed bets appear in the card.

### First-time setup

Once, to build the match history the model learns from (no API key — Jeff
Sackmann's free archives):

```bash
python3 -m tennis.fetch --seed 2019 2020 2021 2022 2023 2024 2025
python3 -m tennis.model --fit --tour atp
python3 -m tennis.model --fit --tour wta
```

Day to day, `bash tennis/update.sh` accumulates new results and refits both
tours; then `python3 -m tennis.season` is all you run.

## In the app

The **Predict / Simulate / Edge** tabs drive the same engine (head-to-head with
set/games sub-markets, full-draw outright Monte-Carlo, and staked match-winner
edges). `tennis.season` is the command-line equivalent that hands you a whole
draw, round by round, in one page.

---

## Under the hood

`tennis.season` is a thin orchestrator over the modelling, which is where the
quality lives:

| File | Role |
|---|---|
| `season.py` | **the front door**: schedule → draw → model → round-by-round card |
| `providers.py` | Sackmann/TML/MatchCharting history → `matches.csv`; ESPN draw scraper |
| `fetch.py` | `--seed` / `--accumulate` history; `--odds-template` |
| `model.py` | surface-split Bradley–Terry fit (ridge logistic, time-decay) + `predict_match` |
| `simulate.py` | Markov chain (game/set/match, tiebreak) + draw / bracket Monte-Carlo |
| `market.py` | two-way & power de-vig, log-odds market blend, CLV tracking |
| `calibrate.py` | per-market isotonic calibration maps (outright nesting guard) |
| `portfolio.py` | simultaneous-Kelly staking (per-player + total caps, drawdown brake) |
| `validate.py` | walk-forward backtest (match + outright markets) + regression gate |
| `engine.py` | in-process command API the app tabs call |
| `data/` | `matches.csv` (source of truth), `*_model_params.json`, `draw.csv`, `odds.csv`, `card.md`, `calibration.json`, … |

### The model

```
logit P(A beats B) = skill_A − skill_B
                   + surface_offset_A[s] − surface_offset_B[s]
                   + form_weight · (form_A − form_B)
                   + h2h_weight · h2h_log_odds(A, B, s)
```

Fitted by penalised (ridge) logistic regression over a sparse design with
time-decay sample weights (≈52-week half-life), L-BFGS, no scikit-learn. Low
sample players regress to a rank-based prior; surface offsets are kept only above
a minimum sample. ATP and WTA are fitted separately.

The Markov chain gives **exact** game/set/match probabilities from point-on-serve
rates, so set/games sub-markets stay consistent with the match probability. A
matchup-specific serve base (`serve_base()`) sets the total-games regime and a
fitted `games_cal` corrects the idealised model's ~9% over-prediction of totals,
making over/under priceable. The only stochastic layer is the bracket.

### Validation & calibration

```bash
python3 -m tennis.validate --since 2023-01-01 --gate              # match markets
python3 -m tennis.validate --since 2023-01-01 --outright --sims 20000
python3 -m tennis.calibrate --fit                                 # isotonic maps + OOS
```

`validate.py` refits on matches strictly before each retrain date (no
look-ahead), orients matches neutrally, and scores match_winner / set_hcp /
first_set, plus reconstructed-bracket win/final/sf/qf with `--outright`. It writes
the predictions calibration fits on and a baseline for the `--gate` check.
`calibrate.py` reports an honest grouped K-fold out-of-sample Brier improvement;
predict and edge apply calibration and the market blend by default.

See [`plans/tennis_engine_plan.md`](../plans/tennis_engine_plan.md) for the full
design and [`app/engines/tennis.py`](../app/engines/tennis.py) for the adapter.
