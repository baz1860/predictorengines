# Tennis Prediction Engine (ATP + WTA)

A tennis betting engine. It does the same four things the World Cup engine does,
on a week-by-week, event-by-event basis:

1. **pulls all active tournaments** (ESPN ATP/WTA scoreboards),
2. **gets every draw** for the selected tour,
3. **prices every match with a fitted model** (surface-split Bradley–Terry, with
   an exact Markov-chain set/games simulator and a bracket Monte-Carlo), and
4. **prints the best bets — round by round** (R128 → … → final).

## Use it

See what's on this week, then price every active event for a tour:

```bash
python3 -m tennis.season --schedule                 # live ATP tournaments + draws
python3 -m tennis.season --schedule --tour wta

python3 -m tennis.season                            # all active ATP events
python3 -m tennis.season --tour wta                 # all active WTA events
python3 -m tennis.season --tour both                # both tours, all active events
python3 -m tennis.season --tour wta --event Berlin  # narrow to one event
python3 -m tennis.season --event Wimbledon --odds-api # also pull book prices
```

That pulls all matching draws from ESPN, including completed matches and their
winners, saves them to `data/draw.csv`, and writes
[`data/card.md`](data/card.md) — the only file you normally read. For every match
in every round it shows the model's pick and win probability; where you've added
book prices it also shows the de-vigged market, the edge, and a staked bet
(**bold** = backed).

With `--tour both`, the index links to separate [`data/card_atp.md`](data/card_atp.md)
and [`data/card_wta.md`](data/card_wta.md) cards so one tour cannot overwrite the
other.

```bash
python3 -m tennis.season --no-fetch                 # reprice the saved draw offline
python3 -m tennis.season --min-edge 2               # only count ≥2% edges as bets
```

### Adding prices

Match probabilities come for free; **bets need book odds**, which are yours to
provide or fetch from The Odds API:

```bash
python3 -m tennis.fetch --odds-api --tours atp --event Wimbledon
python3 -m tennis.fetch --odds-api --tours wta --event Wimbledon
```

You can also do it in the normal card run:

```bash
python3 -m tennis.season --tour atp --event Wimbledon --odds-api --min-edge 2
```

For manual entry, write a skeleton and fill it in:

```bash
python3 -m tennis.fetch --odds-template             # → data/odds.csv
```

`draw.csv` columns: `tour, tourney_name, event_id, surface, best_of, round,
player_a, player_b, state, winner, score, match_id`. `tourney_name` and
`event_id` keep simultaneous events separate. `state=post` plus `winner` locks
a completed result; future rounds are simulated from the surviving players.

`odds.csv` columns: `tour, tourney_name, event_id, surface, best_of, player_a,
player_b, odds_a, odds_b`. Event names are used when matching prices, with a
blank event retained as a backwards-compatible fallback.
Any match in the draw whose names match a row gets priced, blended toward the
market, and staked with fractional Kelly. Rerun `python3 -m tennis.season` and
the backed bets appear in the card.

### First-time setup

Once, to build the match history the model learns from:

```bash
python3 -m tennis.fetch --seed 2019 2020 2021 2022 2023 2024 2025
python3 -m tennis.model --fit --tour atp
python3 -m tennis.model --fit --tour wta
```

ATP history uses TML's free season archives. WTA history uses the public WTA
results JSON feed (completed singles, winners, scores, rounds and surfaces),
cached under `data/api_cache/`; Sackmann and MatchCharting remain offline
fallbacks. The WTA fit is therefore trained on real tour results rather than
the old heuristic-only seed.

Day to day, `bash tennis/update.sh` accumulates new results and refits both
tours; then run `python3 -m tennis.season --tour atp` or
`python3 -m tennis.season --tour wta` to refresh and price all active events.

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
| `providers.py` | TML/WTA official history → `matches.csv`; ESPN multi-event draw scraper |
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
