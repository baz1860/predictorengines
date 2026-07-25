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
[`data/card_atp.md`](data/card_atp.md) or [`data/card_wta.md`](data/card_wta.md).
For every match
in every round it shows the model's pick and win probability; where you've added
book prices it also shows the de-vigged market, the edge, and a staked bet
(**bold** = backed).

With `--tour both`, [`data/card.md`](data/card.md) is an index linking to separate [`data/card_atp.md`](data/card_atp.md)
and [`data/card_wta.md`](data/card_wta.md) cards so one tour cannot overwrite the
other.

```bash
python3 -m tennis.season --no-fetch                 # reprice the saved draw offline
python3 -m tennis.season --min-edge 0.02            # only count ≥2% EV edges as bets
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
python3 -m tennis.season --tour atp --event Wimbledon --odds-api --min-edge 0.02
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

Current-season ATP results use ESPN's free, no-key scoreboard JSON. The first
run downloads the season to date; later runs refresh a rolling 21-day overlap,
merge corrected events into `data/api_cache/espn_atp_<year>.json`, and append
only unseen completed singles to `matches.csv`. Completed ATP seasons already
cached from TML remain the historical source. WTA history uses the public WTA
results JSON feed; Sackmann and MatchCharting remain offline fallbacks.

Day to day, `bash tennis/update.sh` accumulates new results and refits both
tours; then run `python3 -m tennis.season --tour atp` or
`python3 -m tennis.season --tour wta` to refresh and price all active events.
The update is fail-fast: a tour more than 21 days behind its fit date stops the
pipeline and saved stale parameters are refused by prediction/card commands.

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
| `providers.py` | ESPN current ATP + TML history/WTA API → `matches.csv`; ESPN draw scraper |
| `fetch.py` | `--seed` / `--accumulate` history; `--odds-template` |
| `model.py` | surface-split Bradley–Terry fit (ridge logistic, time-decay) + `predict_match` |
| `simulate.py` | Markov chain (game/set/match, tiebreak) + draw / bracket Monte-Carlo |
| `rounds.py` | shared round vocabulary, inference and draw schema |
| `market.py` | two-way de-vig and log-odds market blend |
| `calibrate.py` | orientation-safe match-winner isotonic calibration |
| `portfolio.py` | simultaneous-Kelly staking (per-player + total caps, drawdown brake) |
| `validate.py` | walk-forward backtest (match + outright markets) + regression gate |
| `engine.py` | in-process command API the app tabs call |
| `data/` | `matches.csv` (source of truth), `*_model_params.json`, `draw.csv`, `odds.csv`, per-tour cards, `calibration.json`, … |

### The model

```
logit P(A beats B) = skill_A − skill_B
                   + surface_offset_A[s] − surface_offset_B[s]
```

Fitted by penalised (ridge) logistic regression over a sparse design with
time-decay sample weights (≈52-week half-life), L-BFGS, no scikit-learn. Low
sample ATP players regress to a rank-based prior; the WTA prior is explicitly
zero-centred because the current WTA results feed has no ranking fields. Surface
offsets are kept only above a minimum sample. ATP and WTA are fitted separately.

The Markov chain gives **exact** game/set/match probabilities from point-on-serve
rates, so set/games sub-markets stay consistent with the match probability. A
matchup-specific serve base (`serve_base()`) sets the total-games regime, with
separate ATP/WTA fallbacks when serve stats are absent. A fitted `games_cal`
corrects the remaining level error. The only stochastic layer is the bracket,
where best-of-5 increases the stronger player's advancement probability.

### Validation & calibration

```bash
python3 -m tennis.validate --since 2023-01-01 --gate              # match markets
python3 -m tennis.validate --since 2023-01-01 --outright --sims 20000
python3 -m tennis.calibrate --fit                                 # isotonic maps + OOS
```

`validate.py` refits on matches strictly before each retrain date (no
look-ahead), orients matches neutrally, and scores match_winner / set_hcp /
first_set alongside rank-logistic and Elo controls, plus reconstructed-bracket
win/final/sf/qf with `--outright`. It writes the predictions used by calibration
and a baseline for the `--gate` check. `calibrate.py` fits only match-winner in
fold-sorted orientation; predict then re-inverts the Markov chain so first-set
and handicap probabilities remain consistent. Edge and card staking share the
same portfolio caps and fractional edge units.

See [`plans/tennis_engine_plan.md`](../plans/tennis_engine_plan.md) for the full
design and [`app/engines/tennis.py`](../app/engines/tennis.py) for the adapter.
