# World Cup 2026 — Best Bets

_Generated 2026-06-30 08:58 · blend model · fitted Elo + Poisson_

The model rates every nation on its full international history, turns each fixture into an expected scoreline, and then weighs its own probabilities against every price it can find. It makes **Argentina** the tournament favourite at 24% to lift the trophy. This week it backs **5 bets** (total stake £4.05) — each explained below, with the model's number, the price, and exactly where the edge comes from. Stakes are fractional-Kelly on a £110 bankroll.

## How the model thinks

Every number below comes from one pipeline, so it's worth knowing what drives it:

1. **Strength (Elo).** Each nation carries an Elo rating built from every international it has played, with bigger swings for World Cups and big wins than for friendlies. The gap between two ratings is the model's core read on who is better, and by how much.
2. **Expected goals (Poisson).** That Elo gap is fed through a goal model fitted on every match since 2010. It turns the gap into an *expected scoreline* — e.g. 1.8 goals for the stronger side, 0.8 for the weaker — with a home-field bump only for the host nations (USA, Mexico, Canada) playing at home.
3. **The full grid (Dixon-Coles).** From those two goal expectations the model builds the probability of every scoreline, nudged to fit how often low-scoring draws really happen. Summing the grid gives win/draw/loss, both-teams-to-score and over/under numbers.
4. **Edge.** A bet is only listed when the model's probability is enough above the bookmaker's implied probability to clear the threshold. The title %s come from running the whole tournament tens of thousands of times, so they fold in group draws and bracket luck, not just raw strength.

## Match bets

Each bet pits the model's probability against the bookmaker's price on an upcoming match; it only fires when its own number is the bigger one. **5 bets cleared the threshold** (total stake £4.05), strongest edge first.

### United States win — United States v Bosnia and Herzegovina
**1.38** · model 71.5% vs market 70.0% · **+1.6pp edge** · stake **£0.98**

The model rates United States at Elo 1900 against Bosnia and Herzegovina's 1703, a 197-point edge to United States. Run through the goal model that comes out as an expected **1.99–0.72** in United States's favour — and the host-nation home bump on top, and once every scoreline is added up United States win it **72%** of the time. The 70% price baked into the odds is too generous for a side the model likes this much over Bosnia and Herzegovina.

### Over 2.5 goals — United States v Bosnia and Herzegovina
**1.82** · model 53.8% vs market 52.4% · **+1.5pp edge** · stake **£0.74**

The model rates United States at Elo 1900 against Bosnia and Herzegovina's 1703, a 197-point edge to United States. The two attacks project to **1.99 + 0.72 = 2.71** expected goals, above the 2.5 line. That makes Over a **54%** shot, where the price only allows 52% — the market is pricing a tighter game than the model sees.

### France win — France v Sweden
**1.29** · model 75.5% vs market 74.6% · **+0.9pp edge** · stake **£1.03**

The model rates France at Elo 2175 against Sweden's 1812, a 363-point edge to France. Run through the goal model that comes out as an expected **2.43–0.59** in France's favour, and once every scoreline is added up France win it **76%** of the time. The 75% price baked into the odds is too generous for a side the model likes this much over Sweden.

### Mexico win — Mexico v Ecuador
**2.23** · model 43.7% vs market 43.0% · **+0.7pp edge** · stake **£0.60**

The model rates Mexico at Elo 2016 against Ecuador's 1981, a 35-point edge to Mexico. Run through the goal model that comes out as an expected **1.45–0.98** in Mexico's favour — and the host-nation home bump on top, and once every scoreline is added up Mexico win it **44%** of the time. The 43% price baked into the odds is too generous for a side the model likes this much over Ecuador.

### Under 2.5 goals — England v DR Congo
**1.88** · model 51.2% vs market 50.7% · **+0.5pp edge** · stake **£0.70**

The model rates England at Elo 2100 against DR Congo's 1811, a 289-point edge to England. Between them the sides project to only **2.10 + 0.68 = 2.78** expected goals, above the 2.5 line, so the model leans Under at **51%** against the 51% the price implies — it expects a cagier match than the bookmaker.

## Title outlook

**Argentina** head the field: highest Elo in the draw (2211) and champions in **24%** of simulated tournaments. Spain are the closest challenger at 14%, with France, Colombia heading the chasing pack.

These aren't bets — they're the model's read on the title race, straight from the tournament simulation. Each side's chance to lift the trophy already folds in its group draw and likely knockout path, which is why raw Elo order and these numbers don't match exactly.

| Team | Grp | Champion | Reach final |
|---|---|--:|--:|
| Argentina | J | 24.5% | 38% |
| Spain | H | 14.4% | 26% |
| France | I | 12.3% | 22% |
| Colombia | K | 7.3% | 13% |
| Brazil | C | 6.1% | 13% |
| England | L | 5.9% | 13% |
| Portugal | K | 4.3% | 9% |
| Netherlands | F | 3.7% | 8% |
| Mexico | A | 3.2% | 7% |
| Morocco | C | 2.8% | 7% |
| Belgium | G | 2.6% | 5% |
| Germany | E | 2.0% | 5% |

## Fixtures forecast

Not bets — the model's read on the next matchday (2026-06-30). For each game: the expected scoreline that falls out of the Elo gap, and where the probability lands.

- **Ivory Coast v Norway** (Elo 1864 v 1979): expected **0.95–1.50**, most likely 1-1 — Norway favoured at 48%.
- **France v Sweden** (Elo 2175 v 1812): expected **2.43–0.59**, most likely 2-0 — France favoured at 77%.
- **Mexico v Ecuador** (Elo 2016 v 1981): expected **1.45–0.98**, most likely 1-1 — Mexico favoured at 47%.

## Notes

- Bankroll £109.87. Settled 63 bets (36 won), net £+12.89 on a £100 start.
- Model adjustments active this run: calibrated+market-blend(w=0.16,1X2+OU+BTTS)+totals-calib(lam x1.09)+context+stakes(coef=0.15)+squad-adj.
- Same numbers as charts: `dashboard.html` (`python3 scripts/worldcup/report.py`).
