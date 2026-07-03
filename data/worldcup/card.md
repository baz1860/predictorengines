# World Cup 2026 — Best Bets

_Generated 2026-07-03 00:33 · blend model · fitted Elo + Poisson_

The model rates every nation on its full international history, turns each fixture into an expected scoreline, and then weighs its own probabilities against every price it can find. It makes **Argentina** the tournament favourite at 24% to lift the trophy. This week it backs **5 bets** (total stake £20.00) — each explained below, with the model's number, the price, and exactly where the edge comes from. Stakes are fractional-Kelly on a £111 bankroll.

## How the model thinks

Every number below comes from one pipeline, so it's worth knowing what drives it:

1. **Strength (Elo).** Each nation carries an Elo rating built from every international it has played, with bigger swings for World Cups and big wins than for friendlies. The gap between two ratings is the model's core read on who is better, and by how much.
2. **Expected goals (Poisson).** That Elo gap is fed through a goal model fitted on every match since 2010. It turns the gap into an *expected scoreline* — e.g. 1.8 goals for the stronger side, 0.8 for the weaker — with a home-field bump only for the host nations (USA, Mexico, Canada) playing at home.
3. **The full grid (Dixon-Coles).** From those two goal expectations the model builds the probability of every scoreline, nudged to fit how often low-scoring draws really happen. Summing the grid gives win/draw/loss, both-teams-to-score and over/under numbers.
4. **Edge.** A bet is only listed when the model's probability is enough above the bookmaker's implied probability to clear the threshold. The title %s come from running the whole tournament tens of thousands of times, so they fold in group draws and bracket luck, not just raw strength.

## Match bets

Each bet pits the model's probability against the bookmaker's price on an upcoming match; it only fires when its own number is the bigger one. **5 bets cleared the threshold** (total stake £20.00), strongest edge first.

### Over 2.5 goals — Colombia v Ghana
**2.12** · model 57.8% vs market 44.9% · **+12.9pp edge** · stake **£6.09**

The model rates Colombia at Elo 2075 against Ghana's 1693, a 382-point edge to Colombia. The two attacks project to **2.52 + 0.57 = 3.09** expected goals, comfortably above the 2.5 line. That makes Over a **58%** shot, where the price only allows 45% — the market is pricing a tighter game than the model sees.

### Over 2.5 goals — Portugal v Croatia
**2.32** · model 50.7% vs market 40.1% · **+10.6pp edge** · stake **£4.03**

The model rates Portugal at Elo 2046 against Croatia's 1965, a 81-point edge to Portugal. The two attacks project to **1.40 + 1.02 = 2.42** expected goals, just below the 2.5 line. That makes Over a **51%** shot, where the price only allows 40% — the market is pricing a tighter game than the model sees.

### Over 2.5 goals — Switzerland v Algeria
**2.21** · model 53.0% vs market 43.4% · **+9.6pp edge** · stake **£4.22**

The model rates Switzerland at Elo 1983 against Algeria's 1879, a 104-point edge to Switzerland. The two attacks project to **1.46 + 0.98 = 2.44** expected goals, just below the 2.5 line. That makes Over a **53%** shot, where the price only allows 43% — the market is pricing a tighter game than the model sees.

### Colombia win — Colombia v Ghana
**1.43** · model 73.5% vs market 66.8% · **+6.7pp edge** · stake **£3.62**

The model rates Colombia at Elo 2075 against Ghana's 1693, a 382-point edge to Colombia. Run through the goal model that comes out as an expected **2.52–0.57** in Colombia's favour, and once every scoreline is added up Colombia win it **74%** of the time. The 67% price baked into the odds is too generous for a side the model likes this much over Ghana.

### Over 2.5 goals — Argentina v Cape Verde
**1.62** · model 64.3% vs market 58.5% · **+5.9pp edge** · stake **£2.04**

The model rates Argentina at Elo 2211 against Cape Verde's 1704, a 507-point edge to Argentina. The two attacks project to **3.21 + 0.44 = 3.65** expected goals, comfortably above the 2.5 line. That makes Over a **64%** shot, where the price only allows 58% — the market is pricing a tighter game than the model sees.

## Title outlook

**Argentina** head the field: highest Elo in the draw (2211) and champions in **24%** of simulated tournaments. Spain are the closest challenger at 13%, with France, Brazil heading the chasing pack.

These aren't bets — they're the model's read on the title race, straight from the tournament simulation. Each side's chance to lift the trophy already folds in its group draw and likely knockout path, which is why raw Elo order and these numbers don't match exactly.

| Team | Grp | Champion | Reach final |
|---|---|--:|--:|
| Argentina | J | 23.6% | 36% |
| Spain | H | 13.3% | 24% |
| France | I | 12.3% | 23% |
| Brazil | C | 7.4% | 13% |
| England | L | 6.9% | 13% |
| Colombia | K | 6.4% | 13% |
| Mexico | A | 4.4% | 9% |
| Portugal | K | 3.9% | 9% |
| Morocco | C | 3.2% | 7% |
| Netherlands | F | 3.0% | 7% |
| Belgium | G | 2.4% | 6% |
| United States | D | 2.1% | 6% |

## Fixtures forecast

Not bets — the model's read on the next matchday (2026-07-03). For each game: the expected scoreline that falls out of the Elo gap, and where the probability lands.

- **Australia v Egypt** (Elo 1904 v 1852): expected **1.32–1.08**, most likely 1-1 — Australia favoured at 41%.
- **Argentina v Cape Verde** (Elo 2211 v 1704): expected **3.21–0.44**, most likely 3-0 — Argentina favoured at 89%.
- **Colombia v Ghana** (Elo 2075 v 1693): expected **2.52–0.57**, most likely 2-0 — Colombia favoured at 79%.

## Notes

- Bankroll £110.61. Settled 70 bets (40 won), net £+12.57 on a £100 start.
- Model adjustments active this run: totals-calib(lam x1.09).
- Same numbers as charts: `dashboard.html` (`python3 scripts/worldcup/report.py`).
