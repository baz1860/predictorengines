# World Cup 2026 — Best Bets

_Generated 2026-07-02 09:17 · blend model · fitted Elo + Poisson_

The model rates every nation on its full international history, turns each fixture into an expected scoreline, and then weighs its own probabilities against every price it can find. It makes **Argentina** the tournament favourite at 24% to lift the trophy. This week it backs **6 bets** (total stake £23.14) — each explained below, with the model's number, the price, and exactly where the edge comes from. Stakes are fractional-Kelly on a £111 bankroll.

## How the model thinks

Every number below comes from one pipeline, so it's worth knowing what drives it:

1. **Strength (Elo).** Each nation carries an Elo rating built from every international it has played, with bigger swings for World Cups and big wins than for friendlies. The gap between two ratings is the model's core read on who is better, and by how much.
2. **Expected goals (Poisson).** That Elo gap is fed through a goal model fitted on every match since 2010. It turns the gap into an *expected scoreline* — e.g. 1.8 goals for the stronger side, 0.8 for the weaker — with a home-field bump only for the host nations (USA, Mexico, Canada) playing at home.
3. **The full grid (Dixon-Coles).** From those two goal expectations the model builds the probability of every scoreline, nudged to fit how often low-scoring draws really happen. Summing the grid gives win/draw/loss, both-teams-to-score and over/under numbers.
4. **Edge.** A bet is only listed when the model's probability is enough above the bookmaker's implied probability to clear the threshold. The title %s come from running the whole tournament tens of thousands of times, so they fold in group draws and bracket luck, not just raw strength.

## Match bets

Each bet pits the model's probability against the bookmaker's price on an upcoming match; it only fires when its own number is the bigger one. **6 bets cleared the threshold** (total stake £23.14), strongest edge first.

### Over 2.5 goals — Colombia v Ghana
**2.22** · model 57.8% vs market 42.9% · **+14.9pp edge** · stake **£7.02**

The model rates Colombia at Elo 2075 against Ghana's 1693, a 382-point edge to Colombia. The two attacks project to **2.52 + 0.57 = 3.09** expected goals, comfortably above the 2.5 line. That makes Over a **58%** shot, where the price only allows 43% — the market is pricing a tighter game than the model sees.

### Colombia win — Colombia v Ghana
**1.52** · model 73.5% vs market 63.0% · **+10.5pp edge** · stake **£6.83**

The model rates Colombia at Elo 2075 against Ghana's 1693, a 382-point edge to Colombia. Run through the goal model that comes out as an expected **2.52–0.57** in Colombia's favour, and once every scoreline is added up Colombia win it **74%** of the time. The 63% price baked into the odds is too generous for a side the model likes this much over Ghana.

### Over 2.5 goals — Switzerland v Algeria
**2.17** · model 53.0% vs market 44.0% · **+9.0pp edge** · stake **£3.82**

The model rates Switzerland at Elo 1983 against Algeria's 1879, a 104-point edge to Switzerland. The two attacks project to **1.46 + 0.98 = 2.44** expected goals, just below the 2.5 line. That makes Over a **53%** shot, where the price only allows 44% — the market is pricing a tighter game than the model sees.

### Over 2.5 goals — Argentina v Cape Verde
**1.67** · model 64.3% vs market 56.8% · **+7.5pp edge** · stake **£3.34**

The model rates Argentina at Elo 2211 against Cape Verde's 1704, a 507-point edge to Argentina. The two attacks project to **3.21 + 0.44 = 3.65** expected goals, comfortably above the 2.5 line. That makes Over a **64%** shot, where the price only allows 57% — the market is pricing a tighter game than the model sees.

### Over 2.5 goals — Spain v Austria
**1.73** · model 56.9% vs market 54.8% · **+2.0pp edge** · stake **£0.86**

The model rates Spain at Elo 2197 against Austria's 1897, a 300-point edge to Spain. The two attacks project to **2.14 + 0.67 = 2.81** expected goals, above the 2.5 line. That makes Over a **57%** shot, where the price only allows 55% — the market is pricing a tighter game than the model sees.

### Argentina win — Argentina v Cape Verde
**1.17** · model 83.8% vs market 82.8% · **+1.0pp edge** · stake **£1.27**

The model rates Argentina at Elo 2211 against Cape Verde's 1704, a 507-point edge to Argentina. Run through the goal model that comes out as an expected **3.21–0.44** in Argentina's favour, and once every scoreline is added up Argentina win it **84%** of the time. The 83% price baked into the odds is too generous for a side the model likes this much over Cape Verde.

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

Not bets — the model's read on the next matchday (2026-07-02). For each game: the expected scoreline that falls out of the Elo gap, and where the probability lands.

- **Spain v Austria** (Elo 2197 v 1897): expected **2.14–0.67**, most likely 2-0 — Spain favoured at 71%.
- **Portugal v Croatia** (Elo 2046 v 1965): expected **1.40–1.02**, most likely 1-1 — Portugal favoured at 44%.
- **Switzerland v Algeria** (Elo 1983 v 1879): expected **1.46–0.98**, most likely 1-1 — Switzerland favoured at 47%.

## Notes

- Bankroll £110.61. Settled 70 bets (40 won), net £+12.57 on a £100 start.
- Model adjustments active this run: totals-calib(lam x1.09).
- Same numbers as charts: `dashboard.html` (`python3 scripts/worldcup/report.py`).
