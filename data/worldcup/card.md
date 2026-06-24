# World Cup 2026 — Best Bets

_Generated 2026-06-24 20:09 · blend model · fitted Elo + Poisson_

The model rates every nation on its full international history, turns each fixture into an expected scoreline, and then weighs its own probabilities against every price it can find. It makes **Argentina** the tournament favourite at 23% to lift the trophy. This week it backs **7 bets** (total stake £19.46) — each explained below, with the model's number, the price, and exactly where the edge comes from. Stakes are fractional-Kelly on a £113 bankroll.

## How the model thinks

Every number below comes from one pipeline, so it's worth knowing what drives it:

1. **Strength (Elo).** Each nation carries an Elo rating built from every international it has played, with bigger swings for World Cups and big wins than for friendlies. The gap between two ratings is the model's core read on who is better, and by how much.
2. **Expected goals (Poisson).** That Elo gap is fed through a goal model fitted on every match since 2010. It turns the gap into an *expected scoreline* — e.g. 1.8 goals for the stronger side, 0.8 for the weaker — with a home-field bump only for the host nations (USA, Mexico, Canada) playing at home.
3. **The full grid (Dixon-Coles).** From those two goal expectations the model builds the probability of every scoreline, nudged to fit how often low-scoring draws really happen. Summing the grid gives win/draw/loss, both-teams-to-score and over/under numbers.
4. **Edge.** A bet is only listed when the model's probability is enough above the bookmaker's implied probability to clear the threshold. The title %s come from running the whole tournament tens of thousands of times, so they fold in group draws and bracket luck, not just raw strength.

## Match bets

Each bet pits the model's probability against the bookmaker's price on an upcoming match; it only fires when its own number is the bigger one. **7 bets cleared the threshold** (total stake £19.46), strongest edge first.

### Japan win — Japan v Sweden
**1.90** · model 61.3% vs market 50.1% · **+11.3pp edge** · stake **£5.21**

The model rates Japan at Elo 2010 against Sweden's 1796, a 214-point edge to Japan. Run through the goal model that comes out as an expected **1.81–0.79** in Japan's favour, and once every scoreline is added up Japan win it **61%** of the time. The 50% price baked into the odds is too generous for a side the model likes this much over Sweden.

### Under 2.5 goals — Bosnia and Herzegovina v Qatar
**2.42** · model 50.3% vs market 39.0% · **+11.3pp edge** · stake **£4.38**

The model rates Bosnia and Herzegovina at Elo 1669 against Qatar's 1585, a 84-point edge to Bosnia and Herzegovina. Between them the sides project to only **1.41 + 1.01 = 2.42** expected goals, just below the 2.5 line, so the model leans Under at **50%** against the 39% the price implies — it expects a cagier match than the bookmaker.

### Under 2.5 goals — Ecuador v Germany
**2.05** · model 55.1% vs market 46.1% · **+9.1pp edge** · stake **£3.52**

The model rates Germany at Elo 2036 against Ecuador's 1943, a 93-point edge to Germany. Between them the sides project to only **1.00 + 1.43 = 2.43** expected goals, just below the 2.5 line, so the model leans Under at **55%** against the 46% the price implies — it expects a cagier match than the bookmaker.

### Mexico win — Czech Republic v Mexico
**1.92** · model 56.3% vs market 50.1% · **+6.2pp edge** · stake **£2.47**

The model rates Mexico at Elo 2000 against Czech Republic's 1769, a 231-point edge to Mexico. Run through the goal model that comes out as an expected **2.13–0.67** in Mexico's favour, and once every scoreline is added up Mexico win it **56%** of the time. The 50% price baked into the odds is too generous for a side the model likes this much over Czech Republic.

### Over 2.5 goals — Czech Republic v Mexico
**2.05** · model 51.2% vs market 46.3% · **+4.9pp edge** · stake **£1.36**

The model rates Mexico at Elo 2000 against Czech Republic's 1769, a 231-point edge to Mexico. The two attacks project to **0.67 + 2.13 = 2.80** expected goals, above the 2.5 line. That makes Over a **51%** shot, where the price only allows 46% — the market is pricing a tighter game than the model sees.

### Over 2.5 goals — Japan v Sweden
**1.77** · model 58.9% vs market 54.1% · **+4.8pp edge** · stake **£1.45**

The model rates Japan at Elo 2010 against Sweden's 1796, a 214-point edge to Japan. The two attacks project to **1.81 + 0.79 = 2.60** expected goals, just above the 2.5 line. That makes Over a **59%** shot, where the price only allows 54% — the market is pricing a tighter game than the model sees.

### Over 2.5 goals — Scotland v Brazil
**1.83** · model 56.2% vs market 52.2% · **+4.1pp edge** · stake **£1.07**

The model rates Brazil at Elo 2058 against Scotland's 1842, a 216-point edge to Brazil. The two attacks project to **0.78 + 1.82 = 2.60** expected goals, just above the 2.5 line. That makes Over a **56%** shot, where the price only allows 52% — the market is pricing a tighter game than the model sees.

## Title outlook

**Argentina** head the field: highest Elo in the draw (2205) and champions in **23%** of simulated tournaments. Spain are the closest challenger at 16%, with France, England heading the chasing pack.

These aren't bets — they're the model's read on the title race, straight from the tournament simulation. Each side's chance to lift the trophy already folds in its group draw and likely knockout path, which is why raw Elo order and these numbers don't match exactly.

| Team | Grp | Champion | Reach final |
|---|---|--:|--:|
| Argentina | J | 23.2% | 34% |
| Spain | H | 15.6% | 26% |
| France | I | 8.7% | 17% |
| England | L | 7.1% | 13% |
| Brazil | C | 5.7% | 11% |
| Colombia | K | 5.5% | 11% |
| Mexico | A | 3.8% | 8% |
| Germany | E | 3.5% | 9% |
| Portugal | K | 3.3% | 7% |
| United States | D | 3.2% | 9% |
| Morocco | C | 3.1% | 7% |
| Netherlands | F | 2.9% | 7% |

## Fixtures forecast

Not bets — the model's read on the next matchday (2026-06-24). For each game: the expected scoreline that falls out of the Elo gap, and where the probability lands.

- **Mexico v Czech Republic** (Elo 2000 v 1769): expected **2.13–0.67**, most likely 2-0 — Mexico favoured at 70%.
- **South Africa v South Korea** (Elo 1664 v 1883): expected **0.78–1.83**, most likely 0-1 — South Korea favoured at 61%.
- **Canada v Switzerland** (Elo 1889 v 1953): expected **1.20–1.19**, most likely 1-1 — Canada favoured at 35%.
- **Bosnia and Herzegovina v Qatar** (Elo 1669 v 1585): expected **1.41–1.01**, most likely 1-1 — Bosnia and Herzegovina favoured at 45%.
- **Scotland v Brazil** (Elo 1842 v 2058): expected **0.78–1.82**, most likely 0-1 — Brazil favoured at 61%.
- **Morocco v Haiti** (Elo 2012 v 1661): expected **2.37–0.60**, most likely 2-0 — Morocco favoured at 76%.

## Notes

- Bankroll £113.44. Settled 47 bets (25 won), net £+16.46 on a £100 start.
- Model adjustments active this run: totals-calib(lam x1.09).
- Same numbers as charts: `dashboard.html` (`python3 scripts/worldcup/report.py`).
