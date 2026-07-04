# World Cup 2026 — Best Bets

_Generated 2026-07-04 12:44 · blend model · fitted Elo + Poisson_

The model rates every nation on its full international history, turns each fixture into an expected scoreline, and then weighs its own probabilities against every price it can find. It makes **Argentina** the tournament favourite at 23% to lift the trophy. This week it backs **4 bets** (total stake £3.40) — each explained below, with the model's number, the price, and exactly where the edge comes from. Stakes are fractional-Kelly on a £107 bankroll.

## How the model thinks

Every number below comes from one pipeline, so it's worth knowing what drives it:

1. **Strength (Elo).** Each nation carries an Elo rating built from every international it has played, with bigger swings for World Cups and big wins than for friendlies. The gap between two ratings is the model's core read on who is better, and by how much.
2. **Expected goals (Poisson).** That Elo gap is fed through a goal model fitted on every match since 2010. It turns the gap into an *expected scoreline* — e.g. 1.8 goals for the stronger side, 0.8 for the weaker — with a home-field bump only for the host nations (USA, Mexico, Canada) playing at home.
3. **The full grid (Dixon-Coles).** From those two goal expectations the model builds the probability of every scoreline, nudged to fit how often low-scoring draws really happen. Summing the grid gives win/draw/loss, both-teams-to-score and over/under numbers.
4. **Edge.** A bet is only listed when the model's probability is enough above the bookmaker's implied probability to clear the threshold. The title %s come from running the whole tournament tens of thousands of times, so they fold in group draws and bracket luck, not just raw strength.

## Match bets

Each bet pits the model's probability against the bookmaker's price on an upcoming match; it only fires when its own number is the bigger one. **4 bets cleared the threshold** (total stake £3.40), strongest edge first.

### Over 2.5 goals — United States v Belgium
**1.77** · model 56.9% vs market 53.7% · **+3.2pp edge** · stake **£0.86**

The model rates Belgium at Elo 1976 against United States's 1916, a 60-point edge to Belgium. The two attacks project to **1.21 + 1.18 = 2.39** expected goals, just below the 2.5 line. That makes Over a **57%** shot, where the price only allows 54% — the market is pricing a tighter game than the model sees.

### Under 2.5 goals — Canada v Morocco
**1.68** · model 58.8% vs market 56.8% · **+2.1pp edge** · stake **£0.89**

The model rates Morocco at Elo 2025 against Canada's 1877, a 148-point edge to Morocco. Between them the sides project to only **0.90 + 1.59 = 2.49** expected goals, just below the 2.5 line, so the model leans Under at **59%** against the 57% the price implies — it expects a cagier match than the bookmaker.

### Over 2.5 goals — Brazil v Norway
**1.75** · model 56.4% vs market 54.4% · **+2.0pp edge** · stake **£0.84**

The model rates Brazil at Elo 2104 against Norway's 1999, a 105-point edge to Brazil. The two attacks project to **1.47 + 0.97 = 2.44** expected goals, just below the 2.5 line. That makes Over a **56%** shot, where the price only allows 54% — the market is pricing a tighter game than the model sees.

### Brazil win — Brazil v Norway
**1.82** · model 53.2% vs market 52.7% · **+0.5pp edge** · stake **£0.81**

The model rates Brazil at Elo 2104 against Norway's 1999, a 105-point edge to Brazil. Run through the goal model that comes out as an expected **1.47–0.97** in Brazil's favour, and once every scoreline is added up Brazil win it **53%** of the time. The 53% price baked into the odds is too generous for a side the model likes this much over Norway.

## Title outlook

**Argentina** head the field: highest Elo in the draw (2214) and champions in **23%** of simulated tournaments. Spain are the closest challenger at 15%, with France, England heading the chasing pack.

These aren't bets — they're the model's read on the title race, straight from the tournament simulation. Each side's chance to lift the trophy already folds in its group draw and likely knockout path, which is why raw Elo order and these numbers don't match exactly.

| Team | Grp | Champion | Reach final |
|---|---|--:|--:|
| Argentina | J | 23.2% | 36% |
| Spain | H | 15.4% | 26% |
| France | I | 12.4% | 23% |
| England | L | 7.0% | 13% |
| Colombia | K | 6.6% | 13% |
| Brazil | C | 5.3% | 10% |
| Portugal | K | 4.9% | 10% |
| Mexico | A | 4.4% | 9% |
| Netherlands | F | 2.8% | 7% |
| Morocco | C | 2.8% | 7% |
| Belgium | G | 2.2% | 6% |
| Germany | E | 1.8% | 5% |

## Fixtures forecast

Not bets — the model's read on the next matchday (2026-07-04). For each game: the expected scoreline that falls out of the Elo gap, and where the probability lands.

- **Canada v Morocco** (Elo 1877 v 2025): expected **0.90–1.59**, most likely 1-1 — Morocco favoured at 53%.
- **Paraguay v France** (Elo 1898 v 2187): expected **0.68–2.10**, most likely 0-2 — France favoured at 70%.

## Notes

- Bankroll £106.66. Settled 76 bets (42 won), net £+2.34 on a £100 start.
- Model adjustments active this run: totals-calib(lam x1.09).
- Same numbers as charts: `dashboard.html` (`python3 scripts/worldcup/report.py`).
