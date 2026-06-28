# World Cup 2026 — Best Bets

_Generated 2026-06-28 08:51 · blend model · fitted Elo + Poisson_

The model rates every nation on its full international history, turns each fixture into an expected scoreline, and then weighs its own probabilities against every price it can find. It makes **Argentina** the tournament favourite at 24% to lift the trophy. This week it backs **2 bets** (total stake £1.60) — each explained below, with the model's number, the price, and exactly where the edge comes from. Stakes are fractional-Kelly on a £109 bankroll.

## How the model thinks

Every number below comes from one pipeline, so it's worth knowing what drives it:

1. **Strength (Elo).** Each nation carries an Elo rating built from every international it has played, with bigger swings for World Cups and big wins than for friendlies. The gap between two ratings is the model's core read on who is better, and by how much.
2. **Expected goals (Poisson).** That Elo gap is fed through a goal model fitted on every match since 2010. It turns the gap into an *expected scoreline* — e.g. 1.8 goals for the stronger side, 0.8 for the weaker — with a home-field bump only for the host nations (USA, Mexico, Canada) playing at home.
3. **The full grid (Dixon-Coles).** From those two goal expectations the model builds the probability of every scoreline, nudged to fit how often low-scoring draws really happen. Summing the grid gives win/draw/loss, both-teams-to-score and over/under numbers.
4. **Edge.** A bet is only listed when the model's probability is enough above the bookmaker's implied probability to clear the threshold. The title %s come from running the whole tournament tens of thousands of times, so they fold in group draws and bracket luck, not just raw strength.

## Match bets

Each bet pits the model's probability against the bookmaker's price on an upcoming match; it only fires when its own number is the bigger one. **2 bets cleared the threshold** (total stake £1.60), strongest edge first.

### Over 2.5 goals — Brazil v Japan
**1.98** · model 50.7% vs market 47.9% · **+2.8pp edge** · stake **£0.76**

The model rates Brazil at Elo 2082 against Japan's 1993, a 89-point edge to Brazil. The two attacks project to **1.42 + 1.01 = 2.43** expected goals, just below the 2.5 line. That makes Over a **51%** shot, where the price only allows 48% — the market is pricing a tighter game than the model sees.

### Under 2.5 goals — South Africa v Canada
**1.75** · model 55.7% vs market 54.5% · **+1.2pp edge** · stake **£0.84**

The model rates Canada at Elo 1859 against South Africa's 1711, a 148-point edge to Canada. Between them the sides project to only **0.90 + 1.59 = 2.49** expected goals, just below the 2.5 line, so the model leans Under at **56%** against the 55% the price implies — it expects a cagier match than the bookmaker.

## Title outlook

**Argentina** head the field: highest Elo in the draw (2211) and champions in **24%** of simulated tournaments. Spain are the closest challenger at 15%, with France, England heading the chasing pack.

These aren't bets — they're the model's read on the title race, straight from the tournament simulation. Each side's chance to lift the trophy already folds in its group draw and likely knockout path, which is why raw Elo order and these numbers don't match exactly.

| Team | Grp | Champion | Reach final |
|---|---|--:|--:|
| Argentina | J | 24.2% | 37% |
| Spain | H | 14.6% | 26% |
| France | I | 11.4% | 22% |
| England | L | 6.9% | 13% |
| Colombia | K | 6.5% | 13% |
| Brazil | C | 6.3% | 12% |
| Portugal | K | 4.0% | 9% |
| Mexico | A | 3.3% | 7% |
| Morocco | C | 3.0% | 7% |
| Netherlands | F | 3.0% | 7% |
| Germany | E | 2.2% | 6% |
| Japan | F | 2.1% | 4% |

## Fixtures forecast

Not bets — the model's read on the next matchday (2026-06-28). For each game: the expected scoreline that falls out of the Elo gap, and where the probability lands.

- **South Africa v Canada** (Elo 1711 v 1859): expected **0.90–1.59**, most likely 1-1 — Canada favoured at 53%.

## Notes

- Bankroll £108.63. Settled 61 bets (34 won), net £+11.65 on a £100 start.
- Model adjustments active this run: totals-calib(lam x1.09).
- Same numbers as charts: `dashboard.html` (`python3 scripts/worldcup/report.py`).
