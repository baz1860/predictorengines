# World Cup 2026 — Best Bets

_Generated 2026-06-27 08:56 · blend model · fitted Elo + Poisson_

The model rates every nation on its full international history, turns each fixture into an expected scoreline, and then weighs its own probabilities against every price it can find. It makes **Argentina** the tournament favourite at 23% to lift the trophy. This week it backs **6 bets** (total stake £17.49) — each explained below, with the model's number, the price, and exactly where the edge comes from. Stakes are fractional-Kelly on a £112 bankroll.

## How the model thinks

Every number below comes from one pipeline, so it's worth knowing what drives it:

1. **Strength (Elo).** Each nation carries an Elo rating built from every international it has played, with bigger swings for World Cups and big wins than for friendlies. The gap between two ratings is the model's core read on who is better, and by how much.
2. **Expected goals (Poisson).** That Elo gap is fed through a goal model fitted on every match since 2010. It turns the gap into an *expected scoreline* — e.g. 1.8 goals for the stronger side, 0.8 for the weaker — with a home-field bump only for the host nations (USA, Mexico, Canada) playing at home.
3. **The full grid (Dixon-Coles).** From those two goal expectations the model builds the probability of every scoreline, nudged to fit how often low-scoring draws really happen. Summing the grid gives win/draw/loss, both-teams-to-score and over/under numbers.
4. **Edge.** A bet is only listed when the model's probability is enough above the bookmaker's implied probability to clear the threshold. The title %s come from running the whole tournament tens of thousands of times, so they fold in group draws and bracket luck, not just raw strength.

## Match bets

Each bet pits the model's probability against the bookmaker's price on an upcoming match; it only fires when its own number is the bigger one. **6 bets cleared the threshold** (total stake £17.49), strongest edge first.

### Over 2.5 goals — Croatia v Ghana
**2.33** · model 50.4% vs market 40.7% · **+9.8pp edge** · stake **£4.01**

The model rates Croatia at Elo 1953 against Ghana's 1704, a 249-point edge to Croatia. The two attacks project to **1.94 + 0.73 = 2.67** expected goals, just above the 2.5 line. That makes Over a **50%** shot, where the price only allows 41% — the market is pricing a tighter game than the model sees.

### Croatia win — Croatia v Ghana
**1.91** · model 59.8% vs market 50.3% · **+9.5pp edge** · stake **£4.68**

The model rates Croatia at Elo 1953 against Ghana's 1704, a 249-point edge to Croatia. Run through the goal model that comes out as an expected **1.94–0.73** in Croatia's favour, and once every scoreline is added up Croatia win it **60%** of the time. The 50% price baked into the odds is too generous for a side the model likes this much over Ghana.

### Under 2.5 goals — DR Congo v Uzbekistan
**1.77** · model 62.1% vs market 54.1% · **+8.0pp edge** · stake **£3.87**

The model rates DR Congo (Elo 1765) and Uzbekistan (Elo 1774) as near-equals — a gap of just 9 points. Between them the sides project to only **1.18 + 1.21 = 2.39** expected goals, just below the 2.5 line, so the model leans Under at **62%** against the 54% the price implies — it expects a cagier match than the bookmaker.

### Over 2.5 goals — Jordan v Argentina
**1.52** · model 69.1% vs market 63.1% · **+6.0pp edge** · stake **£2.90**

The model rates Argentina at Elo 2205 against Jordan's 1717, a 488-point edge to Argentina. The two attacks project to **0.46 + 3.10 = 3.56** expected goals, comfortably above the 2.5 line. That makes Over a **69%** shot, where the price only allows 63% — the market is pricing a tighter game than the model sees.

### Under 2.5 goals — Colombia v Portugal
**2.02** · model 50.2% vs market 47.2% · **+3.0pp edge** · stake **£0.76**

The model rates Colombia at Elo 2077 against Portugal's 2043, a 34-point edge to Colombia. Between them the sides project to only **1.28 + 1.12 = 2.40** expected goals, just below the 2.5 line, so the model leans Under at **50%** against the 47% the price implies — it expects a cagier match than the bookmaker.

### Argentina win — Jordan v Argentina
**1.15** · model 84.4% vs market 83.5% · **+0.9pp edge** · stake **£1.27**

The model rates Argentina at Elo 2205 against Jordan's 1717, a 488-point edge to Argentina. Run through the goal model that comes out as an expected **3.10–0.46** in Argentina's favour, and once every scoreline is added up Argentina win it **84%** of the time. The 84% price baked into the odds is too generous for a side the model likes this much over Jordan.

## Title outlook

**Argentina** head the field: highest Elo in the draw (2205) and champions in **23%** of simulated tournaments. Spain are the closest challenger at 14%, with France, Brazil heading the chasing pack.

These aren't bets — they're the model's read on the title race, straight from the tournament simulation. Each side's chance to lift the trophy already folds in its group draw and likely knockout path, which is why raw Elo order and these numbers don't match exactly.

| Team | Grp | Champion | Reach final |
|---|---|--:|--:|
| Argentina | J | 22.5% | 34% |
| Spain | H | 14.0% | 24% |
| France | I | 9.4% | 18% |
| Brazil | C | 6.6% | 12% |
| Colombia | K | 6.0% | 12% |
| England | L | 5.7% | 11% |
| Portugal | K | 5.1% | 11% |
| Mexico | A | 4.1% | 8% |
| Netherlands | F | 3.5% | 8% |
| Morocco | C | 3.4% | 8% |
| Germany | E | 2.8% | 7% |
| United States | D | 2.4% | 7% |

## Fixtures forecast

Not bets — the model's read on the next matchday (2026-06-27). For each game: the expected scoreline that falls out of the Elo gap, and where the probability lands.

- **Algeria v Austria** (Elo 1877 v 1899): expected **1.14–1.25**, most likely 1-1 — Austria favoured at 37%.
- **Jordan v Argentina** (Elo 1717 v 2205): expected **0.46–3.10**, most likely 0-3 — Argentina favoured at 88%.
- **Colombia v Portugal** (Elo 2077 v 2043): expected **1.28–1.12**, most likely 1-1 — Colombia favoured at 39%.
- **DR Congo v Uzbekistan** (Elo 1765 v 1774): expected **1.18–1.21**, most likely 1-1 — Uzbekistan favoured at 36%.
- **Panama v England** (Elo 1775 v 2087): expected **0.65–2.20**, most likely 0-2 — England favoured at 72%.
- **Croatia v Ghana** (Elo 1953 v 1704): expected **1.94–0.73**, most likely 2-0 — Croatia favoured at 65%.

## Notes

- Bankroll £111.67. Settled 52 bets (27 won), net £+14.69 on a £100 start.
- Model adjustments active this run: totals-calib(lam x1.09).
- Same numbers as charts: `dashboard.html` (`python3 scripts/worldcup/report.py`).
