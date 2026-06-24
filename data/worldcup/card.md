# World Cup 2026 — Best Bets

_Generated 2026-06-24 19:03 · blend model · fitted Elo+Poisson_

The model simulated the World Cup field and weighed every price it could find against its own probabilities. It makes **Argentina** the favourite at 23% to lift the trophy. This week it backs **7 bets** (total stake £19.46) — each explained below, with the model's number, the price, and why there's an edge. Stakes are fractional-Kelly on a £113 bankroll.

## Match bets

Each bet backs the model against the bookmaker's price on an upcoming match. It bets only when its own probability beats the price. **7 bets cleared the threshold** (total stake £19.46), strongest edge first:

- **Japan win** (Japan v Sweden) — 1.90. The model makes this 61.3%, against 50.1% implied by the price — a +11.3pp edge. Stake **£5.21**.
- **Under 2.5 goals** (Bosnia and Herzegovina v Qatar) — 2.42. The model makes this 50.3%, against 39.0% implied by the price — a +11.3pp edge. Stake **£4.38**.
- **Under 2.5 goals** (Ecuador v Germany) — 2.05. The model makes this 55.1%, against 46.1% implied by the price — a +9.1pp edge. Stake **£3.52**.
- **Mexico win** (Czech Republic v Mexico) — 1.92. The model makes this 56.3%, against 50.1% implied by the price — a +6.2pp edge. Stake **£2.47**.
- **Over 2.5 goals** (Czech Republic v Mexico) — 2.05. The model makes this 51.2%, against 46.3% implied by the price — a +4.9pp edge. Stake **£1.36**.
- **Over 2.5 goals** (Japan v Sweden) — 1.77. The model makes this 58.9%, against 54.1% implied by the price — a +4.8pp edge. Stake **£1.45**.

Also backed, at smaller edges (1):

| Bet | Match | Odds | Model | Edge | Stake |
|---|---|--:|--:|--:|--:|
| Over 2.5 goals | Scotland v Brazil | 1.83 | 56.2% | +4.1pp | £1.07 |

## Title outlook

Not bets — just the model's own read on the title race, from the tournament simulation: each side's chance to lift the trophy and to reach the final.

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

Not bets — the model's read on the next matchday (2026-06-24): each side's win/draw/loss chance and its single most likely scoreline.

| Match | Home | Draw | Away | BTTS | Likely |
|---|--:|--:|--:|--:|---|
| Mexico v Czech Republic | 70% | 20% | 10% | 44% | 2-0 |
| South Africa v South Korea | 14% | 24% | 61% | 46% | 0-1 |
| Canada v Switzerland | 35% | 30% | 35% | 50% | 1-1 |
| Bosnia and Herzegovina v Qatar | 45% | 29% | 26% | 49% | 1-1 |
| Scotland v Brazil | 15% | 24% | 61% | 47% | 0-1 |
| Morocco v Haiti | 76% | 17% | 7% | 42% | 2-0 |

## Notes

- Bankroll £113.44. Settled 47 bets (25 won), net £+16.46 on a £100 start.
- Model adjustments active: totals-calib(lam x1.09).
- Dashboard with charts: `dashboard.html` (`python3 scripts/worldcup/report.py`).
