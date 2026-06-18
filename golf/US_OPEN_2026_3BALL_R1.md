# 2026 U.S. Open — Round 1 3-Ball Card

Prices: **Sky Bet**, Round 1 3-balls (settles on lowest Round-1 score only).
Model: fitted single-round model — `score ~ Normal(−rating, σ)`, Shinnecock + major σ,
200k sims. Staking: quarter-Kelly on £100 bankroll. 36 of the 52 groups priced
(the ones you pasted).

**Why a separate pricer:** the main engine settles 3-balls on the 72-hole total, but
this Sky Bet market is decided by Round 1 alone. `price_threeballs_r1.py` models a single
round instead, so the probabilities actually match how the bet settles. Sanity check: in
Scheffler's group the model gives him 59.5% vs the de-vigged market's 57.4% — tight
agreement at the top, which is what you want from the anchor.

## Recommended (11 bets · £15.77 total)

| EV% | Player | Group | Odds | Model | Market | Stake | Rounds |
|----:|--------|-------|-----:|------:|-------:|------:|-------:|
| +22.9 | **Jackson Suber** | Schaper / Suber / Fang | 2.50 | 49.2% | 36.1% | £3.82 | 131 |
| +12.0 | Akshay Bhatia | M. Lee / Ortiz / Bhatia | 2.88 | 38.9% | 31.9% | £1.59 | 323 |
| +11.9 | Ryan Fox | Fox / Conners / Hisatsune | 2.75 | 40.7% | 34.3% | £1.70 | 247 |
| +11.6 | Viktor Hovland | M. Fitzpatrick / DeChambeau / Hovland | 3.25 | 34.3% | 29.0% | £1.29 | 322 |
| +10.2 | Davis Thompson | Thompson / Puig / Stout | 2.75 | 40.1% | 34.0% | £1.46 | 329 |
| +10.1 | Justin Thomas | Schauffele / Thomas / Matsuyama | 3.10 | 35.5% | 29.5% | £1.20 | 343 |
| +9.0 | Padraig Harrington | Harrington / Smith / Russell | 3.75 | 29.1% | 25.3% | £0.82 | 82 |
| +8.5 | Brandon Wu | B. Wu / Stanger / Yuan | 3.10 | 35.0% | 29.6% | £1.02 | 276 |
| +7.7 | Ben Silverman | Silverman / Grillo / Dumont | 3.10 | 34.7% | 29.8% | £0.91 | 201 |
| +7.3 | Nico Echavarria | MacIntyre / Højgaard / Echavarria | 3.75 | 28.6% | 25.0% | £0.66 | 313 |
| +7.2 | Wyndham Clark | Johnson / Clark / Woodland | 2.38 | 45.0% | 39.6% | £1.30 | 393 |

The headline play is **Jackson Suber** (+22.9%): the market made Jayden Schaper favourite in
that group, but the model has Suber — a solid PGA pro (skill +0.28, 131 rounds) — clearly
ahead of a DP World Tour player and an amateur. The rest are mid-single-digit-to-low-double-digit
overlays on well-sampled players. Note Harrington's edge is "market made a 55-year-old the
longest price in a flat group" rather than positive skill — the model has him ~average and
doesn't model age, so treat that one as the softest of the eleven.

## Excluded — positive edge, but don't bet

| EV% | Player | Odds | Rounds | Why excluded |
|----:|--------|-----:|-------:|--------------|
| +11.8 | Ethan Fang | 4.33 | 6 | Amateur, 6 rounds — rating is basically a guess |
| +6.1 | Alex Fitzpatrick | 2.38 | 38 | Thin sample inflates his skill (same artefact that wrongly ranked him 7th in the outright sim) |

The pricer auto-excludes any player with fewer than 60 fitted rounds (`--min-rounds`),
because a thin history makes the skill estimate — and therefore the "edge" — unreliable.

## How to refresh / extend

```bash
cd golf
# paste a fresh board into data/threeballs_r1_raw.txt, then:
python3 price_threeballs_r1.py --min-edge 4 --kelly 0.25 --bankroll 100
```

Same approach works for **tournament matchups** and **top-5/10/20 / make-cut** — paste those
Sky Bet lines and I'll wire equivalent pricers (tournament matchups use the existing 72-hole
sim; place markets compare model `predictions.csv` probabilities to your pasted prices).

## Files
- `golf/price_threeballs_r1.py` — the single-round 3-ball pricer.
- `golf/data/threeballs_r1_edges.csv` — full 108-player card with EV, stake, sample size.
- `golf/data/threeballs_r1_raw.txt` — the Sky Bet prices you pasted (re-paste to refresh).
