# 2026 U.S. Open — Model Card

**Shinnecock Hills Golf Club, Southampton NY · 18–21 June 2026 (major).**
Field: 156 official (158 on the board incl. 2 alternates). Odds as of 17 Jun 2026,
best price across BetMGM / DraftKings / Betfair Exchange / BetRivers (decimal).
Engine: fitted strokes-gained + variance model, 50,000-sim Monte Carlo, major σ bump,
calibrated and market-blended.

---

## What I fixed first

The golf module was built but its last outputs were broken in two ways, both now resolved:

- **Rating floor on accented names.** The fitted model stores `Ludvig Åberg`,
  `Nicolai Højgaard` etc. with their real spelling, but the odds/field feed sends ASCII
  (`Aberg`, `Hojgaard`), so the name lookup missed and dumped those players onto the
  default-skill floor (−1.9). Åberg — a genuine top-5 player — was being modelled as one
  of the weakest in the field. I made name resolution accent-insensitive (incl. `ø`/`æ`,
  which don't decompose normally) plus a small alias map for nickname cases
  (`Matthew → Matt Fitzpatrick`, `Joohyung → Tom Kim`, etc.). Åberg now rates +1.40 / 2.75%.
- **`cut_pct = 100%` for everyone.** That was correct for the old 30-player stub field (a
  65-man cut can't bind on 30 players). With the real 156-player field loaded, the cut binds
  and make-cut now ranges ~57–88% as it should.

Both fixes are in `model.py` (zero name collisions, all modules import clean).

---

## Model contenders (top 20)

| # | Player | Win% | Top-5 | Top-10 | Top-20 | Cut% | Best odds |
|--:|--------|-----:|------:|-------:|-------:|-----:|----------:|
| 1 | Scottie Scheffler | 10.6 | 34.3 | 51.0 | 69.7 | 87.9 | 8.0 |
| 2 | Rory McIlroy | 5.1 | 18.8 | 31.0 | 48.2 | 76.0 | 15.0 |
| 3 | Wyndham Clark | 3.6 | 14.2 | 24.3 | 39.3 | 70.0 | 65.0 |
| 4 | Jon Rahm | 3.3 | 13.1 | 22.5 | 36.9 | 67.8 | 20.0 |
| 5 | Ludvig Åberg | 2.8 | 11.4 | 20.2 | 34.3 | 65.7 | 32.0 |
| 6 | Sam Burns | 2.5 | 11.0 | 19.7 | 33.6 | 65.8 | 44.0 |
| 7 | Alex Fitzpatrick* | 2.3 | 9.9 | 17.5 | 30.4 | 62.6 | 150.0 |
| 8 | Cameron Young | 2.2 | 9.8 | 17.8 | 30.8 | 62.9 | 29.0 |
| 9 | Justin Thomas | 2.2 | 9.0 | 16.4 | 28.8 | 61.5 | 75.0 |
| 10 | Tommy Fleetwood | 2.2 | 11.3 | 21.8 | 39.4 | 72.4 | 23.0 |
| 11 | Kristoffer Reitan* | 2.1 | 8.4 | 14.9 | 25.6 | 57.2 | 140.0 |
| 12 | Xander Schauffele | 1.9 | 9.3 | 18.1 | 32.9 | 66.7 | 22.0 |
| 13 | Patrick Reed | 1.8 | 8.6 | 16.2 | 29.5 | 62.9 | 60.0 |
| 14 | Patrick Cantlay | 1.8 | 9.8 | 19.1 | 35.3 | 69.3 | 70.0 |
| 15 | Maverick McNealy | 1.7 | 8.2 | 15.8 | 28.7 | 62.4 | 110.0 |

\* *Small-sample caveat — see below.*

---

## Where the model disagrees with the market (win market)

Ratio = model win% ÷ de-vigged market win%. Above 1 = model likes them more than the
price; below 1 = model is colder.

**Model overlays (model warmer than market)**

| Player | Odds | Model W% | Market W% | Ratio |
|--------|-----:|---------:|----------:|------:|
| Sam Burns | 44 | 2.52 | 1.75 | 1.44 |
| Patrick Reed | 60 | 1.84 | 1.28 | 1.44 |
| Ludvig Åberg | 32 | 2.75 | 2.40 | 1.14 |
| Scottie Scheffler | 8 | 10.62 | 9.62 | 1.10 |

**Model fades (model colder than market)**

| Player | Odds | Model W% | Market W% | Ratio |
|--------|-----:|---------:|----------:|------:|
| Matthew Fitzpatrick | 24 | 1.54 | 3.21 | 0.48 |
| Xander Schauffele | 22 | 1.87 | 3.50 | 0.53 |
| Tommy Fleetwood | 23 | 2.18 | 3.34 | 0.65 |
| Cameron Young | 29 | 2.19 | 2.65 | 0.83 |

The standout reads are the fades: the market prices Schauffele, Fleetwood and Fitzpatrick
as a clear second tier (22–24), while the model's skill+form blend has them a notch below
that. Treat these as the model's opinion, not gospel — the model has **no Shinnecock
course-fit signal** (the venue hasn't appeared in the 2022–2026 round history it learned
from), so it can't reward links/major pedigree the market may be paying for.

---

## Betting verdict

**No outright (win) bet clears the staking bar — and that's the right answer.** After
de-vigging the win board and blending the model toward this sharp major market, the only
positive-EV outrights are sub-1% longshots (Aaron Rai, McNealy, Kitayama) where quarter-Kelly
on a £100 bankroll stakes literal pennies. The win market here is efficient and the model
broadly agrees with it.

The engine's edges normally come from **matchups, 3-balls and place markets**, where the
blend leans toward the model rather than the market. The Odds API only carries the *winner*
market for this event, and DataGolf (which supplies matchup/place lines + richer SG history)
is unreachable from my sandbox. To get that side of the card, run it on your Mac where your
DataGolf key works:

```bash
cd golf
bash update.sh --course "Shinnecock Hills" --major
python3 edge.py --course "Shinnecock Hills" --major --min-edge 1.0
```

That will pull matchup/3-ball/place boards and price them through the same calibrated,
portfolio-staked pipeline.

---

## Caveats

- **Small-sample longshots are noisy.** Alex Fitzpatrick (150), Kristoffer Reitan (140) and
  Ryan Gerard (150) rank higher in the model than their prices imply — that's thin/recent
  data inflating their skill estimate, not real value. Trust the relative-value signal most
  among well-sampled, established players.
- **No course-fit term for Shinnecock** (not in the learned history), so every player's
  course adjustment is neutral (0). The major σ bump *is* applied.
- **158 vs 156:** the board lists two alternates; harmless, they sit at the bottom.

## Files
- `golf/data/predictions.csv` — full-field model probabilities (win/top-5/10/20/cut).
- `golf/data/edge_report.csv` — all 79 priced win-market lines with EV and Kelly stake.
- `golf/data/field.csv`, `golf/data/odds.csv` — the live Shinnecock field + win odds I pulled.
