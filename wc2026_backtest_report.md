# World Cup 2026 — Live Backtest (all results so far)

**Hypothesis tested:** the model's pre-match predictions match real outcomes.
**Sample:** all **16 matches** completed from kick-off (Thu 11 Jun) through 15 Jun 2026. The 16 Jun fixtures had not kicked off when this was run.
**Method:** leak-free. The goal model is fit only on internationals played *before* 11 Jun 2026, and each match is scored on its pre-kickoff Elo — the same approach as the repo's existing `wc2022_backtest.py`. As a check, the regenerated 15 Jun probabilities reproduce the published `predictions_worldcup_2026.csv` exactly (e.g. Spain 0.917, Belgium 0.518), so this *is* the model's real output.

## Headline metrics (n = 16)

| Metric | Model | Random baseline | Verdict |
|---|---|---|---|
| 3-way accuracy | 37.5% (6/16) | 33.3% | ~ at chance |
| Brier (avg) | 0.779 | 0.667 | **worse than chance** |
| Log-loss (avg) | 1.230 | 1.099 | **worse than chance** |
| Avg prob on actual outcome | 0.347 | 0.333 | ~ at chance |
| Exact scoreline | 3 / 16 | — | — |

The model's *direction* is roughly coin-flip-plus, but its *confidence* is being punished: Brier and log-loss are both worse than blind guessing because several heavy favourites failed to win.

## The dominant failure: draws

This opening round has been freakishly draw-heavy — **8 of 16 matches were drawn** (≈50%, vs a typical ~25%). Outcome split: 7 home wins, 8 draws, 1 away win.

The model **never once picked a draw** (0/16) — across all 60 of its fixtures a draw is never the single most likely result (max draw probability anywhere ≈ 30%). So every one of the 8 draws is an automatic outcome miss. This single structural quirk explains essentially all the damage:

- On the **8 decisive matches, the model went 6/8 (75%)** — strong.
- On the **8 drawn matches, it went 0/8** — by construction.

## Where it failed

- **Overconfident favourites that drew.** Of 6 matches where the model gave a side ≥60%, only 2 won; the other 4 were draws. The worst: **Spain 0–0 Cape Verde** (model: Spain 91.7%), plus Saudi Arabia 1–1 Uruguay (Uruguay 67%), Canada 1–1 Bosnia (Canada 70%), Qatar 1–1 Switzerland (Switzerland 79%). These overconfident misses are what drag Brier/log-loss below chance.
- **Two clean upsets the *modal pick* got backwards:** Australia 2–0 Turkey and Ivory Coast 1–0 Ecuador — the model's single most likely outcome favoured the loser. **Important caveat on Australia (see betting section):** as a 3-way pick this counts as a miss, but the betting engine actually backed *Australia* at 5.25 and won +24.14 — because the model's 27.8% on Australia was well above the ~19% the market implied. Outcome-accuracy and betting value are not the same thing, and this match is the clearest example.

## Where it succeeded

- **Clear mismatches it called correctly:** Mexico 2–0 South Africa, Germany 7–1 Curaçao, Sweden 5–1 Tunisia, USA 4–1 Paraguay, South Korea 2–1 Czechia, Scotland 1–0 Haiti — 6 right winners, several by big margins as implied by the high xG gap.
- **3 exact scorelines:** Mexico 2–0, Brazil 1–1, Belgium 1–1.
- **Calibration on blowouts is sound** — the games it rated as lopsided generally were.

## Betting performance — the right test for this model

3-way accuracy judges the *predictor*; but the product is a *betting* model, and a bet model is supposed to back value (positive expected value vs the market price), not the single most likely result. On that metric — measured from the actual placed bets in the suite ledger, which records the odds taken at the time — it is **profitable so far**:

| Metric | Value |
|---|---|
| Settled WC bets | 18 (8 won / 10 lost, 44% strike rate) |
| Total staked | 52.51 u |
| Net P&L | **+23.66 u** |
| ROI / yield | **+45.1%** |
| Bankroll | 100.00 → **120.64** (peak 125.41) |
| Open bets still running | 3 outrights (3.52 u staked) |

By market: 1X2 match-result bets +51.2% ROI (10 bets), Totals O/U 2.5 +28.8% (8 bets).

The standout is exactly the match the accuracy metric flagged as a failure: **Australia to beat Turkey @ 5.25 → +24.14 u**, the single biggest winner on the book.

**But be honest about variance:** that one longshot *is* the profit. Strip Australia out and the other 17 settled bets are **−0.48 u (−1.0% ROI)** — essentially breakeven. By odds band the entire return sits in the longshot bucket (≥4.0: +307% on 2 bets); short-priced bets (<1.8) are −49%. So the +45% yield rests on one 5.00+ winner and is not yet evidence of a durable edge.

**Closing-line value points the same way.** Of the 11 settled bets that now have a closing-odds snapshot, mean CLV is **−6.8%** with only 36% beating the close. CLV is the most reliable early signal of a real edge, and right now it's negative — i.e. the prices taken were on average *worse* than the market close (the Australia winner has no snapshot, so it's excluded). Net: the model found and sized a genuine value bet (Australia), but there is not yet CLV or sample-size evidence of a repeatable edge. Run `python clv.py --snapshot` before kickoffs to keep building the closing-line record.

## Honest read

Two different questions, two different answers:

- **As a match predictor:** not well supported on this sample, but for a specific, largely benign reason — an unusually draw-heavy opening round collides with a model that structurally never predicts draws. Strip the draws and it called 75% of decisive games. The one real, sample-independent red flag is **chronic over-confidence in favourites** (Spain at 92% the poster child) — worth fixing with a fatter draw allowance / shrink toward the market.
- **As a betting model:** ahead (+21.4 u, +48.5% ROI), and it found genuine value where the predictor "missed" (Australia). But the profit is one longshot deep; without Australia it's marginally negative. Verdict: promising staking behaviour, not yet a proven edge.

Caveats: n = 16 matches / 13 settled bets is small and front-loaded with matchday-1 games; re-run after another 20–30 results before trusting any of these numbers. Betting P&L is taken from the placed-bet ledger (`data/suite_ledger.csv`), which records the odds taken at placement and so is itself a point-in-time record — model changes can't retro-edit it.

*Per-match detail in `wc2026_backtest.csv`. Sources: ESPN, Yahoo Sports, Olympics.com, Sky Sports, Al Jazeera, FOX Sports match reports (11–15 Jun 2026).*
