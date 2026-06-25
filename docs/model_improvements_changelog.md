# Model improvements — changelog & results

Implementation pass following the calibration/CLV diagnostic and the 1X2 feature draft.

## 1. Fixed CLV reporting (`core/clv.py`)
`--report` was crashing on a tz-aware vs tz-naive datetime comparison, so closing-line
value — your most important betting KPI — wasn't being computed at all. Snapshots are
now parsed `utc=True` and the cutoff is built tz-aware.

**Result:** report runs; rolling mean CLV **−0.72%** over 44 settled bets — independently
confirming the diagnostic's −0.75%. The betting layer is not beating the close.

## 2. Recency weighting in the goal model (`engines/worldcup/features_1x2.py`)
Your point was correct: the 1X2 regression scaffold trained on *all* internationals back
to 1872 with equal weight — **41% of the training matches predate 1990, 58% predate 2000.**
Added exponential time-decay (`decay_weights`, half-life configurable, 8y default) threaded
through a weighted Poisson IRLS (`fit_poisson(... w=...)`).

**Result (held-out 1X2 log-loss, competitive 2018+, n=5,884):**

| Scheme | log-loss |
|--------|----------|
| equal weight, all history | 0.8501 |
| half-life 20y | 0.8497 |
| half-life 8y | 0.8499 |
| cutoff ≥2000 only | 0.8498 |
| cutoff ≥2010 only | 0.8503 |

So old data is **nearly worthless but also nearly harmless on the current model** — the
Elo-gap→goals slope (~0.78) is era-stable, so ancient matches barely distort it; they only
inflate the baseline scoring level (intercept drifts 0.30→0.20). Decay is now the default
because it becomes important once **era-sensitive features** (form, squad value, styles) are
added — those genuinely change over time and must not be equal-weighted against 1990s data.
See `engines/worldcup/exp_recency.py` to reproduce.

## 3. Refit the market blend (`engines/worldcup/fit_market_blend.py`)
Refit the logit-space 1X2 blend on **128** WC matches (2018+2022) with leave-one-tournament-out
CV, replacing the old 64-game fit.

**Result:** optimal weight **w → 1.0** — i.e. the published World Cup market is sharper than
the model and the best blend is *pure market* (model-only log-loss 1.019 vs market-only 0.971).
This is a **no-edge** result, not a deployable weight (w=1.0 = copy the market and pay the vig).
`data/market_blend.json` is left **inactive** (prior `w=0.163` preserved; original backed up to
`.bak.preblend`) with the interpretation recorded.

**Operational read:** don't bet WC 1X2 on model edge — the WC market is too efficient. The
model's edge (it beats base-rate by ~0.20 log-loss over 7,138 competitive matches) lives in
**softer markets**: less-watched internationals and specific totals where the line moves slower.

## 4. Soft-market probe — club corners model (`club_soccer/corners_model.py`)
Followed the "hunt softer markets" recommendation into corners. Built a corners model in
the club_soccer style (recency-weighted attack/defence, per-league baseline, home adj,
full O/U pricing) on `fixtures.csv` (16,775 matches, top leagues 2022-26), then wired the
shots lever (Poisson GLM with team shot-strength) and validated head-to-head.

**Result (walk-forward, test n=3,403):**

| model | MAE (total corners) | O/U 9.5 log-loss | calibration (ECE) |
|-------|---------------------|------------------|--------------------|
| league baseline | **2.71** | **0.693** | 0.007 |
| corners (team strength) | 2.78 | 0.707 | 0.064 |
| corners + shots GLM | 2.77 | 0.706 | 0.066 |

The shots GLM fits a shot coefficient of ~0.08 (corner ~0.99) — shots are collinear with
corner rate and add nothing. **Neither corners nor shots beats simply quoting the league
average**, and both are overconfident. Corners are governed by in-game state and noise, not
stable pre-match team traits. A pre-match strength model is the wrong tool for this market;
the only paths left are in-play models or market-specific inefficiencies (line shopping).
Honest conclusion: **do not bet corners with this approach.**

## 5. International TOTALS probe (`engines/worldcup/totals_probe.py`)
Tested the "softer international totals" idea directly, using the model's own logged edges
(`edge_snapshots.csv`: p_model vs de-vigged p_book) joined to results (32 settled matches).

**Result — the book beats the model on totals too:**
- O/U 2.5 log-loss: **model 0.692 vs book 0.657** (model even worse than base-rate 0.676).
- The model leans Under (mean P(over) 0.47 vs book 0.51 vs actual 0.59 on this hot sample).
- When model and book disagree on side, **book right 71%, model 29%**.
- Following the model's value side lost **-10% ROI**, and *worse the bigger the claimed edge*
  (edge >=5% -> -66%). That inverse edge->ROI relationship is the signature of the "edges" being
  the Under bias, not signal.

Small live-tournament sample (re-run as matches settle), but the direction is unambiguous and
contradicts the earlier guess that totals would be soft. The book is sharper there too.

**Older World Cups (book-free check, `--wc-calibration`).** No totals odds exist for 2018/2022
in the repo, so a model-vs-book test isn't possible there — but the model-vs-reality test is.
It corrects one thing and confirms another:

| Tournament | model P(over) | actual | avg goals | vs base-rate |
|---|---|---|---|---|
| WC2018 | 0.469 | 0.484 | 2.64 | loses |
| WC2022 | 0.468 | 0.469 | 2.69 | loses |
| WC2026 so far | 0.500 | 0.521 | 2.94 | loses |

- **Correction:** the model is NOT Under-biased on WC totals — predicted ≈ actual every time.
  The "Under lean" in the 32-match probe was just 2026 running hot on goals (2.94/game).
- **Confirmation:** at *every* World Cup the model's totals **lose to the base rate** per match
  (~0.700 vs ~0.692 log-loss). At WC level the teams are bunched, so expected goals barely vary
  (~0.47 over for nearly every game) — no discrimination. If it can't beat the base rate, it
  can't beat a sharp book. Same conclusion you predicted, different mechanism (no edge, not bias).

## Bottom line
The goals model is well-calibrated and genuinely predictive at the population level — but in
every market we could actually price-test, **it does not beat the closing line**: WC 1X2 (blend
-> pure market), corners (lose to the league average; pure noise), and now international totals
(book sharper, the Under lean loses money). The honest conclusion is that the available markets
are efficient enough that this model's edge is forecasting value, not betting value. Productive
directions, in order: (a) keep CLV as the only scorecard (now that it runs); (b) if betting
continues, restrict to the smallest, least-watched competitions where lines are genuinely lazy,
and *prove* edge by CLV before staking; (c) otherwise treat the model as a calibrated forecasting
tool rather than a market-beater. Bigger-sample re-runs of the totals probe and corners
validation are the cheapest next checks.

## Files
- `core/clv.py` — tz fix
- `engines/worldcup/features_1x2.py` — weighted IRLS + decay + gate (extended)
- `engines/worldcup/exp_recency.py` — recency experiment (new)
- `engines/worldcup/fit_market_blend.py` — market-blend refit (new)
- `engines/worldcup/data/market_blend.json` → `data/market_blend.json` — refit recorded, left inactive (`.bak.preblend` = original)
- `club_soccer/corners_model.py` — corners model + shots GLM + validation (new)
- `engines/worldcup/totals_probe.py` — international totals model-vs-book probe (new)
