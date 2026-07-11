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

## Club Soccer Phase 4 — league structure & position (2026-07-02)

Following `plans/club_soccer_engine_plan.md` Phase 4. Three fitted/gated candidates,
each evaluated honestly against the incumbent on walk-forward Brier; none clears the bar,
so all three ship **inactive** (`"active": false`), exactly as designed.

### P4.4 Fitted competition strength (`model.py::fit_comp_strength`)
Mean end-of-fit Elo of each league's >=6-match teams in the last completed season (2025),
min-max rescaled to `[0.15, 1.10]`; cups at 0.95x their parent league. Written to
`data/comp_strength.json`.

**Result (full walk-forward, n=16714):** active=false (incumbent hand-set constants)
Brier **0.61256**; active=true (fitted) Brier **0.61275** — worse by +0.00018. Rejected;
stays inactive.

### P4.5 Promoted/relegated-team shrinkage prior (`model.py::tune_promo_prior`)
Grid-searched pi in {0.80, 0.85, 0.90, 0.95, 1.00} — a promoted team's attack/defence
shrinkage prior seeded from its own previous-season rate x pi instead of `global_avg`
(relegated: symmetric). Evaluated on promoted/relegated teams' first 10 league matches,
seasons 2023-2026 (n=826).

**Result:** baseline (no prior) Brier **0.6496**; every grid point was worse (pi=0.80 best
of the active runs at 0.6500). Rejected; stays inactive.

### P4.6 Season-boundary Elo regression + half-life re-tune (`model.py::tune_season_boundary`)
Grid-searched `season_regress_rho` in {0.0, 0.1, 0.2, 0.3, 0.4} x `HALF_LIFE_DAYS` in
{180, 270, 365} on Aug-Oct fixtures only (n=4466 per cell; the walk-forward months outside
Aug-Oct don't need refitting for this check, since a given month's `fit()` call only
depends on its own train cutoff — an exact optimization, not an approximation).

**Result:** the incumbent (rho=0, half_life=365) was already the best cell in the grid
(Brier 0.6206); every other combination was equal or worse. Confirms the existing
365-day half-life needs no change and Elo carrying straight across July is, on this data,
not costing anything. No promotion needed (nothing to promote — incumbent already wins).

### P4.7 Cup tier_gap (`context.py::_cup_tier_gap`)
Feature wired into the context GLM design matrix, but **fixtures.csv has zero PLAYED
domestic-cup rows** across the whole 2022-2026 history (FA Cup/EFL Cup/Scottish Cup/
DFB-Pokal/Coppa Italia/Coupe de France/Copa del Rey all show only upcoming, unplayed
fixtures) — the original historical seed never ingested cup results. `tier_gap` is
therefore a constant-zero column; `context.py::fit_context` now generically drops any
zero-variance design column (was previously only xi_load14_diff-specific) rather than
crashing on the resulting singular IRLS matrix. Nothing to fit yet — revisit once
`season.py`'s daily pipeline (P8) has accumulated real cup results.

### Files
- `club_soccer/standings.py` — point-in-time league tables (new)
- `club_soccer/motivation.py` — pos/ppg/fight/dead features (new)
- `club_soccer/competitions.py` — `teams_n`/`releg_spots`/`promo_spots`/`euro_spots` + fitted-strength file consultation (extended)
- `club_soccer/model.py` — `fit_comp_strength`, `tune_promo_prior`, `tune_season_boundary`, `_promo_relegation_priors`, season-boundary Elo regression in the Elo loop (extended)
- `club_soccer/context.py` — `ppg_diff`/`fight_diff`/`dead_diff`/`tier_gap` terms, generic zero-variance column guard (extended)
- `data/comp_strength.json`, `model_params.json["promo_prior"/"season_regress_rho"/"half_life_days"]` — diagnostic artifacts, all inactive

### P4.6b Continuous Elo time-decay (`model.py::tune_elo_decay`)

Follow-up investigation: raw Elo has no decay between matches, so a rating
earned in a past hot spell can carry forward at full strength for years,
eroding only through actual results. Prompted by a case study where Lincoln
City held the highest Elo of any team in the Championship/League One pool
despite a 44% win rate across its full 2022-2026 history.

Added `elo_decay_half_life_days` to `fit()`: before each team's match, its
Elo decays toward `BASE_ELO` by `0.5 ** (gap_days / half_life)`, where
`gap_days` is the calendar gap since that team's previous match (any
competition) — so decay accrues continuously, not just across close-season
gaps. `tune_elo_decay()` grid-searches
`{None, 1095, 730, 365, 180, 90}` days on the **full** walk-forward Brier
(all 43 months, n=16714 — decay can matter any time in the season, unlike
`season_regress_rho` which only bites at the July boundary).

**Result:** every decay half-life tested made Brier *worse* than the
undecayed incumbent, monotonically with decay strength:

| half-life | none (incumbent) | 1095d | 730d | 365d | 180d | 90d |
|---|---|---|---|---|---|---|
| Brier | 0.6126 | 0.6128 | 0.6129 | 0.6133 | 0.6139 | 0.6150 |

Rejected by the gate; stays inactive (`elo_decay_half_life_days: null`).

Re-examined the Lincoln case that motivated this: their overall 44% win
rate average hides a complete in-sample turnaround —
30%/43%/35% win rate in 2022-23/23-24/24-25, then **31W-10D-5L (67%)** in
2025-26, all league matches, the most recent ending 29 days before the
dataset's latest date. Lincoln's high Elo isn't stale — it's Elo correctly
tracking a team currently running away with League One. Decay does pull
Lincoln's rating down at every tested half-life (1750->1666 at 90d), but
even the most aggressive setting tested doesn't unseat them as the pool's
top-rated team, because the signal decay is fighting is real, not stale.
This explains the aggregate result: decay discounts genuine current form
(the common case) as readily as it discounts genuinely stale form (the rare
case Lincoln was mistaken for), a net loss on held-out accuracy. Ships as a
gated, inactive `fit()` parameter + `--tune-elo-decay` CLI diagnostic;
`predict.py`'s default call path is unaffected.

### Files (P4.6b)
- `club_soccer/model.py` — `elo_decay_half_life_days` param on `fit()`, decay applied in the Elo loop before season-boundary regression, `tune_elo_decay()`, `--tune-elo-decay` CLI flag (extended)

## Club Soccer Phase 5 — weather (2026-07-02)

`data/venues.csv` (265 teams, city-level lat/lon, 100% coverage on the 12
core tracked leagues). `weather.py` backfilled `data/weather.csv` from
Open-Meteo's archive+forecast APIs: **15,104 of 17,008** matched fixtures
got real weather (89%; the remainder mostly hit late-stage rate limiting on
a multi-year backfill and were skipped, not fabricated). Wired into
`context.py` as symmetric terms (`wind_high`, `precip`, `temp_cold`,
`temp_hot` — same value for both sides, since weather shifts totals, not
which side benefits) plus the OU2.5-Brier-specific gate criterion from plan
§12 (1X2 log-loss allowed to move ≤ 0.0005; primary metric is OU2.5 Brier).

**Result:** none of the four weather terms cleared |t| >= 2
(wind_high t=-0.76, precip t=1.65, temp_cold t=-1.30, temp_hot t=0.49, on
33,946 side-observations) — no detectable weather effect on goals in this
sample at city-level precision. All four pruned from the fit; nothing to
gate-check against OU2.5 Brier since none survived. Ships inactive.
(`context.py::fit_context` also gained a generic zero-variance-column drop,
needed independently for `tier_gap` — see Phase 4's P4.7 note — reused here
before real weather data existed, verified again with real data.)

### Files
- `club_soccer/data/venues.csv`, `club_soccer/weather.py` (new)
- `club_soccer/context.py` — weather terms + OU2.5-specific validate() gate (extended)

## Club Soccer Phase 6 — market layer (2026-07-02)

`snapshot_odds.py` (P6.1): discovered BSD's real multi-bookmaker data lives
at `/api/v2/events/{id}/odds/comparison/`, not the `event["bookmakers"]`
field the plan assumed (doesn't exist on list/detail responses) — added
`bsd_client.odds_comparison()`. Only populated close to kickoff empirically
(same-day, even for major leagues) — no market snapshot data was available
for any of our tracked competitions' upcoming fixtures at write time (all
are 1+ weeks out, mid-close-season).

`market_model.py` (P6.2): `line_history`/`do_not_bet`, wired into
`edge.py` (auto-on past 30 days of snapshot history; prints "warming up:
Nd" before that — currently 0.0d, correctly inactive).

`fit_market_blend.py` (P6.3): time-series CV blend weight (1X2 and OU2.5
separately) vs `market_history.csv`'s Bet365 pre-match odds, splits
{2024, 2025}. **Result: w=0.0 (pure market) wins every split, both
markets** — the grid search degenerates to "just use the market" because
nothing beats it, consistent with `backtest_market.py`'s finding (P1.5:
model 1X2 log-loss 1.023 vs market 0.997). Since w=0 ties rather than
strictly beats the market, `beats_both` is correctly False — rejected,
`app/market_blend.DEFAULT_BLEND_ON` untouched.

Found + fixed a real bug during testing: `snapshot_odds.append_snapshots`'s
dedupe pass used `pd.to_datetime` without `format="mixed"`, so a column
with a mix of `T`-separator (freshly generated) and space-separator
(post-CSV-round-trip `str(Timestamp)`) timestamps silently NaT'd (and
dropped) every row after the first format pandas inferred — would have
made every snapshot after the first look like a fresh event forever,
defeating the whole point of the movement-tracking dedupe window.

### Files
- `bsd_client.py` — `odds_comparison()` (new)
- `club_soccer/snapshot_odds.py`, `club_soccer/market_model.py`, `club_soccer/fit_market_blend.py` (new)
- `club_soccer/edge.py` — do-not-bet wiring (extended)
- `data/market_blend_suite.json["club_soccer"/"club_soccer_ou25"]` — both 0.0, inactive

## Club Soccer Phase 7 — real xG from Understat: BLOCKED, not implemented (2026-07-02)

The plan called for scraping Understat's embedded match JSON for top-5-league
real xG, citing "no aggressive bot protection." Verified live before writing
any scraper:

1. **The embedded-JSON scraping pattern no longer works.** `understat.com/league/EPL/{season}`
   returns an identical 17,480-byte page for every season tested (2024, 2025)
   with no `var datesData/teamsData/playersData = JSON.parse(...)` anywhere
   in the HTML — the site has been rewritten to load match/team data via a
   client-side call after page load, not embedded server-side. A handful of
   guessed REST/GraphQL endpoint paths all 404.
2. **`understat.com/robots.txt` is now `User-agent: * / Disallow: /`** — a
   blanket disallow on the entire site. The plan's own ground rule ("Respect
   robots/ToS") makes this a hard stop regardless of (1): even if a working
   data endpoint were found, scraping it would violate the site's stated
   policy.

Neither fact was true when `docs/DATA_SOURCING_PLAN.md` §2 was written (or
wasn't checked against the live site) — this is a real, current change, not
a bug in this session's code. **P7 is not implemented.** The `xg`/`xgf`
ensemble components keep the existing SoT x conversion-rate proxy for every
competition, including the top-5 leagues Understat would have covered. If
real xG is wanted later, it needs a different source that (a) actually
serves the data server-side or via a documented API, and (b) permits
automated access — e.g. a paid provider, or re-checking Understat's policy
in case it's reverted.

### Files
- none changed for this phase

## Club Soccer Phase 8 — canonical identities and league environment experiment (2026-07-11)

Added canonical match reconciliation on date + competition + home + away. This
fixed duplicate provider records where football-data and BSD assigned different
IDs to the same match; the richer BSD detail row is retained and conflicting
scores remain a health failure rather than being averaged. The clean fixture
file now contains 17,830 played matches and the health check reports zero
duplicate identities.

Added a gated hierarchical league-season scoring environment and home-advantage
candidate. League-only estimates use recency-weighted empirical-Bayes shrinkage
to both season-level and competition-level fallbacks; cup and European fixtures
retain the existing competition-strength path.

**Result:** on 16,879 identical walk-forward predictions, the candidate was
rejected. Incumbent Brier was **0.612495** and log-loss **1.021347**; candidate
Brier was **0.612744** (+0.000250) and log-loss **1.021674** (+0.000327).
OU2.5 Brier improved slightly (0.245318 → 0.245214), but the primary 1X2
metrics worsened. The fitted tables are stored with
`league_adjustments_active=false` for future re-testing; live probabilities
remain on the incumbent model.

### Files
- `club_soccer/identities.py`
- `club_soccer/model.py`, `club_soccer/validate.py`
- `club_soccer/data/validation_league_adjustments.json`

## Club Soccer Phase 9 — temperature calibration promoted (2026-07-11)

The prior isotonic calibration candidate improved Brier on one held-out slice
but worsened log-loss, so it stayed inactive. Replaced it with a single
multiclass temperature parameter fitted on prior walk-forward predictions.

The fixed time-split gate improved both primary metrics on every split:

- 2025-01: ΔBrier **−0.000178**, Δlog-loss **−0.000344**
- 2025-07: ΔBrier **−0.000275**, Δlog-loss **−0.000466**
- 2025-12: ΔBrier **−0.000411**, Δlog-loss **−0.000629**

The all-data fitted temperature is **0.925**. Across 16,879 walk-forward
predictions, Brier improved **0.612495 → 0.612271** and log-loss improved
**1.021347 → 1.020937**. The calibration is now active for the displayed
1X2 probabilities and pricing paths; the underlying score model remains
unchanged.

### Files
- `club_soccer/calibrate.py`, `club_soccer/validate.py`
- `club_soccer/data/calibration.json`

## Club Soccer Phase 10 — BSD historical backfill and context promotion (2026-07-11)

Backfilled BSD event details for tracked fixtures from 2025-08-01 through
2026-05-31. The fixture history grew to **20,348 played matches**, real xG
coverage rose from **3.1% to 16.5%** (3,366 matches), and player cache
coverage now spans **2025-08-05 to 2026-07-09** with 27,690 applications.
Canonical identity and score-conflict checks remain clean.

On the same clean walk-forward framework, the raw model improved from the
pre-backfill **0.612495 / 1.021347** (Brier / log-loss) to **0.611685 /
1.020271** after the data backfill. BSD-detail rows are also materially more
predictive in the diagnostic split: xG component Brier **0.610338** versus
**0.618763** for rows without BSD xG. This is a source diagnostic, not a
separate promotion claim.

The context fit was re-run with the expanded cup and European history. The
minutes-load player term remains inactive because point-in-time coverage is
only 5%. Rest differential, European hangover and cup tier gap passed an
ensemble held-out gate: Brier **0.6076 → 0.6068** and log-loss
**1.0149 → 1.0139** on 2025-12 onward. Those coefficients are now active.

Combined production walk-forward performance is **0.611244 / 1.019665**
before calibration and **0.611079 / 1.019382** after the re-gated
temperature calibration. Ensemble weight re-tuning was still rejected by
the time-split gate.

### Files
- `club_soccer/data/fixtures.csv`, `club_soccer/data/bsd_cache/`
- `club_soccer/data/player_stats_cache.json`
- `club_soccer/data/context_coef_club.json`
- `club_soccer/context.py`, `test_club_soccer.py`
