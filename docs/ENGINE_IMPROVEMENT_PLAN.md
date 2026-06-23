# Engine Predictive-Improvement Plan

Date: 2026-06-22

---

## Implementation results (2026-06-23)

Four self-contained, no-new-data items were implemented and validated walk-forward.
Discipline followed throughout: a change only becomes the default if it beats the
engine's held-out metric; otherwise the capability is built and gated OFF.

| Engine | Change | Held-out result | Status |
|---|---|---|---|
| **Golf** | Per-player round correlation (hot/cold week) + Student-t(6) round tails in the Monte Carlo (`round_corr=0.3`, `tail_df=6.0`, `golf/data/sim_config.json`) | Headline Brier **0.14726 → 0.14553**, winner log-loss **0.0501 → 0.0453**, every place market improved (139 events, 4000 sims) | **PROMOTED (default)** |
| **CFB** | Shrink the projected **total** toward the league mean (`TOTAL_SHRINK=0.80`), margin/win-prob untouched | Total MAE **13.05 → 12.86** in the gate; LOSO 2019-25 pooled **13.103 → 13.086**; moneyline/margin unchanged | **PROMOTED (default)** |
| **World Cup** | Data-driven Dixon-Coles **ρ + home advantage** via held-out-gated MLE (`--fit-dc`) | Fitted values **lose** on held-out competitive log-loss (0.85099 vs 0.84965) | Built + **gated OFF**; defaults confirmed near-optimal |
| **Club soccer** | Per-competition **HFA + ρ** (EB-shrunk), `comp_adj` table | Neutral-to-slightly-worse held-out Brier (0.61207 global vs 0.61216–0.61234) | Built + **gated OFF** (`comp_adj_active=False`) |
| **Tennis** | **Per-player serve/return rates** → matchup-specific serve base, **+ fitted total-games calibration** (`games_cal`) | Games-per-set correlation **+0.11 → +0.18**; held-out total-games MAE **6.13 → 5.49**, bias **+2.5 → 0.0** (ATP, train<2025/test 2025); match-winner / set-handicap / first-set unchanged by construction | **PROMOTED (default)**, falls back to fixed base + no cal when serve stats absent (WTA) |

Two landed as genuine accuracy gains (golf, CFB totals). The other two returned
honest negative results — the existing hardcoded constants were already near
optimal — so the validated defaults were kept and the new machinery sits behind a
flag that auto-promotes if future data shifts. The CFB EPA-into-totals idea was
also tested and rejected (held-out preferred 100% power).

What was deliberately NOT done here (needs external data): availability/lineups,
real xG, weather, serve/return stats. Those remain the biggest untapped levers
(see per-engine sections below).

Scope: improvements that should make each engine **more predictive** (lower
held-out Brier / log-loss, better margin/total MAE, better-calibrated outright
and prop probabilities). Codebase tidiness is explicitly out of scope — refactor
later. Every promotion below should clear the engine's existing walk-forward
validation gate before becoming a default; report-only first, default second.

The five production engines (per `app/engines/`): **World Cup / international**,
**club soccer**, **college football (CFB)**, **golf**, **tennis**.

Ranking convention below: items are ordered by expected predictive lift per unit
of effort within each engine.

---

## 1. World Cup / International (`engines/worldcup/`, `wc_v4/`)

### Current state
Elo over all internationals (K scaled by tournament + margin), a single global
Poisson goal model `log λ = α + β·ΔElo/400`, fixed Dixon-Coles `ρ = −0.10`, fixed
`HOME_ADV = 65`. Squad-availability (`squads.py`, `wc_v4/availability.py`) and
rest/altitude context (`context.py`) exist but are **report-only / default-off**.
Market-blend segmentation validated but left report-only.

### Improvement points
1. **Promote the availability + context layers into the default prediction.**
   This is the single biggest untapped signal already half-built. `availability.py`
   prices injuries/absences with lineup-confidence bands and GK/attack/defence
   splits; `context.py` fits rest + altitude corrections to λ. Both are gated off.
   The blocker (per the team's own notes) is lineup/absence labels and historical
   market coverage to validate against — fix the data, then let the gate decide.
2. **Fit `ρ` (Dixon-Coles) and home advantage from data instead of hardcoding.**
   `ρ = −0.10` and `HOME_ADV = 65` are constants. Joint-fit `ρ` (and per-side home
   λ multiplier) by maximum likelihood on the score distribution. Improves draw and
   correct-score calibration directly.
3. **Add over-dispersion (Negative-Binomial goals).** Pure Poisson + a single DC
   nudge understates the variance in international goal totals (blowouts, low-event
   knockout games). A fitted NB or a small λ-dependent dispersion term improves
   totals and BTTS calibration in the tails.
4. **Time-decay / tournament-cycle mean reversion in Elo.** Ratings aggregate
   matches back to 1872 with no recency weighting beyond K. Add an explicit decay
   (or between-tournament regression to a confederation mean) so a team's number
   reflects its *current* generation, not a decade-old peak.
5. **Blend a team attack/defence Poisson layer with the Elo-driven λ.** Elo is a
   single strength scalar; it cannot see "great attack / leaky defence" style. The
   club-soccer engine already has this exact attack/defence structure — port a
   shrunk version and blend it into λ. Style information Elo misses.
6. **Refresh and apply isotonic calibration on 1X2 + totals by default**
   (`calibrate.py`) so the headline probabilities are calibrated, not just the raw
   matrix.

### Implementation plan
- Phase A (data): assemble absence/lineup labels for recent tournaments and expand
  `data/*_odds.csv` historical market coverage (the market-blend gate needs samples).
- Phase B (fit-from-data): add a joint MLE for `ρ` + home multiplier in
  `predictor.fit_goal_model`; add NB option behind a flag; add Elo decay parameter.
  Each guarded by `wc_v4/validate_v4.py` walk-forward (WC2018/2022 replays).
- Phase C (promote): once availability/context clear the held-out log-loss margin,
  wire them into `coherent_board` / `edge.py` defaults; keep the report-only path as
  fallback.
- Validate with: `python3 -m wc_v4.validate_v4`, `predictor.py --backtest`,
  `validate_all.py --gate`.

---

## 2. Club Soccer (`club_soccer/model.py`)

### Current state
Strong ensemble already: goals attack/defence Poisson, Elo, long-run SoT-xG, recent
SoT-form, shot-pressure — blended by walk-forward-tuned weights. Team notes say the
**shot-volume proxies are near their easy-gain limit**.

### Improvement points
1. **Replace the SoT-conversion xG proxy with real shot-quality xG.** The `xg`/`xgf`
   components currently approximate xG as `SoT × league_conversion` — it ignores
   shot location/quality, the whole point of xG. A free per-shot xG source (e.g.
   Understat-style) is the largest remaining signal gain and directly upgrades two
   of the five ensemble components.
2. **Add lineup / injury / suspension availability.** There is currently *no*
   availability adjustment in club soccer (unlike the WC engine). A missing top
   striker or first-choice keeper moves λ materially. Reuse the WC squad-gap pattern.
3. **Fit competition strength from inter-league results.** `competitions.strength()`
   is a static hand-set map feeding both the Elo K and a λ `comp_adj`. Fit relative
   league strength from cross-league matches (continental cups, transfers of form)
   so promoted/relegated teams and cross-border fixtures are priced correctly.
4. **Per-competition home advantage and `ρ`.** `HOME_ADV_ELO = 55` and
   `DC_RHO = −0.08` are global constants; home edge varies a lot by league. Fit per
   competition (shrunk to the global value for thin leagues).
5. **Market open→close movement as a feature / CLV teacher.** The WC engine already
   has `market_model.py` (steam, reversal, "do-not-bet"). Port it to club soccer to
   suppress edges that are really just stale model vs moved market.
6. **Newly-promoted / data-thin team priors.** Teams with few fixtures get the
   global prior via the `+ global_avg·4` shrinkage; a division-strength prior would
   start them in a better place early-season.

### Implementation plan
- Phase A: add an `xg` provider to `club_soccer/fetch.py` writing per-match real xG
  columns; have `fit()` prefer real xG and fall back to the SoT proxy.
- Phase B: add `absences.csv` + a squad-gap λ adjustment loader; add a fitted
  competition-strength table; add per-competition HFA/ρ with shrinkage.
- Phase C: port `market_model` "do-not-bet" into `club_soccer/edge.py`.
- Validate with: `python3 club_soccer/validate.py --tune-ensemble` then `--gate`,
  and `test_club_soccer.py`. Promote only on held-out Brier/log-loss improvement.

---

## 3. College Football (`cfb/`)

### Current state
Elo + offense/defence power, blended (champion `w_elo = 0.60`). EPA/PPA wired as a
challenger but **default-off** (failed the gate). Totals come from the **power
model alone** (≈13.0 total MAE). Team's own next steps: QB / depth-chart / weather /
tempo.

### Improvement points
1. **Starting-QB availability adjustment.** The largest single predictive lever in
   CFB and currently absent. A team with its backup QB is a different team; Elo and
   power both lag by weeks. Add a QB-out / QB-change adjustment to win prob, margin,
   and total. Even a coarse "starter vs not" flag with a fitted point value helps.
2. **Weather for totals.** Wind and precipitation systematically lower totals and
   are knowable pre-kick. Totals are the weakest output (single-model, 13-pt MAE);
   a fitted weather correction is high-value and uncontroversial.
3. **Tempo / pace-adjusted totals.** Power totals ignore plays-per-game. Two teams
   with the same per-play efficiency but different tempo project to very different
   totals. Bring pace into the totals model (EPA module already has per-play data).
4. **Make totals a blend, not power-only.** Route EPA/pace into the total even while
   EPA stays out of the win-prob blend — the EPA ablation rejected EPA *for
   moneyline*, but totals are a different target and were never the reason it lost.
5. **Returning-production / transfer-portal preseason priors.** `priors.py` exists;
   confirm early-season weeks (where Elo is stalest) lean on returning production and
   portal movement. This is where the model is weakest each August/September.
6. **Venue-specific HFA** (altitude, travel distance) instead of the single
   `HFA_ELO = 62`.

### Implementation plan
- Phase A (totals): add weather + pace features to a totals model in `power.py`/
  `epa.py`; backtest with `totals_backtest.py` (currently total MAE is flat because
  totals are power-only).
- Phase B (QB): add a `qb_status.csv` + fitted adjustment in `predictor.blend_predict`;
  walk-forward via `cfb/validate.py --since 2023`.
- Phase C (priors): verify/strengthen `priors.py` returning-production weighting for
  weeks 1–4.
- Validate with: `python3 cfb/validate.py --since 2023 --gate`, `test_cfb_blend.py`,
  `totals_backtest.py`, `ats_backtest.py`. Keep EPA-for-moneyline off unless the gate
  flips.

---

## 4. Golf (`golf/`)

### Current state
Fitted skill/σ/form/course-fit from `score_to_par` via ridge least squares; 4-round
Monte Carlo with a 36-hole cut. **Round scores are drawn iid Normal per player**
(`rng.normal(loc=-rating, scale=sigma)`), with **no round-to-round correlation and
no field-wide common shock**. SG categories are loaded but not used inside the
fitted skill. Team's own next steps: SG categories, tee-wave/weather, course
archetype.

### Improvement points
1. **Add a common per-round field shock (course playing hard/easy).** Right now each
   player's four rounds are independent Normals; in reality a round has a shared
   difficulty (weather, pin positions) that moves everyone together. Model
   `score = field_round_effect + player_mean + idiosyncratic`. Without it the sim
   *understates* the variance of the leaderboard and mis-prices outrights and top-N
   (too confident). High-value, self-contained change to `simulate.py`.
2. **AM/PM tee-wave & weather draw bias (R1/R2).** A real, well-documented golf edge:
   one wave can play a materially easier course. Currently absent. Add a per-wave
   round-1/2 bias when tee-time/forecast data is available.
3. **Heavy-tailed / right-skewed round distribution.** Golf scores are right-skewed
   (blow-up holes); a Normal understates disaster rounds and overstates the floor.
   A skew-normal or t-distributed idiosyncratic term improves win/blow-up tails,
   which is exactly where outright and top-5 pricing lives.
4. **Use strokes-gained categories + course archetype fit.** Skill is a single
   number from total score; the loaded OTT/APP/ATG/PUTT splits are unused in the fit.
   Weight categories by course archetype (bomber vs positional, fast vs soft greens)
   so course-fit is structural, not just a per-course residual with a 4-round
   minimum.
5. **Condition-dependent σ.** σ is per-player only; major/hard setups raise variance
   for everyone. Tie σ to course difficulty, not just the player.

### Implementation plan
- Phase A: refactor the score draw in `simulate_tournament` to
  `field_effect[sim,round] + player_mean + idiosyncratic`, with a fitted
  field-variance share; re-validate calibration (`golf/validate.py --gate --sims`).
  This alone should improve outright/top-N calibration.
- Phase B: add skew/heavy-tail idiosyncratic draw; sweep on the validation harness.
- Phase C: add SG-category skill decomposition + a course-archetype weighting table
  to `model.fit`; add tee-wave bias when `field.csv` carries tee times/forecast.
- Validate with: `python3 golf/validate.py --quiet --gate --sims 4000`,
  `test_golf_config.py`. Only promote on the headline-Brier margin the gate enforces.

---

## 5. Tennis (`tennis/`)

### Current state
Surface-split Bradley-Terry skill + surface offset + EB-shrunk form + tiny H2H,
fitted by ridge logistic regression. Match prob → set/game sub-markets via a Markov
chain, but the serve probabilities are derived as a **symmetric edge around a single
`BASE_SERVE = 0.64`** for both players — i.e. all sub-markets are anchored to one
shared serve baseline.

### Improvement points
1. **Player- and surface-specific serve/return profiles.** Deriving both players'
   serve point-win from one symmetric edge around 0.64 means total-games, set-
   handicap and first-set markets ignore that some players are big servers and
   others are grinders. Fit serve-hold and return rates (or use serve/return stats)
   so the Markov chain reflects real serve dominance. Biggest sub-market accuracy
   gain.
2. **Fatigue / match load.** No rest or cumulative-load term. Sets played and days
   since last match within a tournament measurably shift win prob (especially best-
   of-5 and back-to-back days). Add a fitted fatigue nudge.
3. **Best-of-5 vs best-of-3 awareness.** Match-format variance differs; the favourite
   is more likely to win bo5. Ensure the chain and any calibration are format-aware
   for Slam pricing.
4. **Indoor / altitude / ball-speed context.** Conditions shift serve dominance
   (fast indoor favours servers). A court-speed feature improves both the winner and
   the games markets.
5. **Full surface-specific skill for specialists** (not just an offset) where a
   player has enough surface matches — clay/grass specialists are under-served by an
   offset shrunk to 0.
6. **Walk-forward tuning of `form_weight` and `h2h_weight`** (currently fixed config
   defaults 0.5 / 0.05); let `validate.py` choose them on held-out data.

### Implementation plan
- Phase A: add serve/return rates to `model.fit` (or a serve-stats loader) and feed
  per-player `ps_a/ps_b` into `simulate._set_dp` instead of the symmetric 0.64 edge;
  validate games/sets sub-markets against results.
- Phase B: add fatigue (sets/rest) and court-speed features; format-aware calibration.
- Phase C: enable per-surface skill for high-sample specialists; tune form/H2H weights
  on the walk-forward.
- Validate with: `python3 tennis/validate.py` walk-forward + gate, `test_tennis_contract.py`.

---

## Cross-engine themes

These recur and are worth treating as shared work:

- **Availability/lineups** is the highest-value missing signal in *three* engines
  (WC has it built but off; club soccer and CFB lack it). Prioritise the data plumbing.
- **Fit constants from data** rather than hardcoding (`ρ`, home advantage, league
  strength, σ shape). Several engines bake these in.
- **Market movement as a teacher** (`market_model.py`) exists only for the WC engine;
  porting the "do-not-bet / CLV" logic suppresses false edges everywhere.
- **Promotion discipline**: every change stays report-only until it beats the
  engine's walk-forward gate on held-out log-loss/Brier by the existing margin. Land
  data and report-only signal first; flip defaults only on evidence.
