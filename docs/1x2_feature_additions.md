# 1X2 Feature Additions — Design Draft

**Goal:** sharpen the 1X2 model further and, more importantly, convert its edge into
**positive closing-line value (CLV)**. The calibration/CLV diagnostic showed the model
is already well-calibrated and beats the base rate by ~0.20 log-loss over 7,138
competitive matches — but logged-bet CLV is ~flat-to-negative. So the brief is not
"make it predictive" (it is); it's "find the signal Elo can't see, and beat the
market's number."

All features feed the **expected-goals (λ) layer**, so 1X2, totals and BTTS stay
coherent (one score matrix). Each feature is gated the same way `dc_params.json`
already gates rho/home-adv: **promote only if it beats the incumbent on held-out
3-way log-loss by a set margin.** Calibrated-but-not-promoted features are kept as
inactive artifacts.

Current goal model (`predictor.fit_goal_model`) is a 2-parameter Poisson GLM:
`log λ = α + β · (eloΔ/400)`. Everything below extends that design matrix or adds a
final 1X2 blend.

---

## Priority order

| # | Feature | Why it helps | Data (all present) | Effort | Expected lift |
|---|---------|-------------|--------------------|--------|---------------|
| 1 | **Market blend, refit** | Market is the sharpest benchmark; directly targets CLV | `wc2018_odds.csv`, `wc2022_odds.csv`, `odds_history.csv` | Low | High (CLV) |
| 2 | **Squad availability** | Injuries/suspensions are invisible to Elo | `absences.csv`, `squad_ratings.csv` (`power_avail` vs `power_full`) | Low–Med | Med |
| 3 | **Attack/defence form (Dixon-Coles)** | Elo blurs *how* a team is winning; recent scoring/conceding adds signal | `results.csv` + existing `matchup.py` DC fit | Med | Med |
| 4 | **Rest & congestion differential** | Tournament fatigue; already computed, not in λ | feature store `rest_days_*`, `congestion_*` | Low | Low–Med |
| 5 | **Match context / motivation** | Dead rubbers and knockouts shift goals & draw rate | `tournaments.py` standings + stage | Med | Low–Med (draw) |
| 6 | **Squad market value prior** | Stabilises strength for teams with thin recent data | `ea_players.csv`, `squads.csv` | Med | Low |

Do **1 and 2 first** — highest value-per-hour and they hit the actual problem (CLV)
and the most obvious blind spot (who's actually playing).

---

## 1. Market blend, refit (the CLV lever)

**Status:** partially built. `market_blend.json` holds a logit-space 1X2 blend with
`w = 0.163`, but it was fit on **64 games** (WC2022) and `logloss_market_only (1.0003)`
already essentially matches `logloss_blend (0.9994)` — i.e. on that tiny sample the
market alone was as good as the blend. That weight is unreliable.

**Do:**
- Refit the blend weight `w` (per market: home/draw/away, and separately totals) on
  **all** available odds — `wc2018_odds.csv` + `wc2022_odds.csv` + `odds_history.csv` —
  with time-series CV, not a single tournament.
- Blend in logit space: `p_final = softmax(w · logit(p_market) + (1−w) · logit(p_model))`.
- **Use the blend for bet selection, not just pricing.** Only fire when the model
  disagrees with the market by more than a threshold *and* that disagreement has
  historically produced positive CLV. Track CLV as the KPI (see fix below).
- Your unders are the biggest bet bucket and carry slightly negative CLV — the market
  has already priced the low-scoring lean. Expect the refit blend to *suppress* most
  unders bets. That's the point.

**Bug to fix first (`core/clv.py`):** `--report` crashes comparing tz-aware
`snapshot_time` to a tz-naive cutoff. Fix:
```python
m["t"] = pd.to_datetime(m["snapshot_time"], errors="coerce", utc=True)
cutoff = pd.Timestamp(str(match_date), tz="UTC") + pd.Timedelta(days=1)
```

## 2. Squad availability

**Status:** scaffolded but inert. `squad_ratings.csv` has `power_full`, `power_avail`,
`att_adj`, `def_adj`, `elo_adj` — all the adj columns are currently `0` because
`absences.csv` is an empty template.

**Do:**
- Populate `absences.csv` (you already pull this on update). Each absence maps to a
  player power weight from `squad_ratings`/`ea_players`.
- Feature: `avail_gap = power_avail − power_full` per team → λ multiplier
  `λ_team *= exp(b · avail_gap_team)`. This reuses the exact `context_coef.json`
  mechanism (`λ *= exp(Σ bᵢ·featureᵢ)`).
- **Point-in-time:** absences must be dated; only apply those known before kick-off.

## 3. Attack/defence form (Dixon-Coles team strengths)

**Status:** `matchup.py` already fits a DC attack/defence view as **report-only**.

**Do:** promote `att_home − def_away` (and symmetric) as covariates in the λ GLM,
estimated on a trailing window (e.g. 24 months, half-life weighted) so it captures
*form* rather than all-time strength. Elo already has long-run strength; this adds the
short-run scoring/conceding signal Elo smooths out. Watch collinearity with `eloΔ` —
include the DC term as a *residual* on top of Elo, not a replacement.

## 4. Rest & congestion differential

**Status:** `rest_days_h/a`, `congestion_h/a` are in the feature store; `context_coef`
even stores `rest_cap=14`, `rest_diff_clip=7` — but no rest coefficient is active.

**Do:** add `clip(rest_h − rest_a, ±7)` and a congestion term to the λ GLM/context
multiplier. Small but free; matters most in the compressed knockout schedule.

## 5. Match context / motivation

**Do:** derive per-fixture flags from `tournaments.py` standings + stage:
- `dead_rubber` (a team already qualified/eliminated before the final group game),
- `knockout` (cagier, fewer goals, more draws→ETs),
- `must_win`.
These mainly correct the **draw rate** and the goal environment — exactly the 1X2
dimension where independent-Poisson is weakest. Feed as λ multipliers and/or a small
rho adjustment for knockouts.

## 6. Squad market value prior

**Do:** aggregate `ea_players.csv`/`squads.csv` to a team market-value/age index; use as
a slow prior that shrinks Elo for teams with little recent competitive data (minnows,
post-realignment squads). Lowest priority — overlaps with Elo for the big teams.

---

## The draw problem (cross-cutting)

Independent Poisson under-predicts draws; your `rho` helps but is a single global
constant. Two cheap improvements, both gate-able:
- Fit `rho` **conditional on context** (separate knockout vs group), since knockout
  football is materially drawier in regulation.
- Or add a small **draw-inflation** term to the 1X2 layer calibrated on held-out data.

---

## Integration & gating

1. Generalise `fit_goal_model` to a **multivariate Poisson IRLS** over a pluggable
   feature vector (scaffold: `engines/worldcup/features_1x2.py`).
2. Build features **point-in-time** in the existing feature store (it already enforces
   pre-match Elo, rest, and odds-as-of).
3. For each candidate feature (or set), run the **held-out 3-way log-loss** harness over
   2016+ competitive matches and the last 3–4 World Cups; **promote only on a real
   margin**, mirroring `--fit-dc`. Write rejected fits as inactive artifacts.
4. Re-run the calibration + CLV diagnostic after each promotion. Success = **CLV moves
   positive**, not ROI on a small sample.

See `engines/worldcup/features_1x2.py` for the scaffold (multivariate IRLS + feature
registry + promotion harness, with extractors 1/2/4 partly implemented and 3/5/6
stubbed).
