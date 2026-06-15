# V2_NOTES — implementation log

Dated entries from the agent executing `V2_PLAN.md`. One milestone per entry:
what shipped, fitted parameters, and gate numbers.

---

## M8 — Ops polish (2026-06-14)

**Shipped:** daily-pipeline wiring, a morning bet queue, and an offline dashboard.

- **update.sh** now also runs (after the edge report): `clv.py --snapshot` (closing
  odds for open bets; degrades offline), `validate.py --quiet --gate` (loud warning
  on regression but **never blocks** — guarded with `||`, verified it continues
  under `set -euo pipefail`), and `report.py`. Still finishes offline end-to-end
  (only the existing git clone needs network; CLV snapshot degrades gracefully).
- **bet_queue.csv** (M8b): `edge.py` writes the day's reviewable candidates
  (match, bet, odds, p_model/p_book, edge, sized stake, active-adjustment flags,
  and any squad adjustments in play) — written whether or not bets are recorded,
  so the scheduled summary can surface it before real money goes down.
- **report.py → dashboard.html** (M8c): single self-contained page, **inline SVG
  only — no external JS/CSS**, opens via `file://`. Panels: bankroll curve, CLV
  trend (rolling mean), calibration plot (the fitted isotonic maps vs the
  diagonal), today's bet queue, and title-odds movers (snapshots champion % to
  `data/title_history.csv` and shows ▲▼ vs the previous snapshot). ~15 KB.
- **README**: added a "v2 (new)" section documenting every new flag/command, and
  corrected the now-stale Annex C / squad-power / knockout-settlement descriptions.

**Acceptance:** `update.sh` syntax OK and its new steps run offline; gate-failure
guard confirmed not to abort the script; `report.py` renders `dashboard.html`
(all panels present); every README-documented command spot-checked (including the
combined `--squad-adj --calibrated --market-blend --context`) ✓; `validate.py
--gate` PASS ✓; default outputs unchanged.

**Scheduled task:** the daily-update prompt is standing config — per the plan I did
**not** change it without sign-off; proposed update covers reviewing `bet_queue.csv`
and opening `dashboard.html` in the summary. Awaiting Barrie's go-ahead.

**Files touched:** added `report.py`; edited `update.sh` (CLV/gate/dashboard
steps), `edge.py` (bet_queue.csv), `README.md`. New outputs: `bet_queue.csv`,
`dashboard.html`, `data/title_history.csv`. No change to the ledger or any existing
default output.

---

## M6 — Context features: rest, travel, altitude (2026-06-14)

**Shipped:** `context.py` — a fitted multiplicative λ correction for rest and
altitude, applied behind `edge.py --context` (opt-in; default unchanged).

### Features & fit
- **rest_diff**: own rest days − opponent's (raw rest capped at 14, diff clipped
  ±7), from results.csv match spacing.
- **alt_gap**: how far above a team's usual altitude the venue sits, in km. A
  team's usual altitude = median of its non-neutral home-venue altitudes (data
  driven, from a hardcoded city→metres table covering the 16 WC2026 venues + the
  high-altitude football cities); venues below 1000 m count as lowland. So lowland
  sides at altitude are penalised, altitude sides are not.
- **travel** (great-circle since last match): **dropped** — it needs a city
  coordinate dataset we don't have for historical matches, and the plan expects it
  insignificant. Not shipped as noise.

Fit: Poisson GLM with the model's log-λ as a fixed offset, on competitive
(non-friendly) internationals since 2010 (21,500 side-observations).

| term | coef | SE | t | kept |
|---|---|---|---|---|
| intercept | +0.0211 | 0.0058 | 3.65 | no (not applied) |
| rest_diff | +0.0006 | 0.0026 | 0.24 | **dropped (|t|<2)** |
| alt_gap | **−0.1411** | 0.0248 | **−5.70** | **yes** |

Applied as `λ_side *= exp(Σ bᵢ·featureᵢ)` (kept coefficients only; intercept not
applied). So a sea-level team at Mexico City (gap ≈ 2.24 km) has its λ scaled by
exp(−0.141·2.24) ≈ **0.73** (−27% scoring); an acclimatised side is unaffected.
Saved to `data/context_coef.json`.

### Validation (held-out, fit pre-2022, test after)
- Literal acceptance set (tournament matches after 2022, n=422): mean log-loss
  base 0.9632 → +context 0.9632 (Δ +0.0001, within noise) — **flat by design**,
  since post-2022 tournaments (Qatar, Germany, USA) are all lowland.
- Where the feature actually applies (altitude competitive matches after 2022,
  n=77 — mostly CONMEBOL qualifiers at La Paz/Quito/Bogotá): base 1.0077 →
  **+context 0.9597 (Δ −0.0480)** — a clear, material improvement.

**Acceptance:** context log-loss ≤ baseline on tournament matches (flat, within
tolerance) and materially better where altitude applies ✓; coefficients reported
with std errors ✓; insignificant features dropped (rest, travel) ✓; `edge.py`
without `--context` byte-identical (`market_probs(ctx=None)` is an exact no-op) ✓;
`validate.py --gate` PASS ✓; `test_m6.py` all pass ✓.

**Where it applies:** the correction is applied per concrete fixture in `edge.py`
(date + venue known). It is not applied inside deep `simulate.py` knockout
branches, where future venues/opponents are hypothetical — documented limitation;
the WC2026 high-altitude venues are Mexico City (2240 m) and Guadalajara/Zapopan
(1566 m), all in the group stage and early knockout, which `edge.py` covers.

**Files touched:** added `context.py`, `test_m6.py`, `data/context_coef.json`;
edited `edge.py` (`--context` flag, `market_probs` ctx hook, default no-op). No
change to the ledger, results.csv, or any default output.

---

## M5 — Squad layer v2 (2026-06-14)

**Shipped:** position-aware, starter-weighted squad availability adjustments, plus
a full backfill of the 12 `ea_proxy` squads with official 2026 lists. All changes
live behind `--squad-adj` (opt-in); the default pipeline is untouched.

### Code (squads.py)
- **Starter-weighting** replaces the flat top-18 mean: ranks 1-11 count full,
  12-18 half, the rest zero (likely-minutes proxy). Removing a starter promotes a
  weaker backup, so availability gaps are captured naturally. Elo-per-rating-point
  recalibrated to **24.8** (was ~23; fit on 37 well-covered teams).
- **Position-aware att/def split**: `elo_adj` is split into `att_adj` + `def_adj`
  by the absent players' positions (GK/DF → 75% defence, MF → 50/50, FW → 75%
  attack), weighted by overall. `adjusted_sources` applies them asymmetrically —
  a team's attack gap lowers its OWN λ, its defence gap raises the OPPONENT's λ
  (factor 2k, which reduces exactly to the old symmetric model for a 50/50 split).
  New `att_adj`/`def_adj`/`def_frac` columns in `squad_ratings.csv`;
  `load_adjustments`/`load_adj` (elo_adj) kept for the edge.py combined path.

### Backfill (data/squads.csv)
- Replaced **225 ea_proxy rows** (Iran, Iraq, Jordan, New Zealand, Norway, Panama,
  Portugal, Saudi Arabia, Senegal, Spain, Uruguay, Uzbekistan) with **312 official
  2026 rows** (source `wiki`), captured from Wikipedia "2026 FIFA World Cup squads"
  via the browser (`build_squads_2026.py` reproduces the merge). **0 ea_proxy
  remain; 48/48 squads official.**
- Coverage caveat: EA FC26 lacks ratings for many Iran/Iraq/Jordan/Uzbekistan/
  Panama players, so those teams now match < 15 EA players and correctly fall back
  to **no adjustment** (the safe default) — official squad *membership* is right,
  but EA-rating coverage is thin for those nations. Documented, not a regression.

### Sanity backtest (squad_backtest_wc2022.py)
- 7 well-known WC2022 absences (Mané, Kanté, Pogba, Nkunku, Kimpembe, Werner,
  Jota), EA FC26 used as the FC23 approximation (FC23 not obtainable). Mean
  log-loss on 19 affected matches: no-adj 0.9845, symmetric 0.9919, position-aware
  0.9933 — both deltas tiny and **within the noise band** (±0.01 at n=19). The
  small positive Δ vs no-adjustment is driven entirely by France reaching the 2022
  final despite four listed absences (an ex-post outlier); the method helps the
  underperformers (Senegal). Conclusion: not materially worse, so `--squad-adj`
  stays opt-in (default off), consistent with the working agreement.

**Acceptance:** 48/48 squads `source != ea_proxy` ✓; what-if CLI works
(`--match … --without …`) ✓; WC2022 check not materially worse than no-adjustment
✓; default (no `--squad-adj`) output unchanged — squads.csv is off the default
path; predictor `--worldcup` is deterministic (the only change vs the committed
file was 2 now-played 2026-06-13 fixtures dropping out, from results.csv advancing
between sessions, unrelated to M5) ✓; `validate.py --gate` PASS ✓; `test_m5.py`
all pass ✓.

**Files touched:** `squads.py` (starter-weighting, att/def split, load_adj_split,
asymmetric adjusted_sources); `data/squads.csv` (backfilled); added
`build_squads_2026.py`, `squad_backtest_wc2022.py`, `test_m5.py`;
`data/squad_ratings.csv` regenerated (new columns). No change to the ledger,
results.csv, or any default output.

---

## M7 — Portfolio staking discipline (2026-06-13)

**Shipped:** joint position sizing in `edge.py`'s recording step, plus drawdown
state in `bankroll.py`. Default ON (it's risk management); `--no-portfolio`
restores the old sequential sizing.

`portfolio_size()` applies, in order:
1. **Drawdown brake** — when bankroll < 70% of its running peak, Kelly is halved
   until a new peak. Peak is tracked in `data/bankroll.json` (new `peak` field),
   updated by `settle()`; backward-compatible (migrated from the ledger's
   `bankroll_after` history when the field is absent).
2. **Single-match cap** — no stake exceeds 10% of bankroll.
3. **Correlation guard** — combined exposure on bets sharing a team (incl. existing
   open bets, e.g. an outright EW + a match win on the same side) is capped at
   1.5× the single-match cap (15%). Best-EV bets are filled first.
4. **Simultaneous-Kelly daily cap** — total across all new same-day bets ≤ 25% of
   bankroll; if exceeded, all stakes scale down proportionally (floored to pennies
   so the rounded total never creeps back over the cap).

The rescaled fraction is fed to `place_bets`, which records the disciplined amount;
pre/post stakes are reported.

**Caps (fractions of bankroll):** single-match 0.10, daily 0.25, correlated-group
0.15 (=1.5×0.10), drawdown trigger 0.70, brake ×0.5. Reasonable defaults — tune in
`edge.py` if Barrie wants a different risk appetite.

**Acceptance:** synthetic 6 big-edge same-day bets (£20 each, £120 pre) → £24.96
recorded, i.e. ≤ the £25 daily cap, each ≤ the £10 single cap ✓; correlation guard
caps two same-team bets at £15 and caps a new bet against an existing £12 open
outright on the same team to £3 ✓; drawdown brake halves Kelly below 70% of peak
(£3 → £1.50) ✓; v1 `bankroll.json` (`{"bankroll":…}`) loads, peak migrates to
£114.13 from history ✓; **ledger.csv format unchanged** (11 cols; stake_pre/post
never leak) ✓; `validate.py --gate` PASS ✓; `test_m7.py` all pass ✓.

**Note on default-ON:** unlike the analytical flags (M2/M3, opt-in), M7 changes the
recorded *stake values* by default because it's a safety mechanism the acceptance
exercises with plain `edge.py`. The ledger *format* is unchanged; current bankroll
(£98.25 vs £114.13 peak = 86%) is above the drawdown trigger, so the brake is
currently inactive and typical small quarter-Kelly stakes sit well under the caps —
day-to-day behaviour is largely unchanged until many big edges or a drawdown occur.

**Files touched:** `edge.py` (caps, `_bet_teams`, `portfolio_size`,
`--no-portfolio`, recording-step integration), `bankroll.py` (peak tracking in
state + settle/reset); added `test_m7.py`. No change to ledger history or outputs.

---

## M2 — Probability calibration (2026-06-13)

**Shipped:** isotonic per-outcome calibration of the blend model's 1X2
probabilities, fit in `validate.py`, applied (opt-in) in `edge.py`.

- `validate.py --calibrate`: fits isotonic regression per outcome (H/D/A) on the
  walk-forward **blend** predictions, renormalises the three calibrated probs to
  sum to 1, and writes piecewise-linear knots to `data/calibration.json`
  (≤300 knots/outcome). Uses **sklearn's `IsotonicRegression` when installed**
  (Barrie just added scikit-learn) and falls back to a dependency-free **PAV**
  implementation otherwise — both compute the same fit, so the daily pipeline
  never hard-depends on sklearn. (This sandbox has no sklearn, so the run below
  exercised the PAV path; Barrie's machine will use sklearn.)
- New `calibrate.py` with `apply(p_home, p_draw, p_away)` — **apply only, never
  fits** (refitting is exclusively `validate.py`'s job, per the plan).
- `edge.py --calibrated` (opt-in): calibrates the model's 1X2 **after the model
  blend, before any --market-blend** (so `--calibrated --market-blend` = calibrate
  then anchor). OU2.5/BTTS untouched (calibration was fit on 1X2). Verified the two
  flags combine cleanly.

**Held-out acceptance** (fit on walk-forward blend preds < 2025-12-01, test on the
final ~6 months, n=382):

| | accuracy | Brier | log-loss |
|---|---|---|---|
| raw blend | 61.0% | 0.5170 | 0.8795 |
| calibrated | 60.5% | **0.5144** | **0.8753** |

Brier −0.0026 and log-loss −0.0041 — calibrated ≤ raw on both ✓ (accuracy dips
slightly because isotonic optimises probability quality, not argmax). Reliability
table visibly flattened: the systematic high-end under-confidence shrank, e.g.
[0.6,0.7) gap −0.111 → −0.032, [0.8,0.9) −0.089 → +0.006, [0.0,0.1) +0.040 →
−0.005. Production map saved fit on all 4,556 predictions.

**Acceptance:** calibrated Brier & log-loss ≤ raw on held-out ✓; reliability
flattened ✓; `edge.py` without `--calibrated` byte-identical (1X2 `p_model`
unchanged; non-1X2 untouched; flag-off path skipped) ✓; `validate.py --gate`
PASS ✓; `test_m2.py` all pass ✓.

**Files touched:** added `calibrate.py`, `test_m2.py`, `data/calibration.json`;
edited `validate.py` (`--calibrate`, isotonic/PAV + sklearn routing) and `edge.py`
(opt-in `--calibrated`, default no-op). No change to default outputs or the ledger.

---

## M3 — Market-anchored probabilities + CLV tracking (2026-06-13)

**Shipped:** logit-space market blend for the model's 1X2 probs (opt-in) and a
full closing-line-value (CLV) tracking module.

### Market blend
- New `market_blend.py`: `blend(p_model, p_market, w)` blends per outcome in logit
  space then renormalises (`logit p_final = w·logit p_model + (1−w)·logit p_market`).
- `market_blend.py --fit` fits **w by max-likelihood on WC2022** (`data/wc2022_odds.csv`
  + the same leak-free blend model the replay uses) and writes `data/market_blend.json`.
  **Fitted w = 0.163** (n=64). Log-loss: model-only 1.0278, market-only 1.0003,
  **blend 0.9994** — strictly better than BOTH extremes ✓ (acceptance met). w landed
  a touch below the plan's 0.2–0.4 guess because the WC2022 closing market was very
  sharp relative to the model; the optimiser leans toward the market accordingly.
- `edge.py --market-blend` (opt-in): blends the model's 1X2 toward the de-vigged
  market using stored w; **OU2.5/BTTS untouched** (w was fitted on 1X2). Edges are
  still computed vs the **raw** de-vigged market — the blend only moves p_model.
  Structured so a future M2 `--calibrated` runs first, then the blend.
- **Effect (live odds, back-to-back run):** mean |p_model − p_market| on 1X2
  shrank 0.064 → 0.010; edges ≥3% fell 121 → 44 overall (78 → 1 on 1X2). Far fewer
  fake edges — exactly the intent.

### CLV tracking
- New `clv.py`:
  - `--snapshot`: records current odds for **open** ledger bets to
    `data/odds_history.csv` (`snapshot_time,match_date,home,away,side,odds`) via The
    Odds API; degrades gracefully offline (clear message, writes nothing). Needs
    network → run on Barrie's machine (M8 will add it to update.sh).
  - `--report`: per settled bet CLV% = bet_odds/closing − 1 (closing = latest
    snapshot at/before kick-off); rolling mean CLV, positive-CLV rate, win rate,
    actual P&L vs CLV-expected P&L. Empty history prints the "no snapshots yet"
    message (acceptance ✓).
  - Shared `compute_clv(ledger, hist)` reused by the tracker.
- `refresh_tracker.py`: adds a **`clv` column** (col 12; the note stays in col 13)
  to the Project Ledger sheet, formatted as %. Blank until snapshots exist.

**Acceptance:** fitted w strictly beats both extremes on WC2022 ✓; `edge.py` default
output unchanged (p_model byte-identical to the pre-M3 report; only live-odds-derived
columns moved) ✓; `--market-blend` cuts ≥3% edges sharply ✓; `clv.py --report` on
empty history prints "no snapshots yet" ✓; **ledger.csv format untouched** (CLV is
computed on the fly / stored separately, never added to the ledger) ✓;
`validate.py --gate` PASS ✓; `test_m3.py` all pass ✓.

**Notes / caveats:** the `clv` column was verified on a COPY of `Betting Tracker.xlsx`
because the live workbook appears open in Excel (lock file present); it will populate
on the next `update.sh`/`refresh_tracker.py` run with the workbook closed. The Odds
API was reachable from the sandbox this session (the snapshot path still degrades
gracefully if it isn't).

**Files touched:** added `market_blend.py`, `clv.py`, `test_m3.py`,
`data/market_blend.json`; edited `edge.py` (opt-in `--market-blend`, default
no-op), `refresh_tracker.py` (clv column). `data/odds_history.csv` is created on
first snapshot. No change to ledger/bankroll history or `data/results.csv`.

---

## M1 — Validation harness with regression gates (2026-06-13)

**Shipped:** `validate.py` — walk-forward, no-leakage evaluation of the three match
models (elo / dc / blend), plus a CI-style regression gate.

**How it works**
- Elo is point-in-time already (`compute_elo` records pre-match `elo_h`/`elo_a`), so
  scoring a match with those columns never uses its own result.
- At each calendar-month boundary in the test window, the Elo→goals map
  (`fit_goal_model`) and the Dixon-Coles model (`fit_dc`) are refit on matches
  **strictly before** that month (DC anchor = cutoff − 1 day, so a match dated on the
  first of the month can't leak into its own training set).
- DC fits are cached in `data/validation_cache/dc_<YYYY-MM>.json` (first full run
  ~23s; cached reruns ~1s). Caches are never deleted (mounted `data/` forbids unlink
  from the sandbox) — stale ones are simply ignored.
- Metrics per model: 3-way accuracy, Brier, log-loss, and a 10-bin reliability table
  (predicted vs observed frequency, pooled one-vs-rest over H/D/A).
- Randomness seeded with `numpy default_rng(42)`; harness is otherwise deterministic.

**Flags:** `--since <date>` (default 2022-01-01), `--reliability`, `--gate`,
`--update-baseline`, `--quiet`. All default OFF; default invocation is read-only
except writing the baseline on first run.

**Gate:** `--gate` exits non-zero if blend Brier exceeds the stored baseline by
> 0.002 (`data/validation_baseline.json`). Baseline auto-written on first run.

**Results (walk-forward since 2022-01-01, n=4556):**

| model | accuracy | Brier | log-loss |
|---|---|---|---|
| Elo+Poisson | 60.1% | 0.5138 | 0.8729 |
| Dixon-Coles | 59.8% | 0.5123 | 0.8725 |
| 50/50 blend | **60.3%** | **0.5099** | **0.8679** |

**Reference check (sub-window since 2024-01-01, n=2546):** blend 60.3% / **0.5034** —
matches the v1 reference (blend 60.4% / 0.5038) within noise. ✓

**Stored baseline:** blend Brier 0.5099 (gate_tol 0.002 → limit 0.5119).

**Reliability note for downstream milestones:** Elo is already well-calibrated
(gaps ≤ 0.02 across bins); DC and the blend are mildly over-confident at the high
end (e.g. blend [0.7,0.8) predicts 0.747 but observes 0.787; [0.8,0.9) predicts
0.846 observes 0.895). This is exactly the miscalibration M2 (isotonic calibration)
is meant to remove — the reliability table will be the before/after evidence.

**Acceptance:** `validate.py` prints the three-model table + writes baseline ✓;
`validate.py --gate` exits 0 (PASS) ✓; blend within noise of v1 reference ✓;
`simulate.py` already accepts `--seed` (default 42, no change needed) ✓;
`update.sh` untouched by M1 (syntax OK) ✓.

**Files touched:** added `validate.py`, `data/validation_baseline.json`,
`data/validation_cache/` (new). No existing file modified — v1 behaviour byte-compatible.

**Next:** per the plan's value order, M3 (market anchoring + CLV) or M4 (knockout
correctness, hard deadline before R32 on June 28) are the next candidates.

---

## M4 — Knockout correctness: 90-min settlement + Annex C hook (2026-06-13)

**Shipped:** the deadline-critical 90-minute settlement path for knockout bets,
plus the mechanism for FIFA's exact Annex C third-place table. Done before the
R32 deadline (June 28).

### Part 1 — 90-minute 1X2 in edge.py (no code change needed)
On inspection, `edge.py`'s `market_probs` already builds 1X2 / O-U / BTTS from the
single-match (90-minute) score matrix with the draw kept intact — exactly the
correct settlement basis. Extra-time/penalty logic lives only in `simulate.py`
(progression), never in edge pricing. Added a clarifying docstring; **no
behavioural change** (group-stage and knockout edge output unchanged). The real
"full-time" bug was only on the settlement side (Part 2).

### Part 2 — 90-minute knockout settlement in bankroll.py
`data/results.csv` records the **after-extra-time** score (verified: 2022 final
stored as 3-3, NED-ARG 2-2, etc.; penalties excluded). A knockout that is level at
90' but scores in ET would mis-settle a 90' 1X2 bet. Fix: new
`data/ko_overrides.csv` (`date,home,away,score90`); `settle()` consults it first
for the 90-minute score, else falls back to the results.csv score. The daily task
fills it from news when a knockout goes to extra time.
- **Backward compatible:** file is header-only today → `_load_ko_overrides()`
  returns `{}` → settlement byte-identical to v1 for the group stage. Verified
  `bankroll.py` status (read-only) still loads the live ledger.
- Settlement log now tags override rows with `[90']`.

### Part 3 — Annex C third-place table (COMPLETE — official table active)
**Finding:** the official table is genuinely required — for the 48-team format the
allowed-group constraints leave **3–214 valid matchings for every one of the 495
group combinations** (0 are uniquely determined; measured in-session). FIFA's
choice is not derivable, only tabulated (Annex C of the 2026 regulations; 495
scenarios, no secondary draw).
- **Obtained the full table** via the Claude-in-Chrome browser (Wikipedia is
  proxy-blocked in the sandbox and `web_fetch` times out on that large page).
  Source: Wikipedia "2026 FIFA World Cup knockout stage" rendering of FIFA regs
  Annex C. Transcribed to `data/annexc_raw.txt`; `build_annexc.py` parses it to
  `data/annexc_thirds.json`.
- Column→slot mapping (the 8 result columns are the group winners that play a
  third): `1A→T79, 1B→T85, 1D→T81, 1E→T74, 1G→T82, 1I→T77, 1K→T87, 1L→T80`
  (from the R32 schedule, e.g. Match 79 = Winner A vs 3rd ⇒ 1A = slot T79).
- **Validated all 495 rows** as genuine perfect matchings against `THIRD_SLOTS`
  with full coverage (exactly C(12,8)=495, distinct). Because every slot has a
  different allowed-group set, an incorrect column→slot mapping would have made
  many rows violate the constraints — all 495 passing is strong evidence the
  mapping and transcription are correct.
- `simulate.py` loads `data/annexc_thirds.json` at import (re-validating each
  entry; malformed table fails loudly) and `allocate_thirds` uses it
  deterministically. If the file is ever removed it falls back to the previous
  constraint-valid backtracking allocator unchanged.
- **Effect:** `tournament_odds.csv` now reflects the official allocation (replaces
  the old random-valid fallback). Shifts are small and bounded: max |Δ| ≈ 0.024 on
  reach-R16, 0.022 on reach-QF, 0.005 on champion; no NaNs; probabilities sane.

**Tests:** `test_m4.py` (no pytest dep) — all pass: settlement flips draw
WON(90')↔LOST(FT) via overrides and credits the bankroll correctly;
`allocate_thirds` yields a valid complete slotting for **all 495** combinations;
Annex C loader uses a valid table and rejects a malformed one.

**Acceptance:** group-stage `predictions_worldcup_2026.csv` byte-identical ✓;
all 495 patterns slot without crashing, all validated as perfect matchings ✓;
`validate.py --gate` PASS (blend Brier 0.5099, unaffected — it scores match-level
data, not the bracket) ✓; `update.sh` syntax OK (unchanged) ✓; WC2022 replay runs ✓
(fixed a *pre-existing* import break — `edge.BET_EDGE_MIN` was removed in an earlier
refactor; defined the 3% threshold locally in the replay, not in edge.py).
NB `tournament_odds.csv` intentionally changes now that the official Annex C table
is in use (this is the M4 deliverable, not a regression); it was confirmed
byte-identical *before* the table was added, isolating the change to the table.

**Files touched:** `bankroll.py` (settlement + overrides), `simulate.py` (Annex C
loader + use), `edge.py` (comment only), `wc2022_replay.py` (import fix); added
`data/ko_overrides.csv` (header only), `data/annexc_raw.txt` (495-row source),
`data/annexc_thirds.json` (built/validated), `build_annexc.py`, `test_m4.py`.
No change to ledger/bankroll history or `data/results.csv`.
