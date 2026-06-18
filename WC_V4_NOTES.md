# World Cup V4 — build notes (M1–M7)

**Date:** 2026-06-17 · **Scope built:** V4 World Cup modelling slice (M1–M7).
V3 is fully
landed (git: V3 M1–M9 merged), so the V4 prerequisites — common engine contract,
`event_id`/ledger columns, validation gates, provenance — all exist and are reused
rather than reinvented.

Everything lives in a new, self-contained `wc_v4/` package plus `test_wc_v4.py`.
**No V3 default, engine, settlement, security, or bankroll code was changed**
(guardrail #5). The V4 layers are report-only until they clear the held-out gate.

## What was built

| Plan | Module | What it does |
|------|--------|--------------|
| **M1** | `wc_v4/schema.py` | Single source of truth for the feature row: point-in-time `FEATURE_COLUMNS` vs `OUTCOME_COLUMNS` (result, **closing line**, CLV, P&L). `assert_no_leakage()` is the chokepoint the leakage tests drive. |
| **M1** | `wc_v4/feature_store.py` | Leak-free per-event feature rows. `build_training_matrix()` (historical, labelled) and `build_asof()` (live, features only). Pre-match Elo + month-boundary goal model + earlier-only rest/congestion + opening/current odds. Every row carries `asof, event_id, source, fetched_at, schema_version`. |
| **M2** | `wc_v4/market_model.py` | `line_history()` (open/current/close + steam/reversal movement), `segment_blend_weights()` (group vs knockout, **leave-one-out** decision), `clv_series()` (CLV as a first-class metric), `do_not_bet()` (flags edges that are likely just market info). |
| **M3** | `wc_v4/availability.py` | Lineup confidence from absence-note certainty, replacement value, GK-specific impact, attack/defence/set-piece splits, rotation likelihood, and — the headline — an **uncertainty band** on the availability adjustment that widens when the lineup read is shaky. |
| **M4** | `wc_v4/matchup.py` | Report-only tactical matchup diagnostics: strength-only baseline vs Dixon-Coles attack/defence view, with deltas for 1X2, totals, BTTS, and reason codes. |
| **M5/M6** | `wc_v4/probability.py` | Coherent score-distribution board. 1X2, totals, BTTS, correct-score, fair odds, and confidence ranges all derive from one score matrix. |
| **M6** | `wc_v4/consistency.py` | Cross-market consistency checks and stale/generous market diagnostics so recommendations cannot come from contradictory prices. |
| **M7** | `wc_v4/staking.py` | Uncertainty-aware recommendation wrapper: fair odds, confidence range, availability uncertainty, market-movement reason codes, CLV context, stake haircuts, and pass decisions. |
| — | `wc_v4/tournaments.py` | Generalized, leak-free per-match sample builder for past World Cups. Registry `TOURNAMENTS` (WC2018, WC2022); drop a `data/wc{YEAR}_odds.csv` to fold a new cup into the gate. |
| — | `wc_v4/validate_v4.py` | Held-out harness: the WC2022 market-blend gate (model/market/V3-blend/V4-segment) **plus** pooled model calibration across WC2018+WC2022. Writes `data/wc_v4_validation.json`. |
| — | `test_wc_v4.py` | 19 tests incl. leakage-injection, walk-forward, WC2018 wiring/calibration checks, coherent-board checks, consistency checks, and M7 pass-staking. |

## How acceptance criteria are met

**M1 (point-in-time feature store).**
- Every row has `asof, event_id, source, fetched_at, schema_version` — asserted in
  `test_training_matrix_has_provenance_on_every_row`.
- Walk-forward rebuild reads no future rows: pre-match Elo columns, a goal model
  refit only on matches before each month, and rest/congestion computed only from
  a team's earlier matches. `build_asof(d)` prices only fixtures with
  `match_date ≥ d` and exposes **no** closing/result column.
- Leakage tests inject future-only columns (`result`, `odds_close_*`, `clv`) and
  confirm they are rejected (`assert_no_leakage`) or excluded (`feature_columns`).

**M2 (closing-line teacher & market movement).**
- Open/current/close line history with movement, steam, and reversal features.
- The closing line is a **teacher, not a feature** (guardrail #3): it lives in
  `OUTCOME_COLUMNS`; `test_closing_line_is_a_teacher_not_a_feature` enforces it.
- Segmented blend weights are decided **out-of-sample** (leave-one-out), not by
  in-sample fit. On the 64-match WC2022 sample, segmentation does **not** beat the
  global weight, so the V3 default (`w = 0.163`) stands. This is the correct,
  guardrail-#1/#6 outcome — thin data stays report-only.
- `do_not_bet()` returns reason codes for the "edge is just market info" case.

**M3 (availability & replacement value).**
- Replacement value = squad-power drop (V3's best-18 power already encodes bench
  drop-off); GK absences flagged with their heavier defensive share.
- Lineup confidence erodes with *doubtful* absences (possible-return, fitness
  tests), not with clean ruled-out players.
- Uncertainty is surfaced: the adjustment is returned as `[elo_adj_low, elo_adj,
  elo_adj_high]`, widening when confidence is low or a keeper is involved — the
  M3 acceptance criterion. All M3 output is `status: report_only`.

**M4 (matchup-specific modelling).**
- Tactical diagnostics compare a simpler strength-only board to a Dixon-Coles
  attack/defence board, then emit market deltas and reason codes.
- Held-out evidence is exposed through `heldout_matchup_eval()` and remains
  `report_only`; it does not change V3/V4 defaults.
- Totals and BTTS deltas are evaluated separately from 1X2 deltas.

**M5/M6 (probabilistic backbone and cross-market consistency).**
- `coherent_board()` produces 1X2, totals, BTTS, correct-score, fair odds, and
  confidence intervals from one score matrix.
- `check_board()` verifies complements and probability sums; `market_diagnostics()`
  identifies stale/generous bookmaker legs without allowing incoherent prices to
  pass silently.

**M7 (uncertainty-aware staking).**
- `staking.recommendation()` includes fair odds, probability confidence range,
  availability uncertainty, CLV context, market-movement reason codes, haircuts,
  and final stake.
- If uncertainty overwhelms edge, the recommendation is `pass` and stake is zero.

## Held-out result (data/wc_v4_validation.json)

**Market-blend gate (WC2022, result90)** — the audited ship/no-ship decision:

```
price-maker    n   logloss    brier     acc
model         64    1.0278   0.6025   0.5469
market        64    1.0003   0.5837   0.5312
v3_blend      64    0.9994   0.5836   0.5312
v4_segment    64    0.9994   0.5836   0.5312
```

V4 segmentation ties V3 (it fails the gate, so it fail-safes to the global
weight). **Verdict: V3 stays the default.** That is the intended behaviour — V4's
job here is to build the leak-free substrate, the market/availability *signals*,
and the honest gate that decides when a richer model is allowed to ship.

**Model calibration, pooled held-out (WC2018 + WC2022)** — the larger evidence
base the single-tournament gate was missing:

```
tournament     n   logloss    brier     acc   odds
WC2018        64    0.9759   0.5812   0.5781    no
WC2022        64    1.0315   0.6052   0.5469   yes
POOLED       128    1.0037   0.5932   0.5625
  group-only  96    1.0079   0.5923   0.5729
```

WC2018 is wired in via `wc_v4/tournaments.py`: its 64 matches are scored by a
model trained strictly before the 2018-06-14 kickoff (leak-free), doubling the
held-out base to 128. It has **no** historical 1X2 odds file, so it strengthens
*model calibration* but not the *market-blend gate* — the harness is explicit
about that in its `coverage` block. Dropping a `data/wc2018_odds.csv` (same schema
as `data/wc2022_odds.csv`) folds WC2018 into the gate automatically. 1X2 outcomes
use the final score; a `group_only` cut (no extra-time knockouts) is reported
alongside for a clean 90-minute read.

## How to run

```
python3 -m wc_v4.feature_store --since 2022-01-01      # training matrix
python3 -m wc_v4.feature_store --asof 2026-06-17       # live feature rows
python3 -m wc_v4.market_model                           # segment weights + CLV
python3 -m wc_v4.availability                           # per-team availability
python3 -m wc_v4.matchup Brazil Morocco --asof 2026-06-11
python3 -m wc_v4.probability Brazil Morocco --asof 2026-06-11
python3 -m wc_v4.tournaments                            # past-cup sample coverage
python3 -m wc_v4.validate_v4                            # gate + pooled calibration
python3 test_wc_v4.py                                   # tests (19)
```

## Suggested next steps (not built this session)

- **More held-out tournaments.** WC2018 is now wired in for model calibration
  (n=128 pooled). The remaining gap is *odds*: source historical 1X2 closing odds
  for WC2018 (and qualifiers) into `data/wc2018_odds.csv` so the market-blend and
  segmentation gates get a fair, multi-tournament test rather than WC2022 alone.
- **M3 calibration.** Calibrate the availability adjustment/uncertainty against
  historical absence examples rather than the current heuristic SD, then re-run
  the gate before considering any default change.
- **Default promotion gates.** M4-M7 are implemented as report-only layers. The
  next step is extending `validate_v4.py` to decide which, if any, clear the
  default gate against V3 on held-out accuracy, calibration, CLV, and drawdown.
- **UI surface.** When a signal clears the gate, expose the explainability
  (reason codes, uncertainty band) on the existing dashboard/history screens per
  the plan's UI note — no new UI until V6.
