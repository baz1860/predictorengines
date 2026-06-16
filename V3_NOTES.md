# V3 Implementation Notes

Dated log of the V3 build (see `V3_PLAN.md`). One entry per milestone: files
changed, what was verified, and any rejected approaches. Scope for this pass is
the M1–M4 risk-reduction core (the plan's recommended time-box).

---

## 2026-06-15 · M1 — Common engine contract + contract tests

**Goal:** make all four engines speak one stable app contract before any
modelling change.

**Files changed**

- `app/engines/contracts.py` *(new)* — dependency-free contract helpers:
  - `assert_finite_json()` / `is_finite_number()` — reject NaN/Inf and any
    non-JSON value anywhere in a payload.
  - `validate_prediction()` — accepts outcomes-style (1X2/win-loss) *or*
    table-style (golf head-to-head) predictions.
  - `validate_table()` — `{columns, rows}` structure check.
  - `market_id()` — stable market identifiers (also normalizes golf's
    `side`-carried markets like `matchup:a|b`).
  - `fixture_key()` — deterministic `date|home|away[|competition]` event id.
  - `normalize_edge_row()` / `normalize_edge_result()` — **additive**
    normalization to the canonical edge shape, preserving each engine's existing
    UI keys so `columns` keep rendering unchanged. Sanitizes stray non-finite
    floats to `None` on the live path.
  - Canonical fields: `event_id, match_date, home, away, market, side, line,
    bet, odds, p_model, p_market, p_book, edge, ev_per_unit, kelly_frac,
    stake_gbp, source, model, recommended`.
- `app/engines/worldcup.py`, `cfb.py`, `club_soccer.py`, `golf.py` — each
  `edge()` now returns `normalize_edge_result(...)`. Record/pick logic is
  untouched (normalization is additive and runs last).
- `app/engines/runners/cfb_runner.py` — **fixed model-selection drift**:
  `cmd_edge` now reads `model` (validated against `blend|elo|power`) and passes
  it to `blend_predict(...)`. Previously the requested model was ignored and edge
  always used the blend.
- `test_engines_contract.py` *(new)* — exercises `info`, `predict`, `edge`
  (manual), and `simulate` (WC/golf, tiny seeded) for every registered adapter.
  Missing data/odds = SKIP; contract violations = FAIL.

**Verified**

- `python3 test_engines_contract.py` → 12 pass · 2 skip · 0 fail (exit 0). Skips
  are WC and golf edge, whose local odds files are unfilled templates.
- CFB edge default (`blend`) output unchanged; `power` now differs
  (e.g. ML home 0.822 → 0.787) — matches the acceptance criterion exactly.
- Canonical fields present on every emitted edge row; original UI keys
  (`date`, `match`, …) preserved.
- No engine emits NaN/Inf (asserted by `assert_finite_json`).
- Regression suite green: `test_club_soccer`, `test_m2`, `test_m3`, `test_m4`,
  `test_m5`, `test_m6`, `test_m7` all pass.

**Rejected / deferred**

- Did *not* rewrite the subprocess runners to emit canonical rows directly —
  normalizing in the adapter keeps the runner JSON minimal and avoids editing
  four flat-module codebases. The adapter is the contract boundary.
- Golf `event_id` is currently just the player slug; proper tournament-scoped
  event identity is M4 (where golf settlement is also made event-safe).

---

## 2026-06-15 · M2 — Security & reliability hardening

**Goal:** shrink attack surface without changing any model output.

**Files changed**

- `app/security.py` *(new)*:
  - `redact()` — scrubs known stored/env key values, then masks any remaining
    20+ char key-looking token. Belt-and-braces for unknown keys too.
  - `collect_secrets()` — gathers concrete secrets (api_keys.json values + key
    env vars) to redact against.
  - `safe_runner_env()` — curated env allowlist (system essentials + the API-key
    env vars the engines read) instead of leaking the whole parent environment.
  - `safe_get()` — requests wrapper with timeout, provider label, redacted error.
- `app/engines/_subprocess.py` — hardened `run_engine()`:
  - rejects any command outside `ALLOWED_COMMANDS` *before* spawning;
  - checks the runner file exists;
  - uses `safe_runner_env()`;
  - strict finite-JSON validation of the result (`assert_finite_json`) — NaN/Inf
    or noisy stdout is a hard `RuntimeError`;
  - redacts stderr snippets and engine `error` strings.
- `app/server.py` — bounded Pydantic request models:
  - `EngineRequest.engine` must match `^[a-z0-9_-]{1,40}$` (path traversal /
    injection rejected at the boundary, 422 — never reaches the filesystem);
  - `params` serialized size capped at 50 KB;
  - numeric params (`sims, seed, kelly, min_edge, cut_rule`) clamped to ranges;
  - `BankrollAction.amount` and `SettingsPatch.default_kelly` range-bounded.
- `api_keys.py` — `save_keys()` now `chmod 0o600` (best-effort; no-op on
  Windows/unsupported FS).
- `preflight.py` *(new)* — offline report of per-engine data-file presence/age
  and which API keys are set (masked). Never makes a network call; exit 0 always
  so missing data can't block offline use.
- `test_security.py` *(new)* — 22 checks across all of the above.

**Verified**

- `python3 test_security.py` → 22 passed, 0 failed. Confirms: settings expose
  only masked keys; a synthetic runner error carrying a fake key is redacted in
  the raised error; bad/oversized requests rejected; sims/kelly clamped;
  unrelated env dropped while key env vars survive; key file is 0600;
  non-finite runner JSON rejected.
- No behavior change on default runs: contract test still 12 pass · 2 skip · 0
  fail; `test_club_soccer` (exercises the subprocess path) still green.

**Rejected / deferred**

- Did *not* wire every engine's existing network fetch through `safe_get()` —
  that touches four flat-module codebases and isn't needed for the default
  offline/manual runs the acceptance covers. The helper is in place; routing the
  live fetches through it is a follow-up (and a natural fit alongside M5 CLV).
- Kept the `safe_runner_env` allowlist generous (system + venv/conda vars) on
  purpose: a too-tight env risks breaking Python startup on some machines, which
  would be a worse failure than the leakage we're closing.

---

## 2026-06-15 · M3 — Validation gates across all engines

**Goal:** make "do not degrade modelling" enforceable for every engine, not just
World Cup.

**Files changed**

- `cfb/validate.py` *(new)* — walk-forward CFB validation, consolidating the
  useful parts of `predictor.py --backtest`, `blend_eval.py`, `ats_backtest.py`,
  `totals_backtest.py`:
  - Elo updated game-by-game; spread slope fitted only on seasons **before**
    `--since`; power refit per week with `asof = first kickoff` → no future
    leakage.
  - Metrics: `ml_brier`, `ml_acc`, `margin_mae`, `total_mae`, plus ATS ROI and
    totals ROI per disagreement threshold (vs `closing_spreads.csv` /
    `closing_totals.csv`).
  - Gate fails (exit 1) if Brier (+0.005) or margin/total MAE (+0.5pt) regress.
    ROI is recorded but **not** gated (too noisy). `--update-baseline` is the
    only way to loosen the baseline.
- `cfb/data/validation_baseline.json` *(new)* — stored baseline, window
  2023–2025, 2394 games: ml_brier 0.1885, margin_mae 12.79, total_mae 13.05.
- `validate_all.py` *(new)* — runs all four gates, each in its own
  cwd + PYTHONPATH (engine-folder isolation), writes `data/validation_suite.json`,
  prints a compact per-engine table, exits non-zero if any engine regresses or
  errors. Golf sims default to a small `--sims 5000` for a fast gate.

**Verified**

- `python3 validate_all.py --gate --sims 4000` → all four PASS (worldcup 1.0s,
  club_soccer 9.3s, cfb 3.6s, golf 45.3s; ~59s total). Summary JSON written.
- Negative test: tightening the CFB baseline `ml_brier` to 0.10 makes the CFB
  gate print FAIL and exit 1; restoring the baseline returns exit 0. Confirms the
  regression gate is real and the baseline isn't silently loosened.
- CFB walk-forward confirmed leakage-free (slope fit on `<since`, power refit
  per week with `asof`).

**Rejected / deferred**

- Did not gate on ATS/totals ROI — single-season betting ROI is too high-variance
  to use as a pass/fail signal; it's stored for visibility and trend-watching.
- Defaulted the CFB window to 2023–2025 (2394 games) over 2025-only (807): a
  larger sample makes the Brier/MAE baseline more stable while still being
  leakage-free and fast (~4s).

---

## 2026-06-15 · M4 — Shared bankroll, portfolio & settlement

**Goal:** one safer staking/settlement path for all engines.

**Files changed**

- `app/portfolio.py` *(new)* — suite-level risk caps (pure risk controls, only
  ever reduce a stake): `drawdown_factor()` brake + `apply_caps()` enforcing
  single-event (15%), correlated-per-engine (25%) and daily (40%) exposure
  against the pooled bankroll, accounting for already-open exposure.
- `app/bankroll_store.py`:
  - ledger extended backward-compatibly with `event_id, market, line, source,
    model, closing_odds` (`_CORE_COLS` + `_V3_COLS`); `load_ledger()` backfills
    missing columns as `""`, so old ledgers load unchanged.
  - `place_bets()` rewritten: dedupes on `(engine, event_id, side)` (legacy
    `home/away/side` still honored), runs `apply_caps()` before the pooled-funds
    clamp, and writes the new provenance columns. Accepts an optional `peak`.
  - `settle()` gains `dry_run=True`: computes every verdict and the resulting
    bankroll but writes nothing, returning a per-bet `preview`.
- `app/engines/golf.py` — **event-safe settlement**. Replaced "grade everything
  against the single latest event" with `_completed_events()` (all events, sorted
  by end date) + a per-bet rule: grade only against the earliest completed event
  **on/after the bet's reference date** whose field actually contains the
  participant(s). A stale/future outright stays open. Golf records now stamp the
  placement date so `event_id = date|player` is week-distinct and settlement has
  a date floor.
- `app/engines/{worldcup,cfb,club_soccer}.py` — each `edge()` now flags
  `recommended` on the rows recording would place (the engine's own selection),
  and recording places exactly those rows + writes `source`/`model` — no separate
  ad-hoc filter at record time. Golf already recorded by `recommended`.
- `app/server.py` — new bankroll action `settle_preview` → `settle(dry_run=True)`.
- `test_bankroll.py` *(new)* — 17 checks on temp files.

**Verified**

- `python3 test_bankroll.py` → 17 passed: legacy ledger loads + V3 cols
  backfilled; duplicate placement deduped; single-event exposure capped to ~15%;
  in-event golf bet settles while a stale future outright stays open; dry-run
  previews settlement and leaves ledger/bankroll untouched, then the real settle
  commits.
- Full regression: all 10 suites green (`test_engines_contract`, `test_security`,
  `test_bankroll`, `test_club_soccer`, `test_m2`–`test_m7`).
- End-to-end smoke: `app.server` imports with the bounded models; a real CFB
  `edge()` flags 3/6 rows `recommended` with `event_id` + `stake_gbp`; the
  dashboard still builds against the extended ledger.

**Rejected / deferred**

- Golf `event_id` is a `date|player` key, not the ESPN numeric tournament id:
  the upcoming event isn't in `rounds.csv` at bet time and `field.csv` carries no
  id, so a numeric id can't be captured reliably. Settlement is made event-safe
  by **chronology + field membership** instead, which needs no fragile id mapping
  and directly satisfies the staleness acceptance.
- Left each engine's internal portfolio shaping (WC `edge.portfolio_size`,
  `golf/portfolio.py`) in place; `app/portfolio.py` is the shared backstop at the
  pooled-bankroll boundary, not a rip-and-replace — that keeps the validated
  per-engine staking intact while still enforcing suite-wide caps.

---

## 2026-06-16 · M5 — Market blend + CLV for every priced engine

**Goal:** make betting evaluation less fake-edge prone across the suite, and
track closing line value (CLV) — the most reliable +EV signal — for every engine.

**Files changed**

- `app/market_blend.py` *(new)* — shared, dependency-light (pure-Python `math`)
  generalisation of the World Cup 1X2 logit blend:
  - `blend_probs()` / `blend_two()` — logit-space anchor of model probabilities
    toward the de-vigged market, renormalised; `w` = weight on the model.
  - `anchor_line()` — linear convex blend for point lines (spread/total).
  - `devig()`, `weight_for()` (reads `data/market_blend_suite.json`),
    `is_default_on()`, and `apply_blend_to_rows()` — the adapter-level row
    applier that re-anchors `p_model`, then recomputes edge / EV / Kelly / stake
    in place.
- `app/engines/club_soccer.py`, `cfb.py` — each `edge()` accepts an
  **experimental, default-OFF** `market_blend` flag and exposes it in
  `edge_schema().options`. When set, `apply_blend_to_rows()` anchors the rows
  before `_mark_recommended`, so recommendations reflect the blend. Default runs
  are byte-for-byte unchanged.
- `clv_suite.py` *(new)* — suite-level CLV over the shared `data/suite_ledger.csv`:
  - `--snapshot [--engine X]` records current odds for open bets, matched per
    engine to its odds file (WC `odds.csv`, club `club_soccer/data/odds.csv`,
    CFB `cfb/odds.csv`, golf `golf/data/odds.csv`) → `data/clv_history.csv`,
    keyed by `(engine, event_id, market, side)`.
  - `--report [--write-closing]` computes the closing-odds proxy (latest snapshot
    at/before kick-off) per settled bet, prints per-bet CLV%, rolling mean CLV,
    positive-CLV rate, and a per-engine breakdown; `--write-closing` backfills
    `ledger.closing_odds` (backup first). Empty history → clear no-data message,
    never a crash.
- `test_market_blend.py` *(new)* — 20 checks; `test_clv_suite.py` *(new)* — 10.

**Verified**

- `python3 test_market_blend.py` → 20 pass; `python3 test_clv_suite.py` → 10 pass
  (no-data report is crash-free; snapshot matches the open bet from the odds
  file; CLV% = bet/closing − 1 and excludes a mismatched side; `--write-closing`
  backfills with a backup).
- End-to-end: CFB `edge(market_blend=True)` shrinks every edge toward the market
  (e.g. SPREAD 0.194 → 0.101, ML 0.077 → 0.041 at w=0.50) while the default
  `edge()` is unchanged. Full regression green (all 12 fast suites).

**Rejected / deferred (M5 guardrail)**

- **No default flipped.** Per the M5 acceptance, a generalised blend becomes an
  engine's *default* only once a held-out metric beats *both* pure model and pure
  market for that engine. Club Soccer and CFB ship the blend behind an
  experimental flag with a conservative placeholder `w` (not a fitted value);
  `DEFAULT_BLEND_ON` is empty. Fitting + validating per-engine `w` (and flipping
  defaults) is M6 work, where the validation harness and data live. World Cup
  keeps its own validated 1X2 blend in `edge.py`; golf keeps its runner blend.
- **Probability-space anchoring for CFB**, not line-space. The plan suggested
  blending the model *margin/total* toward the market line. Anchoring the
  cover/over *probability* toward the de-vigged book is equivalent in pulling the
  model toward the market, uniform with the other engines, and avoids re-plumbing
  the spread/total math in the flat-module runner. The adapter is the contract
  boundary (consistent with M1/M2).
- **Offline-first CLV snapshots.** Snapshots read each engine's manual odds file
  (the same file the app Edge "API" source refreshes) rather than calling
  providers directly, so CLV never makes a network call or crashes offline. Live
  provider fetches remain the existing per-engine fetchers feeding those files.

---

## 2026-06-16 · M6 — Engine-specific modelling upgrades (gated)

**Goal:** improve each engine *only* where held-out validation says it helps;
otherwise document the measurement and leave the default unchanged. This pass was
run conservatively: measure first, flip nothing that doesn't clearly and
consistently win, keep new capability default-OFF.

**What shipped (CFB tunable blend weight)**

- `cfb/predictor.py` — `blend_predict()` gained a `w_elo` blend weight (weight on
  Elo for win-prob + margin; power always supplies the total) and a
  `load_blend_weight()` that reads `cfb/data/blend_weight.json` *if present*,
  else **0.50 — the V2 50/50 blend**. No weight file is shipped, so default
  predict/edge output is byte-for-byte unchanged.
- `cfb/validate.py` — `walk_forward()` honours the stored weight; new
  `choose_weight()` (pure, unit-tested) + `--tune-blend [--write]` print a
  before/after table and can opt into the validated weight.
- `test_cfb_blend.py` *(new, 11 checks)*.

**Before/after (walk-forward 2023–2025, 2394 FBS-vs-FBS games, leak-free):**

| w_elo | ml_brier | margin_mae |
|------:|---------:|-----------:|
| 0.50 (default) | 0.18849 | 12.786 |
| **0.60 (chosen)** | **0.18787** | **12.784** |
| 0.70 (Brier min) | 0.18765 | 12.818 |

Brier improves by −0.00062 and is LOSO-stable (every held-out season prefers
w>0.5: Brier optima 0.65/0.75/0.75). `choose_weight` picks the Brier-minimising
weight **subject to margin MAE not regressing** vs the 0.5 blend — so it never
trades away accuracy on ATS (the market CFB actually bets). The improvement is
real but modest, so per the M6 "default-OFF" guardrail it ships as an explicit
opt-in (`--tune-blend --write`, then `--gate --update-baseline`), not a forced
flip.

**Measured and deliberately NOT changed (documented per acceptance)**

- **CFB moneyline calibration.** LOSO Platt scaling moved Brier 0.18849 →
  0.18830 (−0.0002; global fit a≈1.17, b≈−0.03 ≈ identity). The blend is already
  well-calibrated → no calibration layer added.
- **CFB ATS/totals ROI.** Recorded by the gate, still not gated (single-season
  betting ROI is too high-variance) — unchanged from M3.
- **Club Soccer ensemble weights.** `ENSEMBLE_W = {goals .20, elo .25, xg .30,
  xgf .25}` is *already* "chosen by held-out walk-forward search" (model.py).
  Re-tuning blindly risks degrading a validated default for no measured gain;
  left as-is. Market blend + CLV (the M6 prerequisites for more aggressive
  staking) landed in M5 behind an experimental flag — staking aggressiveness is
  unchanged.
- **Golf.** The two structural items are already implemented: cut/no-cut markets
  are suppressed when the cut doesn't bind (`__cut_binds__` / `cut_binds` in
  `golf/edge.py`, `golf/simulate.py`), and market blend + CLV exist in
  `golf/market.py` (`blend`, `closing_fair`, `clv_pct`). **Deferred:**
  simulation vectorization — it's a performance-only change with real risk of
  perturbing seeded outputs and no accuracy benefit; the gate already passes in
  ~45s, so it belongs in a dedicated perf pass, not a conservative modelling one.
- **World Cup.** Richer (totals/BTTS or stage-specific knockout) calibration
  needs enough *post-V2* data; the 2026 tournament only kicked off 2026-06-11, so
  there are too few completed matches to fit a non-overfit calibration. Deferred
  until a meaningful 2026 sample exists; squad/context adjustments stay opt-in
  (unchanged).

**Verified**

- `python3 test_cfb_blend.py` → 11 pass. Full regression green (all 13 fast
  suites). Default CFB gate identical to the M3 baseline (ml_brier 0.1885,
  margin_mae 12.786, total_mae 13.05) with no weight file present.

---

## 2026-06-16 · M7 — Data provenance & refresh hygiene

**Goal:** know what data produced a prediction, warn when it's stale, and catch
manual-odds mistakes with actionable errors — all **offline** (no network, never
blocks operation).

**Files changed**

- `app/provenance.py` *(new)* — dependency-light (csv + json, no pandas):
  - `ENGINE_INPUTS` registry of each engine's key inputs (path, role, source).
  - `build_manifest()` / `write_manifest()` / `read_manifest()` →
    `data_manifest.json` per engine (path, `fetched_at` = file mtime, row count,
    `schema_version`, source label).
  - `freshness()` / `freshness_warnings()` — staleness from file mtimes only
    (results/games 3d, fixtures 3d, rounds 3d, field 2d, odds 1d, model 30d);
    `freshness_warnings` never raises so it can't break a schema call.
  - `validate_odds_file()` — long-format (club/CFB) and wide-format (WC/golf)
    manual-odds schema checks returning **row / column / value / expected** per
    error (bad market, bad side-for-market, non-numeric odds ≤ 1.0, missing
    spread/total line, bad neutral flag, missing header column).
  - CLI: `python3 -m app.provenance --write [--engine X] | --freshness |
    --check-odds <engine>`.
- `app/engines/{worldcup,club_soccer,cfb,golf}.py` — `predict_schema()` now
  includes a `freshness` list (UI staleness badges); CFB and Club Soccer `edge()`
  attach `odds_issues` for a malformed manual odds file (non-fatal — the edge
  still runs on the usable rows).
- `update.sh`, `club_soccer/update.sh`, `golf/update.sh` — write the manifest
  after a refresh (offline-safe, `|| true`; engine scripts run it from repo
  root). CFB has no update script; its manifest is written via the CLI.
- `.gitignore` — ignore generated `data_manifest.json` (local mtimes) and
  `data/clv_history.csv`.
- `test_provenance.py` *(new, 19 checks)*.

**Verified**

- `python3 test_provenance.py` → 19 pass (fresh/stale/missing detection on a temp
  tree with no network; manifest row counts + source + schema_version; CFB and
  WC odds errors carry row/column/expected; valid file → no errors;
  `freshness_warnings` safe on an unknown engine).
- CLI smoke: `--freshness` flags stale odds/games/field across all four engines;
  `--check-odds cfb` passes on the real file. Adapter schemas now carry
  `freshness` (e.g. CFB `['games: 4.5d old (> 3d)', 'odds: 1.9d old (> 1d)']`).
- Full regression green (all 14 fast suites).

**Rejected / deferred**

- Freshness is computed from **live file mtimes**, not the stored manifest, so a
  fresh checkout (or a machine that hasn't run the update scripts) still reports
  correctly. The manifest is the richer record (source/rows), written on refresh.
- Manifests are **gitignored**, not committed: they encode local mtimes and would
  otherwise churn on every refresh.

---

## Scope note

The M1–M4 risk-reduction core, M5 (market blend + CLV), M6 (gated modelling
upgrades) and M7 (data provenance) are delivered. M8–M9 (power-user UX, release
docs) remain open. Leverage available for them: `app/engines/contracts.py`,
`app/security.py` (`safe_get`), `app/portfolio.py`, `app/market_blend.py`,
`clv_suite.py`, `validate_all.py`, and `preflight.py`.
