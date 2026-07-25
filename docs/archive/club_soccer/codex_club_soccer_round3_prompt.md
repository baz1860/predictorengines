# Codex Prompt: Round-3 Review — Remediation Verification + Remaining Deep Issues

Copy everything below the line into Codex.

---

This is round 3 of an adversarial review cycle on `club_soccer/`. Round 1 found the issues; round 2 verified four mechanical fixes and found them incomplete; a larger remediation pass has now landed. Your job: (1) adversarially verify the remediation — assume it is wrong until proven otherwise; (2) re-assess the module against every still-open finding; (3) go deeper on the model-quality items that no fix has touched yet.

Be harsh. No praise, no filler. Every finding: `file:line`, offending code, impact, concrete fix. Do not modify files.

## Part 1 — Adversarially verify the remediation

### A. Evidence gate (`evidence_gate.py`, new file; wired into `edge.rows_from_odds`)
Staking is now OFF by default: preregistered criteria (artifact ≤14 days old, ≥1000 bets, positive flat+Kelly ROI at every threshold, positive CLV with >50% positive fraction, model log-loss ≤ market for 1X2) are checked against `data/backtest_market.json`; ALL markets with evidence must pass or every stake in `rows_from_odds` output is zeroed with `suppressed_reason="evidence-gate: ..."`. Fails closed on any error. The card and the app adapter (`_mark_recommended` now requires nonzero stake and no suppression) both inherit it.
Attack it:
- Can any pricing path still produce a nonzero stake or recommended/recorded bet while the gate is closed? Check `late_lineup_card`, `fit_market_blend`, `market_model`, the CLI entry points in `edge.py`'s `__main__`, `snapshot_edges.py`/`daily_card.py` at repo root, and anything else that consumes `edge_report.csv` or calls `kelly()` directly.
- The gate reads the backtest artifact the module itself generates — from a backtest you established (round 1, finding 5) selects and executes at closing odds. The gate can therefore open on evidence from a flawed simulator. How should gate criteria be hardened so they can only be satisfied by the redesigned decision-time backtest (e.g., require a `backtest_version`/methodology field)?
- Criteria gaps: no confidence intervals (ROI > 0 on 1000 bets can be noise), no per-league breakdown, no requirement that CLV is computed against a sharp book. Propose concrete tightened thresholds.
- Is `MAX_ARTIFACT_AGE_DAYS=14` compatible with how often `backtest_market.py` actually runs (Mondays via the card footer)?

### B. Context-leakage fix (`model.predict_match` gained `context_coef`; `validate.py` passes `{}`)
Walk-forward validation now predicts with `context_coef={}` so the full-history production artifact can't leak into historical predictions. Verified locally: a 2024 prediction differs between production (0.6333 home) and context-free (0.5994) — the leak was real and material.
Attack it:
- Validation now measures the CONTEXT-FREE model while production applies active coefficients (`rest_diff`, `euro_hangover`, `tier_gap`). So validation and production have diverged: the gate metrics no longer describe the deployed predictor. Is that acceptable as an interim state? Design the per-window context refit (which function in `context.py` fits coefficients, what it costs per fold, where the window boundary must sit).
- Are there OTHER production artifacts still read during validation? Audit every `load_*`/file-read reachable from `validate.walk_forward` and the calibration/ensemble sections: calibration maps, ensemble weights, league adjustments, market-blend weights, player feature store, `identities`/name maps. The same leak pattern may repeat.
- `test_club_soccer.py` — does any test now pin the validation path to `context_coef={}` so a regression (someone removing the argument) would be caught?

### C. Per-quote timestamps (`quoted_at_utc` stamped in `fetch_bsd_odds` and `fetch_the_odds_api`; `validate_quotes` enforces 6h live / 2d manual, dropping un-timestamped rows in timestamped frames)
Attack it:
- The timestamp is stamped at FETCH time, not the bookmaker's quote time — a bookmaker's stale feed gets a fresh stamp. Is anything better available from BSD/The Odds API responses (event `last_update` fields)?
- Kickoff granularity is still date-only: a 12:00 UTC kickoff prices at 23:00 UTC same-day. Where do kickoff timestamps exist in the BSD event payload and fixtures schema, and what's the migration path?
- Mixed frames: `snapshot_odds.py` writes odds history — does it stamp quotes consistently with the new column? Does anything downstream break on the new column's presence?

### D. Line-shopping EV bias fix (edge/EV now vs the executing book's own de-vigged probability; cross-book mean kept as `p_consensus`)
Attack it:
- A soft book with high vig and a bad price now sets its own baseline — a systematically mispriced soft book can still show "edge" against itself. Is a sharp-anchor requirement (edge must also be positive vs the sharpest available book / Pinnacle when present) needed before the gate ever opens?
- `do_not_bet` in `market_model.py` consumes `p_book` — verify its logic is still calibrated for the new definition.
- Does anything consume the removed cross-book-mean `p_book` semantics (settlement, reports, `fit_market_blend`)?

### E. Fail-closed pipeline (`_step(required=True)` on fetch/refit/pricing → exit 3 after writing degraded card; `update.sh` propagates nonzero from season and the validation gate)
Attack it:
- Is the required/optional classification right? (Absences, squads, snapshots, fd.co.uk refresh are still optional — argue for/against each.)
- `launchd`/scheduler behavior on exit 3 — does anything actually surface it to the operator? Propose a minimal notification hook.
- `set -euo pipefail` interaction with the new `|| { }` blocks in `update.sh` — any path where a failure still yields exit 0?

## Part 2 — Still-open findings (verify status, no fix has been attempted)

1. **Team identity fragmentation** — 42 collision groups, nondeterministic `names.py` resolution, `identities.py` false-clean health. Design the provider-ID registry + migration in enough detail to implement (data files, ingestion hook points, backfill procedure for `fixtures.csv`).
2. **Closing-odds backtest** — still selects on `p_model − p_close` and executes at close. Specify the redesigned decision-time backtest: which stored odds snapshots exist (`snapshot_odds.py` history), what decision timestamp to use, CLV vs close afterward.
3. **Postponed/void fixtures retaining scores** — `fetch.py` merge + `model.played()` by score presence.
4. **Cross-league strength** — symmetric comp adjustment, tier-blind Elo init.
5. **Fuzzy player/team matching** — one-token guessing, home-side default.
6. **Future congestion ignoring scheduled fixtures**; **correlation reset after adjustments** (`DC_RHO` rebuild); **ensemble promotion without untouched holdout**; **±0.01 Brier gate**; **overdispersion unmodelled**; **provider-ID namespacing**; **seeder overwrite risk**; **openfootball season derivation**; **corners `.notna().all()`**; **no portfolio exposure caps**; **false-green `check()` in the other repo-root test files** (club_soccer's is fixed; the pattern remains in test_market_blend, test_bankroll, test_m2–m7, golf/NFL/NHL/etc.).

## Part 3 — Deeper model-quality assessment (nothing has touched these)

- **Draw calibration**: check H/D/A reliability separately on the stored `validation_predictions.csv`; draws are the classic weak leg. Quantify.
- **Time-decay half-life (365d), `HFA_SHRINK_K=300`, `RHO_SHRINK_K=400`, `RECENT_K=6`, Elo K-factor**: which were ever tuned out-of-sample? Design the experiment that would tune them without contaminating the holdout.
- **Player-availability layer ROI**: ~900 lines of `player_features.py` feeding multiplicative adjustments — is there any stored evidence the adjustments improve Brier/log-loss vs. ignoring them? If not, propose the ablation test.
- **Ensemble weights**: how different are goals/elo optimal weights across leagues and seasons? Is a single global weight defensible?

## Output format

Fix verdicts first (VERIFIED / INCOMPLETE / WRONG / REGRESSION per remediation item A–E, with reproductions), then findings by severity with `file:line`, then an updated operator-delusion list, then a max-10 prioritized remediation plan distinguishing "implement now" from "needs design decision by the owner".

Run before concluding: `python3 -m pytest test_club_soccer.py -q`, `python3 test_club_soccer.py`, `python3 -m compileall club_soccer`, `python3 -m club_soccer.evidence_gate` (expect exit 1 — staking blocked), `python3 -m club_soccer.season --no-network --fast` (expect exit 0, card written, no backed bets), and your own reproductions. Report every command's outcome. Do not modify files.
