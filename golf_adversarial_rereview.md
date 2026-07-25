# Adversarial Re-review — `golf/` (post-fix)

Fixes reviewed at commit `644e848` ("Refactor soccer prediction pipeline and update app UI", 2026-07-19), which contains all golf changes, plus the untracked root `conftest.py`. All reproductions were run from temporary scripts outside the repository; no production code, tracked data, or tests were modified. `git diff --check` clean; no golf files carry uncommitted changes.

---

## 1. Verdict

**FAIL — demonstrated Critical/High defects remain.**

The headline fixes are real — event identity, point-in-time validation, the blocking gate, dead-heat economics, and the paste parser all now hold up under attack — but the same commit introduced a demonstrated Critical regression: with ESPN's actual payloads (competitor status is NULL for every one of ~5,900 competitors in the cached season files), the rewritten history ingester labels **every missed-cut player as having made the cut**, and the rewritten accumulator will retroactively rewrite two seasons of `rounds.csv` with those labels on the next `update.sh` run, silently *improving* the gate metric while destroying the cut market's ground truth. A second demonstrated High defect lets a mistyped odds value (`0.5` for `5.0`) turn a 3-ball into a silently repriced 2-ball through the CSV round-trip.

## 2. Findings

### Critical

**N1 — New `made_cut` rule labels all missed cuts as made with real ESPN data; next accumulate rewrites history and self-inflates the gate.**
- Location: `golf/providers/legacy.py:286` (`made_cut = 0 if out_status and len(rounds) < 3 else 1`), interacting with `golf/providers/legacy.py:430-441` (N2) and `golf/validate.py` (`event_no_cut = made_by_player.eq(1).all()` in `walk_forward`).
- Trigger/state: ESPN's season scoreboard payloads carry **no competitor status**. Verified against the repo's own caches: `espn_pga_2025.json` and `espn_pga_2024.json` → `{'NULL': 5888}` / `{'NULL': 5933}` status counts. `out_status` is therefore always False and the fallback labels everyone `made_cut = 1`.
- Observed wrong result (demonstrated, real cached data): Sony Open in Hawaii 2025 — field 143, **65 players with exactly two rounds (the missed cuts), zero labelled `made_cut = 0`**.
- Consequence chain: the next `python3 -m golf.fetch --accumulate` (update.sh step 1) re-parses the current+previous seasons and — because of N2 — **replaces the existing, correctly-labelled rows** for 2025–2026 with `made_cut = 1` for everyone. `walk_forward` then classifies every such event as no-cut, simulates it without a cut, and scores `p_cut ≈ 1` against `y_cut = 1`: the cut Brier collapses toward zero, the headline metric *improves*, the gate passes, and the next `calibrate --fit` bakes the corruption into `calibration.json`. The system reports its own data destruction as model improvement.
- Why tests missed it: `test_golf_inplay.py` exercises the *live* path with synthetic `STATUS_CUT` competitors; no test parses a real (status-less) season payload through `rounds_for`.
- Smallest safe remediation: restore the round-count rule as the fallback when no status exists — `made_cut = 1 if len(rounds) >= 3 else (0 if <event has 4 rounds> else 1)` — or infer the cut from 36-hole scores the way `espn.completed_round_scores` already does; never default an absent status to "made".
- Regression test: feed `rounds_for` a synthetic FINAL event whose competitors have null status and 2-vs-4 round players; assert two-round players get `made_cut = 0`.

### High

**N2 — Accumulate's replace logic never matches existing rows: every run rewrites all fetched seasons.**
- Location: `golf/providers/legacy.py:430-441` (`merged.get(key) == row` at line 436).
- Trigger: rows loaded from `rounds.csv` are all strings (`{'round': '1', 'score_to_par': '-8.0', 'made_cut': '1'}`); freshly parsed rows are typed (`{'round': 1, 'score_to_par': -8.0, 'made_cut': 1}`). Demonstrated: for 3 real 2025 events, **0 of 1,207 re-parsed rows** compared equal — all 1,207 counted as "new" and replaced.
- Observed wrong result: idempotency is gone (the operation reports thousands of "new rounds" weekly), and — the money path — it is the vehicle by which N1 retroactively corrupts two seasons of cut labels on the very next run.
- Why tests missed it: `test_store_round_import_and_field_export` tests the SQLite import, not `accumulate_rounds`' CSV round-trip.
- Remediation: compare on a normalized representation (cast the CSV row through the `RoundRecord` types before comparing), and log replacements separately from additions.
- Regression test: accumulate twice from the same cached payload; the second run must report 0 new rounds and leave the file byte-identical.

**N3 — Odds ≤ 1.0 pass the parser and silently reload a 3-ball as a 2-ball with the low-odds player dropped.**
- Location: `golf/providers/odds_manual.py` — `_parse_odds`/`finish` accept any numeric token as a price (no `> 1` check in the parse path), while `load_threeballs` filters slots with `od > 1` (`odds_manual.py:200`) and re-derives the market from the surviving count.
- Demonstrated: paste `AARON RAI / 0.5, COLLIN MORIKAWA / 2.38, JASON DAY / 3.50` → parser emits a valid `3ball` group with **no issue recorded**; after `write_threeballs_csv` → `load_threeballs`, it loads back as `[('2ball', 'COLLIN MORIKAWA', 2.38), ('2ball', 'JASON DAY', 3.5)]`. Both names are real field members, the event tag matches, and the group is priced as a twosome on a three-runner book — the exact C3 failure class through a different gap. A `0.5`-for-`5.0` typo is a realistic paste error.
- Why tests missed it: the new parser tests cover missing/misplaced lines, not sub-1.0 numeric odds.
- Remediation: reject odds ≤ 1.0 (and non-finite values) at parse time with an issue; make `load_threeballs` refuse — not shrink — a group whose stored `group_id` market disagrees with the surviving slot count.
- Regression test: the paste above must produce zero groups and one issue; a CSV row with `odds_a=0.5` under a `3ball-…` group_id must be refused.

### Medium

**M-a — `captured_at` timestamps are written in local naive time but decoded as UTC.**
- Location: `golf/providers/odds_manual.py:399-400` (`_ts()` = `time.strftime`, local, no offset) vs `board_captured_at` (naive → `tzinfo=utc`); consumed by `golf/engine.py:118-128` (`_board_fresh`).
- Demonstrated with `TZ=America/Los_Angeles`: a board written *this instant* decodes as **25,200 s (7 h) old** and `_board_fresh` refuses it against a refresh from 2 h ago — freshly pulled boards are excluded from live pricing. In the operator's timezone (UK, UTC+1 in July) the sign flips: every board reads 1 h fresher than reality, extending the intended 30-minute freshness window to ~90 minutes.
- Remediation: write `captured_at` with `datetime.now(timezone.utc).isoformat()` everywhere `_ts()` feeds provenance (bovada quotes, threeballs, matchups); `fetch.write_odds_csv` already does this correctly.
- Regression test: round-trip a board written under a non-UTC `TZ` and assert decoded age ≈ 0.

**M-b — The live fit silently drops all public-stat priors except on the day after play.**
- Location: `golf/model.py:601-603` — `include_public_stats = asof.date() >= today`, where a live fit's `asof` is `max(round date) + 1 day`.
- Trigger: any refit run ≥ 2 days after the last recorded round (i.e. most weekdays) → `asof < today` → 0 priors. Demonstrated: `fit(df)` (live, no asof) returns **0 public_stat_priors** while the shipped `model_params.json` (fitted 2026-07-18, when `asof == today`) carries 151. Two refits a day apart produce materially different ratings (previously measured: ~40% of a field shifts, mean |Δ| 0.045 SG/round) with no warning, and the feature is effectively dead in routine production use.
- Remediation: gate on "the stats snapshot predates `asof`" using the stats file's own capture date, not on `asof >= today`.
- Regression test: live fit two days after the last round with a fresh stats file must include priors; a 2023 `asof` must not.

**M-c — `board_event` trusts the first tagged row: mixed-event files bypass the guard.**
- Location: `golf/providers/odds_manual.py:360-368` — returns the first non-empty `event` value; rows from another event later in the same file are never checked (`engine.cmd_edge` compares only this single tag).
- Trigger: a hand-merged or partially-overwritten board whose first row carries the current event; remaining rows from last week price straight through (name overlap defeats the secondary field check in co-sanctioned weeks).
- Remediation: `board_event` should return the set of tags and callers refuse on more than one distinct value.
- Regression test: two-row matchups.csv with different event tags must be refused entirely.

### Low

**L-a** — Duplicate selections pass the parser (`Rai / Rai / Day` parses cleanly, no issue) — a paste error prices one golfer against himself.
**L-b** — A header without the `- name / name` suffix (`"3 Ball (Round 1)"`) silently swallows its whole group with **no issue recorded** (parser fuzz: `groups=[] issues=[]`); malformed headers should be reported like malformed groups.
**L-c** — `write_round_edges([])` still returns without clearing (`golf/round_pricer.py`), and `weekly_report`'s round-3-ball failure path (unlike `season.build_card`) does not delete a stale `round_edges.csv` before reading it.
**L-d** — Atomic writes cover card/report/edge/live-state, but `predictions.csv` (`simulate.write_predictions`), `field.csv` (`store.export_field_csv`), `players.csv`, and the board CSVs still use plain `open("w")` — a truncation there still yields a plausible partial artifact.
**L-e** — CSV formula-injection (names written verbatim) remains unaddressed; both matchup sides of one pairing can still both be staked (bounded by the new 10% group cap).

## 3. Original-finding closure matrix

| ID | Status | Evidence |
|----|--------|----------|
| C1 stale live_state | **fixed and verified** | `engine.py:163-175` requires event_id or normalized-name match against `field.csv` (fail-closed when either side is blank); `refresh.py:163-166` clears all three in-play artifacts on the leaderboard exception path (`_clear_inplay_artifacts`). Code inspection + `test_golf_inplay.py` passing. |
| C2 point-in-time leakage | **fixed and verified** | `walk_forward` forces `include_public_stats=False` + `safe_flags` disabling public_stat/global_priors/course_arch/weather (`validate.py:118-146`); demonstrated `fit(asof=2023-06-01)` → **0 priors** (was 151); `predict_field` gates file loads by flags (`model.py:1089-1096`); predictions carry `point_in_time_safe=1` and `calibrate.fit_from_csv` refuses files without it. Honest skills dropped accordingly (cut 6.4% vs 9–10% claimed before). |
| C3 parser mis-pairing | **partially fixed** | Rewritten parser fails closed with recorded issues on 11 of my 17 fuzz cases (missing first/middle/final price, missing player, promo line, wrapped name, count mismatches both directions, tie selection, reordered selections, American odds) — verified by execution. Remaining holes: sub-1.0 odds (N3, demonstrated repricing path), duplicate selections (L-a), silent headerless group (L-b). |
| H1 gate never gates | **fixed and verified** | `update.sh` now validates **before** refit with `set -euo pipefail` and `exit 2` on gate failure; calibration only runs after both succeed. `validate.main` updates the baseline only under `--write`, refuses legacy baselines (schema check), and fails the gate when no frozen baseline exists (`validate.py:615-655`). `bash -n` clean. |
| H2 phantom cut on no-cut events | **partially fixed** | Plumbing is complete and correct: `cut_rule`/`no_cut`/`total_rounds` flow ESPN→store→field.csv→`_simulation_rules`→both simulators→`completed_round_scores(no_cut=)`. But the real ESPN payload contains **no `format`/`noCut`/`cutRule`/`numberOfRounds` keys at all** (verified: event keys = competitions/date/…; `format: None`), so every event still gets the defaults (65/False/4). A genuine signature no-cut event is still phantom-cut unless the operator hand-edits the store/field.csv. |
| H3 tie/playoff economics | **fixed and verified** | Fractional dead-heat credit in the pre-tournament sim (`simulate.py:333-368` groups/credit; `_win_frac` tie-share), in-play 3-balls (`simulate_inplay.py` tie_size division), matchup pushes priced with conditional Kelly and push-aware EV (`edge.py:_bet_row`, matchup block), `_actuals` fractional credit (`validate.py:80-100`). Post-blend nesting clamp added (`edge.py` `offered`/`wider` chain). No win>top5 violation found, including an extreme-favourite in-play stress (win 0.904 < top5 0.985). |
| H4 divergent in-play simulator | **fixed and verified** | `simulate_inplay` now draws through `pre._draw_scores` under the shared `sim_config` regime, simulates the upcoming cut after R1, keeps the separate win regime, and settles both-missed-cut matchups by r36. Demonstrated round-zero parity at 100k sims: win 0.2864/0.2855, top5 0.7039/0.7044. In-play calibration disabled by default in `cmd_edge` (explicit `calibrated=1` overrides — caveat). |
| H5 mtime/untagged staleness | **partially fixed** | All odds writers now tag `event` + `captured_at` (bovada, fetch, manual boards); `cmd_edge` guards outrights too; `_board_fresh` reads content capture time and fails closed on missing/malformed provenance (legacy untagged files are refused). Remaining: the timezone skew (M-a, demonstrated) and the mixed-tag first-row bypass (M-c). |
| H6 tests can't fail | **fixed with a caveat** | Root `conftest.py` fails any pytest test whose module `FAIL` counter grew — and it works: it exposed a real latent failure in `test_m3.py::test_fitted_w` that was previously false-green. Caveat: `conftest.py` is **untracked** (`?? conftest.py` in git status); until committed, any other clone/CI regresses to false-green. |
| M1 selection on test set | **fixed and verified** (residual) | `tune_config` selects on [split, holdout) and promotes on an untouched [holdout,) window, raising `SystemExit` on empty windows (no silent fallback); `sweep_win_corr` likewise requires holdout improvement. Residual: the holdout date is a constant — repeated tuning runs still consume it; screening remains at 300–750 sims so selection noise is not quantified. |
| M2 in-sample baseline | **fixed and verified** | `summarize` now uses deployable naive baselines (1/field, N/field, 65/field with no-cut awareness) — `validate.py:184-199`. |
| M3 historical semantics | **regressed** | The good parts landed (18-hole completeness check, event-FINAL gate, id-keyed rows, WD/DQ status read) — but the made_cut rewrite (N1) plus the replace-everything accumulate (N2) make cut ground truth strictly worse than before on real payloads. |
| M4 settlement rules unused | **fixed and verified** | `round_pricer.py:174-186` honors `push_tie` vs dead-heat; `price_threeballs_r1.py` now dead-heat-correct with neutral defaults (course "", major off, rounded draws); matchup pushes handled in `price_all`. Residual: matchups.csv still carries no settlement column — push is assumed for all tournament matchups. |
| M5 not simultaneous Kelly | **fixed and verified** (scope-honest) | Docstring now says "capped fractional-Kelly… not a joint/simultaneous Kelly optimiser"; new `GROUP_CAP` covers mutually exclusive sets — demonstrated: 20 win bets capped to exactly 10.0, both matchup sides share one group key, rounding trimmed to the cap. Weather-wave correlation remains unmodeled (acknowledged limitation, not a claim). |
| M6 CLV dead code | **partially fixed** | `cmd_edge` now snapshots accepted, event-tagged boards; place/cut lines de-vigged individually (`market.py:168-180`); event mandatory for snapshot and lookup; UTC timestamps. Still unsupported: "closing" is merely the latest snapshot — nothing verifies it is pre-start, so CLV outputs remain unusable as closing-line evidence; `odds_history.csv` does not yet exist (no live run since the fix); no isolation between production history and ad-hoc/experimental `cmd_edge` runs. |
| M7 name identity | **partially fixed** | Ambiguous folded collisions now resolve to `None` (never a silent pick) — `model.py:713-727`; `refresh`/`round_pricer` normalizers delegate to `_fold_name`; `store._pid` prefers provider IDs; `accumulate` keys by `dg_id` when present. Remaining: `model.fit` and `import_rounds_csv` still key by display name — two same-named golfers still merge inside the fit itself. |
| M8 weekly report stale edges | **fixed** | `weekly_report._run_optional_steps` now persists `cmd_edge` rows via `write_edge_report` before the report reads them. Residual: stale `round_edges.csv` on the round-3-ball failure path (L-c). |
| M9 partial-board devig | **fixed and verified** | `price_all` passes `complete = len(win_board) ≥ 0.9 × field` (`edge.py:432`); `devig_outright(complete=...)` overrides the implied-sum heuristic. |
| L1 wrong-event fallback | **fixed** | `_events_for` returns `[]` when a requested id is absent (`espn.py:141-144`). |
| L2 `_to_par` zero default | **fixed** | Unparseable/blank values now return `None` and are skipped in both live (`espn.py:273-276`) and historical (18-hole + parse guard) paths. |
| L3 non-atomic writes | **partially fixed** | New `io_utils` used for card, weekly report, edge report, live scores/state; predictions/field/players/board CSVs still plain writes (L-d). |
| L4 weather round mapping | **not fixed** | `wave_features` still maps sorted forecast dates to rounds 1..N (`weather.py:161-178`); unchanged in the commit. Low impact (weather inert in validation, small clamp live). |
| L5 formula injection | **not fixed** | No escaping added to any CSV writer. |
| L6 broken legacy `evaluate_h2h` | **fixed** | Retired: now raises `RuntimeError` with a pointer to the safe path (`edge.py:286-292`). |

## 4. Mandatory scenario matrix

Original 15:

| # | Scenario | Verdict | Evidence |
|---|----------|---------|----------|
| 1 | Field refresh OK, leaderboard fails | **survives** | Exception path clears in-play artifacts (`refresh.py:163-166`); `_live_state` refuses on event mismatch and fails closed when either side lacks identity. |
| 2 | Copied/touched odds file | **survives** | Freshness now reads embedded `captured_at`; touching/copying preserves content time; missing/malformed → refused. (Timezone skew M-a weakens the window by the UTC offset.) |
| 3 | 80% shared field, blank/wrong tag | **survives** (mixed-tag caveat) | Outrights now guarded alongside matchups/3-balls (`engine.py:433-450`); untagged → refused. First-row-only `board_event` (M-c) is the remaining bypass, requiring a hand-mixed file. |
| 4 | Two golfers, same folded name | **partially — not proven safe end-to-end** | Ambiguous folds resolve to None (no silent pick) and store IDs are provider-keyed; but `model.fit` still merges identical display names in `rounds.csv`, so a true same-name pair still shares one skill history. |
| 5 | WD after one hole; DQ after cut | **fails (historical), survives (live)** | Live statuses handled; historically, partial rounds are now excluded (good) but N1 labels *everyone* made_cut=1 on real payloads — worse than the old approximation. |
| 6 | Limited-field no-cut event, default rule | **fails in practice** | The wiring exists, but ESPN's payload carries no cut/no-cut/rounds metadata (verified `format: None`), so defaults (65/False/4) apply to every event; a real no-cut event is still phantom-cut without manual data entry. |
| 7 | Event shortened to 54 holes | **fails** | `total_rounds` is plumbed but sourced from a nonexistent ESPN key → always 4; a shortened event still gets a phantom final round in-play. (Validation now correctly skips non-72-hole events.) |
| 8 | Ties for first / top-5 boundary | **survives** | Fractional dead-heat credit in sim, in-play, matchups (push), 3-balls, and `_actuals`; hand-check of `credit()` and `_win_frac` tie-sharing; win mass sums to 1.0 (demonstrated). |
| 9 | Both matchup players miss cut | **survives** | Pre-tournament unchanged (r36 ordering); in-play now settles via `settlement_totals = 1e6 + r36` too. |
| 10 | 3-ball missing tie price / two selections | **fails** | Tie-price and two-selection cases are now refused with issues (verified) — but the sub-1.0-odds variant (N3) still silently reprices the group as a 2-ball through the CSV round-trip (demonstrated). |
| 11 | Historical event uses today's files | **survives** | Demonstrated: 0 priors in a 2023 fit; safe flags in `walk_forward`/`sweep`; calibration refuses unsafe prediction files. |
| 12 | Hyperparams selected and reported on same events | **survives** (residual) | Disjoint select/holdout windows with loud failures on empty windows; blocking gate; calibration promoted per-market on a temporal 75/25 split (win demoted to identity — visible in `calibration.json`). Residual: fixed holdout date erodes with repeated runs; 300–750-sim screening noise unquantified. |
| 13 | Pre vs in-play at zero rounds | **survives** | Demonstrated parity at 100k sims across win/top5/cut (differences within MC error). |
| 14 | win > top5 after win draw + calibration + blend | **survives** | Post-blend nesting clamp in `price_all`; no violation found in randomized or extreme-favourite stress tests. |
| 15 | Twenty correlated +EV bets | **survives (bounded)** | New 10% group cap on mutually exclusive sets (demonstrated: 20 win bets → exactly £10 of £100) plus 10% player and 40% weekly caps with rounding trims. Weather-wave correlation remains outside any cap. |

New adversarial cases from this review: local-naive `captured_at` under non-UTC TZ **fails** (M-a, demonstrated ±7 h); accumulate idempotency **fails** (N2, 0/1,207 rows judged unchanged); live-fit prior dropout **fails** (M-b, demonstrated 0 vs 151 priors); mixed event tags in one board **fails** (M-c, code-traced); duplicate selections **fails** (L-a, demonstrated); malformed header **fails silently** (L-b, demonstrated); zero-odds round-trip **fails** (N3, demonstrated).

## 5. Validation and calibration audit

- **Point-in-time safety**: genuine for the shipped artifacts. `validation_predictions.csv` (18,873 rows, all `point_in_time_safe=1`, 2024-06-06 → 2026-07-09) was produced by the corrected path: historical fits verifiably exclude public priors (0 in a 2023 refit), and `predict_field` receives explicit disable flags. `calibrate.fit_from_csv` refuses unsafe inputs; `load_maps`/`validate.main` reject legacy artifacts by schema.
- **Metrics reproduce**: recomputing `summarize()` over the on-disk predictions yields **headline Brier 0.12932 exactly**, matching `validation_baseline.json`; per-market skills (win 1.8%, top5 3.7%, top10 5.5%, top20 7.5%, cut 6.4%) are plausibly honest — sharply down from the leaky 9–10% claims. Row count matches the reported 18,873. **The reported "131 events" does not reproduce: the file contains 149 tournaments.** Likely a stale or differently-filtered figure; it should be corrected or explained.
- **Calibration promotion**: `calibration.json._meta` (schema 2, holdout from 2026-02-12) promotes `top5/top10/top20/cut`; the **win market was demoted to an identity map** — consistent with the promotion logic and with honest evaluation, and evidence the artifact came from the corrected code path rather than hand-labelling.
- **Temporal honesty**: selection and promotion windows are disjoint and fail loudly when empty. Residual risks: the holdout dates are constants (reuse across repeated tuning runs), screening sims (300–750) leave selection noise of the same order as the 0.001–0.002 promotion thresholds, and the shipped calibration maps for promoted markets are fit on the full window (promotion decided out-of-sample, map fitted in-sample — conventional but worth stating).
- **Forgeability**: `point_in_time_safe` is a plain flag in JSON/CSV — it defends against *stale* artifacts, not a dishonest operator; nothing cryptographic ties it to the generating code.

## 6. Test results

Executed (sandboxed Linux, Python 3.10; installed for the run: pytest 9.1.1, scipy, scikit-learn, fastapi):

```
python3 -m py_compile golf/*.py golf/providers/*.py      → OK
bash -n golf/update.sh                                   → OK
git diff --check                                         → OK

pytest -q test_golf_app_wrapper.py test_golf_config.py test_golf_weekly_report.py
  → 7 passed (1.3s)
pytest -q test_golf_free_sources.py test_golf_inplay.py → 40 passed (6.9s)
pytest -q test_engines_contract.py test_security.py test_provenance.py
  → 14 passed (0.9s)
```

Full root sweep (run in batches; sandbox constrained long single runs):

```
test_bankroll, test_baseline_ownership, test_blend_gate, test_cfb_blend,
test_clv_suite, test_daily_card, test_edge_api, test_evidence_gate,
test_market_blend, test_model_audit                       → 88 passed
test_club_soccer + 5 horse-racing files                   → 23 passed, 1 failed
test_m2..m7, nfl, nhl, quote_provenance, release,
run_status, status_normalization, tennis                  → 99 passed, 1 error
test_v5, test_v6                                          → 7 passed, 4 failed
test_wc_v4                                                → 12 passed
test_worldcup_live_bracket                                → 1 passed
```

All golf and shared-contract tests pass. Unrelated failures (reported separately, none in golf): `test_m3.py::test_fitted_w` — a **real latent check() failure in the World Cup suite exposed by the new conftest** (previously false-green; validates the H6 fix); `test_club_soccer.py::test_card_written` and 2 of the `test_v5.py` failures are `PermissionError` on this sandbox's mount (environmental; club_soccer also carries unrelated uncommitted changes); 2 `test_v5.py` drift-report failures are in the World Cup engine.

Executed reproductions (scripts outside the repo): parser fuzz (17 cases), sub-1.0-odds CSV round-trip, TZ skew on `captured_at`, real-payload `made_cut` labelling (Sony Open 2025), accumulate idempotency (0/1,207 unchanged), round-zero pre/in-play parity (100k sims), extreme-favourite in-play nesting, 20-bet group-cap/rounding checks, historical-fit prior emptiness, headline-Brier recomputation. Everything else labelled "code-traced" above was inspection, not execution.

## 7. Remaining operator risks

1. **The gate can be fooled by data corruption it caused.** N1+N2: after the next accumulate, cut labels for two seasons flip to "made", validation reclassifies those events as no-cut, cut Brier collapses toward zero, and the blocking gate reads it as improvement. The one number the operator trusts is the one the bug inflates.
2. **No-cut/shortened-event handling looks fixed but is fed by metadata ESPN never sends** — every event runs on defaults (65/False/4) until someone hand-edits the store; the operator will reasonably believe scenario 6/7 are handled because the plumbing and flags exist.
3. **The false-green test fix lives in an untracked file.** If `conftest.py` is never committed, CI and fresh clones silently revert to a suite where four of five golf test files cannot fail — while the operator remembers the hole as closed.
4. **CLV/"closing" outputs remain evidentially empty**: history only starts accumulating now, nothing establishes a snapshot is pre-start, and production history shares a file with any ad-hoc `cmd_edge` invocation.
5. **Silent feature dropout on refit**: a routine mid-week refit drops all 151 public-stat priors (M-b), shifting ~40% of field ratings versus the shipped params with nothing but a missing dict in the JSON to show for it.

## 8. Prioritized remediation

Demonstrated correctness fixes:

1. Fix `made_cut` inference for status-less payloads (N1) **before the next accumulate/update.sh run**, and re-verify Sony-Open-style labelling; add the null-status regression test.
2. Fix accumulate row comparison (N2) so re-runs are no-ops; regression-test byte-identical idempotency.
3. Reject odds ≤ 1 at parse time and refuse shrunken groups on reload (N3).
4. Write all `captured_at`/`timestamp` provenance in UTC with offsets (M-a).
5. Gate public-stat priors on the stats snapshot's own date, not `asof >= today` (M-b).
6. Commit `conftest.py` (H6 closure is otherwise one clean checkout from regressing).
7. Make `board_event` refuse multi-tag boards (M-c); record an issue for headerless groups and duplicate selections (L-a/L-b).

Experiments / missing external evidence:

8. Source real cut-rule/no-cut/round-count metadata (manual per-event table or a second feed) — until then treat signature/no-cut weeks as manually-configured, not automated.
9. Once `odds_history.csv` accumulates, validate the "closing" definition against actual tee-off times before quoting CLV; keep test/experimental runs out of the production file.
10. Quantify screening-stage Monte Carlo noise vs the 0.001–0.002 promotion thresholds (repeat-seed variance study) before the next config promotion.
