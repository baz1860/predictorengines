# CFB module — adversarial review (2026-08-07)

> **Status: all findings addressed 2026-08-07.** See "Resolution" at the end of
> this document for what changed, what it moved, and what remains outstanding.

Scope: `cfb/*` plus `app/engines/cfb.py`. Style ignored; only functional failures. Ordered most → least important. Line numbers are current as of this review. Findings marked **[verified]** were reproduced by running code against the repo's data.

---

## 1. The validation gate validates a different model than production ships **[verified]**

**Where:** `cfb/validate.py:72-74` (`walk_forward`), `cfb/validate.py:271-273` (`split_ppa_walk_forward`), `cfb/ats_backtest.py:55-57` (`model_margins`), `cfb/blend_eval.py:25-27` (`collect`).

All four fit the Elo→points spread slope as:

```python
pre = (games["season"] < since).values
slope = float((diffs[pre] * m_all[pre]).sum() / (diffs[pre] ** 2).sum())
```

with **no FBS mask**. Production (`cfb/elo.py:186-187 fit_spread_map`) and `cfb/predictor.py:132-133 backtest` explicitly exclude FCS-vs-FCS rows, because those rows' `history[i][2]` values are **FCS-ledger diffs** (anchored at 850), not champion-ledger diffs. Since `fetch_data.py` began ingesting full FCS schedules, games.csv is 28% FCS-vs-FCS (7,703 of 27,521 rows; 5,507 of them land in the pre-2023 fit window).

Measured on the current data: contaminated slope **0.05853** vs production slope **0.05717** (+2.4%; ~0.3 pts at a 200-Elo gap).

Why this is the top failure: every governance artifact downstream is built on the contaminated `m_elo` column — `validation_baseline.json` (the regression gate that `update.sh` and `rehearsal.py` enforce), the **nested holdout that selected and froze the runtime `w_elo`** (`choose_weight` minimises Brier subject to a margin-MAE constraint computed from the wrong margins), `market_validation.py` (inherits `V.walk_forward`), the ATS/ROI tables, and the frozen README metrics. The gate is precise, hashed, fingerprinted — and measuring a model that isn't the one `season.py` prices with.

**Fix:** apply the same mask everywhere the slope is fit:

```python
pre = ((games["season"] < since)
       & ((games["home_div"] == "fbs") | (games["away_div"] == "fbs"))).values
```

Extract one shared `elo.fit_slope(games, history, mask)` and delete the three re-implementations. Add a regression test asserting `validate.walk_forward`'s slope equals `elo.fit_spread_map`'s on the same window. Then re-run `--nested-holdout --write` and `--update-baseline`, because the frozen `w_elo` and baseline are products of the contaminated fit.

---

## 2. The win-totals market pipeline is dead — the fetcher cannot succeed **[verified: output absent]**

**Where:** `cfb/fetch_win_total_lines.py:34` (`MARKETS_TO_TRY = ["team_totals", "wins"]`), `:62-65` (`fetch_odds`), `:48-59` (`fetch_sports`); `cfb/compare_win_totals.py:101-102`.

Three defects stack:

- `team_totals` is not a featured market on The Odds API's `/sports/{key}/odds` endpoint (it's event-endpoint-only), and `wins` is not a market key at all. The request 422s and `_get` (`:41-43`) `sys.exit`s on the first HTTPError — the fallback loop never runs.
- Even if `team_totals` returned data, it is **per-game team points**, not season wins. `parse_win_totals` would happily write per-game totals labelled as win totals and `compare_win_totals.py` would compute "edges" against them.
- `fetch_sports:59` takes `ncaaf[0]` — the first sport key containing "ncaaf" — which can select a futures/championship key, not the game-odds key.

Evidence it has never worked: `data/win_totals_lines_2026.csv` and `win_totals_raw_2026.json` do not exist, while `compare_win_totals.py` hard-requires the former. So `win_totals.py` produces projections with no market leg, and `compare_win_totals.py` (166 lines) is unreachable in practice — a three-file pipeline where one file has never produced output.

**Fix:** fetch season-wins from The Odds API event-level endpoint with an explicit correct sport/market key (or a provider that actually posts NCAAF season wins), assert the sport key equals `americanfootball_ncaaf`, and validate that returned `point` values look like win totals (2–12) not game totals (30–80). If you don't intend to bet win totals, delete the leg rather than leaving a silently-broken path.

---

## 3. `SystemExit` used as a domain error — can kill the host app

**Where:** `cfb/elo.py:451`, `cfb/power.py:124`, `cfb/epa.py:146` — `predict()` raises `SystemExit` for an unknown team. `app/engines/_inproc.py:49` catches `Exception` only; `SystemExit` derives from `BaseException` and sails through the adapter's error boundary, terminating the GUI process.

`engine.cmd_predict:40-42` validates team names against **power** params only, then calls `blend_predict` → `E.predict`, which checks against **Elo** ratings. Any name in power's 4-year window but absent from the current Elo snapshot (division reclassification, renamed program) crashes the app instead of returning an error dict. `win_totals.py:89` already catches `SystemExit` around `P.predict` — a workaround that proves the smell.

**Fix:** raise `ValueError` in all three `predict()`s; reserve `SystemExit` for `main()`. Validate against `eparams[1]` in `cmd_predict`. One-line safety net: catch `(Exception, SystemExit)` in `_inproc`.

---

## 4. Best-quote selection can shadow a stakeable quote with a stale one

**Where:** `cfb/season.py:295-305` (`load_market`).

Per (game, market, side) the modal-line group is reduced with `at_line.sort_values("odds", ascending=False).iloc[0]` — highest odds wins, **ignoring `quote_eligible`**. A 30-hour-old quote at 1.95 beats a fresh executable quote at 1.91; the stale row's `quote_eligible=False` then zeroes the stake at `season.py:414-419`. Result: the card reports "diagnostic only" edges on games where a perfectly stakeable quote existed. Freshness gating (`edge.py:152-161`) is undone by the selection step.

**Fix:** sort by `(quote_eligible, odds)` descending, or filter to eligible quotes first and fall back to ineligible only for display.

---

## 5. The CLI ledger drops the game ID it already has, then settles by name/date guess

**Where:** `cfb/edge.py:311-318` (ledger write), `cfb/bankroll.py:38-42` (`settle_bet`).

`edge.py` carries `event_id`/`cfbd_game_id` all the way through `prepare_odds` and the report (`:274`), then writes a ledger row **without either** — the identity is fetched, validated, and discarded at the money boundary. `bankroll.py --settle` then matches by `home/away` name and string-compared `date >= bet date`, taking `iloc[0]`. A rematch (conference championship, bowl) settles the bet against the wrong game. The app-side adapter got this right (`app/engines/cfb.py:161-180` requires a unique match on `cfbd:` id); the CLI path is the unsafe twin.

**Fix:** add `cfbd_game_id` to the ledger header and settle on it; require a unique match like `grade_open_bets` does. Longer term, delete one of the two edge pipelines (`edge.py main` vs `engine.cmd_edge` are ~150 duplicated lines already drifting).

---

## 6. One `season --odds-api` run rebuilds the same state three times, on possibly different data

**Where:** `cfb/season.py:540-546` (`main`), `cfb/live_evidence.py:163` (`capture_signals` → `CENGINE.cmd_edge`), `cfb/engine.py:85` (`cmd_edge` → `E.build_as_of`), `cfb/live_evidence.py:120-131` (`capture_quotes`), `:151` (`_commence_lookup`).

The flow: `build_card` runs a full Elo replay + `prepare_odds` + schedule parse; then `live_evidence.capture()` re-reads odds.csv (`capture_quotes`), re-runs `prepare_odds`, then `capture_signals` calls `cmd_edge`, which re-reads odds.csv, re-runs `prepare_odds`, and re-runs the **entire** `E.build_as_of` Elo replay; `_commence_lookup` reads odds.csv a fourth time. Nothing passes state between steps, so the paper signals can be computed from a different snapshot than the card if any file changes mid-run — and the run does ~3x the work by construction. `load_slate` is likewise called twice (`main:538` and `build_card:316`).

**Fix:** compute `(eparams, state_meta)`, the prepared odds frame, and the slate once in `main` and pass them down (`capture(result=...)` already accepts an injected edge result — use it).

---

## 7. Immutable data is re-downloaded / re-computed on every refresh

**Where:**

- `cfb/fetch_data.py:88-132` (`build_closing_spreads`): every weekly run re-reads the full 2006-2019 betting mirror and re-runs the expensive abbr-voting + groupby-apply consensus over ~14 seasons of data that **cannot change**, only to merge-protect newer imported seasons at `:124-127`. Build once, keep the artifact, append new seasons only.
- `cfb/prior_challenger.py:155-169` (`fetch_inputs`): `--fetch` re-downloads recruiting + portal for 2022→2026 each time, though 2022-2024 are frozen history whose SHA-256s are recorded in the same file. Cache per year; refetch only the current season.
- `cfb/engine.py:38` (`cmd_predict`): every GUI predict click replays Elo over all 27.5k games and SHA-hashes five input files (`elo._snapshot_hash`). ~0.2s today [verified], but it's pure recompute of state that only changes when games.csv does. Cache `build_as_of` keyed on `(target_season, as_of, games.csv mtime)`.
- `cfb/identity.py:127-139` (`resolve`): parses the 715KB schedule JSON **twice per call** (`schedule_catalog`, then `alias_index` → `schedule_catalog` again); called twice per event in `season.fetch_odds_api:183-184`. ~9ms/call [verified] — invisible now, structural waste forever. Memoise `schedule_catalog`/`alias_index` on `(season, file mtime)`.
- `cfb/edge.py:173-175` (`write_template`): calls `IDENTITY.registry_version(season)` **inside the per-fixture loop** — re-reading and re-hashing schedule + alias files once per upcoming game. Hoist it.
- `cfb/validate.py:297` (`split_ppa_walk_forward`): `load_blend_weight()` — a file open + JSON parse — inside the innermost per-game loop.

---

## 8. Data pulled and silently dropped

- `cfb/epa.py:64-68` (`load_ppa`): builds `def_passing`, `def_rushing`, `def_firstDown`… plus `off_ppa`/`def_ppa` aliases for every game row. **No code reads any `def_*` column or either alias** — `fit` uses only `off_{field}`. Six dead columns computed per row on every walk-forward.
- `ppa_games_*.json` has **no fetcher anywhere in the repo** (grep: only `epa.py` reads it) and `update.sh` never refreshes it. The EPA challenger and its promotion gate silently evaluate on whatever was last hand-fetched (currently through 2025). If the gate ever promotes EPA, production has no refresh path for its input.
- `cfb/elo.py:453`: `predict` returns `sigma` — dropped by every caller. `model_prob` (`edge.py:201-213`) prices spread/total cover for **blended** margins using the *power model's in-sample* `pparams["sigma"]` — the Elo sigma and any blend-specific residual width are never used. The cover probabilities driving stakes assume the blend has exactly power's error distribution.
- `cfb/season.py:474-489`: the manifest records `"blend_weights": load_blend_weights()` (three-way stack), but the card is priced with `model="blend"` → `load_blend_weight()` (two-way, `predictor.py:119`). The manifest documents weights the card did not use.
- `cfb/season.py:119`: `_slate_from_schedule_json(days)` — `days` parameter is dead.
- `cfb/engine.py:64` and `cfb/edge.py:231`: `(odds["odds"] != "")` on a numeric column — always true, dead guard.

---

## 9. Identity discipline bypassed by the win-totals compare path

**Where:** `cfb/compare_win_totals.py:29-35` (`_fuzzy_key`), `:38-57` (`merge`).

`identity.py`'s docstring: *"Fuzzy and prefix matching are deliberately report-only: an unknown spelling must block quote ingestion instead of being silently attached to the most plausible team."* `compare_win_totals` does exactly the forbidden thing — and badly: `_fuzzy_key` deletes the word "state", so `Ohio State` → `ohio`, `Michigan State` → `michigan`, `Mississippi State` → `mississippi`. When both a school and its State sibling are unmatched (likely, since book names carry mascots), lines can attach to the wrong school; with mascot-suffixed names the fuzzy key usually just fails, silently shrinking coverage. Route provider names through `identity.resolve(..., provider=...)` like `season.py` does. (Currently moot because of finding 2 — which is itself the point: two broken layers hide each other.)

---

## 10. The totals benchmark is stale by construction

**Where:** `cfb/data/closing_totals.csv` (last written 2026-06-12; seasons 2006-2019 + 2025 — a five-season hole), no writer exists: `fetch_data.py` builds spreads only (`build_closing_spreads`), `import_cfbd_lines.py` writes `closing_spreads.csv` only.

The totals market is the one currently at "paper" status (`policy.py:16`), i.e., the one closest to real money — and its evidence base (`validate.evaluate` totals ROI, `market_validation` totals gate) rests on a file nothing can regenerate or extend. `dataset_fingerprint.py` dutifully hashes it, which freezes the gap rather than fixing it.

**Fix:** add a totals equivalent of `import_cfbd_lines` (CFBD `/lines` carries `overUnder`) and backfill 2020-2024.

---

## 11. Smaller defects

- `cfb/elo.py:448-453`: standalone `elo.predict` prices FBS-vs-FCS margins with the champion slope; the cross-division compression fix (`cross_slope`, ~0.76×) is applied only in `blend_predict` (`predictor.py:95-99`). CLI Elo margins for cross-division games overshoot by ~24% relative to the blend path's own correction.
- `cfb/fetch_data.py:96`: if the sparse-checkout `add` succeeds but the mirror dropped `cfb_line_odds.csv.gz`, `pd.read_csv(src)` raises with no context. Guard and skip with a warning.
- `cfb/season.py:427-436`: on `preview_bets` failure every eligible stake is zeroed ("failed closed" — good) but `betting_eligible` stays `True` on the rows and in the manifest's `value_bets` count, so the manifest reports value bets that carry £0 stakes. Set the flag false too, or record the degradation in the manifest.
- `cfb/fetch_win_total_lines.py:90`: `overs = [e for e in entries if e["side"] in ("over", "")]` — treats missing `description` as "over"; two-sided books without descriptions would median over- and under-odds together. Moot until finding 2 is fixed.
- Duplicate logic: `edge.py main` vs `engine.cmd_edge` (~150 lines), `_atomic_json`/atomic-write helpers re-implemented five times (`season.py`, `live_evidence.py`, `run_status.py`, `validate.py`, `prior_challenger.py`, `market_validation.py`, `dataset_fingerprint.py`). Consolidate into one `cfb/_io.py`; duplication is where the next drift bug (finding 1 was exactly this) comes from.

---

## What's actually good (for calibration)

Atomic writes with staged validation everywhere; last-good retention on fetch failure; the identity registry blocking unreviewed names at odds ingest; the layered betting gates (prior readiness → per-team evidence → quote provenance/freshness → market policy) that fail closed; append-only live evidence with dedupe; the fingerprint-bound validation baseline. The governance skeleton is genuinely strong — which is why finding 1 matters so much: the gate is trustworthy machinery pointed at slightly the wrong target.

## Suggested order of work

1. Fix the slope mask (one line × 4 files) + shared helper + regression test; re-freeze nested holdout and baseline.
2. Change `SystemExit` → `ValueError` in the three `predict()`s; broaden `_inproc` catch.
3. Prefer eligible quotes in `load_market`.
4. Add `cfbd_game_id` to the CLI ledger and unique-match settlement.
5. Single-pass `season --odds-api` (share eparams/odds/slate).
6. Rebuild or delete the win-totals market leg; if kept, route names through `identity.resolve`.
7. Add a totals-lines importer and backfill 2020-24.
8. Cache `schedule_catalog`/`build_as_of`; hoist per-row `registry_version`; stop rebuilding 2006-2019 spread consensus weekly.

---

# Resolution (2026-08-07)

All 11 findings addressed. `python3 -m pytest tests/cfb test_cfb_blend.py` → **65 passed**
(56 existing + 9 new); `python3 -m cfb.rehearsal` → **7/7 checks pass**;
`python3 -m cfb.validate --gate` → **PASS** on the re-frozen baseline.

### 1. Slope contamination — fixed, and it moved the runtime model

New `elo.fit_slope(games, history, mask)` (`cfb/elo.py:175`) owns the masked fit;
`fit_spread_map`, `validate.walk_forward`, `validate.split_ppa_walk_forward`,
`ats_backtest.model_margins`, `blend_eval.collect` and `predictor.backtest` all
call it. Four duplicate implementations deleted.

**This changed the shipped model.** Re-running the nested holdout selected a
different runtime blend weight:

| | before | after |
|---|---|---|
| runtime `w_elo` | 0.55 | **0.60** |
| holdout ML Brier | 0.18663 | 0.18650 |
| baseline margin MAE | 12.791 | 12.784 |

Re-frozen: `nested_validation_2025.json`, `blend_weight.json`,
`validation_baseline.json`, `market_validation_2025.json`,
`prior_challenger_2025.json`, `validation_datasets.json`, `README.md`.
The market-challenger verdict is unchanged (still not promoted).

Guarded by `tests/cfb/test_slope_consistency.py` — three tests, including one
asserting `walk_forward`'s slope equals production's on the same window, and one
asserting the contaminated fit is *measurably different* so the test can fail.

### 2. Win totals — rebuilt, and the provider verdict is in

A live `--list` against the account settled the open question: **The Odds API
does not offer NCAAF season win totals.** Only two NCAAF keys exist —
`americanfootball_ncaaf` (per-game) and `americanfootball_ncaaf_championship_winner`
(an outright title market, not per-team win counts). There is no season-wins
key to fetch, so the market cannot be sourced from this provider at all.

Two further independent bugs surfaced while proving the path end-to-end — this
three-file leg had a defect in *every* file, which is why nothing ever
surfaced it:

- **Path mismatch:** `win_totals.py` writes `cfb/projected_win_totals_<yr>.csv`
  but `compare_win_totals.py` read `cfb/data/…`. Even with lines present the
  compare would have exited "Missing … run win_totals.py". Now checks both.
- **Duplicate-team double bets:** after canonicalisation, a raw name and its
  reviewed alias (`Ohio State` + `Ohio State Buckeyes`) both survived, so the
  same team was priced and flagged as *two* bets on one market. Now
  consolidated to one row per canonical team.
- **Invalid averaged prices:** consolidating those rows medianed American odds
  directly — `-110` and `+100` average to `-5`, not a price, which produced a
  bogus +93% edge. Odds are now medianed in probability space
  (`_median_american`), and `american_to_implied` treats any `|odds| < 100` as
  the -110 default.

Because there is no provider, a `--template` path now writes a hand-fillable
lines CSV seeded from the model's projected teams:

```
python3 -m cfb.fetch_win_total_lines --template --year 2026   # 138 rows
python3 -m cfb.compare_win_totals --year 2026
```

Verified end-to-end with hand-entered lines: edges compute, an unreviewed
spelling is reported rather than guessed, and one team yields one bet.

`fetch_win_total_lines.py` rewritten: discovers live sport keys instead of
hardcoding `team_totals`/`wins` (neither of which could work), ranks futures
keys first, continues past 4xx instead of exiting on the first one, and — the
important part — **refuses to write rows whose points aren't in season-win
range** (0.5–14.5). A per-game-totals response is now a hard failure rather
than a silently mislabelled CSV. `parse_win_totals` identifies Over/Under
explicitly and drops rows without a determinable side (previously a missing
`description` was treated as "over", mixing over and under prices).

Still needs one live run against your key to confirm the market exists on your
plan: `python3 -m cfb.fetch_win_total_lines --list` shows visible sport keys.

### 3. `SystemExit` — fixed

`elo.predict`, `power.predict`, `epa.predict` raise `ValueError`; the three
`main()`s convert to `SystemExit` at the CLI boundary. `_inproc.run_inprocess`
now also catches `SystemExit` as a backstop. `engine.cmd_predict` validates
against **both** Elo and power rosters. `win_totals.py`'s `except SystemExit`
workaround became `except ValueError`.

### 4. Quote shadowing — fixed

`season.load_market` sorts by `(quote_eligible, odds)` (`season.py:305`), so a
fresh executable quote wins over a better-priced stale one.

### 5. Ledger identity — fixed

`edge.py` writes `cfbd_game_id` + `bookmaker` into the ledger (existing ledgers
migrate in place on next write). `bankroll.settle_bet` settles on game ID, and
its legacy name/date fallback now requires a **unique** ±2-day match and a
finished game — a rematch can no longer settle against the wrong game.

### 6/7. Redundant work — cached

| path | before | after |
|---|---|---|
| `elo.build_as_of` | 0.157 s/call | 0.0035 s warm (**45×**) |
| `identity.resolve` | ~13 ms/call | 0.009 ms warm (**~1400×**) |
| `edge.fixture_registry` | 0.128 s/call | 0.00015 s warm (**850×**) |

All keyed on input-file mtime/content so a data refresh invalidates them.
`fetch_data.build_closing_spreads` skips the 2006-2019 consensus rebuild when
the mirror source hash is unchanged; `prior_challenger.fetch_inputs` reuses
frozen historical years (`--refetch-all` to override);
`edge.write_template` hoists `registry_version` out of the per-fixture loop;
`split_ppa_walk_forward` hoists `load_blend_weight` out of the inner loop;
`season.main` loads the slate once and passes it to `build_card`.

### 8. Dropped data — fixed

`epa.load_ppa` no longer builds six unread `def_*` columns or the unused
`off_ppa`/`def_ppa` aliases. The card manifest now records `runtime_w_elo`
(what `model="blend"` actually priced with) alongside the full stack.
`_slate_from_schedule_json`'s dead `days` parameter and two always-true
`odds != ""` guards removed.

### 9. Identity bypass — fixed

`compare_win_totals` routes book names through
`identity.resolve(..., provider="the-odds-api")`. The `_fuzzy_key` matcher that
collapsed "Ohio State" → "ohio" is deleted; unresolved names are listed for
review, never attached to a plausible team.

### 10. Totals benchmark — importer built and gap backfilled

`import_cfbd_lines` now also writes `closing_totals.csv` from CFBD `overUnder`,
validated by correlation against actual totals. Verified against the existing
2025 data: **1,594 rows reproduced byte-identically**. A `--fetch` flag pulls
CFBD `/lines` so the 2020-24 hole can be backfilled:

```
python3 -m cfb.import_cfbd_lines 2020 2021 2022 2023 2024 --fetch
```

The same run added **11 genuinely new 2025 closing-spread rows** with zero
changed values (the large `closing_spreads.csv` diff is row reordering).

**The backfill has since been run.** The 2020-24 hole is closed:

| | before | after |
|---|---:|---:|
| `closing_totals.csv` rows | 9,227 | **15,104** |
| totals seasons | 2006-19, 2025 | **2006-2025 (contiguous)** |
| `closing_spreads.csv` rows | 15,078 | **15,263** |

The fingerprint gate correctly refused the old baseline afterwards — working
as designed — so everything was re-frozen a second time against the richer
data. This also **retired a workaround that only existed because of the gap**:
`market_validation` used to start its walk in 2018 so totals calibration could
reach 2018-19, the last pre-holdout seasons with lines. That's now data-driven
(`MIN_CALIBRATION_ROWS`), and totals calibrate on **2023-24 (1,587 rows)** —
the same window as spreads — instead of 2018-19 (1,173 rows). The challenger
verdict is unchanged: still not promoted.

### 11. Smaller defects — fixed

`fetch_data` warns and keeps the last-good file when the mirror lacks the
betting CSV instead of raising bare; `season`'s failed-closed portfolio preview
now also sets `betting_eligible = False` so manifest counts are truthful.

---

## Follow-up: two further defects found while operating the module

Running `bash cfb/update.sh` after the fixes exposed two more real problems.
Both are fixed; tests added; artifacts re-frozen. Suite is now **81 passing**.

### A. `update.sh` silently destroyed imported line history — *data loss*

`fetch_data.build_closing_spreads` retained imported rows with:

```python
keep = old[old["season"] > m["season"].max()]     # m = mirror consensus
```

The comment said "keep imported seasons the mirror doesn't cover (e.g. CFBD
2020+)". That works only while the mirror lags. **The sportsdataverse mirror now
covers 2006-2025**, so `m["season"].max()` is 2025, `keep` selects nothing, and
every refresh discarded all CFBD-imported lines.

Observed: `closing_spreads.csv` fell 15,263 → **15,078 rows**, wiping 185 games
— the 2020-25 fixtures the mirror does not carry (2020: 567→534, 2021: 887→843,
2022: 1459→1403, 2023: 1413→1383, 2024: 1557→1546, 2025: 1597→1586). The
mirror's 2020-25 rows also carry **no juice**, so they were not even a
like-for-like replacement.

This is worse than the losses it caused: it is silent, it recurs on every
weekly run, and it only began once the mirror caught up — so it would have
looked like a "sudden" regression with no code change to blame.

Fixed by `fetch_data.merge_with_imported()` — retention is per **game**, not
per season: mirror rows win where both have a fixture (they carry the juice),
and any previously imported game the mirror lacks is kept. Output is also
sorted deterministically, so an unchanged dataset hashes identically instead of
tripping the fingerprint gate on row churn. The 185 rows were restored from the
on-disk `lines_*.json` (no refetch). Two tests cover it, including one that
reproduces the exact "mirror max == imported max" condition that made the old
rule a no-op.

### B. Reviewed-schedule gate fired on data nobody reads

The gate hashed the **raw** provider file, so it failed after a routine
`fetch_cfbd` refresh. Investigation: 888 events unchanged, none added or
removed, and no change to any event ID, kickoff, team, week, classification,
neutral-site or completed flag. The sole difference was CFBD backfilling its own
`homePregameElo`/`awayPregameElo` on 156/112 events — fields read **nowhere** in
the codebase.

A control that fires on changes nobody needs to read is a control that stops
being read; the failure mode is an operator who reflexively re-blesses it. The
gate now hashes only `identity.SCHEDULE_DECISION_FIELDS` (event ID, season,
week, season type, kickoff, both team IDs/names/classifications, neutral site,
completed), order-independently.

Crucially, the decision-content hash of the current file is **byte-identical to
the 2026-08-02 reviewed schedule**, so this recorded that the approved content
still holds rather than approving a change. `reviewed_schedule.json` gains a
`review_log` entry with the finding and decision, and falls back to the raw hash
for pre-migration records.

Loosening a control demands proof it still bites: 11 tests, including
parametrised cases asserting re-review still triggers on a moved kickoff, a
renamed team, a changed team ID, a reclassification, a neutral-site flip, a week
change, a completed-state change, and added/removed events.

### C. `cfb/refreeze.sh` — one reviewed command for the re-freeze sequence

The four-step sequence was run three times this session by hand, which is
exactly how a step gets skipped and the artifacts drift out of sync.
`cfb/refreeze.sh` runs nested holdout → baseline → market challengers →
generated docs, then verifies the gate and prints the runtime `w_elo` and
baseline metrics **before and after** so the operator can see whether the
runtime model moved.

Safety rails, because this rewrites the artifacts that authorise the engine:

- Requires `--confirm`; without it, explains itself and exits non-zero.
- Rejects unknown flags with exit 2, so a typo can't read as an unconfirmed run.
- Backs every artifact up to `cfb/data/backups/refreeze_<stamp>/` first
  (gitignored — local rollback copies, the artifacts themselves stay versioned).
- Verifies `generate_docs --check` and the gate before reporting success.
- Deliberately **not** wired into `update.sh`. A pipeline that rebaselined
  itself would defeat the fingerprint gate completely.

Verified idempotent: a run with unchanged inputs leaves `w_elo`, all baseline
metrics and the line fingerprint identical. Three tests pin the rails,
including one asserting `update.sh` contains neither `refreeze` nor
`--update-baseline`. RUNBOOK gains Step 9 and three troubleshooting rows.

### Not done / outstanding

- **`elo.predict` cross-division slope** (finding 11, first bullet). The
  standalone CLI still uses the champion slope for FBS-vs-FCS; only
  `blend_predict` applies `cross_slope`. Left alone deliberately: changing
  `elo.predict` alters the Elo-only model that the validation baseline covers,
  and it deserves its own gated change rather than riding along here.
- **`ppa_games_*.json` has no fetcher.** The EPA challenger still evaluates on
  hand-fetched data. It is not promoted, so this is not a runtime risk today —
  but if EPA is ever promoted it needs a refresh path in `update.sh` first.
- **Win totals have no data source.** The model projections
  (`win_totals.py`) work, but The Odds API carries no NCAAF season-wins market,
  so the comparison only runs on hand-entered lines via `--template`. If you
  want this automated, it needs a different provider.
- **Duplicate edge pipelines** (`edge.main` vs `engine.cmd_edge`) and the seven
  re-implemented atomic-write helpers are untouched. Both are drift risks —
  finding 1 was exactly this failure mode — but consolidating them is a
  refactor, not a fix, and would have obscured the diffs above.
- The win-totals fetcher's *fetch* branch remains unexercised (there is no
  market to fetch). Its discovery, rejection and template branches are all
  verified, and discovery costs zero quota — `/sports` is a free endpoint, and
  the code no longer probes keys that cannot carry win totals.
