# Adversarial Review — `golf/` module

Scope: all Python under `golf/` (incl. `golf/providers/`), the five `test_golf_*.py` files, `golf/update.sh`, the app/engine contract paths, and the schemas/metadata of the JSON/CSV artifacts. All findings verified against code at the cited lines; demonstrated findings were reproduced with temporary scripts outside the repository (commands in §4). No production code or data was modified.

---

## 1. Findings

### Critical

**C1 — A stale `live_state.json` from a previous event silently conditions all pricing on last week's leaderboard.**
- Location: `golf/engine.py:103-148` (`_live_state`), `golf/refresh.py:156-164` and `golf/refresh.py:437-455` (`_write_live_scores`).
- `live_state.json` stores `event_id`/`event_name`, but `_live_state` never compares them to the current field's event. Refresh clears the in-play artifacts only when the leaderboard read *succeeds* and returns `rounds_done < 1` (`refresh.py:447-455`). If the leaderboard read raises (network error, ESPN schema drift), the handler at `refresh.py:163-164` records a QA warning and leaves last week's `live_state.json` and `scores_live.csv` in place.
- Failure scenario: Thursday refresh for a new event — field fetch succeeds, leaderboard fetch fails. `cmd_simulate`/`cmd_edge` auto-route in-play (`engine.py:252-254`, `409`), `_inplay_results` (`engine.py:151-180`) matches this week's field to last week's scores by `name.lower()` — consecutive events share most of the field — and every market is priced as if this week's players held last week's 54-hole scores. The mtime board-freshness guard passes (boards were just refreshed), the card is written, and stakes are recommended.
- Why tests miss it: `test_golf_inplay.py` tests the happy paths of `_write_live_scores` (write and clear), never the exception path, and no test crosses an event boundary.
- Remediation: in `_live_state`, require `norm_event(state["event_name"]) == norm_event(current field event)` (or matching `event_id`) and return `None` otherwise; clear the three in-play artifacts inside the `except` branch of the live-scores step. Regression test: write a `live_state.json` tagged with event A, a `field.csv` tagged event B, and assert `cmd_simulate` runs the pre-tournament projection.

**C2 — Walk-forward validation, calibration, and every tuning/README claim leak *today's* stat and prior files into historical fits (mandatory scenario 11).**
- Location: `golf/model.py:594` (`fit()` calls `load_public_stat_priors()` with no as-of), `golf/model.py:1074-1076` (`predict_field` loads current `course_features.csv` and `global_player_priors.csv`), consumed by `golf/validate.py:128-133` (`walk_forward`).
- Demonstrated: `model.fit(df, asof="2023-06-01")` returns `public_stat_priors` with **151 current-season entries** (e.g. today's SG for Fitzpatrick/Thomas/Fleetwood baked into a "2023" model). These priors blend into ratings at weight 0.15 for fitted players (`model.py:1032-1039`) and become the *entire rating* for players not yet in the fit (`model.py:1013-1023`) — exactly the debutants/LIV players where the backtest looks smartest. `global_player_priors.csv` is a hand-curated 2026 file applied to every historical event.
- Consequence: `validation_predictions.csv` (17,286 rows), `calibration.json`, `model_config` tuning, the sim-shape sweep, and the README's "+9–10% skill" all rest on information unavailable at the historical prediction time. Magnitude, measured on one June-2024 walk-forward event (156-player field): removing today's files shifts **61/156 ratings by >0.01 SG/round, mean |Δ| 0.045, max |Δ| 0.39 SG/round** — 0.39 SG/round is the gap between a mid-field player and a contender, and the shifts concentrate on the thin-history players where the backtest looks smartest.
- Why tests miss it: no test fixes the file-vs-asof relationship; the ablation flags (`validate.py:471-489`) cover `weather`/`course_arch`/`global_priors` but **there is no flag for the public-stat prior at all**, so the largest leak cannot even be ablated.
- Remediation: in `fit()`/`predict_field`, accept an explicit `asof` and return empty priors when the snapshot files postdate it (or thread `feature_flags={"public_stat": False, "global_priors": False, "course_arch": False}` from `walk_forward` by default). Re-run validation and expect the headline numbers to worsen. Regression test: `fit(df, asof="2023-01-01")["public_stat_priors"] == {}` while the current CSV is present.

**C3 — The pasted-board parser silently mis-pairs players and odds; a 3-ball missing one price is priced as a 2-ball with everyone's odds shifted (mandatory scenarios 5-adjacent and 10).**
- Location: `golf/providers/odds_manual.py:248-280` (`parse_skybet_threeball_text`).
- The parser keeps a `pending` name queue and pairs the *oldest* pending name with each odds-like line. Demonstrated:
  - Missing odds line: `RAI / MORIKAWA / DAY` where Rai's price is absent parses to `[('AARON RAI', 2.38), ('COLLIN MORIKAWA', 3.50)]` — Rai gets Morikawa's price, Morikawa gets Day's, Day disappears, and the group is re-labelled a **2-ball**. Every downstream guard passes (event tag matches, both names are in the field), so a 3-runner book is priced as a 2-runner market with wrong odds and staked.
  - A promo line ("Was 3.10") becomes a phantom player and shifts all subsequent pairings; a name wrapped across two lines becomes two phantom players. These usually get the group dropped by the field-mismatch guard — but only when the phantom itself fails the field check; the shifted *real* names sail through.
  - A tie/deadheat selection makes the group 4 entries and it is silently discarded (`len ∈ (2,3)` filter, line 280) with no warning.
- Why tests miss it: `test_golf_free_sources.test_manual_threeball_parser` tests one perfectly-formed paste (and cannot fail under pytest anyway — see H6).
- Remediation: parse strictly alternating name→odds; refuse any group where the count of parsed players disagrees with the header's `A / B / C` name count, and surface refused groups as QA warnings. Regression test: the missing-odds paste above must produce zero groups and a warning, not a 2-ball.

### High

**H1 — The "regression gate" gates nothing, and the baseline ratchets on noise.**
- Location: `golf/update.sh` steps 3–5; `golf/validate.py:564-577`.
- `update.sh` **refits first** (step 3 overwrites `model_params.json`), then runs `validate --gate` with `|| echo "validation gate warning"` — the non-zero exit is swallowed (no `set -e`), and step 5 unconditionally refits calibration from the (possibly regressed) predictions. The README's claim that validate is "the regression gate the daily update.sh runs before trusting a refit" is false on both counts: it runs *after* the refit and its failure changes nothing.
- Separately, `validate.py:574-577` adopts any improvement beyond tolerance as the new baseline. Repeated noisy runs ratchet the baseline to the minimum of a random sequence; future honest runs then "regress" or the recorded baseline (currently 0.14553) overstates true skill.
- Remediation: order gate → refit, propagate the exit code (or restore `model_params.json.bak` on failure), and freeze the baseline except by explicit `--write`.

**H2 — No-cut and limited-field events are priced with a phantom 65-player cut, and the in-play path invents a cut too (mandatory scenario 6).**
- Location: `golf/engine.py:259` and `457` (`cut_rule = 65` default; no `no_cut` parameter exists), `golf/season.py:187` (`base` passes neither), `golf/store.py:114` (`events.no_cut` is written but read by no consumer), `golf/providers/espn.py:289-296` (`completed_round_scores` applies a synthetic top-65 cut).
- Demonstrated: a 72-player no-cut event under defaults yields `cut_binds=True` and `made_cut ≈ 0.904` per player — below the 0.99 CERTAINTY guard, so if `odds_cut` appears on the board a make-cut market that does not exist is priced and staked. The `cut_binds` suppression only rescues fields ≤ 65. In-play the ESPN provider similarly marks players outside its synthetic top-65 line as `made_cut=0` after R2, dropping them from all in-play pricing of a no-cut event.
- Remediation: read `cut_rule`/`no_cut` from the ESPN event (stored in `events`) through `season.build_card` → `cmd_simulate`/`cmd_edge` → `completed_round_scores`. Regression test: 72-player no-cut event must produce `made_cut = 1.0` and price no cut market.

**H3 — Win-market probabilities are internally inconsistent after blending, and tied/playoff winners are double-counted in both simulation and validation labels (mandatory scenarios 8 and 14).**
- Location: `golf/simulate.py:360-361` (win clamped to top5 *before* calibration/blend only), `golf/edge.py:499-508` (per-market blends with different weights: win 0.60, top5 0.45), `golf/simulate.py:116-126` (`_win_frac` counts every tied leader fully), `golf/validate.py:80-93` (`_actuals` `rank(method="min")`).
- Demonstrated: calibrated nested probs (win 0.100 ≤ top5 0.101) blended toward a market that likes the winner become **win 15.3% > top5 10.5%** — the exact scenario-14 inconsistency, and both numbers feed EV/Kelly.
- Demonstrated: a playoff (two players tied at −10 over 72 holes) yields `y_win = 1` for **both** players in `_actuals` — the real event pays one winner. Win base rates, Brier "skill", and the isotonic win map are all trained on inflated labels; live top-5 boundary ties are likewise counted as full wins while books apply dead-heat reduction. Model EV for win/top-N is overstated at exactly the tied-outcome margin.
- Remediation: enforce nesting *after* blending in `price_all`; count a k-way tied win as 1/k (or model playoff explicitly) in both `_win_frac`/`positions` accumulation and `_actuals`; settle top-N ties with dead-heat fractions. Regression test: three-way tie must contribute ⅓ each, and post-blend rows must satisfy win ≤ top5.

**H4 — The in-play simulator is a different, unvalidated model, and pre-tournament calibration is applied to its output (mandatory scenario 13).**
- Location: `golf/simulate_inplay.py:109-114` (independent Gaussian rounds — no `round_corr`, no `tail_df`, no win regime), `golf/engine.py:451-462` (in-play results passed through `price_all` with the calibration maps fitted on the correlated-t pre-tournament simulator).
- Demonstrated at the round-zero boundary with identical fields and 120k sims: pre-tournament top-5 for the favourite 0.753 vs in-play 0.824 (top-10/20 similarly shifted); win agrees only because `win_round_corr` happens to be 0.0. So the moment a round completes, every place probability jumps by several points for reasons unrelated to the leaderboard, and the isotonic maps correct biases of a simulator that is no longer running. Additionally, with `rounds_done=1` the remaining-round simulation applies **no 36-hole cut at all** (survivor filter only; `simulate_inplay.py:89`), and `_inplay_results` hardcodes `made_cut=1.0` after R1 (`engine.py:176`) when the cut hasn't happened yet.
- Remediation: draw remaining rounds under the same validated regime (share `_draw_scores`), simulate the upcoming cut when `rounds_done < 2`, and either calibrate in-play separately or not at all. Regression test: pre vs in-play at zero conditioning must agree within MC error on all markets.

**H5 — Odds staleness protection is mtime-based and does not cover outrights at all pre-tournament (mandatory scenarios 2 and 3).**
- Location: `golf/providers/bovada.py:235-247` (`write_outrights_csv` writes `odds.csv` with **no event column** — confirmed in the live file), `golf/engine.py:380-399` (event-tag guard covers matchups/3-balls only), `golf/engine.py:87-99` + `409-423` (mtime freshness, applied only once in-play), `golf/refresh.py:306-307` (Bovada failure "leaves previous CSVs intact"), `golf/refresh.py:320-335` (paste staleness = file mtime).
- Failure scenarios: (a) a copied or `touch`ed odds file passes every freshness check with last week's prices — `_board_fresh` and the paste guard read mtimes, not content; (b) Monday refresh for the new event with Bovada down (geo-blocked, schema change) leaves last week's untagged `odds.csv` in place, and pre-tournament there is *no* guard between it and the new field — with 80% shared players the outright/place markets price cleanly against wrong odds. The matchup/3-ball tag guard is real, but it is disabled whenever `field.csv` carries no event value (`engine.py:386: if current_event:`).
- Remediation: tag `odds.csv` rows with the event and enforce the same `norm_event` comparison for outrights; embed a `captured_at` timestamp *in the file content* and compare that (not mtime); treat a missing tag as refuse-by-default.

**H6 — Four of the five golf test files cannot fail under pytest.**
- Location: `test_golf_app_wrapper.py`, `test_golf_config.py`, `test_golf_free_sources.py`, `test_golf_weekly_report.py` — all use a print-only `check()` helper (e.g. `test_golf_free_sources.py:23-30`) with **zero assert statements**; failures only matter via `main()`'s exit code when run as scripts. Under `pytest -q` (the run the review and CI would use) every `check` can be false and the suite is still green — verified: 40 passed in 4.3 s.
- Consequence: combined with H1 (swallowed gate), the project has effectively no automated safety net for the fit, parser, store, weekly report, or app wrapper paths; only `test_golf_inplay.py` contains real assertions.
- Remediation: replace `check(name, cond)` with `assert cond, name` (a five-line change per file). Then re-run — any latent failures these tests were masking will surface.

### Medium

**M1 — Hyperparameter selection touches its own "out-of-sample" set; screening is under-powered (mandatory scenario 12).**
- `sweep_win_corr` picks the winner **by** post-split log-loss (`validate.py:378-396`) — the split that is then quoted as the out-of-sample gain (`sim_config.json`: 0.04525 → 0.04500, a 2.5e-4 margin over 6 candidates: winner's-curse sized). `tune_config` reuses the same fixed 2025-01-01 split for every invocation, screens at 300–750 sims (MC noise comparable to the deltas it ranks), and its feasibility/promotion thresholds (±0.002) are of the same order as the reported improvements. Every tuning artifact and the README headline are computed on windows overlapping these selection sets.
- Remediation: three-way temporal split (tune / select / final-report-once), or nested CV over event blocks; report only the untouched final window.

**M2 — The "skill" baseline is computed on the evaluation set itself.** `summarize` uses `base = y.mean()` over the evaluated rows (`validate.py:166-171`) — a non-deployable comparator (it knows the eval-set base rate). The "+9–10% Brier skill" figures are relative to this, not to a real ex-ante baseline (e.g. previous-week market or a fitted historical base rate).

**M3 — Historical result semantics corrupt rounds.csv (feeds every fit).**
- `legacy.py:272-287` (`rounds_for`) has **no 18-hole completeness check** — a suspended/partial round's `displayValue` is ingested as a full round, and `accumulate_rounds` (`legacy.py:413-426`) dedupes on `(tid, player, round)` so a partial score captured mid-event is **frozen forever** (update.sh runs accumulate daily, including during events).
- `made_cut = rounds >= 3` (`legacy.py:279`) mislabels: WD/DQ after making the cut with an unparseable R3 → "missed cut"; a 36-hole weather-shortened event → the whole field "missed the cut" (poisoning the cut calibration); no-cut 54-hole events are fine by accident.
- `_actuals` (`validate.py:71-93`) ranks players by summed `score_to_par` regardless of rounds played: a DQ-after-R3 player's 3-round total competes against 4-round totals, biasing their finish toward the middle (missing round ≈ par), contaminating y_top20/y_top10.
- Round dates are `start + (round−1)` (`legacy.py:283`) — Monday finishes and delays are misdated (minor `asof` boundary effects).

**M4 — Tie/settlement rules are recorded but not used, and a drifted duplicate pricer ships different numbers.**
- Tournament matchups: sim tie mass is excluded from `p_model` and EV treats a tie as a full loss (`edge.py:519-526`) even though the Bovada rule is recorded as `push_tie` — EV systematically understated (biases *which* matchups look bettable).
- Round groups: `round_pricer.py:179-183` always applies dead-heat splitting regardless of the row's `settlement_rule`; for a push-rule 2-ball at odds < 2 this understates, at odds > 2 overstates.
- `price_threeballs_r1.py` is a live drifted duplicate: EV ignores dead-heat entirely (`ev = pm*odds − 1`, line 108), and its defaults are `--course "Shinnecock Hills"` with `--major` **defaulting to True** (lines 46-47) — anyone running it for a normal event applies major sigma and a wrong course fit silently.

**M5 — `portfolio.py` is not simultaneous Kelly.** It brakes and proportionally scales summed independent quarter-Kelly stakes to per-player (10%) and weekly (40%) caps (`portfolio.py:38-86`). There is no joint return distribution, no handling of mutually-exclusive outrights (20 individually +EV win bets can still take the full 40% on one event), no weather-wave/group correlation notion (mandatory scenario 15), and the caps/brake constants carry no empirical support in the repo. The README's "simultaneous-Kelly" description overstates it.

**M6 — CLV tracking is dead code, and its math is inconsistent anyway.** No caller of `snapshot_fair`/`clv_pct`/`clv_report` exists anywhere in the pipeline (grep: only `market.py` itself); `odds_history.csv` does not exist. If it were used: `clv_pct` compares vigged bet odds to a *de-vigged* close (`market.py:204-210`), a biased definition; `closing_fair` with the default empty `event` matches across events (`market.py:190-201`); `snapshot_fair` normalizes *any* market's board to sum 1 — including top-N/cut boards where that is meaningless (`market.py:162-171`). "The latest snapshot before settlement is the close" is asserted, never enforced as pre-start.

**M7 — Player-identity is name-string identity, with three inconsistent normalizers.**
- Demonstrated: two distinct fitted players folding to one key ("Byeong Hun An" / "Byeong-Hun An") silently collapse to a single entry in `_folded_index` (`model.py:704-712`) — last one wins, no warning. `store._pid` (`store.py:228`) is lowercase-name-only, so two different golfers with one name merge in SQLite, and `accumulate_rounds`' `(tid, player, round)` key drops the second same-named player in an event.
- Three normalizers disagree: `model._fold_name` (transliterates ø/æ, strips punctuation), `refresh._norm_name` (no transliteration — tee-time overrides for "Højgaard"-class names written one way silently fail to match, so their weather/wave adjustment is dropped), `round_pricer._norm_name` (casefold only).
- Unresolved field players are silently assigned the 20th-percentile default skill (`model.py:1013-1014`); the thin-sample guard blocks *stakes* (`engine.py:466-478`) but the wrong probabilities still populate every priced row and the card's forecast tables.

**M8 — `weekly_report --edge` reports a fresh edge run but tabulates a stale file.** `_run_optional_steps` calls `cmd_edge` (which computes and does **not** persist) and only records a note (`weekly_report.py:127-137`), then the report body reads the old `edge_report.csv` (`weekly_report.py:72`). Only `season.build_card` writes the report CSV (`season.py:206-207`). The weekly report can therefore show last event's bets under this event's header, with a note implying they're current.

**M9 — Partial outright boards between the 1.10 threshold and the true overround are normalized to 1.** `devig_outright` (`market.py:110-122`) treats implied-sum ≥ 1.10 as complete; a top-N subset of a heavily-margined board summing to e.g. 1.2 (true book overround 1.5+) gets its fair probs inflated ~25% for every listed name, and those feed the win-market blend (`edge.py:479-480, 502`). `OUTRIGHT_MARGIN = 1.30` and the `LINE_MARGIN` table are hand-set constants with no fitting or source.

### Low

**L1** — `espn._events_for` (`espn.py:128-143`) falls back to the *featured* event when the requested id isn't in the payload — a pinned event request can silently return a different same-week tournament's field/leaderboard.
**L2** — `espn._to_par` (`espn.py:373-381`) maps unparseable score strings to 0.0 (even par); a leaderboard rendering quirk becomes a real score in live snapshots.
**L3** — No CSV/JSON write in the pipeline is atomic (e.g. `store.export_field_csv:479`, `write_threeballs_csv`, `write_predictions`); an interrupt leaves a truncated but well-formed-looking file that every reader accepts.
**L4** — Weather round mapping assumes rounds are consecutive forecast dates from `start_date` (`weather.py:161-178`); any delay shifts wave penalties onto the wrong rounds. Weather is also structurally untestable in the walk-forward (field names carry no tee times), so the "weather" ablation is a no-op that reports sim noise as the feature's value.
**L5** — Scraped player/outcome names are written verbatim into CSVs (`=`, `+` prefixes not escaped) — spreadsheet formula-injection risk on manual inspection of e.g. `pgatour_stats.csv`.
**L6** — Legacy `evaluate_h2h` (`edge.py:301-345`) applies each pair's probability to *every* row holding `h2h_a_odds` (odds not keyed to the pair) — broken, but unreachable from the v2 CLI/engine; should be deleted, not fixed.

---

## 2. Most likely ways the system is fooling its operator

1. **Backtests that saw the future.** Every reported validation number was produced with today's `pgatour_stats.csv`, `global_player_priors.csv`, and `course_features.csv` inside "historical" fits (C2; demonstrated — 151 current priors in a 2023 fit). The players this helps most (thin history, LIV/international) are exactly the ones the operator will believe the model "found early".
2. **Selection and reporting on the same events.** The win-corr sweep picks its winner on the post-2025 split it then reports (margin 2.5e-4); config tuning reuses that split every run at noise-level sim counts; the gate baseline auto-ratchets to the best noisy run (M1, H1, `validate.py:378-396, 574-577`). The recorded metrics are approximately the max of repeated draws, not expected performance.
3. **Brier skill is presented where betting ROI is needed.** No bet-level backtest exists anywhere: calibration + blend + Kelly + caps — the layer that actually loses money — has never been evaluated; CLV tracking is uncalled dead code with no odds history (M6). Meanwhile tied/playoff winners are counted twice in the labels and paid in full in the sim (H3), so precisely the markets with dead-heat settlement look better than they can pay.
4. **Edges manufactured against stale or mis-parsed boards.** Untagged `odds.csv` survives event transitions with no guard pre-tournament; mtime "freshness" blesses copied/touched files; the paste parser can shift every price one player down and still pass all guards (H5, C3). A score-aware model against last week's prices reads as a huge, confident edge.
5. **Green dashboards that cannot turn red.** Four of five test files pass under pytest regardless of their check results (H6); `update.sh` swallows the gate exit code after already refitting (H1); refresh degrades every provider failure to a warning that the card renders as a footnote (`season.py:402-424`). The operator sees "40 passed", "gate", and a tidy card either way.

---

## 3. Adversarial scenario matrix

| # | Scenario | Verdict | Evidence |
|---|----------|---------|----------|
| 1 | ESPN refresh succeeds for field, fails for leaderboard | **fails** | Exception path `refresh.py:163-164` preserves stale `live_state.json`; `_live_state` (`engine.py:103-148`) never checks the event; name-overlap match prices new field on old scores (C1). |
| 2 | Yesterday's odds file copied/touched after today's refresh | **fails** | `_board_fresh` (`engine.py:87-99`) and paste guard (`refresh.py:320-335`) are mtime-only; touched old content passes. `test_golf_inplay.py:117-131` tests exactly this mechanism and encodes the mtime assumption. |
| 3 | Consecutive events share 80% of field; board has blank/wrong event tag | **fails (outrights), survives (matchups/3-balls)** | Wrong/blank *board* tag → refused (`engine.py:380-399`, `521-538`). But `odds.csv` is written untagged (`bovada.py:241-247`, confirmed in the live file) and has no pre-tournament guard at all; and a blank `field.csv` event disables the guard entirely (`engine.py:386`). |
| 4 | Two distinct golfers normalize to the same folded name | **fails** | Demonstrated: `_folded_index` keeps one silently (`model.py:704-712`); `store._pid` is name-only (`store.py:228`); `accumulate_rounds` key drops the second same-named player (`legacy.py:413`). No detection anywhere. |
| 5 | WD after one hole; DQ after making the cut | **fails (historical), survives (live)** | Live: `_is_out` catches WD/DQ statuses (`espn.py:384-387`). Historical: partial rounds ingest as full rounds with no 18-hole check and freeze via dedupe (`legacy.py:272-287, 413-426`); DQ-after-cut with unparseable R3 → `made_cut=0` (`legacy.py:279`); `_actuals` ranks 3-round totals against 4-round totals (`validate.py:80-88`). |
| 6 | Limited-field no-cut event run with default cut rule | **fails** | Demonstrated: 72-player no-cut field under defaults → `made_cut 0.904` per player (below the 0.99 guard) and `cut_binds=True`; `no_cut` is stored but read by nothing (`engine.py:259,457`; `store.py:114`); ESPN in-play applies a synthetic 65-cut (`espn.py:289-296`). |
| 7 | Event shortened to 54 holes after pricing | **fails** | `TOTAL_ROUNDS = 4` is fixed (`simulate_inplay.py:32`); after R3 of a shortened event the engine still simulates a phantom R4 (`engine.py:141` only stops at `rounds_done >= 4`). Historically a 36-hole completion marks the entire field `made_cut=0` (`legacy.py:279`). |
| 8 | Three-way tie for first; top-5 boundary tie | **fails (economically)** | Each tied winner counts in full (`simulate.py:120-126, 344-352`); competition ranking gives all boundary-tied players full top-N credit; `_actuals` labels playoff losers `y_win=1` (demonstrated: sum y_win = 2). Bookmaker dead-heat/playoff settlement pays less than the sim's indicator. |
| 9 | Both matchup players miss the cut with unequal 36-hole scores | **survives** | `rank_score = 1e6 + r36` (`simulate.py:313`) settles by better 36-hole score, matching standard book rules; in-play, groups naming a cut player are dropped rather than sim-priced (`simulate_inplay.py:123-129`). |
| 10 | 3-ball board omits its tie price or has only two valid selections | **fails** | Tie price appears as a 4th "player" → group silently discarded (`odds_manual.py:280`). Two-valid-selection case demonstrated: parses as a 2-ball with odds shifted onto the wrong players, passes all guards, gets priced (C3). |
| 11 | Historical walk-forward event uses today's stats/priors/course/weather files | **fails** | Demonstrated: 151 current public-stat priors inside `fit(asof=2023-06-01)`; `predict_field` reads current global-prior and course-feature files (C2). Weather alone is inert in the backtest (no tee times) — which also makes its ablation meaningless. |
| 12 | Calibration and sim hyperparameters selected and reported on the same post-2025 events | **fails** | `sweep_win_corr` selects by post-split score (`validate.py:378-396`; `sim_config.json` records the 2.5e-4 win); `tune_config` reuses the same split each run; `update.sh` refits calibration daily on the full `validation_predictions.csv` whose window overlaps every reported claim. |
| 13 | Pre-tournament vs in-play at zero holes completed | **fails** | Demonstrated with identical fields, 120k sims: top-5 0.753 (pre, corr-t regime) vs 0.824 (in-play, independent normals); the engines are different models (H4). |
| 14 | Win probability > top-5 after separate win draw, calibration, and blend | **fails** | Demonstrated: calibrated (0.100, 0.101) → blended (0.153, 0.105). The nesting guard runs before the blend; blend weights differ per market (`edge.py:499-508`). |
| 15 | Twenty +EV bets correlated to one golfer or weather wave | **fails (bounded)** | Per-player 10% and weekly 40% caps bound the loss (`portfolio.py:17-18`), but there is no joint Kelly, no mutual-exclusivity handling, and no wave/group correlation notion — 20 correlated bets can still take the full 40% at proportionally scaled, not jointly optimal, sizes. |

---

## 4. Test and tooling results

Environment: sandboxed Linux, Python 3.10. Installed into the sandbox to run the suite: `scipy`, `scikit-learn`, `pytest`, `fastapi`, `uvicorn` (none of this touched the repo).

Commands and results:

```
python3 -m pytest -q test_golf_app_wrapper.py test_golf_config.py \
  test_golf_free_sources.py test_golf_inplay.py test_golf_weekly_report.py
→ 40 passed in 4.32s

python3 -m pytest -q test_engines_contract.py test_security.py test_provenance.py
→ 14 passed in 0.67s
```

Caveat on the 40-passed number: only `test_golf_inplay.py` contains assertions; the other four files use a print-only `check()` helper and **cannot fail under pytest** (H6), so "40 passed" verifies little beyond import-and-run.

Targeted reproductions (temporary scripts under `/tmp/golfrev/`, run against the repo read-only):

- `repro_parser.py` — paste-parser fuzz: promo line shifts pairings; missing odds → mispriced 2-ball; wrapped name → phantom players; tie line → silent group drop. All four confirmed (§C3, scenario 10).
- `repro_nesting.py` — post-blend nesting violation: win 0.1531 > top5 0.1050 from calibrated-nested inputs (scenario 14).
- `repro_boundary.py` — pre vs in-play at zero conditioning: top-5 0.7532 vs 0.8238, top-10/20 similar (scenario 13).
- `repro_actuals.py` — playoff labels: two `y_win=1` in one event (scenario 8 / H3).
- `repro_cutbind.py` — 72-player no-cut event with defaults: `made_cut 0.904`, `cut_binds=True` (scenario 6).
- `repro_leak.py` / `leak_mag2.py` — `fit(asof="2023-06-01")` contains 151 current-season public-stat priors; `predict_field` reads current global-prior/course files; removing today's files from a June-2024 refit shifts 61/156 field ratings (mean |Δ| 0.045, max 0.39 SG/round) (scenario 11).
- fold-collision check — two distinct fitted names collapse to one folded index entry; `store._pid` name-only (scenario 4).
- `repro_devig2.py` — partial-board devig behavior at the 1.10 threshold (M9).

Limitations: no live network calls were made (providers untested end-to-end); the leak's magnitude was quantified at the rating level for one event (above), but a full walk-forward Brier A/B across all ~25 recent events did not fit the session's compute budget — the Brier-scale impact of C2 remains to be measured after the fix; standalone `python3 test_*.py` executions (the mode where `check()` failures count) were not all completed.

---

## 5. Prioritized remediation

Must-fix correctness defects:

1. **Event-identity check on in-play state** (C1): compare `live_state.json` event to the current field event in `_live_state`; clear in-play artifacts on leaderboard-read failure. Largest single live-money risk.
2. **Point-in-time discipline in walk-forward** (C2): thread an `asof` into `fit`/`predict_field` file loads (or default the public-stat/global/course features off in `walk_forward`), then re-run validation, tuning, and calibration and re-baseline. Until then, treat all reported skill as unverified.
3. **Strict paste parsing** (C3): refuse groups whose parsed player count mismatches the header; alternating name/odds only; surface refusals. Prevents mispriced stakes from routine paste noise.
4. **Make the gate gate** (H1): validate before refit (or auto-restore `model_params.json.bak`), propagate exit codes in `update.sh`, freeze the baseline.
5. **Wire `no_cut`/`cut_rule` end-to-end** (H2), including `completed_round_scores`.
6. **Tie/dead-heat consistency** (H3): fractional credit for ties in sim and `_actuals`; enforce nesting after the blend; use recorded `settlement_rule` in matchup/round EV (M4); delete or fix `price_threeballs_r1.py` and legacy `evaluate_h2h`.
7. **Unify the in-play simulator with the validated scoring regime** (H4) and stop applying pre-tournament calibration to it until it has its own evidence.
8. **Content-based board provenance** (H5): event tag + captured-at inside `odds.csv`/all boards; refuse untagged; drop mtime as the freshness signal.
9. **Convert `check()` tests to asserts** (H6) and add the regression tests named above (stale-state routing, parser refusal, nesting property, no-cut event).

Experiments needed to validate methodology (not code fixes):

10. **Nested temporal holdout + bet-level backtest**: tune on pre-2025, select on 2025, report once on an untouched 2026 window; simulate the full calibrate→blend→Kelly→caps pipeline against recorded odds with dead-heat/playoff settlement before believing any EV. Start recording `odds_history.csv` (the CLV code is currently never called) so closing-line evidence can accumulate.

---

## 6. Residual uncertainty

What cannot be established from the repository alone:

- **True out-of-sample performance.** No point-in-time snapshots of `pgatour_stats.csv`, OWGR, global priors, or course features exist, so the honest historical backtest cannot be reconstructed from this repo — only approximated by disabling those features.
- **Bookmaker settlement terms.** Dead-heat vs push rules per book/market, each-way terms, and rule-4-style deductions are assumed, not documented; several EV formulas hinge on them.
- **Executable prices and limits.** `odds_history.csv` was never written; there are no bet timestamps, no evidence any recorded price was available at stakeable size, and no slippage model. The gap between simulated Brier skill and realizable ROI is unmeasured.
- **ESPN feed semantics.** Whether `displayValue` is always round-to-par (vs strokes) across tours/years, and how partial/suspended rounds render, cannot be verified offline; both directly affect `rounds.csv` integrity (M3, L2).
- **Provenance of the existing `rounds.csv`** (143,379 rows): whether any rows were captured mid-event (and thus frozen partial scores) is not determinable after the fact, because the dedupe key erases the distinction.
- **Bankroll inputs.** `bankroll.json`/peak provenance and whether the drawdown brake has ever engaged are outside the repo's evidence.
