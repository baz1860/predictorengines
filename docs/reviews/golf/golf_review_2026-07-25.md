# Golf module — adversarial review, 2026-07-25

Scope: `golf/` at working-tree state (source files last touched 2026-07-19). All claims below were reproduced against the repo's own data and live free endpoints.

---

## 1. Verdict

The core is a sound sparse ridge fit (player skill + tournament-round difficulty, `model.py:491-560`). Almost everything wrapped around it is either broken, inert, or complexity bought to patch a self-inflicted problem.

Three findings are load-bearing:

- **The cut market is trained on corrupt labels.** Already in the data, not a future risk.
- **The `course` column contains tournament names, not courses.** This silently kills the course-features feature entirely and makes "course fit" wrong for every rotating venue.
- **The free data you need is available and you are already downloading it, then throwing it away.**

The two prior review documents (`golf_adversarial_review.md`, `golf_adversarial_rereview.md`) are accurate. Nothing from the re-review's Critical or High list has been fixed — I re-verified N1, N2, N3, M-a and M-b are all still present. That's the single biggest process problem: you are generating reviews faster than fixes.

---

## 2. Data integrity (fix before anything else)

### 2.1 Missed cuts are labelled as made cuts — already in `rounds.csv`

`providers/legacy.py:286`:

```python
out_status = any(k in pstatus for k in ("CUT", "WD", "WITHDR", "DQ", "DISQUAL"))
made_cut = 0 if out_status and len(rounds) < 3 else 1
```

ESPN's **season scoreboard** payload — the one this parser reads — carries no competitor status. Verified against your own caches: status is `None` for all 5,888 competitors in `espn_pga_2025.json`. So `out_status` is always `False` and everyone is labelled made-cut.

Observed damage in `data/rounds.csv`:

| Season | Rows | `made_cut=0` | Events flagged no-cut (all made) |
|---|---|---|---|
| 2022 | 23,158 | 27.3% | — |
| 2023 | 32,721 | 27.5% | 17 / 90 |
| 2024 | 34,234 | 26.3% | 21 / 96 |
| **2025** | 33,198 | **12.6%** | **41 / 93** |
| **2026** | 20,388 | **11.4%** | **23 / 55** |

Roughly half the missed-cut labels for the last two seasons are wrong, and 44% of 2025 events now look like no-cut events to `validate.walk_forward`. Those events get simulated without a cut and scored `p_cut≈1` against `y_cut=1`, so the cut Brier improves as the data degrades, the gate passes, and `calibrate --fit` bakes it in.

Blast radius is narrower than the re-review implies — `fit()` never reads `made_cut`, so skill ratings are unaffected. But the cut market, the cut leg of the validation headline, and the promoted `cut` calibration map are all built on it.

**Fix (free):** the correct endpoint exists and returns real statuses:

```
https://site.web.api.espn.com/apis/site/v2/sports/golf/leaderboard?league=pga&event={id}
```

Verified live on event 401703492 (Farmers 2025): 155 competitors, `STATUS_CUT` × 85, `STATUS_FINISH` × 70. Note the constant both providers currently use — `site.api.espn.com/.../golf/pga/leaderboard` — **404s**. The live in-play path is pointed at a dead URL.

### 2.2 Accumulate rewrites every row, every run

`providers/legacy.py:436`, `merged.get(key) == row`. CSV rows load as strings, parsed rows are typed, so the comparison never matches. Every run replaces the whole history. That is the vehicle by which 2.1 keeps overwriting good 2022–24 labels the moment you extend `--since`. Normalise through `RoundRecord` before comparing.

### 2.3 Sub-1.0 odds silently reprice a 3-ball as a 2-ball

Still open (`providers/odds_manual.py`, parse path vs `od > 1` filter at line ~200). A `0.5`-for-`5.0` typo drops a runner and reprices the group. Reject odds ≤ 1.0 at parse time; make `load_threeballs` refuse a group whose surviving slot count disagrees with its `group_id` market rather than shrinking it.

### 2.4 Local-naive timestamps

`odds_manual._ts()` is `time.strftime` with no offset; `_board_fresh` decodes as UTC. In UK summer time every board reads one hour fresher than it is, so your 30-minute freshness window is really 90 minutes. One-line fix.

---

## 3. The `course` column is not a course

`data/rounds.csv` `course` values are tournament names: `The American Express`, `PGA Championship`, `The Open`, `U.S. Open`. Two consequences.

**`course_features.csv` is dead.** Zero of 399 events match a row in it (I checked after `_fold_name` folding). `_course_arch_adjustment` (`model.py:932-957`) therefore returns `0.0` on every call, every time. That's ~80 lines of model code, a data file, and the `COURSE_ARCH_MAX_ABS` constant delivering nothing.

Separately: the ten rows in `course_features.csv` are hand-invented — every value is a multiple of 0.05. Even if the join worked, you'd be moving ratings by up to ±0.45 strokes on numbers someone typed.

**"Course fit" is really "event fit."** `params["courses"]` has 146 keys, including `The Open` (217 players), `U.S. Open` (210), `PGA Championship` (214). Those events rotate venues annually. You are pooling Portrush, Troon and St Andrews into one "course" and calling the mean residual a course fit. For the fixed-venue events (Augusta, Harbour Town, TPC Sawgrass) it's fine; for the rotating ones it's noise dressed as signal, and majors are exactly where you're pricing outrights.

**Fix (free):** the per-event endpoint above returns `courses[]` with real names, `shotsToPar`, `totalYards`, and per-hole par/yardage. Backfill a `course_id`/`course_name` column, key the fit on that, and drop the hand-made archetype file in favour of measured yardage/par.

---

## 4. Features that compute zero

The shipped `data/model_params.json` (asof 2026-07-20) contains **`public_stat_priors: {}`** — zero entries.

Cause is `model.py:601`: `include_public_stats = asof.date() >= today`. A live fit's `asof` is `max(round date) + 1`, so any refit two or more days after the last round silently drops every prior. That is most weekdays. Consequences:

- `PUBLIC_STAT_BLEND`, `_public_stat_alignment`, `_aligned_public_prior`, `_public_stat_components` (~150 lines) — inert.
- `providers/pgatour_stats.py` (352 lines) and `data/pgatour_stats.csv` (158 KB of genuine SG components) — collected, never used.
- `_course_arch_adjustment` reads `_public_stat_components`, so it is dead twice over.
- Two refits a day apart produce materially different ratings with no warning.

Gate on the stats file's own capture date versus `asof`, not on `asof >= today`.

`_weather_wave_adjustment` is a third: it needs `player.tee_time_rN`, which comes from `tee_times.py` — a **manual paste parser**, because "the reliable free feeds do not always carry tee times." They do. The per-event leaderboard returns `linescores[].teeTime` (`"2025-01-22T17:12Z"`) for every player and every round. 133 lines of paste parser replaced by a field read.

Net: roughly a third of `model.py` and two of five providers currently contribute nothing to a price.

---

## 5. Too complex — cut this

### 5.1 The separate win regime

`sim_config.json` runs the win market at `round_corr=0.0` while place/cut markets run at `0.3`. Two different generative processes over the same tournament means win and top-5 no longer come from one joint distribution, so they aren't nested by construction — which is why `edge.py:459-467` needs a post-blend clamp to stop `p(win) > p(top5)`.

The justification is a screening sweep: win logloss 0.04500 vs 0.04525 at 300–750 sims. That is a 0.5% relative difference, plausibly inside the Monte-Carlo noise of the screen itself. You bought a coherence bug and a clamp for a gain you can't distinguish from zero.

**Simplify:** one regime, `round_corr=0.3`, delete `win_round_corr`/`win_tail_df` and the separate `_win_frac` draw. If you want the win market to be less correlated, model the reason (leaders play in the final group in the same conditions) rather than running a second universe.

### 5.2 The calibration layer

`calibration.json` is 43 KB of isotonic knots, with `win` already demoted to identity. The remaining four markets (`top5/10/20/cut`) are calibrated **independently**, which breaks nesting again — a second cause of the same clamp. You reached the same conclusion in `club_soccer` (calibration added, not recommended). Apply it here: drop the layer, or calibrate one market and derive the rest.

### 5.3 Two ESPN clients

`providers/legacy.py` (490 lines, `EspnProvider`) and `providers/espn.py` (456 lines, `EspnGolfProvider`) both wrap the same free source, with separate `_to_par`, `_course_name`, and status parsers. `providers/__init__.py` re-exports both. The made-cut bug lives in the one you'd expect to be retired. Collapse to one client — and the one that survives should use the per-event endpoint.

### 5.4 Two stores of the same rounds

`data/golf.db` (30 MB, 143,701 rounds) and `data/rounds.csv` (11 MB, 143,700 rounds) hold the same history. `model.load_rounds_df` reads the CSV, so the SQLite copy plus most of `store.py` (620 lines) is a second source of truth to keep in sync with no consumer on the pricing path. Either make the DB authoritative and delete the CSV, or delete the DB and keep `store.py` for `odds_quotes` only.

---

## 6. Too simple — add this (all free)

You already download everything below and discard it.

**6.1 Per-hole scores are in the payloads you cache.** `providers/legacy.py:279-283` reads `ls["linescores"]`, checks `len(holes) >= 18`, and throws the values away. Verified in `espn_pga_2025.json`: 18 hole objects per round with gross `value` and `scoreType`. Combined with per-hole `shotsToPar` from the per-event endpoint, that gives you free, honest proxies for the SG components you can't buy:

- scoring average split by par-3 / par-4 / par-5 (a distance/approach proxy),
- birdie rate vs bogey-avoidance rate (separates aggressive and steady players who share a stroke average),
- blow-up-hole frequency (double+), which is the real driver of the fat tail you currently model with a global `tail_df=6`.

That last one is the biggest modelling upgrade available to you: replace one global `t(6)` with a per-player skewness estimated from their own double-bogey rate. A player's variance is not symmetric, and outright pricing lives in the right tail.

**6.2 Real cut rules and round counts.** `tournament.cutRound / cutScore / cutCount / numberOfRounds / major` are all returned per event. The hardcoded `65 / False / 4` defaults go away, LIV's 54-hole no-cut events stop being phantom-cut, and weather-shortened events stop getting a phantom fourth round.

**6.3 Real tee times** — see §4. Wakes up the weather feature you already wrote.

**6.4 Real course metadata** — par, total yardage, per-hole yardage. Replaces the invented archetype file with measured values, and lets you regress event difficulty on yardage/par rather than asserting it.

**6.5 A note on residual double-counting.** `sigma_p`, `form` and `courses` are all computed from the same `resid` vector with no orthogonalisation (`model.py:558-600`). A hot player gets a rating bump from `form` *and* an inflated "course fit" if the streak happened at one event, *and* a distorted `sigma`. `FORM_WEIGHT=0.7` partly absorbs this by tuning, but you're tuning around a structural overlap rather than removing it. Fit form on residuals net of the course/event term.

---

## 7. Junk

Tracked in git and shouldn't be:

- `golf/threeballs_r1_raw.txt` **and** `golf/threeballs_r1_raw 2.txt` — byte-identical copies of the same June paste, both tracked. A third copy sits in `data/`.
- `golf/US_OPEN_2026_REPORT.md`, `golf/US_OPEN_2026_3BALL_R1.md`, `golf/reports/2026-06-21_*.md` — June output artifacts.
- `golf/data/course_features.csv` — dead per §3.
- `golf/data/tee_times.example.csv` — dead once §6.3 lands.

Untracked litter in `data/`: `rounds.csv.bak` (6.6 MB), `model_params.json.bak` (476 KB), `.sqlite_ioprobe.db`, `.sqlite_ioprobe.db-journal`, `.fuse_hidden0000000400000001`, `_wtest.txt`, `.DS_Store`. Test probes are writing into the production data directory — that's how a `.bak` becomes a source of truth by accident.

Likely superseded: `price_threeballs_r1.py` (156 lines) largely duplicates `round_pricer.py` (307 lines) for the round-1 case.

At repo root: two golf review docs and eight `codex_club_soccer_*` prompt files. Move them to `docs/reviews/` — root-level clutter makes it harder to see that the fixes never landed.

---

## 8. Order of work

1. Repoint every ESPN read to `site.web.api.espn.com/.../golf/leaderboard?league=pga&event={id}`; take status, `tournament.*`, `courses[]`, `teeTime` from it. (Fixes §2.1, §3, §6.2, §6.3, §6.4 and the dead 404 URL at once.)
2. Rebuild `rounds.csv` from scratch with the corrected parser, add `course_name`. Do **not** incrementally accumulate over the corrupt file.
3. Fix the accumulate comparison (§2.2) and prove idempotency: two runs, byte-identical file.
4. Fix `include_public_stats` gating (§4) and re-fit. Confirm `public_stat_priors` is non-empty in the shipped params.
5. Re-run validation. The cut Brier will get **worse**. That is the correct outcome; freeze the new baseline.
6. Only then: collapse the win regime (§5.1), reconsider calibration (§5.2), and add the per-hole features (§6.1).

Steps 1–5 are corrections. Step 6 is the only place new modelling belongs, and it shouldn't start until the ground truth is trustworthy.
