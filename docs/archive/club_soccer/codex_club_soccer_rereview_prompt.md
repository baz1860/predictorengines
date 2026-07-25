# Codex Prompt: Verify Fixes + Full Re-Review of `club_soccer/`

Copy everything below the line into Codex.

---

You previously performed an adversarial review of the `club_soccer/` module and found 8 critical, 7 high, and several medium issues. Four of the mechanical criticals have since been fixed by another agent. Your job now is twofold: (1) adversarially verify those fixes — assume they may be wrong, incomplete, or may have introduced regressions — and (2) re-assess the whole module, because the remaining findings from your first review are still open and the fixes may have shifted behavior elsewhere.

Be harsh. No praise, no filler. Every finding needs `file:line`, the offending code, why it's wrong, and a concrete fix.

## Part 1 — Adversarial verification of the four fixes

### Fix 1: Per-bookmaker de-vig (`edge.py`, `rows_from_odds`)
What changed: odds now get a `bookmaker` column defaulted to `"manual"` when absent; each book is de-vigged independently and only if it quotes the market's complete outcome set (`MARKETS` sides); output is one row per outcome — best executable price across books, `p_book` = mean of per-book de-vigged probabilities.
Attack it:
- Feed it multi-book frames with duplicate sides within one book, NaN odds, odds ≤ 1.0, mixed-case side strings, and totals with lines other than 2.5. Does anything slip through or get double-staked?
- Is `p_book` as the cross-book mean the right consensus? Taking the *best* price but a *mean* probability inflates EV at the best book by construction (best-price bias / line-shopping EV). Should the consensus exclude the executed book, use median, or use a sharp-book anchor? Quantify the bias on a realistic two-book example.
- The complete-set requirement now rejects manual odds files that quote only two of three 1X2 sides. Confirm this is enforced and check whether any existing data files (`data/odds.csv`, snapshot history) or callers (`snapshot_odds.py`, `backtest_market.py`, `fit_market_blend.py`, `market_model.py`, the app adapter in `app/engines/club_soccer.py`) relied on the old behavior — including anything that consumed multiple rows per outcome from `edge_report.csv`.
- `drop_duplicates(subset=["side"])` after `sort_values("odds", ascending=False)` keeps each book's *highest* price per side — is best-within-book correct for de-vigging, or should a book quoting the same side twice be rejected as malformed?

### Fix 2: Stale-odds fallback (`season.py`)
What changed: `_future_odds_only()` filters quotes to fixtures dated today or later; manual `odds.csv` is now opt-in via `--allow-manual-odds`, age-limited to `MANUAL_ODDS_MAX_AGE_DAYS = 2.0` (file mtime), and future-filtered; `_backed_bets_section` additionally drops any row dated before today and displays a `pricing_note` explaining the pricing source or failure.
Attack it:
- Date-only granularity: a fixture kicking off today at 12:00 passed at 23:00 still prices as "future". Is intraday staleness acceptable, or should kickoff timestamps be used where available?
- File mtime as freshness proxy: `touch odds.csv` defeats the age gate. Is a quote timestamp column needed?
- Timezone: `pd.Timestamp(datetime.now(timezone.utc).date())` vs naive `date` column parsing — any off-by-one at UTC day boundaries for a user west of UTC?
- The BSD path still swallows exceptions via `_step`. The card now *says* pricing failed instead of silently staking stale prices — but should a pricing failure make the scheduled run exit nonzero? Check `update.sh` interaction.
- Regression check: `--no-network` runs, the app adapter, and `test_card_written` — do they still produce a card?

### Fix 3: Blank-line parsing (`edge.py`)
What changed: `raw_line` is coerced with `pd.to_numeric(..., errors="coerce")` before `float()`.
Attack it: other `float(...)` calls on API-shaped strings elsewhere in `edge.py`, `snapshot_odds.py`, and the fetchers. The original crash was one instance of a pattern — enumerate the remaining instances.

### Fix 4: Test harness (`test_club_soccer.py`)
What changed: `check()` raises `AssertionError` under pytest (via `PYTEST_CURRENT_TEST`), backed by an autouse fixture asserting no new `_fails` per test; script mode still collects and exits nonzero at the end.
Attack it:
- Do any existing checks now fail that were silently failing before? Run `python3 -m pytest test_club_soccer.py -q` and report every failure — these are pre-existing bugs the false-green harness was hiding.
- The immediate raise means later checks in the same test function no longer execute under pytest — is coverage materially reduced vs. the fixture-only approach?
- `tests/` directory and other `test_*.py` files at repo root: do any use the same false-green `check()` pattern? Fix-worthy instances elsewhere are in scope.

## Part 2 — Full module re-assessment

Re-run your complete adversarial review of `club_soccer/` (code quality and model/statistical quality, as before). Specifically re-test and update the status of every finding from the first review, with emphasis on the ones NOT yet addressed:

1. **Team identity fragmentation** — `Bayern Munich` vs `FC Bayern München` etc. still splitting histories? Check ingestion in `fetch.py`, the `names.py` set-ordering nondeterminism, and whether `identities.py`'s accent/case normalization still reports false health.
2. **Context-artifact leakage in walk-forward validation** — does `validate.py` still load the current full-history context artifact when predicting historical matches?
3. **Backtest at closing odds** — `backtest_market.py` still selecting on `p_model − p_close` and executing at `odds_close`?
4. **No evidence gate on betting output** — negative ROI/CLV backtests, market-blend weight 0, yet `season.py` still stakes every internally positive edge. Given the new pricing-note plumbing, propose (or implement as a recommendation) a concrete evidence gate: what metrics, what thresholds, what sample sizes, where it hooks in.
5. **Cross-league strength identification** — competition adjustment still applied symmetrically; Elo still tier-blind at initialization.
6. **Fail-open pipeline** — `_step` still swallows all exceptions; `update.sh` still converts failures to informational output.
7. **Postponed/void fixtures retaining scores** — `fetch.py` merge semantics and `model.py`'s score-presence "played" definition.
8. **Future rest/congestion ignoring scheduled fixtures**; **player/team fuzzy-matching guessing**; **ensemble promotion without untouched holdout**; **±0.01 Brier gate**; and the medium items (provider ID namespacing, seeder overwrite risk, openfootball season derivation, overdispersion, correlation reset after adjustments, corners `.notna().all()`, no portfolio exposure limits, duplicated provider logic).

Also hunt for anything NEW: the four fixes touched pricing, card assembly, and the harness — look for fresh bugs adjacent to those edits, and for any behavior the old (buggy) downstream consumers depended on.

## Output format

- **Fix verdicts first**: for each of the four fixes — VERIFIED / INCOMPLETE / WRONG / REGRESSION, with evidence and reproductions.
- Then findings ranked by severity (Critical → High → Medium → Low), each with `file:line`, code, impact, fix.
- Updated "ways this system fools its operator" list.
- A prioritized remediation list (max 10), noting which first-review items remain open.
- Run `python3 -m pytest test_club_soccer.py -q`, `python3 test_club_soccer.py`, `python3 -m compileall club_soccer`, and your own targeted reproductions before writing conclusions. Report every command's outcome. Do not modify files.
