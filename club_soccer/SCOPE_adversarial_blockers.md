# Scope — the three deferred adversarial blockers

**Status:** Scoping, for review before build.
**Source:** Codex adversarial review, findings 2, 3/9, 5. These were left unfixed in the
remediation pass because each is architecture-level, not a patch. This document scopes
all three and proposes a build order.

The connective tissue: all three are about **evidence integrity**. None corrupts live
pricing today. Each makes a *claim about the model's edge* untrustworthy — and the whole
point of the staking gate is to only ever bet on trustworthy evidence. Building these in
the wrong order wastes calendar time, so order matters as much as content.

---

## Blocker A (finding 2) — league-seed validation leaks future coefficients *(DONE)* ✅

**Built.** `uefa_coefficients_history.json` holds per-year snapshots (2021–2026, all 55
associations, each scraped from its own `crankYYYY` page with a `published_on` date).
`strength_prior(country, tier, as_of=)` selects the latest snapshot published on or
before `as_of`, anchored *dynamically* on England=1.00 / Scotland=0.58 within that
snapshot (scale-invariant — the source rescales its absolute values between captures).
`model.fit(coef_as_of=)` and `walk_forward` (per-fold cutoff) thread it through; the
history file is in the cache fingerprint.

**Re-validated cold, leak-free (2024-06 onward, n=27,896):**

| | Brier | Log-loss |
|---|---|---|
| seed OFF | 0.61468 | 1.02472 |
| seed ON | 0.61279 | 1.02207 |
| delta | **−0.00189** | −0.00265 |

The leak-free delta (−0.00189) is essentially the contaminated one (−0.00198) — **the
leak was not the source of the benefit.** Seeding genuinely helps, so the promotion
**stands, now on honest evidence** (`league_seed_evidence.json` → `revalidation_leak_free`,
with provenance hashes). Full-history gate still PASS (Brier 0.6124). Original scope
preserved below.

---

### Original scope (for the record)

## Blocker A (finding 2) — league-seed validation leaks future coefficients

### What's wrong
`uefa_coefficients.json` is a single artifact: the 2025 five-season ranking. Every
walk-forward fold reads it, so a 2022 fold is seeded with coefficients that embed
2023–2025 European results. The seeded arm gets a faint signal from the future; the
unseeded arm does not. The −0.002 Brier that justified promoting `LEAGUE_SEED_DEFAULT`
is therefore measured on contaminated evidence.

### What's NOT wrong (important)
Live production pricing is correct. Today's card uses today's coefficients, which is
exactly right — there is no future to leak from at the live edge. The leak exists **only
in the historical validation**. So this is not a "stop the presses" data-corruption bug;
it is "the promotion decision rests on an overstated number."

### The fix
Time-versioned coefficient snapshots. Confirmed feasible: kassiesa publishes a ranking
page per year (`crankYYYY.html`, verified 2021–2026 all return 200). Build:

1. `uefa_coefficients_history.json` — `{publication_year: {country: coefficient}}` for
   2021–2026, each scraped from its own `crankYYYY` page, with a `published_on` date per
   snapshot (UEFA coefficients publish end of season, ~June).
2. `strength_prior(country, tier, as_of=None)` — when `as_of` is a date, use the most
   recent snapshot whose `published_on <= as_of`; else the latest (production behaviour,
   unchanged).
3. `model.fit(..., coef_as_of=None)` and the walk-forward: each fold passes its training
   cutoff as `as_of`, so a 2022 fold seeds from the coefficients that existed in 2022.
4. Re-run the seeded-vs-unseeded A/B cold, with time-correct priors. **Re-promote only if
   it still wins.** If it doesn't, `LEAGUE_SEED_DEFAULT` goes back to `False` and the
   card/model lose nothing they didn't earn.

### Acceptance
- No fold reads a coefficient snapshot published after its training cutoff (assert in a
  test, not by eye).
- The A/B artifact carries the fixture/code/prior hashes it currently lacks, so the
  result is reproducible.
- A clear yes/no on re-promotion, recorded in `league_seed_evidence.json`.

### Cost / risk
Medium. The scrape is 6 pages. The `as_of` plumbing touches `strength_prior`, the ELO
seed in `model.fit`, and `walk_forward`. Risk: the cache fingerprint must now include the
history file (already added `uefa_coefficients.json`; add the history file too). Low risk
of production regression — worst case seeding is de-promoted and we return to a known-good
state.

---

## Blocker B (findings 3 + 9) — the decision-time backtest uses hindsight identity and a synthetic price *(DONE — ships empty, accumulates forward)* ✅

**Built.** `decision_ledger.py` — an append-only ledger written at snapshot time:

- **`record()`** freezes one immutable row per (fixture, market, side) for fixtures in the
  60–180 min pre-kickoff window: provider fixture id, kickoff, raw + resolved names,
  `resolver_version` (hash of alias map + registry), the ONE executable quote
  (`select_executable_quote` picks the best-priced book that quotes the *complete* market
  — never a median), that book's own de-vig, `p_model`, per-row `decision_lead_min`, and
  model/code/prior hashes. Idempotent per decision id; re-runs never rewrite a row.
- **`settle()`** appends results + Pinnacle-close CLV later, keyed on fixture id. The
  decision is never touched.
- **`decision_time_backtest.build_bets()`** now reads ONLY the joined ledger. The
  reconstruction path is retained as `build_bets_reconstructed` (deprecated, unused).
- Wired into the daily pipeline: record → settle → backtest.

**Acceptance met:**
- The backtest reads only frozen rows — `build_bets` calls neither `canonical_name()`
  nor the alias-map file (asserted). Deleting `club_alias_map.json` cannot change a
  historical metric.
- Every bet carries one named `book` and `odds_executed`; no synthetic median anywhere in
  selection or settlement.
- `decision_lead_min` is per-row, and the window (`MIN_LEAD_MIN=60`, `MAX_LEAD_MIN=180`)
  sits inside the gate's `[60, 10080]`.

**Ships empty and accumulates forward** — a decision cannot be recorded after kickoff, so
the ledger fills only from live runs; the gate stays correctly closed until ~1,000 settled
decisions exist (a season). The run ledger's bets-toward-1,000 counter tracks it. This is
the calendar-bound cost the scope flagged; the point is that accumulation now begins on
trustworthy, executable, frozen records. Original scope preserved below.

---

### Original scope (for the record)

## Blocker B (findings 3 + 9) — the decision-time backtest uses hindsight identity and a synthetic price

This is the big one, and Codex's "one change before Phase B." Two defects, one fix.

### What's wrong
1. **Hindsight identity (finding 3).** `decision_time_backtest.build_bets` resolves past
   fixtures with *today's* `club_alias_map.json` and `club_registry`. Six of the eleven
   settled fixtures only match because of aliases created after the snapshots were taken.
   Sample inclusion, bet counts and ROI all change with the resolver version (11 fixtures
   with the current map, 5 with an empty one). The replay is not measuring what a bettor
   could have known at decision time.
2. **Synthetic price (finding 9).** `snapshot_odds.py` stores only `odds_median` across an
   unknown, possibly-differing set of bookmakers per side. `_devig` then builds a market
   no single book offered and selects against it. Phase B would measure profitability
   against non-executable prices.

### Why patching in place doesn't work
Both defects come from **reconstructing** the past from mutable current state. You cannot
fix reconstruction; you have to stop reconstructing. The decision that mattered — which
club, which league, which executable price, which model probability — has to be **frozen
at the moment it was made**.

### The fix — an append-only decision ledger
Written at snapshot time (in the daily run, when the fixture is still upcoming), never
rewritten:

```
decision_ledger.csv  (append-only, one row per priced side)
  decision_ts            when the row was written (>=60min pre-kickoff)
  provider_fixture_id    stable BSD/provider id — the identity anchor
  kickoff_utc
  competition            as known at decision time
  raw_home, raw_away     provider spellings, frozen
  club_id_home, club_id_away   resolved stable ids AT THIS TIME
  resolver_version       hash of the alias map + registry used
  market, side
  book                   the actual bookmaker selected
  book_market_complete   that book's full market (all sides), for an honest de-vig
  odds_executed          one real, simultaneously-available quote
  decision_lead_min      the individual lead (not a pooled median)
  p_model                model probability at decision time
  train_cutoff           data the model was fit on
  model_hash, code_hash, prior_hash
```

Settlement is a **separate append**, later, keyed on `provider_fixture_id`: result,
win/loss, Pinnacle closing price, CLV. The decision is never touched.

The backtest then reads the ledger — pure frozen records — and computes the same
confidence/threshold/gate metrics it does now, but on data that is honest about what was
knowable when.

### Snapshot changes required (feeds the ledger)
- `snapshot_odds.py` must store **per-bookmaker** quotes, not a median: `book`,
  `side`, `odds`, plus the fixture's `provider_fixture_id` and `kickoff_utc`.
- Capture at a disciplined lead (the 60–180 min pre-kickoff window), so `decision_lead`
  is real and within gate bounds.

### Acceptance
- The backtest reads only the ledger; deleting `club_alias_map.json` does not change any
  historical metric (test: run twice with the map present and absent, assert identical).
- Every bet has one named executable `book` and `odds_executed`; no synthetic median
  anywhere in the selection or settlement path.
- `decision_lead_min` is per-row and every row is in `[60, 10080]`.

### Cost / risk
High — this is the largest of the three, and it is mostly **new infrastructure plus
calendar time**. The engine rewrite is a few sessions; but the ledger only fills forward
(same constraint as the current backtest), so useful volume is still a season away. The
value of building it now is that the accumulation, when it starts, is trustworthy — which
is the entire point.

---

## Blocker C (finding 5) — totals permanently veto the whole gate *(DONE)* ✅

**Built.** Closing totals sourced (`PC>2.5`/`PC<2.5`, Pinnacle closing, ~85% coverage
across the 12 European domestic leagues) and wired into the backtest's totals CLV. The
gate is now per-market: `evaluate()` returns `{markets: {mkey: {active, open}}}`,
`market_staking_allowed()` exposes it, and `edge.apply_evidence_gate` zeroes stakes per
market. A failing/inactive OU2.5 no longer vetoes a passing 1X2; BTTS is never stakeable
(no CLV reference). Verified: passing-1X2 + empty-OU2.5 opens 1X2 only; the money-safety
invariant fires per market. Non-UEFA totals have no closing feed, so they stay
information-only, correctly. Original scope preserved below for the record.

---

### Original scope (for the record)

## Blocker C (finding 5) — totals permanently veto the whole gate

### What's wrong
`market_history.csv` has no closing-totals feed, so OU2.5 CLV is always `None`. The gate
requires **every** market with evidence to pass (global AND), so a market that *cannot* be
CLV-scored permanently blocks staking on *every* market — including a 1X2 book that has
earned it. The artifact note (now corrected) previously implied volume alone would open
it; it never could.

### The fix — per-market activation
1. `evidence_gate.evaluate()` already computes `market_pass[mkey]`. Return it, rather than
   collapsing to a single global `allowed`. A market with **no evidence** (all `n_bets =
   0`) is neither pass nor veto — it is simply inactive.
2. `edge.apply_evidence_gate` zeroes stakes per market: a row's stake survives only if its
   own market's gate is open. A 1X2 pass never unlocks OU2.5 and vice versa.
3. Totals stay permanently inactive until a closing-totals source exists — which is
   correct and now explicit, not an accidental global veto.

### The deeper question this raises
Should we add a closing-totals feed at all? Options: (a) accept that totals are
information-only forever and only ever stake 1X2; (b) source closing totals (fd.co.uk
carries `>2.5`/`<2.5` closing for the main leagues in the `mmz4281` files — the
`Avg>2.5`/`AvgC>2.5` columns — so it is obtainable for the UEFA leagues, though not the
non-UEFA ones). This is a product decision, not just an engineering one, and belongs in
the review of this scope.

### Acceptance
- A hand-crafted artifact where 1X2 passes and OU2.5 has zero evidence opens 1X2 staking
  only.
- A hand-crafted artifact where OU2.5 has evidence but fails does not block a passing 1X2.
- The stake-zeroing test proves a non-UEFA totals row is never stakeable.

### Cost / risk
Low–medium. The gate change is small and well-contained; the risk is in the stake-zeroing
path (`apply_evidence_gate`), which is the last line before money, so it needs the hard
per-market invariant test the global version already has.

---

## Build order

The order is forced by dependency and by not wasting calendar time:

1. **C first (per-market gate).** Smallest, unblocks the concept of 1X2-only staking, and
   is a prerequisite for B's evidence to ever matter (no point accumulating totals
   evidence that can't open anything). ~1 session.
2. **A second (coefficient snapshots).** Independent of B, and it settles whether league
   seeding stays promoted — which affects the `model_hash` that B's ledger will freeze. Do
   it before B starts writing frozen model hashes, or every pre-A ledger row is stamped
   with a possibly-de-promoted model. ~1–2 sessions.
3. **B last (decision ledger).** The big build, and the one whose value is calendar-bound.
   Start it only once A has settled the production model, so the accumulation begins on a
   stable, honestly-validated model. Engine ~2–3 sessions; useful volume ~a season.

**Do not start Phase B accumulation on the current replay.** Every day of it is wasted —
the metrics can't survive an alias-map change. The first real accumulation day is the day
B's ledger ships.

## One decision for you before the build
Blocker C asks it directly: **stake 1X2 only forever, or source closing totals for the
UEFA leagues?** The first is simpler and honest about the non-UEFA gap; the second doubles
the stakeable market surface but only for Europe. Everything else in this scope proceeds
the same either way — I only need the answer before wiring C's stake-zeroing.
