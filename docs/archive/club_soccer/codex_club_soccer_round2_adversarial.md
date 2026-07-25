# Codex Prompt: Round-2 Adversarial Review — Verify the Remediation, Attack the Three New Builds

Copy everything below the line into Codex.

---

Round 2. Your first review found 18 issues across the league expansion, identity layer,
and staking evidence. The author worked them and then built the three deferred blockers
(closing totals + per-market gate, time-versioned coefficients, decision ledger). As
before: assume the self-verification is compromised and the most confident claims hide
the bugs. Several of the "fixes" reversed a guard the author had earlier deleted — check
whether the reversal is correct, or whether it broke something in the other direction.

**Standing rules:** harsh. `file:line` + offending code + breaking input + impact + fix.
Do NOT modify files. Reproduce every number; when you can't, say why. "Looks fine" is
worthless — produce the input that breaks it.

Priorities: (0) the new staking-evidence path, since it is the road to real money;
(1) whether a remediation over-corrected; (2) everything the first round could not reach
because the code has changed under it.

---

## Part 0 — Falsify these seven headline claims

**0.1 — "The decision ledger cannot be corrupted by hindsight; deleting the alias map
changes no historical metric."** (`decision_ledger.py`, `decision_time_backtest.build_bets`)
The ledger is empty, so this property has NEVER run on real data. Attack it structurally
AND with a synthetic ledger:
- `select_executable_quote` picks the book offering the **best price for our side** among
  books quoting the complete market, then de-vigs **that** book. A book that is generous
  on our side is, by construction, skewed on our side — so the de-vig `p_book_devig` (and
  therefore `edge`) is biased in our favour by our own selection. Quantify the bias. Is
  the edge the ledger records systematically overstated?
- `settle()` joins decisions to results on `provider_fixture_id`, looking up
  `fixtures.csv` by `fixture_id`. Prove the BSD event id written at decision time equals
  the `fixture_id` stored in fixtures.csv for the same match. If BSD's event id and our
  stored fixture id ever differ (dedup, re-id, provider change), **every settlement
  silently misses** and the gate never opens no matter how long it accumulates.
- `resolver_version` is frozen per row but nothing READS it back to reject a row resolved
  under a since-changed map. Is it decorative? What is it for if no consumer checks it?

**0.2 — "The gate epsilon hole is closed."** (`evidence_gate.py` lines ~231, ~262)
The author added a `flat_roi_lb95` requirement and a Wilson lower bound on CLV fraction.
- Confirm the produced artifact actually WRITES `flat_roi_lb95` at every threshold — if it
  doesn't, the gate is closed by an absent field, not by statistics, and the "it can open
  after volume" claim is false. Which is it?
- `kelly_roi` still checks only the point estimate — there is no `kelly_roi_lb95`
  requirement while `flat_roi` has one. Asymmetric. Can a kelly-staked strategy open the
  gate on a point estimate the flat path would reject?
- The block-bootstrap `flat_roi_lb95` the artifact writes: with n≈50 the bootstrap is
  meaningless. Confirm the gate can only pass at n≥1000 where it is meaningful, and that a
  hand-crafted artifact with n=1000 and a genuinely positive LB opens exactly the right
  markets and no others.

**0.3 — "League seeding still wins on leak-free evidence, so promotion stands."**
(`uefa_registry.strength_prior(as_of=)`, `model.fit(coef_as_of=)`, `validate.walk_forward`)
Reproduce the A/B cold, cache cleared, and confirm `coef_as_of` actually flows:
- Assert that in a 2022 fold, `strength_prior` returns the 2022 snapshot value, not the
  2026 one. If `coef_as_of` is silently `None` anywhere in the path, the "leak-free"
  −0.00189 is identical to the contaminated −0.00198 by accident, not by fix. Prove the
  number moved because the priors are period-correct.
- Dynamic anchoring divides by `(England_coef − Scotland_coef)` within each snapshot.
  Find the snapshot (or a plausible future one) where England or Scotland is missing, or
  where the two are close, and show whether `strength_prior` returns junk or the
  `DEFAULT_PRIOR` for the WHOLE snapshot.
- The `published_on` dates are hard-coded to `YYYY-06-15`. A fold whose cutoff is
  2023-06-01 gets the 2022 snapshot; a fold at 2023-07-01 gets 2023. Is that boundary
  right, or does it misassign a month of fixtures to the wrong coefficient era?

**0.4 — "Per-market gating means a CLV-less totals market no longer vetoes 1X2."**
(`evidence_gate.evaluate`, `market_staking_allowed`, `edge.apply_evidence_gate`)
- The money-safety invariant now runs per market. Construct the row that should crash it
  (an open-market row that refuses to zero) and one that should NOT (a closed-market row
  correctly zeroed). Confirm BTTS — absent from `_GATE_MARKET` — can never carry a stake.
- `market_active` is "any threshold has n_bets ≥ 1". A market with 1 bet is active and can
  therefore contribute a `reason`/failure. Is a 1-bet market meaningfully different from
  an inactive one, and can a tiny active market produce a confusing gate state?

**0.5 — "Fixture 10034 was the only corrupt row; the reclassifier is now safe."**
(`fetch.reclassify_by_club_country`, `_bsd_to_fixture_row`)
- The author's audit for the mis-weld class excluded Monaco, Wales, Scotland, Northern
  Ireland, Liechtenstein as "legitimate cross-border". Prove that exclusion did not also
  hide a real mis-weld. Re-run the audit WITHOUT the exclusions and inspect each.
- Canonicalization now scopes by the COMPETITION country. Find the legitimate club that
  plays in another country's league (a Welsh club in the English pyramid, Monaco, a
  Liechtenstein club in Switzerland) whose alias now FAILS because its own country ≠ the
  competition country. Did fixing the Brazil weld break a real cross-border merge?

**0.6 — "Restoring the same-day veto (fail-closed to review) fixed the CSKA merge without
breaking true merges."** (`club_identity.propose_domestic_merges`, `club_registry.confirms_same_club`)
- Registry coverage is ~226/355 unambiguous non-UEFA clubs. Auto-merge now requires
  POSITIVE registry confirmation, so any club the registry doesn't list goes to review.
  Quantify: of the merges that previously auto-applied, how many now require manual
  review? If it's most of them, the auto-merge path is effectively dead and every future
  ingest dumps a large review queue — is that the intended cost, or a regression dressed
  as a safety fix?
- Find a true same-country merge the registry cannot confirm (both clubs absent) that now
  wrongly sits in review forever with no path to auto-apply.

**0.7 — "The season-derivation bug is fully centralized and migrated."**
(`fetch.season_for_date`, the 3,054-row migration)
- Confirm EVERY writer now calls `season_for_date` — grep for any remaining inline
  `month >= 7` season logic across the package (seeders, repair paths, backtests).
- The migration relabelled 3,054 rows. Confirm it did not change any WINTER-league row
  (those were already correct), and that Liga MX Apertura vs Clausura within one calendar
  year did not collapse into a single season key.

---

## Part 1 — Did any remediation over-correct?

1. **`reset_country_index` in the write boundary.** The write path now resets the country
   cache. Confirm it does not reset mid-fetch in a way that makes two rows in one batch
   resolve against different indices. Is the invalidation at the right granularity?
2. **`1st_half -> LIV` + `LIV -> LIV` self-map.** A LIV status now persists on a
   past-dated fixture (the match is over but the row says live). Does anything train,
   settle, or price on a LIV row? Is a stale-LIV row inert everywhere, or did admitting
   the status quietly let a non-terminal state into a settled context?
3. **Cache fingerprint expansion.** The author added 8 files. Confirm none of them is read
   at IMPORT time in a way that makes the fingerprint itself unstable, and that the added
   `club_alias_map.json` doesn't churn the whole cache on every identity edit (which would
   make the cache near-useless during identity work).

## Part 2 — The new staking-evidence path, in depth

4. **`decision_ledger.record` idempotency under re-run.** A decision is keyed on
   `(fixture_id, market, side)`. If the same fixture re-enters the window on a later run
   with a DIFFERENT price, the second is dropped. Is "first quote in the window" the right
   policy, or should it be the last quote before the 60-min floor? Either is defensible —
   but is the choice deliberate and does the recorded `decision_lead_min` match the quote
   actually stored?
5. **The window is 60–180 minutes.** A fixture whose only snapshot lands 45 minutes out
   (a late add, a missed run) is never recorded — silent non-coverage. Quantify how much
   of a normal fixture list would fall outside a once-daily capture at this window. Is the
   accumulation rate the author implied (a season to 1000) actually achievable, or does
   the narrow window make it far slower?
6. **`build_bets_reconstructed` still exists.** Prove nothing calls it. If it is truly
   dead, why keep a hindsight path in the tree at all — could a future edit wire it back
   in by accident?
7. **Closing totals (`psc_over25`/`psc_under25`).** Coverage is ~85% of the European
   domestic leagues and ZERO elsewhere. Confirm a non-UEFA totals decision therefore can
   never get CLV and is correctly frozen out of staking. Confirm the de-vig of
   `psc_over25`/`psc_under25` is a real 2-way de-vig, not a passthrough.

## Part 3 — Re-probe the rest (changed under the first round)

8. **`model.fit` after the coefficient change.** The ELO seed offset now depends on
   `coef_as_of`. Confirm the DEFAULT (`coef_as_of=None`, production) reproduces the
   pre-change production ratings within noise — i.e. that fixing the validation leak did
   not silently move live pricing. If Sturm Graz / a Dutch club's live Elo moved, say by
   how much and whether it was intended.
9. **The full suite.** `python3 -m pytest -q` from the repo root: collected / passed /
   failed / skipped, runtime. Confirm the suites that mutate `data/` restore it —
   especially `fixtures.csv`, `model_params.json`, `club_alias_map.json`,
   `decision_ledger.csv`, `settlement_ledger.csv`, `backtest_market.json`. Name any test
   that leaves shared state dirty (the first round found one).
10. **Gate reproduction.** `python3 -m club_soccer.validate --gate` and
    `python3 -m club_soccer.decision_time_backtest`. Confirm gate PASS at the claimed
    Brier and that the decision-time artifact is structurally valid and CLOSED for the
    right (per-market, no-evidence) reason.
11. **One step back.** The staking gate can now, in principle, open per market once ~1000
    frozen settled decisions accumulate. Given everything you found: if that ledger fills
    over a season and the gate opens 1X2 staking, would you trust it with real money — or
    is there a residual defect (selection bias in the quote, the settlement join, the
    edge computation) that would make the first real stakes unsound? This is the question
    that matters; answer it plainly.

---

## Deliverable

Numbered findings, severity-ranked (blocker/high/medium/low), each with `file:line`, code,
breaking input, impact, fix. Then:

- a yes/no on each of the seven Part-0 claims with the reproduction that settles it;
- an explicit verdict on whether any first-round remediation over-corrected (0.5, 0.6 are
  the suspects), with the input that proves it;
- the single change you would make before the decision ledger is trusted with the first
  real stake;
- any NEW instance of "fixed by deleting/■weakening a guard rather than fixing the cause"
  — the author's recurring failure mode.

Do not soften. Two rounds of self-verification compound the blind spots, not reduce them.
