# Codex Prompt: Adversarial Review — UEFA + non-UEFA Expansion, Identity Layer, Staking Evidence

Copy everything below the line into Codex.

---

You are reviewing a large body of work that expanded a club-soccer model from 5 fitted
leagues to ~50 (UEFA + non-UEFA), rebuilt the identity layer, added an operational
monitoring stack, reframed the daily card around win-probability, and built the
decision-time backtest that gates staking. It was built fast, by one agent, with the
author verifying its own work. Assume that self-verification is compromised and that the
most confident claims are where the bugs hide.

**Standing rules:** harsh. Every finding as `file:line` + the offending code + concrete
impact + the fix. Do **not** modify files. Reproduce every number you can; when you
cannot reproduce, say so and say why. A "looks fine" is worthless — find the input that
breaks it.

Priorities, highest first: (0) leakage in the backtest and validation, (1) the safety
guard that was **inverted**, (2) the model change that was **promoted to production**,
(3) the identity guards that silently weld clubs together, (4) everything else.

---

## Part 0 — Falsify these five headline claims first

Each is load-bearing and each was asserted by the author on evidence they also produced.

**0.1 — "The decision-time backtest has no look-ahead."**
`decision_time_backtest.py` claims each fixture is priced by a model trained only on
prior results (`_price_month`: monthly refit on `played[_ko < cutoff]`, `context_coef={}`,
`DEFAULT_ENSEMBLE_W`). Attack the claim on a different axis than the model fit:

- `build_bets()` calls `canonical_name(...)` and `_key(...)` to match snapshots to
  fixtures, and `canonical_name` reads `data/club_alias_map.json` — which was built from
  the **full history including future merges and manual review verdicts**. So a past
  decision is being resolved with identity knowledge that did not exist at that decision
  time. Is that leakage that matters for the metric, or benign relabelling? Prove which.
- Same question for `club_registry.py` (openfootball) and `reclassify_by_club_country`:
  the model at time T is being handed team identities and league labels fixed with
  post-T information.
- `_closing_probs()` reads the Pinnacle close — correct for CLV, but confirm the close
  never leaks into `p_model` or the bet-selection path.
- The confidence/threshold tables use `p_model` from the same monthly params — confirm no
  fixture is in its own training fold at a month boundary (a match on the 1st of the
  month, `cutoff = month_matches["ko"].min()`).

**0.2 — "The same-day-collision guard was exactly backwards, so I inverted it."**
This is the single most dangerous change in the review. In `club_identity.py`,
`propose_domestic_merges` previously **vetoed** a cross-source merge when the two
identities shared a match date (reasoning: a club can't play twice a day). The author
found the opposite on 6 pairs — true merges (`FC Twente`/`Twente`) show 1–2 same-day
"clashes" from source date disagreements, genuine different clubs (`Sparta Praha`/`Sparta
Rotterdam`) show 0 — and **removed the veto**, keeping only the affinity floor, the
already-present veto, the blocklist, and now the club-registry country veto.

Attack it hard:
- Six pairs is a tiny basis for deleting a safety guard. Construct the adversarial case:
  two genuinely different clubs, same country (so the registry country veto does NOT
  fire), similar names (so affinity passes), that share a date. Does anything now stop
  the merge? Walk `propose_domestic_merges` line by line for that input.
- The registry veto (`same_club_possible`) only fires when BOTH clubs are in
  openfootball AND in different countries / reserve mismatch. What fraction of the
  incoming non-UEFA clubs are actually in openfootball? For clubs absent from it, the
  registry returns "no objection" — so the only remaining guards are name affinity and
  the human review queue. Is that enough without the date veto?
- Was the original date logic definitely wrong, or wrong *only* because canonicalization
  now collapses the two sources' spellings so the "same club" pairs coincide on date?
  I.e. did the author fix the symptom by deleting the guard rather than fixing why true
  pairs collided?

**0.3 — "League-prior seeding improves the model, so it is promoted (ON in production)."**
`model.LEAGUE_SEED_DEFAULT = True`. The A/B (`data/league_seed_evidence.json`) claims
walk-forward Brier 0.61730 → 0.61521 (2024-07, n=21,596) on "identical folds both arms".

- Confirm the two arms actually ran identical folds. The walk-forward cache keys on
  `league_seed` (`validate.py`) — verify a stale fold from one arm cannot be served to
  the other. Delete the cache and reproduce both arms from cold. Do the deltas hold?
- `validate.walk_forward(league_seed=None)` resolves to `M.LEAGUE_SEED_DEFAULT`. Confirm
  the gate therefore measures the production model and cannot silently score the other
  arm. What happens to a cached fold produced under the OLD default the first time the
  new default is read?
- −0.002 Brier is small. Is it inside the run-to-run noise of the monthly refit? Estimate
  the noise (e.g. reshuffle the fold boundary by a day) and report whether the
  improvement survives.
- `_primary_league_map` + the ELO seed offset (`ELO_PER_STRENGTH`, `LEAGUE_SEED_ANCHOR`,
  `strength_prior`): a club's seed depends on its most-played league across ALL history.
  In a walk-forward fold this again uses future league membership. Leakage or not?

**0.4 — "Country-scoped identity makes cross-confederation collisions impossible."**
`canonical_name(name, country=...)` refuses an alias whose target is in a different
country (registry or fixtures-derived). `apply_alias_map(scope_by_country=True)` does the
same per row.

- Find the false NEGATIVE: a legitimate merge that this now REFUSES because the target's
  country is unknown, wrong, or the row's competition maps to the wrong country. The
  Denmark/Romania "Superliga" case proves labels can be wrong — does the country scope
  ever block a correct merge for a mislabelled row?
- `reclassify_by_club_country`: acts when the clubs it can place agree on one country ≠
  the label's. Construct a real cross-border domestic fixture (playoffs, or a club that
  genuinely moved association) and show whether it mis-moves. Confirm it never
  reclassifies into a league with a colliding name in a third country.
- `team_countries()` is cached process-wide (`_country_index_cache`) and built from
  `fixtures.csv`. After a merge/ingest rewrites fixtures, is the cache invalidated
  everywhere it is read? Find a call path that uses a stale index.

**0.5 — "The evidence gate cannot be opened by the artifact I now produce."**
`decision_time_backtest.run()` writes `data/backtest_market.json`; the gate stays CLOSED.
Confirm it is CLOSED for the RIGHT reasons and cannot be tricked:

- Feed the gate hand-crafted artifacts: `n_bets = 1000` exactly, all ROIs `1e-9`, CLV
  `1e-9`, `frac_positive = 0.5`, `decision_lead_minutes = 60`, generated now. Does it
  open? Should it, given the preregistered TODO tightenings (block-bootstrap lower
  bounds) are documented but NOT enforced in `evaluate()`? State whether the gate as
  coded is weaker than the gate as commented.
- NaN/Inf: can any metric the backtest writes (`flat_roi`, `clv_mean`) reach the gate as
  a value that passes `_finite` but is semantically wrong? Check the totals CLV path
  (`None`) and the `_block_bootstrap_lb` output (can it be NaN and does anything read
  it?).

---

## Part 1 — Identity layer, in depth

`club_identity.py`, `club_registry.py`, `identity_review.py`, `names.py`.

1. **Cumulative alias map + chain flattening.** `--write` merges new aliases into the
   prior map and flattens chains (`a→b, b→c ⇒ a→c`). Construct a cycle (`a→b, b→a`) or a
   chain that flattens to a self-map. Does `_resolve` terminate? Can a bad manual verdict
   plus an auto-merge produce a target that no longer exists in fixtures?
2. **`propose_domestic_merges` one-to-one.** It claims two Europe-only spellings may
   collapse onto one club, but one domestic identity must not absorb two different clubs.
   Verify with the `Olympiacos` case and then break it: two distinct incoming clubs that
   both best-match the same existing identity.
3. **`identity_review` consistency check.** It flags "merged but still in fixtures" and
   "distinct but still aliased". Confirm it actually catches the FK Žalgiris-class
   reversal (verdict flipped, alias not removed) and that `--apply` truly removes the
   alias on reversal rather than only recording intent.
4. **`club_registry` ambiguous names.** A name in ≥2 countries is marked `ambiguous` and
   returns no country. Confirm `same_club_possible` and `canonical_name` both treat
   ambiguous as "no objection / pass through" and never as a country assignment. Find a
   name that is ambiguous in openfootball but which our data treats as one club.
5. **openfootball parser.** `parse_clubs_file`: reserve detection (`ii)`/`b)`),
   language-tag stripping (`[en]`), founding-year fields (`Club, 1976, City`). Feed it
   malformed lines and confirm no club name is silently lost or a city becomes a club.

## Part 2 — Ingest, refresh, season labelling

`seed_fdcouk_leagues.py`, `fetch.py`, `competitions.py`.

6. **Shrink guard vs. reclassification.** `write_fixtures` refuses a write dropping >50%
   of rows. `reclassify_by_club_country` + dedupe + the self-match filter all mutate row
   counts. Construct a legitimate operation that trips the guard, and a catastrophic one
   that slips under 50%. Is 50% the right line?
7. **Self-match filter.** `home == away` rows are dropped at the write boundary. Confirm
   it is string-exact after canonicalization and cannot drop a legitimate row (e.g. a
   placeholder "TBD v TBD", or a genuine same-named-different-club edge the identity layer
   failed to separate).
8. **`_season_of` / `calendar_season`.** Nordic + Irish + all non-UEFA leagues use
   calendar-year seasons; Liga MX runs two tournaments per year registered as two
   competitions. Find the fixture date that lands in the wrong season label, and confirm
   the Liga MX Apertura/Clausura split does not double-count or mis-window a club-season.
9. **Season-aware staleness.** `_active_months` (≥25% of peak month) + `refresh_health`
   (source-ahead-of-us). Find a league whose real mid-season break exceeds the 21-day
   warn window but is suppressed as "off-season", and a pre-season gap that wrongly warns.
   Confirm `refresh_health` fails safe (network down → not a false "behind").
10. **BSD league-name collisions.** Exact-match `comp_from_bsd_league`. Enumerate every
    registered league whose BSD name is a substring/superset of another continent's
    (`Super League`, `Superliga`, `Serie A/B`, `Championship`, `Primera`, `Premier
    League`). Prove none resolve to the wrong competition, now that ~30 more names exist.

## Part 3 — Model, calibration, caching

`model.py`, `league_strength.py`, `uefa_registry.py`, `walkforward_cache.py`.

11. **xG-evidence gating.** `_weights_for_match` scales shot components by
    `n/(n+8)` on the weaker club's `xg_evidence`. Confirm: (a) a club with partial shot
    history is scaled correctly, not just zero/full; (b) the renormalisation cannot divide
    by zero or produce weights that don't sum to 1; (c) `xg_evidence` in stored params
    matches what the live path recomputes.
12. **Walk-forward cache fingerprint.** `code_fingerprint` hashes a fixed file list
    (`model.py`, `competitions.py`, `validate.py`, `coverage.py`, `comp_strength.json`,
    `ensemble_weights.json`). Name a change that alters predictions but is NOT in the
    fingerprint (e.g. `uefa_registry.py`, `league_strength.py`, `club_identity.py`,
    `names.py`). Does a stale fold then get served? This is a correctness hole if so.
13. **`estimate_k` stability.** Split-half variance decomposition, `K = sigma²/tau²`,
    `tau² = max(total_var − noise, 1e-6)`. Force `tau²` to the floor (leagues nearly
    identical) and show K → 500 cap; force `sigma²` huge (few shared leagues) and show the
    fallback. Is the 22-league basis enough for the reported K=10.5, or noise?
14. **`prior_is_informative` + per-league K.** Default-prior leagues use `K_DEFAULT_PRIOR
    = 3`. Confirm this is only ever applied to leagues genuinely on the 0.75 fallback and
    never silently down-weights a real coefficient prior. Confirm the plausibility gate
    still holds and `comp_strength.json` stays `active:false`.
15. **Variance inflation.** Implemented, `VARIANCE_INFLATION_DEFAULT = False`. Confirm it
    is genuinely inert on every production path (predict, predict_match, walk_forward,
    edge) and that the A/B artifact's "OFF is better" still reproduces.

## Part 4 — Card, ledger, operations

`season.py`, `run_ledger.py`, `monitor.py`, `health.py`, `edge.py`.

16. **Likely-winners card.** `_likely_winners_section` leads by `p_model`, full-evidence
    only, excludes do-not-bet but NOT evidence-gate suppression. Confirm a gated (£0)
    pick appears but a genuine do-not-bet line does not; confirm no `p_model` is a bet on
    the wrong side (side-awareness of `p_model` per market/side in `edge.rows_from_odds`).
17. **Run ledger readiness.** `green_streak` breaks on gap >30h, failure, or stale-league.
    Empty ledger must read UNKNOWN not healthy. Coverage-erosion check (latest vs median
    of last 10). Find the sequence of records that wrongly reports READY, or wrongly
    breaks a legitimate streak.
18. **`_write_last_run` + ledger append.** Best-effort, must never fail the pipeline.
    Confirm an exception in `run_ledger.snapshot()` (e.g. unreadable params) cannot crash
    the season run, and that failures ARE recorded (a success-only ledger can't measure a
    streak).

## Part 5 — Cross-cutting

19. **Full suite.** Run `python3 -m pytest -q` from the repo root. Report collected /
    passed / failed / skipped and the runtime. Several suites delete/rewrite files under
    `data/` — confirm they restore state and do not leave `fixtures.csv`,
    `model_params.json`, `club_alias_map.json`, or `backtest_market.json` mutated for the
    next suite. Name any test that pollutes shared state.
20. **The gate can never open on today's data — but should the ENGINE ever be trusted?**
    Step back: even at 1,000 bets, is the decision-time methodology as coded actually
    deployable, or are there silent approximations (median decision lead reported as a
    scalar while individual leads vary 60min–7d; odds_median across n_books vs a real
    executable price; canonicalization leakage from 0.1)? Give your verdict on whether
    Phase B accumulation on THIS engine would produce trustworthy evidence, or whether the
    engine needs changes first.

---

## Deliverable

A numbered findings list, severity-ranked (blocker / high / medium / low), each with
`file:line`, the code, the breaking input, the impact, and the fix. End with:

- a yes/no on each of the five Part-0 claims, with the reproduction that settles it;
- the one change you would make before Phase B accumulation starts (since every day of
  accumulation on a flawed engine is wasted calendar time);
- any place where the author "fixed" a problem by deleting a guard rather than fixing the
  cause (0.2 is the prime suspect; find others).

Do not soften. The author has been marking their own homework for a long session and the
blind spots are cumulative.
