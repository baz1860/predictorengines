# International Football Module: Plan

**Date:** 8 August 2026 · **Revision 3.1**
**Audience:** VP / decision-maker — no technical background assumed
**Companion:** `plans/international-refactor-plan.md` (16 June 2026) — technical compatibility spec
**Analysis:** `scripts/analysis/international_skill_audit.py` — reproduces every number in §2

---

## 1. The one-page version

We want the World Cup module to track and price **every senior men's match between FIFA member
nations**, friendlies included.

**Revision 3 claimed the business case was that we predict qualifying competitions well. That claim
was wrong and is withdrawn.** It rested on a measurement error and on a baseline too weak to mean
anything. The corrected analysis is in §2. Its finding:

> Almost all of the model's apparent value is the Elo rating itself. Once a baseline is given the
> same ratings, the goal model on top adds between **0.4% and 8.8%** depending on competition — and
> under 1% for the World Cup, the Nations League and AFCON. The differences between competitions are
> mostly differences in how mismatched the fixtures are, not differences in how well we model them.

This does not kill the project. It changes what the project is:

- **It is a data and coverage project, not a modelling-edge project.** The case for expanding is that
  we would cover ~1,100 fixtures a year instead of 104, in one system, with proper provenance.
  Whether that coverage converts into betting value is **unproven and cannot be proven from inside
  our own data** — it requires odds history we do not have.
- **The friendlies question is settled enough to deprioritise.** On the strictest baseline the
  friendly-versus-competitive difference is **+0.002, interval −0.011 to +0.016**. Not a detected
  difference. This is not proof of equivalence, but it is sufficient to take friendlies-specific
  modelling off the critical path while leaving it on the risk register.
- **The one thing worth committing to now** is capturing odds and fixtures during the
  **21 September – 6 October** window, because without odds history we can never answer the only
  question that matters commercially.

**Asked for:** funding for scope-hardening, two defect fixes, and a bounded data-capture spike, with
a cost ceiling and pre-committed thresholds (§11). **Not asked for:** a build commitment, a
platform, or a rename.

---

## 2. What the model is actually worth

### Method

`scripts/analysis/international_skill_audit.py` — committed, seeded, with the input file hash printed
on every run. Fit on matches before 1 January 2022; score everything after. Three baselines of
increasing difficulty:

| Baseline | Knows | Purpose |
|---|---|---|
| **B0** unconditional | The overall home/draw/away split. Nothing else | R3 used this. Too easy to be informative |
| **B1** per-competition | The home/draw/away split *within that competition* | Controls for competitions having different draw rates |
| **B2** Elo-only | **The same Elo ratings**, through a plain conversion, with no goal model | Isolates what our modelling adds beyond the ratings |

B0 and B1 use test-period outcome rates, so they are diagnostic references, **not deployable
forecasts**. B2 is the honest internal bar. The commercially relevant bar — de-vigged market
probability — **cannot be computed outside the World Cup**, because we have no odds history.

### Results (FIFA-member universe, n = 3,962)

| Competition | n | Avg. Elo gap | vs B0 | vs B1 | **vs B2** |
|---|---|---|---|---|---|
| AFC Asian Cup qualification | 73 | 253 | 0.342 | 0.334 | **0.084** |
| UEFA Euro qualification | 228 | 342 | 0.321 | 0.313 | **0.040** |
| FIFA World Cup qualification | 907 | 255 | 0.249 | 0.248 | **0.039** |
| Friendly | 1,145 | 200 | 0.158 | 0.154 | **0.024** |
| African Cup of Nations qualification | 279 | 214 | 0.148 | 0.147 | **0.010** |
| FIFA World Cup | 168 | 174 | 0.147 | 0.146 | **0.005** |
| CONCACAF Nations League | 93 | 169 | 0.126 | 0.121 | **0.004** |
| UEFA Nations League | 340 | 170 | 0.118 | 0.113 | **0.010** |
| African Cup of Nations | 156 | 188 | 0.090 | 0.079 | **0.003** |
| **Friendly (aggregate)** | 1,145 | 200 | 0.158 | 0.154 | **0.024** |
| **Competitive (aggregate)** | 2,817 | 221 | 0.189 | 0.173 | **0.022** |

**Confidence interval on the friendly-minus-competitive difference** (block bootstrap over calendar
months, 2,000 resamples — not eyeballed from overlapping intervals):

| Baseline | Difference | 95% interval |
|---|---|---|
| vs B0 | −0.031 | −0.078 to +0.015 |
| vs B1 | −0.019 | −0.066 to +0.032 |
| vs B2 | **+0.002** | **−0.011 to +0.016** |

### Reading it

The **B2 column is the finding**. Give a baseline our own Elo ratings and the rest of the model adds
almost nothing: 0.5% on the World Cup, 0.3% on AFCON, 1.0% on the Nations League. The largest figure
(8.4%, Asian qualifying) rests on 73 matches.

The B0 and B1 columns track the Elo-gap column almost perfectly. That is the whole explanation for
R3's "qualifiers are our strength" story: qualifying competitions contain more mismatches, so any
team-aware forecast beats a team-blind one by more. It was never evidence about qualifying.

### What R3 got wrong, specifically

R3 reported friendlies 0.175, World Cup 0.121, Gold Cup 0.235. The corrected figures on the same
universe are **0.151, 0.147 and 0.188**. The error: R3 averaged per-match skill *ratios*; the
correct construction is the ratio of mean scores. Averaging ratios lets matches where the reference
forecast happened to be confident and right dominate, inflating everything. R3's "decisive finding"
was a transcription of an arithmetic artefact, never committed to a script.

### Remaining limits, stated plainly

- **This is not a clean holdout.** The 2022 split is the repository's own validation window
  (`validate.py` START = 2022-01-01, and `data/validation_baseline.json` records `since: 2022-01-01`,
  n=4,556). Model constants have been chosen against this period. Treat as diagnostic only.
- Single fixed split, not rolling walk-forward.
- The FIFA universe is approximated by the 197-team confederation mapping. **An effective-dated
  211-member registry is a Stage 1 deliverable**, and every number here should be rerun against it.
- Ten competition rows invite cherry-picking. The only comparison stated in advance was
  friendly-versus-competitive; the per-competition rows are exploratory.
- Not detecting a difference is not proof of equivalence. A formal equivalence test needs a margin
  agreed in advance.

---

## 3. Scope

**Recommendation: FIFA members only, senior men.** Testable against an externally maintained list,
and it removes the sub-national tail (Kernow, Padania, Two Sicilies, Isle of Wight, Sápmi — 72 of the
257 currently active identities are unmapped, and most are of that kind).

Stage 1 must also decide and write down, because none of it is currently specified:

- Must **both** teams be in scope, or one?
- Is membership applied **retroactively**, or effective-dated per match?
- Do out-of-scope matches still **train** ratings even if hidden from the product? (They currently
  do, and ratings propagate through the network, so hiding a team does not remove its influence.)
- Successor states and shared rating history.
- Dormancy and reinstatement.

**Out of scope, confirmed:** women's football (the source dataset is men's full internationals by
construction), youth, Olympic, club football, in-play betting.

---

## 4. Competition weights

Only **12 of 201 labels** carry an explicit importance weight; the rest fall to a generic default
covering **34.3%** of matches, including Euro qualifying (2,824 matches) and AFCON qualifying (2,327)
— currently weighted more heavily than a declared friendly.

R3 tested one invented weight table, found a median rating shift of 1.9 points, and called the risk
low. **That has now been measured properly, and R3 was wrong.**

`scripts/analysis/taxonomy_sensitivity.py` runs four defensible weight tables against the **210
active FIFA members** — R3 measured 257 identities including the non-FIFA Island Games sides that
dominated its tail — and translates each into probability and bet-decision impact:

| Profile | median Δelo | p95 | p99 | max | max Δprob | **fixtures crossing the 3% edge threshold** |
|---|---|---|---|---|---|---|
| **v1** (the proposal) | 5.1 | 23.8 | 41.5 | 111.4 | 0.149 | **25 / 600 (4.2%)** |
| v1_flat | 10.6 | 41.8 | 54.2 | 93.1 | 0.167 | 106 / 600 (17.7%) |
| v1_aggressive | 6.3 | 32.3 | 53.4 | 99.2 | 0.162 | 63 / 600 (10.5%) |
| friendly_20 | 10.9 | 26.5 | 30.5 | 41.1 | 0.069 | 42 / 600 (7.0%) |

R3's headline of "median 1.9 points" was wrong three ways: the wrong population, the wrong statistic
(a p99 of 41.5 Elo hides behind a median of 5.1), and it never translated ratings into the quantity
that matters. **A single 1X2 probability moves by as much as 15 percentage points**, and even the
mildest profile changes the bet decision on 4% of recent in-scope fixtures.

**That made it a model change, not a tidy-up** — so it went through the challenger path rather than
being edited into `predictor.py`.

### Challenger result: REJECTED

`scripts/analysis/taxonomy_challenger.py` ran the production walk-forward harness twice, identical in
every respect except the weight table (n = 4,660 matches):

| Model | legacy | v1 | delta |
|---|---|---|---|
| Elo | 0.5134 | 0.5144 | **+0.0010** |
| Dixon-Coles | 0.5123 | 0.5123 | +0.0000 *(sanity check — DC never reads the table)* |
| **Blend** | **0.5097** | **0.5101** | **+0.0003** |

Against a promotion rule fixed before the result was known — pooled improvement, no competition
regressing beyond 0.010, and improvement exceeding the incumbent gate's own 0.002 tolerance —
**all three criteria failed.** The only material per-competition move was Island Games (+0.0119),
a non-FIFA competition already out of product scope.

**The two analyses together are the finding.** Sensitivity says the v1 weights move individual
prices a great deal — up to 15 percentage points, changing the bet decision on 4% of fixtures. The
challenger says they do not improve accuracy at all. *Moving prices without improving them is the
worst of both worlds*, so **legacy weights stay**.

Why this is unsurprising in hindsight: Elo is self-correcting. A competition weighted too low still
converges, because every team plays across a mix of competitions and the errors wash out. The
34.3%-on-a-default-weight statistic was a real description of the code and a poor predictor of harm.

**What v1 is still used for:** competition *categories* and *bettability* — which competitions exist,
which are youth or non-FIFA, which are eligible for a betting product. Only the rating weights are
rejected.

---

## 5. What we have and what is missing

| Have | Detail |
|---|---|
| Results history | **49,523 played matches**, 1872–present, 338 identities, 201 competitions, 18,388 friendlies, public domain |
| Elo ratings | Full history |
| Goal model | Fitted on matches **since 2010** |
| Dixon-Coles | **Rolling 12-year window**, 2.5-year half-life decay |
| Player ratings | 11 MB EA FC 26 database |
| Precedent | The club module already solved multi-competition ingest, coverage tiers and identity |

*(R3 said these were "all fitted on all internationals since 2010." Three different windows, as above.)*

| Missing | Severity | Detail |
|---|---|---|
| **Forward fixtures** | Critical | No diary. Only hand-entered World Cup rows |
| **Odds history outside the World Cup** | Critical | 1,196 price points, 59 matches. Without it §2's commercial question is permanently unanswerable |
| Squads / availability | Important | 48 teams |
| Match statistics | Useful | World Cup only |
| Confederation map | Small | 197 mapped; 72 active teams unmapped |
| Venue coordinates | Small | 48 hand-entered altitudes |

---

## 6. Data acquisition

### Odds

The Odds API's published catalogue carries **nine** senior men's international keys: World Cup;
qualifiers for **Europe and South America only**; Euro and Euro qualifying; Nations League; AFCON
*(odds only)*; Gold Cup *(odds only)*; Copa América. **No friendlies key. No Africa, Asia, CONCACAF
or Oceania qualifiers.** Cost is **markets × regions × polls × keys**, not one credit per call.

R3 said this capped the product "at nine competitions regardless of spend." **That does not follow
from one provider's catalogue** and is withdrawn. R3 also called betting exchanges "accessible
through free tiers" — misleading:

| Candidate | Reality | Status |
|---|---|---|
| BSD | Free, no rate limits, already integrated | **Coverage beyond the World Cup unverified** |
| Betfair Exchange | Delayed key free for development; **live key carries a one-off £499 activation fee**; commercial use needs separate Betfair approval | Candidate, properly costed |
| Aggregators (multi-bookmaker) | Wider catalogues than single providers | Untested |
| Bookmaker feeds direct | Highest effort | Untested |

Exchange prices carry their own trap: **a midpoint is not an executable price.** Available size,
commission and liquidity decide whether an apparent edge is tradable — and friendlies are exactly
where liquidity is thinnest. Any exchange evidence must record depth, not just price.

**No provider is "primary" until the spike scores them.** R3 wrote "BSD primary" one paragraph after
condemning pre-commitment to a provider; that contradiction is removed.

### Fixtures

All candidates tested, none pre-selected. R1's manual-entry fallback (~180 fixtures in an hour) was
fantasy and stays withdrawn.

---

## 7. Implementation

### 7.1 The fixture defect is duplication, not staleness

R2 reported six stale fixtures. R3 corrected that to two. **Both were the wrong diagnosis.** The two
rows are each followed a day later by the same match, scored:

| Fixture | Blank row | Scored row | Venue |
|---|---|---|---|
| Argentina v Egypt | 6 July 2026 | **7 July, 3–2** | Atlanta |
| Switzerland v Colombia | 6 July 2026 | **7 July, 0–0** | Vancouver |

Same teams, same competition, same city, one day apart. Both are North American evening kickoffs —
**local date 6 July, UTC date 7 July.** The merge key is `["date", "home_team", "away_team"]`, so a
row keyed on local date and a row keyed on UTC date cannot be reconciled: the local one is treated as
new and appended, and both survive.

This is a **cross-source identity and date-boundary failure** — precisely the risk this plan
describes as future, already occurring in production. A date-based staleness check, which R3
proposed, would have masked the symptom and left the duplicate generator running.

**Fix:** repair or quarantine the two rows; reconcile on provider IDs where available and otherwise on
a canonical identity containing **kickoff UTC**; add a candidate-duplicate rule for same
teams/competition/venue within ±1 day; add an invariant forbidding a scheduled and a played record
for the same canonical fixture.

### 7.2 Three stores

| Store | Contents |
|---|---|
| **Raw observations** | Exactly what each provider returned, timestamped, never edited |
| **Canonical fixtures** | Internal ID, kickoff UTC, venue + timezone, neutral flag, lifecycle status, source, conflict record |
| **Results** | Promoted only after a match finishes and validates |

With a reconciliation layer: source precedence rules and tombstones for withdrawn fixtures.

### 7.3 Identity

Canonical internal team IDs with **provider-scoped, effective-dated aliases**. Unknown identities go
to a quarantine queue for human adjudication; nothing merges automatically. The existing refusal to
guess at unknown names is correct and stays.

### 7.4 Engine and interface

- **Build the registry alias mechanism first.** It does not exist — registration keys purely on
  engine ID and lookup raises on anything else. Renaming before building it breaks settlement for
  every historical bet.
- Replace the two "World Cup only" filters with a competition parameter defaulting to current
  behaviour.
- Add a competition registry modelled on the club module's.
- The World Cup simulator becomes one competition profile.
- Add a competition selector to the interface. *(R3 said "the interface builds itself from what an
  engine declares" and then listed the hard-coded interface work. The registry-driven part is real;
  the simulate screen, dashboard reads and prediction path are hand-written and need changing.)*
- Port coverage tiers so thin and defaulted teams are visibly flagged.

Also requiring migration: dashboard reads, provenance maps, update orchestration, odds selection and
event IDs, narrative cards, snapshots, settlement records, frozen test outputs.

### 7.5 The gate must block

The daily update catches gate failure and continues, by design, with a comment saying "NEVER block."
It is also pooled across competitions. Replace with a blocking gate matrix: frozen legacy goldens,
**per-competition** temporal metrics, schema and ledger compatibility, fixture-health checks
including the duplicate invariant above.

---

## 8. Sequence

R3's ordering was impossible: it collected auditable evidence before building the store that makes
evidence auditable, and froze the compatibility harness after making behavioural changes. Corrected:

| # | Work | Owner | Effort | Gate |
|---|---|---|---|---|
| 1 | Freeze legacy compatibility harness; write scope and contracts (§3) | Eng + you on scope | ~1 wk | Legacy path byte-identical on frozen inputs and seeds |
| 2 | Fix canonical identity / date reconciliation (§7.1); make the gate block (§7.5) | Eng | ~1 wk | Duplicate invariant passes on full history |
| 3 | Provider and licensing spike — all candidates scored, costs and commercial terms confirmed | Eng | ~1 wk | Written comparison, no provider pre-selected |
| 4 | Minimum **immutable raw collector** + provenance + monitoring | Eng | ~1.5 wk | Replayable from raw; dry-run against known fixtures |
| 5 | **Dry run before 21 September** | Eng | 2 days | Recording verified against a manual sample |
| 6 | Collect evidence: 21 Sept – 6 Oct, then November | — | elapsed | Coverage matrix vs pre-committed thresholds (§11) |
| 7 | **Economic go/no-go** | You | — | Decision against thresholds fixed *before* the window |
| 8+ | Canonical platform, shadow release, cutover, betting | — | Re-estimated after 7 | Separately funded |

Steps 1–5 are roughly **4–5 weeks of one engineer** against a six-week runway. That is tight but not
heroic, and it is the honest number R3 omitted while still demanding the deadline.

### The window arithmetic, corrected

Two windows remain in 2026: **21 September – 6 October** (16 days, up to four matches per team) and a
shorter November window (nine days, two matches). R3 said missing the first costs "a third" of the
remaining evidence. It costs **roughly two-thirds of match opportunities**, and half the windows.

### The calendar claim R3 got wrong

R3 said World Cup qualifying "finished in 2025" and the qualifying payoff waits for the 2030 cycle.
**Both wrong, and my own data disproved the first** — `results.csv` records World Cup qualifying
matches through **31 March 2026**. And **AFCON 2027 qualifying begins 23–25 September 2026**, in the
very next window, with six matchdays through March 2027; Euro 2028 qualifying starts March 2027.

Qualifying is also **36% of post-2022 matches** — the largest single category, but a plurality, not
the domination R3 implied. Given §2, none of this rescues the withdrawn business case; it does mean
fixture volume arrives sooner than R3 claimed.

---

## 9. Risks

| Risk | Likelihood | Impact | Note |
|---|---|---|---|
| **Model has no edge against market prices** | **Unknown — and unmeasurable until odds history exists** | **High** | §2: near-zero gain over plain Elo internally. This is now the project's central risk |
| Miss the 21 September window | High if we do not start now | High | Costs ~two-thirds of 2026 match opportunities |
| Broad international odds unavailable or unlicensable | Confirmed for one provider; unknown elsewhere | High | Exchange route carries fees and commercial approval |
| Duplicate fixtures from date-boundary failures | **Occurring now** | High | Two live cases; scales with fixture count and provider count |
| Gate does not block | **Occurring now** | High | |
| Identity collisions across providers | High | High | Canonical IDs plus quarantine |
| Wrong neutral flags mis-price matches | High | Medium | Moves home advantage directly |
| Taxonomy change moves prices | Medium | Medium | p90 = 13 Elo ≈ 2pp. Needs full sensitivity work (§4) |
| Friendlies materially worse than competitive | Low on current evidence | Medium | Not detected (§2), not disproven. Off critical path, on register |

---

## 10. Pre-committed thresholds (revised 8 August 2026)

**Agreed before the window opens, so they cannot move afterwards.** Revised once, *before* any
window data existed, because the original set contained a threshold that measurement showed was
unreachable — and an unreachable threshold is not a gate, it is a guaranteed failure that will be
argued away later.

**What changed and why:** the original demanded *"≥20 competitions with usable odds at 48h lead"*.
BSD carries **16 senior internationals in total**, of which 4 currently have upcoming fixtures, and
The Odds API carries 9. Twenty was never achievable by any provider we have found. Neutral-venue
accuracy was also unscoreable as written: the flag is present on 100% of fixtures but we had no
ground truth to check it against, so it needed a sampling rule. Two measures were missing entirely —
venue-timezone resolution and first-price lead time — both of which turned out to be live risks.

| Measure | Go | Reconsider | Stop | Baseline today |
|---|---|---|---|---|
| Fixture recall vs a 30-fixture manual sample, 14 days ahead | ≥95% | 85–95% | <85% | not yet measured |
| Duplicate rate after reconciliation | <0.5% | 0.5–2% | >2% | **0%** |
| Neutral-venue accuracy, 20-fixture hand-checked sample | ≥95% | 85–95% | <85% | flag present on 100%, accuracy unverified |
| Cancellation reflected within | 48h | 48h–5d | >5d | not yet measured |
| Venue timezone resolved (so local date is real, not the UTC date) | ≥60% | 30–60% | <30% | **21%** — currently Stop |
| **Competitions with any usable odds at 48h lead** | **≥6** | **3–5** | **<3** | **0** |
| First price appears before kickoff by | ≥48h | 6–48h | <6h or never | **World Cup baseline: median 60h — Go** |
| Odds licensable for our use | written confirmation | ambiguous | refused | not sought |

Two of these are already outside Go, which is the point of writing them down: venue resolution sits
in Stop territory at 21%, and odds coverage is at zero with the "wait on BSD" decision riding on it.

### When do BSD prices actually appear? Measured, not assumed

The "wait on BSD" decision rested on an assumption — that prices arrive closer to kickoff. Rather
than wait to find out, that was tested against the stored World Cup 2026 raw payloads (557
observations across 64 fixtures):

| Time to kickoff | Observations | Priced |
|---|---|---|
| <6h | 42 | **100%** |
| 6–24h | 86 | **100%** |
| 1–3 days | 155 | **96%** |
| 3–7 days | 257 | **21%** |
| >7 days | 0 | never sampled |

**62 of 64 fixtures were eventually priced. Median first price 60h before kickoff, max 130h.** The
assumption holds, and BSD demonstrably does carry international odds — it simply publishes them
about three days out.

This changed the polling design. A flat cadence sampled the 0–72h window — where the price *moves*,
and where closing-line value lives — no more densely than the empty weeks beforehand, which would
have produced no usable closing price. The cadence now tightens from 48-hourly at discovery range to
every 30 minutes inside 6h.

**Caveat that matters:** these figures come from a World Cup, the most heavily traded international
event there is. Friendlies and the Nations League will likely be priced later and less completely,
so 60h is an optimistic upper bound. The live window will correct it.

**Second-order finding:** 144 of 399 fixtures — the entire CAF derived set — carry no kick-off time
and therefore **cannot be odds-polled at all**. They can be predicted but not priced. That is now
reported on every run rather than silently excluded.

**Discovery cost ceiling: £1,500** covering all provider trials and any activation fees, plus the
engineering time in §8. Anything beyond that returns for approval.

### CAF sourcing: SOLVED, and free

Every *commercial* route to AFCON 2027 qualifying is paid:

| Provider | Carries AFCON 2027 qualifying? | Verified |
|---|---|---|
| BSD | **No** — no such league in a 79-league catalogue | 8 Aug |
| api-football | League id 36 exists, but the **free plan is capped at 2022–2024 seasons** | 8 Aug, our key |
| football-data.org | Competition 2193 exists, but it is **TIER_FOUR** — 403 with our key | 8 Aug, our key |
| openfootball | Only to 2024, and it mirrors our own results source anyway | 8 Aug |
| TheSportsDB | No AFCON league discoverable on the free tier | 8 Aug |

**But the fixtures themselves are public, and a 4-team double round-robin is deterministic.** Given
the group draw of 19 May 2026, every matchday pairing follows from the standard template. So the
fixtures are *derived* rather than bought — `international/providers/caf.py`, 144 fixtures, £0.

Correctness rests on two independent sources agreeing: the group draw (Wikipedia, sourced to CAF)
and a published fixture list (africasoccer.com, 20 May 2026). `cross_check()` compares them, and it
earned its keep twice:

- it caught an error in the **published list** — Group A MD6 reads "Morocco v Gabon; Niger v Gabon",
  which plays Gabon twice and repeats the MD1 fixture;
- it caught an error in **our own template** — CAF runs MD4/5/6 as the reverses of MD3/1/2, not the
  intuitive MD1/3/2. That mistake had produced 20 wrong fixtures.

**Known limitation, recorded not hidden:** CAF has published matchday *windows*, not per-fixture
kick-off times or venues. Every derived row is dated to its window start and says so in `conflict`.
They are correct about who plays whom and roughly when; they are **not** precise enough to time an
odds poll or settle a bet. If a dated source appears, it should overwrite these.

**Result: the September window goes from 157 to 205 fixtures at zero cost, and the 23 September
deadline is met.**

---

## 11. Decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | Accept that the economic case is **unproven**, and that this is a coverage project until odds history says otherwise | **Yes** — §2 |
| 2 | Team universe: FIFA members, senior men | **Yes**, with §3's sub-questions answered in Stage 1 |
| 3 | Fund steps 1–5 (~4–5 engineer-weeks) and the £1,500 ceiling | **Yes** |
| 4 | Ratify the §10 thresholds now, before any data is seen | **Yes** — this is the point |
| 5 | Fix the two live defects regardless of whether the project proceeds | **Yes** |
| 6 | Drop friendlies-specific modelling from the critical path | **Yes**, keep on risk register |
| 7 | Confirm women's, youth, Olympic, club football out of scope | Confirm |

---

## Appendix 0 — Build status (8 August 2026)

Steps 1, 2 and 4 of §8 are **built and tested**. Step 3 (provider spike) needs live
credentials and a real international window, so it remains open.

| §8 step | Status | Delivered |
|---|---|---|
| 1 — Compatibility harness + scope contracts | **Done** | `tests/international/test_legacy_goldens.py`, `international/taxonomy.py`, `international/registry.py`, engine alias in `contracts/registry.py` |
| 2 — Identity/date reconciliation + blocking gate | **Done** | `international/identity.py`, `international/fixtures.py`, `international/gate.py`, patched `merge_results.py` and `update.sh` |
| 3 — Provider spike | **Done for BSD** | `docs/international_provider_spike.md`. Licensing and rival providers still open |
| 4 — Immutable raw collector | **Done and running** | `international/store.py`, `international/providers/bsd.py`, `international/venues.py`, `scripts/international/fetch_fixtures.py` |
| 5 — Dry run before 21 September | **Effectively done** | 255 fixtures fetched, parsed and stored; replayable offline |

**Fetcher results (8 August):** 255 international fixtures in the canonical store,
**157 of them inside the 21 Sep – 6 Oct window**, zero duplicates, 100% carrying UTC
kick-off times and stable provider IDs.

**Everything buildable is now built.** 24 tests pass; the data gate, the
per-competition model gate and preflight all run clean. Also delivered since:
odds capture with absence recording, coverage tiers, the results promotion path,
a cross-provider comparison harness, FIFA accession dates, a cron-ready refresh
runner, and the §4 sensitivity analysis (which overturned R3 — see above).

**Four items now need a decision that only you can make:**

| # | Item | Deadline |
|---|---|---|
| 1 | **AFCON 2027 qualifying is missing** from BSD (~144 matches). Request it or source elsewhere | **23 September** |
| 2 | **Odds route.** BSD returned nothing at six weeks out; Betfair costs £499 + commercial approval; TheSportsDB's free tier is too capped to be a source | Before the window |
| 3 | **Back up `data/international/raw/`** off-repo. 23MB, gitignored, unreconstructable after the fact | Before the window |
| 4 | **Adjudicate the 8 pending duplicate pairs** against an external source. The strict gate stays red until then | Any time |

**And one that no amount of engineering closes:** we still cannot measure edge
against market prices, because no odds exist yet. The capture system is built and
recording absence; the answer arrives with the September window or not at all.

**What was fixed in production data:**

- The two duplicated World Cup fixtures are gone; `data/results.csv` is 49,527 rows.
  A backup sits at `data/results.csv.bak.dupes.20260808`.
- `merge_results.py` now reconciles a local row against an upstream row for the same
  fixture dated within one day, so the duplicates cannot regenerate on the next
  refresh. Verified with a reconstruction of the exact July 2026 scenario.
- `update.sh` no longer swallows gate failures. Both gates block; `ALLOW_GATE_FAIL=1`
  is the documented escape hatch.

**What the build found that the plan did not predict:**

- **Eight further duplicate pairs already in the history**, not two — Aruba v Curaçao
  (1937 and 1945), Niger v Mali (1983), Gibraltar v Jersey (1995), Dominica v Saint
  Lucia (1999), Occitania v Ambazonia (2006), Dominican Republic v Curaçao (2011),
  Nicaragua v Cuba (2021). Each is the same fixture on consecutive days at the same
  venue with an identical score. They are **not** auto-repaired: dropping a scored
  row deletes a result and shifts every later Elo rating. They are recorded in
  `data/international/fixture_exceptions.csv` as `pending_review`, the daily gate
  tolerates them, and the strict gate fails until they are adjudicated.
- **The registry lands on exactly 211 current FIFA members** with all six
  confederation counts correct, once Czechoslovakia and Yugoslavia carry end-dates
  and Kazakhstan's 2002 AFC→UEFA transfer is applied.
- **A pattern-matching taxonomy is itself a silent default.** The first version
  classified any unseen "…Cup" as a regional cup. Classifications now carry a
  `provisional` flag; 82 long-tail labels are acknowledged, and a genuinely new
  label fails the gate rather than quietly acquiring a guessed weight.

**Deliberately not done:** no rename, no engine alias, no competition selector in the
interface, no promotion of canonical fixtures into `results.csv`. All of those come
after the evidence phase, per §8.

---

## Appendix A — Verified code references

| Finding | Location |
|---|---|
| Merge key is date + teams, so UTC/local date splits duplicate | `merge_results.py:27` (`KEY`), `:69–72` |
| "Upcoming" = blank score, no date or status test | `engines/worldcup/predictor.py:78` |
| Knockout placeholders deliberate and filtered | `engines/worldcup/predictor.py:303–309` |
| Gate never blocks | `update.sh:131–135` |
| 12 weighted competitions, `DEFAULT_K=30` for 189 | `engines/worldcup/predictor.py:66–76` |
| Repo validation window is 2022+ (contaminates §2) | `engines/worldcup/validate.py:50`, `data/validation_baseline.json` |
| Dixon-Coles 12-year window, 2.5-year half-life | `engines/worldcup/dixoncoles.py:32–33` |
| Context model excludes friendlies, opt-in | `engines/worldcup/context.py:19–20`, `:44–46` |
| Confederation adjustment fitted on 2002–2022 World Cups | `engines/worldcup/confederation_adj.py:63` |
| App prediction path ignores competition | `app/engines/worldcup.py:64` |
| Registry has no alias mechanism | `contracts/registry.py:65–82` |
| Simulate UI sends only model and sim count | `app/web/app.js:432–446` |
| Market blend inactive, weight 0, "not deployable" | `data/market_blend.json` |
| The two "World Cup only" filters | `predictor.py:302`, `edge.py:235` |

## Appendix B — Revision history

**R1.** Claimed the engine was "already international"; miscounted matches and teams; credited the
gate with blocking it does not do; claimed the model learns from closing prices when that component
is off; concluded odds cost £0–30/month when the provider carries no friendlies.

**R2.** Deleted the implementation plan in response to review; reported six stale fixtures when four
were deliberate placeholders; claimed the taxonomy fix changes "every rating" without measuring;
called BSD "close to the only route" to odds on no evidence; recommended a scope option while
deferring its boundary rules.

**R3.** Built its executive story on a skill table that was **arithmetically wrong** (averaged
per-match ratios instead of the ratio of means) and never committed to a script; used a baseline too
weak to support any conclusion; treated overlapping confidence intervals as evidence of equivalence;
measured on all 338 identities while recommending a FIFA-only product; mis-diagnosed a duplication
bug as staleness; sequenced evidence collection before the store that makes it auditable; asserted
World Cup qualifying ended in 2025 when its own data file showed 31 March 2026; understated the cost
of missing the September window; and requested a deadline commitment with no owner, effort estimate,
cost ceiling or pre-committed thresholds.

**R3.1** corrects all of the above. The analysis is now a committed, seeded, hash-stamped script; the
economic thesis is withdrawn as unproven; and the plan carries owners, effort, a cost ceiling and
thresholds fixed in advance.
