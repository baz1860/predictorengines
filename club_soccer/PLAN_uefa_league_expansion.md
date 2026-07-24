# Club Soccer — UEFA-Wide League Expansion Plan

**Status:** Draft for review
**Date:** 2026-07-22
**Trigger:** Sturm Graz priced at 18% away to Hearts; won 4–0.

---

## 1. Diagnosis

The instinct was right, but the cause is not simply "Austria isn't fitted." There are
three independent defects, and fixing only the obvious one would leave the model
producing the same class of error.

### 1.1 Under-evidenced teams collapse to *exactly average* — invisibly

**Corrected during P0 implementation.** The first draft of this plan claimed unknown
teams are silently defaulted at pricing time. That is not quite what happens, and the
real mechanism is more subtle.

In `model.py` every rating lookup does use a silent default:

```python
ah = params["attack"].get(home, 0.0)      # 0.0 == league-average attack
eh = params["elo"].get(home, BASE_ELO)    # BASE_ELO == 1500.0
```

But `predict()` raises `ValueError` for a team absent from `params["teams"]`, and
`edge.rows_from_odds` catches it and `continue`s. Verified: the rating maps (`attack`,
`defence`, `attack_xg`, `elo`, …) have **zero missing entries** for fitted teams. So
completely unknown clubs are *silently dropped from the card*, not mispriced. That is a
different fault — invisible non-coverage rather than confident error — and the
walk-forward validation inherits it (§1.6).

The live mispricing mechanism is teams that **are** in the fit on almost no data. The
attack/defence shrinkage carries a prior weight of 4 matches, and Elo starts at 1500,
so a club with a handful of results is pulled back to the global mean — a mean computed
across all 892 fitted teams, including English and Scottish League Two.

Sturm Graz is exactly this: present as `SK Sturm Graz`, **23 matches, all European**,
Elo 1505.06 (0.06 above default), attack coefficient **−0.267**. The shrinkage is doing
its job correctly. The defect is that nothing distinguishes *"average because we
measured it"* from *"average because we have no idea"* — and the card reported
`lineup_confidence 1.0` for the fixture regardless.

### 1.2 Team identities are fragmented across data sources

`names.py` defines a canon (`OVERRIDES`, `make_canon`) mapping source spellings onto
football-data.co.uk names. It is applied in the openfootball seeder but **not** on the
BSD ingest path. The result, measured in the current `fixtures.csv`:

| Canonical club | Identity A | Identity B |
|---|---|---|
| Bayern Munich | `Bayern Munich` — 136 BL, 20 CL | `FC Bayern München` — 102 BL, 29 CL |
| Inter | `Inter` — 204 SA, 28 CL | `FC Internazionale Milano` — 25 CL |
| Hearts | `Hearts` — 152 SPFL (Elo 1717) | `Heart of Midlothian` — 109 SPFL, 8 euro (Elo 1646) |

Two consequences:

1. **Fitted teams are already being under-rated in Europe.** Ajax appears only as
   `AFC Ajax`, euro-matches only, Elo 1437.
2. **The Hearts side of the fixture is itself suspect.** Which of the two Hearts
   identities priced that match determines a ~70-point Elo swing.

Ingesting 50 more leagues before fixing this would multiply the fragmentation, not
reveal it.

### 1.3 No minimum-evidence gate

175 teams currently in the model have **no domestic league data at all** — they exist
only through UEFA matches. Median sample: **8 matches**.

Sturm Graz specifically: present as `SK Sturm Graz`, 24 matches, Elo 1505.06 (i.e.
0.06 above the default), attack coefficient **−0.267** — the model had actively
concluded Sturm Graz are a *below-average attacking side*, on a handful of results
against opposition it also could not rate.

Meanwhile `forward_predictions_club.csv` records `lineup_confidence 1.0` and
`n_missing 0` for these fixtures. The evidence gate reports full confidence on teams
the model knows nothing about.

### 1.4 Reconstructing the 18%

Hearts ≈ 1717 Elo (fitted, real data) versus Sturm Graz ≈ 1505 (default), plus home
advantage. A ~212-point gap plus HFA produces roughly the 18/22/60 shape observed.
The model was not miscalculating. It was correctly computing the consequences of an
input that asserted, without evidence and without flagging it, that Sturm Graz are an
average team drawn from a pool containing League Two.

### 1.5 The validation is structurally blind to this failure

`validate.walk_forward` **skips** any match whose teams were unseen in the training
window, and reports one pooled metric. So the fixtures most at risk are either excluded
outright or averaged into 13,000 well-evidenced ones. Pooled Brier can improve while
cross-league pricing stays broken. Measured over 2024-07 onward: **764 matches
unpriceable and silently dropped** by this rule.

### 1.6 Existing league-strength fit is already unreliable

`data/comp_strength.json` carries `"active": false`, and its fitted values are not
credible — Bundesliga 0.545 and La Liga 0.584 rank *below* Scottish Premiership 0.698.
This is the signature of an under-identified fit, and it is a warning about the
approach to avoid in §4.

---

## 2. Design principles

These follow from the diagnosis and are what make the solution durable rather than a
one-off patch.

1. **No silent defaults, ever.** Every rating lookup for an unknown team must record
   the fact. Absence of data is information and must propagate to the output.
2. **There will always be unfitted teams.** Even at 55 leagues, newly promoted clubs,
   reformed clubs and early qualifying entrants will fall below threshold. The system
   must degrade *gracefully and permanently*, not just until coverage improves.
3. **Seed from the league prior, not the global mean.** A new Austrian Bundesliga club
   should start at the Austrian Bundesliga mean, anchored to that league's estimated
   strength — never at the pooled global mean.
4. **Do not fit 55 free league-strength parameters.** See §4 — this is the decision
   that determines whether the long tail works or produces noise.
5. **Identity before ingest.** Sequencing is load-bearing.

---

## 3. Scope (confirmed)

- **Coverage:** all 55 UEFA member associations — **top *and* second tier**, wherever
  the data exists. Second tiers matter: they are where promoted clubs come from, and a
  club's first season in Europe usually follows a season the model would otherwise have
  no record of. Where a second tier is unavailable, its clubs fall to the low-evidence
  path below rather than blocking the wave.
- **History:** 5 seasons per league.
- **Fallback behaviour:** below-threshold teams are still priced (P0 keeps pricing on by
  design), seeded from the league prior with inflated variance, and **flagged at the
  point the bet suggestion is made** — already implemented in P0.
- **UEFA coefficients:** scraped to start, refreshed **annually** after the season ends
  and the new coefficients publish (a `--refresh-coefficients` step with the fetch date
  recorded in the artifact, so a stale table is visible rather than assumed current).

> ⚠ **Superseded by P2.** The feasibility claim originally recorded here was wrong — it
> was inferred from the local BSD cache rather than a live catalogue pull. All 55
> associations is **not** achievable on free sources; the reachable set is 22
> associations / 35 divisions, and second tiers exist only for the big five plus
> Scotland. See P2 for the verified source map. Scale is genuinely not a constraint: the
> refit is **0.5s for 22,408 matches / 892 teams**.

---

## 4. The hard part: cross-league calibration

This deserves stating plainly because it is where the plan most likely fails if rushed.

**Team strength and league strength are confounded.** Within a league, only *relative*
team strength is identified from results; the league's absolute level is not. The only
things that link leagues into a common scale are:

- UEFA competition matches (inter-league), and
- teams moving between divisions via promotion/relegation (links tiers *within* a
  country only).

Across 55 leagues this link graph is severely unbalanced. England–Spain is connected by
hundreds of matches. Gibraltar or San Marino are connected by a handful of qualifying
ties per season, nearly all defeats, against a different opponent league each year.
Fitting a free strength parameter per league from that will produce exactly the
implausible output already sitting in `comp_strength.json` (§1.5).

**Proposed approach — hierarchical shrinkage to the UEFA coefficient.**

Estimate one strength parameter per league, but shrink it toward the published UEFA
country coefficient, with shrinkage weight inversely proportional to the number of
observed inter-league matches:

```
strength_league = (n_eff · strength_observed + K · strength_uefa_prior) / (n_eff + K)
```

Leagues with heavy European exposure (Netherlands, Portugal, Belgium) move freely to
their data. Leagues with almost no exposure (San Marino, Andorra) stay near the UEFA
prior, which is itself a defensible external estimate rather than noise. `K` is tuned
on held-out cross-league matches.

This is the same shrinkage pattern already used for `league_hfa` via
`LEAGUE_HFA_SHRINK_K` in `model.py`, so the machinery and idiom exist. Per-league goal
environment (`league_env_by_comp`) and home advantage (`league_hfa_by_comp`) extend
naturally alongside it.

**Additionally:** UEFA publishes *club* coefficients, not only country coefficients.
For a thin-sample team that is a repeat European participant — Sturm Graz among them —
the club coefficient is a materially better prior than the league mean. Use it as the
team-level seed where available.

---

## 5. Phased plan

Each phase has an explicit exit gate. No phase begins before its predecessor's gate
passes.

### P0 — Instrument and measure *(no new data; done)* ✅

Makes the failure visible without altering pricing. **Pricing deliberately continues**
so the build can be judged against live behaviour; the safeguard is that every
suggestion is flagged rather than blocked.

- `coverage.py` — per-team evidence tiers (`full` / `thin` / `defaulted`). A team with
  no domestic data is never `full` regardless of match count, which is the Sturm Graz
  signature a naive threshold would miss.
- `model.fit()` emits `team_evidence`; `model.predict()` attaches a `coverage` block.
  Probabilities are untouched.
- `edge.rows_from_odds` carries `evidence_tier` / `evidence_note` / match counts onto
  every bet row; `season.write_card` renders an Evidence column plus a
  **⚠ Low-evidence fixtures** section naming each under-evidenced team and why.
- `cross_league_eval.py` — the acceptance harness (§6), with the baseline stored to
  `data/cross_league_baseline.json`.
- Tests: `test_coverage.py` (13). Existing suites: 78 passed, no regressions.

**Gate: passed.** Coverage flows to the card; baseline recorded. Fleet state: 892 teams
— 301 `full`, 591 `thin`, 169 with no domestic data at all.

**But see §7 — the baseline result changes the case for the phases below.**

### P1 — Identity resolution *(done)* ✅

Escalated during implementation: this was **worse than fragmentation**. The overlapping
identities were not merely splitting ratings — they were *the same matches stored
twice*, with identical scores. `identities.dedupe_fixtures` keys on
(date, competition, home, away), so two spellings produce two keys and every duplicate
slipped through. The fit was double-counting them.

`club_identity.py` derives the merge map from **hard evidence** rather than string
similarity: two names are candidates when a match on the same date, competition and
score has one side spelled identically and the other differently. Five guards then
filter the candidates, each earning its place:

| Guard | Purpose | Caught in live data |
|---|---|---|
| G1 head-to-head | two identities that played each other are different clubs | `Bolton Wanderers` / `Wolverhampton`, `PSG` / `Racing Club de Lens`, `Auxerre` / `RC Strasbourg` |
| G2 country | never merge across countries | — |
| G3 reserve/youth | "II", "B", "U21" never merge into the senior side | — |
| G4 name affinity | evidence alone is not enough | `Reims` / `Stade Rennais` (18 collisions, different clubs) |
| G5 evidence floor | ≥3 corroborations unless cores are identical | — |

A second pass was needed after the first application. Pure **normalisation variants**
(accent/punctuation only) are invisible to the evidence collector — it compares
normalised names, so it sees them as already identical and never proposes them — yet
they remain distinct identities to the model. `Atletico Madrid` (128) and
`Atlético Madrid` (109) had split one club almost in half; `St Mirren`/`St. Mirren` and
`St Johnstone`/`St. Johnstone` likewise.

The similarity threshold was deliberately **not** loosened to absorb the residue
(`QPR`/`Queens Park Rangers`, `PSV`/`PSV Eindhoven`, `Rennes`/`Stade Rennais`), because
the genuine false positive `Reims`/`Stade Rennais` sits right beside them in similarity
space. Those nine are a reviewed `MANUAL_ALIASES` table instead.

**Results**

- **3,429 duplicate rows removed** (25,058 → 21,629) — ~13% of the dataset was
  double-counted in every fit.
- **93 identities merged** (923 → 830).
- Ratings consolidated: Bayern `1886`/`1976` → **1950**; Inter `1829`/`1685` → **1837**;
  Hearts `1717`/`1646` → **1652**; Atletico `1749`/`1681` → **1693**.
- Real-xG coverage `0.151` → **0.178** (dedupe retained the richer rows).
- Verified no data loss: 0 match-keys lost, 0 residual normalisation variants,
  1 pre-existing score conflict (unrelated to the merge).

**Root cause fixed.** `fetch.py` — the *daily* live path — wrote raw BSD spellings with
no canon applied (`seed_real.py` accepted a `canon` callable; `fetch.py` never had one).
Left alone, every nightly run would have silently rebuilt the duplication.
`club_identity.canonical_name()` now runs at ingest, and passes unknown clubs through
unchanged so the 55-league expansion cannot bend new clubs onto existing identities.

**Collateral fix:** the merge changed some canonical spellings, which broke the
football-data.co.uk market-history join (La Liga fell to 72%). `names.FDCOUK_ALIASES`
realigned; coverage back to 100%.

**Gate: passed.** 196 tests green (`test_club_identity.py` 17 new).
`validate --gate` PASS (Brier 0.6127 vs limit 0.6212).

### P2 — Registry expansion *(done)* ✅

**Scope correction — the 55-league target is not reachable on free sources.**

§3 claimed feasibility was "confirmed". It was not: that was inferred from the local
BSD cache, which only ever held the leagues we already fetch. A live pull of BSD's
catalogue (`/api/v2/leagues/`) returns **72 leagues in total**, of which just 18 are
UEFA-member domestic competitions — and **Austria is not one of them**. The league whose
absence started this entire investigation cannot be sourced from BSD at all.

Probing the alternatives changed the plan's centre of gravity: **football-data.co.uk is
the primary source for this work, not BSD.**

| Source | Coverage |
|---|---|
| fd.co.uk `/mmz4281/` | England ×5, Scotland ×4, Germany ×2, Italy ×2, Spain ×2, France ×2, Netherlands, Belgium, Portugal, Turkey, Greece |
| fd.co.uk `/new/` | Austria, Denmark, Finland, Ireland, Norway, Poland, Romania, Russia, Sweden, Switzerland |
| BSD | Bulgaria, Portugal Liga 3, plus the leagues already carried |

Austria alone has **14 seasons (2012/13–2025/26, 2,638 matches)** on fd.co.uk, well
beyond the 5 required, and those files carry closing odds as a bonus.

**Reach: 22 of 55 associations, 35 divisions.** Still missing, all regular European
participants: Czechia (rank 10), Israel (11), Ukraine (19), Serbia (20), Croatia (21),
Hungary (23), Slovakia (24), Azerbaijan (25), Cyprus (26), Moldova (30), plus 23 smaller
associations.

**On second tiers in all leagues** — requested, but only available for England,
Scotland, Germany, Italy, Spain and France. No free source carries the Eerste Divisie,
Portugal's Liga 2, or any second tier outside the big five. Portugal's *third* tier
(Liga 3) is available via BSD and is registered.

**Built**

- `uefa_registry.py` — all **55** associations registered with coefficient, rank, and
  per-division source mapping. Unavailable associations are recorded with an explicit
  reason rather than omitted, so the gap is tracked rather than silently absent. Clubs
  from them fall to the P0 low-evidence path and are flagged at bet-suggestion time.
- `data/uefa_coefficients.json` — all 55, with source URL, fetch date and an annual
  refresh cadence recorded in the artifact. Top-ten ordering **cross-checks exactly**
  against UEFA's published 2025/26 figures.
- `competitions.py` — 22 new competitions (46 total), with `fdcouk_code` / `fdcouk_new`
  / `fdcouk_league` source fields. Synthetic `api_id` range 9000+ since these are not
  api-football sourced.
- Strength priors derived from the coefficient, **anchored on the existing hand-set
  scale** (PL 1.00 at coef 89.16, Scottish Premiership 0.58 at 27.50). The anchoring
  reproduces the untouched hand-set values well (Serie A 0.931 vs 0.91; La Liga 0.896 vs
  0.92), so it is used only to seed *new* leagues — existing strengths are unchanged and
  `comp_strength.json` stays gated off.

**Deliberately not guessed:** `releg_spots` / `promo_spots` / `euro_spots` are left at 0
for the new leagues rather than invented. They only parameterise the P4.3 motivation
bands, and 0 already means "not a table", so the effect is that motivation does not yet
apply to these leagues. `teams_n` is set only where observed live. Populating the rest
is a P3 follow-up.

**Gate: passed.** Every UEFA domestic league BSD returns resolves to a Competition; the
only unresolved entries are national-team competitions (Euro, Nations League, WC
qualifying, U19) and two domestic cups, all correctly out of scope. No cross-continent
collisions — the exact-match rule holds with 22 more names in play, including the two
genuine traps: BSD's `Super League` is **Switzerland's** (Greece's is
`Stoiximan Super League`) and `Superliga` is **Romania's** (Denmark's comes from
fd.co.uk and is not in BSD). 230 tests green; `validate --gate` PASS.

### P3 — Data ingest *(done)* ✅

Four waves by coefficient rank, 5 seasons each, via `seed_fdcouk_leagues.py`.

| Wave | Leagues | Rows |
|---|---|---|
| 1 | Eredivisie, Liga Portugal, Belgian Pro League, Super Lig, Greek Super League, **Austrian Bundesliga** | 8,641 |
| 2 | Swiss Super League, Danish Superliga, Ekstraklasa, Eliteserien, Allsvenskan, Romanian Superliga | 8,329 |
| 3 | Serie B, Segunda División, 2. Bundesliga, Ligue 2, National League | 10,204 |
| 4 | Veikkausliiga, League of Ireland, Russian Premier League | 3,273 |

**fixtures.csv 21,629 → 52,074 rows.** Fitted matches 18,991 → 49,436. Fit time 2.0s.

#### The hard part was identity, not fetching

fd.co.uk spells it `Sturm Graz`; we already held `SK Sturm Graz` from BSD's European
coverage. Ingesting naively would have re-created the P1 fragmentation on precisely the
club this project exists to fix — and neither existing mechanism catches it. The P1
evidence matcher keys on same date/competition/score collisions, and an Austrian
Bundesliga match never collides with a Champions League one. `canonical_name()` passes
unknown names through by design.

So `propose_domestic_merges()` matches newly ingested domestic clubs against existing
Europe-only identities by name affinity, with guards. The first run proposed 30 merges
including several confident and wrong ones:

| Proposed | Reality |
|---|---|
| `Cercle Brugge` → `Club Brugge` | two different Bruges clubs |
| `AC Sparta Praha` → `Sparta Rotterdam` | Czechia vs Netherlands |
| `AEK Larnaca` → `AEK` | Cyprus vs AEK Athens |
| `IF Vestri` → `Estoril` | Iceland vs Portugal |
| `İstanbul Başakşehir` → `Istanbulspor` | two different Istanbul clubs |
| `GNK Dinamo Zagreb` → `Dinamo Bucuresti` | Croatia vs Romania |

There is no country field for a Europe-only club, so nothing in the data separates
these on name alone. Three mechanisms fixed it:

1. **Same-day conflict veto** — one club cannot play two matches on one day, so a date
   collision proves two identities are different clubs. This is the decisive guard.
2. **Already-present veto** — if the Europe-only club appears in the incoming league
   under its own spelling it is already reconciled and cannot be absorbed by a
   similarly-named neighbour. This is what saved Cercle Brugge.
3. **Auto-accept only at ≥0.80 or identical core.** Everything between 0.62 and 0.80
   goes to a review queue rather than being applied. The queue caught 6 more on wave 2 —
   3 genuine (`Mjällby AIF`, `Tromsø IL`, `Universitatea Craiova`) and 3 cross-country
   false positives.

**33 Europe-only identities reconciled** onto domestic names. Romania needed care:
three Craiova spellings exist and they are *not* one club — `Univ. Craiova` (the
European participant) and `U Craiova 1948` are different, while `U Craiova` is a
spelling fd.co.uk used for 7 matches in 2020 before switching.

**Also fixed:** the alias map is now **cumulative**. `build_alias_map` only proposes
merges it can still see evidence for, and an applied merge erases its own evidence — so
a rebuild would have silently shrunk the map, and `canonical_name()` reads it on every
daily fetch. The duplication would have quietly returned.

#### Results

| | Before P3 | After P3 |
|---|---|---|
| Fitted matches | 18,991 | **49,436** |
| Teams | 799 | 1,176 |
| Full-evidence teams | 225 | **585** |
| Europe-only teams (no domestic data) | 154 | **97** |
| UEFA matches priced on thin evidence | 776 | **390** |

**Sturm Graz: Elo 1517 → 1686.** Ajax 1448 → 1604. Rated on real domestic data instead
of 23 European matches against opposition the model also could not rate.

**Target subset improved: `uefa_cross_league` Brier 0.58777 → 0.57361 (−0.0142).**

**Regression check.** A pooled Brier comparison across the expansion is meaningless —
the match population changed, and the new leagues are intrinsically harder (ten arrive
via `/new/` files with **no shot data**, so the xg and xpress ensemble components have
nothing to work with there). The comparable check is pre-expansion leagues only, same
window: **0.61281 → 0.61591, +0.0031**, inside the ±0.01 gate tolerance but a real
directional effect — `global_avg` and the Elo pool mean now span 25 leagues rather than
5.

**Gate: passed.** `validate --gate` → **Brier 0.6168 vs baseline 0.6112, limit 0.6212 →
PASS** (n=47,598). This was initially unrunnable — the full-history walk-forward took
longer than a single sandbox command allows — and became runnable once the fold cache
below landed.

One honest caveat: the descriptive OU2.5 (+0.0127) and BTTS (+0.0084) deltas against
the promotion baseline are *worse*. Both markets lean on the shot-derived components,
and ten of the new leagues arrive with no shot data at all, so this is the expected
cost of the sparse sources rather than a modelling regression. Worth confirming against
the pre-expansion split before promoting anything that depends on those two markets.

### P3a — Walk-forward fold cache *(done)* ✅

Raised during review: *"do we need to run a full walk-forward on every run?"* No — and
the reason matters.

The pass was running **twice**: `update.sh` calls `validate --gate` on every invocation,
and on Mondays `season.py`'s weekly footer ran `walk_forward` again for the same number.
It also grew from ~30s to **~2 minutes** with the P3 data.

Almost all of it was redundant. A fold trains only on data before its test month, so
roughly **66 of 67 folds recompute byte-identical results daily**. So the fix is not
"validate less often" — it is "stop redoing identical work".

`walkforward_cache.py` keys each fold on a fingerprint of everything that can change it:
every fixture column feeding a fit or prediction **including row order** (Elo is
sequential), the source of `model.py` / `competitions.py` / `validate.py` / `coverage.py`,
plus `comp_strength.json` and `ensemble_weights.json` — data files that change behaviour
with no code edit at all. When in doubt it misses: an unnecessary recompute costs
seconds, a false hit costs correctness.

| | Time |
|---|---|
| Cold (empty cache) | ~2 min |
| Fully warm | **0.4s** |
| Typical daily run (trailing folds stale) | ~14s |

Verified **exact, not approximate**: cached and uncached runs produce identical rows and
identical metrics. Verified invalidation: perturbing `comp_strength.json` forces a full
recompute, and reverting restores the original fingerprint and metrics.

Two bugs found and fixed while building it:

- `prune()` originally deleted every entry not referenced by the current run, so a
  windowed call — which the eval harness and tuning helpers make — would have silently
  wiped the rest of the cache. Now scoped to superseded keys for months actually
  processed.
- Cache deletion is best-effort: a filesystem refusing an unlink is a disk-space
  problem, not a reason to fail validation.

The Monday duplicate now reads `validation_latest.json` instead of recomputing, with a
timestamp and an 8-day staleness guard — a month-old Brier displayed as current would
defeat the point of putting it on the card.

**Invalidation cascades forward**, which is correct but worth knowing operationally: a
backfill to a match on date D invalidates every fold after D. BSD's routine shot/xG
backfills touch recent matches only (cheap); a retroactive correction to an old season
will be slow, and should be.

### P4 — Hierarchical league-strength calibration *(done)* ✅

#### P4a — the estimator (`league_strength.py`)

Measured connectivity first, and it settled the design. Inter-league matches per league,
post-P3:

    Premier League 274 · La Liga 250 · Bundesliga 211 · Serie A 206 · Ligue 1 189
    Liga Portugal 127 · Eredivisie 112 · Austrian Bundesliga 60
    Ekstraklasa 6 · Scottish Championship 2 · Veikkausliiga 2
    Parva Liga, Liga 3, Russian Premier League, Scottish League One/Two: ZERO

Five leagues have **no** inter-league matches, so their strength is identified by
nothing at all. Fitting a free parameter there is fabrication, and the incumbent
estimator did exactly that.

The incumbent fails its own sanity check on four counts:

    La Liga (0.5835) <= Scottish Premiership (0.6979)
    Serie A (0.6235) <= Scottish Premiership (0.6979)
    Bundesliga (0.5451) <= Scottish Premiership (0.6979)
    Ligue 1 (0.6270) <= Scottish Premiership (0.6979)

Three causes: min-max rescaling lets the two extreme leagues set the scale; nothing
weights a league by how much evidence it has; and mean Elo is only comparable across
leagues *through* inter-league matches, so a weakly-connected league sits near BASE_ELO
by construction — not because it is average, but because nothing has measured it.

Replaced with `strength = (n_eff·observed + K·prior) / (n_eff + K)`, prior from the P2
UEFA coefficients, observed Elo mapped onto the strength scale via the **same anchors**
P2 uses (min-max would put the two components on incomparable scales).

**K is estimated, not chosen.** In this form K has a definition — within-league sampling
variance over between-league true variance — and both are measurable. Split-half
variance decomposition over 22 leagues gives **K = 10.5**. My first attempt grid-searched
K against "let strong leagues follow data, keep weak leagues near prior"; that objective
is monotone in K, so it always returned the grid's smallest value. It was measuring the
grid edge, not the data.

**Gate: PASS.** Premier League 1.000, Serie A 0.774, La Liga 0.770, Bundesliga 0.725, all
above Scottish Premiership 0.580; every second tier below its top tier. Zero-connectivity
leagues land exactly on their prior — the honest answer.

**Known limitation, visible in the output:** a UEFA coefficient is earned by the three or
four clubs a nation sends to Europe; mean league Elo describes the whole division. So
top-heavy leagues show a large one-directional gap (Eredivisie observed 0.495 vs prior
0.760; Liga Portugal 0.527 vs 0.723) while flat leagues agree closely (Scotland 0.580 vs
0.580, Denmark 0.551 vs 0.572). The prior is therefore biased upward for top-heavy
leagues. The low fitted K limits the damage, and it is a further reason the artifact
stays gated. A cleaner fix weights the prior by each league's European *participants* —
a later refinement, not a blocker.

`comp_strength.json` written with `"active": false`; verified `strength()` still returns
the hand-set values.

#### P4b — league-prior team seeding *(gated, but it works)*

This is the mechanism behind the original complaint. `elo = {t: BASE_ELO for t in teams}`
starts every club at the pooled average — and after P3 that pool spans 25 leagues from
the Premier League to the National League. An under-measured club was being seeded as
"average of everything", which is how Sturm Graz arrived at 1505.

With `league_seed=True` a club instead starts at its own league's coefficient-implied
level, and the goals prior comes from its league's scoring rate rather than `global_avg`.

**Walk-forward evidence, both arms on identical folds:**

| Window | n | Brier default → seeded | Log-loss default → seeded |
|---|---|---|---|
| 2025-07 → | 11,261 | 0.61799 → **0.61600** | 1.02952 → **1.02675** |
| 2024-07 → | 21,596 | 0.61730 → **0.61521** | 1.02865 → **1.02572** |

**−0.0021 Brier, −0.0029 log-loss, replicated across two windows.** This is the first
change in this project that improves pooled predictive accuracy rather than coverage or
data integrity.

#### PROMOTED — 2026-07-22

`model.LEAGUE_SEED_DEFAULT = True`. Recorded in code rather than toggled at runtime, the
same discipline as the evidence gate and the market blend.

**Full-history walk-forward after promotion: Brier 0.6168 → 0.6153** (n=47,598), gate
**PASS** against baseline 0.6112, limit 0.6212. Consistent with both windowed
measurements.

Three things were needed to make the promotion safe rather than just switched on:

1. **One source of truth.** `validate.walk_forward` now defaults to
   `model.LEAGUE_SEED_DEFAULT` rather than carrying its own `False`. Had those drifted
   apart, the gate would have scored a model nobody runs — a green light on an
   unvalidated production model. There is a test pinning it.
2. **Resolved before caching.** `league_seed=None` is resolved to a concrete bool before
   it reaches the walk-forward cache key; caching under `None` would collide two
   different models under one entry the moment the production default moved.
3. **Params carry their provenance.** Stored fits now record `league_seed_active`, so a
   cached `model_params.json` can never be mistaken for the other arm.

The pre-promotion model stays reproducible via `fit(league_seed=False)`, so the A/B that
justified this can be re-run at any time.

**Not promoted: `comp_strength.json` stays `active: false`.** The UEFA prior is biased
upward for top-heavy leagues (Eredivisie observed 0.495 vs prior 0.760), and until the
prior is weighted by each league's European participants rather than the whole division,
that bias would flow into live pricing. The estimator is sound; the prior needs work.

One cache-correctness note: `league_seed` had to be added to the walk-forward cache key.
A fit option absent from that key means the cache serves results produced under different
settings — a silent wrong answer rather than a slow one. There is now a test pinning it.

**249 → 301 tests green.** `validate --gate` PASS (Brier 0.6168, limit 0.6212).

### P6a — Keeping the expansion fresh *(done)* ✅

Found by asking whether the new leagues would actually stay current. They would not.

The P2 aliases mean the daily BSD fetch already resolves **14 of the 20** new leagues.
But **8 had no refresh path at all** — BSD does not carry them and P3 loaded them from
fd.co.uk as a one-off, so they were frozen at May 2026 with no upcoming fixtures and no
new results. **Austria was among them.** Left alone, P3's value would have decayed within
weeks of the new season starting, on the exact league that prompted the project.

- `seed_fdcouk_leagues.refresh()` — re-fetches current results for the BSD-less set,
  derived from the registry rather than hard-listed, so adding a BSD alias later drops a
  league out automatically. Idempotent: deterministic fixture ids mean a re-fetch updates
  in place (verified: second run adds 0 rows).
- Wired into `season.run_network_steps`, so it runs every pipeline invocation.
- `health.py` now reports **per-league** staleness. A single global "days since last
  result" hid exactly this failure — one league silently stopping while 40 others flow.

**Also fixed a bug P3 introduced.** `_season_of` applied Aug–May logic to every league,
but Norway, Sweden, Finland and Ireland play **calendar-year** seasons. Each real season
was split across two labels at the July boundary, merging two squads into one "season" —
Eliteserien read 19–20 teams for a 16-team league. That corrupts standings, promoted/
relegated detection and per-league-season scoring rates. `calendar_season` added to the
registry; **2,163 rows relabelled**; Eliteserien now reads 16.

### P6b — Motivation geometry *(done)* ✅

`releg_spots` / `promo_spots` / `euro_spots` were 0 for every expansion league, so the
P4.3 motivation bands silently did not apply to half the dataset. Now populated for all
34 league competitions — relegation derived empirically from season-to-season roster
departures (for a top tier, a departure is a relegation), European places from the UEFA
access list by coefficient rank. Note this is currently inert: `context_coef_club.json`
is `active: false`. It matters *before* any context fit, so the features are correct
whenever that happens rather than trained on zeros for half the leagues.

### P6c — Evidence-weighted ensemble *(done, LIVE)* ✅

Twelve competitions (~31% of fitted matches) arrive with **no shot data at all**. Their
`attack_xg`/`defence_xg` are identically zero — league-average by construction, not by
measurement — yet the xg and xgf components held **40% of the ensemble weight**, emitting
a flat matrix and diluting the goals and Elo components that do know something. That was
the mechanism behind the post-P3 OU2.5/BTTS regression.

Shot components are now scaled by `n/(n+8)` where n is the weaker club's shot-data
weight. Smooth rather than a cliff, because the spread is wide (Arsenal 71.8, Sturm Graz
7.1 from European matches only, 449 clubs at exactly 0).

| Metric | Without | With | Delta |
|---|---|---|---|
| Brier | 0.61519 | **0.61209** | −0.00310 |
| OU2.5 | 0.25664 | **0.25352** | −0.00312 |
| BTTS | 0.25429 | **0.25231** | −0.00198 |

Full-history gate: **Brier 0.6153 → 0.6119**. The regressions roughly halve —
OU2.5 +0.0126 → **+0.0074**, BTTS +0.0083 → **+0.0048**. The residual is the genuine cost
of shot-less leagues, not dilution.

### P5 — Variance inflation *(implemented, measured, left OFF)* ⛔

The plan's rationale was real when written: the P1 baseline showed thin favourites badly
overconfident (0.50–0.65 bucket predicted 0.541, observed 0.322, error **−0.221**).

It is no longer true. On current data that bucket reads **−0.002**, and 0.40–0.50 moved
from −0.095 to −0.019. Overall bias −0.0564 → −0.0208. P3's domestic data, P4b's league
seeding and the xg gating fixed it — **the cure for miscalibration on thin clubs was
measuring them, not widening their intervals.**

A/B confirms enabling it would now hurt:

| | Brier | Log-loss |
|---|---|---|
| OFF | **0.61209** | **1.02119** |
| ON | 0.61221 | 1.02146 |
| delta | +0.00012 | +0.00027 |

The mechanism and its harness stay in place — coverage can degrade — but it is off, and a
test asserts the A/B still favours OFF, so if that ever flips the suite prompts a rethink
rather than the question being re-argued from theory.

### P5 (original) — Graceful degradation, made permanent

- Formalise the evidence threshold (proposed: ≥20 matches within the last 2 seasons for
  `full`; below that `thin`).
- For `thin` and `defaulted`, inflate predictive variance toward the base rate in
  proportion to the evidence deficit, and surface the flag on the card and dashboard.
- Replace P0's blanket staking block with the threshold-based rule.

**Gate:** calibration curve on the `thin` subset is flat — i.e. when the model says 18%
for a thin-evidence team, it happens about 18% of the time.

### P6 — Promotion and ongoing operation *(done)* ✅

Per-league fetch scheduling and staleness monitoring landed in P6a. What remained was
the acceptance gate itself — *"14 consecutive green daily runs"* — which was
**unmeasurable**: `last_run.json` holds exactly one run, so `monitor.py` could answer
"did the last run work?" but never "has this been working?".

That distinction matters because the failure modes this project introduced are **slow**.
A league quietly stops updating. Coverage erodes as clubs drop below the evidence bar.
The fitted-match count drifts after a bad merge. Every individual run reports green;
only the sequence shows it.

**`run_ledger.py`** — every run appends one line to `run_history.jsonl`, carrying the
run outcome plus a coverage snapshot (fixtures rows, identities, full-evidence teams,
Europe-only count, stale leagues without a BSD path, gate Brier and verdict).

`readiness()` evaluates the preregistered gate:

| Check | |
|---|---|
| 14 consecutive green runs | a gap >30h breaks the streak — 14 greens over two months is not a healthy daily pipeline |
| latest run succeeded | |
| no stale leagues without a BSD path | the P6a failure mode, checked every run |
| validation gate passing | |
| full-evidence coverage not eroding | latest vs median of the last 10, 5% tolerance |

Design points worth recording:

- **Failures are recorded too.** A ledger of successes alone cannot measure a streak.
- **An empty ledger reports UNKNOWN, not healthy.** The first version passed the
  "no stale leagues" check on zero history — absence of evidence dressed as evidence of
  absence, the same mistake as a green gate on an unvalidated model.
- **Appending never raises.** Observability must not be able to fail the pipeline it
  observes.
- **`monitor.py` reports readiness but never fails on it.** A young ledger is not an
  incident, and escalating it would train the operator to ignore the alert.

**Current state: NOT READY, 0/14** — correctly, since the ledger starts empty. It fills
from the next scheduled run. Note `monitor` currently also reports a genuine failure: a
run started and was killed 11h ago, so the pipeline has not completed today.

**Still not promoted: `comp_strength.json`.** P4's estimator passes its plausibility
gate, but the UEFA prior is biased upward for top-heavy leagues (Eredivisie observed
0.495 vs prior 0.760). Weighting the prior by each league's European *participants*
rather than the whole division is the fix, and is the natural next piece of work.

---

## 6. Acceptance test — how we know it worked

Defined now, measured before any change, so the result is falsifiable rather than
narrative.

**Test set:** completed UEFA matches involving at least one team with no domestic-league
data. This set already exists in `fixtures.csv`: **1,094 matches**, of which **543 have
both sides unfitted**.

**Metrics:**

1. Brier score and log-loss on that 1,094-match subset, versus the P0 baseline.
2. Calibration curve on the subset — the decisive one. The current model is almost
   certainly *overconfident against* unfitted teams; the curve should flatten.
3. Ordinal sanity of fitted league strengths (§ P4 gate).
4. No regression in walk-forward Brier on the five currently fitted leagues.

**Sturm Graz vs Hearts is one row in this set.** The plan should not be judged on
whether it re-prices that single fixture more favourably — it should be judged on
whether the whole 1,094-match subset becomes properly calibrated. Tuning to the
anecdote is the main way this effort could go wrong.

---

## 7. P0 baseline result — read this before approving P1

P0 is built and the baseline is measured (`cross_league_baseline.json`, walk-forward
from 2024-07, 13,980 predictions). **The headline finding does not support the premise
this project started from, and that should change how we proceed.**

Underdog reliability — when the model gave an under-evidenced side probability *p* of
winning, how often did it actually win:

| Subset | n | Model said | They actually won | Bias |
|---|---|---|---|---|
| All competitions | 1,723 | 32.5% | 29.0% | **−0.035** |
| UEFA only | 497 | 26.8% | 26.6% | **−0.002** |

A positive bias would mean under-evidenced teams are being systematically under-rated —
the Sturm Graz hypothesis. The measured bias is **negative and near zero**. On 497 UEFA
matches the model is essentially perfectly calibrated on exactly the fixtures we
suspected. If anything it slightly *over*-rates weak-evidence teams; the 0.40–0.50 and
0.50–0.65 buckets show them underperforming their price by 6–7 points.

**Sturm Graz at 18% winning 4–0 is, on this evidence, variance rather than bias.** An
18% shot comes in roughly one time in five and a half. One result cannot distinguish a
broken model from an unlucky one, which is why the baseline was measured before
building anything.

Note also that pooled Brier is *better* on thin teams (0.606) than on full-evidence
ones (0.612) — thin fixtures are disproportionately mismatches, which are easy to call.
Any metric that ignores this would have produced a falsely reassuring green light in the
other direction.

### What still justifies the work

The premise was wrong; three of the findings were not, and they are independent of it:

1. **Identity fragmentation is real and definitely harmful** (§1.2). Bayern, Inter,
   Hearts and Ajax each have their ratings split across two identities. This degrades
   *currently fitted* leagues and has nothing to do with European coverage.
2. **764 matches could not be priced at all** and were silently dropped from both the
   card and the validation. Invisible non-coverage, not visible error.
3. **169 teams have no domestic data**, so their ratings are weakly identified even
   where they happen to land near the right value. Calibrated-on-average is not the
   same as reliable per fixture, and it is not a basis for staking.

### Post-P1 update — the identity fix improved exactly the target subset

Re-baselined on clean data. Pooled Brier barely moved (0.6127, gate PASS), but the
**UEFA cross-league calibration curve flattened substantially** — which is the metric
this project is actually judged on:

| Bucket | Before (corrupt data) | After (clean data) |
|---|---|---|
| 0.00–0.10 | −0.031 | **+0.015** |
| 0.10–0.20 | −0.006 | **+0.010** |
| 0.20–0.30 | −0.009 | **+0.002** |
| 0.30–0.40 | +0.061 | **−0.007** |
| 0.40–0.50 | −0.091 | **+0.026** |
| **Max abs error** | **0.091** | **0.026** |

UEFA underdog bias is now **+0.0015** — effectively zero. Note the pooled Brier
comparison across the fix is *not* interpretable (the evaluation set itself shrank by
the removed duplicates); the calibration curve is the meaningful readout, and it
improved by a factor of ~3.5.

### New finding — P5 should target thin *favourites*, not thin underdogs

The full underdog curve reveals a pattern the original plan assumed backwards. When the
model makes an under-evidenced team a **favourite**, they badly underperform:

| Bucket | n | Predicted | Observed | Error |
|---|---|---|---|---|
| 0.40–0.50 | 151 | 0.440 | 0.344 | **−0.095** |
| 0.50–0.65 | 59 | 0.543 | 0.322 | **−0.221** |

Low buckets are well calibrated; high buckets are badly overconfident. So the model is
not writing off teams it cannot see — it is *over-promoting* the ones it half-sees, most
likely because a thin record of favourable results is not shrunk hard enough. Sample is
small (n=59 in the top bucket) and domestic-heavy, so this needs confirming after P3,
but it points P5's variance inflation at the opposite end of the distribution from where
the original plan aimed it.

### Recommended change to the plan

Reorder. **P1 (identity resolution) should proceed immediately** — it is the
best-evidenced defect and it improves the existing model regardless of the expansion.
The 55-league ingest (P2–P3) is still worth doing for coverage, but it should be
justified as *"we cannot currently price 764 matches"* rather than *"our prices on
those matches are wrong"* — and the shrinkage design of P4 should be treated as
protection against introducing a bias that the data says is not there today, rather
than as a fix for one.

I would not tune anything to the Sturm Graz result.

---

## 8. Risks and open questions

| Risk | Mitigation |
|---|---|
| BSD lacks 5 seasons of history for smaller leagues | Verify per wave before committing; fall back to openfootball / football-data.co.uk where thin. Austrian Bundesliga absent from sampled cache — check first. |
| Long-tail leagues stay statistically weak regardless | Accepted by design — §4 shrinkage keeps them near the UEFA prior, and P5 flags them rather than pretending confidence. |
| Historical rewrite of `fixtures.csv` in P1 corrupts existing data | Snapshot before rewrite (a `.bak` convention already exists in `data/`); the identity merge must be a pure function that can be re-derived and re-run. |
| Name collisions multiply at 55 leagues (multiple "Dinamo", "Sparta", "Flora") | Canon must be keyed on (name, country), not name alone. The existing exact-match-only rule in `comp_from_bsd_league` is the right instinct and must be preserved. |
| Tuning to the Sturm Graz result | The §6 acceptance test is defined on 1,094 matches, and baseline is recorded before any change. |

**Open questions for review:**

1. Given §7, should P2–P3 (the 55-league ingest) proceed at full scope now, or should
   P1 (identity resolution) land and be measured first? P1 is the defect with the
   clearest evidence behind it; the ingest is a coverage argument, not an accuracy one.
2. `MIN_RECENT_MATCHES = 20` over a two-season window currently classes 591 of 892 teams
   as `thin`. That is defensible — much of the tail is historical clubs no longer being
   priced — but if the low-evidence section on the card proves noisy in practice, the
   threshold is the dial to turn.
3. The P5 variance inflation now needs re-justification. The baseline says the model is
   *already* well calibrated on thin fixtures, so widening those intervals would make it
   worse, not better, on the measured evidence. Recommend holding P5 until after P3,
   then re-measuring: inflation is the right protection for genuinely *new* teams
   entering from newly ingested leagues, but not for the current thin population.
