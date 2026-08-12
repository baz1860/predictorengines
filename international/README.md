# International football module

Scope contracts, fixture ingest and data integrity for matches between national
teams. **No modelling code lives here** — the Elo / goal model / Dixon-Coles path
stays in `engines/worldcup` until the compatibility harness proves a move is safe.

Plan: `plans/international_football_module_plan.md` (revision 3.1)
Provider evidence: `docs/international_provider_spike.md`

---

## Why this exists

The World Cup module's model was always international — it trains on 49,523 matches
back to 1872. What was not international was everything around it: fixtures, scope,
identity, competition weights and the product.

Two live defects motivated most of the design:

1. **Two World Cup fixtures were duplicated in `results.csv`** — the same match
   recorded on 6 July (local) and 7 July (UTC), because the merge key included the
   date. The blank row then satisfied `load_matches()`'s test for an upcoming
   fixture, so a finished match was presented as forthcoming for a month.
2. **The validation gate never blocked.** `update.sh` caught its failure and
   carried on, by design, with a comment saying so.

Both are fixed. The architecture below is mostly a set of guards against them
recurring at 200 competitions instead of one.

---

## Modules

## Time: one rule

**Every instant is stored, compared and reasoned about in UTC**, through
`international/timeutil.py`. There is exactly one exception, and it is a boundary
rather than a compromise:

> `data/results.csv` is dated in **local** time.

That is the upstream martj42 convention and we do not control it. Writing UTC dates
there would put us a day ahead of upstream on every evening kick-off in the
Americas — which is exactly the disagreement that duplicated two World Cup
fixtures. So `local_date()` exists, is used only at that boundary, and always
records which timezone produced it.

Consolidating removed **30+ ad-hoc conversions across 10 files**. Three silent bugs
had already come out of the gaps between them: a naive/aware comparison that raised
only on an untested path, a `tz_of()` returning the string `"nan"` that then fell
back to UTC without saying so, and a `NaN >= interval` test that scheduled nothing
at all. The gate now enforces the invariant — every stored `kickoff_utc` must end
`+00:00`.

Vocabulary, chosen deliberately:

| Call | Returns | Use for |
|---|---|---|
| `to_utc(x)` | tz-aware UTC | anything stored |
| `naive_utc(x)` | tz-naive UTC wall time | **only** comparisons against `results.csv` |
| `utc_iso(x)` | canonical string ending `+00:00` | serialisation |
| `local_date(x, tz)` | `(date, tz_used)` | the `results.csv` boundary |

An empty `tz_used` is a **flag, not a default**: that date is right for roughly half
the world and a day out for the rest.

| Module | Responsibility |
|---|---|
| `timeutil.py` | The single source of truth for time. Nothing else may convert timezones |
| `taxonomy.py` | Every competition label → category + importance weight. Two profiles: `legacy` (reproduces `predictor.py` exactly) and `v1` (the challenger). Pattern-matched labels are marked **provisional** so a new competition cannot silently acquire a guessed weight |
| `registry.py` | Effective-dated team registry. 211 current FIFA members, all six confederation counts correct. **Fails closed**: an active unclassified team raises rather than being silently in or out |
| `identity.py` | Canonical fixture identity that survives UTC/local date disagreement. Provider ID > kickoff timestamp > date+signature |
| `fixtures.py` | Duplicate detection and the invariants the gate enforces. Conservative: only blank-vs-scored pairs auto-resolve; two scored rows go to a human |
| `venues.py` | Venue timezones from coordinates, with country-level inference where every known venue in a country agrees. Multi-timezone countries stay unresolved rather than guessed |
| `home_venues.py` | Where each team *actually* plays at home, as a distribution over a decade of matches, with coordinates, elevation and timezone. Feeds both timezone fallback and the altitude model |
| `store.py` | Append-only raw observations + canonical fixture store with lifecycle (`scheduled → played / postponed / cancelled / abandoned`) |
| `odds.py` | Odds snapshot history. Records **absence** as a real observation |
| `coverage.py` | Evidence tiers (`full` / `thin` / `defaulted`), including opponent connectivity — a team that plays only its neighbours is not well-anchored however many matches it has |
| `gate.py` | Blocking health gate. Daily mode tolerates a known backlog; `--strict` does not |
| `providers/bsd.py` | Primary fixture source. Pure parsers + network functions kept separate so parsing is testable offline |
| `providers/caf.py` | AFCON 2027 qualifying, **derived from the draw rather than fetched** — every commercial route is paid, but a 4-team double round-robin is deterministic. Cross-checked against a published list |
| `providers/thesportsdb.py` | Cross-check only. Free tier returns 1 event per call and cannot enumerate leagues |

---

## Data

All under `data/international/`:

| File | Contents |
|---|---|
| `team_registry.csv` | 338 teams; 211 current FIFA members, 51 non-FIFA, with reasons |
| `fixtures.csv` | Canonical fixtures, keyed on `fixture_id` |
| `venues.csv` | 1,786 venues; 1,247 with a resolved timezone |
| `odds_snapshots.csv` | Append-only price history, including absence rows |
| `fixture_exceptions.csv` | Adjudicated duplicate pairs; 8 pending review |
| `raw/` | Append-only provider payloads. **Gitignored, ~23MB, unreconstructable** — back this up before an international window |

---

## Commands

```bash
# fixtures
python3 -m scripts.international.fetch_fixtures --venues     # build venue table
python3 -m scripts.international.fetch_fixtures --write      # fetch + store
python3 -m scripts.international.fetch_fixtures --replay     # reparse offline

# odds (self-throttling: only fixtures within 14 days of kickoff)
python3 -m scripts.international.fetch_odds --write
python3 -m scripts.international.fetch_odds --report

# integrity
python3 -m international.gate                # daily, blocking
python3 -m international.gate --strict       # release; fails on the backlog

# analysis
python3 -m international.coverage
python3 -m scripts.international.compare_providers
python3 scripts/analysis/taxonomy_sensitivity.py
python3 scripts/analysis/international_skill_audit.py

# results history
python3 -m scripts.international.promote_results --dry-run

# scheduled
./scripts/international/refresh.sh fixtures|odds|weekly
```

---

## Design rules

**Fail closed.** An unclassified active team, an unmapped competition or an
unrecognised fixture name stops the run or is quarantined. Nothing is guessed
silently. The cost of a wrong guess here is a confident, wrong price.

**Absence is data.** "We asked and there were no odds" is recorded as a row.
Otherwise a gap in the data is indistinguishable from a gap in collection, and only
one of those is a finding.

**Raw before parsed.** Every network payload is stored verbatim before anything
interprets it, so a coverage claim can be rechecked after the window closes.

**Challenger, not edit.** The `v1` competition weights went through the challenger
path rather than being edited into `predictor.py` — and were **rejected**. They move
a 1X2 probability by up to 15 points and change the bet decision on 4% of fixtures
(`taxonomy_sensitivity.py`), while making prediction very slightly *worse*
(blend Brier 0.5097 → 0.5101, `taxonomy_challenger.py`). Moving prices without
improving them is the worst of both worlds, so **`k_for(..., "legacy")` remains the
production profile**. The `v1` profile stays in use for competition *categories* and
*bettability*, which are unaffected by that result.

Elo turns out to be self-correcting here: a competition weighted too low still
converges, because teams play across a mix of competitions. The "34.3% of matches on
a default weight" statistic was an accurate description of the code and a poor
predictor of harm — worth remembering before the next alarming-looking metric.

**Two layers on identity.** A name heuristic rejects obvious non-national sides;
the registry rejects anything it does not know. The heuristic alone missed a
Japanese club sitting in BSD's international friendlies league.

---

## Known gaps

- **CAF fixtures are derived, not observed.** 144 AFCON 2027 qualifying fixtures are
  computed from the group draw because every commercial route is paid. Pairings are
  cross-checked against a published list; **kick-off times and venues are unknown**,
  so each row is dated to its matchday window and flagged. Good enough to predict,
  not precise enough to time an odds poll or settle a bet.
- **28 of 255 BSD fixtures still have no resolvable timezone** (down from 202).
  BSD supplies no venue for most Nations League fixtures; where the match is not on
  neutral ground the home team's usual ground now fills the gap, and the derivation
  basis is recorded in `conflict` so an inferred date is never mistaken for a
  observed one.
- **No odds exist yet.** BSD returned nothing for September fixtures at six weeks
  out. Until prices arrive we cannot measure edge, which is the project's central
  open question.
- **8 duplicate pairs await adjudication**; the strict gate fails until they are resolved.
- Only one provider is properly tested. TheSportsDB's free tier is too capped to
  serve as a real second source.
