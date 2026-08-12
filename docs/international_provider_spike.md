# Provider spike: BSD international coverage

**Date:** 8 August 2026 · **Provider:** BSD (Bzzoiro Sports Data) · **Cost:** free, key already held
**Plan reference:** `plans/international_football_module_plan.md` §8 step 3
**Reproduce:** `python3 -m scripts.international.fetch_fixtures --replay --dry-run`
(offline, from the raw observations stored under `data/international/raw/bsd/`)

This is the evidence phase for one provider. It is **not** a coverage SLA — that needs a
live window, which starts 21 September. What it does establish is what BSD carries today.

---

## Verdict

**BSD is a viable primary fixtures source, with one gap that matters more than the rest.**

| | |
|---|---|
| Reachable, key valid | Yes |
| Catalogue | 79 leagues, of which **16 are senior men's international** |
| Upcoming international fixtures | **280** (255 after FIFA-scope filtering) |
| Inside the 21 Sep – 6 Oct window | **157** |
| Kick-off timestamps | **100% present**, offset-aware |
| Stable provider event IDs | **100% present** |
| Neutral-ground flag | 100% present (all `false` in the current set) |
| Cost | £0 |

**The gap:** AFCON 2027 qualifying — which begins **23 September**, inside the very first
window — is **absent**. It is roughly 144 matches across 48 teams in 12 groups. BSD's
African competition is `Africa Cup of Nations 2023`, a past-edition league with zero
upcoming fixtures.

---

## Competition coverage

BSD files internationals under continental country labels, not under `International`.
An initial filter on `country == "International"` returned only 2 leagues and would have
produced a badly wrong "BSD does not cover internationals" conclusion.

**Carried, with upcoming fixtures:**

| Competition | Upcoming | In Sept window |
|---|---|---|
| UEFA Nations League | 156 | 105 |
| CONCACAF Nations League | 51 | 51 |
| AFC Asian Cup | 36 | 0 |
| International Friendly Games | 12 | 1 |

**Carried, but currently empty** (past editions or between cycles): World Cup 2026;
World Cup Qualification for **all six** confederations (UEFA, CONMEBOL, CAF, AFC,
CONCACAF, OFC); UEFA Euro 2024; Africa Cup of Nations 2023; Copa America 2024;
CONCACAF Gold Cup 2025.

**Notable against The Odds API:** BSD carries World Cup qualifying for **all six**
confederations. The Odds API carries Europe and South America only. On qualifying
coverage BSD is strictly better, and free.

**Structural risk:** continental tournaments are **per-edition leagues** — "UEFA Euro
2024", "AFC Asian Cup 2023", "Copa America 2024". They do not roll forward. Each new
edition appears as a new league that must be mapped, and the missing AFCON 2027
qualifying is the first instance of that failure mode, not a one-off.

**Only 12 upcoming friendlies.** For a window that will feature dozens, friendlies
populate late. Whether they arrive in time to price is a question only the live window
answers.

---

## Data quality findings

### 1. Team naming breaks scope filtering silently

The first run dropped **24 fixtures involving FIFA members** as "out of scope" because
BSD's spellings did not match ours: `Czechia`, `Türkiye`, `Ireland`,
`Bosnia & Herzegovina`, `US Virgin Islands`. Aliases added; the parser now canonicalises
before the scope test. Pinned by `tests/international/test_provider_bsd.py`.

### 2. The friendlies league contains clubs and youth sides

`International Friendly Games` included **Albirex Niigata** (a Japanese club),
**Miami United** and **Jamaica U20**. Two layers of defence now:

- a name heuristic rejects explicit markers (`U20`, `B`, `Women`, `XI`);
- the team registry rejects anything it does not know.

The heuristic does **not** catch a club with an ordinary name — Albirex Niigata passes it
and is stopped by the registry, which is why both layers exist. Such fixtures are
**quarantined and reported**, never silently dropped.

### 3. Venue coverage is the weakest link

| | |
|---|---|
| Venues in catalogue | 1,786 |
| With coordinates | 784 (44%) |
| Timezone resolved | 1,247 (70%) — 782 from coordinates, 465 inferred from country |
| **Fixtures with no venue at all** | **183 of 255** |

BSD supplies `venue_id: null` for most Nations League fixtures. Those fixtures fall back
to the UTC date for `local_date` and are flagged in `conflict`. **202 of 255 fixtures
currently carry that flag.**

This matters because a wrong local date is exactly what duplicated two World Cup fixtures
in `results.csv`. Demonstrated end to end:

```
local_date("2026-07-07T00:00:00Z", venue=AT&T Stadium) -> ("2026-07-06", "America/Chicago")
local_date("2026-07-07T00:00:00Z", venue=None)         -> ("2026-07-07", "")
```

Country-level inference was added for venues without coordinates, but **only where every
coordinate-bearing venue in that country agrees on one timezone**. Multi-timezone
countries (USA, Russia, Brazil, Australia) stay unresolved rather than guessed.

### 4. Two API generations disagree about time

- v1 `/api/events/` serves `event_date` with a **+04:00** offset and embeds league, venue
  (with lat/lon) and odds.
- v2 `/api/v2/events/` serves **+00:00** and returns `league_id` / `venue_id` as integers.

Both are offset-aware, so UTC conversion is unambiguous. The pre-existing
`scripts/worldcup/live_data.py::_local_match_date()` converts every kick-off to a single
hardcoded US Pacific timezone — correct for a North American tournament, wrong worldwide.
The new path does not use it.

### 5. Odds are embedded, for one implied book

v1 event payloads carry `odds_home`, `odds_draw`, `odds_away`, `odds_btts_*`,
`odds_over_*`, `odds_under_*` inline. That is free odds on international fixtures, which
The Odds API does not offer for friendlies at all. **But it is a single implied price with
no bookmaker attribution, no line movement and no closing price**, so it cannot support
closing-line-value measurement. Not yet assessed; a separate spike.

---

## Against the pre-committed thresholds

From plan §10. Most cannot be scored until a live window.

| Measure | Threshold (Go) | Now | Status |
|---|---|---|---|
| Fixture recall vs manual sample, 14 days ahead | ≥95% | not measured | **needs the window** |
| Duplicate rate after reconciliation | <0.5% | **0%** (255 unique IDs) | **Go** |
| Neutral-venue accuracy | ≥98% | flag present on 100%, correctness unverified | needs the window |
| Cancellation reflected within | 24h | not measured | needs the window |
| Competitions with usable odds at 48h lead | ≥20 | 4 with fixtures; odds quality unassessed | **at risk** |
| Odds licensable for our use | written confirmation | not sought | open |

---

## What this changes in the plan

1. **BSD is confirmed as a viable fixtures source.** It is not confirmed as an odds source.
2. **The "capped at nine competitions" worry is eased for fixtures** — BSD has all six
   qualifying confederations — but qualifying is between cycles, so that advantage is
   latent until 2027.
3. **AFCON 2027 qualifying must be sourced elsewhere or requested from BSD**, before
   23 September. This is now the single most actionable item.
4. **Venue data is a real dependency**, not reference polish. 183 fixtures with no venue is
   183 fixtures whose local date we are guessing.
5. Provider comparison is still owed: openfootball, TheSportsDB and football-data.org are
   untested, and the plan is explicit that no provider is primary until they are scored
   against each other.
