# Horse-racing provider decision

Date: 2026-07-14

## Decision

Use **rpscrape** as the user-approved free research provider, complemented by
**Betfair Historical BASIC** for genuine cutoff-market replay. Keep **The Racing
API** as the supported paid API path.

The rpscrape integration is deliberately isolated in `.providers/rpscrape`,
pinned to commit `f2c6977cd9a8b6ca20af823b3b2fac0ca39bce51`, and excluded from
version control. A hash-verified local patch bounds an upstream infinite retry
when a result page lacks the expected marker. The adapter enables and retains
Racing Post race/course/horse/trainer/jockey IDs, archives raw CSVs by checksum,
and excludes starting prices from canonical cutoff odds.

The accepted tradeoffs are that rpscrape has no declared software licence,
depends on Racing Post page structure, and supplies retrospective snapshots
rather than field-level publication timestamps. Its validation grade is always
`research_only`.

### Initial research dataset and result

The first bounded pull covers 2026-06-01 through 2026-07-13:

- 842 races and 7,844 runners;
- 689 GB and 153 Irish races;
- 735 turf and 107 all-weather races;
- no duplicate race or runner-version keys;
- all 842 races have exactly one winner;
- 22.46% of runners lack an official rating.

With the first 500 races as the warm-up and 342 strictly later held-out races:

| Forecast | Log loss | Race Brier |
|---|---:|---:|
| Conditional-logit model | 2.05279 | 0.85533 |
| Uniform field | 2.12356 | 0.87310 |
| Official-rating softmax | 2.23954 | 0.90195 |
| De-vigged official starting price | 1.72629 | 0.77510 |

Paired race bootstrap differences in log loss were model minus uniform
`-0.07077` (95% CI `[-0.11645, -0.02519]`), model minus official rating
`-0.18675` (`[-0.26304, -0.11028]`), and model minus starting price `+0.32650`
(`[+0.26779, +0.39303]`). The model adds signal over elementary baselines but
does not beat the closing market. This is not evidence of a profitable edge.

## Betfair Historical BASIC complement

`horse_racing/providers/betfair_historical.py` streams the downloaded tar
directly, so the multi-gigabyte archive is never duplicated on disk. It retains
only GB/IE, single-winner WIN markets that make an exact, unique match on
jurisdiction, date, course, scheduled off and the complete active runner-name
set. Non-runners and ambiguous identities are quarantined rather than guessed.

The stream is reconstructed as a market-state cache. Each canonical row uses
the final full-board snapshot timestamp no later than the 15-minute cutoff;
unchanged LTP deltas persist in that state. Maximum component-LTP staleness is
recorded per matched market. BASIC has no available-size ladder, so
`available_size` remains blank and the source can never make a recommendation
eligible. It is a point-in-time observational benchmark, not evidence that a
quoted price was executable for a proposed stake.

The supplied `horse_racing/data.tar` is 3,451,060,224 bytes with SHA-256
`1496927c5effdd1e0f8dd84e5cbdc3162fe2f15ac02365da5aa8757a67fe3a76`.
Its readable prefix contains 695,763 members and 813 distinct path dates, but it
is truncated inside
`BASIC/2025/Mar/17/34126467/1.240920927.bz2`; there are no tar end markers.
The manifest records `complete=false`, the error and the last readable member.
The isolated 2023-06-07 validation joined Betfair market `1.215008786` exactly
to Racing Post race `840683`: all 15 runners matched, the board snapshot was
44.269 seconds old at cutoff, raw LTP overround was 1.01026, and the winner's
de-vigged market probability was 0.20201 (log loss 1.59944). Component LTPs had
at most 360.032 seconds of staleness. This validates parsing, cutoff replay and
identity joining, but one race is not predictive or economic validation.

The continuation `horse_racing/data1.tar` is a complete 172,981,760-byte POSIX
tar with SHA-256
`119f77c039618e4016317e9a57f5be3f37eb9b4758bc4cfe7b1733cc4db89d1c`.
Its 17,744 readable members cover 2025-03-17 through 2026-07-09 and overlap the
main Racing Post research dataset. The exact identity join found 690 markets.
After enforcing a maximum one-hour component-LTP age, 668 cutoff boards covering
6,010 runners remain. Of the 755 canonical races through the archive endpoint,
668 have accepted boards (`88.48%`). Across all archive WIN markets considered,
quarantines were 69 without canonical date scope, 305 without a scheduled-off
candidate, 460 without a course candidate, 37 with a runner-field mismatch and
22 with a stale component LTP.

On the 235 held-out walk-forward races having complete fresh Betfair boards:

| Forecast | Log loss | Race Brier |
|---|---:|---:|
| Conditional-logit model | 2.00412 | 0.84600 |
| Betfair LTP at 15-minute cutoff | 1.73272 | 0.77814 |
| Official starting price, same races | 1.71370 | 0.77184 |

Paired model-minus-Betfair log loss was `+0.27140` with 95% bootstrap CI
`[+0.18728, +0.35123]`: the model is decisively worse than the cutoff market.
Betfair-minus-starting-price was `+0.01902`, CI
`[-0.00297, +0.04096]`; this sample does not establish a meaningful difference
between the 15-minute LTP benchmark and official starting prices. Neither result
tests execution, commission, slippage or profitability.

Selection diagnostics are now reported alongside the comparison. The 235
matched held-out races have mean field size 8.68 and model log loss 2.00412; the
107 unmatched held-out races have mean field size 9.21 and model log loss
2.15968. The market subset is therefore measurably easier and must not be treated
as representative of all live races.

An adversarial follow-up added additive checksum-owned multi-archive upserts,
preservation of Betfair odds across rpscrape refreshes, strict tar end-marker
completeness, canonical-cutoff alignment, bounded compressed/decompressed member
and JSON-line sizes, and whole-checkout integrity verification for rpscrape.

## Paid API alternative

The Racing API was selected because its official API exposes provider-stable
race, horse, trainer, jockey and course IDs; GB/IE historical results; tracked
historical racecards from 2023; official ratings; and bookmaker price-change
history from 2025. The provider documents Basic authentication and paid plan
boundaries. The adapter uses only documented endpoints and does not scrape.

Official references:

- API documentation: https://api.theracingapi.com/documentation
- Data coverage: https://www.theracingapi.com/data-coverage
- Terms: https://www.theracingapi.com/terms-of-service
- Service status: https://status.theracingapi.com/
- Betfair developer and historical-data overview:
  https://developer.betfair.com/

## Contract boundaries

- Scope is GB/IE flat racing only.
- Credentials are read from `THE_RACING_API_USERNAME` and
  `THE_RACING_API_PASSWORD`; they are never command arguments or archived.
- Every response is archived with endpoint, non-secret parameters, retrieval
  time and payload SHA-256 before canonicalization.
- Provider IDs remain canonical. Race-specific runner IDs are deterministic
  `race_id:horse_id` values.
- Races with unsupported surfaces, fewer than two runners, or missing stable
  horse/trainer/jockey IDs are quarantined by exclusion and counted in the
  manifest rather than name-matched.
- The API's bookmaker odds do not include available size. They are retained for
  market-baseline evaluation but fail the recommendation eligibility check.

## Historical validation limitation

The historical racecard schema does not publish field-level update timestamps,
and the results schema does not publish result-availability timestamps. The
adapter therefore uses two explicit conservative research assumptions:

1. historical racecard fields are pinned to the configured prediction cutoff;
2. results become available at 00:00 UTC on the following day.

Both are recorded in `provider_manifest.json`, whose `validation_grade` is
`research_only`. The validation command may report chronological metrics on
this dataset but refuses to write or pass a production baseline unless the
manifest is `point_in_time`. Live pre-race accumulation and post-race polling
are required to establish that stronger grade.

## Acquisition still needed

For continuity with the Racing Post sample, Betfair data from 2026-07-10 onward
is still missing. The larger limitation is unchanged: Racing Post declarations
and result availability remain `research_only`. A fitted artifact cannot become
deployment-eligible without genuinely timestamped declarations/results and an
executable exchange feed containing available sizes. The current model also
needs substantial feature/model improvement before economic testing—the
point-in-time market comparison now demonstrates a material forecasting gap.
