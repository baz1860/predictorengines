# Non-UEFA leagues — status and how to finish the ingest

## What this is for

You want to **bet** these leagues. That is the right reason, and it works —
but read the limitation below before using the ratings for anything else.

## What's built

| | |
|---|---|
| Country-scoped identities | `canonical_name(name, country=...)` refuses cross-confederation merges |
| 15 competitions registered | MLS, USL Championship, J1, K League 1, Brasileirão A/B, Primera A, Saudi Pro League, CSL, Botola Pro, Liga MX Apertura + Clausura, Copa Libertadores, Copa Sudamericana, CAF Champions League |
| BSD aliases | exact-match only, verified against every historic collision name |
| Shrink guard | `write_fixtures` refuses a write that drops >50% of rows |
| Tests | `test_nonuefa.py` (17) |

A 10-day slice is already ingested as proof the path works — 13 competitions,
no duplicate identities, Athletic Bilbao uncontaminated.

## Finish the backfill

The BSD merge onto a 55k-row fixtures.csv takes ~30s per slice, which exceeded
the sandbox's per-command limit. On your machine there is no such limit:

```bash
# Backfill in slices. Each is idempotent — re-running merges rather than
# duplicating, because fixture ids are deterministic.
for range in 2024-01:2024-06 2024-07:2024-12 2025-01:2025-06 \
             2025-07:2025-12 2026-01:2026-07; do
  from=${range%%:*}-01; to=${range##*:}-28
  python3 -c "
from club_soccer import fetch as F
F.fetch_fixtures(current=True, date_from='$from', date_to='$to', enrich_stats=False)"
done

# Then report unresolved identities, refit, and check
python3 -m club_soccer.identity_review              # read-only report
python3 -c "from club_soccer import model as M; M.save_params(M.fit())"
python3 -m club_soccer.validate --gate
```

**`current=True` is not optional.** `current=False` does not mean "dry run" —
it means "don't merge, then overwrite the file with just this slice". That is
what truncated fixtures.csv from 55,329 rows to 2,704 during this build. The
shrink guard now refuses it, but use `current=True` regardless.

After the backfill, run `identity_review` and check the new rows. The report
does not mutate fixtures; confirmed aliases are added explicitly to
`data/club_alias_map.json`.
Brazil, Portugal and Spain share a lot of club names — América, Nacional,
Inter, Sport, Vitória, Santos — and the country guard only blocks merges onto
clubs whose country we already know.

## The limitation, stated plainly

**Within-league ratings: sound.** Elo and the goals model rank teams against
their own league perfectly well without any external reference, and BSD's data
here is good — 95–100% xG coverage, better than the ten fd.co.uk European
leagues that carry no shot data at all.

**Cross-confederation comparisons: not supported.** The UEFA expansion worked
because European competition *links* those leagues — the Premier League alone
has 274 inter-league matches, which is what puts every European league on a
single Elo scale. MLS, J1 and K League have essentially **zero** competitive
matches against the fitted European set. The only bridge is pre-season Club
Friendlies, which are worthless for rating (rotated squads, no intensity).

So "is this MLS side better than this Bundesliga side" is **not answerable**
from this data, and the `strength` values for these competitions are
placeholders on an unanchored scale — not comparable with the European ones.
Copa Libertadores and CAF Champions League do link their own confederations
internally, so relative strength *within* South America and Africa is
identifiable.

If an intercontinental fixture ever needs pricing (Club World Cup), the P0
evidence tiers will flag it, but treat the number with real suspicion.

## Structural caveats

- **Liga MX** plays Apertura and Clausura — two championships per calendar
  year — and is registered as two competitions. The season model still assumes
  one competition-season per year, so promotion priors and Elo season
  regression are only approximately right for these two.
- **MLS and USL** have no relegation (conferences and playoffs instead), so
  `releg_spots=0` is correct rather than unknown — motivation bands built on a
  relegation battle do not apply.
- Calendar-year seasons are handled (`calendar_season=True`).
