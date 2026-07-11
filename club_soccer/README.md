# Club Soccer Engine

Predicts club football across the Premier League, Championship, League
One/Two, the four Scottish divisions, Bundesliga, Serie A, Ligue 1, La Liga,
the major domestic cups, and the Champions/Europa/Conference Leagues.

## Use it

One command. It refreshes results/fixtures/absences/squads/odds, refits the
model, prices upcoming matches, and writes a card:

```bash
python3 -m club_soccer.season
```

That writes [`data/card.md`](data/card.md) — the file you normally read. It
has:

- **Freshness header** — days since the last result, upcoming fixture count,
  absence rows recorded today, odds-snapshot age.
- **Next 7 days** — grouped by day then competition: model H/D/A%, fair
  odds, an availability note (▲/▼ multipliers + lineup confidence) when
  players are missing, a rest/congestion note, and league positions for
  domestic league fixtures.
- **Backed bets** — the market/side/odds/edge/stake the model actually
  recommends (quarter-Kelly, haircut by lineup confidence), with a footnote
  for anything the market-model's do-not-bet filter suppressed.
- **Transfers & absences** — recently observed squad changes and notable
  absences for upcoming matches.
- **Mondays**: a walk-forward validation + market-backtest summary appended
  to the footer.

Other useful flags:

```bash
python3 -m club_soccer.season --fast          # skip the model refit
python3 -m club_soccer.season --no-network    # cached data only, no API calls
```

### Scheduling

- **App open**: the app's own scheduler (`data/update_schedule.json`) can
  fire `./club_soccer/update.sh` on whatever cadence you configure —
  matches the pattern every other engine uses (see `golf/update.sh`).
- **App closed** (optional): a launchd job runs it standalone at 07:30
  local time.
  ```bash
  # Edit the WorkingDirectory in the plist to this repo's absolute path first.
  cp club_soccer/com.sportspredictor.clubsoccer.plist ~/Library/LaunchAgents/
  launchctl load ~/Library/LaunchAgents/com.sportspredictor.clubsoccer.plist
  # to stop: launchctl unload ~/Library/LaunchAgents/com.sportspredictor.clubsoccer.plist
  ```
  Logs go to `/tmp/club_soccer_season.log`.

`club_soccer/update.sh` is a thin wrapper around the same command for the
app's scheduler / launchd-style invocation (`./club_soccer/update.sh --fast`).

## How it's built

Everything below is what `season.py` calls, in order. You don't normally
need to run these individually — they're here for debugging, backfills, and
one-off fits.

### Data

`data/fixtures.csv` is the source of truth (BSD — free, no rate limits, key
required). `python3 -m club_soccer.health` checks it for future-dated
"finished" rows, duplicate provider IDs, and duplicate canonical match
identities. Provider IDs are not assumed to be globally unique: rows are
reconciled on date + competition + home + away, retaining the richest detail
record.

```bash
python3 -m club_soccer.fetch --current           # results + upcoming fixtures
python3 -m club_soccer.fetch --current --status upcoming \
  --date-from 2026-07-11 --date-to 2027-06-30 --no-details  # next-season book
python3 -m club_soccer.fetch --repair            # fix/drop implausible future-dated results
python3 -m club_soccer.health                    # data health checks (exit 1 on hard failures)
```

BSD key: `data/api_keys.json` → `"bsd"`, or env `BSD_API_KEY`. Register free
at https://sports.bzzoiro.com/register/.

### Model

```bash
python3 -m club_soccer.model --fit
python3 -m club_soccer.model "Arsenal" "Chelsea" --competition "Premier League"
```

5-component ensemble (goals Poisson, Elo, xG, xG-form, shot-pressure),
Dixon-Coles low-score correction, walk-forward-tuned weights, 365-day
recency decay. BSD's real team xG is used wherever both sides are available;
the SoT conversion remains the explicit fallback for older or uncovered
competitions. Enhancements are stored with explicit held-out gates; a
candidate is only activated when it improves the required metrics on the
fixed time splits, with the result recorded in
`docs/model_improvements_changelog.md`.

| Feature | Fit with | Gated by |
|---|---|---|
| Per-competition home advantage + rho | `model.py::fit()` (automatic) | `comp_adj_active` in `model_params.json` |
| League-season scoring environment + HFA | `model.py::fit(league_adjustments=True)` | `league_adjustments_active` in fitted params; currently rejected by walk-forward gate |
| 1X2 temperature calibration | `python3 -m club_soccer.validate --calibrate` | `active` in `data/calibration.json`; promoted after all three fixed splits improved |
| Fitted competition strength | `python3 -m club_soccer.model --fit-comp-strength` | `active` in `data/comp_strength.json` |
| Promoted/relegated shrinkage prior | `python3 -m club_soccer.model --tune-promo-prior --write` | `promo_prior.active` in `model_params.json` |
| Season-boundary Elo regression + half-life | `python3 -m club_soccer.model --tune-season-boundary --write` | `season_regress_rho.active` / `half_life_days.active` |
| Context GLM (rest/congestion/motivation/tier-gap/weather) | `python3 -m club_soccer.context --fit` | `active` in `data/context_coef_club.json` |
| Market blend (1X2 / OU2.5) | `python3 -m club_soccer.fit_market_blend --write` | `app/market_blend.DEFAULT_BLEND_ON` (code change) |

### Player layer

Minutes, squads and transfers are all derived by **observation** (who
actually played for whom, lately) from BSD per-match player data — no
transfer feed. The cache uses BSD's canonical `/api/player-stats/` rows when
available (provider player ID, minutes, rating, xG/xA, shots, passing and
defensive actions), with confirmed lineups as a fallback. Unused substitutes
are retained as zero-minute selection records and never treated as appearances.
Refreshes select the newest events first, ingest chronologically, and
checkpoint every 25 matches so a slow provider response cannot lose a rebuild.

```bash
python3 -m club_soccer.player_features --refresh --max-events 400 --days-back 90  # BSD stats
python3 -m club_soccer.player_features --from-cache                    # deterministic offline rebuild
python3 -m club_soccer.player_features --pull-absences              # dated absences -> data/absences_club.csv
python3 -m club_soccer.club_squads --write                          # squads + detected transfers
python3 -m club_soccer.minutes --write                              # xi_load features
```

Availability adjustments (attack/defence multipliers + an uncertainty band
that widens on doubtful absences or a missing GK) are computed by
`club_soccer/availability.py` and applied via `edge.py --player-adj`
(alias `--availability`) — haircuts the Kelly stake by lineup confidence,
never the point estimate.

### League structure

```bash
python3 -m club_soccer.standings "Premier League" --date 2026-05-01
```

Point-in-time tables (3-1-0 points, uniform points/GD/GF tiebreak — no
per-league head-to-head rule). Feeds motivation features (`ppg_diff`,
title/Europe/relegation "fight" and mathematically-"dead" flags) into the
context GLM.

### Market layer

```bash
python3 -m club_soccer.fetch_fdcouk                # historical closing odds (backtest teacher)
python3 -m club_soccer.snapshot_odds                # live multi-bookmaker snapshots
python3 -m club_soccer.backtest_market              # model vs de-vigged closing-line log-loss + simulated ROI
```

`edge.py` applies the do-not-bet filter (suppress steam-chasing or
unanimous-books-thin-edge bets) automatically once 30 days of snapshot
history exist; before that it prints "market-model warming up: N days".

### Weather

```bash
python3 -m club_soccer.weather --missing-venues     # teams with no data/venues.csv row
python3 -m club_soccer.weather --build              # backfill (played) + forecast (<=16d out)
```

City-level (not stadium-level) lat/lon, Open-Meteo (free, no key).
`data/venues.csv` covers the 12 core tracked leagues fully; Champions/Europa/
Conference League opponents are filled in iteratively as `--missing-venues`
surfaces them.

### Additional xG sources

BSD now supplies real team xG in event detail (`live_stats.expected_goals`
and its direct xG variants), so the model consumes that feed without an
additional scraper. Coverage is strongest for the domestic cup rows that BSD
details and partial across league/UEFA history; run
`python3 -m club_soccer.health` to see exact coverage by type and cup.
Where BSD has no xG, the model falls back to its SoT-based estimate. Understat
remains an optional future backfill source for historical top-five-league rows,
but is not required by the production path.

### Cup results

Fixture detail preserves `result_scope` (`regulation`, `extra_time`, or
`penalties`), half-time/full-time scores, round/group metadata, venue, and
shootout scores/winner. Shootout advancement is stored separately so cup
settlement and regulation markets are not conflated.

### Validation

```bash
python3 -m club_soccer.validate            # walk-forward report (1X2, OU2.5, BTTS Brier)
python3 -m club_soccer.validate --gate     # pass/fail vs data/validation_baseline.json
python3 -m club_soccer.validate --tune-ensemble [--write]
python3 -m club_soccer.validate --calibrate       # temperature calibration; activates only after multi-split gate
python3 -m club_soccer.validate --compare-league-adjustments --write  # diagnostic only
python3 -m club_soccer.fit_market_blend           # diagnostic; promotion remains gated
python3 -m club_soccer.validate --benchmark-clubelo   # report-only sanity check, never a model input
python3 test_club_soccer.py
```

Calibration and market anchoring are explicit promotion gates. Temperature
calibration is currently active because it improved both primary metrics on
all fixed splits; market anchoring remains off because it has not beaten the
market benchmark under the stricter gate.
