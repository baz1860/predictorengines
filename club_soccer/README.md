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
required). `python3 -m club_soccer.health` checks it for the two things that
would silently corrupt the model: future-dated "finished" rows and duplicate
`fixture_id`s.

```bash
python3 -m club_soccer.fetch --current           # results + upcoming fixtures
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

5-component ensemble (goals Poisson, Elo, SoT-xG, SoT-form, shot-pressure),
Dixon-Coles low-score correction, walk-forward-tuned weights, 365-day
recency decay. Every enhancement below this line ships **gated off by
default** and only becomes a default once a held-out walk-forward gate
passes (`club_soccer/validate.py`'s `--gate`, and the criteria in
`plans/club_soccer_engine_plan.md` §12) — the fitted artifact is always kept
on disk even when the gate says no, with the honest result recorded in
`docs/model_improvements_changelog.md`.

| Feature | Fit with | Gated by |
|---|---|---|
| Per-competition home advantage + rho | `model.py::fit()` (automatic) | `comp_adj_active` in `model_params.json` |
| Fitted competition strength | `python3 -m club_soccer.model --fit-comp-strength` | `active` in `data/comp_strength.json` |
| Promoted/relegated shrinkage prior | `python3 -m club_soccer.model --tune-promo-prior --write` | `promo_prior.active` in `model_params.json` |
| Season-boundary Elo regression + half-life | `python3 -m club_soccer.model --tune-season-boundary --write` | `season_regress_rho.active` / `half_life_days.active` |
| Context GLM (rest/congestion/motivation/tier-gap/weather) | `python3 -m club_soccer.context --fit` | `active` in `data/context_coef_club.json` |
| Market blend (1X2 / OU2.5) | `python3 -m club_soccer.fit_market_blend --write` | `app/market_blend.DEFAULT_BLEND_ON` (code change) |

### Player layer

Minutes, squads and transfers are all derived by **observation** (who
actually played for whom, lately) from BSD per-match player data — no
transfer feed.

```bash
python3 -m club_soccer.player_features --refresh --max-events 500   # build the player-apps cache
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

### Understat xG — not implemented

The plan called for scraping Understat for real top-5-league xG. Verified
before writing anything: Understat's `robots.txt` is now `Disallow: /` for
the whole site, and the old embedded-JSON scraping pattern is gone (the
site now loads match data client-side). Both are blockers on their own; see
`docs/model_improvements_changelog.md` for the full writeup. The `xg`/`xgf`
ensemble components keep the shots-on-target proxy for every league.

### Validation

```bash
python3 -m club_soccer.validate            # walk-forward report (1X2, OU2.5, BTTS Brier)
python3 -m club_soccer.validate --gate     # pass/fail vs data/validation_baseline.json
python3 -m club_soccer.validate --tune-ensemble [--write]
python3 -m club_soccer.validate --benchmark-clubelo   # report-only sanity check, never a model input
python3 test_club_soccer.py
```
