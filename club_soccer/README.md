# Club Soccer Engine

Predicts club football across the Premier League, Championship, League
One/Two, the four Scottish divisions, Bundesliga, Serie A, Ligue 1, La Liga,
the major domestic cups, and the Champions/Europa/Conference Leagues.

## Use it

One command. It refreshes results/fixtures/absences/squads/odds, updates the
model when training data changed, prices upcoming matches, and writes a card:

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
  local time, monitors the run, and captures decision-time odds every 15
  minutes so the forward evidence ledger does not miss its pre-kickoff window.
  ```bash
  bash deploy/install_launchagents.sh
  ```
  Logs go to `~/Library/Logs/club_soccer_{season,monitor,capture}.log`.

`club_soccer/update.sh` is a thin wrapper around the same command for the
app's scheduler / launchd-style invocation (`./club_soccer/update.sh --fast`).
The normal daily path is incremental:

- one 90-day BSD event-index response is shared by fixtures, absences, player
  adjustments, pricing and the decision ledger;
- the player store ingests only new or changed cached event bundles;
- model parameters are reused when the played training rows and model inputs
  are unchanged;
- the fixed-window promotion gate and decision-time evidence are reused only
  on an exact fingerprint match;
- recoverable cache pruning runs weekly, and far-ahead odds are polled every
  three days rather than every day.

The first run after deploying model/validation code or creating the player
manifest can still be slow. That is an intentional full rebuild; subsequent
unchanged runs take the incremental path.

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
| Opponent-adjusted xG attack/defence | `model.py::fit()` (automatic) | promoted evidence in `data/opponent_adjusted_xg_evidence.json` |
| 1X2 temperature calibration | `python3 -m club_soccer.validate --calibrate` | `active` in `data/calibration.json`; promoted after all three fixed splits improved |
| Market blend (1X2 / OU2.5) | `python3 -m club_soccer.fit_market_blend --write` | `app/market_blend.DEFAULT_BLEND_ON` (code change) |

Rejected experiments and their evidence remain listed in `experiments.json`;
their runtime implementations are not kept in the production package.

### Player layer

Minutes, squads and transfers are all derived by **observation** (who
actually played for whom, lately) from BSD per-match player data — no
transfer feed. The cache uses BSD's canonical `/api/player-stats/` rows when
available (provider player ID, minutes, rating, xG/xA, shots, passing and
defensive actions), with confirmed lineups as a fallback. Unused substitutes
are retained as zero-minute selection records and never treated as appearances.
Refreshes select the newest events first, ingest chronologically, and
checkpoint every 25 matches so a slow provider response cannot lose a rebuild.
The offline daily refresh keeps an event-file manifest and therefore parses
only new or changed bundles. `--from-cache` remains the explicit full rebuild.

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

```bash
python3 -m club_soccer.player_features --refresh-cached --oldest-first --max-events 500  # BSD backfill for cached events
```

The rejected point-in-time player-quality experiment is retained only as a
result artifact in `data/player_quality.json`; its unused runtime
implementation has been removed.

### Bzzoiro v2 enrichment

The Bzzoiro-compatible v2 collector keeps raw, resumable match payloads for
shotmaps, team stats, confirmed lineups, incidents and per-match player stats.
It is an experimental cache and does not alter `fixtures.csv` or production
predictions by itself. The default pull is the 2025/26 league season for the
Premier League and the other four major European leagues:

```bash
python3 -m club_soccer.bsd_enrichment --collect \
  --league-ids 1,3,4,5,6 --date-from 2025-08-01 --date-to 2026-05-31 \
  --oldest-first --workers 12
python3 -m club_soccer.bsd_enrichment --refresh-players --workers 12
python3 -m club_soccer.bsd_enrichment --summary --join
```

Raw records are stored under `data/bsd_enrichment/`; the flattened join is
`data/bsd_enriched_matches.csv`. `candidate_fixtures()` can fill missing
historical xG pairs from shotmaps for a gated walk-forward experiment while
preserving existing xG observations. Current-season player rows are available
through the v2 event player-stats endpoint; older historical matches may have
shotmaps without player-level coverage.

### League structure

```bash
python3 -m club_soccer.standings "Premier League" --date 2026-05-01
```

Point-in-time tables use 3-1-0 points and uniform points/GD/GF tiebreaks; they
do not attempt competition-specific head-to-head rules.

### Market layer

```bash
python3 -m club_soccer.fetch_fdcouk                # historical closing odds (backtest teacher)
python3 -m club_soccer.snapshot_odds                # live multi-bookmaker snapshots
python3 -m club_soccer.decision_time_backtest       # settled decision-time evidence used by the staking gate
python3 -m club_soccer.market_diagnostics           # read-only model-vs-market audit
python3 -m club_soccer.runtime_safety --json        # list Syncthing conflicts; never merges them
```

The staking artifact is `decision_time_v3`. Complete raw closing markets are
captured per book, fair CLV uses a power de-vig, and raw price CLV compares the
executed quote with the same bookmaker at close. The legacy proportional CLV is
diagnostic only. Evidence cohorts use the explicit strategy contract in
`strategy_contract.py`; code comments, model refits and unrelated alias changes
do not reset compatible history. Identity exclusions apply only to reviewed
fixtures.

### Surfaced prediction scope

Collection, training, odds capture, decision evidence and the append-only
forecast ledger continue across every supported competition. User-facing cards,
app predictions, app/CLI edge reports and manual odds templates surface only:

- Premier League, La Liga, Bundesliga, Serie A and Ligue 1;
- Scottish Premiership;
- Champions League, Europa League, Conference League and UEFA Super Cup;
- Copa Libertadores and Copa Sudamericana.

This is a presentation boundary in `prediction_scope.py`, not a data-retention
filter. Non-surfaced competitions continue accumulating shadow forecasts and
settlements for research and possible later expansion.

`edge.py` applies the do-not-bet filter (suppress steam-chasing or
unanimous-books-thin-edge bets) automatically once 30 days of snapshot
history exist; before that it prints "market-model warming up: N days".

### Forward forecast scoring

Every supported fixture probability—surfaced or shadow—is frozen before
rendering in an append-only forecast ledger. This is deliberately separate from
betting evidence: it measures the complete forecast universe even when a
fixture has no odds, while `decision_ledger.py` measures whether an executable
quote-backed strategy makes money.

```bash
python3 -m club_soccer.forecast_ledger --status
python3 -m club_soccer.forecast_ledger --settle --report
```

The normal season run handles both automatically. Runtime artifacts are:

- `data/forecast_ledger.csv` — every surfaced and shadow snapshot, with model/data/code
  provenance, final H/D/A, OU2.5 and BTTS probabilities, evidence tier,
  availability adjustments and lead time;
- `data/forecast_settlements.csv` — one immutable official result per frozen
  fixture identity;
- `data/forecast_performance.json` — first-published, T-24 and latest
  pre-kickoff Brier/log-loss/accuracy summaries, plus rolling, competition and
  calibration views.

Repeated card appearances are retained so movement toward kick-off can be
measured, but the headline T-24 cohort selects only the closest forecast at or
before 24 hours for each fixture. Cached/offline runs are recorded as
non-primary evidence; only successful network production rows enter headline
metrics.

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
python3 -m club_soccer.validate --gate     # fixed-window pass/fail vs promotion_baseline.json
python3 -m club_soccer.promote_baseline    # deliberate re-pin after an audited population change
python3 -m club_soccer.validate --opponent-xg-ab --test-from 2024-07-01 \
    --test-to 2026-07-01 --write-evidence  # reproducible promoted-feature A/B
python3 -m club_soccer.validate --calibrate       # temperature calibration; activates only after multi-split gate
python3 -m club_soccer.fit_market_blend           # diagnostic; promotion remains gated
python3 -m club_soccer.validate --benchmark-clubelo   # report-only sanity check, never a model input
python3 -m pytest tests/club_soccer
```

The promotion gate uses an order-independent identity/outcome hash. A promoted
reference also includes `promotion_population.json`, so a later population
failure reports counts and examples of added/removed fixtures in
`validation_population_diff.json`. Commit the baseline and population snapshot
together after reviewing that diff; validation itself never moves either
reference.

Calibration and market anchoring are explicit promotion gates. Temperature
calibration is currently held inactive after the 2024–25 backfill because it
failed the early fixed splits; market anchoring remains off because it has not
beaten the market benchmark under the stricter gate.
