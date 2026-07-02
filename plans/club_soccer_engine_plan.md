# Club Soccer Engine — Functional Repair & Full-Data Upgrade Plan

Date: 2026-07-02
Audience: the implementing model (Sonnet). This document is the specification.
Follow it literally. Where a judgement call is genuinely required, the rule to
apply is written in the task. Do not invent alternative designs.

---

## 0. Ground rules (read before writing any code)

1. **Dependencies:** numpy + pandas + stdlib only. No new pip packages. sklearn
   may be used only behind the existing try/except pattern (see
   `club_soccer/validate.py::_isotonic`).
2. **Imports/invocation:** `club_soccer` is a package. All new modules use
   package-relative imports and run as `python3 -m club_soccer.<module>`.
3. **Offline-first:** every network fetcher must (a) cache raw responses to
   disk, (b) degrade gracefully offline (print a "skipped" line, never raise
   out of a pipeline), and (c) never be required for `model.predict` /
   `validate.py` to run. This mirrors `fetch.py` and the WC engine.
4. **Promotion discipline (the single most important rule):** every
   model-affecting change lands **report-only / gated OFF** first. It becomes a
   default only if it passes the promotion gate defined in §12. Rejected fits
   are kept on disk as inactive artifacts (the existing `comp_adj` /
   `comp_adj_active: False` pattern in `model.py` is the template).
5. **Point-in-time:** any feature used in a backtest must be computable from
   data strictly dated before the match. New dated files are append-only with a
   `recorded_at` column where the underlying fact can change over time
   (absences, odds, standings are derived so exempt).
6. **Don't break the app contract:** `club_soccer/engine.py` `COMMANDS` and
   `app/engines/club_soccer.py` (`ClubSoccerAdapter`) must keep working. Only
   add optional fields to their outputs. After every phase run
   `python3 run_checks.py` and `python3 test_club_soccer.py` — both must pass.
7. **No API keys in git.** Keys come from `data/api_keys.json` /
   env vars via `api_keys.get_key` (existing pattern).
8. **One front door:** the end state (Phase 8) is
   `python3 -m club_soccer.season` → `club_soccer/data/card.md`, matching the
   golf/tennis/worldcup pattern. Sub-scripts remain internals.

Commands you will use constantly:

```bash
python3 -m club_soccer.model --fit
python3 -m club_soccer.validate            # walk-forward report
python3 -m club_soccer.validate --gate     # pass/fail vs baseline
python3 -m club_soccer.validate --tune-ensemble [--write]
python3 test_club_soccer.py
python3 run_checks.py
```

---

## 1. Current-state audit (verified 2026-07-02)

What exists and works:

- `data/fixtures.csv`: 17,496 rows, 2022-07 → 2026-05, 15 competitions,
  team-level shots/SoT/corners on 16,775 rows. Source of truth.
- `model.py`: 5-component ensemble (goals Poisson, Elo, SoT-xG, SoT-form,
  shot-pressure), Dixon-Coles rho, walk-forward-tuned weights, 365-day decay.
  Per-competition HFA/rho fitted but gated off (validated neutral).
- `validate.py`: monthly-refit walk-forward, 1X2 Brier gate, isotonic
  calibration, ensemble tuner with a real promotion gate. Good bones — extend,
  don't replace.
- `player_features.py`: BSD-based player xG/minutes cache + absence →
  attack/defence multipliers. **Built but inert** — nothing calls it in the
  default predict/edge path, and its team attribution is buggy (see P2.1).
- `edge.py`: 1X2 / OU2.5 / BTTS, quarter-Kelly, manual/API odds.
- `fetch.py`: BSD (free, no rate limits) → fixtures.csv.

Why it is **not currently functional**:

- **Zero upcoming fixtures.** `M.upcoming()` returns 0 rows, so `edge.py
  --template` writes an empty file and nothing can be predicted or priced.
  Data effectively stops 2026-05-24 (season end); the 2026-27 fixture list has
  never been fetched.
- **80 corrupted rows**: Champions League 2025-26 league-phase results are
  dated **2027-01-20/21** with status FT and filled scores (e.g. "FK Kairat vs
  Club Brugge, 2027-01-20, FT"). Future-dated "played" matches poison Elo
  recency weighting and season inference.
- No rest/congestion, no availability in the default path, no standings, no
  weather, no minutes data, no odds history, no market teacher — all the
  things the WC engine has (feature store, context GLM, availability bands,
  market model) or that club soccer specifically needs (transfers, league
  position, promoted-team priors).

## 2. What we take from the WC engine — and what we deliberately do NOT

Port these patterns (they are proven in this repo):

| WC artifact | Pattern to port |
|---|---|
| `wc_v4/feature_store.py` | Point-in-time training matrix + `build_asof` live path; state updated *after* emitting each row; provenance columns; schema module with FEATURE vs OUTCOME column split and a leak check. |
| `engines/worldcup/context.py` | Poisson IRLS with `offset = log(model λ)`; keep only \|t\| ≥ 2 coefficients; apply as `λ *= exp(Σ bᵢ·featureᵢ)`; JSON coefficient file; `--fit` / `--validate` CLI. |
| `wc_v4/availability.py` | Absence certainty classification, GK-specific flag, uncertainty band that widens stakes-haircuts when the lineup read is shaky. |
| `wc_v4/market_model.py` | Opening/current/closing odds history, steam/reversal movement features, `do_not_bet()` suppression, closing line as teacher-not-oracle. |
| `update.sh` + `scripts/worldcup/card.py` | Morning/pre-kickoff/post-match flows and a single human-readable best-bets card. |

Do **not** port (international-specific, wrong for club):

- **Altitude** (`context.py` alt_gap): irrelevant for UK/EU club venues. Drop.
- **Confederation adjustments**: replaced by fitted competition strength (P4.4).
- **EA-ratings squad power** (`squads.py`/`ea_players.csv`): club player value
  comes from observed minutes/xG (we have per-match data), not a static
  ratings file.
- **Tournament bracket simulation**: out of scope. (League-winner outright sim
  is a possible future item; do not build it in this plan.)

Club-specific realities the WC engine never had to handle — these drive the
new work: **transfers and season-boundary squad churn; promotion/relegation;
heavy midweek congestion and rotation; league-table motivation (title,
relegation, Europe, dead mid-table); cup ties across tiers; winter weather
across a 10-month season.**

## 3. Free data source inventory (all verified patterns already in repo or documented in docs/DATA_SOURCING_PLAN.md)

| Source | What we take | Cost / limits | Used in phase |
|---|---|---|---|
| **BSD** (`bsd_client.py`, sports.bzzoiro.com) | Fixtures/results, per-match team+player stats, **unavailable_players**, **confirmed lineups**, **multi-bookmaker odds** embedded per event | Free, no rate limits (key required) | 0, 2, 6 |
| **football-data.co.uk** | Historical CSVs per league-season: results, shots/SoT/corners, **Bet365 pre-match + Pinnacle closing 1X2 and OU2.5 odds** | Free, no key. Codes: E0,E1,E2,E3 (England tiers), SC0–SC3 (Scotland), D1, I1, SP1, F1. URL `https://www.football-data.co.uk/mmz4281/{yy}{yy+1}/{code}.csv` e.g. `2526/E0.csv` | 1 (backtest teacher), 6 |
| **Open-Meteo** | `archive-api.open-meteo.com/v1/archive` (hourly wind/precip/temp, 1940→now) for backfill; `api.open-meteo.com/v1/forecast` (≤16 days) for upcoming | Free, no key, non-commercial | 5 |
| **Understat** | Real per-match xG, top-5 leagues only (EPL, La Liga, Serie A, Bundesliga, Ligue 1) | Free scrape (JSON in `<script>` tags). FBref is stale (lost Opta licence Jan 2026) — do not use it for current xG | 7 |
| **football-data.org** (`fetch_fdorg.py`, exists) | Backup fixtures for PL/ELC/EL1/EL2/BL1/SA/FL1/PD/CL/FAC | Free tier, 10 req/min | 0 (fallback only) |
| **The Odds API** (`edge.py`, exists) | Live odds at bet time | Free 500 credits/mo — use sparingly; BSD odds are the default snapshot source | 6 |
| **ClubElo** (`api.clubelo.com`) | Independent club Elo CSV, free, no key | Optional sanity benchmark only — report-only comparison in validate; never a model input | 1 (optional) |

---

# Phases

Order is dependency-driven. Do not start a phase before its prerequisites.

```
P0 (repair) → P1 (feature store + backtest) → P2 (player layer)
                                            → P3 (context GLM: rest/load)  [needs P1; P2 for load terms]
                                            → P4 (league structure)        [needs P1]
                                            → P5 (weather)                 [needs P1, P3 machinery]
                                            → P6 (market layer)            [needs P1]
                                            → P7 (Understat xG)            [independent, needs P1 gate]
P8 (season.py front door + scheduling)      [needs P0–P4 minimum; absorbs the rest as they land]
```

---

## Phase 0 — Data repair: make the engine functional again

### P0.1 Repair the 80 future-dated rows

New CLI flag `python3 -m club_soccer.fetch --repair`:

1. Select rows in `fixtures.csv` where `status == "FT"` (or any finished
   status) **and** `date > today + 1 day`.
2. For each, call `bsd_client.get_event(key, fixture_id)`; take
   `event_date_utc(event)`. If it returns a valid past date, rewrite the row's
   `date` (and recompute `season` with the existing rule: month ≥ 7 → year,
   else year − 1).
3. If the refetch fails or still yields a future date, **drop the row** and
   print it.
4. Write the repaired file; print a summary: `repaired N, dropped M`.

Also add a **permanent guard** in `_bsd_to_fixture_row`: if the event status is
finished but the kickoff date is more than 1 day in the future, return `None`
(caller skips and counts; print the count once per fetch).

Acceptance: after repair, `python3 - <<'EOF'` check
`df[(df.status=='FT') & (df.date > 'TODAY')]` is empty; `--fit` runs clean.

### P0.2 Restore upcoming fixtures + season rollover

- Extend the default `fetch_fixtures` behaviour: when called with no `status`,
  it already fetches all — the problem is operational (nobody fetched since
  May). The fix is procedural + one code change:
  - In `fetch.py`, after merging, print counts:
    `played / upcoming / finished-in-last-7-days` so staleness is visible.
  - `upcoming` rows must never be dropped by the finished-status score guard
    (verify: scores nulled for non-finished — already correct — and dedupe by
    `fixture_id` keeps latest — already correct).
- Add `--days-ahead N` filter option (default: keep everything BSD returns).
- Document in README: 2026-27 fixtures appear on BSD during July/August; the
  daily pipeline (P8) picks them up automatically.

Acceptance: `python3 -m club_soccer.fetch --current` followed by
`M.upcoming(M.load_fixtures())` returns > 0 rows once BSD publishes 2026-27
fixtures (test with `--status upcoming` now; if BSD has none yet the code path
is still exercised and prints `0 upcoming`).

### P0.3 Data health checks

New module `club_soccer/health.py` with `run_checks() -> dict` and CLI
`python3 -m club_soccer.health`:

- `future_ft_rows`: count of finished rows dated in the future (must be 0).
- `duplicate_fixture_ids`: count (must be 0).
- `days_since_last_result`: today − max(date of played row). Warn > 7 in
  Aug–May; info only Jun–Jul (off-season).
- `upcoming_count`: rows with null goals and date ≥ today.
- `stats_coverage`: fraction of played rows with SoT present (report).
- Exit code 1 only on the two "must be 0" checks.

Wire into `club_soccer/update.sh` (first step) and later into `season.py`.
Add a test in `test_club_soccer.py` calling `run_checks()` on the shipped CSV.

---

## Phase 1 — Point-in-time feature store + honest backtest harness

This is the substrate everything else validates against. Mirror
`wc_v4/feature_store.py` closely.

### P1.1 `club_soccer/schema.py`

Constants: `SCHEMA_VERSION = 1`,
`PROVENANCE_COLUMNS = ["asof","source","fetched_at","schema_version"]`,
`ID_COLUMNS = ["event_id","match_date","home","away","competition","season","neutral"]`,
`FEATURE_COLUMNS` (grows by phase; starts with
`elo_h, elo_a, elo_diff, lam_h, lam_a, p_model_h, p_model_d, p_model_a,
rest_days_h, rest_days_a, matches_7d_h, matches_7d_a, matches_14d_h,
matches_14d_a, matches_30d_h, matches_30d_a`),
`OUTCOME_COLUMNS = ["home_goals","away_goals","result","p_close_h","p_close_d",
"p_close_a","odds_close_h","odds_close_d","odds_close_a","odds_close_over25",
"odds_close_under25"]`, and
`feature_columns(cols)` which raises if any OUTCOME column leaks into the
feature list (copy the WC implementation).

### P1.2 `club_soccer/feature_store.py`

Two entry points, signatures fixed:

```python
def build_training_matrix(since: str = "2022-08-01", until: str | None = None,
                          competitions: list[str] | None = None) -> pd.DataFrame
def build_asof(asof: str, fixtures: pd.DataFrame | None = None) -> pd.DataFrame
```

- Schedule features are computed **across all competitions per club** (a
  Wednesday Champions League match counts toward Saturday's league-match rest
  and congestion — this is the club-specific point the WC store didn't need).
  Copy `_schedule_features` from `wc_v4/feature_store.py` (update state after
  emitting the row) and extend the deque logic to also count 7- and 30-day
  windows.
- Strength/model columns: refit `model.fit` at **calendar-month boundaries**
  on strictly-prior matches (copy the `_MonthlyBeta` caching idea; here the
  cached object is the full `params` dict). `p_model_*` uses the ensemble.
- Market columns join from `data/market_history.csv` (P1.3) on
  `(match_date, home, away)` after alias normalisation via
  `club_soccer/names.py`; closing odds land in OUTCOME columns only.
- `build_asof` = live path: only data dated < asof; used later by season.py
  and by any backtest of "what would we have said that morning".

CLI smoke: `python3 -m club_soccer.feature_store --since 2024-08-01` prints
row count and head, like the WC module.

### P1.3 Historical closing odds: `club_soccer/fetch_fdcouk.py`

- Download `https://www.football-data.co.uk/mmz4281/{ss}/{code}.csv` for
  `ss ∈ {2223,2324,2425,2526,2627}` ×
  `code ∈ {E0,E1,E2,E3,SC0,SC1,SC2,SC3,D1,I1,SP1,F1}` (skip 404s silently —
  next season's file appears mid-August). Cache raw files under
  `data/fdcouk_cache/{ss}_{code}.csv`; refetch only the current season's file.
- Emit `data/market_history.csv` with columns:
  `match_date, competition, home, away, b365_h, b365_d, b365_a,
  ps_h, ps_d, ps_a, psc_h, psc_d, psc_a, b365_over25, b365_under25,
  max_over25, max_under25, source_code`.
  Column mapping from fd.co.uk: `Date→match_date` (day-first!), `HomeTeam`,
  `AwayTeam`, `B365H/D/A`, `PSH/D/A`, `PSCH/PSCD/PSCA` (Pinnacle **closing**),
  `B365>2.5→b365_over25`, `B365<2.5→b365_under25`, `Max>2.5`, `Max<2.5`.
  Older/lower-league files may lack Pinnacle columns — leave NaN.
- Team-name aliasing: fd.co.uk uses short names ("Man United",
  "Nott'm Forest", "Sheffield Weds", "QPR"…). Extend `club_soccer/names.py`
  with an `FDCOUK_ALIASES` dict mapping fd.co.uk name → fixtures.csv name.
  Build it empirically: after the first join, print unmatched names and add
  aliases until join coverage ≥ 95% of league rows per competition; assert
  that coverage in the test.

### P1.4 Extend `validate.py` beyond 1X2

- `walk_forward()` rows gain `p_over25`, `p_btts` and labels
  `total_goals`, `btts_actual`.
- `metrics()` gains `brier_ou25` (2-outcome Brier of P(over2.5)) and
  `brier_btts`. Print all three.
- Baseline JSON gains the two new keys the first time it is rewritten
  (`--update-baseline`); the **gate still passes/fails on 1X2 Brier only**,
  but prints the totals/BTTS deltas. (Weather in P5 will add a totals gate —
  see §12.)

### P1.5 Market-anchored backtest: `club_soccer/backtest_market.py`

The honest scoreboard. Join walk-forward predictions (P1.4 rows) to
`market_history.csv`.

Report (printed table + `data/backtest_market.json`):

- n matched rows per competition.
- Model 1X2 log-loss vs **de-vigged Pinnacle closing** log-loss on the same
  rows (the market benchmark to beat or approach).
- Simulated betting vs closing prices: for edge thresholds
  `{2%, 4%, 6%}` (model prob − de-vigged closing prob), flat 1-unit ROI and
  quarter-Kelly ROI, bet count, per-market split (1X2 / OU2.5).
- CLV proxy: for each simulated bet taken at Bet365 pre-match price, CLV =
  de-vigged Pinnacle closing prob of the side − de-vigged B365 prob of the
  side. Report the mean and the fraction positive. Rows lacking either book:
  excluded, count reported.

No gate — this is a diagnostic. It must run offline once market_history.csv
exists.

### P1.6 (Optional, report-only) ClubElo benchmark

`validate.py --benchmark-clubelo`: fetch `api.clubelo.com/{date}` once, cache,
convert Elo diff → 1X2 via the existing `_lambdas_elo` shape, report its Brier
next to ours on the same walk-forward months. Never a model input. Skip
cleanly offline.

---

## Phase 2 — Player layer: minutes, squads, transfers, absences

### P2.1 Fix player→team attribution and store dated appearances

`player_features.py::refresh` currently guesses a player's team by index
position (`players.index(p) < len(players)//2`) — wrong and it breaks on
transfers. Rework:

- `_players_from_event` must return `(entry, side)` using the lineups
  structure (shape 1 already knows home/away; shapes 2/3 also carry side).
  Only fall back to the index heuristic if no shape provides sides, and tag
  those entries `side_confident=False`.
- Cache schema v2 (`player_stats_cache.json`, bump an internal `"v": 2` key):
  per player keep `apps: [{"date": "YYYY-MM-DD", "team": str, "mins": float,
  "xg": float, "xa": float}]`, most recent **60** kept. Preserve `pos`.
  On load, if `"v"` missing → discard and rebuild via
  `--from-cache` (bsd_cache event files carry dates; use `event_date_utc`).

### P2.2 Squads & transfers: `club_soccer/club_squads.py`

Transfers are handled by **observation, not by a transfer feed**: a player
belongs to the club he most recently appeared for.

- `build_squads() -> pd.DataFrame` from the v2 apps history:
  current squad of club T = players whose most recent app in the last 120
  days was for T. Columns:
  `team, player, pos, last_seen, apps_season, mins_season, mins_30d`.
  Write `data/squads_club.csv` (name avoids clashing with the WC
  `data/squads.csv`).
- **Transfer detection (report-only):** a player whose latest app team ≠ his
  previous app team generates a row in `data/transfers_detected.csv`
  (`date, player, from_team, to_team`). Surface new arrivals/departures in the
  daily card (P8). No automatic model adjustment from transfers beyond the
  re-attribution itself — squad churn is priced at the team level by P4.6
  (season-boundary Elo regression), which is fitted, not assumed.
- Manual override `data/transfers_manual.csv`
  (`effective_date, player, from_team, to_team`) applied on top, for known
  moves before a debut (empty template with header committed).

### P2.3 Minutes-load features: `club_soccer/minutes.py`

Definitions (exact):

- Per player, as of date D: `mins_7d` = Σ minutes over apps with
  `D-7 < date ≤ D`; same for 14d/30d; `mins_season` = Σ since the season
  boundary (July 1 preceding D).
- Likely XI of club T as of D = top 11 current-squad players by `mins_30d`
  (ties → `mins_season`). If fewer than 11 have data, use what exists.
- Team features: `xi_load_7d`, `xi_load_14d`, `xi_load_30d` = mean of the
  likely XI's per-player minutes in the window, divided by 90 (units:
  matches-worth of minutes).
- `python3 -m club_soccer.minutes --write` → `data/player_minutes.csv`
  (`asof, team, player, pos, mins_7d, mins_14d, mins_30d, mins_season,
  starts_season`) and prints team-level XI loads.
- Feature-store integration: add `xi_load_14d_h/a` (and 7d/30d) to
  FEATURE_COLUMNS; in `build_training_matrix` compute them point-in-time from
  the apps history (only apps dated < match date). These become candidate GLM
  terms in P3.

### P2.4 Dated absences

- New job (part of season.py, also standalone
  `python3 -m club_soccer.player_features --pull-absences`): for each BSD
  upcoming event in our competitions, read `unavailable_players(event)` and
  **append** to `data/absences_club.csv`:
  `recorded_at, match_date, competition, team, player, reason, status`.
  Append-only; dedupe on `(match_date, team, player)` keeping the latest
  `recorded_at`.
- Point-in-time rule: a backtest as-of date A may only use rows with
  `recorded_at ≤ A`. (We start accumulating history now; there is no
  historical absence archive — accept that the availability gate needs a
  season of accumulation, exactly like the WC engine's blocker.)
- Wire availability into the live path **report-first**:
  - `season.py` card and `edge.py` print each match's
    `adjustments_for_match` multipliers and the WC-style uncertainty
    band (port the certainty regexes and `lineup_confidence` from
    `wc_v4/availability.py`; a GK absence widens the band).
  - `edge.py` gains `--availability` flag: applies the (already clamped
    [0.80, 1.25]) multipliers via the existing `player_adj` parameter of
    `M.predict`, and **haircuts the Kelly stake by the confidence factor**
    (stake × lineup_confidence).
  - Default-ON only after the §12 gate passes on accumulated data.

### P2.5 Confirmed-lineup mode

`edge.py --lineups`: for events within ~90 minutes of kickoff where BSD
provides confirmed lineups, use
`PlayerFeatureStore.adjustments_from_lineups` and re-price. Report-only output
(a "late card" section); never auto-bets.

---

## Phase 3 — Context GLM: rest, congestion, minutes load (fitted, gated)

`club_soccer/context.py`, a direct port of `engines/worldcup/context.py`
minus altitude:

- Offset = `log(λ)` from the **goals component** (`_lambdas_goals`) — a
  deterministic, params-only λ (do not use the blended ensemble as offset).
- Per-side features (all diffs clipped):
  - `rest_diff` = clip(own rest − opp rest, ±7), rest capped at 14
    (constants identical to WC: `REST_CAP=14`, `REST_DIFF_CLIP=7`).
  - `cong14_diff` = own matches_14d − opp matches_14d, clipped ±4.
  - `euro_hangover` = 1 if the side played a Champions/Europa/Conference
    League match 2–4 days before this match, else 0 (only for domestic
    fixtures).
  - `xi_load14_diff` = own xi_load_14d − opp xi_load_14d, clipped ±3
    (requires P2.3 data; fit without it if minutes coverage < 50% of rows).
- Fit: Poisson IRLS with offset (copy `_poisson_fit` verbatim), on played
  fixtures since 2022-08 from the feature store (point-in-time). Keep
  |t| ≥ 2 terms → `data/context_coef_club.json`.
- Apply: `model.predict` gains `context_adj: dict | None = None` parameter;
  when provided, `λ_h *= exp(Σ b·f_h)`, `λ_a *= exp(Σ b·f_a)` before the
  score matrix — same mechanism as `apply_player_adj`, applied to the blended
  matrix's extracted λs.
- Validate: `python3 -m club_soccer.context --validate` = refit on pre-split
  (< 2025-12-01), evaluate 1X2 log-loss with/without on the held-out months,
  same layout as the WC validate. Promote per §12.

---

## Phase 4 — League structure & position

### P4.1 `club_soccer/standings.py`

```python
def table_asof(competition: str, season: int, date: str,
               fixtures: pd.DataFrame | None = None) -> pd.DataFrame
```

League matches only (`type == "league"`), played, `date <` asof. 3-1-0
points; columns `team, played, points, gf, ga, gd, position` sorted by
points, gd, gf (uniform rule across leagues — do not implement per-league
head-to-head tiebreaks; note this as a documented approximation).
CLI: `python3 -m club_soccer.standings "Premier League" --date 2026-05-01`.

### P4.2 League-geometry metadata on `Competition`

Extend the dataclass in `competitions.py` with
`teams_n: int, releg_spots: int, promo_spots: int, euro_spots: int`
(fill: PL 20/3/–/7ish → use `euro_spots=7`; Championship 24/3/3/0;
League One 24/4/3/0; League Two 24/2/4/0; Scottish Prem 12/1/1/4;
Scottish lower 10/…; Bundesliga 18/3/–/7; Serie A 20/3/–/8; La Liga 20/3/–/8;
Ligue 1 18/3/–/7; cups/europe: zeros). Values needn't be legally perfect —
they parameterise motivation bands; document that.

### P4.3 Motivation & position features (into the feature store)

As of each match date, for league fixtures with ≥ 8 rounds played:

- `pos_diff` = home position − away position (integer).
- `ppg_diff` = home points-per-game − away points-per-game.
- Rounds remaining R = expected matches per team (2·(teams_n−1)) − played.
- Flags per side, computed only when `R ≤ 8`, else 0:
  - `fight` = 1 if within 3 points of the relegation line (either side of it)
    OR within 3 points of 1st OR within 3 points of the last euro spot.
  - `dead` = 1 if not `fight` and mathematically/practically parked: more
    than `3·R` points from every boundary above.
- Features into the context GLM (P3 machinery):
  `fight_diff = fight_own − fight_opp`, `dead_diff` likewise, `ppg_diff`.
  (`pos_diff` is collinear with strength — include `ppg_diff` only.)
  Fitted, |t|-pruned, gated exactly like P3 terms.

### P4.4 Fitted competition strength (replaces hand-set constants, gated)

`python3 -m club_soccer.model --fit-comp-strength`:

- For each competition, compute the mean end-of-fit Elo of the teams that
  played ≥ 6 matches in it during the last completed season (Elo is already
  shared across competitions in `fit()`, so cross-league cup/Europe matches
  and promoted teams propagate information).
- Map to the existing strength scale: fitted_strength(c) =
  `clip(0.15 + 0.85 · (mean_elo_c − min_e) / (max_e − min_e), 0.15, 1.10)`
  where min/max are over league competitions only; cups keep
  `0.95 × parent league's fitted strength` (parent = same country top flight
  for top-flight cups; document mapping in code).
- Write `data/comp_strength.json` `{name: value, "active": false}`.
  `competitions.strength()` consults it **only when `active` is true**.
  Promotion per §12 (walk-forward with the file active vs not).

### P4.5 Promoted-team priors

In `model.fit()`: identify teams whose first league match of a season is in a
different tier than their last league match of the previous season (use
`Competition.tier`). For a **promoted** team, seed its attack/defence
shrinkage prior not with `global_avg` but with its previous-season rates
scaled by a fitted promotion penalty `π` (single global multiplier on
attack, 1/π on defence). Fit `π` by grid `{0.80, 0.85, 0.90, 0.95, 1.00}` on
walk-forward Brier of promoted-team matches in the first 10 rounds,
2023–2026 seasons (three promotion cohorts exist in the data). Relegated
teams symmetric with 1/π. Store in `model_params.json` under
`"promo_prior": {"pi": float, "active": bool}`; gate per §12.

### P4.6 Season-boundary handling (transfers at team level)

Club sides churn every summer; the current model carries Elo and rates
straight across July.

- In the Elo loop in `fit()`: when a team's consecutive matches straddle a
  July 1 boundary, regress its Elo toward the mean Elo of its upcoming
  competition: `elo ← (1−ρ)·elo + ρ·league_mean`. Grid
  `ρ ∈ {0.0, 0.1, 0.2, 0.3, 0.4}` on walk-forward Brier restricted to
  August–October fixtures (where the effect lives). Store as
  `"season_regress_rho"`; gate per §12 (note: ρ=0 is the incumbent).
- Re-tune the recency half-life while there: grid
  `HALF_LIFE_DAYS ∈ {180, 270, 365}` on the same walk-forward. Keep 365
  unless the gate says otherwise.

### P4.7 Cup context

- Feature `tier_gap` for cup ties = away league tier − home league tier
  (league members only; Europe = tier 0/skip). Candidate GLM term
  (captures giant-killing base rates + rotation-by-favourites).
- Two-legged European knockouts: add a card annotation only ("2nd leg,
  aggregate X-Y") — **no model change**; the 1X2 price of the leg is still
  the deliverable. Explicitly out of scope: pricing qualification markets.

---

## Phase 5 — Weather (totals-oriented, gated on totals Brier)

### P5.1 `club_soccer/data/venues.csv`

Columns: `team, city, lat, lon, approx`. One row per team appearing in
`fixtures.csv`. Generate from general knowledge at **city level**
(`approx=1`); stadium-level precision is unnecessary for weather. Cover 100%
of teams (≈350); a helper `python3 -m club_soccer.weather --missing-venues`
lists teams without a row so the file can be completed iteratively.

### P5.2 `club_soccer/weather.py`

- Backfill: for each played fixture since 2022-08 with a venue row, GET
  `archive-api.open-meteo.com/v1/archive?latitude=..&longitude=..&start_date=D&end_date=D&hourly=temperature_2m,precipitation,wind_speed_10m`
  and take the 15:00 local values (kickoff time is unknown in fixtures.csv —
  document the approximation). Batch one request per (city, date) with an
  on-disk cache `data/weather_cache/{lat}_{lon}_{date}.json`; be polite
  (0.2 s sleep between uncached calls).
- Forecast: same fields from the forecast endpoint for upcoming fixtures
  ≤ 16 days out.
- Output `data/weather.csv`: `fixture_id, temp_c, precip_mm, wind_kmh`.
- Features (both sides symmetric, they shift totals not sides):
  `wind_high = max(0, wind_kmh − 25)/10`, `precip = min(precip_mm, 10)/5`,
  `temp_cold = max(0, 0 − temp_c)/5`, `temp_hot = max(0, temp_c − 28)/5`.
- Fit through the P3 GLM machinery (they enter both sides' λ equally).
  **Gate: held-out OU2.5 Brier** (this is why P1.4 exists), tolerance rules
  in §12; 1X2 must not regress beyond tolerance.

---

## Phase 6 — Market layer: odds history, movement, do-not-bet, blend

### P6.1 Daily odds snapshots from BSD: `club_soccer/snapshot_odds.py`

For each BSD upcoming event in our competitions, read the embedded
multi-bookmaker odds (`event["bookmakers"]`-style list — reuse the parsing in
`player_features.market_dispersion`). Append to `data/odds_history_club.csv`:
`snapshot_time (UTC ISO), match_date, competition, home, away, market
(1x2|total25), side, odds_median, n_books, disp` where `disp` = std of
de-vigged implied probs across books. Dedupe: at most one snapshot per
(event, market, side) per 6 hours. Run from season.py; BSD is free so 2×/day
is fine.

### P6.2 Movement features + do-not-bet: `club_soccer/market_model.py`

Port from `wc_v4/market_model.py`:

- `line_history(home, away, match_date, asof)` → open/current probs +
  movement per side from `odds_history_club.csv`.
- `do_not_bet(row)` rule (deterministic): suppress a recommended bet when
  **either** (a) the market has already moved ≥ 3 implied-prob points toward
  our side since open (steam we're chasing — the value is gone), or
  (b) `disp` < 0.005 and our edge < 4% (books unanimous, thin edge = model
  noise). Suppressed rows still appear in `edge_report.csv` with
  `suppressed_reason` filled; they are excluded from the card and stakes.
- `edge.py` applies do-not-bet by default once ≥ 30 days of snapshots exist
  (before that, prints "market-model warming up: N days").

### P6.3 Market blend refit (pricing layer, not model layer)

- Using P1 walk-forward probs joined to `market_history.csv` **B365 pre-match
  odds as the market input** (they proxy "current" odds available at bet
  time): fit logit blend
  `p_final = softmax((1−w)·logit(p_model) + w·logit(p_market))` with
  time-series CV over season splits (train seasons < S, test S; S ∈
  {2024, 2025}); pick w per market (1X2, OU2.5) by held-out log-loss.
- Store in the existing generalized `app/market_blend.py` config for the
  club engine; **used at edge/pricing time only** (bet selection), never fed
  back into fit(). Promote from experimental default-OFF to ON per §12 with
  the additional criterion that simulated ROI-vs-closing (P1.5) does not get
  worse.

---

## Phase 7 — Real xG from Understat (top-5 leagues)

Per `docs/DATA_SOURCING_PLAN.md` §2, unchanged in substance:

- `club_soccer/fetch_understat.py` → adds `home_xg, away_xg` columns to
  `fixtures.csv` (match on date+teams via names.py aliases; add an
  `UNDERSTAT_ALIASES` dict). Cache raw JSON under `data/understat_cache/`.
  Coverage: EPL, La Liga, Serie A, Bundesliga, Ligue 1 only.
- `model.fit()`: where `home_xg` is present, build `attack_xg/defence_xg`
  from real xG; else keep the SoT×conv proxy (per-row decision, so mixed
  datasets work).
- Rerun `--tune-ensemble` (the xg/xgf weights may shift). Promote per §12.
- Respect robots/ToS: single-threaded, cached, ~1 req/s, only current season
  incrementally after backfill.

---

## Phase 8 — Front door + daily updates

### P8.1 `club_soccer/season.py` (the only user-facing command)

`python3 -m club_soccer.season [--fast] [--no-network]` runs, in order, each
step wrapped in the `|| skipped` pattern (a step failure never kills the run):

1. `health.run_checks()` (abort only on the two hard checks).
2. Fetch results + upcoming (`fetch.fetch_fixtures(current=True)`).
3. Pull absences (P2.4), refresh player apps cache from new bsd_cache events,
   rebuild squads + minutes (P2.2/2.3).
4. Snapshot odds (P6.1). Refresh current-season fd.co.uk file (P1.3) and
   Understat (P7) — weekly (skip if fetched < 6 days ago).
5. Refit: `model.fit` + save (skipped with `--fast`).
6. Standings refresh (P4.1) for card display.
7. Edge: BSD-odds-based pricing with availability report, context
   multipliers (whatever is promoted), do-not-bet filter.
8. Write **`club_soccer/data/card.md`**:
   - Header: date, data freshness (days since last result, upcoming count,
     absence rows today, odds snapshot age).
   - Next 7 days, grouped by day then competition: `Home vs Away — model
     H/D/A %, fair odds, availability note (▲/▼ multipliers + confidence),
     rest/congestion note, league positions (e.g. "3rd v 17th")`.
   - **Backed bets only** section: market, side, odds, edge %, stake
     (quarter-Kelly × confidence haircut), suppressed-count footnote.
   - New transfers detected (P2.2) and notable absences.
9. Weekly (Mondays): `validate --gate` + `backtest_market` summary appended
   to the card footer.

### P8.2 Scheduling (manual or automatic — support both)

- Manual: `python3 -m club_soccer.season` — document as the one command in
  `club_soccer/README.md` (rewrite the README to lead with it).
- App-integrated: add one line to the root `update.sh` `morning` flow and the
  full default flow:
  `python3 -m club_soccer.season --fast || echo "   club soccer skipped"`.
  The existing v6 in-app scheduler (`data/update_schedule.json`, fires
  `./update.sh <mode>`) then covers club soccer with zero new scheduler code.
- Standalone (app closed): ship
  `club_soccer/com.sportspredictor.clubsoccer.plist` (launchd, 07:30 local,
  RunAtLoad false) + README install instructions
  (`launchctl load ~/Library/LaunchAgents/...`). Optional for the user.
- `club_soccer/update.sh` becomes a thin wrapper calling season.py then
  `validate --gate` and provenance (keep for compatibility).

---

## 12. Promotion gate (applies to every phase; copy, don't re-derive)

A candidate change may flip its `active` flag / become default only when ALL
hold, evaluated by the monthly-refit walk-forward (P1.4) with the candidate ON
vs OFF:

1. Primary metric improves: 1X2 Brier strictly lower — except weather (P5),
   whose primary metric is OU2.5 Brier with 1X2 Brier allowed to move ≤ +0.0005.
2. 1X2 log-loss not worse.
3. Time-split robustness: of the three splits `2025-01-01, 2025-07-01,
   2025-12-01` (train < split, test ≥ split), the candidate wins ≥ 2, and the
   worst split regression ≤ 0.0015 Brier (identical to `tune_ensemble`).
4. `python3 -m club_soccer.validate --gate` still passes.
5. For market-layer changes (P6): simulated ROI vs closing (P1.5) at the 4%
   threshold does not decrease.

On promotion: update `data/validation_baseline.json` via `--update-baseline`
and record the change + numbers in `docs/model_improvements_changelog.md`.
On rejection: keep the fitted artifact with `"active": false` and record the
negative result in the changelog (the honest-negative pattern already used).

## 13. Tests to add (extend `test_club_soccer.py`; keep it offline-runnable)

- `test_health`: `health.run_checks()` hard checks pass on shipped data.
- `test_repair_guard`: `_bsd_to_fixture_row` returns None for a finished
  event dated in the future.
- `test_feature_store_pit`: build a tiny synthetic fixtures frame where team
  A plays on d1, d5, d8; assert rest/congestion on the d8 row see only
  d1/d5, and that OUTCOME columns never appear in
  `schema.feature_columns()` output.
- `test_fdcouk_alias_coverage`: join coverage ≥ 95% per league on cached
  files (skip if cache absent).
- `test_minutes_windows`: synthetic apps → exact mins_7d/14d/30d values.
- `test_transfer_reattribution`: player with apps for A then B lands in B's
  squad and appears in transfers_detected.
- `test_standings_asof`: synthetic 4-team league; assert table and that a
  match on the asof date is excluded.
- `test_context_apply`: coefficients {rest_diff: 0.02} shift λ_h/λ_a in
  opposite directions by exp(±0.02·rd).
- `test_do_not_bet`: both suppression rules fire on constructed rows.
- `test_card_written`: season.py with `--no-network` on shipped data writes a
  card.md containing the freshness header.

## 14. Task order & sizing

| # | Task | Depends | Size |
|---|------|---------|------|
| 1 | P0.1–P0.3 repair + health | — | S |
| 2 | P1.1–P1.2 schema + feature store | P0 | M |
| 3 | P1.3 fd.co.uk odds + aliases | — | M |
| 4 | P1.4 validate totals/BTTS | P0 | S |
| 5 | P1.5 backtest_market | 2,3,4 | M |
| 6 | P2.1 player cache v2 | P0 | M |
| 7 | P2.2–P2.3 squads/transfers/minutes | 6 | M |
| 8 | P2.4–P2.5 absences + lineups | 6 | S |
| 9 | P3 context GLM (rest/cong/load) | 2,7 | M |
| 10 | P4.1–P4.3 standings + motivation | 2 | M |
| 11 | P4.4–P4.7 comp strength, promo priors, season boundary | 2 | M |
| 12 | P5 weather | 2,9 | M |
| 13 | P6 market layer | 3,5 | M |
| 14 | P7 Understat xG | 2 | M |
| 15 | P8 season.py + scheduling + README | 1–11 min | M |

Work tasks 1–5 first (repair + honest measurement), then 6–9 (the biggest
missing signal: who plays and how tired they are), then 10–11, then 12–15 in
any order. After every task: `python3 test_club_soccer.py && python3
run_checks.py`, and for model-affecting tasks the §12 gate.

## 15. Definition of done

- `python3 -m club_soccer.season` runs end-to-end on a clean checkout with a
  BSD key, and with `--no-network` on cached data, producing a card.md with
  upcoming fixtures, availability/rest/position context, and backed bets.
- `health` hard checks pass; no future-dated results; upcoming fixtures
  present in-season.
- Feature store builds a leak-free training matrix ≥ 15k rows with market
  outcome columns joined for ≥ 90% of top-league rows.
- `backtest_market` reports model-vs-closing log-loss and simulated ROI.
- Rest/congestion, availability, motivation, weather, market-blend and xG
  each exist as fitted artifacts with an honest promoted/rejected verdict in
  the changelog — whichever way the data falls.
- README leads with the front door; all suites green.
