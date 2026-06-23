# Data Sourcing & Integration Plan

Date: 2026-06-23

Where to get the external data behind the high-value levers, and exactly where it
plugs into the repo. Ordered by payoff-per-effort. Each engine already has a
`fetch.py` + a source-of-truth CSV + a `fit()/predict()`; the work is extending
the fetcher to write new columns and teaching `fit()` to use them — not new
infrastructure.

Provider facts below were checked June 2026; pricing/terms move, so re-confirm at
the linked pages before committing.

---

## 1. Tennis — serve/return profiles (EASIEST; data already flows)

**The data is already in the source we pull.** Jeff Sackmann's `tennis_atp` /
`tennis_wta` match CSVs carry per-match serve stats: `w_ace, w_df, w_svpt,
w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced` (and `l_*` for the
loser). `tennis/providers.py` (`SackmannProvider`) already reads these files — it
just drops the serve columns when writing `matches.csv`.

- **Source:** github.com/JeffSackmann/tennis_atp + tennis_wta (CSV, free).
  License **CC BY-NC-SA 4.0** (non-commercial — fine for research/paper-trading;
  matters only if this ever goes commercial).
- **Plug in:**
  1. `tennis/fetch.py` / `providers.py`: keep the serve columns when building
     `data/matches.csv` (add `w_svpt, w_1stWon, w_2ndWon, w_ace, w_df, ...`).
  2. `tennis/model.py fit()`: aggregate to a per-player **serve-points-won %** and
     **return-points-won %** (time-decayed, EB-shrunk to tour means, like the
     existing form fit), store in `*_model_params.json`.
  3. `tennis/simulate.py`: replace the symmetric `BASE_SERVE = 0.64` in
     `point_edge_for_target` with each player's fitted serve/return rate (still
     re-centred to stay consistent with the Bradley-Terry match prob). This is the
     fix that makes total-games / set-handicap / first-set markets reflect who is
     actually a big server.
- **Effort:** low. **Cost:** free. **Coverage:** ATP/WTA tour-level main draws;
  `adf_flag` marks the few matches with no ace/DF data — skip those rows.

---

## 2. Club soccer — real xG (replaces the SoT proxy)

The `xg`/`xgf` ensemble components currently approximate xG as `SoT ×
league_conversion`. Real shot-quality xG is the single biggest signal upgrade.

- **Source:** **Understat** (free; top-5 European leagues only — EPL, La Liga,
  Serie A, Bundesliga, Ligue 1). Serves data as JSON inside `<script>` tags, no
  aggressive bot protection. **Avoid FBref for current xG** — it lost its Opta
  licence in Jan 2026, so its advanced stats no longer update (historical only).
  For history/back-test, **StatsBomb open data** (free, CC) is an option but covers
  limited competitions.
- **Access:** scrape Understat match JSON directly in Python (e.g. an
  `understat` provider), or the `understatapi` package. No key.
- **Plug in:**
  1. `club_soccer/fetch.py`: add an Understat fetch that writes `home_xg`,
     `away_xg` columns onto `data/fixtures.csv` (match on date + team, via
     `names.py` aliasing).
  2. `club_soccer/model.py fit()`: when `home_xg/away_xg` are present, build the
     `attack_xg/defence_xg` maps from **real xG** instead of `SoT × conv`; keep the
     SoT proxy as the fallback for leagues Understat doesn't cover.
- **Caveat:** Understat does **not** cover the Championship, League One/Two, or the
  Scottish leagues — those keep the SoT proxy. So this lifts the top-5-league
  subset only. **Effort:** medium. **Cost:** free.

---

## 3. CFB — weather for totals

`power.py` totals are weather-blind; wind and precipitation are knowable pre-kick
and systematically lower scoring.

- **Source (weather):** **Open-Meteo Historical Weather API** (`/v1/archive`,
  ERA5 reanalysis, 1940→present, hourly wind/precip/temp by lat/lon). **Free, no
  key** for non-commercial use, plain HTTP GET → JSON.
- **Source (venue coordinates):** **CollegeFootballData** `/venues` endpoint
  returns stadium lat/lon (and dome flag). The repo already uses CFBD (the
  `data/cfbd/ppa_games_*.json` came from it). Free tier = 1,000 calls/mo; Patreon
  Tier 3 ($10/mo) = 75k calls + the Patreon-only `/games/weather` endpoint if you'd
  rather skip Open-Meteo entirely.
- **Plug in:**
  1. One-off: build `cfb/data/venues.csv` (venue_id → lat, lon, dome) from CFBD.
  2. `cfb/fetch_data.py`: for each game, query Open-Meteo at the venue/kickoff →
     `cfb/data/weather.csv` keyed by `game_id` (wind_mph, precip, temp, dome).
  3. `cfb/power.py`: add a fitted weather term to the **total** only (dome ⇒ no
     adjustment; high wind / precip ⇒ negative). Re-validate with
     `totals_backtest.py` and the gate — promote only if held-out total MAE
     improves (the totals shrinkage already landed; weather stacks on top).
- **Effort:** medium (mostly the venue table + back-filling historical weather).
  **Cost:** free via Open-Meteo.

---

## 4. Availability / lineups (club soccer + World Cup)

### Club soccer injuries
- **Source:** **API-Football** `injuries` endpoint (list of players likely to miss
  a fixture). The repo already authenticates to API-Football in
  `club_soccer/fetch.py` (key `api-football`). Free plan = **100 requests/day**
  (enough for a handful of leagues if cached); paid tiers lift that.
- **Plug in:** fetch → `club_soccer/data/absences.csv` (team, player, status) →
  add a squad-gap λ adjustment in `model.predict` (port the World Cup
  `squads.py` "available minus full-strength" pattern; gate it behind the
  validation harness).

### World Cup absences (mostly already built)
- `engines/worldcup/squads.py` already reads `data/absences.csv` and
  `wc_v4/availability.py` already prices absences with lineup-confidence bands —
  but they're **report-only / default-off**. The missing piece is *data*, not code.
- **Source:** during the tournament, API-Football `injuries` for national teams, or
  a hand-maintained `data/absences.csv` (the format squads.py expects:
  `team,player,note`). Then let the existing validation gate decide whether to flip
  availability into the default prediction.
- **Effort:** low–medium (WC) / medium (club). **Cost:** free tier workable.

### CFB starting QB (highest value, weakest free feed)
- There is **no clean free "who's starting/injured this week" feed.** Options:
  - CFBD rosters + `player/usage` to identify the season's primary QB, then a
    **hand-maintained `cfb/data/qb_status.csv`** (team, week, qb_out|backup) from
    depth-chart/injury news — practical and accurate, low volume.
  - Scrape a depth-chart site (brittle, ToS risk).
- **Plug in:** `cfb/predictor.py blend_predict` applies a fitted points/win-prob
  adjustment when the listed starter is out. Validate via `cfb/validate.py`.
- **Effort:** medium (ongoing manual upkeep). This is the biggest single CFB lever
  but the least automatable.

---

## 5. Golf — strokes-gained categories + tee-wave (bonus)

- **Free path (already partly in repo):** `golf/providers/pgatour_stats.py`
  already scrapes PGATour.com SG categories (OTT/APP/ATG/PUTT) into
  `pgatour_stats.csv`; ESPN endpoints give tee times. Use SG categories to give
  `model.fit` a structural course-archetype fit, and tee times to add the R1/R2
  wave bias on top of the round-correlation change already shipped.
- **Paid gold standard:** **DataGolf** API (requires the **Scratch Plus**
  subscription; Scratch Basic is $20/mo, Plus is higher) — round-level
  strokes-gained, tee times, and course-fit model predictions across 22 tours.
  Worth it only if the free SG scrape proves the signal first.
- **Effort:** medium. **Cost:** free (PGATour/ESPN) → optional paid (DataGolf).

---

## Suggested order

1. **Tennis serve rates** — data already pulled, just parse + fit. Fast, clean win.
2. **Club xG (Understat)** — biggest soccer signal; top-5 leagues.
3. **CFB weather (Open-Meteo + CFBD venues)** — stacks on the totals shrinkage.
4. **Availability:** flip the WC absences path on (code exists) → club injuries →
   CFB QB (manual CSV).

Every one lands report-only first and only becomes a default if it beats the
engine's existing walk-forward gate — same discipline as the four changes already
implemented.

## Sources
- [CollegeFootballData API tiers](https://collegefootballdata.com/api-tiers) · [CFBD site/venues & weather](https://collegefootballdata.com/)
- [API-Football pricing](https://www.api-football.com/pricing) · [API-Football injuries endpoint](https://www.api-football.com/news/post/new-endpoint-injuries)
- [Understat](https://understat.com/) · [Best xG sites 2026 (FBref lost Opta licence)](https://statpair.com/blog/best-xg-websites-2026-comparison)
- [Sackmann tennis_atp](https://github.com/JeffSackmann/tennis_atp) · [tennis_wta](https://github.com/JeffSackmann/tennis_wta)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [DataGolf API access](https://datagolf.com/api-access) · [DataGolf subscribe](https://datagolf.com/subscribe)
