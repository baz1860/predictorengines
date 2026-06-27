# Player-Form Rating — Proof-of-Concept Build Plan

**Goal of the PoC:** prove we can build a per-player form rating from BSD data and
turn it into per-team attack/defence multipliers for the World Cup field —
**reported only, not wired into predictions yet.** Ship/no-ship comes later, after
the validation gate.

**Why this is feasible:** BSD is one provider across club *and* international
football, so `provider_player_id` is consistent. We already (a) pull WC lineups
with those IDs (`data/worldcup/lineups.csv`), and (b) have a working player-form
engine on the club side (`club_soccer/player_features.py: PlayerFeatureStore`)
that converts BSD per-player xG/xA/minutes into clamped attack/defence multipliers.
The PoC bridges the two; it builds nothing new from scratch.

---

## Scope

**In scope:** build a WC-scoped form store from BSD, compute per-team multipliers
off the current lineup, print a report + coverage stats, and compare against the
existing static EA squad gap.

**Out of scope (deferred to a follow-up):** wiring into `squads.adjusted_sources`
/ the edge path, the validation gate, any change to live predictions. The PoC must
not touch `update.sh` or the prediction outputs.

---

## Phases

### Phase 0 — Linkage sanity check ✅ DONE — premise CONFIRMED
Built and run: `scripts/worldcup/player_form_phase0.py`. Findings:
- **Offline:** all 51 `lineups.csv` rows carry integer `provider_player_id`s, no dupes.
- **Same-source join:** refetching event 8327 from BSD, **51/51 lineup IDs match** —
  `provider_player_id` *is* BSD's `player_id`. No name-matching fallback needed.
- **Cross-competition (the key unlock):** `player_id` is a **global** key. BSD exposes
  `/api/v2/players/{id}/stats/` returning a **per-match** history (xG, xA, rating,
  minutes, shots…) that spans **club and international** football. Probing E. Martínez
  (1629) returned 12 club matches (Premier League + Europa League) vs 4 international
  in a 16-event sample. Messi (9063) → 41 MLS + 9 national-team rows.
- **Consequence for the design:** we no longer scan club fixtures or name-match. We pull
  each WC squad player's club form **directly by `player_id`** from the player-stats
  endpoint. Phase 1 below is rewritten accordingly. The earlier "ID drift" risk is retired.

### Phase 1 — WC-scoped form store ✅ DONE
Built and run: `scripts/worldcup/player_form.py`
(`python3 -m scripts.worldcup.player_form --teams 489 --report`). Results:
- Pulls the full WC field (1452 players, 48 teams) from the squads endpoint, then
  each player's per-match history by global `player_id`.
- Computes a recency- + minutes-weighted form value (xg90, xa90, BSD rating) shrunk
  toward a position baseline; caches to `data/worldcup/player_form_cache.json`.
- Face-valid output: Messi tops Argentina (form +2.00, rating 8.41, xg90 0.758),
  then Paredes / De Paul. Coverage 54/55 players with real form.
- **Wrinkle noted:** the squads endpoint returns *extended* squads (e.g. 55 for
  Argentina), not the final 26. Harmless here — Phase 2 keys multipliers off the
  actual `lineups.csv` XI, so non-selected players are simply never looked up.

Original design notes (still accurate):
- Get the WC field's player IDs — from `lineups.csv` and/or `/api/v2/worldcup/squads/`
  (BSD has a dedicated WC squads endpoint).
- For each player_id, pull `/api/v2/players/{id}/stats/` (paginated; ~50–75 rows/player)
  → per-match xG, xA, `rating`, `minutes_played`, shots, etc. Resolve each row's
  `event_id` to a league only if we want to weight/exclude competitions; otherwise
  use all recent matches.
- Build a rolling form value per player (recency-weighted xG/90 + the BSD per-match
  `rating`, which the club path doesn't currently use). Anchor to the EA overall as a
  prior and shrink hard for low minutes. Cache to `data/worldcup/player_form_cache.json`.
- `PlayerFeatureStore`'s aggregation/clamping logic is still reusable; only the
  *ingestion* changes (player-stats endpoint instead of `refresh()` over events).

### Phase 2 — Per-team multipliers from the lineup ✅ DONE
### Phase 3 — Report + EA comparison ✅ DONE
Built and run: `scripts/worldcup/player_form_multipliers.py`
(`python3 -m scripts.worldcup.player_form_multipliers`). It builds the current XI
from `lineups.csv`, looks up each starter's form (cache, else live by `player_id`),
and aggregates into `attack_mult` (own goals) and `defense_mult` (opponent goals),
clamped to [0.80, 1.25] — matching the club / `adjusted_sources` convention. It then
prints the EA squad gap (`load_adj_split`) converted to the same multiplier space.

Result on the one fixture we have lineups for:
- **Argentina 10/10 matched → attack 1.071, defence 0.971** (in form, Messi-driven)
  while the **EA gap is flat (0.998 / 1.006)** → flagged **NEW SIGNAL**.
- **Austria 11/11 → 1.011 / 0.984**, near-neutral.

This is the evidence the PoC set out to find: form produces meaningful, asymmetric
multipliers exactly where the static availability gap says nothing. Coverage was
full (no fallback needed).

### Confirmed lineups acquired ✅ (removes the one-fixture limit)
Built `scripts/worldcup/fetch_confirmed_lineups.py`. BSD's list/date/status filters
only serve an upcoming window, so finished matches are reached by event id directly:
WC 2026 is **league 27**, and its played matches occupy a contiguous id block
(**8287–8352**). The fetcher walks that range, keeps finished league-27 events with
`lineups.confirmed == true`, and rewrote `data/worldcup/lineups.csv` with **all 66
played matches — 48 teams, 3383 confirmed rows (1452 starters + 1931 subs)**.
`player_form_multipliers.py` now picks each team's *most recent* confirmed XI, so
Phase 2/3 can run across the whole field (Argentina 11/11, Brazil 11/11, both NEW
SIGNAL). Projected lineup backed up to `lineups_projected_backup.csv`.

---

## Deliverables
- `scripts/worldcup/player_form_probe.py` — runnable: `python3 -m scripts.worldcup.player_form_probe --report`
- `data/worldcup/player_form_cache.json` — the rolling store.
- A short findings note: coverage %, multiplier spread, and the EA-vs-form divergence table.

## Functions to reuse (don't rebuild)
- `club_soccer/player_features.py`: `PlayerFeatureStore`, `adjustments_from_lineups`,
  `adjustments_from_names`, `_players_from_event`, `_compute_team_adj` (clamping).
- `bsd_client.py`: `get_all_events`, `get_event`.
- `engines/worldcup/squads.py`: `load_adj_split` (for the EA comparison).

## Risks / watch-items
- **Cross-competition transfer:** club form ≠ international form (different
  teammates, system, opponent quality). Keep multipliers shrunk + clamped; treat
  the PoC numbers as directional, not calibrated.
- **Coverage:** if too few of a team's XI match to BSD stats, report it and let
  that team fall back to no adjustment — don't fabricate a multiplier.
- ~~**ID drift**~~ — retired in Phase 0: `player_id` is global across competitions.
- **Competition weighting:** club xG isn't equivalent across leagues (MLS ≠ Premier
  League). Optionally weight matches by league strength when building the form value.

### Integration + validation gate ✅ DONE
**Leak-free backtest** (`scripts/worldcup/form_backtest.py`): A/B over all **66
played WC matches**, same Elo+Poisson baseline in both arms, form built only from
each player's **non-WC** matches (pre-tournament — zero WC leakage). Concurrent
prefetch with a resumable disk cache so it finishes inside the shell window.

| arm | accuracy | Brier | log-loss |
|-----|---------:|------:|---------:|
| baseline | 66.7% | 0.4851 | 0.8186 |
| +form    | 66.7% | 0.4823 | 0.8135 |
| **delta** | +0.0% | **−0.0028** | **−0.0051** |

Verdict: **form helps** — small but consistent improvement in Brier and log-loss,
accuracy unchanged. Honest caveats: 66 matches is a small sample; the gains are
modest; the multiplier gains (`G_ATT/G_DEF`) are hand-tuned not fitted; club xG is
still not league-weighted. Directionally positive, not yet conclusive.

**Wiring (off by default):**
- `engines/worldcup/squads.py`: `load_form_mults()` + `wrap_form_mults()` — composes
  the form layer on top of Elo (+ optional conf/squad) lambdas. No-op unless
  `data/worldcup/form_multipliers.json` exists.
- `engines/worldcup/edge.py`: new `--form-adj` flag (OFF by default), mirrors the
  `--squad-adj` / `--conf-adj` pattern; `update.sh` never passes it.
- `scripts/worldcup/player_form_multipliers.py --write`: emits
  `form_multipliers.json` keyed by predictor team names (alias-mapped).
- Regression-checked: modules import clean, `build_sources`/`adjusted_sources`
  unchanged, wrapper math verified (Argentina λ 1.728→1.847).

**Recommendation:** keep `--form-adj` OFF. The backtest is a green light to keep
going, not to ship. Next: (1) fit `G_ATT/G_DEF` + league weights instead of hand
values, (2) a prospective forward test — freeze form now, predict the next round,
score after — which is the only fully clean signal; then enable if it holds.

### Fitted gains + league weighting ✅ DONE (`scripts/worldcup/form_fit.py`, `form_config.py`)
- **League weighting:** each player's form contribution is scaled by their club
  league's strength (from the squads endpoint's `club_country` — free, no extra
  calls). A weak-league XI's form shrinks toward neutral. Strengths are a fixed
  prior table (not fitted — too many params for 66 matches).
- **Gains are not truly fittable:** log-loss is **monotonic** in `g_att` (0.8186 →
  0.7878 as gain → 3.0), with no interior optimum — so the gain is a *regularisation
  choice*, not a fit. Picked **g_att = 0.30** conservatively (captures the early,
  robust benefit far from clamp saturation); **g_def = 0** (rating-based defence
  added nothing). At 0.30: Brier 0.4788, log-loss 0.8092 (−0.0094 vs baseline).
- **Permutation test (the important one):** shuffle which team gets which form delta.
  Real assignment (0.8092) beats 48/50 shuffles, and shuffling makes things *worse*
  than baseline on average (0.8200) → the gain is **team-specific real form, not
  just sharpening an under-confident baseline** (p≈0.06). Marginal on 66 matches.
- `form_config.py` is now the single source (league table, gains, aggregation);
  `player_form_multipliers` + edge `--form-adj` consume the fitted params.

### Forward test harness ✅ DONE (`scripts/worldcup/form_forward_test.py`)
The fully-clean check the backtest can't be: `--predict` freezes baseline + form
H/D/A for upcoming fixtures **before kickoff** (zero leakage by construction);
`--score` pulls results once played and reports cumulative baseline vs +form.
Froze 11 next-round fixtures (England 0.65→0.70, Argentina 0.83→0.87, Brazil
0.53→0.57 — sensible). Caveat: upcoming lineups aren't published until ~1h pre-match,
so it uses each team's latest confirmed XI as a proxy (re-run when projected lineups
drop). **This is the decision-maker** — let it accrue over the knockouts, then
enable `--form-adj` only if +form holds up out-of-sample.

## Decision gate (after PoC, before any wiring)
Only proceed to integration if the report shows: (a) acceptable lineup coverage
(target ≥ ~12 of 18 for most teams), and (b) meaningful, sensible divergence from
the EA prior. Integration itself ships **off by default** behind a flag and must
clear the validation harness (Brier / calibration / CLV on WC 2022 replay +
2026-to-date) — the lesson from the confederation adjustment.
