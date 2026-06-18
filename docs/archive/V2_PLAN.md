# World Cup Prediction Engine — v2 Plan

Implementation spec for the next iteration of this engine. Written to be executed
cold, milestone by milestone, by Claude Opus (or any capable agent) without access
to the conversation that produced it. Read this whole file before starting M1.

---

## 0. Context: what v1 is

A World Cup 2026 match/tournament prediction and value-betting engine, live mid-
tournament (group stage started June 11, 2026). All paths relative to this folder.

| File | Role |
|---|---|
| `predictor.py` | Elo (K by tournament, margin-scaled) + Poisson goal model on Elo diff, fixed DC rho −0.10 |
| `dixoncoles.py` | Attack/defence MLE fit (2.5y decay half-life, 12y window, L2 0.001, fitted rho). `build_sources(model)` is the central lambda-source factory used by simulate.py and edge.py. `--model elo\|dc\|blend`, blend (50/50) is default |
| `simulate.py` | 20k Monte Carlo: groups (played matches fixed), FIFA tiebreakers, R32 bracket, approx Annex C third-place mapping → `tournament_odds.csv` |
| `edge.py` | De-vig median odds (The Odds API or manual `odds.csv`), EV, quarter-Kelly; auto-records bets (edge ≥3%, best outcome/match, ≤36h, exposure cap) to ledger. `--squad-adj` opt-in |
| `bankroll.py` | Ledger settle/compound; bankroll in `data/bankroll.json` (started £100, 2026-06-10). Real money is placed manually by Barrie at bookmakers; the ledger mirrors it |
| `squads.py` | EA FC 26 squad power (mean top-18 overall); only full-vs-available gap adjusts predictions (~23 Elo/rating pt, cross-team calibration); <15 EA-matched players → no adjustment |
| `injuries.py` | API-Football feed — **dead on free tier** (no injuries post-2024). Absences come from `data/absences.csv`: manual rows + `news:` rows maintained by the daily scheduled task |
| `update.sh` | Daily: refresh results (git clone martj42/international_results) → settle → DC refit → squad refresh → predictions → simulation → edge report → tracker xlsx |
| `wc2022_replay.py` / `wc2022_backtest.py` | WC2022 replay harness; `data/wc2022_odds.csv` has historical odds |

Reference backtest (matches since 2024-01-01, n≈2,538): Elo 60.2% / 0.5070 Brier,
DC 59.8% / 0.5089, blend 60.4% / **0.5038**. These are the numbers to beat.

### Environment constraints (verified, do not rediscover painfully)

- Sandbox network is allowlisted: `git clone github.com` works; raw.githubusercontent,
  api.the-odds-api.com, wikipedia, api-football are **blocked** (proxy 403). Scripts
  needing those must run on Barrie's machine — follow the edge.py pattern: try, fail
  with a clear message telling him what to run in Terminal.
- `mcp web_fetch` truncates at ~64K chars and only fetches URLs already seen in chat.
- The mounted folder forbids `unlink` of files in `data/` from the sandbox — avoid
  temp-file-and-delete patterns; pass DataFrames in memory instead.
- Team names must match `data/results.csv` exactly ("United States", "South Korea",
  "Ivory Coast", "Czech Republic"). Alias maps exist in squads.py / injuries.py /
  edge.py — extend those, don't create new ones.

### Invariants — break these and Barrie's daily pipeline breaks

1. Default behaviour of every CLI without new flags must be byte-compatible with v1
   (same columns in predictions_worldcup_2026.csv, tournament_odds.csv, ledger.csv).
2. `update.sh` must finish offline (no network beyond the git clone, which already
   has a fallback) — every new step needs a `|| echo skipped` guard.
3. Never auto-place real money. The ledger is a paper record Barrie mirrors manually.
4. The ledger format (`placed_on,match_date,home,away,side,bet,odds,stake,status,pnl,
   bankroll_after`) is append-plus-settle only; outright rows use `—OUTRIGHT—`.
5. Don't refactor for elegance. Each milestone touches the minimum files listed.

---

## 1. Milestones, impact-ordered

Execute in order. Each has acceptance criteria — run them, show the output, and do
not proceed while one fails. If a milestone's gate can't be met, revert its changes
and record why in `V2_NOTES.md` rather than shipping a regression.

### M1 — Validation harness with regression gates (foundation, ~1 session)

Everything later needs a trustworthy yardstick first.

Build `validate.py`:
- Walk-forward evaluation on matches since 2022-01-01 (point-in-time Elo already
  exists via `compute_elo`; for DC, refit at each month boundary using only prior
  matches — slow is fine, cache fits in `data/validation_cache/`).
- Metrics per model (elo/dc/blend): 3-way accuracy, Brier, log-loss, and a
  reliability table (10 probability bins: predicted vs observed frequency).
- `--gate` mode: exits non-zero if blend Brier exceeds the stored baseline by >0.002.
  Store baseline in `data/validation_baseline.json` on first run.
- Seed all randomness (numpy default_rng(42)); simulate.py should accept `--seed`.

Acceptance: `python3 validate.py` prints the three-model table and writes the
baseline; `python3 validate.py --gate` passes; blend numbers within noise of the
v1 reference above; `./update.sh` still completes.

### M2 — Probability calibration (highest accuracy ROI, ~1 session)

Match models are usually miscalibrated at the extremes, which directly distorts
Kelly stakes and the 3% edge threshold.

- In `validate.py`, fit isotonic regression per outcome (H/D/A) on walk-forward
  blend predictions (sklearn if available — `pip install scikit-learn
  --break-system-packages` — else implement PAV, ~30 lines, no dependency).
  Renormalise the three calibrated probabilities to sum to 1.
- Save the fitted maps to `data/calibration.json` (piecewise-linear knots).
- New module `calibrate.py` with `apply(p_home, p_draw, p_away)`; edge.py gets
  `--calibrated` flag applying it after the blend, before edge computation.
- Calibration must be refit only by `validate.py` (never inside edge.py).

Acceptance: walk-forward Brier and log-loss of calibrated blend ≤ raw blend on a
held-out final 6 months (fit isotonic on pre-2025-12 predictions only, test after);
reliability table visibly flattened; `edge.py` without `--calibrated` byte-identical
to before; `validate.py --gate` passes.

### M3 — Market-anchored probabilities + CLV tracking (highest betting ROI, ~1-2 sessions)

The model alone beats noise, not closing lines. Anchoring to the market removes
most fake edges; CLV measures whether the operation is genuinely +EV.

Market blend:
- In edge.py: when odds exist, compute de-vigged market probs (already done) and
  blend in logit space: `logit(p_final) = w·logit(p_model) + (1−w)·logit(p_market)`,
  renormalised. Fit w by maximising log-likelihood on `data/wc2022_odds.csv` +
  wc2022 replay predictions (expect w ≈ 0.2–0.4; report the fitted value).
- Store w in `data/market_blend.json`. Flag `--market-blend` (opt-in), combinable
  with `--calibrated` (calibrate first, then blend).
- Recompute edges against the *raw* market probs as before — the blend changes
  p_model only. Expect far fewer ≥3% edges; that is the point, not a bug.

CLV tracking:
- `clv.py --snapshot`: record current odds for open ledger bets into
  `data/odds_history.csv` (needs The Odds API → runs on Barrie's machine; degrade
  gracefully offline). The last snapshot before kickoff = closing proxy.
- `clv.py --report`: per settled bet, CLV% = (bet_odds/closing_odds − 1); rolling
  mean CLV, win rate, P&L vs CLV-expected P&L.
- Add a `clv` column to the Betting Tracker refresh in `refresh_tracker.py`.

Acceptance: fitted w on WC2022 strictly improves log-loss vs both pure model and
pure market on that sample; edge.py default output unchanged; `clv.py --report`
runs with an empty history (prints "no snapshots yet"); ledger format untouched.

### M4 — Knockout correctness: 90-minute settlement + exact Annex C (~1 session)

- Split knockout predictions: 1X2 markets settle on 90 minutes — `market_probs` in
  edge.py must use the 90-minute score matrix (draw stays a draw). Progression
  (for simulate.py) keeps ET/penalty logic. Today both use full-time; group stage
  was unaffected but R32 starts June 28 — **this must land before then.**
- bankroll.py settlement: for knockout matches, settle 1X2 on the 90-minute result.
  The results dataset reports full-time; add `data/ko_overrides.csv`
  (`date,home,away,score90`) which the daily task fills from news when a match went
  to extra time — settlement consults it first, else assumes the FT score is 90'.
- Replace the third-place constraint-matching in simulate.py with FIFA's exact
  Annex C allocation table (hardcode the table; cite the FIFA regulations PDF URL
  in a comment so it can be re-verified).

Acceptance: re-run WC2022 replay — bracket third-place slotting must match the
actual WC2022 R16 (table for 2026 differs in size but logic is verifiable on 2026
regulations; verify no crash + plausible slots for all 81 qualification patterns);
group-stage predictions byte-identical; `validate.py --gate` passes.

### M5 — Squad layer v2 (~1-2 sessions)

- Backfill the 12 `ea_proxy` squads in `data/squads.csv` with official lists
  (FIFA squad PDF / Wikipedia — needs network: do it via chat tools or Barrie's
  machine, then re-run `squads.py`). Teams: see `source` column.
- Position-aware adjustment: an absent GK/DF should mostly raise opponent λ, an
  absent FW mostly lower own λ. Split elo_adj into att_adj/def_adj by the absent
  player's position (GK/DF → 75% defence; MF → 50/50; FW → 75% attack) and apply
  asymmetrically in `adjusted_sources`.
- Starter-weighting: weight each player by likely minutes — proxy: rank in squad
  by overall; ranks 1–11 weight 1.0, 12–18 weight 0.5, rest 0. (Replaces flat
  top-18 mean. Re-run the Elo-per-point calibration after.)
- Sanity backtest: for 5–10 known WC2022 absences (e.g. Sadio Mané, N'Golo Kanté,
  Pogba — verify list via news), apply the method with EA FC 23 ratings if
  obtainable, else current ratings as approximation, and check the adjusted
  probabilities don't *worsen* WC2022 replay log-loss on affected matches.

Acceptance: 48/48 squads with `source != ea_proxy` (or documented exceptions);
what-if CLI still works; WC2022 absence check not worse than no-adjustment;
default (no `--squad-adj`) output unchanged.

### M6 — Context features: rest, travel, altitude (~1 session, expect small gains)

- Build per-fixture features for WC2026: days rest differential (from fixture
  list), great-circle km travelled since last match (stadium coords — hardcode the
  16 venues), altitude flag (Mexico City 2240m, Guadalajara 1566m, Monterrey 540m).
- Fit a single multiplicative λ correction on historical tournament data
  (rest-diff and altitude-vs-lowland teams; travel likely insignificant — drop
  features whose coefficient |t| < 2 rather than shipping noise).
- Apply inside `build_sources` behind `--context` flag; document fitted
  coefficients in V2_NOTES.md.

Acceptance: walk-forward log-loss with `--context` ≤ without, on tournament
matches specifically; coefficients reported with std errors; default unchanged.

### M7 — Portfolio staking discipline (~1 session)

- Simultaneous Kelly: when multiple same-day bets are recommended, size them
  jointly (independent approximation: scale each stake by total-exposure solve)
  rather than sequentially; respect existing exposure cap.
- Correlation guard: forbid combined exposure >1.5× single-match cap on bets whose
  outcomes are correlated (same match different markets, outright + matches of the
  same team). Outright EW + match win on the same team counts as correlated.
- Drawdown brake: if bankroll < 70% of its running peak, halve Kelly fraction until
  a new peak. State in `data/bankroll.json` (add fields; stay backward-compatible).
- All inside edge.py's recording step; report the pre/post-scaling stakes.

Acceptance: synthetic test — feed edge.py a fabricated odds.csv with 6 big-edge
same-day outcomes and confirm total recorded stake ≤ caps; bankroll.json from v1
loads without error; ledger format unchanged.

### M8 — Ops polish (~1 session, after the above)

- `update.sh`: run `clv.py --snapshot` and `validate.py --gate` (gate failure
  prints a loud warning in the daily summary, never blocks the update).
- Morning bet queue: edge.py writes `bet_queue.csv` (match, bet, odds, stake,
  edge raw/calibrated/market-blended, squad adjustments in play) — the scheduled
  task includes it in the summary so Barrie reviews before placing real bets.
- Single-page HTML dashboard `report.py` → `dashboard.html`: bankroll curve,
  CLV trend, calibration plot, today's queue, title-odds movers (matplotlib or
  inline SVG; no external JS dependencies so it renders offline).
- README: new "v2" section documenting every new flag; update the scheduled-task
  prompt (ask Barrie before changing it — it's standing config).

Acceptance: `./update.sh` completes offline end-to-end; dashboard renders with
file:// open; README accurate (spot-check every documented command actually runs).

---

## 2. Sequencing and effort summary

```
M1 harness ──► M2 calibration ──► M3 market+CLV ──► M7 staking
        │                                   ▲
        ├─────► M4 knockout (DEADLINE: before June 28, 2026)
        ├─────► M5 squads v2
        └─────► M6 context features
M8 ops last (depends on M1–M3 outputs existing)
```

If time-boxed, the order of value is: M1, M3, M4 (deadline), M2, M7, M5, M6, M8.
M4 jumps the queue as June 28 approaches.

## 3. Working agreements for the executing agent

- One milestone per session/PR-equivalent; run the full acceptance block before
  declaring done; append a dated entry to `V2_NOTES.md` (created at M1) with what
  shipped, fitted parameters, and gate numbers.
- Anything needing live network goes in Barrie-runs-it-locally scripts with the
  edge.py failure pattern, OR is fetched via chat tools at build time and committed
  as data. Never curl/requests around the sandbox proxy.
- New flags default OFF until their gate has passed twice (two daily runs); then
  propose — don't silently make — a default flip to Barrie.
- Don't touch: bankroll history, ledger rows, `data/results.csv` (regenerated daily).
- When the tournament ends (final July 19, 2026), M4/M5 urgency dies; the harness,
  calibration, market blend and CLV generalise to any competition — prefer them.
