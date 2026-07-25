# Codex Prompt: Aggressive Review of `club_soccer/`

Copy everything below the line into Codex.

---

You are performing an adversarial code and model review of the `club_soccer/` module (~8,700 lines of Python). Your job is to find problems, not to summarize the code. Assume the author is competent and the obvious things work — hunt for the subtle failures that lose money or corrupt predictions silently. Be harsh. No praise, no filler, no "overall the code is well-structured."

## Context

This module predicts club football match outcomes (Premier League through League Two, Scottish divisions, top European leagues, cups, UCL/UEL/UECL) and recommends bets. Real money is staked on its output, so model errors are financial losses.

Architecture: `season.py` orchestrates a daily pipeline — fetchers (`fetch.py`, `fetch_fdorg.py`, `fetch_fdcouk.py`, seeders) refresh `data/fixtures.csv`; `model.py` fits a Poisson attack/defence model with Dixon-Coles rho, an Elo model, and an ensemble blend (with Empirical-Bayes shrinkage of per-competition HFA and rho); `player_features.py` / `feature_store.py` / `availability.py` / `minutes.py` adjust for lineups; `context.py`, `motivation.py`, `weather.py` add situational factors; `market_model.py` + `fit_market_blend.py` blend with market odds; `edge.py` computes edges and quarter-Kelly stakes (haircut by lineup confidence); `validate.py` runs walk-forward validation; `backtest_market.py` backtests against the market; `calibrate.py` handles calibration. Tests live in `test_club_soccer.py` at repo root. Read `club_soccer/README.md` and `AGENTS.md` first.

## Part 1 — Code quality (aggressive)

Examine every file. Report:

1. **Correctness bugs**: off-by-one errors, wrong pandas semantics (silent NaN propagation, misaligned indexes, chained-assignment traps, timezone/date-parsing bugs), mutation of shared state, float comparison bugs, incorrect merges/joins that drop or duplicate rows.
2. **Silent failure paths**: bare/broad `except` blocks, fetchers that fail and leave stale data that downstream code treats as fresh, missing-file fallbacks that mask real problems, default values that hide errors.
3. **Data integrity**: schema drift between fetchers and consumers, team-name normalization gaps (`names.py`) that silently split one team into two, CSV read/write asymmetries, encoding issues, no validation on external API responses.
4. **Structure**: `model.py` (1,132 lines), `player_features.py` (912), `edge.py` (673) — identify god functions, duplicated logic across files (especially between the three fetchers and three seeders), dead code, and copy-paste divergence where two versions of the same logic have drifted apart.
5. **Robustness**: what happens on empty fixtures, a new team, a new competition, a postponed/abandoned match, mid-season squad changes, API rate limits, partial fetch failure mid-pipeline?
6. **Test coverage**: map `test_club_soccer.py` (536 lines) against the 8,700-line module. List the highest-risk untested paths, and identify tests that assert too little to catch regressions.

## Part 2 — Model quality (aggressive)

This is the part that matters most. Scrutinize the statistical methodology:

1. **Look-ahead bias / leakage**: Does any feature use information unavailable at prediction time? Check walk-forward validation in `validate.py` rigorously — is the train/test split truly temporal everywhere, including feature construction in `feature_store.py` and `player_features.py`? Are ensemble weights, calibration parameters, market-blend weights, or EB shrinkage constants fit on data that overlaps the evaluation window? Is `backtest_market.py` using odds that were actually available at bet time, or closing/settled odds?
2. **Poisson model soundness**: Is independence between home/away goals handled correctly beyond the Dixon-Coles low-score correction? Is overdispersion addressed or ignored? Is `MAX_GOALS = 10` truncation handled properly in probability normalization? Is the time-decay half-life (365 days) justified or arbitrary? Are promoted/relegated teams and cross-competition strength (`competitions.py`) handled without distorting ratings?
3. **Shrinkage and hyperparameters**: `HFA_SHRINK_K = 300`, `RHO_SHRINK_K = 400`, `DC_RHO = -0.08`, `RECENT_K = 6`, Elo K-factor, home advantage — which of these were tuned on the same data used to validate? Flag every magic number that functions as an untested hyperparameter.
4. **Ensemble and market blend**: How are goals/Elo/ensemble weights fit? Is the market blend circular (model influenced by odds, then edge measured against those same odds)? Does the do-not-bet filter in the market model have selection bias baked in from backtest overfitting?
5. **Calibration**: Is `calibrate.py` (52 lines) doing enough? Are probabilities calibrated per-outcome (H/D/A), per-league, per-odds-band? Draw probabilities are the classic weak point — check them specifically.
6. **Staking and edge**: Is quarter-Kelly computed on true (vig-removed) probabilities? Is the vig removal method correct (proportional vs. Shin/power)? Does the lineup-confidence haircut have any empirical basis? Is bankroll/simultaneous-bet correlation handled (multiple correlated bets same day)?
7. **Backtest validity**: Sample size, multiple-comparison problems across markets/leagues, survivorship in the odds snapshots, whether reported edge/ROI in `backtest_market.py` would survive realistic execution (odds movement, limits).
8. **Player-feature layer**: 900+ lines of lineup/minutes/availability logic — is there evidence any of it improves predictions, or is it unvalidated complexity that adds noise and failure modes?

## Output format

- **Findings ranked by severity** (Critical → High → Medium → Low). Critical = can lose money or corrupt predictions silently.
- Every finding: `file:line`, the specific code, why it's wrong, concrete fix.
- A **"most likely ways this system is fooling its operator"** section: the top 5 ways reported backtest/validation performance could be optimistic.
- A short prioritized remediation list (max 10 items).
- Do not report style nits (naming, line length) unless they cause bugs. Do not pad with praise or restate what the code does.

Run the tests (`pytest test_club_soccer.py`) and any quick static checks you want before writing conclusions. Where you claim a statistical flaw, demonstrate it with a small reproduction or cite the exact lines that prove it.
