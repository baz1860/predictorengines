# Model Signal Sprint Notes

Date: 2026-06-18

This sprint focused on core-model lift using current/free/local data only. UI,
ops, auto-betting, and paid-provider work were intentionally left out.

## Promoted Changes

### CFB: retuned champion blend

- Default CFB moneyline/margin blend uses `cfb/data/blend_weight.json` with
  `w_elo = 0.60`.
- Previous champion: `0.50 Elo / 0.50 points-power`.
- New champion: `0.60 Elo / 0.40 points-power`.
- Walk-forward window: 2023-2025, 2,394 FBS-vs-FBS games.
- Brier improved from `0.18849` to `0.18787`.
- Margin MAE held flat/slightly improved from `12.786` to `12.784`.
- Total MAE stayed `13.048` because totals still come from the power model.
- CFB baseline was tightened to the promoted champion.

### Club soccer: shot-pressure ensemble champion

- Added a local shot-pressure component from shots on target, non-SOT shots, and
  corners. Coefficients are fit only on the training slice and clipped to
  non-negative plausible ranges.
- Added `club_soccer/validate.py --tune-ensemble [--write]`.
- Added `club_soccer/data/ensemble_weights.json`; prediction now loads it when
  present and falls back to hardcoded champion weights otherwise.
- Previous champion: goals `0.20`, Elo `0.40`, long-run SoT-xG `0.20`,
  recent-form SoT-xG `0.20`, shot-pressure `0.00`.
- New champion: goals `0.15`, Elo `0.45`, long-run SoT-xG `0.00`,
  recent-form SoT-xG `0.20`, shot-pressure `0.20`.
- Walk-forward validation: 16,794 predictions.
- Brier improved from `0.612500` to `0.612352`.
- Log-loss improved from `1.021347` to `1.021128`.
- Initial three split checks all improved Brier within tolerance, so the artifact
  and club baseline were updated.
- A dry-run after promotion now rejects further changes because the artifact is
  already the current champion.

## Rejected / Monitored Challengers

### CFB EPA/PPA overall

EPA/PPA remains wired as an explicit challenger:

- `cfb/predictor.py --model epa`
- `cfb/predictor.py --model blend3`
- `cfb/validate.py --ablation`

Latest ablation rejected EPA for default promotion:

- Champion Brier: `0.18787`, margin MAE `12.78`, total MAE `13.05`.
- EPA-only Brier: `0.20415`.
- Equal-thirds Brier: `0.19175`.
- Best constrained stack assigned EPA weight `0.00`.

### CFB split PPA

Added `cfb/validate.py --ablation --ppa-splits` for pass/rush and down-split
PPA challengers.

- Champion Brier: `0.18787`, margin MAE `12.78`, total MAE `13.05`.
- Pass/rush 10% challenger Brier: `0.18947`.
- Down-split 10% challenger Brier: `0.18915`.
- Best constrained grid stayed `100%` champion.
- Split PPA remains rejected/default-off.

### Golf config tuning

Added optional config loading and `golf/validate.py --tune-config [--write]`.

- Search selected `course_k = 20` on the earlier split.
- Later validation headline Brier improved from `0.14595` to `0.14509`
  (`0.00086`), short of the `0.0010` promotion threshold.
- Top10/top20/cut all improved on the later split, but the headline threshold was
  not met.
- Full-window Brier moved from `0.14377` to `0.14309`.
- No `golf/data/model_config.json` was written.

### World Cup V4 market segmentation

- Validated local `data/wc2018_odds.csv` and `data/wc2022_odds.csv`.
- The market-blend gate now uses schema-valid local odds only and reports odds
  validation status.
- Market-covered sample size: `127`.
- V3 blend log-loss/Brier: `0.9779` / `0.5753`.
- V4 segmented blend log-loss/Brier: `0.9779` / `0.5753`.
- Sample-size and Brier gates passed, but log-loss improvement did not clear the
  `0.005` margin, so V4 segmentation remains report-only and default stays
  `v3_blend`.

## Validation Commands

```bash
python3 test_club_soccer.py
python3 test_cfb_blend.py
python3 test_golf_config.py
python3 cfb/validate.py --since 2023 --quiet --ablation
python3 cfb/validate.py --since 2023 --quiet --ablation --ppa-splits
python3 cfb/validate.py --since 2023 --quiet --gate
python3 club_soccer/validate.py --tune-ensemble
python3 club_soccer/validate.py --gate
python3 golf/validate.py --tune-config --sims 4000
python3 golf/validate.py --quiet --gate --sims 4000
python3 -m wc_v4.validate_v4
python3 validate_all.py --gate --sims 4000
```

Focused tests, engine gates, and full-suite gate passed.

## Next Model-Signal Work

1. CFB: add QB/depth-chart/weather/tempo features before trying more EPA blends.
2. Club soccer: source real xG, lineups, injuries/suspensions, manager changes,
   and market close/open movement. The local shot-volume proxies are near their
   easy-gain limit.
3. Golf: expand the tuning harness to a small coordinate descent or Bayesian
   search only after adding stronger inputs such as strokes-gained categories,
   tee-wave/weather, and course archetype features.
4. World Cup: expand historical market coverage and lineup/absence labels before
   promoting V4 report-only layers.
