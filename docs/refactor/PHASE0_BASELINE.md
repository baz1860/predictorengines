# Phase 0 — Baseline & Safety Net

Captured 2026-06-18, off `main` at the start of `refactor/phase-0-safety-net`.
This is the reference every later refactor phase must hold to.

## Test baseline (canonical runner)

`python run_checks.py` — **21/21 checks passed in 194.9s**.

```
PASS test_engines_contract (6.6s)   PASS test_clv_suite (0.2s)
PASS test_security (0.5s)           PASS test_cfb_blend (0.4s)
PASS test_bankroll (0.2s)          PASS test_provenance (0.0s)
PASS test_club_soccer (1.9s)       PASS test_model_audit (0.2s)
PASS test_m2 (1.4s)                PASS test_edge_api (0.2s)
PASS test_m3 (0.2s)                PASS test_golf_config (0.3s)
PASS test_m4 (0.2s)                PASS test_release (0.2s)
PASS test_m5 (0.7s)                PASS test_v5 (0.8s)
PASS test_m6 (1.0s)                PASS test_v6 (0.2s)
PASS test_m7 (0.2s)                PASS test_wc_v4 (179.2s)  ← dominates runtime
PASS test_market_blend (0.0s)
```

Reproduce: `python run_checks.py` (fast suites only) or `python run_checks.py --gates`
(also runs validation gates, ~1 min more).

## Why `run_checks.py` is canonical and flat `pytest` is not (yet)

`run_checks.py` runs **each suite in its own subprocess**. That isolation is currently
load-bearing: several engines define generic module names (`edge.py`, `predictor.py`,
`model.py`) in different packages, and bare imports like `from edge import ...` /
`from predictor import ...` resolve to the *wrong* sport's module when all tests share
one Python process. Demonstrated during this phase:

```
$ python -m pytest --ignore=test_wc_v4.py -q
ERROR test_v5.py  -> ImportError: cannot import name 'DC_RHO' from 'predictor'
                      (resolved to cfb/predictor.py, not the root predictor.py)
ERROR ...          -> cannot import name 'portfolio_size' from 'edge'
                      (resolved to club_soccer/edge.py)
5 errors during collection
```

Individually, the same files pass under pytest (e.g.
`pytest test_market_blend.py test_provenance.py test_edge_api.py` → 12 passed). The
collision only appears under shared-process collection. **Fixing this is the point of
the refactor** (Phases 3–4 package the engines so names no longer collide; Phase 5 then
unifies discovery under pytest). Until then, do not rely on flat `pytest`.

## Golden-output tripwire

`tests/golden/SHA256SUMS.txt` pins the promoted model artifacts and key outputs.
Structural-only phases must keep these byte-identical. Verify with:

```
bash tests/golden/verify.sh
```

Pinned files (git-tracked, promoted artifacts only — generated/gitignored reports such
as `edge_report.csv` are deliberately excluded since they don't exist in a fresh clone):
`predictions_worldcup_2026.csv`, `tournament_odds.csv`, `data/dc_params.json`,
`data/squad_ratings.csv`, `data/validation_suite.json`,
`club_soccer/data/validation_baseline.json`, `cfb/data/validation_baseline.json`,
`golf/data/calibration.json`.

If a phase is *meant* to change an output, re-baseline deliberately and note it in that
phase's PR:

```
shasum -a 256 <files...> > tests/golden/SHA256SUMS.txt
```

## Environment

- Python 3.12.7 (pyenv)
- Deps present: numpy, pandas, scipy, requests (+ fastapi/uvicorn/pydantic for app)
- pytest 9.1.0 installed during this phase (dev dependency)
