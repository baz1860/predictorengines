# Refactor Plan

A phased restructuring of the multi-sport prediction/betting system. Each phase is
an independently shippable PR. Phases run strictly in order 0 → 5; the early phases
de-risk the later ones.

Status legend: ⬜ not started · 🟡 in progress · ✅ merged

| Phase | Title | PR | Status |
|-------|-------|----|--------|
| 0 | Safety net | `refactor/phase-0-safety-net` (#10) | ✅ |
| 1 | Cleanup & archive | `refactor/phase-1-cleanup` | 🟡 |
| 2 | Core + contracts extraction | — | ⬜ |
| 3 | Package the worldcup engine | — | ⬜ |
| 4 | Kill the subprocess hack | — | ⬜ |
| 5 | Tests & layer rename | — | ⬜ |

---

## Why

The project grew fast and organically. The symptoms:

1. **The repo root is a ~62-file flat dump.** The World Cup soccer engine
   (`edge.py`, `dixoncoles.py`, `simulate.py`, `squads.py`, `confederation_adj.py`,
   `predictor.py`, `context.py`, …) lives loose at the root, intermixed with 21 test
   files, 9 one-off backtest/replay scripts, ~12 data CSVs, and ~12 `V*_PLAN/NOTES.md`
   docs.

2. **Engine naming collisions, worked around by a subprocess hack.** Every sport
   defines generic `model.py` / `edge.py` / `simulate.py`. They cannot coexist on one
   import path, so `app/engines/runners/*` launch each engine as a subprocess with
   `PYTHONPATH=<sport dir>` so a bare `import edge` resolves to the right one. This is
   the central architectural smell — fragile, hard to test, slow.

3. **Inconsistent packaging.** `golf/`, `cfb/`, `club_soccer/` are proper packages.
   The *soccer* engine — the original core — is the one that is NOT packaged. The
   `worldcup` adapter imports root modules directly (same path), while the others use
   subprocess runners. That asymmetry is what the refactor erases.

4. **Misleadingly-named layers, not version iterations.** `wc_v4`, `v5`, `v6` are not
   successive versions of one engine — they are distinct functional layers:
   - **canonical engine** = the registered adapter suite in `app/engines/` (the
     V3/contract layer) with promoted artifacts, gated by `validate_all.py`. Default
     blend remains `v3_blend`.
   - **wc_v4** = World Cup research/report substrate (not promoted).
   - **v5** = governance/advisory layer (registry, drift, review, portfolio, scenario).
   - **v6** = operations/product layer (health, backup, daily-run, release status).

5. **No project scaffolding.** No `pyproject.toml`, no pytest config, no `tests/`
   directory. Tests run via a bespoke `run_checks.py` with a hardcoded `ORDERED` list
   that includes milestone tests (`test_m2`…`test_m7`).

6. **Committed/lingering cruft.** `edge.py.bak`, `_bt_tmp.py`, `.DS_Store`,
   `.launch_error.log`, `~$Betting Tracker.xlsx`, a 193 KB `dashboard_preview.html`,
   a private `Betting Tracker.xlsx` (mode 600), `odds_sample.csv`. Plus `api_keys.py`
   at the root — a secrets-handling concern.

## Target structure

```
pyproject.toml            # packaging + pytest/ruff config
src/predictors/
  core/                   # shared primitives: bankroll, market_blend, clv,
                          #   calibration, context, provenance
  contracts/              # the registry + engine contract (today's
                          #   app/engines/base.py + contracts.py) — canonical interface
  engines/
    worldcup/             # the root soccer engine, finally packaged
    club_soccer/
    cfb/
    golf/
  research/               # was wc_v4 — WC research/report substrate
  governance/             # was v5 — registry, drift, review, portfolio, scenario
  operations/             # was v6 — health, backup, daily-run, release
app/                      # web server + web UI; depends on contracts + engines only
tests/                    # mirrors src/, pytest-discovered
data/                     # input data (gitignored where appropriate)
docs/archive/             # old V*_PLAN/NOTES, session handoffs
scripts/backtests/        # the one-off backtest/replay scripts
```

Because the engines become real subpackages
(`predictors.engines.golf.model`), the name collisions vanish and the
subprocess + `PYTHONPATH` hack can be deleted — runners become direct imports.

The engine **registry** (`app/engines/__init__.py`) is the one thing already right.
It stays the single wiring point throughout; everything routes through it.

---

## Phases

### Phase 0 — Safety net  🟡
Prerequisite for everything else. No structural change.
- Add `pyproject.toml` (packaging metadata + pytest/ruff config).
- Make the test suite discoverable and record a green baseline.
- Capture golden outputs (sample `edge_report.csv`, predictions) so later phases can
  diff behavior rather than guess.
- Add this plan document.

**Acceptance:** baseline recorded; `pyproject.toml` present; golden snapshot captured;
no behavior change.

### Phase 1 — Cleanup & archive  🟡
Low risk, high signal. Pure file moves/deletes.
- Delete cruft: tracked `dashboard_preview.html` (193 KB, orphaned) and root
  `odds_sample.csv` (unreferenced); plus local ignored junk (`.bak`, `_bt_tmp.py`,
  `.DS_Store`, lock files).
- Move the 13 `V*`/`GUI`/`SESSION`/`NOTES` planning docs → `docs/archive/`; fix the
  README links that pointed at them.

**Deferred to Phase 3:** moving the root backtest/replay scripts to
`scripts/backtests/`. They do bare `from predictor import ...` / `from edge import ...`
and import each other (`wc_backtest_history` ← `wc2022_sim_backtest`); relocating them
now would break those imports, violating Phase 1's "no import changes" rule. They move
cleanly once the worldcup engine is a package (Phase 3).

**`api_keys.py` — no change needed.** It is a *loader*, not a secret: it reads keys
from the already-gitignored `data/api_keys.json` with env-var precedence, and is
imported by 16+ modules. Verified no secret is committed (`data/api_keys.json`
untracked, no hardcoded keys). Renaming/moving it would break 16 imports for zero
security gain.

**Acceptance:** root file count sharply reduced; tests still green; no import changes.

### Phase 2 — Core + contracts extraction  ⬜
- Extract genuinely shared modules into `core/` (bankroll, market_blend, clv, context,
  provenance, calibration).
- Lift `app/engines/base.py` + `contracts.py` into `contracts/` as the canonical
  interface.
- De-duplicate copies (e.g. root `market_blend.py` vs `app/market_blend.py`).

**Acceptance:** one source of truth per shared concern; registry unchanged; tests green.

### Phase 3 — Package the worldcup engine  ⬜  (highest risk)
- Move the loose root soccer files into `engines/worldcup/`, matching the other sports.
- Gate strictly on Phase 0 golden outputs.

**Acceptance:** worldcup engine is a package; golden outputs match; registry unchanged.

### Phase 4 — Kill the subprocess hack  ⬜
- With everything packaged, rewrite `app/engines/runners/*` as direct imports.
- Delete `_subprocess.py` and the `PYTHONPATH` plumbing.

**Acceptance:** no subprocess engine launches; app faster; tests green.

### Phase 5 — Tests & layer rename  ⬜
- Consolidate tests into `tests/` mirroring `src/`.
- Replace `run_checks.py`'s manual `ORDERED` list with pytest markers
  (`-m fast` / `-m gates`).
- Rename layers to their roles with import shims: `wc_v4`→`research`, `v5`→`governance`,
  `v6`→`operations`.

**Acceptance:** `pytest` discovers everything; markers select fast vs gate suites;
layer names reflect roles.

---

## Conventions for every phase PR

- Branch name: `refactor/phase-N-<slug>`.
- Run the baseline check (`python run_checks.py`) before and after; the PR body records
  both results.
- No phase changes behavior; structural-only. Any behavior change is called out
  explicitly and justified.
- Keep the registry (`app/engines/__init__.py`) as the single wiring point.
