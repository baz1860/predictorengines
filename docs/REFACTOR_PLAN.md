# Refactor Plan

A phased restructuring of the multi-sport prediction/betting system. Each phase is
an independently shippable PR. Phases run strictly in order 0 → 5; the early phases
de-risk the later ones.

Status legend: ⬜ not started · 🟡 in progress · ✅ merged

| Phase | Title | PR | Status |
|-------|-------|----|--------|
| 0 | Safety net | `refactor/phase-0-safety-net` (#10) | ✅ |
| 1 | Cleanup & archive | `refactor/phase-1-cleanup` (#11) | ✅ |
| 2 | Core extraction | `refactor/phase-2-core-contracts` (#12) | ✅ |
| 2b | Contracts extraction | `refactor/phase-2b-contracts` (#13) | ✅ |
| 3a | Package worldcup core (+shims) | `refactor/phase-3-worldcup` (#14) | ✅ |
| 3b | Migrate importers, drop shims | `refactor/phase-3b-migrate-importers` (#15) | ✅ |
| 3c | Move backtest/analysis scripts | `refactor/phase-3c-tooling` (#16) | ✅ |
| 3c-2 | Move WC validate + WC scripts | `refactor/phase-3c2-engine-tooling` | 🟡 |
| 4a | In-process club_soccer engine | `refactor/phase-4-kill-subprocess` (#18) | ✅ |
| 4b | In-process cfb engine | `refactor/phase-4b-cfb` (#19) | ✅ |
| 4c | In-process golf engine | `refactor/phase-4c-golf` | 🟡 |
| 4d | Delete _subprocess + rework sec tests | — | ⬜ |
| 5 | Tests & layer rename | — | ⬜ |

> **Phase 4 was split per-sport** after investigation: the subprocess is **not only**
> a collision workaround — it's a tested **security boundary** (curated env, command
> allowlist, secret-redacted errors). Decision (with the user): **kill it, preserve
> redaction** — engines go in-process (consistent with worldcup, which already runs
> in-process), redaction is replicated at the adapter boundary via the new
> `app/engines/_inproc.py`; the curated-env isolation is dropped (engines are
> first-party). Relativizing a sport's imports breaks its subprocess runner, so each
> sport flips atomically: 4a club_soccer, 4b cfb, 4c golf, 4d deletes `_subprocess.py`
> + the runners and reworks the now-obsolete `safe_runner_env` security tests.

> **Phase 3 was split into 3a/3b/3c** after investigation showed it's far bigger than a
> pure move: ~15-20 tightly-coupled engine modules with **import cycles**
> (`predictor ↔ confederation_adj ↔ dixoncoles`, `edge ↔ market_blend`), all resolving
> `data/` paths via `__file__`, plus ~25 external importers (the adapter, all of `wc_v4/`,
> backtests, `test_m2`–`m7`, `core/clv`).
>
> **Data decision (chosen): keep the shared root `data/`.** Moved modules anchor on
> `Path(__file__).resolve().parents[2]` to reach it, so the tripwire-pinned files stay
> put and `app/`/`core/`/tests are unaffected. (The other engines own private `data/`
> dirs; worldcup stays asymmetric by design for now.)

> **Phase 2 was split** after investigation. The original Phase 2 bundled three
> items; two turned out riskier/different than assumed:
> - **`market_blend` "de-dup" is not a structural move.** Root `market_blend.py` (WC
>   1X2 blend) and `app/market_blend.py` (generalized V3 blend) are *different
>   implementations*. Unifying them changes behavior — handled separately as an
>   explicit behavior-change task with golden re-baselining, not in a structural phase.
> - **Contracts extraction is high-blast-radius** (the registry + `contracts.py` are
>   imported from ~14 sites across tests, `app/`, **and `wc_v4/`+`v5/` reaching up into
>   `app.engines`**). Pulled out into its own **Phase 2b** so the core move stays small.

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

### Phase 2 — Core extraction  🟡
Create the `core/` package (root-level for now; relocates under `src/predictors/core/`
in Phase 3) and move the sport-agnostic betting infrastructure into it:
- `bankroll.py` → `core/bankroll.py` (ledger bankroll management)
- `clv.py` → `core/clv.py` (closing-line-value)
- `clv_suite.py` → `core/clv_suite.py` (suite-wide CLV reporting)

All 14 import sites migrated to `from core import …` (textual grep confirmed no
dynamic/importlib usage, so the migration is provably complete — a final straggler grep
returns empty).

**Excluded (belong to the worldcup engine, → Phase 3), despite being root modules:**
- `context.py` — imports the WC `predictor`; it's WC-specific, not shared.
- `calibrate.py` — model calibration; golf already has its own copy.

**Known smell, deferred to Phase 3:** `core/clv.py` lazily does `from edge import …`
(WC closing-odds fetch). It works because the repo root stays on `sys.path`, but `core/`
ideally should not depend on a specific engine. When `edge` moves into
`engines/worldcup/` (Phase 3) this cross-package edge is formalized or inverted.

**Acceptance:** `core/` exists with the 3 modules; zero straggler imports; registry
untouched; tests green; golden tripwire unchanged.

### Phase 2b — Contracts extraction  🟡
Lifted `app/engines/base.py` → `contracts/registry.py` (registry + `EngineAdapter`) and
`app/engines/contracts.py` → `contracts/protocol.py` (fixture/market identity, edge
normalisation, JSON checks). `contracts/__init__.py` re-exports the public API, so every
caller now uses `from contracts import …`. Both modules had zero non-stdlib imports —
genuinely self-contained, so the lift is clean. `app/engines/_subprocess.py` stays put
(it's the subprocess hack removed in Phase 4).

Sites updated: the protocol-vocabulary imports in `wc_v4/feature_store.py`,
`v5/registry.py`, `core/clv_suite.py`, `test_model_audit.py`, `test_engines_contract.py`,
and the relative `.base`/`.contracts` imports inside the four adapters + `_subprocess.py`.

**Gotcha handled — adapter registration is an import side-effect.** Three callers
(`app/server.py`, `daily_summary.py`, `test_engines_contract.py`) need the *populated*
registry, which only exists once `app/engines/__init__.py` runs its `register(...)` calls.
So their `from app.engines import registry` was **kept** (not redirected to `contracts`):
the registry singleton lives in `contracts`, but importing `app.engines` is what fills it.
`app/engines/__init__.py` now does `from contracts import registry` then registers the
four adapters — staying the single wiring point.

**Gotcha caught by the baseline — `__file__`-relative paths.** `enrich_template_result`
computed `repo_root = Path(__file__).resolve().parents[2]`, valid at the old
`app/engines/` depth but off-by-one at `contracts/` (one level shallower). The golden
baseline caught it (`test_model_audit` failed: row count `None`); fixed to `parents[1]`.
**Action item for Phase 3:** every file moved between directory depths must have its
`__file__`/`parents[N]` path math re-checked — grep moved files for `__file__` and
`parents[` before trusting a green import.

**Separate task (not a phase): unify `market_blend`.** Reconcile the root WC 1X2 blend
with `app/market_blend.py` into one implementation. This *changes behavior*, so it runs
outside the structural phases with a deliberate golden re-baseline and its own
validation.

### Phase 3 — Package the worldcup engine  🟡  (highest risk, split 3a/3b/3c)

**3a — Package the core cluster (this PR).** Create `engines/worldcup/` and move the 9
mutually-dependent core modules: `predictor`, `dixoncoles`, `confederation_adj`,
`simulate`, `squads`, `context`, `calibrate`, `market_blend`, `edge`. Because of the
import cycles they move as one unit. Inside the package, sibling imports became relative
(`from .predictor import …`); `data/` anchors were repointed to
`Path(__file__).resolve().parents[2]` (root). **Transparent root shims** were left for
each module (`import sys; from engines.worldcup import X as _m; sys.modules[__name__]=_m`)
so the ~25 external importers — adapter, `wc_v4/`, backtests, `test_m2`–`m7`, `core/clv`
— keep working unchanged. Verified: shim identity is exact (incl. private names), data
loads from root, all 4 engines register.

**3b — Migrate importers, drop shims (done).** Rewrote 31 in-process importers
(tests, backtests, WC tooling, `core/clv`, the adapter, all of `wc_v4/`) to
`from engines.worldcup import …` via a line-anchored migration script, then deleted the
9 root shims. Straggler grep is empty. The `core/clv → edge` coupling is now *explicit*
(`from engines.worldcup.edge import fetch_api_odds, …`) — note this is the odds-API
plumbing, not WC model logic; relocating it into `core/` to invert the dependency is a
worthwhile follow-up, left out of this structural phase.

**Collision trap — two sport-test gotchas:** bare `import edge`/`import predictor` in a
root-run test can mean a *sport* module via `sys.path` insertion, so import-resolution had
to be traced, not assumed.
- `test_cfb_blend.py` inserts `cfb/` at `sys.path[0]` → its `predictor`/`validate` are CFB.
  **Excluded** up front.
- `test_club_soccer.py` looked root-first (it inserts `CLUB` then `ROOT`), so it was
  rewritten — **wrong**. Its `ROOT` re-insert is guarded by `if str(ROOT) not in sys.path`,
  and `ROOT` (the script's own dir) is *already* on `sys.path` at startup, so the guard
  skips it and `CLUB` stays first. Its `import edge` is the *club_soccer* edge (whose
  `devig` returns an `ndarray`; worldcup's returns a tuple). The golden baseline caught it
  (`AttributeError: 'tuple' object has no attribute 'sum'`); reverted that one line.

**Lesson for 3c / future moves:** never infer import resolution from `sys.path.insert`
order alone — account for the interpreter putting the script's own directory on `sys.path`
first, and for `not in sys.path` guards. When unsure, run the suite and read the failure.

**3c — Move standalone scripts (done).** Relocated the 15 standalone WC scripts into
`scripts/backtests/` (9 backtest/replay scripts — the Phase-1 deferral) and
`scripts/analysis/` (6 calibration/build scripts: `draw_calibration`, `draw_lopsided`,
`rho_sweep`, `totals_calibration_check`, `build_annexc`, `build_squads_2026`). Each now
prepends a repo-root `sys.path` bootstrap (so `from engines.worldcup import …` resolves
from a subdir) and anchors data on `parents[2]`. Inter-backtest sibling imports
(`wc2018`/`wc_backtest_history` ← `wc2022_sim_backtest`) still resolve because the
scripts co-locate (the script's own dir is on `sys.path` at runtime).

Verification: not covered by `run_checks` (these aren't tests), so each moved file was
import-smoked under faithful runtime (script dir + repo root on path); the analysis
scripts additionally ran their full computation during the smoke. Golden tripwire +
`run_checks` unaffected. *Gotcha caught:* the auto-injected bootstrap landed inside a
`try:` block in `backtest_betting.py` (its engine import is lazy) → `IndentationError`;
moved the bootstrap to module top.

**3c-2 — Move WC validate into the package; WC scripts to scripts/worldcup/ (done).**
Investigation refined the taxonomy:
- **`validate.py` → `engines/worldcup/validate.py`** — the genuine WC validation module
  (imported by `test_m2`, run as the worldcup gate). Sibling imports → relative
  (`from .predictor import …`), `HERE` → `parents[2]`. Rewired the three by-path callers:
  `validate_all.py` (worldcup cmd → `["-m", "engines.worldcup.validate", "--quiet",
  "--gate"]`), `update.sh` (→ `python3 -m engines.worldcup.validate …`), and `test_m2`
  (`import validate` → `from engines.worldcup import validate`). Verified the gate runs
  end-to-end via `run_checks --gates`.
- **`report.py`, `injuries.py`, `outrights.py` → `scripts/worldcup/`** — standalone WC
  operational scripts (dashboard, injury fetch, outright odds), not imported anywhere.
  `HERE` → `parents[2]`; `report` got a repo-root `sys.path` bootstrap for its lazy
  `core.clv` import; `update.sh`'s `report.py` call repointed. Verified by import-smoke.
- **`preflight.py` stays at root** — it's **suite-level** ("reports all engines"), not a
  WC module, and is invoked by-path in `test_security`. Out of scope for the WC package;
  revisit alongside the other suite orchestrators (`daily_summary`, `refresh_tracker`,
  `merge_results`, `validate_all`, `run_checks`) in Phase 5 if at all.

This completes Phase 3: the WC engine + its validation live in `engines/worldcup/`; its
standalone scripts live under `scripts/`; the repo root holds only suite-level
orchestration, tests, and config.

**Acceptance (per sub-PR):** golden outputs byte-identical; registry untouched; tests
green. `__file__`/`parents[N]` re-checked on every moved file (the Phase 2b lesson).

### Phase 4 — Kill the subprocess hack  🟡  (split per-sport)

Each sport flips to in-process atomically (relativizing its imports breaks its
subprocess runner, so the two can't coexist). Shared helper added once:
`app/engines/_inproc.py` — `run_inprocess(commands, command, params)` enforcing the
allowlist, **redacting secrets from any error**, and asserting finite JSON (the
subprocess guarantees, minus the dropped curated-env isolation).

**4a — club_soccer (done).** Relativized all 15 intra-package imports; added
`club_soccer/engine.py` (the runner's command logic, package imports); rewrote the
adapter to dispatch via `_inproc` + `club_soccer.engine.COMMANDS` and replaced its
`importlib` grader hack with `from club_soccer import edge`; deleted
`club_soccer_runner.py`. Rewired CLI callers to `-m club_soccer.X`: `validate_all.py`,
`club_soccer/update.sh`, seed-script docstrings + `seed_openfootball`'s validate
subprocess. `test_club_soccer` now imports the package + exercises the in-process path
(incl. the rejected-command guard). Verified via `run_checks --gates` (club_soccer gate
runs through the new `-m` entry).

**4b — cfb (done).** Added `cfb/__init__.py`; relativized all 26 intra-package imports;
added `cfb/engine.py` (runner logic, package imports); adapter dispatches via `_inproc` +
`cfb.engine.COMMANDS`; deleted `cfb_runner.py`. Rewired callers to `-m cfb.X`:
`validate_all.py` (cfb gate), `test_cfb_blend` (was inserting `cfb/` on `sys.path`), and
the README `cfb/predictor.py`/`cfb/validate.py` examples. Verified via `run_checks --gates`
(cfb gate through `-m cfb.validate`). cfb's `grade_open_bets` is pure pandas — untouched.

**4c — golf (done).** Added `golf/__init__.py`; relativized 19 intra-package imports;
added `golf/engine.py` (runner logic — `model`/`simulate`/`edge`/`portfolio` — package
imports); adapter dispatches via `_inproc` + `golf.engine.COMMANDS`; deleted
`golf_runner.py`. Rewired callers to `-m golf.X`: `validate_all.py`, `golf/update.sh`
(6 invocations + usage echoes), `test_golf_config`, README, and golf script docstrings.
`grade_open_bets` is pure pandas — untouched. All three runners now gone; `runners/`
holds only `__pycache__`. Verified via `run_checks --gates` (golf gate through
`-m golf.validate`).

**4d — teardown (next).** No adapter imports `_subprocess` now. Delete `_subprocess.py`
and the empty `runners/`, drop `safe_runner_env` from `app/security.py`, and rework the
`test_security` cases that exercised the subprocess (`test_safe_env`,
`test_run_engine_*`) — replacing them with `_inproc` equivalents. Keep the redaction
tests (`_inproc` reuses `redact`/`collect_secrets`).

**Acceptance (per sub-PR):** that sport runs in-process; registry + golden unchanged;
`run_checks` (and `--gates` for validate-rewires) green; security redaction preserved.

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
