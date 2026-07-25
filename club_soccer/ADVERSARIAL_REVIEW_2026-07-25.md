# Club Soccer — adversarial review, 2026-07-25

19,115 lines across 49 modules. Every claim below was verified against the code
or by running it.

---

## 1. Bug: neutral player adjustments move the price — **FIXED 2026-07-25**

> Adjustments now enter at the component lambdas (`_adj_multipliers` +
> `component_matrices(mult_h, mult_a)`) instead of being applied to the blended
> matrix. Identity is now exactly identity (0.000e+00 matrix delta), the three
> corrections compose exactly, and the per-competition rho survives.
> The matrix-level appliers remain as a documented legacy path and now take a
> `rho` argument. Regression test:
> `test_club_soccer.py::test_adjustments_enter_at_lambda_level`.
>
> Production impact, measured over 150 upcoming fixtures with a typical
> absence profile — the pre-fix code was systematically overstating the
> over/BTTS side:
>
> | market | mean Δ | p95 \|Δ\| | max \|Δ\| |
> |---|---|---|---|
> | P(home) | −0.0008 | 0.0152 | 0.0321 |
> | Over 2.5 | −0.0105 | 0.0285 | 0.0660 |
> | BTTS yes | −0.0160 | 0.0429 | 0.0517 |
>
> Original diagnosis follows.


`apply_player_adj`, `apply_context_adj` and `apply_quality_adj` (model.py
1266–1334) all do the same thing: read the marginal expected goals off the
matrix, multiply, and rebuild with `score_matrix(lam_h, lam_a)`.

The ensemble matrix is a **mixture of five Dixon-Coles matrices**. It is not
itself a DC matrix. Rebuilding from its marginals throws the mixture away and
replaces it with a single Poisson that only matches the means. Measured, with
an identity adjustment that should be an exact no-op:

```
                base      adj     delta
home          0.4678   0.4704   +0.0026
away          0.2681   0.2660   -0.0021
over25        0.5296   0.5327   +0.0031
btts_yes      0.5617   0.5658   +0.0041
```

41 basis points on BTTS from multiplying by 1.0. Every fixture with absence
data is priced off a structurally different distribution than one without, and
the discontinuity lands hardest on totals and BTTS — the markets where the
mixture's extra dispersion was the whole point.

Second defect in the same three functions: they call `score_matrix` without a
`rho` argument, so they silently substitute the global `DC_RHO = -0.08` for the
per-competition rho that `predict` computed on line 1462. Currently masked only
because `comp_adj_active` is `False` (see §3).

**Fix.** Apply multipliers to the component lambdas, not the blended matrix —
scale inside `component_matrices` and re-blend. Adjustment then commutes with
the ensemble and identity is exactly identity. If that's too invasive, at
minimum collapse the three functions into one (`_rescale(M, mult_h, mult_a,
rho)`) and pass rho through; that fixes the rho bug and deletes ~50 lines, but
leaves the mixture-collapse.

---

## 2. The module is mostly dormant

The following are computed, stored, documented, tested — and never used in
production:

| Thing | State | Cost |
|---|---|---|
| `comp_adj` (58 comps: shrunk HFA + DC rho) | `comp_adj_active: False`, **hardcoded at model.py:705** | fit every run |
| `league_env` / `league_hfa` (46 comps) | `league_adjustments_active: False` | fit every run |
| `comp_strength.json` | `active: false` | `fit_comp_strength`, 70 lines |
| `player_quality.json` | `active: false` | player_quality.py, 406 lines |
| `context_coef_club.json` | `active: false` | context.py, 590 lines |
| `ensemble_weights.json` | `active: false` | validate.py tuner, ~230 lines |
| variance inflation | `VARIANCE_INFLATION_DEFAULT = False` | ~45 lines |
| promo prior / season boundary / Elo decay | all `active: false` | 3 tuners, 263 lines |
| `league_strength.py` artifact | `active: false` **and zero callers** | 484 lines |
| market blend | gated off at app level | — |
| staking | evidence gate closed (56 ledger rows vs 1000 bar) | — |

What actually ships: a five-component ensemble on global HFA, global rho, and
league-seeded Elo. That's maybe 900 lines of model.py. The other ~18,000 lines
are scaffolding around a prediction the module then refuses to bet.

The "measure it, park it, promote by commit" discipline is the right instinct
and I'm not arguing against it. But nothing is ever *deleted*, so the parked
inventory only grows, and each parked item still has to be read, imported,
kept compiling, and kept out of the live path by a flag. `comp_adj_active` is
literally a hardcoded `False` in a dict literal — it is not a switch, it is a
comment with syntax.

**Fix.** Give parked work an expiry. If an experiment hasn't promoted in 60
days, delete the code and keep the evidence JSON plus the git SHA in a one-line
register. `git log` is the archive; the working tree is not. That alone removes
~1,500 lines.

---

## 3. Dead files — 1,458 lines with zero importers

Verified by scanning every `.py`, `.sh` and `.md` in the repo:

- `cross_league_eval.py` (356) — referenced only in `PLAN_uefa_league_expansion.md`
- `league_strength.py` (484) — test-only (`test_league_strength.py`)
- `fetch_fdorg.py` (294) — referenced only in a 3-month-old plan doc
- `corners_model.py` (324) — **no reference anywhere**, including tests
- `build_bets_reconstructed` in decision_time_backtest.py (85) — marked
  DEPRECATED; its only test asserts that the string "DEPRECATED" is in its
  source. A test that greps a comment isn't a test.

`seed_footballdata.py`, `seed_openfootball.py` and `seed_real.py` have test
coverage but no production caller — either they're one-shot backfills (say so
in the filename or move them to `scripts/`) or they're orphaned.

---

## 4. Four name normalisers and a repair layer

Seven `_norm`-shaped functions across the module; four of them normalise team
names:

```
names.simplify          identities._norm
club_registry._norm     club_identity._norm / _core
```

`club_identity.py`'s own docstring diagnoses the problem exactly: *"names.py
defines a canon for this, but it is applied only on some ingest paths."*
Confirmed:

```
fetch.py               names=no   club_identity=yes
fetch_fdcouk.py        names=yes  club_identity=no
seed_real.py           names=yes  club_identity=no
seed_openfootball.py   names=yes  club_identity=no
seed_fdcouk_leagues.py names=no   club_identity=yes
seed_footballdata.py   names=no   club_identity=no
fetch_fdorg.py         names=no   club_identity=no
```

So there are two competing canon systems, two ingest paths using neither, and
then **957 lines of `club_identity.py`** — fuzzy matching, country guards,
head-to-head evidence, reserve-team detection, plus 37 lines of
`DOMESTIC_MANUAL_MERGES` and 13 of `DOMESTIC_MERGE_BLOCKLIST` — repairing the
damage downstream. Plus `identity_review.py` (524) to review the repairs.

This is the module's worst structural problem, and it's the *opposite* of too
complex: the complexity is real, it's just in the wrong place. Cleaning
identities at ingest is a small, well-defined job. Reconciling them afterward
from statistical evidence is unbounded, and the manual merge/blocklist tables
are the proof — they only grow.

**Fix.** One `canonicalise(raw_name, source, country_hint) -> canonical_id`
called by *every* writer to fixtures.csv, backed by the openfootball registry
that `club_registry.py` already downloads. Make it the only way to write a
team name; enforce with a schema check in `health.run_checks`. Then
`club_identity.py` shrinks to the reconciliation of genuinely ambiguous cases
and `identity_review.py` becomes a report rather than a workflow.

---

## 5. Five overlapping evaluation harnesses

`validate.walk_forward`, `backtest_market`, `decision_time_backtest`,
`decision_ledger`, `cross_league_eval`.

Their own docstrings adjudicate this:

- `backtest_market.py` selects and executes at the close — `evidence_gate.py`
  is explicitly written to refuse its output. It exists to be rejected.
- `decision_ledger.py` says of `decision_time_backtest`: *"You cannot fix
  reconstruction; you have to stop reconstructing."* Then `season.py:792`
  still runs it every day, and it writes the artifact the gate reads.

Two of the five are declared invalid by the code that replaced them, one is
dead, and the two that matter (`walk_forward`, `decision_ledger`) are fine.

**Fix.** Keep `walk_forward` (model quality) and `decision_ledger` (staking
evidence). `decision_time_backtest` keeps only the code that computes the gate
artifact *from the frozen ledger* — that's maybe 150 of its 616 lines. Delete
`backtest_market.py` or demote it to a clearly-labelled diagnostic that cannot
write `data/backtest_market.json`.

---

## 6. The gate's own artifact contradicts the gate

`data/backtest_market.json` carries this note:

> *"under the current global-AND gate that means staking can never open on ANY
> market until a closing-totals source is added OR the gate is made per-market.
> 1X2 alone cannot open it."*

The gate **is** per-market — `evidence_gate.market_staking_allowed()` and
`market_league_staking_allowed()` both exist. The note is stale and describes a
permanent blocker that was fixed. Anyone reading the artifact concludes the
system is bricked when it isn't; the real blocker is just volume (56 settled
decisions against a 1,000 bar).

Stale prose is worse than no prose when the prose is about whether the thing
can ever work. Regenerate the note from the code, or delete it.

---

## 7. Too simple: the ensemble is three copies of one signal

Production weights: `goals 0.20, elo 0.40, xg 0.20, xgf 0.20, xpress 0.00`.

`_lambdas_xg` and `_lambdas_xgf` are **the same function** — `xgf` is
`_lambdas_xg(form=True)`. `_lambdas_xpress` is a byte-for-byte copy of
`_lambdas_xg` with different dict keys. All three read shot data; 49% of that
"xG" is `shots_on_target × global_conversion` (`proxy_xg_coverage = 0.487`),
against 21% real xG. So 40% of the ensemble sits on two near-collinear
rescalings of one shot count, and the third copy is weighted zero.

The tuner's own numbers say the search found nothing: `chosen_brier 0.612352`
vs `previous_brier 0.612500`. That is a 1.5e-4 improvement — noise on ~17k
matches — and it's why the artifact is deactivated.

**This is where the module is under-complicated.** Adding a fifth
transformation of the same shot count is not diversification. The signals a
club model at this scale is actually missing:

1. **Rest and travel as model inputs, not a gated post-hoc multiplier.**
   `context.py` computes them and `context_coef` is `active: false`. Fit
   rest-days and travel distance *inside* the Poisson as covariates. They're
   causal, weakly correlated with Elo, and cheap.
2. **Home/away split attack and defence.** Currently one attack term plus a
   competition-level HFA. Team-specific home effects are large and real
   (altitude, pitch size, crowd) and you have 56,745 matches — enough for a
   shrunk per-club home term.
3. **Bivariate Poisson or a negative-binomial margin.** Dixon-Coles patches
   four cells of the 0–1 corner. It does not fix the dispersion in the tail,
   which is exactly where OU2.5 and BTTS live — the two markets whose
   calibration the module keeps flagging.
4. **A proper opponent-adjusted xG.** Raw SoT×conversion doesn't know that 12
   shots against Bayern mean something different from 12 against Kaiserslautern.

Any one of those is worth more than the xg/xgf/xpress triplet combined.

---

## 8. Smaller, concrete

**Redundant predictions.** `rows_from_odds` groups by
`(date, comp, home, away, market, line)` and calls `M.predict_match` inside the
loop. Three markets in `MARKETS` → three full predictions per fixture, each
re-running the context lookup. Hoist the prediction to a per-fixture cache.

**Tests write to production data.** `test_club_soccer.py:605` does
`S.CARD.unlink()` on the real `club_soccer/data/card.md`, then runs
`season.py --no-network` to regenerate it. Running the test suite destroys the
live card. Point the tests at a tmpdir.

**"Most likely to land" bypasses the evidence gate** (season.py:378). The
comment is candid about it. But the entire gate architecture exists so
unproven picks aren't presented as backable, and the *lead section of the card*
is a ranked table of picks with odds and a disclaimer. If the gate is right,
this section shouldn't lead. If it's fine to show them, the gate is theatre.
Pick one.

**`_poisson_pmf` recomputes `math.factorial(0..10)` on every call** — once per
component per prediction. Precompute the array at module level.

**`apply_evidence_gate` is called twice** — inside `rows_from_odds` (edge.py:399)
and again in the adapter (`app/engines/club_soccer.py`). Defensible as
defence-in-depth, and the comments explain why, but it means "who is
responsible for zeroing stakes" has no single answer.

---

## 9. Repository hygiene

- **`.git` is 2.9 GB.** 712 individual `bsd_enrichment/event_*.json` files (69 MB)
  are tracked, while `fixtures.csv` — the actual artifact — is gitignored
  (`.gitignore:200`). Exactly backwards. Tar the enrichment cache or move it
  out of the tree.
- **13 `fixtures.csv.bak.*` files, ~100 MB**, named `pre_wave1`…`pre_wave4`,
  `pre_identity`, `pre_nonuefa`, `pre_zalgiris_revert`. That's a hand-rolled
  VCS inside a git repo. Gitignored, so they're pure local clutter with no
  recovery guarantee.
- `club_soccer/data` is 420 MB, of which 109 MB is `bsd_cache` and 56 MB
  `weather_cache` with no eviction policy.
- **8 `codex_club_soccer_*.md` prompt files at repo root** (~90 KB) plus
  `PLAN_uefa_league_expansion.md` (47 KB), two `SCOPE_*.md`, and
  `NONUEFA_INGEST.md` inside the module. Review scaffolding checked in beside
  the code. Move to `docs/archive/`.
- **25 `test_club_soccer*`-adjacent files at repo root** rather than in
  `tests/`. `tests/` exists and holds only `golden/`.

---

## Priority

1. Fix `apply_*_adj` — it is silently mispricing every fixture with absence data (§1).
2. Delete the four dead modules and the deprecated reconstruction path (§3, §5).
3. Unify canonicalisation at ingest (§4).
4. Fix the stale gate note (§6) — cheapest item here, and it's actively misleading.
5. Set an expiry policy on parked experiments (§2).
6. Then, and only then, spend modelling effort on §7 — not on a sixth shot transform.
