# Ready-to-run Claude Review Request — Horse Racing V1

Copy the block below into Claude Code/Cowork with read access to this repository.

---

Follow every instruction in
`docs/CLAUDE_HORSE_RACING_REVIEW_PROMPT.md` as the governing review rubric.
Do not modify any files.

Review mode:

`DIFF`

Repository:

`/Users/lucky/AI Models/Soccer Prediction`

Review scope:

Review the implemented provider-neutral horse-racing V1 and only its integration
changes. The in-scope files are:

```text
.gitignore
app/engines/__init__.py
app/engines/horse_racing.py
app/provenance.py
app/web/app.js
docs/HORSE_RACING_ENGINE_PLAN.md
horse_racing/README.md
horse_racing/__init__.py
horse_racing/__main__.py
horse_racing/config.py
horse_racing/schema.py
horse_racing/features.py
horse_racing/model.py
horse_racing/edge.py
horse_racing/engine.py
horse_racing/validate.py
horse_racing/data/races.csv
horse_racing/data/runners.csv
horse_racing/data/results.csv
horse_racing/data/odds.csv
test_horse_racing.py
```

The worktree contains unrelated pre-existing user changes in World Cup, tennis,
club-soccer data, the server, and other files. Do not review or attribute those
changes to this implementation. Inspect the untracked in-scope files directly;
ordinary `git diff` does not display untracked content.

Implementation claims to challenge:

1. Canonical inputs are restricted to GB/IE flat racing and fail closed outside
   the declared scope.
2. Runner/race/result/odds records use stable IDs and source timestamps.
3. Features for a race use only declarations and result versions available by
   its 15-minute pre-off cutoff.
4. Full result corrections become visible at their update time and trigger a
   chronological history-state rebuild without changing earlier predictions.
5. The pure model is a regularized race-level conditional logit using
   race-relative official rating, weight, draw, age, recency, EB-shrunk dynamic
   horse form/win history, trainer/jockey effects, and surface/distance/course
   history.
6. Temperature calibration preserves race probability normalization.
7. Walk-forward validation is chronological and nested for fitting/calibration,
   with uniform, official-rating, and cutoff-market baselines plus race-level
   paired bootstrap intervals.
8. Odds after cutoff, stale/incoherent/incomplete boards, non-positive available
   size, and mismatched field versions cannot generate recommendations.
9. App edge output is analytical only with zero stakes. Shared-ledger recording
   is disabled because the current ledger cannot settle dead heats exactly.
10. Model artifacts contain source checksums, code hash, feature schema, scope,
    calibration details, environment versions, training period/counts, and the
    current Git commit.

Known intentional limitations for this milestone:

- No live or historical racing provider has been selected or integrated.
- No real racing dataset or fitted production model is committed.
- The production validation gate is therefore UNPROVEN; synthetic tests cannot
  establish predictive or economic performance.
- Input CSVs are canonical provider outputs, not yet an immutable raw-payload
  archive or cross-provider identity service.
- V1 supports pre-race win probabilities only: no jumps, in-play, place,
  each-way, laying, forecasts, or complete-order simulation.
- Shared bankroll recording and automated racing settlement are disabled.
- Race metadata is canonical one-row-per-race and post-cutoff metadata is
  rejected; versioned race-metadata correction replay is not implemented.

These limitations may be classified as `DEFERRED BY SCOPE` only where the plan's
current implementation milestone permits it. Flag any limitation that makes the
implemented predictor internally invalid even as an offline provider-neutral V1.

Commands already run:

```text
python3 test_horse_racing.py
PASS: horse_racing: PASS

python3 test_provenance.py
PASS: 23 passed, 0 failed

python3 test_engines_contract.py
PASS: 23 pass, 4 data-dependent skips, 0 fail
horse_racing info PASS; prediction/edge skipped because committed templates are empty

python3 -m compileall -q horse_racing app/engines/horse_racing.py test_horse_racing.py
PASS

node --check app/web/app.js
PASS

git diff --check
PASS

python3 run_checks.py
31/32 passed. The sole failure was pre-existing and outside this scope:
test_m3 reported World Cup market blend log-loss did not beat market-only.
The horse-racing test passed. Do not treat the unrelated World Cup failure as a
horse-racing finding unless you can demonstrate a causal link from an in-scope
change.
```

Required extra work during review:

- Run `python3 test_horse_racing.py` yourself.
- Run `python3 test_engines_contract.py` and `node --check app/web/app.js`.
- Inspect every in-scope source and test file, not only tracked diffs.
- Trace at least one historical prediction through race cutoff, declaration
  snapshot, result-version event processing, feature calculation, model
  normalization, odds selection, and edge eligibility.
- Attempt the falsification cases in the governing prompt, especially partial
  corrections, out-of-order publication, post-cutoff metadata, mixed-source
  odds, independently calibrated runners, extreme utilities, field-version
  mismatches, and incomplete boards.
- Assess whether the synthetic tests can pass while a real-data production path
  remains wrong.
- Verify that app UI race mode does not leave stale hidden values or break other
  engines when switching between them.
- Check whether artifact provenance is sufficient to reproduce a model fitted
  from an uncommitted/dirty worktree.

Return the exact required verdict and report structure from
`docs/CLAUDE_HORSE_RACING_REVIEW_PROMPT.md`. A clean code review alone must not
mark predictive or economic validity as supported.

---

End of review request.
