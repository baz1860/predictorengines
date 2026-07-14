# Claude Adversarial Re-review — Horse Racing V1

Copy the block below into Claude Code/Cowork with read access to this repository.

---

Follow every instruction in
`docs/CLAUDE_HORSE_RACING_REVIEW_PROMPT.md` as the governing rubric. Do not
modify any files.

Review mode: `DIFF / REMEDIATION RE-REVIEW`

Repository: `/Users/lucky/AI Models/Soccer Prediction`

The previous review returned `REVISE` with six findings. Adversarially verify
the corrections below; do not assume a fix is correct because a regression
test exists. Inspect all files listed in
`docs/CLAUDE_HORSE_RACING_DIFF_REVIEW_REQUEST.md`, including untracked files.
The worktree also contains unrelated pre-existing changes; keep the same scope
boundaries as the original review.

## Claimed remediation

1. **Suspended-board fallback (P1):** `latest_odds_snapshot` now selects the
   source and latest quote/state for every declared runner before checking
   market status. If any latest runner state is not open/active, the board
   fails closed instead of falling back to an older open quote.
2. **Self-result leakage (P1):** result-version construction now rejects a
   complete result version whose availability timestamp is at or before its
   own race prediction cutoff, and also rejects timestamps before scheduled
   off. This guard is in the executed `build_feature_frame` path used by fit.
3. **Partial-board strict JSON (P2):** `cmd_edge` sanitizes every optional
   numeric output, converting all non-finite values to JSON `null`; the board
   remains incomplete and all recommendations remain false.
4. **Non-monotone explicit cutoffs (P2):** history replay now processes races
   in decision-cutoff order, with scheduled-off and race ID only as stable
   tie-breakers, so the monotone event pointer cannot carry later information
   into an earlier explicit cutoff.
5. **Missing liquidity (P3):** edge eligibility requires every declared
   runner to have populated, finite, strictly positive `available_size`.
6. **Dirty-worktree provenance (P3):** fitted artifacts record both Git commit
   and whether the `horse_racing` path is dirty/untracked. Artifact loading
   also recomputes and verifies `code_sha256`, rejecting code/artifact drift.

## Required falsification work

- Re-run each original executable counterexample, not merely the test suite.
- For suspension, test full open quotes followed by suspended rows before the
  cutoff and confirm no older quote can become executable.
- For self-result leakage, test both initial publication and corrected versions
  timestamped at/before the race's own cutoff; confirm feature construction and
  fit fail closed.
- For replay, test multiple deliberately non-monotone explicit cutoffs and
  compare each feature row with independently truncated ground-truth bundles.
- For partial boards, invoke `edge` through `run_inprocess`; require strict JSON,
  null optional prices/metrics, and zero recommendations.
- For liquidity, try blank, NaN, zero, negative, and infinite sizes.
- For provenance, test modified and untracked `horse_racing` files, a clean
  repository, and artifact loading after changing executable source code.
- Look for regressions caused by the fixes, especially source selection,
  correction replay, same-timestamp ordering, prediction of future races,
  artifact portability, and strict JSON behavior.
- Reassess all ten original implementation claims, not only the six findings.
- Keep predictive and economic validity explicitly `UNSUPPORTED`: there is
  still no provider integration, real racing dataset, or production baseline.

## Commands already run after remediation

```text
python3 test_horse_racing.py
horse_racing: PASS

python3 test_engines_contract.py
23 pass, 4 data-dependent skips, 0 fail

python3 test_provenance.py
23 passed, 0 failed

python3 test_security.py
20 passed, 0 failed

python3 -m compileall -q horse_racing app/engines/horse_racing.py test_horse_racing.py
PASS

node --check app/web/app.js
PASS

git diff --check
PASS
```

Return the exact verdict/report structure required by the governing rubric.
State whether each old finding is `FIXED`, `PARTIALLY FIXED`, or `OPEN`, with
file/line evidence and reproducible commands. Report any newly introduced
finding normally by severity. Approval applies only to the offline,
provider-neutral implementation milestone and must not imply demonstrated
real-world predictive accuracy or profitability.

---

End of re-review request.
