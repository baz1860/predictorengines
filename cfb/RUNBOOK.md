# CFB Week 0 Operations Runbook

**Owner:** CFB model operator  
**Frequency:** Daily during preseason; refresh again on each slate day  
**Last updated:** 2026-08-02  
**Last run:** 2026-08-02 (diagnostic rehearsal passed; betting no-go)

## Purpose

Refresh and validate the CFB engine, review provider identity and schedule drift,
and publish a reproducible weekly card without allowing stale, ambiguous, or
regression-only inputs to create a real-money recommendation.

## Prerequisites

- Run commands from the repository root.
- Use the project Python environment with the repository dependencies installed.
- Configure `CFBD_API_KEY`/`collegefootballdata` and
  `THE_ODDS_API_KEY`/`the-odds-api`; never put keys in command history or output.
- Treat `cfb/data/market_policy.json` as the release authority. A market is
  recordable only when its status is `eligible`.
- Do not manually edit generated `card.md`, `card_manifest.json`, validation
  artifacts, or the generated README metrics section.

## Procedure

### 1. Confirm frozen documentation and core inputs

```bash
python3 -m cfb.generate_docs --check
python3 preflight.py --engine cfb --require-diagnostic
```

Expected: README metrics are current and the CFB engine reports either `ready`
or `diagnostic only`; the second command exits zero.

If it fails: restore or refresh the missing required schedule, completed games,
or power parameters before continuing. Do not produce a replacement card.

### 2. Review current provider identities before widening the slate

For Week 1 (through 6 September 2026):

```bash
python3 -m cfb.identity --season 2026 --live --through 2026-09-06 --out /tmp/cfb_identity_review.csv
```

Expected: `183/183 resolved` for the provider snapshot reviewed on 2026-08-02.
The count may grow as the provider lists more events, but unresolved must remain
zero.

If it fails: inspect `/tmp/cfb_identity_review.csv`; verify each spelling against
the CFBD team ID in `cfb/data/schedule_2026.json`, add only explicitly reviewed
aliases to `cfb/data/team_aliases.json`, run the focused tests, then repeat this
step. Never enable fuzzy or prefix matching in production.

### 3. Refresh data and fit the current model

```bash
bash cfb/update.sh
```

Expected: fetch steps either succeed or explicitly retain validated cached data;
diagnostic preflight, power fit, and the frozen validation gate pass. The final
status in `cfb/data/update_status.json` is `success` and `step` is `complete`.

If it fails: read the named failing step in `update_status.json`. Network-fetch
failures may retain the last-good artifact, but power-fit, core preflight, and
validation failures are blocking. Do not bypass the non-zero exit.

### 4. Run the production rehearsal

```bash
python3 -m cfb.rehearsal
```

Expected: every check is `PASS`. Until preseason priors and executable quotes
pass readiness, the command must also print `NO-GO`, the model state must be
`regression_only`, and total stake must be £0. The status is written atomically
to `cfb/data/rehearsal_status.json`; history is appended to
`cfb/data/rehearsal_history.json`.

If it fails: stop publication and use the failing check's detail. A schedule-hash
failure requires review of event IDs, kickoff changes, and Week 0 teams before
updating `cfb/data/reviewed_schedule.json`.

### 5. Check betting readiness independently

```bash
python3 preflight.py --engine cfb --require-ready
```

Expected before any real-money action: exit zero with no prior-coverage,
identity, odds provenance, quote freshness, model-state, or update-status issue.

If it fails: the card remains diagnostic. Do not change market policy, suppress
the issue, or treat `diagnostic_ready` as betting approval.

### 6. Pull current executable quotes and build the slate card

Run only after the identity review for the same date range:

```bash
python3 -m cfb.season --odds-api --days 1
python3 -c 'from cfb.rehearsal import verify_card; print(verify_card("cfb/data/card.md", "cfb/data/card_manifest.json"))'
```

Expected: the fetch retains bookmaker, event ID, kickoff, and quote timestamps;
all in-window teams resolve; the manifest hash matches the exact card. If
readiness is still no-go, `safe_diagnostic_card` must be `True` and stake £0.

If it fails: retain the last-good odds file and card. Unknown teams, incomplete
same-book pairs, stale quotes, event mismatches, or a manifest mismatch are
blocking.

### 7. Record the go/no-go decision by market

Review `cfb/data/market_policy.json`, `cfb/data/rehearsal_status.json`, and the
frozen evidence in `cfb/README.md`. ML, spread, and total must each have an
explicit decision. As of 2026-08-02: ML and spread are diagnostic, total is
paper-only, and no market is eligible for real-money recording.

Expected: no recommendation is inferred from a displayed edge alone. Three
consecutive clean rehearsal days are required before operational go, and betting
go still requires separate data and market-policy approval.

If it fails: leave the affected market non-recordable and escalate the unmet
criterion; do not promote policy to meet a deadline.

## Verification

- `python3 -m cfb.generate_docs --check` exits zero.
- `python3 -m cfb.validate --gate --quiet` exits zero.
- Live provider identity review reports zero unresolved names for the card range.
- `rehearsal_status.json` reports all checks passed and the expected release
  posture.
- `card_manifest.json.card_sha256` matches `card.md`.
- Every priced quote is fresh, event-matched, and paired within one bookmaker.
- A regression-only snapshot has zero eligible bets and £0 total stake.
- The pooled bankroll preview enforces event, engine, and daily caps.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `diagnostic_preflight` fails | Missing/corrupt core schedule, games, or power artifact | Restore the last-good artifact or rerun the required refresh; do not build a card |
| `prior_mode=regression_only` | Talent/returning coverage is below the safety threshold | Keep betting disabled; acquire and validate target-season priors |
| Identity review lists a name | New provider spelling or team-name change | Verify CFBD ID and add a bounded reviewed alias; rerun focused tests and review |
| In-window Odds API event blocks refresh | Alias, fixture, or kickoff cannot be matched exactly | Review identity and CFBD schedule; never force a guessed match |
| Odds provenance issues | Legacy/stale schema, missing book/time/event, or incomplete pair | Fetch a fresh bookmaker snapshot; keep last-good file if response is malformed |
| Schedule hash differs | CFBD event or kickoff data changed since review | Diff schedules, review affected games, then update reviewed hash and date |
| Validation fingerprint differs | Historical games/line data changed | Investigate provenance and explicitly rebaseline only after review |
| Card hash mismatch | Card or manifest was edited or only one artifact published | Rebuild both from unchanged validated inputs; do not use the card |
| Update/rehearsal status is `failure` | A required control stopped the pipeline | Use the recorded failing step/check; fix and rerun from Step 1 |

## Rollback

1. Stop card publication and bet recording.
2. Leave the affected market `diagnostic`, `paper`, or `disabled`; never promote
   it during rollback.
3. Keep atomically retained last-good data/odds/model artifacts. Do not replace
   them with an empty or partially parsed provider response.
4. Redeploy the last reviewed code/config revision through the normal versioned
   release process if the failure is code-related.
5. Rerun Steps 1–5. Rebuild the card only after all required controls pass.
6. Record the incident and reset the consecutive-clean-rehearsal count through a
   real failed rehearsal entry; do not edit history to hide a failure.

## Escalation

- **Data/provider:** escalate repeated CFBD or Odds API failures, empty target-year
  priors, quota exhaustion, or unexplained schedule drift to the data owner.
- **Model:** escalate a failed frozen gate, fingerprint change, calibration drift,
  or prior-mode downgrade to the model owner.
- **Risk/operations:** escalate any non-zero stake while readiness is no-go, cap
  bypass, incorrect settlement, or manifest mismatch immediately; stop recording.
- Include `update_status.json`, `rehearsal_status.json`, relevant hashes, UTC
  timestamps, and sanitized command output. Never attach API keys.

## Change history

| Date | Change | Author |
|---|---|---|
| 2026-08-02 | Initial Week 0 runbook; added identity, rehearsal, manifest, rollback, and go/no-go controls | Codex |
