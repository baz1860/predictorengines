# Sports Predictor Suite - V6 Product + Operations Plan

V6 turns the V3-V5 research suite into a daily operator app. V3 made the suite
safe and measurable; V4 made the line engine stronger; V5 added governance,
recommendation audit, drift, scenarios, and human review. V6 should make those
capabilities usable without script-hunting.

## 0. Theme

Build a polished trading-desk workflow: one place to inspect health, review
recommendations, stress-test lines, manage model governance, run daily ops, and
recover safely.

V6 is not auto-betting. It is productization, reliability, and operator control.

## 1. Guardrails

1. No automatic real-money execution.
2. Existing V3-V5 gates remain authoritative.
3. Operations must be auditable: every daily run, backup, restore, and release
   migration should leave a local record.
4. Destructive actions require explicit user intent.
5. The app must fail soft: missing data, stale feeds, or unavailable validators
   show warnings, not blank screens.
6. UI additions should extend the existing app before any native rewrite.

## 2. Pillars

### P1 - Operator Home

- One health page for validation status, data freshness, bankroll risk, V5 drift,
  recommendation counts, and backup state.
- Clear status: `ok`, `warn`, `fail`, `unknown`.
- Direct links/actions for daily run, validation, backup, and review workflow.

### P2 - Governance UI

- Surface V5 model registry, champions/challengers, feature snapshots, drift
  alerts, and recommendation records.
- Promote/reject challenger workflow remains gated by validation reports.

### P3 - Review Queue

- Recommendation queue with accept/reject/watch/manual-adjust states.
- Reason tags and post-event review.
- Analytics on whether human overrides improved CLV/P&L.

### P4 - Scenario Lab UI

- Controls for World Cup what-if deltas first.
- Later extend to CFB injuries/weather, golf draw/weather, club soccer lineups.
- Scenario outputs are synthetic and exportable.

### P5 - Daily Operations

- A daily checklist that can run or preview:
  - write/refresh manifests;
  - snapshot CLV odds;
  - run validation gates;
  - run V5 drift/research report;
  - create backup;
  - emit operator report.
- Dry-run by default.

### P6 - Backup, Migration, Release

- One-command local backup of data artifacts that matter: bankroll, ledgers,
  V5 audit tables, model registry, validation reports, and settings.
- Migration status for V3/V4/V5/V6 artifacts.
- Release manifest with app/data schema versions.

### P7 - Native Rewrite Readiness

- Keep V6 web UI dense and operational.
- Isolate API payloads and view models so a future native shell can reuse them.

## 3. Milestones

### M1 - Operations API + Ops Panel

Create `/api/v6` health/ops endpoints and a dashboard panel showing validation,
freshness, bankroll, V5 governance, backup readiness, and daily-run preview.

### M2 - Backup + Release Manifest

Add backup creation and release/migration status.

### M3 - Review Queue UI

Expose V5 recommendation records and review actions in the app.

### M4 - Scenario Lab UI

Build a first World Cup scenario control surface using V5 scenario APIs.

### M5 - Governance UI

Model registry and challenger gate workflow.

### M6 - Daily Run Execution

Add explicit execution mode for safe commands with logs and per-step status.

### M7 - Native-Ready View Models

Stabilize API schemas for a future native front end.

## 4. Definition Of Done

V6 is done when the app can be used as a daily operator console: the user can see
system health, run or preview daily operations, back up critical state, review
recommendations, inspect governance/drift, and run scenario analysis from the UI
without weakening any validation, settlement, security, or bankroll guardrails.
