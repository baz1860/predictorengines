# Sports Predictor Suite - V6 Plan

V6 assumes V5 is fully implemented: the models are bookmaker-class, validated,
auditable, adaptive, and portfolio-aware. V6 is not about improving the engines.
It is about moving the whole product into a native Apple app that can be built,
debugged, profiled, signed, and shipped from Xcode.

The target is a native Apple Silicon build using SwiftUI, Swift Concurrency, and
Apple-native storage/UI patterns, while preserving the modelling quality and
operational guarantees from V3-V5.

App size is not a constraint. Additional dependencies are acceptable. Code
usability is a constraint: the final app should not become an opaque bundle of
scripts hidden behind buttons. The migration should leave behind clean Swift
packages, testable engine APIs, reproducible validation, and a maintainable
architecture.

> **Status (2026-06-15): long-horizon plan, not started.** V6 is a native-SwiftUI
> rewrite that assumes V5 (and V3–V4) are complete; none are yet. Read this as
> the eventual target, not the next task. Two reconciliations with today's
> codebase matter:
>
> - **V6 deliberately supersedes the current front end.** The app today is
>   PyWebView + FastAPI + HTML/JS (see `GUI_PLAN.md`), recently upgraded with a
>   live dashboard, history/fixtures/outrights screens, and an in-house SVG chart
>   kit (see `GUI_UPGRADE_PLAN.md`). That web UI is the *interim* product and the
>   concrete **screen-parity target** for V6's SwiftUI screens — the "UI parity"
>   acceptance in M7 should be measured against those screens, not against an
>   abstract spec.
> - **The `PredictionEngine` protocol must map to what exists.** Today each adapter
>   exposes `predict` / `simulate` / `edge` plus `grade_open_bets` (settlement),
>   and validation lives in separate `validate.py` scripts — there is no per-engine
>   `validate()` or `settle()` method yet. The protocol's `settle()` and
>   `validate()` cases are therefore new surface to define during the port.
> - **M2's import scope is gated on upstream work.** Importing "odds snapshots,
>   validation runs, and data manifests" presumes those artifacts exist; manifests
>   (V3 M7) and the model/recommendation tables (V5) are not built. V6 M2
>   acceptance should be conditional on that provenance work, or scoped to what
>   actually exists today: `suite_ledger.csv`, `suite_bankroll.json`,
>   `calibration.json`, `validation_baseline.json`, odds/title history CSVs.

## 0. V6 Theme

Build **Sports Predictor** as a native Apple application.

V6 should deliver:

- SwiftUI app shell built in Xcode;
- native Apple Silicon runtime;
- local-first data and model execution;
- first-class macOS app behaviors: menus, keyboard shortcuts, settings, document
  export/import, sandbox-aware file access, signing/notarization path;
- maintainable Swift engine modules;
- parity with the mature V5 engine outputs;
- no loss of security, settlement, validation, audit trail, or bankroll controls.

## 1. Strategic Migration Choice

Use a staged migration, not a big-bang rewrite.

V6 should have two runtime modes during development:

1. **Compatibility runtime** - SwiftUI app calls the existing V5 Python engines
   through a controlled local bridge. This keeps the native UI useful early and
   provides a parity oracle.
2. **Native runtime** - engines are migrated into Swift packages and called
   directly from the app.

The final production goal is native Swift engine modules for the stable,
performance-critical paths. Keeping a Python compatibility runtime as an
optional developer/parity tool is acceptable because app size and dependencies
are not blockers, but the user-facing app should not depend on fragile terminal
scripts for normal operation once an engine has been ported.

## 2. Target Xcode Architecture

Create an Xcode workspace:

```text
SportsPredictor.xcworkspace
  SportsPredictorApp/          SwiftUI macOS app target
  Packages/
    AppCore/                   navigation, settings, feature flags
    EngineCore/                common engine protocols and result types
    ModelRuntime/              math, distributions, calibration, simulation tools
    DataStore/                 SQLite/SwiftData, migrations, repositories
    MarketKit/                 odds, CLV, market movement, line conversion
    PortfolioKit/              bankroll, ledger, staking, risk constraints
    WorldCupEngine/            native migrated engine
    ClubSoccerEngine/          native migrated engine
    CFBEngine/                 native migrated engine
    GolfEngine/                native migrated engine
    PythonCompatibility/       development/parity bridge, optional in release
    ValidationKit/             golden outputs, validation runners, reports
```

Recommended Apple/native technologies:

- **SwiftUI** for all app screens.
- **Swift Concurrency** (`async/await`, actors) for long-running simulations,
  fetches, validation, and portfolio calculations.
- **SwiftData or SQLite via GRDB** for local storage. Prefer GRDB if query
  control, migrations, and large odds/history tables matter more than declarative
  convenience.
- **Keychain** for API keys.
- **Security-scoped bookmarks** for user-selected import/export folders if the
  app is sandboxed.
- **Accelerate** for linear algebra, distributions, vectorized simulation, and
  numerical routines.
- **Metal / MLX Swift / Core ML** only where they make simulation or model
  inference materially better; do not force ML frameworks where plain Swift and
  Accelerate are clearer.
- **Swift Package Manager** for internal modules and external dependencies.

Dependencies are allowed, but they should be deliberate and wrapped behind local
interfaces so the app is not hostage to any one library.

## 3. Native Domain Boundaries

The V6 app should treat each engine as a plugin-like Swift module implementing
one common protocol:

```swift
public protocol PredictionEngine: Sendable {
    var id: EngineID { get }
    var displayName: String { get }
    var sport: Sport { get }
    var capabilities: Set<EngineCapability> { get }

    func schema() async throws -> EngineSchema
    func predict(_ request: PredictionRequest) async throws -> PredictionResult
    func simulate(_ request: SimulationRequest) async throws -> SimulationResult
    func priceEdges(_ request: EdgeRequest) async throws -> EdgeResult
    func settle(_ request: SettlementRequest) async throws -> SettlementResult
    func validate(_ request: ValidationRequest) async throws -> ValidationReport
}
```

Shared types belong in `EngineCore`, not inside individual engines:

- `EngineID`, `EventID`, `MarketID`, `SelectionID`;
- `Probability`, `DecimalOdds`, `AmericanOdds`, `FairLine`;
- `PredictionResult`, `SimulationResult`, `EdgeResult`;
- `LedgerEntry`, `SettlementResult`;
- `ValidationReport`;
- `FeatureSnapshot`;
- `Recommendation`, `ReasonCode`, `ConfidenceInterval`.

This preserves the V3-V5 contract but makes it native, typed, and easier to
reason about than JSON dictionaries.

## 4. Data Layer Migration

Move from scattered CSV/JSON files to an app-owned local data store, while still
supporting import/export for transparency.

Build `DataStore` around these concepts:

- `events`;
- `participants`;
- `features`;
- `odds_snapshots`;
- `market_lines`;
- `model_versions`;
- `recommendations`;
- `ledger_entries`;
- `settlements`;
- `validation_runs`;
- `data_manifests`;
- `user_reviews`.

Rules:

- Every imported legacy CSV/JSON gets a migration path.
- Every record keeps provenance: source, fetched_at/imported_at, schema version,
  and as-of timestamp where relevant.
- Export remains simple: any table can be exported to CSV/JSON/Parquet-like
  format if a dependency makes that reasonable.
- API keys move to Keychain, never SQLite.
- Ledger migrations are backed up and reversible.

Acceptance:

- Existing V5 data can be imported into a clean V6 install.
- The imported ledger and bankroll match V5 totals exactly.
- Validation reports can be regenerated from stored feature snapshots.

## 5. UI Product Plan

V6 should feel like a real Mac app, not a web view inside a window.

Core SwiftUI screens:

- **Dashboard** - bankroll, CLV, model health, stale data warnings, today's
  recommendations, portfolio exposure.
- **Engines** - World Cup, Club Soccer, CFB, Golf, each exposing Predict,
  Simulate, Edge, Validate, and Model Audit where supported.
- **Line Lab** - what-if scenarios, fair-line sensitivity, model-vs-market
  movement.
- **Portfolio** - bankroll, open exposure, correlation clusters, stress tests,
  settled history.
- **Market Board** - odds snapshots, stale prices, line movement, cross-market
  consistency.
- **Validation Center** - latest gates, drift, champion/challenger status,
  golden-output parity.
- **Data Center** - imports, provider status, manifests, freshness, cache health.
- **Settings** - API keys in Keychain, providers, default stake constraints,
  model feature flags, storage location, export options.

Native behaviors:

- sidebar navigation with `NavigationSplitView`;
- toolbar actions for refresh, validate, export, and record;
- menus for import/export, validation, data refresh, and diagnostics;
- keyboard shortcuts for common workflows;
- native tables with sorting/filtering;
- charts using Swift Charts where suitable;
- long-running tasks shown with progress and cancellation;
- system notifications for completed refreshes or validation warnings if enabled.

## 6. Runtime and Performance

V6 should be fast enough to feel native even when models are heavy.

Build:

- actor-isolated engine registry;
- background task runner for simulations, validation, and data refresh;
- cancellation-aware Monte Carlo loops;
- progress reporting for every long operation;
- deterministic seeded execution for validation;
- simulation kernels using Accelerate/vectorization where useful;
- optional Metal/MLX path only after a CPU baseline exists;
- memory-budget diagnostics for large simulations and odds histories.

Acceptance:

- UI never blocks during simulations or validation.
- Long tasks can be cancelled safely.
- Seeded outputs match golden baselines within declared tolerance.
- Profiling identifies hot paths before adding GPU/ML dependencies.

## 7. Python Compatibility Runtime

The compatibility bridge exists to reduce migration risk, not to define the
final architecture.

Options, in order of preference:

1. **Embedded Python runtime** bundled inside the app for developer/parity builds.
2. **Local engine service** launched by the app as a managed helper during
   migration.
3. **Command-line bridge** only for one-off migration scripts and golden-output
   generation.

Bridge rules:

- fixed commands only;
- no shell strings;
- sanitized errors;
- key redaction;
- bounded input/output;
- explicit timeouts;
- versioned request/response schemas.

Acceptance:

- Swift-native result types can be populated from compatibility responses.
- Golden-output tests can compare Swift engine outputs to V5 Python outputs.
- The app can disable the compatibility runtime per engine once native parity is
  achieved.

## 8. Native Engine Porting Strategy

Port one engine at a time. Do not port all engines halfway.

Recommended order:

1. **PortfolioKit / MarketKit / DataStore** - shared foundation first.
2. **GolfEngine** - simulation-heavy, well-contained, good for testing native
   performance and seeded Monte Carlo parity.
3. **CFBEngine** - strong candidate for typed score/spread/total distributions.
4. **ClubSoccerEngine** - richer feature/lineup dependencies, benefits from the
   shared data layer being mature.
5. **WorldCupEngine** - tournament-specific logic and historical complexity; port
   after generic soccer pieces are stable.

For each engine:

- define typed inputs/outputs;
- port pure math first;
- port data loading second;
- port edge/pricing third;
- port validation fourth;
- compare against V5 golden outputs;
- run held-out gates;
- retire compatibility runtime for that engine only when parity is accepted.

Acceptance per engine:

- prediction parity versus V5 on a fixed golden set;
- edge parity on manual odds fixtures;
- settlement parity on a synthetic ledger;
- validation metrics within tolerance of V5;
- no unexplained change to recommended stakes.

## 9. Testing and Quality

Native migration should be test-first because modelling regressions can be subtle.

Test layers:

- Swift unit tests for math, odds conversion, distributions, Kelly, settlement;
- golden-output tests comparing V5 Python and V6 Swift;
- snapshot tests for major SwiftUI screens;
- data migration tests from real V5 files;
- validation gate tests per engine;
- security tests for Keychain, redaction, sandbox file access, and bridge
  command allowlists;
- performance tests for simulations and validation runs.

Create a `ValidationKit` command-line target:

```text
sports-predictor-validate --engine golf --golden
sports-predictor-validate --all --gate
sports-predictor-validate --migration-check /path/to/v5-data
```

Acceptance:

- Xcode test suite can be run locally with no network.
- Network/provider tests are separately tagged and skipped by default.
- A release cannot be cut while golden-output gates fail.

## 10. Packaging, Signing, and Distribution

Build for Apple Silicon first.

Targets:

- macOS arm64 primary target;
- optional Universal 2 later if Intel support is desired;
- iPadOS/iOS only after macOS data/model workflows are stable.

Release plan:

- app sandbox strategy decided early;
- Keychain entitlements configured;
- file import/export entitlements configured;
- app icon and bundle identifiers;
- code signing;
- notarization;
- crash/log diagnostics that never include API keys or private betting data;
- local backup/export before any data migration.

Acceptance:

- clean Xcode archive builds on Apple Silicon;
- signed/notarized `.app` opens without Gatekeeper friction;
- first launch can import V5 data or start fresh;
- uninstall/reinstall does not silently destroy user data.

## 11. Milestones

### M1 - Xcode Workspace and SwiftUI Shell

Create the workspace, app target, package layout, navigation, settings shell, and
empty engine registry.

Acceptance: native app launches from Xcode, shows Dashboard/Engines/Portfolio/
Settings skeleton, and has no Python dependency yet.

### M2 - DataStore and Migration Import

Implement local database schema, migrations, V5 importers, Keychain key storage,
and export tools.

Acceptance: V5 bankroll, ledger, odds snapshots, validation runs, and manifests
import with exact totals and provenance.

### M3 - Compatibility Runtime

Add Python compatibility bridge for existing V5 engines, feeding typed Swift
results into the SwiftUI app.

Acceptance: all engines can run through the native app via bridge, with sanitized
errors and golden-output capture.

### M4 - Native Shared Kits

Port shared math, odds, calibration, market movement, portfolio, settlement, and
validation primitives into Swift packages.

Acceptance: shared Swift unit tests pass and match V5 golden fixtures.

### M5 - First Native Engine

Port Golf or the smallest agreed engine fully native.

Acceptance: compatibility runtime can be disabled for that engine; predictions,
edges, settlement, and validation match V5 within tolerance.

### M6 - Remaining Native Engines

Port CFB, Club Soccer, and World Cup one at a time.

Acceptance: each engine clears golden-output, validation, settlement, and
performance gates before its bridge is retired.

### M7 - Native UX Completion

Build the real Dashboard, Line Lab, Portfolio, Market Board, Validation Center,
Data Center, and Settings flows.

Acceptance: all V5 workflows are available natively with progress, cancellation,
exports, and clear model audit trails.

### M8 - Performance and Profiling

Profile simulations, validations, imports, and large tables. Add Accelerate,
Metal, MLX, or caching where profiling justifies it.

Acceptance: app remains responsive during heavy work and key workflows meet
declared performance budgets.

### M9 - Signing, Notarization, and Release

Prepare the production macOS app.

Acceptance: clean Apple Silicon archive, signed/notarized app, first-launch data
migration, backup/export, and release notes.

## 12. Suggested Execution Order

```text
M1 Xcode shell
  -> M2 DataStore + migration
  -> M3 compatibility runtime
  -> M4 native shared kits
  -> M5 first native engine
  -> M6 remaining engines
  -> M7 complete native UX
  -> M8 performance pass
  -> M9 signed release
```

The most important trick is to keep the native app usable before every engine is
ported. The compatibility runtime gives the SwiftUI product immediate value and
creates the golden-output harness needed to port safely.

## 13. Definition of Done for V6

V6 is done when **Sports Predictor** is a native SwiftUI Apple Silicon macOS app
that builds in Xcode, stores secrets in Keychain, imports V5 data safely, exposes
all V5 workflows through native UI, runs each engine through typed Swift APIs,
passes golden-output and validation gates against V5, remains responsive during
heavy modelling work, and can be signed/notarized for normal macOS use.
