# Prediction Suite — Desktop GUI Plan

A native-feel macOS app ("**Sports Predictor**") that wraps your prediction engines (World Cup, CFB, golf) behind one window, with an architecture that lets you bolt on any future engine by dropping in a single adapter file.

## Goal

Stop running everything from the terminal. One Mac app named **Sports Predictor**, dock icon and native window, that exposes for each engine: match/event predictions, the tournament/event simulator, the edge finder (with odds entry/pull), and bankroll + bet tracking. New engines plug in without touching the UI.

Three settled decisions baked into this plan:

- **One shared bankroll across every engine.** All bets — soccer, CFB, golf, future — draw from and settle into a single bankroll and ledger, not per-sport pots.
- **Odds API keys live in the app**, managed in a Settings area where you can add/change keys per data source.
- **App is called "Sports Predictor"** in the dock.

## Recommended stack

**PyWebView + FastAPI + a lightweight HTML/JS frontend, packaged to a `.app` with PyInstaller.**

Why this and not the alternatives:

- Your engines are already Python. PyWebView runs a real macOS window (WKWebView — no browser chrome, no separate tab) and ships as a signed `.app` with a dock icon, so it *feels* native while reusing 100% of the existing model code in-process. No rewrite, no language bridge.
- Electron would mean shipping a whole Chromium runtime and a JS↔Python sidecar — heavier and more moving parts for a single-user tool.
- A pure native SwiftUI app would look the most Mac-like but forces a Swift↔Python bridge and duplicated logic — large effort for marginal polish.
- Tauri is lighter than Electron but adds a Rust toolchain you don't currently need.

PyWebView hits the "native-feel" bar at the lowest build cost given the Python codebase.

## Architecture — the part that makes it extensible

Three layers:

```
┌─────────────────────────────────────────────┐
│  Frontend (HTML/JS in a native WKWebView)    │
│  Sidebar = engines · Tabs = capabilities     │
└───────────────┬─────────────────────────────┘
                │  JSON over localhost
┌───────────────┴─────────────────────────────┐
│  FastAPI backend (in-process)                │
│  Generic routes: /engines, /predict,         │
│  /simulate, /edge, /bankroll                 │
└───────────────┬─────────────────────────────┘
┌───────────────┴─────────────────────────────┐
│  Engine adapter registry                     │
│  worldcup.py · cfb.py · golf.py · <future>   │
└──────────────────────────────────────────────┘
```

The key piece is a common **EngineAdapter** interface. Each engine ships one adapter that declares its identity and which capabilities it supports, and wraps the existing scripts. Adding an engine = write one adapter + drop it in the registry folder; the UI discovers it automatically and renders only the panels it supports.

```python
class EngineAdapter:
    id: str                  # "worldcup", "cfb", "golf"
    name: str                # "World Cup 2026"
    sport: str               # "soccer", "cfb", "golf"
    capabilities: set        # {"predict", "simulate", "edge", "bankroll"}

    def predict(self, params) -> dict: ...      # match or field prediction
    def simulate(self, params) -> dict: ...      # MC tournament/event
    def edge(self, params) -> dict: ...          # edges, EV, Kelly stakes
    def bankroll(self, action, params) -> dict:  # status / settle / reset
```

Where possible adapters call your functions directly (import `predictor`, `simulate`, `edge`); where a script is CLI-only, the adapter shells out and parses the CSV it writes. Either way the frontend never knows the difference.

### Capability mapping (why it can't be hardcoded tabs)

| Capability | World Cup | CFB | Golf |
|---|---|---|---|
| Predict | match (1X2 + scoreline) | match (win/spread/total) | **field** (win/T5/T10/T20/cut) |
| Simulate | group + knockout bracket | win totals | 4-round MC w/ cut |
| Edge | 1X2, O/U, BTTS, outrights | ML / spread / totals | outright/T5/T10/T20/H2H/cut |
| Bankroll | shared | shared | shared |

Golf has no two-team "match" screen — it's field-based. The capability set per engine drives which tabs appear, so each sport shows the right controls and nothing irrelevant.

### Shared bankroll (architecture change)

Today the engines keep separate `bankroll.json` and `ledger.csv` files (World Cup and CFB each have their own; golf has none). The app unifies these into **one bankroll store and one ledger** at the app level, with each ledger row tagged by engine/sport. Practically:

- A single `data/bankroll.json` + `data/ledger.csv` at the suite root become the source of truth.
- Each engine's edge finder records bets into the shared ledger (tagged `sport`), and settlement compounds the one bankroll.
- The Bankroll tab is a suite-level view (not per-engine), filterable by sport, showing total bankroll, open bets across all sports, and combined P&L.
- One-time migration: fold the existing per-engine bankroll/ledger files into the unified store so history isn't lost.

This also gives golf bankroll/tracking for free, since it just writes into the shared ledger.

## Screen layout

- **Left sidebar:** engine picker (World Cup / CFB / Golf / …), each showing its sport icon, plus suite-level **Bankroll** and **Settings** entries that sit outside any single engine.
- **Top tabs (per engine):** Predict · Simulate · Edge — only the ones the selected engine supports light up.
- **Predict:** team/field pickers (typeahead validated against each engine's dataset names), model toggle (elo/dc/blend etc.), results card with probabilities + likely scorelines/spreads.
- **Simulate:** run count slider, "go" button with a progress spinner (sims are seconds-long), sortable results table (trophy/round/win-total odds).
- **Edge:** odds entry grid *or* "pull live odds" (your existing API-key path), results table with edge %, EV, quarter-Kelly stake; a "record bets" toggle.
- **Bankroll (suite-level):** the one shared bankroll, open bets across all sports (filterable by sport), combined P&L, a "settle results" button, reset.
- **Settings (suite-level):** odds-API keys per data source (the-odds-api, api-football, etc.) — add, edit, remove; stored locally on disk. Also default bankroll, default Kelly fraction, and default model per engine.

State stays on disk in your existing CSV/JSON files — the GUI is a front end over the same data, so terminal and app stay in sync.

## Build phases

1. **Skeleton (½–1 day).** PyWebView window + FastAPI + sidebar that lists engines from the registry. One adapter (World Cup) wired to the Predict tab end-to-end.
2. **Unified bankroll + World Cup full (1–1.5 days).** Build the shared bankroll store and migrate the existing per-engine bankroll/ledger files into it. Complete Simulate, Edge (manual + live odds), and the suite-level Bankroll view. This proves every capability type against the most feature-complete engine.
3. **CFB + Golf adapters (1 day).** Write two more adapters against the now-stable interface; confirm the UI adapts (golf shows a field screen, no match tab) and that all three engines record into the one shared ledger.
4. **Settings + polish + package (1 day).** Settings area for odds-API keys and defaults; loading/empty/error states, input validation; then PyInstaller `.app` bundle named "Sports Predictor", app icon, and a note on macOS gatekeeper / optional code-signing so it opens without the "unidentified developer" prompt.
5. **Verification.** Run each engine's each capability through the GUI and confirm outputs match the CLI for the same inputs (golden-output check), plus a quick pass on a fresh data refresh.

Rough total: ~4–4.5 working days to a packaged app.

## Risks / decisions to settle

- **Long-running sims:** large Monte Carlo runs shouldn't freeze the window — run them in a worker thread/process with a progress callback. Planned for, worth confirming the run sizes you actually use.
- **Dataset name matching:** team/player names must match each engine's data exactly. The typeahead solves this but needs each adapter to expose its valid-name list.
- **Packaging weight:** PyInstaller bundles pandas/numpy — the `.app` will be ~150–250 MB. Fine for personal use; flagging so it's not a surprise.
- **Code-signing:** unsigned apps need a right-click-open the first time. If you want a clean double-click launch we can note the signing steps (needs an Apple developer account).
- **Shared-bankroll migration:** the one-time fold of existing per-engine bankroll/ledger files into the unified store needs care so no bet history is lost — I'll back up the originals before migrating.
- **API key storage:** keys live in a local settings file. Fine for a personal Mac; if you'd prefer they're encrypted in the macOS Keychain rather than plain on disk, that's a small add — flag if you want it.

## Settled

- Single shared bankroll across all engines and bets.
- Odds API keys stored in-app, editable per source under Settings.
- Dock name: **Sports Predictor**.
