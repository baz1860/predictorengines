# Session handoff — 2026-06-15

Quick context for the next Cowork session (this one ends when the folder is moved
out of `~/Documents`, which disconnects it). After moving, reconnect the folder at
its new location to continue.

## Decisions made this session

- **No Swift rewrite.** V6's native SwiftUI app is shelved. The web UI (PyWebView +
  FastAPI + HTML/JS) is the permanent front end. Goal: make it world-class + open
  it without the terminal.
- **Visual direction: warm editorial** — paper/ink palette, terracotta accent,
  earthy win/draw/loss, serif display headings.
- **Launcher: lightweight `.app` wrapper** (no PyInstaller for now; revisit after V3).

## What was built (all front-end only — no backend/engine changes)

- **GUI upgrade (phases 1–6 complete)** — new design system in `app/web/style.css`,
  charting module `app/web/charts.js`, live Dashboard, plus new screens: Fixtures,
  Outrights, Bet-history explorer, and model comparison on Predict. Sortable/
  filterable tables (`mountTable`), row drill-downs, toasts, theme toggle.
- **New backend routes** (added earlier this session): `/api/dashboard`,
  `/api/history`, `/api/fixtures`, `/api/outrights` in `app/server.py`, backed by
  `app/dashboard_data.py`. `report.py` left intact.
- **Bug fixed:** bankroll "Open bets" table was showing settled rows after Settle
  (mountTable nested-`.table-scroll` clobbering positional lookup) — fixed with
  stable host IDs.
- **Warm-editorial reskin** — `style.css` tokens reworked (light + dark), serif
  headings, softer shadows.
- **`dashboard_preview.html`** (project root) — a self-contained, interactive
  offline preview of the whole UI with real data. Open to review the look.

## Launcher status — IMPORTANT

`Sports Predictor.app` (in the project folder) double-click launcher. Long debug
chain resolved these in turn: wrong Python (Xcode stub) → dotfile clobbering our
`DIR` var → and finally the real blocker:

- **Root cause:** the project lived in `~/Documents`, which macOS privacy (TCC)
  protects. The app could *stat* files but Python couldn't *read* the directory
  (`PermissionError: Operation not permitted`), so `import app` failed. Granting the
  app Full Disk Access didn't help (TCC blames the child pyenv Python, not the app).
- **Fix in progress:** move the whole `Soccer Prediction` folder **out of
  `~/Documents`** to the home folder (`/Users/lucky/`) or `/Users/Shared` — NOT
  Desktop/Downloads (also protected). The launcher is now **location-independent**
  (finds the project relative to the app, which must stay inside the folder), so it
  works wherever the folder lands.
- **After moving:** double-click `Sports Predictor.app` — should open with no
  terminal and no prompts. Launch errors (if any) are written to
  `.launch_error.log` in the project root and shown in an alert.
- Minor: a leftover diagnostic `Sports Predictor.app/Contents/MacOS/_run.py` is no
  longer used (couldn't delete due to a permission quirk) — safe to delete.
- The launcher uses pyenv 3.12.7 (the Python that has `webview, fastapi, uvicorn,
  pandas`). Deps: `app/requirements.txt`.

## Plan docs status

- **V3** — being implemented by Claude Code in parallel (safety/correctness:
  common engine contract, security, validation gates, event-safe settlement, etc.).
  Real bugs it fixes were confirmed: CFB edge ignores the `model` param; Golf
  settles against the latest event rather than the bet's event.
- **V4 / V5 / V6** — reviewed and annotated this session with dated "Status &
  prerequisites" callouts (none are started; the whole chain depends on V3). V6
  reconciled with the web UI as the interim product. `V4_PLAN.MD` renamed to
  `V4_PLAN.md`.
- **`GUI_UPGRADE_PLAN.md`** — the full plan + completion log for the GUI work.

## Suggested next steps

1. Confirm the app launches from its new (moved) location; delete `_run.py`.
2. Optional: further warm-editorial polish once you've seen it (palette/serif
   intensity is easy to tune in `style.css`).
3. When V3 lands and engines are stable: optionally build a PyInstaller bundle so
   the app needs no installed Python, and fold the V4/V5 ledger columns
   (`event_id`, `market`) into `dashboard_data.py` (currently market is inferred
   from bet text).
4. Keep front-end work in `app/web/` to avoid colliding with V3 backend changes.
