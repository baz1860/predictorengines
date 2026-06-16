# Sports Predictor (desktop app)

Native-feel macOS app wrapping the prediction engines. See `../GUI_PLAN.md` for the full plan.

**Phase 1 (done):** app scaffold, engine-adapter registry, FastAPI backend, frontend shell, and the World Cup **Predict** tab working end-to-end.

## Run

```bash
cd "Soccer Prediction"
pip3 install -r app/requirements.txt --break-system-packages
python3 -m app.main          # opens the native window
```

Headless / dev (no window — use a browser or curl):

```bash
uvicorn app.server:app --port 8765
# then open http://127.0.0.1:8765
```

## Layout

```
app/
  main.py              # PyWebView window + starts backend
  server.py            # FastAPI: /api/engines, /api/predict  (engine-agnostic)
  engines/
    base.py            # EngineAdapter interface + registry
    __init__.py        # register adapters here (only wiring step for a new engine)
    worldcup.py        # World Cup adapter (wraps predictor.py)
  web/                 # index.html, app.js, style.css
```

## Adding an engine

1. Write `engines/<id>.py` with a class subclassing `EngineAdapter`: set `id`,
   `name`, `sport`, `capabilities`, and implement `predict_schema()` + the
   capability methods you declared.
2. Register it in `engines/__init__.py`.

The sidebar, tabs, and Predict form all build themselves from what the adapter
reports — no server or frontend changes needed.

## V3 suite modules

- `market_blend.py` — shared market-anchored blending used by adapters.
- `provenance.py` — data manifests, freshness warnings, manual-odds schema
  checks (`python3 -m app.provenance --freshness | --check-odds <engine>`).
- `model_audit.py` — per-engine audit (validation status, params age, freshness,
  active flags), served at `GET /api/engines/{id}/audit` and shown in the Edge
  tab's model-audit panel.

Each engine's `predict_schema()` carries a `freshness` list; `edge()` attaches
`odds_issues` for a malformed manual odds file. See `../V3_NOTES.md`.
