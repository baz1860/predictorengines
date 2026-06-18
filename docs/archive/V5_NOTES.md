# Sports Predictor Suite - V5 Build Notes

**Scope built:** V5 foundation layer for adaptive intelligence and governance.
Everything is advisory/local/offline-first. No automatic real-money execution was
added.

## What Was Built

| Plan | Module | What it does |
|------|--------|--------------|
| **M1/P2** | `v5/registry.py` | Model registry with versioned artifacts, champion/challenger records, gated promotion, feature snapshot fingerprints, and recommendation records. |
| **M2/P2** | `v5/drift.py` | Drift reports over recommendation samples plus CLV/P&L context. Thin samples reduce confidence instead of creating false promotions. |
| **M3/P3** | `v5/portfolio.py` | Cross-sport advisory allocation with hard caps by event, engine, market, team, and day. |
| **M4/P1** | `v5/live.py` | Narrow soccer live 1X2 advisory prototype using score/minute/red-card state. Missing or stale state returns `pass`. |
| **M5/P5** | `v5/scenario.py` | Deterministic World Cup what-if line lab. Outputs are explicitly synthetic and never overwrite production feature snapshots. |
| **M6/P4** | `v5/research.py` | Research backlog generated from drift alerts and human-review signals. |
| **M8/P7** | `v5/review.py` | Human review states, tags, notes, adjusted-stake capture, and analytics excluded from model training by default. |
| — | `v5/report.py` | One-call V5 summary for registry, drift, portfolio, reviews, and research. |
| — | `app/server.py` | `/api/v5/*` endpoints for registry, recommendations, drift, portfolio, scenario, live advisory, review, and research. |
| — | `test_v5.py` | 7 focused tests for gates, audit records, drift confidence, hard caps, synthetic scenarios, live-state pass behavior, and review isolation. |

## Data Artifacts

V5 writes local files under `data/` only when used:

- `v5_model_registry.json`
- `v5_feature_snapshots.csv`
- `v5_recommendations.csv`
- `v5_reviews.csv`
- `v5_drift_report.json`
- `v5_research_backlog.json`

Recommendation records are intentionally separate from the betting ledger. The
ledger remains the source of placed bets and settlement; V5 stores the broader
decision/audit trail.

## Guardrails

- No auto-betting or execution path was added.
- Live output is `advisory` or `pass`, with `live_unvalidated` clearly marked.
- Promotion requires an explicit gate report; failed challengers are rejected and
do not replace champions.
- Human feedback analytics are `excluded_by_default` from model training.
- Portfolio allocation respects hard caps before expected return.

## How To Run

```bash
python3 test_v5.py
python3 -m v5.report
```

Example API endpoints:

```text
GET  /api/v5
GET  /api/v5/registry
POST /api/v5/recommendation
GET  /api/v5/drift
GET  /api/v5/portfolio
POST /api/v5/scenario/worldcup
POST /api/v5/live/soccer
POST /api/v5/review
GET  /api/v5/research
```

## Remaining V5 Work

- Promote live models only after archived live-state and live-odds validation.
- Add richer feature drift tests once every engine emits V4 feature snapshots.
- Build UI controls for scenario/review workflows on the existing dashboard.
- Add provider interfaces for any paid/fragile V5 data sources before use.
