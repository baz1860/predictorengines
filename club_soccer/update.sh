#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root so the club_soccer package resolves (modules use
# package-relative imports since Phase 4; invoke them with `python3 -m club_soccer.X`).
cd "$(dirname "$0")/.."

SEASON="${1:-2025}"

python3 -m club_soccer.fetch --season "$SEASON" --current || echo "fetch skipped"
python3 -m club_soccer.model --fit
python3 -m club_soccer.edge || echo "edge skipped"
python3 -m club_soccer.validate --gate || echo "validation warning"

# Record data provenance (offline, never blocks).
python3 -m app.provenance --engine club_soccer --write || echo "manifest skipped"
