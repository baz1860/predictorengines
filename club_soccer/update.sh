#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SEASON="${1:-2025}"

python3 fetch.py --season "$SEASON" --current || echo "fetch skipped"
python3 model.py --fit
python3 edge.py || echo "edge skipped"
python3 validate.py --gate || echo "validation warning"

# Record data provenance (offline, never blocks) — run from the repo root.
(cd .. && python3 -m app.provenance --engine club_soccer --write) || echo "manifest skipped"
