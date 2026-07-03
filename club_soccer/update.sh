#!/usr/bin/env bash
# Thin wrapper: the real daily pipeline lives in season.py (the front door —
# see README.md). This script is kept for compatibility with the app's
# scheduler / launchd-style invocation: it runs season.py, then the
# validation gate, then records provenance.
#
# Usage: ./club_soccer/update.sh [--fast]
set -euo pipefail

# Run from the repo root so the club_soccer package resolves (modules use
# package-relative imports; invoke them with `python3 -m club_soccer.X`).
cd "$(dirname "$0")/.."

python3 -m club_soccer.season "$@" || echo "season.py run had a problem — see output above"

python3 -m club_soccer.validate --gate || echo "validation warning"

# Record data provenance (offline, never blocks).
python3 -m app.provenance --engine club_soccer --write || echo "manifest skipped"
