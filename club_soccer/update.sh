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

# Required steps fail CLOSED: their failures propagate as a nonzero exit so
# the scheduler (and anyone reading its status) sees a broken run instead of
# a false-green one. Only provenance recording is best-effort.
status=0

python3 -m club_soccer.season "$@" || { status=$?; echo "season.py FAILED (exit $status) — see output above"; }

python3 -m club_soccer.validate --gate --if-needed || { s=$?; echo "validation gate FAILED (exit $s)"; if [ "$status" -eq 0 ]; then status=$s; fi; }

# Record data provenance (offline, best-effort — never blocks).
python3 -m app.provenance --engine club_soccer --write || echo "manifest skipped"

exit $status
