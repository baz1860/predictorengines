#!/usr/bin/env bash
# Refactor safety-net tripwire (Phase 0). Verifies that promoted model artifacts and
# key output files are byte-identical to the Phase 0 snapshot. Structural-only refactor
# phases (1-5) must NOT change these; a mismatch means a phase altered behavior.
#
# Usage (from repo root):  bash tests/golden/verify.sh
# Re-baseline intentionally (only when outputs are *meant* to change):
#   shasum -a 256 <files...> > tests/golden/SHA256SUMS.txt
set -euo pipefail
cd "$(dirname "$0")/../.."
shasum -a 256 -c tests/golden/SHA256SUMS.txt
echo "golden outputs unchanged ✓"
