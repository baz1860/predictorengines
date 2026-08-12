#!/usr/bin/env bash
# cfb/refreeze.sh — re-freeze the CFB validation artifacts after a REVIEWED
# change to validation inputs (games.csv, closing_spreads.csv, closing_totals.csv).
#
# This rewrites the artifacts that authorise the engine:
#   cfb/data/nested_validation_2025.json   (+ blend_weight.json — the RUNTIME model)
#   cfb/data/validation_baseline.json      (the regression gate's reference)
#   cfb/data/market_validation_2025.json
#   cfb/README.md                          (generated metrics section)
#
# It is deliberately NOT wired into update.sh. The fingerprint gate exists to
# stop the dataset moving unnoticed; a script that re-baselined automatically
# would defeat it. Run this only after you have reviewed WHY the inputs changed.
#
# Usage:
#   bash cfb/refreeze.sh --confirm
#   bash cfb/refreeze.sh --confirm --with-challenger   # also refit the prior challenger
#
# Every overwritten artifact is copied to cfb/data/backups/refreeze_<stamp>/ first.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIRM=0
WITH_CHALLENGER=0
for arg in "$@"; do
  case "$arg" in
    --confirm) CONFIRM=1 ;;
    --with-challenger) WITH_CHALLENGER=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ARTIFACTS=(
  cfb/data/nested_validation_2025.json
  cfb/data/blend_weight.json
  cfb/data/validation_baseline.json
  cfb/data/market_validation_2025.json
  cfb/data/validation_datasets.json
  cfb/README.md
)

summary() {
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
label = sys.argv[1]
def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}
weight = load("cfb/data/blend_weight.json")
base = load("cfb/data/validation_baseline.json")
fp = (base.get("data_fingerprint") or {}).get("line_sha256")
print(f"  {label:<8} w_elo={weight.get('w_elo', '?')} "
      f"brier={base.get('ml_brier', '?')} margin_mae={base.get('margin_mae', '?')} "
      f"total_mae={base.get('total_mae', '?')}")
print(f"  {'':<8} line fingerprint {str(fp)[:16]}")
PY
}

echo "════════════════════════════════════════════"
echo "  CFB validation re-freeze  $(date '+%Y-%m-%d %H:%M')"
echo "════════════════════════════════════════════"
echo ""
echo "Current state:"
summary "before"
echo ""

if (( ! CONFIRM )); then
  cat <<'EOF'
This rewrites the frozen validation artifacts, including the runtime blend
weight and the regression baseline the gate compares against.

Only do this when you have REVIEWED why the validation inputs changed. If the
gate failed and you do not know why, investigate first:

  git diff --stat cfb/data/closing_spreads.csv cfb/data/closing_totals.csv
  python3 -m cfb.dataset_fingerprint

Re-run with --confirm to proceed.
EOF
  exit 1
fi

STAMP="$(date '+%Y%m%dT%H%M%S')"
BACKUP="cfb/data/backups/refreeze_${STAMP}"
mkdir -p "$BACKUP"
for f in "${ARTIFACTS[@]}"; do
  [ -f "$f" ] && cp "$f" "$BACKUP/$(basename "$f")"
done
echo "backup -> $BACKUP"

echo ""; echo "── 1/4 Nested holdout (selects + freezes runtime w_elo) ──"
python3 -m cfb.validate --nested-holdout --quiet --write

echo ""; echo "── 2/4 Validation baseline ──"
python3 -m cfb.validate --quiet --update-baseline

echo ""; echo "── 3/4 Market challengers ──"
python3 -m cfb.market_validation --write

if (( WITH_CHALLENGER )); then
  echo ""; echo "── 3b/4 Preseason-prior challenger ──"
  python3 -m cfb.prior_challenger --write > /dev/null
  echo "  prior_challenger_2025.json rewritten"
fi

echo ""; echo "── 4/4 Generated documentation ──"
python3 -m cfb.generate_docs --write

echo ""; echo "── Verification ──"
python3 -m cfb.generate_docs --check
python3 -m cfb.validate --gate --quiet
echo "  validation gate: PASS"

echo ""
echo "State after re-freeze:"
summary "after"
echo ""
echo "Backup of the previous artifacts: $BACKUP"
echo ""
echo "Now confirm the whole engine still rehearses cleanly:"
echo "  python3 -m cfb.rehearsal"
