#!/usr/bin/env bash
# One command for the whole suite: refresh every engine (data, fits,
# validation, odds), then write the combined narrative betting card.
#
#   ./daily_card.sh              # full refresh -> daily_card.md
#   ./daily_card.sh --fast       # skip the slower refits where supported
#   ./daily_card.sh --card-only  # skip refreshes, just re-render the card
#
# Every step is guarded: one engine having a bad day (offline feed, empty
# draw, offseason) never stops the others or the final card.
set -uo pipefail
cd "$(dirname "$0")"

FAST=""
CARD_ONLY=0
for arg in "$@"; do
  case $arg in
    --fast) FAST="--fast" ;;
    --card-only) CARD_ONLY=1 ;;
  esac
done

if [[ "$CARD_ONLY" -eq 0 ]]; then
  echo "════════════════════════════════════════════"
  echo "  Daily card — full suite refresh  $(date '+%Y-%m-%d %H:%M')"
  echo "════════════════════════════════════════════"

  echo ""; echo "═══ 1/6 World Cup ═══"
  ./update.sh morning || echo "  World Cup refresh had a problem — card will use last good data"

  echo ""; echo "═══ 2/6 Club soccer ═══"
  bash club_soccer/update.sh $FAST || echo "  club soccer refresh had a problem"

  echo ""; echo "═══ 3/6 Golf ═══"
  bash golf/update.sh || echo "  golf refresh had a problem"
  python3 -m golf.season || echo "  golf pricing skipped (no current tournament?)"

  echo ""; echo "═══ 4/6 Tennis ═══"
  bash tennis/update.sh || echo "  tennis refresh had a problem"
  python3 -m tennis.season || echo "  tennis pricing skipped (no live draw?)"

  echo ""; echo "═══ 5/6 NFL / CFB / NHL ═══"
  if [[ -z "$FAST" ]]; then
    python3 -m nfl.fetch_data || echo "  NFL data refresh skipped (offline / offseason)"
    python3 -m nfl.power --fit || echo "  NFL refit skipped"
    python3 -m nfl.validate --gate \
      || echo "  ##### NFL VALIDATION GATE FAILED — review before betting #####"
    python3 -m cfb.fetch_data || echo "  CFB data refresh skipped (offline / offseason)"
    python3 -m cfb.power --fit || echo "  CFB refit skipped"
    python3 -m cfb.validate --gate --quiet 2>/dev/null \
      || echo "  ##### CFB VALIDATION GATE FAILED — review before betting #####"
    python3 -m nhl.validate --quiet --gate \
      || echo "  ##### NHL VALIDATION GATE FAILED — review before betting #####"
  else
    echo "  --fast: skipping NFL/CFB data pulls, refits, and validation gates"
  fi
  python3 -m app.provenance --write --engine nfl >/dev/null 2>&1 || true
  python3 -m app.provenance --write --engine cfb >/dev/null 2>&1 || true
  python3 -m app.provenance --write --engine nhl >/dev/null 2>&1 || true

  echo ""; echo "═══ 6/6 Combined card ═══"
fi

python3 scripts/daily_card.py || { echo "daily card FAILED"; exit 1; }
echo ""
echo "Done: daily_card.md — every engine's bets for the day, explained."
