#!/usr/bin/env bash
#
# Remove the club_soccer LaunchDaemons from this Mac. Idempotent.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shopt -s nullglob
PLISTS=("$SCRIPT_DIR"/com.*.sportspredictor.clubsoccer.*.plist)

for src in "${PLISTS[@]}"; do
  label="$(basename "$src" .plist)"
  dest="/Library/LaunchDaemons/$label.plist"
  sudo launchctl bootout "system/$label" 2>/dev/null || true
  sudo rm -f "$dest"
  echo "removed $label"
done

echo
sudo launchctl list | grep clubsoccer || echo "no clubsoccer daemons loaded."
