#!/bin/bash
# Install the club_soccer season + monitor + decision-capture LaunchAgents.
# Run this ON THE MAC (it needs your real ~/Library and launchctl).
#
#   bash "/Users/lucky/AI Models/Soccer Prediction/deploy/install_launchagents.sh"
#
# The monitor runs TWICE (09:00 and 10:00). That is deliberate: with the
# current 2-hour dead-run deadline, a 07:30 job killed at 07:31 is only 90
# minutes old at 09:00 and still reports OK. The 10:00 pass is the one that
# actually catches it.
set -euo pipefail

SRC="/Users/lucky/AI Models/Soccer Prediction/deploy"
DEST="/Users/lucky/Library/LaunchAgents"
SEASON="com.barrie.sportspredictor.clubsoccer.season"
MONITOR="com.barrie.sportspredictor.clubsoccer.monitor"
CAPTURE="com.barrie.sportspredictor.clubsoccer.capture"

mkdir -p "$DEST" "/Users/lucky/Library/Logs"

cp "$SRC/$SEASON.plist"  "$DEST/$SEASON.plist"
cp "$SRC/$MONITOR.plist" "$DEST/$MONITOR.plist"
cp "$SRC/$CAPTURE.plist" "$DEST/$CAPTURE.plist"

plutil -lint "$DEST/$SEASON.plist"
plutil -lint "$DEST/$MONITOR.plist"
plutil -lint "$DEST/$CAPTURE.plist"

launchctl bootout "gui/$(id -u)" "$DEST/$SEASON.plist"  2>/dev/null || true
launchctl bootout "gui/$(id -u)" "$DEST/$MONITOR.plist" 2>/dev/null || true
launchctl bootout "gui/$(id -u)" "$DEST/$CAPTURE.plist" 2>/dev/null || true

launchctl bootstrap "gui/$(id -u)" "$DEST/$SEASON.plist"
launchctl bootstrap "gui/$(id -u)" "$DEST/$MONITOR.plist"
launchctl bootstrap "gui/$(id -u)" "$DEST/$CAPTURE.plist"

launchctl enable "gui/$(id -u)/$SEASON"
launchctl enable "gui/$(id -u)/$MONITOR"
launchctl enable "gui/$(id -u)/$CAPTURE"

launchctl print "gui/$(id -u)/$SEASON"  | head -20
launchctl print "gui/$(id -u)/$MONITOR" | head -20
launchctl print "gui/$(id -u)/$CAPTURE" | head -20

echo
echo "Installed. Logs:"
echo "  /Users/lucky/Library/Logs/club_soccer_season.log"
echo "  /Users/lucky/Library/Logs/club_soccer_monitor.log"
echo "  /Users/lucky/Library/Logs/club_soccer_capture.log"
