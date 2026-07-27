#!/usr/bin/env bash
#
# Install the club_soccer LaunchDaemons on THIS Mac (run it on the mini, from
# the repo). Renders the plist templates in this folder, installs them to
# /Library/LaunchDaemons with root:wheel/644, and (re)loads them. Idempotent:
# safe to re-run after editing anything.
#
# Auto-detects the user, project dir and python. Override any of them:
#   CS_USER=barrie CS_DIR="/Users/barrie/AI Models/Soccer Prediction" \
#   CS_PYBIN="/Users/barrie/.pyenv/versions/3.12.7/bin/python3" ./install.sh
#
set -euo pipefail

# ── resolve paths ──────────────────────────────────────────────────────────
INVOKING_USER="${CS_USER:-$(whoami)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${CS_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"   # deploy/mac-mini -> repo root

if [[ -n "${CS_PYBIN:-}" ]]; then
  PYBIN="$CS_PYBIN"
elif command -v pyenv >/dev/null 2>&1 && pyenv which python3 >/dev/null 2>&1; then
  PYBIN="$(pyenv which python3)"
else
  PYBIN="$(command -v python3 || true)"
fi
PYDIR="$(dirname "$PYBIN")"

DAEMON_DIR="/Library/LaunchDaemons"

# ── sanity checks ──────────────────────────────────────────────────────────
[[ "$(uname)" == "Darwin" ]]        || { echo "ERROR: this installer is for macOS."; exit 1; }
[[ "$INVOKING_USER" != "root" ]]    || { echo "ERROR: run as your normal user (the script uses sudo itself)."; exit 1; }
[[ -n "$PYBIN" && -x "$PYBIN" ]]    || { echo "ERROR: python3 not found — set CS_PYBIN."; exit 1; }
[[ -d "$PROJECT_DIR/club_soccer" ]] || { echo "ERROR: no club_soccer under '$PROJECT_DIR' — set CS_DIR."; exit 1; }
"$PYBIN" -c "import pandas, numpy" 2>/dev/null \
  || { echo "ERROR: '$PYBIN' can't import pandas/numpy — install deps first."; exit 1; }

echo "user        : $INVOKING_USER"
echo "project dir : $PROJECT_DIR"
echo "python      : $PYBIN"
echo

mkdir -p "$PROJECT_DIR/logs"

# ── install each daemon ────────────────────────────────────────────────────
# The plists in this folder are FINAL (fully-filled absolute paths), so this
# just validates and copies them — no templating, no placeholders.
shopt -s nullglob
plists=("$SCRIPT_DIR"/com.*.sportspredictor.clubsoccer.*.plist)
[[ ${#plists[@]} -gt 0 ]] || { echo "ERROR: no clubsoccer plists found in $SCRIPT_DIR"; exit 1; }

for src in "${plists[@]}"; do
  base="$(basename "$src")"
  label="${base%.plist}"

  # Fail before touching the system if the plist is malformed.
  plutil -lint "$src" >/dev/null || { echo "ERROR: $base is not a valid plist"; exit 1; }

  dest="$DAEMON_DIR/$base"
  echo "installing $dest"
  sudo launchctl bootout "system/$label" 2>/dev/null || true   # unload any previous copy
  sudo cp "$src" "$dest"
  sudo chown root:wheel "$dest"
  sudo chmod 644 "$dest"
  sudo launchctl bootstrap system "$dest"
done

echo
echo "loaded daemons:"
sudo launchctl list | grep clubsoccer || echo "  (none found — check the logs)"
echo
echo "Done. Capture ran once just now and repeats every 15 min; transfer data"
echo "updates incrementally at 06:30; the incremental card job runs at 07:30."
echo "Logs: $PROJECT_DIR/logs/"
echo "Tail it:  tail -f \"$PROJECT_DIR/logs/club_soccer_capture.log\""
