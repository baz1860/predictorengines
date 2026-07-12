#!/usr/bin/env bash
# Launch the Sports Predictor Command Center (browser command console).
#   ./command_center.sh              # start on port 8799 and open the browser
#   ./command_center.sh --port 9000  # use a different port
#   ./command_center.sh --no-open    # don't auto-open the browser
cd "$(dirname "$0")"
exec python3 command_center.py "$@"
