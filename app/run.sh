#!/usr/bin/env bash
# Launch Sports Predictor (native window). Run from the "Soccer Prediction" folder.
set -e
cd "$(dirname "$0")/.."   # -> Soccer Prediction/
exec python3 -m app.main
