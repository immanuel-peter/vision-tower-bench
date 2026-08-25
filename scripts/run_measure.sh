#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
session=vtb
tmux kill-session -t "$session" 2>/dev/null || true
tmux new-session -d -s "$session" \
  "uv run python -m vtb.extract --model dinov2 --images data/val2017 --limit ${1:-2000} 2>&1 | tee cache/measure.log; echo; echo '[done] press any key'; read -n 1"
echo "started tmux session '$session'; attach with: tmux attach -t $session"
