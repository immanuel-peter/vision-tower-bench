#!/usr/bin/env bash
# Idempotent MiniMax-M3 diagnosis driver. Safe to start while the live
# L32 attention probe (pid in LIVE_ATTN_PID) is still running: this script
# waits for that process and never relaunches extract if caches exist.
set -u
cd /home/shadeform/vision-tower-bench

ROOT=/home/shadeform/vision-tower-bench
LOGDIR=/tmp/mm3_logs
LABELS=data/imagenet100/validation_labels.json
FULL=cache/minimax_m3_448_full
POOL4=cache/minimax_m3_448_pool4
ATTN_JSON=/tmp/mm3_full_attn.json
MEAN_JSON=/tmp/mm3_full_mean_L32.json
# In-flight job started by the Grok session. Do not kill these.
LIVE_ATTN_PID=${LIVE_ATTN_PID:-25905}
LIVE_BASH_PID=${LIVE_BASH_PID:-25895}

mkdir -p "$LOGDIR"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() { echo "[$(date -Is)] $*"; }
phase() {
  local name=$1
  echo "=== $(date -Is) $name ===" | tee -a "$LOGDIR/timings.txt"
  log "$name"
}

need() {
  # skip if output file exists and is non-empty
  local out=$1
  if [ -s "$out" ]; then
    log "skip, exists: $out"
    return 1
  fi
  return 0
}

run() {
  log "+ $*"
  if ! "$@"; then
    log "FAILED: $*"
    echo "FAILED: $*" >> "$LOGDIR/failures.txt"
    return 1
  fi
}

probe() {
  uv run python scripts/minimax_full_probe.py \
    --run "$FULL" --labels "$LABELS" --device cuda --match-capacity \
    --stages tower --depth-points 32 "$@"
}

wait_pid() {
  local pid=$1
  local what=$2
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    log "$what pid $pid not running"
    return 0
  fi
  log "waiting for $what pid $pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
  log "$what pid $pid exited"
}

cell_top1() {
  python3 - "$1" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
cells = d["cells"] if isinstance(d, dict) and "cells" in d else [d]
print(cells[0]["test_accuracy"])
PY
}

exec > >(tee -a "$LOGDIR/continue.log") 2>&1
phase "CONTINUE start"
log "cwd=$ROOT live_attn=$LIVE_ATTN_PID live_bash=$LIVE_BASH_PID"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader || true
free -h | head -2
df -h / | tail -1

# ----- Step 2: wait for in-flight attention, retry only if it died dirty -----
phase "STEP2 wait live attention"
if [ -s "$ATTN_JSON" ]; then
  log "attention json already present"
else
  wait_pid "$LIVE_ATTN_PID" "live attention"
  wait_pid "$LIVE_BASH_PID" "live bash (may still be writing)"
  if [ ! -s "$ATTN_JSON" ]; then
    log "live job produced no $ATTN_JSON; running attention in tmux"
    phase "STEP2 retry attention"
    probe --readout attention --token-view all --out "$ATTN_JSON" || true
  fi
fi

if [ -s "$ATTN_JSON" ]; then
  log "full-grid L32 attention top1=$(cell_top1 "$ATTN_JSON")"
else
  log "ATTENTION JSON MISSING after retry"
fi

# Mean is queued on LIVE_BASH_PID after attention. Wait it out, then fill gap.
phase "STEP2 mean L32"
if [ -s "$MEAN_JSON" ]; then
  log "mean json already present"
else
  wait_pid "$LIVE_BASH_PID" "live bash before mean"
  # If the live python is still the original attention job, wait.
  wait_pid "$LIVE_ATTN_PID" "live attention before mean"
  if [ ! -s "$MEAN_JSON" ]; then
    log "running mean-pooled full-grid L32"
    probe --readout mean --mean-pool-first --out "$MEAN_JSON" || true
  fi
fi

ATTN_TOP1=0
if [ -s "$ATTN_JSON" ]; then
  ATTN_TOP1=$(cell_top1 "$ATTN_JSON")
fi
log "H1 gate: full-grid attention L32=$ATTN_TOP1 (need >= 0.7 to run token-0)"

# ----- Step 3 token-0 -----
phase "STEP3 token0"
if python3 -c "import sys; sys.exit(0 if float('$ATTN_TOP1') >= 0.7 else 1)"; then
  if need /tmp/mm3_token0.json; then
    probe --readout attention --token-view token0 --out /tmp/mm3_token0_attn.json || true
    probe --readout mean --token-view token0 --mean-pool-first --out /tmp/mm3_token0_mean.json || true
    probe --readout attention --token-view no_token0 --out /tmp/mm3_notoken0_attn.json || true
    probe --readout mean --token-view no_token0 --mean-pool-first --out /tmp/mm3_notoken0_mean.json || true
    python3 - <<'PY'
import json
from pathlib import Path
cells = []
for p in [
    "/tmp/mm3_token0_attn.json", "/tmp/mm3_token0_mean.json",
    "/tmp/mm3_notoken0_attn.json", "/tmp/mm3_notoken0_mean.json",
]:
    path = Path(p)
    if not path.exists() or path.stat().st_size == 0:
        continue
    d = json.loads(path.read_text())
    for c in d.get("cells", []):
        c["source_file"] = p
        cells.append(c)
Path("/tmp/mm3_token0.json").write_text(json.dumps({"cells": cells}, indent=2) + "\n")
print("wrote /tmp/mm3_token0.json", len(cells), "cells")
PY
  fi
else
  log "H1 not supported by full-grid number; skipping token-0"
  echo '{"skipped": true, "reason": "full-grid attention L32 < 0.7"}' > /tmp/mm3_token0.json
fi

# ----- Step 4 forward parity -----
phase "STEP4 forward parity"
if need /tmp/mm3_forward_parity.json; then
  run uv run python scripts/minimax_forward_parity.py \
    --images data/imagenet100/validation --n 8 --device cuda \
    --out /tmp/mm3_forward_parity.json || true
fi

# ----- Step 5 native 672 pool-4 -----
phase "STEP5 672 pool4"
if [ ! -s cache/minimax_m3_672_pool4/summary.json ]; then
  run uv run python -m vtb.extract --model minimax_m3 \
    --images data/imagenet100/validation --device cuda --workers 8 \
    --resolution 672 --pool 4 --final-stages tower --out cache || true
else
  log "skip 672 extract, cache exists"
fi
if need /tmp/mm3_672.json; then
  if [ -d cache/minimax_m3_672_pool4 ]; then
    run uv run python -m vtb.probe_run \
      --run cache/minimax_m3_672_pool4 \
      --labels "$LABELS" --readout attention --match-capacity \
      --device cuda --stages tower --out /tmp/mm3_672.json || true
  fi
fi

# ----- Step 6 stats + optional MLP -----
phase "STEP6 feature stats"
if need /tmp/mm3_feature_stats.json; then
  run uv run python scripts/minimax_feature_stats.py \
    --run "$FULL" --labels "$LABELS" --layers 20 32 \
    --out /tmp/mm3_feature_stats.json || true
fi

KNN=0
if [ -s /tmp/mm3_feature_stats.json ]; then
  KNN=$(python3 - <<'PY'
import json
d=json.load(open("/tmp/mm3_feature_stats.json"))
vals=[]
for layer in d.get("layers", []):
    km=layer.get("knn_mean") or {}
    if "test_top1" in km:
        vals.append(km["test_top1"])
print(max(vals) if vals else 0)
PY
)
fi
log "max mean-pool cosine-kNN=$KNN"
phase "STEP6 MLP"
if python3 -c "import sys; sys.exit(0 if float('$KNN') >= 0.45 else 1)"; then
  if need /tmp/mm3_mlp.json; then
    run uv run python scripts/minimax_mlp_probe.py \
      --run "$FULL" --labels "$LABELS" --layer 32 --device cuda \
      --out /tmp/mm3_mlp.json || true
  fi
else
  log "kNN not high enough to justify MLP; skipping"
  echo '{"skipped": true, "reason": "knn < 0.45"}' > /tmp/mm3_mlp.json
fi

# ----- Report -----
phase "REPORT"
run uv run python scripts/minimax_write_report.py --out /tmp/MM3_DIAGNOSIS.md || true
cp -f /tmp/MM3_DIAGNOSIS.md "$LOGDIR/MM3_DIAGNOSIS.md" 2>/dev/null || true
ls -lh /tmp/mm3_*.json /tmp/MM3_DIAGNOSIS.md 2>/dev/null || true
phase "CONTINUE done"
echo "DONE $(date -Is)"
# Keep the tmux pane alive so attach still shows the log.
exec bash
