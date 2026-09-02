#!/usr/bin/env bash
# Usage: scripts/correspondence_matrix.sh <correspondence_data_root> <out_dir>
set -euo pipefail

DATA_ROOT="${1:?correspondence data root}"; shift
OUT="${1:?output dir}"; shift

MODELS="${MODELS:-dinov2 siglip2 moonvit_v2 kimi_k26 qwen3_5 muse_glimmer}"
DATASETS="${DATASETS:-scannet navi spair}"
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
IFS=, read -ra GPU <<< "$GPUS"
LANES="${LANES:-${#GPU[@]}}"
THREADS="${THREADS:-2}"

mkdir -p "$OUT"
ALERTS="$OUT/alerts.log"
: > "$ALERTS"
FAILURE_PATTERNS='Traceback|CUDA out of memory|RuntimeError|Killed'

root_for() {
    case "$1" in
        scannet) echo "$DATA_ROOT/scannet/scannet_test_1500" ;;
        navi) echo "$DATA_ROOT/navi/navi_v1" ;;
        spair) echo "$DATA_ROOT/spair/SPair-71k" ;;
    esac
}

jobs=()
for model in $MODELS; do
    for dataset in $DATASETS; do
        jobs+=("$model $dataset")
    done
done

for ((i = 0; i < LANES; i++)); do queue[i]=""; done
for i in "${!jobs[@]}"; do
    lane=$((i % LANES))
    queue[lane]+="${jobs[i]}"$'\n'
done

lane() {
    local index="$1" gpu="$2" item model dataset root out rc
    while read -r model dataset; do
        [ -n "$model" ] || continue
        root=$(root_for "$dataset")
        out="$OUT/${model}_correspondence_${dataset}.json"
        rc=0
        echo "[$(date -u +%T)] start $model $dataset on gpu $gpu" >> "$OUT/lane$index.log"
        CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
        uv run python scripts/correspondence_run.py \
            --model "$model" --dataset "$dataset" --root "$root" --out "$out" \
            > "$OUT/${model}_${dataset}.log" 2>&1 || rc=$?
        echo "[$(date -u +%T)] done $model $dataset rc=$rc" >> "$OUT/lane$index.log"
        [ "$rc" -eq 0 ] || echo "$model $dataset exited rc=$rc" >> "$ALERTS"
        grep -nE "$FAILURE_PATTERNS" "$OUT/${model}_${dataset}.log" \
            | sed "s|^|$model $dataset |" >> "$ALERTS" || true
        [ "$rc" -eq 0 ] && touch "$OUT/${model}_${dataset}.DONE"
    done <<< "${queue[$index]}"
    touch "$OUT/lane$index.DONE"
}

echo "$LANES lanes over gpus $GPUS, $THREADS threads per lane, ${#jobs[@]} jobs"
pids=()
for ((i = 0; i < LANES; i++)); do
    [ -n "${queue[i]}" ] || continue
    lane "$i" "${GPU[i % ${#GPU[@]}]}" &
    pids[i]=$!
done
for i in "${!pids[@]}"; do wait "${pids[i]}"; done

if [ -s "$ALERTS" ]; then
    cat "$ALERTS"
else
    echo "no alerts"
fi
