#!/usr/bin/env bash
# Usage: scripts/perturbation_matrix.sh <identity_images> <condition_images> <cache_root> <labels> <manifest> <out>
set -euo pipefail

IDENTITY="${1:?identity image root}"; shift
CONDITIONS="${1:?condition image root}"; shift
CACHE="${1:?cache root}"; shift
LABELS="${1:?labels json}"; shift
MANIFEST="${1:?transform manifest}"; shift
OUT="${1:?result directory}"; shift

MODELS="${MODELS:-kimi_k26 moonvit_v2 qwen3_5 muse_glimmer dinov2 siglip2}"
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
IFS=, read -ra GPU <<< "$GPUS"
LANES="${LANES:-${#GPU[@]}}"
THREADS="${THREADS:-2}"
WORKERS="${WORKERS:-8}"
mkdir -p "$CACHE" "$OUT"
: > "$OUT/alerts.log"
: > "$OUT/expected_cells.tsv"

batch_for() {
    case "$1" in
        dinov2|siglip2) echo 16 ;;
        moonvit_v2|kimi_k26) echo 1 ;;
        qwen3_5) echo 8 ;;
        muse_glimmer) echo 4 ;;
    esac
}

expected_for() {
    case "$1" in
        dinov2|siglip2) echo 1 ;;
        *) echo 3 ;;
    esac
}

jobs=($MODELS)
for ((i = 0; i < LANES; i++)); do queue[i]=""; done
for i in "${!jobs[@]}"; do
    lane_index=$((i % LANES))
    queue[lane_index]+="${jobs[i]}"$'\n'
done

lane() {
    local index="$1" gpu="$2" model batch rc
    while read -r model; do
        [ -n "$model" ] || continue
        batch=$(batch_for "$model")
        rc=0
        echo "[$(date -u +%T)] start $model extraction on gpu $gpu" >> "$OUT/lane$index.log"
        CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
        uv run python scripts/perturbation_extract.py \
            --model "$model" --identity-images "$IDENTITY" --condition-images "$CONDITIONS" \
            --transform-manifest "$MANIFEST" --out "$CACHE" --batch-size "$batch" \
            --workers "$WORKERS" --device cuda > "$OUT/${model}_extract.log" 2>&1 || rc=$?
        echo "[$(date -u +%T)] done $model extraction rc=$rc" >> "$OUT/lane$index.log"
        [ "$rc" -eq 0 ] || { echo "$model extraction exited rc=$rc" >> "$OUT/alerts.log"; continue; }

        rc=0
        echo "[$(date -u +%T)] start $model fixed-readout scoring" >> "$OUT/lane$index.log"
        CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
        uv run python scripts/perturbation_run.py \
            --clean-run "$CACHE/identity/${model}_448_pool4" --features-root "$CACHE" \
            --model-dir "${model}_448_pool4" --labels "$LABELS" \
            --transform-manifest "$MANIFEST" --out "$OUT/${model}_perturbation.json" \
            --device cuda > "$OUT/${model}_probe.log" 2>&1 || rc=$?
        echo "[$(date -u +%T)] done $model scoring rc=$rc" >> "$OUT/lane$index.log"
        [ "$rc" -eq 0 ] || echo "$model scoring exited rc=$rc" >> "$OUT/alerts.log"
    done <<< "${queue[$index]}"
}

for model in $MODELS; do
    printf '%s\t%s\n' "${model}_perturbation.json" "$(expected_for "$model")" \
        >> "$OUT/expected_cells.tsv"
done
sort -o "$OUT/expected_cells.tsv" "$OUT/expected_cells.tsv"

pids=()
for ((i = 0; i < LANES; i++)); do
    [ -n "${queue[i]}" ] || continue
    lane "$i" "${GPU[i % ${#GPU[@]}]}" &
    pids[i]=$!
done
for i in "${!pids[@]}"; do wait "${pids[i]}"; done

if [ -s "$OUT/alerts.log" ]; then
    cat "$OUT/alerts.log"
else
    echo "no alerts"
fi
