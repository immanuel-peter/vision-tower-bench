#!/usr/bin/env bash
# Usage: scripts/count_matrix.sh <features_dir> <prep_dir> <out_dir> <readouts_dir> <model...>
#
# One lane per GPU. Each lane extracts a model's final Stages on the prepared
# FSC-147 crops, validates the cache, then trains the count readouts. Restarts
# skip validated caches and finished cells; a partial cache is moved aside.
set -euo pipefail

FEATURES="${1:?features dir}"; shift
PREP="${1:?prep dir holding images/ and fsc147_targets.npz}"; shift
OUT="${1:?output dir}"; shift
READOUTS="${1:?readouts dir}"; shift
[ "$#" -gt 0 ] || { echo "give at least one model"; exit 1; }

GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
IFS=, read -ra GPU <<< "$GPUS"
LANES="${LANES:-${#GPU[@]}}"
GRID="${GRID:-1e-4 3e-4 1e-3 3e-3 1e-2 3e-2}"
SEEDS="${SEEDS:-3}"
EPOCHS="${EPOCHS:-10}"
ARMS="${ARMS:-free exemplar}"
CAPACITIES="${CAPACITIES:-raw matched}"
THREADS="${THREADS:-8}"
WORKERS="${WORKERS:-8}"
ROSTER="${ROSTER:-}"
# Space-separated name|images|manifest specs, e.g. "countqa|$VTB_WORK/data/countqa/images|$VTB_WORK/data/countqa/manifest.json".
TRANSFER="${TRANSFER:-}"
TARGETS="$PREP/fsc147_targets.npz"
MANIFEST="$PREP/fsc147_manifest.json"
TOWER_ONLY=" dinov2 dinov3 siglip2 "

mkdir -p "$OUT/logs" "$FEATURES"
ALERTS="$OUT/alerts.log"
touch "$ALERTS"
FAILURE_PATTERNS='Traceback|CUDA out of memory|RuntimeError|Killed|FAILED'

resolution() { [ "$1" = dinov3 ] && echo 512 || echo 448; }
stages() { [[ "$TOWER_ONLY" == *" $1 "* ]] && echo tower || echo "tower merged projected"; }
batch_size() {
    [ -n "$ROSTER" ] && [ -f "$ROSTER" ] || { echo "${EXTRACT_BATCH:-16}"; return; }
    uv run python -c "
import json, sys
entry = json.load(open(sys.argv[1]))['models'].get(sys.argv[2], {})
print(entry.get('extraction', {}).get('batch_size') or sys.argv[3])
" "$ROSTER" "$1" "${EXTRACT_BATCH:-16}"
}
source_of() {
    [ -n "$ROSTER" ] && [ -f "$ROSTER" ] || { echo "unrecorded"; return; }
    uv run python -c "
import json, sys
source = json.load(open(sys.argv[1]))['models'][sys.argv[2]]['source']
print(f\"{source.get('kind')}: {source.get('repo')}@{source.get('revision')}\")
" "$ROSTER" "$1"
}

extract() {
    # extract <model> <gpu> [features dir] [images dir] [manifest] [log tag]
    local model="$1" gpu="$2" features="${3:-$FEATURES}" images="${4:-$PREP/images}"
    local manifest="${5:-$MANIFEST}" tag="${6:-extract}" res run log layer
    res=$(resolution "$model")
    run="$features/${model}_${res}_full"
    log="$OUT/logs/${model}_$tag.log"
    mkdir -p "$features"
    if [ -f "$run/complete.json" ]; then
        echo "[$(date +%T)] $model $tag cache already validated" >> "$OUT/logs/lane_gpu$gpu.log"
        return 0
    fi
    if [ -d "$run" ]; then
        mv "$run" "$run.partial-$(date +%Y%m%dT%H%M%S)"
    fi
    local extra=()
    [ "$model" = nemotron35_bolt ] && extra=(--stages tower merged projected)
    # shellcheck disable=SC2046
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    uv run python -m vtb.extract --model "$model" --images "$images" --out "$features" \
        --resolution "$res" --device cuda --batch-size "$(batch_size "$model")" --workers "$WORKERS" \
        --final-stages $(stages "$model") "${extra[@]}" > "$log" 2>&1 || return 1
    layer=$(uv run python -c "
import sys
from vtb import cache
print(max(layer for _, layer in cache.slices(sys.argv[1])))
" "$run") || return 1
    # shellcheck disable=SC2046
    uv run python scripts/validate_cache.py --run "$run" --manifest "$manifest" --model "$model" \
        --stages $(stages "$model") --layer "$layer" --log "$log" \
        --source "$(source_of "$model")" >> "$log" 2>&1
}

train() {
    local model="$1" gpu="$2" arm="$3" capacity="$4" res run name log rc=0
    res=$(resolution "$model")
    run="$FEATURES/${model}_${res}_full"
    name="${model}_count_${arm}_${capacity}"
    log="$OUT/logs/$name.log"
    local extra=()
    [ "$capacity" = matched ] && extra+=(--match-capacity)
    [ "$arm" = exemplar ] && extra+=(--stages tower --deepest-only)
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    uv run python -m vtb.geometry_run --task count --run "$run" --targets "$TARGETS" \
        --arm "$arm" --name "$model" --readouts "$READOUTS" --device cuda --features-device cuda \
        --learning-rates $GRID --seeds "$SEEDS" --epochs "$EPOCHS" "${extra[@]}" \
        --out "$OUT/$name.json" >> "$log" 2>&1 || rc=$?
    return "$rc"
}

lane() {
    local gpu="$1"; shift
    local log="$OUT/logs/lane_gpu$gpu.log" model started rc
    for model in "$@"; do
        started=$SECONDS rc=0
        echo "[$(date +%T)] extract $model on gpu $gpu" >> "$log"
        extract "$model" "$gpu" || rc=$?
        echo "[$(date +%T)] extract $model rc=$rc in $((SECONDS - started))s" >> "$log"
        if [ "$rc" -ne 0 ]; then
            echo "$model extraction rc=$rc" >> "$ALERTS"
            continue
        fi
        for arm in $ARMS; do
            for capacity in $CAPACITIES; do
                started=$SECONDS rc=0
                echo "[$(date +%T)] start ${model}_count_${arm}_${capacity} on gpu $gpu" >> "$log"
                train "$model" "$gpu" "$arm" "$capacity" || rc=$?
                echo "[$(date +%T)] done ${model}_count_${arm}_${capacity} rc=$rc in $((SECONDS - started))s" >> "$log"
                [ "$rc" -eq 0 ] || echo "${model}_count_${arm}_${capacity} exited rc=$rc" >> "$ALERTS"
            done
        done
        # Frozen-transfer caches: name|images dir|manifest, extracted after the FSC readouts.
        for spec in $TRANSFER; do
            IFS='|' read -r set images manifest <<< "$spec"
            started=$SECONDS rc=0
            extract "$model" "$gpu" "$FEATURES/../$set" "$images" "$manifest" "extract_$set" || rc=$?
            echo "[$(date +%T)] extract $set $model rc=$rc in $((SECONDS - started))s" >> "$log"
            [ "$rc" -eq 0 ] || echo "$model $set extraction rc=$rc" >> "$ALERTS"
        done
        grep -hnE "$FAILURE_PATTERNS" "$OUT/logs/${model}_"*.log | sed "s|^|$model |" >> "$ALERTS" || true
    done
    touch "$OUT/logs/lane_gpu$gpu.DONE"
}

# Larger Stage counts first, alternating lanes by load.
declare -a load queue
for ((i = 0; i < LANES; i++)); do load[i]=0; queue[i]=""; done
for model in $(for m in "$@"; do echo "$(stages "$m" | wc -w) $m"; done | sort -k1,1nr -k2 | cut -d' ' -f2); do
    best=0
    for ((i = 1; i < LANES; i++)); do (( load[i] < load[best] )) && best=$i; done
    load[best]=$(( load[best] + $(stages "$model" | wc -w) + 1 ))
    queue[best]+="$model "
done

echo "$LANES lanes over gpus $GPUS, $THREADS threads each, grid [$GRID], $SEEDS seeds, $EPOCHS epochs"
pids=()
for ((i = 0; i < LANES; i++)); do
    [ -n "${queue[i]}" ] || continue
    echo "gpu ${GPU[i]}: ${queue[i]}"
    # shellcheck disable=SC2086
    lane "${GPU[i]}" ${queue[i]} &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || true; done

{
    for model in "$@"; do
        cells=$(stages "$model" | wc -w)
        for arm in $ARMS; do
            for capacity in $CAPACITIES; do
                [ "$arm" = exemplar ] && n=1 || n=$cells
                printf '%s\t%s\n' "${model}_count_${arm}_${capacity}.json" "$n"
            done
        done
    done
} | sort > "$OUT/expected_cells.tsv"
echo "matrix finished $(date)"
[ -s "$ALERTS" ] && { echo "== alerts =="; cat "$ALERTS"; } || echo "no alerts"
