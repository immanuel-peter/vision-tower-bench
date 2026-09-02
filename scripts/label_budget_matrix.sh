#!/usr/bin/env bash
# Usage: scripts/label_budget_matrix.sh <features_dir> <labels_json> <out_dir> <model[:output_name]...>
set -euo pipefail

FEATURES="${1:?features dir}"; shift
LABELS="${1:?labels json}"; shift
OUT="${1:?output dir}"; shift
[ "$#" -gt 0 ] || { echo "give at least one model cache directory"; exit 1; }

GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
IFS=, read -ra GPU <<< "$GPUS"
LANES="${LANES:-${#GPU[@]}}"
THREADS="${THREADS:-$(( $(nproc) / LANES ))}"
[ "$THREADS" -ge 1 ] || THREADS=1
GRID="${GRID:-1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1}"
SEEDS="${SEEDS:-3}"
EPOCHS="${EPOCHS:-20}"
READOUTS="${READOUTS:-attention mean}"
LABEL_FRACTIONS="${LABEL_FRACTIONS:-0.01 0.05 0.20 1.0}"

mkdir -p "$OUT"
ALERTS="$OUT/alerts.log"
: > "$ALERTS"
FAILURE_PATTERNS='Traceback|CUDA out of memory|RuntimeError|Killed'

deepest_tower() {
    uv run python -c '
import sys
from vtb import cache
print(max(layer for stage, layer in cache.slices(sys.argv[1]) if stage == "tower"))
' "$1"
}

specs=()
for spec in "$@"; do
    dir="${spec%%:*}"
    tag="${spec#*:}"
    [ "$tag" = "$spec" ] && tag=$(sed -E 's/_[0-9]+_(full|pool[0-9]+)$//' <<< "$dir")
    [ -d "$FEATURES/$dir" ] || { echo "no cache at $FEATURES/$dir"; exit 1; }
    layer=$(deepest_tower "$FEATURES/$dir")
    specs+=("$dir"$'\t'"$tag"$'\t'"$layer")
done

: > "$OUT/expected_cells.tsv"
for readout in $READOUTS; do
    for item in "${specs[@]}"; do
        IFS=$'\t' read -r dir tag layer <<< "$item"
        for fraction in $LABEL_FRACTIONS; do
            fraction_tag="f${fraction//./}"
            printf '%s\t1\n' "${tag}_probe_${readout}_matched_${fraction_tag}.json" \
                >> "$OUT/expected_cells.tsv"
        done
    done
done
sort -o "$OUT/expected_cells.tsv" "$OUT/expected_cells.tsv"

run_phase() {
    local readout="$1" i lane_index
    local -a jobs queue pids
    jobs=()
    for item in "${specs[@]}"; do
        IFS=$'\t' read -r dir tag layer <<< "$item"
        for fraction in $LABEL_FRACTIONS; do
            jobs+=("$dir"$'\t'"$tag"$'\t'"$layer"$'\t'"$fraction")
        done
    done
    for ((i = 0; i < LANES; i++)); do queue[i]=""; done
    for i in "${!jobs[@]}"; do
        lane_index=$((i % LANES))
        queue[lane_index]+="${jobs[i]}"$'\n'
    done

    lane() {
        local index="$1" gpu="$2" item dir tag layer fraction fraction_tag out_name rc started elapsed
        while IFS=$'\t' read -r dir tag layer fraction; do
            [ -n "$dir" ] || continue
            fraction_tag="f${fraction//./}"
            out_name="${tag}_probe_${readout}_matched_${fraction_tag}"
            echo "[$(date -u +%T)] start $out_name on gpu $gpu, tower L$layer" \
                >> "$OUT/lane$index.log"
            started=$SECONDS
            rc=0
            CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
            uv run python -m vtb.probe_run \
                --labels "$LABELS" --device cuda --run "$FEATURES/$dir" \
                --readout "$readout" --match-capacity --stages tower --depth-points "$layer" \
                --label-fraction "$fraction" --learning-rates $GRID --seeds "$SEEDS" \
                --epochs "$EPOCHS" --out "$OUT/$out_name.json" \
                > "$OUT/$out_name.log" 2>&1 || rc=$?
            elapsed=$((SECONDS - started))
            echo "[$(date -u +%T)] done $out_name rc=$rc in ${elapsed}s" \
                >> "$OUT/lane$index.log"
            [ "$rc" -eq 0 ] || echo "$out_name exited rc=$rc" >> "$ALERTS"
            grep -nE "$FAILURE_PATTERNS" "$OUT/$out_name.log" \
                | sed "s|^|$out_name |" >> "$ALERTS" || true
        done <<< "${queue[$index]}"
        touch "$OUT/lane${index}_${readout}.DONE"
    }

    echo "starting uniform $readout phase: ${#jobs[@]} one-cell invocations"
    pids=()
    for ((i = 0; i < LANES; i++)); do
        [ -n "${queue[i]}" ] || continue
        lane "$i" "${GPU[i % ${#GPU[@]}]}" &
        pids[i]=$!
    done
    for i in "${!pids[@]}"; do wait "${pids[i]}"; done
}

for ((i = 0; i < LANES; i++)); do : > "$OUT/lane$i.log"; done
for readout in $READOUTS; do run_phase "$readout"; done

echo "$(wc -l < "$OUT/expected_cells.tsv") expected one-cell outputs"
if [ -s "$ALERTS" ]; then
    cat "$ALERTS"
else
    echo "no alerts"
fi
