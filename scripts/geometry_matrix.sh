#!/usr/bin/env bash
# Usage: scripts/geometry_matrix.sh <features_dir> <targets_npz> <out_dir> <model[:output_name]...>
set -euo pipefail

FEATURES="${1:?features dir}"; shift
TARGETS="${1:?prepared targets npz}"; shift
OUT="${1:?output dir}"; shift
[ "$#" -gt 0 ] || { echo "give at least one model cache directory"; exit 1; }

GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
IFS=, read -ra GPU <<< "$GPUS"
LANES="${LANES:-${#GPU[@]}}"
GRID="${GRID:-1e-4 3e-4 1e-3 3e-3 1e-2 3e-2}"
SEEDS="${SEEDS:-3}"
TASKS="${TASKS:-depth normal}"
ARMS="${ARMS:-raw matched}"
EPOCHS="${EPOCHS:-10}"
FINAL_STAGES="${FINAL_STAGES:-}"

# Split CPU cores across lanes to avoid Torch oversubscription.
THREADS="${THREADS:-$(( $(nproc) / LANES ))}"
[ "$THREADS" -ge 1 ] || THREADS=1

mkdir -p "$OUT"
ALERTS="$OUT/alerts.log"
: > "$ALERTS"

FAILURE_PATTERNS='Traceback|CUDA out of memory|RuntimeError|Killed'

cells_in() {
    uv run python -c "
import sys
from vtb.geometry_run import selected_slices
stages = sys.argv[2].split() or None
print(len(selected_slices(sys.argv[1], stages=stages, deepest_only=bool(stages))))
" "$1" "$FINAL_STAGES"
}

work=()
for spec in "$@"; do
    dir="${spec%%:*}"
    name="${spec#*:}"
    [ "$name" = "$spec" ] && name=$(sed -E 's/_[0-9]+_(full|pool[0-9]+)$//' <<< "$dir")
    [ -d "$FEATURES/$dir" ] || { echo "no cache at $FEATURES/$dir"; exit 1; }
    cells=$(cells_in "$FEATURES/$dir")
    for task in $TASKS; do
        for arm in $ARMS; do
            work+=("$cells	$dir	$name	$task	$arm")
        done
    done
done

# Assign the next largest job to the least-loaded lane.
for ((i = 0; i < LANES; i++)); do load[i]=0; queue[i]=""; done
while IFS= read -r item; do
    best=0
    for ((i = 1; i < LANES; i++)); do
        (( load[i] < load[best] )) && best=$i
    done
    load[best]=$(( load[best] + ${item%%	*} ))
    queue[best]+="$item"$'\n'
done < <(printf '%s\n' "${work[@]}" | sort -t'	' -k1,1nr -k2,2)

    lane() {
    local index="$1" gpu="$2" name="lane$1" item
    local log="$OUT/$name.log"
    : > "$log"
        while IFS='	' read -r cells dir tag task arm; do
        [ -n "$dir" ] || continue
        local out_name="${tag}_geometry_${task}_${arm}"
            local extra=()
            [ "$arm" = matched ] && extra=(--match-capacity)
            local stage_filter=()
            [ -n "$FINAL_STAGES" ] && stage_filter=(--stages $FINAL_STAGES --deepest-only)
        echo "[$(date +%T)] start $out_name on gpu $gpu, $cells cells" >> "$log"
        local started rc=0
        started=$SECONDS
        CUDA_VISIBLE_DEVICES="$gpu" \
        OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
        uv run python -m vtb.geometry_run \
            --targets "$TARGETS" --device cuda \
                --run "$FEATURES/$dir" --task "$task" "${extra[@]}" \
                "${stage_filter[@]}" \
            --learning-rates $GRID --seeds "$SEEDS" --epochs "$EPOCHS" \
            --out "$OUT/$out_name.json" \
            > "$OUT/$out_name.log" 2>&1 || rc=$?
        # Capture the exit code before another command changes $?.
        local elapsed=$((SECONDS - started))
        echo "[$(date +%T)] done $out_name rc=$rc in ${elapsed}s" >> "$log"
        [ "$rc" -eq 0 ] || echo "$out_name exited rc=$rc" >> "$ALERTS"
        grep -nE "$FAILURE_PATTERNS" "$OUT/$out_name.log" \
            | sed "s|^|$out_name |" >> "$ALERTS" || true
    done <<< "${queue[$index]}"
    touch "$OUT/$name.DONE"
}

echo "$LANES lanes over gpus $GPUS, $THREADS threads each, ${#work[@]} invocations, grid [$GRID], $SEEDS seeds"
pids=()
for ((i = 0; i < LANES; i++)); do
    [ -n "${queue[i]}" ] || continue
    printf 'lane%d gpu %s %d cells:\n%s' "$i" "${GPU[i % ${#GPU[@]}]}" "${load[i]}" "${queue[i]}"
    lane "$i" "${GPU[i % ${#GPU[@]}]}" &
    pids[i]=$!
done

for i in "${!pids[@]}"; do
    rc=0
    wait "${pids[i]}" || rc=$?
    # A missing marker means the lane died before finishing its queue.
    if [ ! -f "$OUT/lane$i.DONE" ]; then
        echo "lane$i vanished rc=$rc without writing its DONE marker" >> "$ALERTS"
    fi
done

echo
echo "== expected cells per output =="
for item in "${work[@]}"; do
    IFS='	' read -r cells dir tag task arm <<< "$item"
    printf '%s\t%s\n' "${tag}_geometry_${task}_${arm}.json" "$cells"
done | sort > "$OUT/expected_cells.tsv"
cat "$OUT/expected_cells.tsv"

echo
if [ -s "$ALERTS" ]; then
    echo "== alerts =="
    cat "$ALERTS"
else
    echo "no alerts"
fi
echo "matrix finished $(date)"
