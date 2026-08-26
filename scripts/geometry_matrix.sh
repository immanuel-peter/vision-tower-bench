#!/usr/bin/env bash
# Run a geometry matrix across GPU lanes, one lane pinned to one GPU.
#
# The matrix is one invocation per (model, task, arm). Lane count is a parameter and the
# models are arguments, so the same script covers the two-model probe and the six-model
# roster run. Cell counts come from the feature cache rather than a table here, because
# they are what balances the lanes and what the completeness check verifies against.
#
# Usage: scripts/geometry_matrix.sh <features_dir> <targets_npz> <out_dir> <model...>
#   model is a cache subdirectory, optionally with a shorter output name:
#   moonvit_v2_448_full:moonvit
#
# Environment: LANES (default: visible GPUs), GPUS, GRID, SEEDS, TASKS, ARMS, EPOCHS.
set -euo pipefail

FEATURES="${1:?features dir}"; shift
TARGETS="${1:?prepared targets npz}"; shift
OUT="${1:?output dir}"; shift
[ "$#" -gt 0 ] || { echo "give at least one model cache directory"; exit 1; }

GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
IFS=, read -ra GPU <<< "$GPUS"
LANES="${LANES:-${#GPU[@]}}"
GRID="${GRID:-3e-4 1e-3 3e-3 1e-2}"   # ADR-0014
SEEDS="${SEEDS:-3}"
TASKS="${TASKS:-depth normal}"
ARMS="${ARMS:-raw matched}"
EPOCHS="${EPOCHS:-10}"

mkdir -p "$OUT"
ALERTS="$OUT/alerts.log"
: > "$ALERTS"

# Anything in a per-invocation log that means the cell did not really run. A lane keeps
# going after one bad cell, so these are collected and reported at the end instead.
FAILURE_PATTERNS='Traceback|CUDA out of memory|RuntimeError|Killed'

cells_in() {
    uv run python -c "
import sys
from vtb import cache
print(len(cache.slices(sys.argv[1])))
" "$1"
}

# One work item per line, as cells, cache directory, output name, task, arm. Cell count
# leads so the balancer can sort on it.
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

# Longest-processing-time-first: hand each invocation to the lane holding the fewest
# cells. With four lanes and eight invocations this puts one wide model and one narrow
# one on every lane, so no lane carries both of the ten-cell models.
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
        echo "[$(date +%T)] start $out_name on gpu $gpu, $cells cells" >> "$log"
        local started rc=0
        started=$SECONDS
        CUDA_VISIBLE_DEVICES="$gpu" uv run python -m vtb.geometry_run \
            --targets "$TARGETS" --device cuda \
            --run "$FEATURES/$dir" --task "$task" "${extra[@]}" \
            --learning-rates $GRID --seeds "$SEEDS" --epochs "$EPOCHS" \
            --out "$OUT/$out_name.json" \
            > "$OUT/$out_name.log" 2>&1 || rc=$?
        # Read $? into rc above before anything else runs. Reporting it after a command
        # substitution such as $(date) prints that command's status instead.
        local elapsed=$((SECONDS - started))
        echo "[$(date +%T)] done $out_name rc=$rc in ${elapsed}s" >> "$log"
        [ "$rc" -eq 0 ] || echo "$out_name exited rc=$rc" >> "$ALERTS"
        grep -nE "$FAILURE_PATTERNS" "$OUT/$out_name.log" \
            | sed "s|^|$out_name |" >> "$ALERTS" || true
    done <<< "${queue[$index]}"
    touch "$OUT/$name.DONE"
}

echo "$LANES lanes over gpus $GPUS, ${#work[@]} invocations, grid [$GRID], $SEEDS seeds"
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
    # A lane that dies before its marker was killed mid-invocation, so its remaining
    # cells never ran and no per-invocation log records the gap.
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
