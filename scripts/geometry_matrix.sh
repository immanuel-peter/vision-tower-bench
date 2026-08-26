#!/bin/bash
# Run the full geometry matrix: two models, two tasks, two capacity arms, 72 cells.
#
# Two lanes share one GPU. Splitting by task rather than by model keeps the lanes even,
# since MoonViT-V2 carries ten cells per arm against DINOv2's eight.
#
# Usage: scripts/geometry_matrix.sh <features_dir> <targets_npz> <out_dir>
set -euo pipefail

FEATURES="${1:?features dir}"
TARGETS="${2:?prepared targets npz}"
OUT="${3:?output dir}"
GRID="${GRID:-3e-4 1e-3 3e-3 1e-2}"   # ADR-0014
SEEDS="${SEEDS:-3}"

mkdir -p "$OUT"

lane() {
  local name="$1"; shift
  for spec in "$@"; do
    IFS=: read -r dir task arm tag <<< "$spec"
    local extra=()
    [ "$arm" = matched ] && extra=(--match-capacity)
    echo "[$(date +%T)] start $tag" >> "$OUT/$name.log"
    local rc=0
    uv run python -m vtb.geometry_run \
      --targets "$TARGETS" --device cuda \
      --run "$FEATURES/$dir" --task "$task" "${extra[@]}" \
      --learning-rates $GRID --seeds "$SEEDS" \
      --out "$OUT/$tag.json" \
      >> "$OUT/$tag.log" 2>&1 || rc=$?
    # Capture the status before any other command runs, or $? reports on that command.
    echo "[$(date +%T)] done $tag rc=$rc" >> "$OUT/$name.log"
  done
  touch "$OUT/$name.DONE"
}

lane A \
  dinov2_448_full:depth:raw:dinov2_depth_raw \
  dinov2_448_full:depth:matched:dinov2_depth_matched \
  moonvit_v2_448_full:normal:raw:moonvit_normal_raw \
  moonvit_v2_448_full:normal:matched:moonvit_normal_matched &

lane B \
  moonvit_v2_448_full:depth:raw:moonvit_depth_raw \
  moonvit_v2_448_full:depth:matched:moonvit_depth_matched \
  dinov2_448_full:normal:raw:dinov2_normal_raw \
  dinov2_448_full:normal:matched:dinov2_normal_matched &

wait
echo "matrix complete $(date)"
