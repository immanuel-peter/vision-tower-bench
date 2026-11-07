# Resume after explicit GPU 0 regrant

Issue 21 is CPU-only while issue 22 owns GPU 0. Do not launch these commands
until the parent explicitly grants GPU 0 back. Every shell uses bash -ic.

The recognition cache at
/tmp/vtb-delivery/issues/21/features/recognition/minimax_m31_flash_448_pool4
is complete and content-validated. Preserve it.

From this worktree, use .venv/bin/python with the per-issue run-suite.py.
--category CATEGORY --list prints the exact commands without running CUDA.
--only SUBSTRING selects a single command within a category. Each command
acquires flock /tmp/vtb-delivery/locks/gpu0.lock for its full lifetime and records
command, UTC start, elapsed time, exit status and source hashes in execution.jsonl.
GPUS=0, LANES=1, THREADS=10, direct CUDA_VISIBLE_DEVICES=0, and at most four
loader workers are fixed. No existing primary LR or seed defaults change.

Recommended order after grant:
1. extract --only extract-diode-minimax_m31_flash
2. correspondence --only minimax_m31_flash
3. bootstrap --only diode
4. recognition
5. standardized
6. label-budget
7. geometry
8. Remaining extract commands for M3 recognition, both KITTI caches and M3 DIODE.
9. Remaining correspondence, perturbation, kitti, throughput and bootstrap jobs.
10. sibling jobs for independent paired M3 controls.

Do not repeat the already completed Flash recognition extraction. Verify the
output content of each command before resuming its dependents. Matrix scripts
write DONE markers even when a child fails, so inspect alerts and actual payloads.

postprocess.py prepares CPU correspondence and perturbation paired intervals
from the actual measurements and combines the four diagnostic label-budget
outputs into a separate standardized payload. It preserves per-image pairing
and points from the same paired run. All model weights and runtime grids are
pinned in validation.json and the release manifests.

Before copying measurements into the standard results directories, verify every
expected cell, LR search count, selected-LR seed count, ID/split hash, checkpoint,
finite metric and bootstrap pairing against expected-inventory.json. Register
the diagnostic payload through figure_sources.label_budget_additional, render
all figures, update per-Stage deltas and replace pending wording. No performance
delta may be inferred from validation or weight drift alone.
