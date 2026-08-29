# Roster run, August 29 2026

## Outcome

448 cells, all verified, no alerts anywhere. `results/README.md` is rewritten from the
JSONs and the JSONs are copied into `results/`. Headline: MoonViT-V2 is the only one of the
four Projectors that sits below its own lossless yardstick, so the two-model conclusion was
a MoonViT-V2 result rather than a Projector result.

Total wall 07:55 to 18:15, about 10.3 hours at $4.22/hr, roughly $43.


Everything long-running lives in tmux. `tmux ls` to see sessions, `/ephemeral/run/status.sh`
to read their logs with progress bars collapsed, `/ephemeral/run/launch.sh` to start a new one.

Scratch is `/ephemeral`. `$HOME/scratch` is a symlink to it, added this run because the
brief said `$HOME/scratch` and `scripts/brev_setup.sh` had made `/ephemeral`.

## Box

46 vCPUs, not the 88 in the brief. Every `--workers` and thread-cap number below divides 46.
4x L40S 48 GB, 2.3 TB free.

## Done

- Suite: 37 passed, 1 skipped, 6m23s. The skip is `tests/test_moonvit_v2.py`, a module-level
  skip because `data/val2017` does not exist. Re-run with
  `VTB_TEST_IMAGES=/ephemeral/data/diode/val`: 5 passed, including the bit-exact `merged`
  regrouping check the lossless yardstick rests on. No `VTB_SKIP_WEIGHTS` anywhere.
- ImageNet-100 validation: 13,000 images, 100 classes, `/ephemeral/data/imagenet100`.
- DIODE val: 771 images, 325 indoors, 446 outdoor, depth to 299.83 m, normals present.
  Both archives downloaded on the first attempt, no reset. `/ephemeral/data/diode`.

## Gotchas found the hard way

- A new tmux session inherits the tmux *server* environment, not the calling shell's, so
  `GPU=3 launch.sh ...` silently ran on GPU 0. Put env vars inside the command string.
- The first batch sweep overlapped a CPU-heavy pytest run and was thrown out.
- `uv run` queues on `/ephemeral/cache/uv/.lock`, and a long-lived `uv tool uvx markitdown-mcp`
  held it for 18 minutes, so a second lane sat at 0% GPU waiting. `UV_NO_SYNC=1` skips that
  lock. It is exported from `env.sh` and must be set for any multi-lane run.
- Batch-size numbers need a large enough sample. At `--limit 256` the DataLoader worker
  startup sits inside the timed window and the same cell read 24.6 then 7.2 img/s. The
  sweep uses 768 images and runs one model at a time.

## Batch sweep, 768 ImageNet-100 images, 4x4 pooled, one model at a time

| batch | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| qwen3_5 img/s | 14.33 | 24.93 | 40.08 | **42.52** | 36.84 |
| muse_glimmer img/s | 9.93 | 13.04 | **14.79** | 11.70 | |

Neither Tower shows the packed-mask collapse ADR-0009 measured on MoonViT-V2, where
throughput fell from 15.5 at batch 1 to 5.3 at batch 8. Both climb with batch size instead.
qwen3_5 runs at 8, muse_glimmer at 4. Read the muse batch-8 point with care: it overlapped
the start of extraction on the other three GPUs, while batch 4 was measured clean.

## Thread limits determine extraction throughput

The first extraction attempt ran MoonViT-V2 at 1.5 img/s against the 15.5 ADR-0009 measured
on an A100. `geometry_matrix.sh` caps `OMP_NUM_THREADS` and `MKL_NUM_THREADS` per lane and
my extraction driver did not, so Torch took one intra-op thread per core in every process.
Load average hit 90 on 46 cores and the GPUs sat under 20 percent.

With 10 workers and 2 threads per lane, on four lanes:

| model | before | after |
|---|---|---|
| moonvit_v2 | 1.5 | 12.3 img/s |
| kimi_k26 | 1.3 | 16.4 img/s |
| siglip2 | 14.1 | 21.5 img/s |

Load average settles near 13 and GPU utilization reaches 33 to 70 percent. The brief says to
set `--workers` high, but workers and Torch threads compete for the same 46 cores. On this
box, limiting Torch threads improves throughput more than adding workers does.

## Validation done before the long runs

- `scripts/semantic_matrix.sh` smoke-tested on a synthetic 4-cell cache at 2 lanes: correct
  lane split, `expected_cells.tsv`, DONE markers, no alerts, and `verify_outputs.py` clean.
- `roster_summary.py` reproduces the published two-model figures exactly: the 8.2 ratio,
  the 47 sd lossless yardstick, the 1.3 sd Projector step, the ADR-0013 Stage swap, and
  29 of 72 edge-selecting cells. It is trustworthy on the new JSONs.

## Measured cell costs, and what the matrices should take

Timed with four lanes running at once and 11 threads each, so these are matrix conditions
rather than a lone invocation on an idle box.

| cell | seconds |
|---|---|
| geometry `tower`, 32x32 grid, 1024 wide | 540 |
| geometry `merged`, 16x16 grid, 4096 wide | 210 |
| semantic, 1024 wide | 50 |

| pillar | lane-hours | wall on four lanes | cost at $4.22/hr |
|---|---|---|---|
| geometry | 30.7 | 7.7 h | $32.35 |
| semantic | 3.6 | 0.9 h | $3.75 |

The brief's geometry estimate of 28 to 33 lane-hours and 7 to 8 hours wall holds. There was
no prior number for the semantic pillar; 3.6 lane-hours is the first one.

Both timing cells reproduce the published two-model results, so the widened grid has not
moved them: dinov2 `tower` at Relative Depth 0.125 reads d1 0.3191 against 0.3199, and
moonvit `merged` reads 0.5204 against 0.5205.

Run semantics first. It takes under an hour and produces a complete result before the
eight-hour geometry run starts.

## Lane balancing is by cell count, and cost is not uniform per cell

Both lane scripts weigh a job by how many cells it has. Cells are not equal. The raw arm
trains the head at full token width and the matched arm reduces every cell to 512 first, so
on the semantic pillar kimi_k26 measured 114 s/cell raw against 39 s/cell matched, near 3x.

The sort that breaks ties by directory name puts one arm on each lane, so lane0 got every
attention-matched invocation and lane1 every attention-raw one. Wall time is then set by
the raw lanes while the matched lanes finish early and leave their GPU idle. Total work is
unchanged; the run just does not pack.

Worth fixing before the next matrix by weighting a job by cells times an arm factor rather
than by cells alone. Not worth restarting a run that is already going.

The geometry run avoids it without touching the script. `geometry_matrix.sh` already takes
an `ARMS` knob, so running `ARMS=raw` and then `ARMS=matched` as two four-lane phases makes
each phase uniform in cost per cell and lets it pack. Estimated 4.2 h then 2.3 h, against
7.7 h for one phase whose matched lanes idle for the last stretch.

Splitting lanes between the two pillars instead is worse. Two lanes of geometry raw takes
8.4 h on its own, so overlapping it with the tail of the semantic run finishes later than
just waiting for four free GPUs.

## The semantic grid truncates the attention arm almost everywhere

107 of the 112 attention cells select 3e-4, the floor of the eight-point semantic grid, and
none of the 112 mean cells do. Three cells select the 1.0 ceiling, all mean. The split is by
readout, not by capacity arm:

| readout | arm | cells at the 3e-4 floor |
|---|---|---|
| attention | matched | 53 of 56 |
| attention | raw | 54 of 56 |
| mean | matched | 0 of 56 |
| mean | raw | 0 of 56 |

PLAN.md picked `[3e-4 ... 1.0]` to suit the small attention pool. It suits mean pooling
instead. The attention arm is the headline readout, so its absolute levels are reported
below their optimum, which is the same failure ADR-0014 recorded on the geometry side.

How far below is not knowable from these JSONs. `geometry_run` stores a
`learning_rate_search` dict per cell and `probe_run` stores only the selected rate, so there
is no way to see whether an attention cell's validation curve is still climbing at the
floor. Adding that dict to `probe_run` is the cheap fix, and it has to land before any
re-run is worth costing.

Cache directory names carry the resolution and grid, and both lane scripts strip that back
to a model name with the same regex, so `moonvit_v2_448_full` becomes `moonvit_v2`.

    scripts/geometry_matrix.sh /ephemeral/features \
        /ephemeral/data/diode/val_targets.npz /ephemeral/run/geometry \
        dinov2_448_full siglip2_448_full moonvit_v2_448_full \
        kimi_k26_448_full qwen3_5_448_full muse_glimmer_448_full

    scripts/semantic_matrix.sh /ephemeral/features \
        /ephemeral/data/imagenet100/validation_labels.json /ephemeral/run/semantic \
        dinov2_448_pool4 siglip2_448_pool4 moonvit_v2_448_pool4 \
        kimi_k26_448_pool4 qwen3_5_448_pool4 muse_glimmer_448_pool4

Both need `LANES=4` and the `UV_NO_SYNC=1` from `env.sh`. Cells: 8 tower each for dinov2 and
siglip2, 10 for the four Towers with a Projector, so 56 per (task, arm) pair and 224 per
pillar.

## Scripts written this run

- `scripts/semantic_matrix.sh` in the repo, mirroring `geometry_matrix.sh`.
- `/ephemeral/run/batch_sweep.sh` profiles batch size for the packing Towers.
- `/ephemeral/run/extract_all.sh` runs the 12 extractions over 4 lanes from `jobs.txt`.
- `/ephemeral/run/verify_outputs.py` checks a matrix output dir against `expected_cells.tsv`.
- `/ephemeral/run/roster_summary.py` answers the six report questions from the JSONs.
