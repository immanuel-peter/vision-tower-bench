# Semantic re-run and pooling validation, August 31 2026

## Outcome

Semantic pillar re-run complete: 224 cells on the eleven-point grid, all verified, no
alerts. The grid truncation ADR-0014 recorded is gone: 4 of 224 cells select an edge,
against 107 of 224 before, and the attention readout gained 0.002 to 0.011 top-1 at the
deepest cells. One ranking change: unmatched attention now reads SigLIP2 over Muse Glimmer.
The Muse Glimmer matched Projector loss the roster recorded dissolves at the extended grid.

Pooling validation (ADR-0005) ran on SigLIP2 and Muse Glimmer, 1,500 images, pooled 4x4
against full patch tokens, 72 cells. Split verdict, recorded in ADR-0019: the mean readout
validates (rankings and curves agree; raw cells identical to four decimals), the attention
readout does not - full-token attention heads collapse at mid Relative Depth on 1,050
training images, where the pooled curves rise smoothly. The pooled cache stays for the mean
readout; the attention readout's cross-model comparison is carried by pooled features whose
control is inconclusive.

Cost: this run. Everything below was measured on the box, not estimated.

## Box

Same as the roster box: 4x L40S 48 GB, 46 vCPUs, 2.3 TB free on `/ephemeral`, $4.22/hr.
`nvidia-smi -L | wc -l` printed 4 before anything ran. No DIODE this run; the MoonViT-V2
tests pointed at the ImageNet images with `VTB_TEST_IMAGES`.

## Timeline (UTC)

| step | start | end | wall |
|---|---|---|---|
| box verify, clones, env | 01:47 | 01:55 | |
| ImageNet-100 export (13,000 images) | 01:50 | 01:55 | ~5 min, download-bound |
| release bundles restored (12 GB) | 01:52 | 02:00 | |
| pytest 57 passed | 01:58 | 02:04 | 6m15s |
| pooled extraction, 6 models, 4 lanes | 02:03:43 | 02:24:45 | 21.0 min |
| semantic matrix, `ARMS=raw` phase | 02:26:38 | 03:08:29 | 41.9 min |
| semantic matrix, `ARMS=matched` phase | 03:08:53 | 03:25:00 | 16.1 min |
| poolcheck extraction (4 grids, 1,500 images) | 03:31:19 | 03:34:46 | 3.5 min |
| poolcheck matrix | 03:42:27 | | |

## Extraction, measured

Same commands as the brief, thread caps inside every command string, four lanes, 10 workers
and 2 Torch threads per lane.

| model | batch | img/s this box | brief (roster box) | wall |
|---|---|---|---|---|
| dinov2 | 16 | 37.9 | ~25 | 5.7 min |
| siglip2 | 16 | 28.7 | 21.5 | 7.2 min |
| moonvit_v2 | 1 | 15.8 | 12.3 | 13.9 min |
| kimi_k26 | 1 | 21.7 | 16.4 | 10.1 min |
| qwen3_5 | 8 | 65.8 | 42.5 | 3.4 min |
| muse_glimmer | 4 | 21.2 | 14.8 | 10.2 min |

All six lanes ran above the roster box's numbers, 15 to 50 percent. The four-lane second
wave (qwen3_5 then siglip2) finished the whole extraction in 21 minutes against the brief's
20-30 estimate. Cache landed at 40 GB against the predicted 42.

## Semantic matrix, eleven points

Run as two four-lane phases, `ARMS=raw` then `ARMS=matched`, per the brief. The two phases
packed correctly: no lane idled while another worked.

| phase | lane-hours | wall | s/cell |
|---|---|---|---|
| raw (112 cells) | 2.70 | 42 min | 86.7 |
| matched (112 cells) | 0.88 | 16 min | 28.4 |

The brief estimated about 5.5 lane-hours for the matrix; measured 3.58. The roster run
measured 114 s/cell raw and 39 s/cell matched on kimi_k26; this box ran kimi attention raw
at 104 s/cell (1039 s for 10 cells) and matched at 35 s/cell, so the box is roughly 1.3x
faster per cell than the roster box at eleven points versus eight.

Two operational notes for the next run:

- `semantic_matrix.sh` truncates the lane logs but not the `.DONE` markers or
  `expected_cells.tsv` when a second phase reuses the same output directory. Reading
  `.DONE` mid-run or `expected_cells.tsv` mid-run verifies the wrong phase. Harmless when
  each phase gets its own output directory; a trap when it does not.
- Cells are cheapest last. `cache.slices` sorts stages alphabetically, so `merged` and
  `projected` come first and the eight cheaper `tower` cells after. The first cell of a raw
  invocation looks 4x slower than the average because it is the widest one.

## Pooling validation, ADR-0005

1,500 images, first 1,500 in sorted order for both grids (`--limit 1500`); pooled and full
caches verified to agree on image ids exactly, per model. Sizes: siglip2 full 27 GB
(ADR predicted 28.3), muse_glimmer full 31 GB against 47.6 predicted. Muse Glimmer's
projected Stage writes 6,656-wide tokens for 1,500 images and that dominates; the ADR's
per-model estimate assumed the roster's 13,000-image Stage mix.

| extraction | img/s | wall |
|---|---|---|
| siglip2 pooled | 29.2 | 56 s |
| siglip2 full | 19.0 | 79 s |
| muse_glimmer pooled | 18.5 | 81 s |
| muse_glimmer full | 11.9 | 126 s |

| poolcheck extraction (4 grids, 1,500 images) | 03:31:19 | 03:34:46 | 3.5 min |
| poolcheck matrix, 4 phases | 03:42:27 | 05:54:14 | 2.20 h |

## Pooling validation, measured

The matrix ran as four uniform phases (full raw, pooled raw, full matched, pooled matched)
after a first attempt that mixed pooled and full caches in one phase put every full-token
invocation on two lanes - the same packing failure the roster run recorded for the
raw/matched arms. Each phase is uniform, so no lane idles.

| phase | wall | lane-hours | s/cell |
|---|---|---|---|
| full raw (36 cells) | 94 min | 4.77 | 473 (sig mean 381, sig attention 457, muse mean 489, muse attention 565) |
| pooled raw (36 cells) | 2.3 min | 0.09 | 27 |
| full matched (36 cells) | 34 min | 2.28 | 206 |
| pooled matched (36 cells) | 0.9 min | 0.04 | 4 |
| total | 2.20 h | 7.31 | |

The brief derived 6 min/cell (360 s) for full-token cells. Measured: 381-457 s/cell on
SigLIP2 and 489-565 s/cell on Muse Glimmer, so 1.06x to 1.57x - within the brief's 2x stop
threshold, and the first completed invocation (siglip2 mean raw, 381 s/cell) was within 6
percent of the estimate. The brief's 7.2 lane-hours estimate for the whole poolcheck matrix
was close: measured 7.7.

Full-token cell costs are per-cell averages over invocations whose first cell is the widest
(`merged`/`projected` come first in `cache.slices` order); the tower cells that dominate the
count are cheaper.

## What the brief got wrong, this run

- Extraction throughputs are roster-box numbers; this box beat all six by 15-50 percent.
  The ranking of batch sizes held exactly.
- The semantic matrix estimate of 5.5 lane-hours was 35 percent high; measured 3.58
  (roster-box per-cell costs scaled to eleven points; this box is faster per cell).
- The full-token cell estimate of 6 min/cell was honest: measured 473 s/cell raw across
  both Towers (siglip2 381-457, muse_glimmer 489-565), within 1.6x, and the matched full
  cells came in under it.
- The brief's per-model full-token cache sizes assumed the roster's Stage mix; Muse
  Glimmer's `projected` Stage dominates at 6,656 tokens wide and 31 GB landed against 47.6
  predicted. SigLIP2 matched its 28.3 GB estimate.
- Nothing else. The two-phase ARMS workaround packed correctly and needed no code change;
  the four-phase poolcheck variant (full/pooled x raw/matched) is the same fix applied to
  the second imbalance the brief's own roster note recorded.

## Scripts written this run

- `/ephemeral/run/extract_poolcheck.sh`, `poolcheck_all.sh` - the pooling-validation driver.
- `/ephemeral/run/semantic_v2_phase1.sh` / `_phase2.sh` - the two matrix phases.
