# Results

Start with the [published v2 report](https://ipeter.dev/blog/vision-tower-bench),
which covers all eleven Towers. This directory contains the v2 measurements,
the original six-Tower v1 records, supporting evidence, corrections, and failed controls.
[index.json](index.json) records each cohort, experiment protocols, and the
source files used for deepest-stage geometry figures. Paths in that index are
relative to this directory. Model IDs inside the JSONs identify the checkpoints;
historical filename slugs `qwen3_5` and `moonvit_v2` mean Qwen3.8 and Kimi K3.

## Find an experiment

| Directory | Contents and status |
|---|---|
| [recognition/](recognition/) | ImageNet-100 layer sweeps, attention and mean readouts, raw and matched capacities. Includes the corrected eleven-rate search. |
| [geometry/](geometry/) | DIODE depth and normals, full layer sweeps in both capacities. Original six-Tower matched values used unseeded PCA; see the correction below. |
| [geometry/deterministic/](geometry/deterministic/) | Corrected matched deepest-stage measurements for the original six Towers. Does not replace the full layer sweeps. |
| [bootstrap/](bootstrap/) | Paired intervals and their per-image or per-pair evidence. Use these files for interval claims, including their own rerun point estimates. |
| [correspondence/](correspondence/) | NAVI, ScanNet, and SPair stage scores and per-pair measurements. |
| [label-budget/](label-budget/) | Recognition at 1%, 5%, 20%, and 100% of training labels; throughput records for the original six Towers. |
| [perturbation/](perturbation/) | Frozen-readout perturbation measurements. All Towers have occlusion; newer payloads also include scale and motion blur. The shared transform manifest describes the original occlusion run. |
| [kitti/](kitti/) | Matched depth transfer measurements on KITTI. Paired intervals are in `bootstrap/`. |
| [diagnostics/](diagnostics/) | Pooling checks, independent repeats, extended learning-rate searches, and the Bolt Tower check. Includes failed and inconclusive controls. |
| [manifests/](manifests/) | Original six-Tower matrix completion lists. Run-specific lists remain beside their measurements. |
| [history/roster-run.md](history/roster-run.md) | Historical run narrative, including superseded interpretations and later corrections. |

## Choose the right source

The v1 cohort is DINOv2, SigLIP2, Muse Glimmer, Kimi K2.6, Qwen3.8, and
Kimi K3. The published v2 report adds GLM 5.3 Flash, Nemotron Omni, DeepSeek V4.1,
Gemma 4, and MiniMax M3. The default figure command draws this eleven-Tower
v2 cohort. The six-Tower grouping identifies the scope of older runs and
corrections, not the limits of the current publication.

The v3 cohort adds Nemotron Super (`nemotron35_super`) and Nemotron Bolt
(`nemotron35_bolt`). Their recognition, geometry, correspondence, and
perturbation files sit beside the v2 files but are not in the default figures,
and they have no KITTI, label-budget, or bootstrap records yet. Super has every
Stage. Super's Tower cells precede its final LayerNorm, which runs before the
merge. Bolt has no such LayerNorm and only `merged` and `projected` cells; its
single final Tower cell is the
[Bolt Tower check](diagnostics/nemotron35-bolt-tower/README.md).

For matched DIODE comparisons at the deepest layer, use
`geometry/deterministic/` for the original six Towers. The other five use
their matrices in `geometry/`. `index.json` lists each source explicitly under
`figure_sources.geometry_stage_levels`, which the figure generator reads.
Bootstrap intervals keep their own paired rerun estimates; do not attach them
to point estimates from a different run.

For geometry layer sweeps and peak selection, retain the complete matrices in
`geometry/`. The deterministic rerun only covers Relative Depth 1.0 and cannot
establish a corrected peak across layers. Original six-Tower matched sweep
magnitudes remain historical. Native patch grids also differ between Towers.

Recognition files contain the extended learning-rate search. Pooling diagnostics
still leave full-token attention comparisons inconclusive. Read the
[pooling diagnostic](diagnostics/pooling-diagnostic/README.md) and
[roster check](diagnostics/pooling-roster/README.md) before interpreting those
controls; the latter preserves both batch-size variants for Kimi.

## Use the files

From the repository root:

```bash
uv sync --group figures
uv run --group figures python scripts/figures.py
uv run python scripts/semantic_table.py results/recognition/*_probe_attention_matched.json
uv run python scripts/geometry_table.py results/geometry/deterministic/*_geometry_depth_matched.json
```

Matrix scripts accept an output directory. Use `results/recognition/` for
`scripts/semantic_matrix.sh` and `results/geometry/` for
`scripts/geometry_matrix.sh` when collecting a new full matrix. Preserve prior
runs in a separate directory when a protocol changes, and update the index with
the correction's scope. JSON, JSONL, and TSV measurements remain in Git so a
clone contains the evidence needed to inspect the figures.
