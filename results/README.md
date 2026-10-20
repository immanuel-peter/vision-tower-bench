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
| [diagnostics/](diagnostics/) | Pooling checks, independent repeats, extended learning-rate searches, the Bolt Tower check, and the DINOv3 equal-pixel and conditioning checks. Includes failed and inconclusive controls. |
| [manifests/](manifests/) | Original six-Tower matrix completion lists. Run-specific lists remain beside their measurements. |
| [history/roster-run.md](history/roster-run.md) | Historical run narrative, including superseded interpretations and later corrections. |

## Choose the right source

The v1 cohort is DINOv2, SigLIP2, Muse Glimmer, Kimi K2.6, Qwen3.8, and
Kimi K3. The published v2 report adds GLM 5.3 Flash, Nemotron Omni, DeepSeek V4.1,
Gemma 4, and MiniMax M3. The default figure command draws this eleven-Tower
v2 cohort. The six-Tower grouping identifies the scope of older runs and
corrections, not the limits of the current publication.

The v3 cohort adds Nemotron Super (`nemotron35_super`), Nemotron Bolt
(`nemotron35_bolt`), and DINOv3 (`dinov3`). Their recognition, geometry,
correspondence, and perturbation files sit beside the v2 files but are not in
the default figures, and they have no KITTI or label-budget records yet. Only
DINOv3 has bootstrap records. Super has every
Stage. Super's Tower cells precede its final LayerNorm, which runs before the
merge. Bolt has no such LayerNorm and only `merged` and `projected` cells; its
single final Tower cell is the
[Bolt Tower check](diagnostics/nemotron35-bolt-tower/README.md).

DINOv3 is Meta's ViT-L/16, a second self-supervised control beside DINOv2
(issue #4). The adapter loads `facebook/dinov3-vitl16-pretrain-lvd1689m` at
revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`. Its files use 512 px,
where 16 px patches give a 32 by 32 grid, equal to DINOv2 at 448 px. Issue #4
assumed DINOv2 ran at 512 px; it runs at 448 px. Preprocessing matches DINOv2:
bicubic short-side resize and center crop, then ImageNet mean and standard
deviation, instead of the parent processor's bilinear square resize. The
adapter drops CLS and the four registers and probes layers 3 to 24 in steps of
three. Intermediate layers are raw residual streams; only layer 24 has the
learned final norm, as with DINOv2. The perturbation payload has occlusion
only. `vtb.extract`, `scripts/correspondence_run.py`, and
`scripts/perturbation_extract.py` default to 448 px, so pass `--resolution 512`
for DINOv3; `scripts/perturbation_matrix.sh` does this itself. The
[equal-pixel check](diagnostics/dinov3-equal-pixels/README.md) reruns the final
layer at 448 px, and the [conditioning check](diagnostics/dinov3-conditioning/README.md)
compares stream scales with DINOv2. DINOv3 weights use the DINOv3 License, not
this repository's MIT license, and publications using them must acknowledge
Meta's DINO materials.

DINOv3's paired intervals (`bootstrap/*-dinov3-vs-dinov2.json`) compare the
final Tower layers of DINOv3 at 512 px and DINOv2 at 448 px. Recognition and
geometry intervals come from fresh paired reruns with their own point
estimates. Correspondence intervals pair `correspondence/dinov3_correspondence_*.json`
with `correspondence/dinov2_correspondence_*.json`; a fresh DINOv2 rerun
reproduced those per-pair measurements exactly. The occlusion interval pairs
with a fresh DINOv2 occlusion rerun,
`bootstrap/occlusion-dinov3-vs-dinov2.dinov2-rerun.json`. That rerun selected
learning rate 1e-3 where `perturbation/dinov2_perturbation.json` selected 3e-3,
and it scores 3 to 4 points higher at the two heaviest occlusion levels. Read
the occlusion interval with the rerun, not the perturbation file. The fresh
DINOv2 controls loaded `facebook/dinov2-large` at revision
`47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`.

DINOv3 leads on matched recognition, matched depth and normals, and ScanNet.
Its NAVI and SPair intervals include zero, although the DINOv3 paper reports
gains on both. Under occlusion it trails DINOv2 by 3 to 15 points.

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
