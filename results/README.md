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
| [bootstrap/](bootstrap/) | Paired intervals and their per-image or per-pair evidence. Use these files for interval claims, including their own rerun point estimates. Geometry refits are not bit-reproducible, because the GPU backward of bilinear interpolation is nondeterministic. A paired refit can therefore differ slightly from the matrix cell it repeats. |
| [correspondence/](correspondence/) | NAVI, ScanNet, and SPair stage scores and per-pair measurements. |
| [label-budget/](label-budget/) | Recognition at 1%, 5%, 20%, and 100% of training labels; throughput records for the original six Towers. |
| [perturbation/](perturbation/) | Frozen-readout perturbation measurements. All Towers have occlusion; newer payloads also include scale and motion blur. The shared transform manifest describes the original occlusion run. |
| [kitti/](kitti/) | Matched depth transfer measurements on KITTI. Paired intervals are in `bootstrap/`. |
| [counting/](counting/) | Class-agnostic counting (issues #8, #10): FSC-147 count readouts on every final Stage for all 16 Towers, with paired intervals, per-bin tables and a letterbox transfer check; plus an experimental, supplementary CountQA-derived transfer column. See [counting/README.md](counting/README.md). |
| [diagnostics/](diagnostics/) | Pooling checks, independent repeats, extended learning-rate searches, the Bolt Tower check, the DINOv3 equal-pixel and conditioning checks, and the MiMo attention sink check. Includes failed and inconclusive controls. |
| [manifests/](manifests/) | Original six-Tower matrix completion lists. Run-specific lists remain beside their measurements. |
| [history/roster-run.md](history/roster-run.md) | Historical run narrative, including superseded interpretations and later corrections. |

## Choose the right source

The v1 cohort is DINOv2, SigLIP2, Muse Glimmer, Kimi K2.6, Qwen3.8, and
Kimi K3. The published v2 report adds GLM 5.3 Flash, Nemotron Omni, DeepSeek V4.1,
Gemma 4, and MiniMax M3. The default figure command draws this eleven-Tower
v2 cohort. The six-Tower grouping identifies the scope of older runs and
corrections, not the limits of the current publication.

The v3 cohort adds Nemotron Super (`nemotron35_super`), Nemotron Bolt
(`nemotron35_bolt`), DINOv3 (`dinov3`), MiMo 2.6 Pro (`mimo26_pro`), and MiMo
2.6 Flash (`mimo26_flash`). Their recognition, geometry,
correspondence, and perturbation files sit beside the v2 files but are not in
the default figures, and they have no KITTI or label-budget records yet. Only
DINOv3 and the MiMo pair have bootstrap records. Super has every
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

MiMo 2.6 Pro and MiMo 2.6 Flash are the vision Towers of Xiaomi's MiMo V2.6 Pro
and Flash (issue #5), loaded from `XiaomiMiMo/MiMo-V2.6-Pro-RL` at revision
`b61f21852680ee02a29e946e18cf8697f4844061` and `XiaomiMiMo/MiMo-V2.6-Flash-RL`
at `dcd3fd83fb181622fbc606fbb1709901982a62c2`. They share the architecture, 28
blocks of width 1280, but not the weights, and their Projectors end at 6144 and
4096. Both run at 448 px with every Stage. The common bicubic short-side resize
and center crop replace the parent's dynamic resizing, normalization uses the
parent's ImageNet mean and standard deviation, and each still image enters as
two identical frames. Resolutions must be multiples of 32. Tower depths are 4,
7, 10, 14, 18, 21, 24, and 28. Some windowed blocks reorder the 2x2 merge blocks
column-major, so the adapter restores every captured depth to raster order.
`merged` is the normalized 5120-wide input to the merger MLP. The weights are
MIT. The unmodified parent vision class cannot strictly load either checkpoint:
it creates three merger biases the checkpoint lacks. The releases omit them, and
the parity reference in `scripts/validate_mimo26.py` sets only those biases to
zero. The [Pro](../hf/MiMoViT-2.6-Pro/NOTICE) and
[Flash](../hf/MiMoViT-2.6-Flash/NOTICE) notices record the change. The [attention sink check](diagnostics/mimo26-attention-sinks/README.md)
compares the Hugging Face sink handling with vLLM.

The MiMo intervals (`bootstrap/*-mimo26-pro-vs-flash*.json`) give Pro minus
Flash at the same Stage. Recognition and geometry files are fresh paired reruns
of one Stage, named by a Stage suffix, with their own point estimates. The
recognition reruns match `semantic_bootstrap.py` with `--first-stage` and
`--second-stage`; the geometry reruns used `geometry_bootstrap.py`. The
correspondence and perturbation files hold all three Stages and pair the
`correspondence/mimo26_*` and `perturbation/mimo26_*` files directly, listing
them with their SHA256. `cross_run_bootstrap.py` reproduces every one of those
intervals to within 4e-9, including the SPair class-macro PCK and the
clean-accuracy intervals.

Pro leads Flash on matched-attention recognition at the Tower, by 2.3 points;
the merged and projected intervals include zero. Flash leads on matched depth
and normals at merged and projected. At the Tower, Pro's normal error is 1.2
degrees lower and the depth interval includes zero. Flash has higher NAVI and
ScanNet recall at every Stage and higher SPair PCK at the Tower; its other SPair
intervals include zero. The perturbation Tower readouts are near chance on
clean images, 3.6% for Pro and 2.0% for Flash. Merged and projected readouts
reach 69% to 70%, and the final Tower reaches 77.8% (Pro) and 75.5% (Flash) in
the recognition sweep. The clean perturbation features match the recognition
cache for the same images bit for bit. The cause is not established, so do not
read Tower robustness from these files.

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

## Counting

[counting/](counting/) holds protocol count-v4 for the `counting_sixteen` cohort, the
union of v2 and v3. FSC-147 is the primary counting experiment. It uses the official
class-disjoint split, and all 116 cells are complete. Counts are sums of a 64 x 64
density whose Gaussians use a sigma of one grid cell, 6 source px. Unlike the DIODE
geometry inputs, which are downsampled, FSC-147 crops are upsampled: 384 px goes to
448 px (1.1667x), or to 512 px for DINOv3 (1.3333x).

- Paired intervals: `counting/bootstrap/fsc147/`
- Per-bin tables: `counting/tables/`
- Letterbox transfer diagnostic: `counting/fsc147/letterbox/`

[counting/countqa/](counting/countqa/) is **experimental and supplementary**. Frozen
FSC-147 heads are applied to a curated CountQA-derived subset. These are not official
CountQA benchmark scores and are not comparable to MLLM results. Its conditioned arm
uses AI-authored exemplar boxes that no human has reviewed, under
`countqa/ai_exemplars/`. No human-approved exemplars exist.

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
