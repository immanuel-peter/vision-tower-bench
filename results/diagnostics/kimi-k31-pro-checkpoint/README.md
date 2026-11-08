# Kimi K3.1 Pro checkpoint validation

This is an in-progress delivery for issue #24, with 229 of 286 required task cells
validated and 67 completed paired comparisons out of 102. Immutable release reload passed all
17 file hashes, 168 parent tensors and 30 bit-exact CPU BF16 math-reference cases. The immutable bundle
is published at revision 6ab95e2daf86e8100be2e712a7a71c0fba26dac4. Independent
math-reference conversion parity passes in CPU and CUDA BF16. Production
TORCH_SDPA BF16 parity failed for both checkpoints and remains a limitation.
Complete label budgets, standardized diagnostics, mean recognition and primary perturbation measurements
are available below.
The remaining benchmark suite and final default figures are required before
delivery is complete. [PR #29](https://github.com/immanuel-peter/vision-tower-bench/pull/29)
is open for review with these experiments explicitly pending.

The pinned Pro weights differ from K3 in all 168 tensors. The complete
per-tensor cosine and relative L2 measurements are in weight-drift.json.
This is a matched vision-architecture comparison; it does not isolate the
training procedure responsible for the changes.

The concatenated Projector has a 10.35% relative L2 difference from K3,
compared with 0.80% to 0.93% for individual Tower blocks. The learned spatial
positions differ by 1.15%, the patch stem by 0.55%, and the final Tower norm
by 0.37%. These ratios use the K3 norm as denominator and FP64 accumulation.
The [weight drift figure](weight-drift-overview.png) is generated from the
complete measured parameter groups, without inferring task performance.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash -ic '
.venv/bin/python -m scripts.plot_kimi_weight_drift \
  --source results/diagnostics/kimi-k31-pro-checkpoint/weight-drift-aggregated.json \
  --out results/diagnostics/kimi-k31-pro-checkpoint/weight-drift-overview
'
```

## Independent reference

Official vLLM commit `1ecc9cc4ce6626352df39e4364579ec376c10761`
registers `KimiK31ForConditionalGeneration` to its Kimi K3 loader.
That loader constructs the Tower and Projector from
`vllm/model_executor/models/kimi_k25_vit.py`, using Pro configuration
and Pro tensors. reference-provenance.json records source hashes.

The harness executes that upstream source with single-rank PyTorch substitutes
for engine dependencies. Masked three-dimensional SDPA matches the bench's
existing attention backend. Neither the bench attention nor the primary PCA
reducer changes. FP32 and CPU BF16 parity cover three seeded 448px images,
eight Tower depths, merged, and projected. The final Tower includes its
RMSNorm. CPU BF16 masked SDPA is bit-exact.

Four-dimensional fused SDPA and explicit eager attention failed the CPU BF16
diagnostic tolerance. Their logs remain in the issue scratch evidence.
Agreement with production vLLM fused kernels has not been established.

The temporal sincos buffer is initialized by upstream source; still-image
parity uses T=1, where the temporal embedding is not added.
Source vision configs and the two preprocessing implementation files match
between K3 and Pro. compatibility.json records their hashes.

## Reproduce validation

```bash
CUDA_VISIBLE_DEVICES="" HF_HOME=/tmp/vtb-delivery/hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash -ic '
uv run python -m scripts.fetch_kimi_reference --out /tmp/kimi-validation
uv run python -m scripts.export_moonvit_v2 --checkpoint kimi_k31_pro --out /tmp/kimi-validation/bundle
uv run python -m scripts.export_moonvit_v2 --checkpoint moonvit_v2 --out /tmp/kimi-validation/k3-bundle
VTB_KIMI_VALIDATION_ROOT=/tmp/kimi-validation VTB_PARITY_DTYPE=bfloat16 uv run pytest -q tests/test_moonvit_v2_forward_parity.py
'
```

A CUDA run also requires an explicit physical GPU grant and the exclusive
device flock. CPU evidence does not authorize GPU use.

## Planned measurements

expected-inventory.json enumerates both checkpoints, resolutions, grids,
learning-rate searches and seed protocol. The primary suite retains
seed-0 learning-rate selection followed by three selected-rate retrainings.
The separate opt-in standardized reducer follows the archived patch with
training-token FP32 channel means and sample standard deviations before PCA.

The selected correspondence manifests contain 555 NAVI, 1500 ScanNet and
3600 SPair pairs, with exact IDs and dataset roots. Dense and correspondence
features retain full token grids. Semantic extraction uses a 4x4 token grid.

## Preserved failed diagnostics

The first short GPU phase used native rotary and failed BF16 agreement for
all three attention variants. These failures remain in gpu-parity-summary.json.
The second phase executed the official packed rotary source and isolated
attention execution. Its controlled results establish the accepted scope below.
Both phase reports and original acceptance gates are preserved.

## Accepted conversion scope and immutable release

The parent accepted independent mathematical-reference conversion parity after
controlled GPU tests. Official packed Triton rotary and the unchanged upstream
vision code give exact BF16 outputs for both checkpoints under the bench's
masked SDPA math attention. The official TORCH_SDPA wrapper agrees in FP32 but
fails the original BF16 tolerance. Holding packed rotary fixed and changing
only attention reproduces this discrepancy. No failed production test was
reclassified as passed and no bound was widened.

The extracted bundle is published at
[immutable revision 6ab95e2](https://huggingface.co/immanuelpeter/MoonViT-K3.1-Pro/tree/6ab95e2daf86e8100be2e712a7a71c0fba26dac4).
The adapter pins both Tower and Projector to this revision. K3 retains its
existing loading behavior. The full capability suite and final figures remain
required before final delivery. Fresh K3 controls are retained separately;
existing K3 measurement files will not be replaced.

release-reload.json records the immutable reload. fullsuite-plan.json records the remaining suite and reuses the two verified CPU geometry caches. K3 existing release identity is separately audited in k3-existing-release-audit.json.

The final focused preparation checks passed 47 tests, including seed-protocol
regression, tensor-content validation, and source-backed SPair exclusions.
The independent official FP32 and mathematical BF16 reference checks are
separate from the failed production BF16 diagnostics.

## Complete CPU DIODE feature caches

Both siblings now have the entire 771-image DIODE set at all ten full-grid
slices in the separate CPU4/BF16/batch1 namespace. The extraction and final
content audits completed in 42.7 minutes total. Each model has 970 shards.
Every image was checked for Stage shape, finite BF16 values, exact identity
and lossless raster-order merging before its chunk was committed.
The final audit reread every shard and verified all ordered image IDs and
content hashes. [The pair audit](cpu-feature-extraction/pair-audit.json)
confirms identical extraction settings, with no CPU/GPU mixing.

The remaining suite must reuse these caches. Raw and matched geometry probes,
attention recognition, correspondence, transfer and remaining paired
intervals remain pending.

The source configuration and modeling files of the existing pinned K3 release
are byte-identical to the independently validated export; see
k3-existing-code-audit.json. The source parameter drift also has concatenated
per-block cosine and relative L2 metrics in weight-drift-aggregated.json,
rather than relying only on averages of individual tensor metrics.

The CPU extraction uses a separate namespace with identical settings for both
siblings. The full-suite plan now references these completed caches for geometry
probes and paired fits. Mean-recognition results now have complete actual-content
validation, as described below.

## Remaining GPU extraction phase

An explicit exclusive GPU1 extraction reservation follows the completed CPU
DIODE extraction. `scripts.extract_kimi_gpu` pins both immutable releases,
uses BF16 batch size one at 448, and writes a separate GPU namespace.
Recognition retains all ten slices after semantic 4x4 pooling. KITTI retains
full-grid final Tower and projected slices. Each of the 13 perturbation
conditions retains all three final Stages at 4x4 pooling for all 2,000 images.
All full forward slices are checked before pooling, including raster order.
Atomic chunks carry source, hardware, ordered-ID and file hashes for resume.
The final content audit rereads every saved tensor before marking a cache
complete. No task-quality score follows from extraction alone.

The GPU phase completed in 2,554.31 seconds with 80,000 image-condition
entries, 1,660 shards and 48.18 GB of feature payloads. All 30 caches passed
content audits, and all 15 sibling pairs have identical producer settings.
The [pair audit](gpu-feature-extraction/pair-audit.json) retains the complete
checkpoint and hardware provenance. [The release record](gpu-feature-extraction/release.json)
confirms GPU1 had no CUDA processes and zero MiB allocated after extraction,
and that the exclusive lock was released. These are extraction timings,
not the prescribed Tower throughput measurement.

Each checkpoint now has the following verified caches:

| Dataset | Images | Saved slices | Token grid | Producer |
| --- | ---: | ---: | --- | --- |
| ImageNet-100 | 13,000 | 10 | 4×4 pooled | GPU1, BF16, batch 1, 8 threads |
| DIODE | 771 | 10 | Full native grids | CPU, BF16, batch 1, 4 threads |
| KITTI | 1,000 | Final Tower and projected | Full native grids | GPU1, BF16, batch 1, 8 threads |
| Perturbation | 2,000 per condition, 13 conditions | 3 per condition | 4×4 pooled | GPU1, BF16, batch 1, 8 threads |

CPU and GPU features occupy separate namespaces. Both siblings use the same
producer for every paired task; no cache mixes producer devices. Geometry
probes reuse the completed CPU DIODE caches. The remaining attention and geometry
probes, correspondence, remaining paired intervals and final default figures
are still required.

## Complete CPU mean measurements

Both checkpoints have all ten raw and matched mean-recognition cells on the
full 13,000-image set. The 40 cells took 1,786.26 seconds with CPU4 fits and
the same GPU1 feature producer. Eleven learning rates were searched with
seed 0, followed by three selected-rate retrainings. Actual image and split
hashes, seed metrics, parameter counts and finite scores passed validation.
Pro measurements are in the standard recognition directory; fresh K3 controls
are in sibling-control/recognition. Historical K3 files are preserved.

The [mean comparison figure](mean-recognition-comparison.png) covers all eight
Tower depths and final Stage differences. These are descriptive point estimates,
without paired-image intervals or a causal training attribution.

| Final Stage, matched mean | Fresh K3 accuracy | Pro accuracy | Pro − K3, percentage points |
| --- | ---: | ---: | ---: |
| Tower | 83.93% | 84.07% | +0.14 |
| Merged | 84.72% | 83.85% | −0.87 |
| Projected | 84.22% | 83.78% | −0.44 |

Eight mean label-budget cells also passed the full-data protocol audit.
Six were newly fitted at nominal 1%, 5% and 20% label budgets. The two 100%
cells reuse the identical fitted final-Tower cells from mean recognition,
with source hashes and compatibility evidence. Both cases use the same
training, validation and test IDs, train-only PCA, independent per-cell seeds,
LR grid, epochs, CPU4 device settings and three retraining seeds. No new
training was claimed for those two reused cells.

completed-inventory.json maps these 48 mean cells and the 78 perturbation cells
below to the inventory written before experiments. The attention budgets and
compatible reuse below, followed by NAVI, bring remaining work to 118 task cells
and 35 paired comparisons after the complete matched-attention phase below.

## Complete primary perturbation measurements

Both checkpoints completed all 13 conditions at all three final Stages.
The original clean-trained attention readout and train-only PCA were retained,
with eleven learning rates searched at seed 0 and three selected-rate
retrainings. Features contain all 2,000 selected IDs per condition. The
prescribed split is 1,400 training, 300 validation and 300 held-out test images.
Both checkpoints use the same GPU1 feature producer and CPU4 fits.
The full phase took 910.85 seconds. All actual prediction rows, source hashes,
split identities, seed metrics and finite aggregates passed validation.

The 52 prescribed sibling and Pro Projector-minus-Tower comparisons use
the same saved three-seed predictions on the same 300 held-out IDs.
Every interval uses 10,000 paired-image resamples with seed 0. Points were
recomputed from actual rows, rather than taken from separately rounded scores.
The [complete paired comparison figure](perturbation-paired-comparison.png)
shows all 52 intervals. These are per-cell 95% intervals without a
multiple-testing correction.

| Clean, matched attention | Fresh K3 accuracy | Pro accuracy | Pro − K3, pp, 95% paired interval |
| --- | ---: | ---: | ---: |
| Tower | 70.67% | 74.67% | +4.00 [1.22, 6.89] |
| Merged | 72.00% | 74.78% | +2.78 [0.11, 5.56] |
| Projected | 73.78% | 72.33% | −1.44 [−4.00, 1.22] |

These are fixed-readout results on the smaller perturbation split. They do
not isolate intrinsic model robustness or the training cause of checkpoint
changes. Pro raw measurements are in results/perturbation; fresh K3 controls
remain under sibling-control/perturbation, preserving historical K3 results.

An opt-in prediction-retention flag now saves selected-fit class predictions
for audited paired reuse. Tests cover both readout families and verify identical
fit scores and RNG state with retention enabled or disabled. LR selection,
retraining, epochs and the primary reducer do not change.

## Complete attention budgets and standardized diagnostic

All eight primary attention-budget cells and eight standardized diagnostic
cells passed actual-content validation. The unchanged 11-LR seed-0 selection
and three-seed retraining protocol ran on CPU4 for both siblings, using their
complete GPU1 BF16 batch-one feature caches. The full phase took 4,599.05 seconds.

| Nominal label budget | K3 primary | Pro primary | K3 standardized | Pro standardized |
| --- | ---: | ---: | ---: | ---: |
| 1% | 21.81% | 22.85% | 22.96% | 23.40% |
| 5% | 56.96% | 57.16% | 57.54% | 56.92% |
| 20% | 77.42% | 77.30% | 77.23% | 78.02% |
| 100% | 83.93% | 84.12% | 84.70% | 85.11% |

The standardized path follows the pinned archived patch and remains diagnostic.
The primary reducer is unchanged. Pro's four standardized rows were appended
to the default figure source; all 64 historical rows were preserved. The
label-budget figure content test now passes for all 17 Towers. Regenerated
default label-budget and occlusion panels are under ready-default-figures.
Fresh K3 diagnostic controls remain separate from historical K3 results.

The two primary 100% attention fits are also the exact final-Tower matched
recognition cells: identical full data, split, PCA seed, LR grid, selected
retraining seeds, epochs and CPU4 settings. They are retained under
recognition-cells with explicit single-cell scope, not presented as a complete
ten-cell matrix. The recorded per-image predictions and true labels also
supply the prescribed final-Tower sibling bootstrap on all 1,950 test IDs.
Its Pro-minus-K3 point is +0.19 percentage points, with a 95% paired-image
interval of [−0.53, +0.91]. The point and interval come from those same fits.
Full-budget reuse, source hashes and compatibility evidence are recorded in
cpu-attention-budgets/full-budget-reuse.json. Later matrix assembly must retain
each cell's actual fit device and avoid repeating compatible completed fits.

## Complete NAVI correspondence pair

Both pinned checkpoints completed all 555 selected NAVI pairs at all three
Stages. CPU4 BF16 vision inference uses batch size one and full native grids;
the unchanged matcher uses evaluation side 64 and 1,000 correspondences.
All ten native slices, finite BF16 values, shapes and raster merging were
checked on every image before matching, covering 1,110 image forwards per
checkpoint. Every actual pair ID and every nonempty-pair metric passed
validation, with no exclusions. Both producers and matchers use the same CPU4
settings. The full phase took 4,881.03 seconds.

| Stage | Fresh K3 recall within 2cm | Pro recall within 2cm | Pro − K3, pp, 95% paired interval |
| --- | ---: | ---: | ---: |
| Tower | 33.65% | 34.55% | +0.90 [0.68, 1.12] |
| Merged | 39.32% | 39.63% | +0.31 [0.09, 0.54] |
| Projected | 39.74% | 40.15% | +0.41 [0.20, 0.63] |

The fourth required comparison, Pro Projector minus Tower, is +5.60 percentage
points with interval [5.11, 6.08]. All four points were independently checked
against the same 555 raw pair measurements used for their 10,000-resample,
seed-0 bootstrap. Within-Stage intervals retain the original pair-mean protocol.
The [NAVI figure](navi-stage-comparison.png) reports the controlled Stage scores
and sibling intervals. These are per-cell intervals and do not establish a
training cause. Production official TORCH_SDPA BF16 agreement is still not claimed.

Matched-attention recognition ran as complete full-data cells,
with explicit partial-matrix scope, saved test predictions and the unchanged
LR/seed protocol. Both full ten-cell matrices were assembled after every required
cell validated. Compatible completed cells and paired predictions are reused.

The coordinator subsequently raised the CPU budget from four to eight threads
at a natural process boundary. Completed fits retain their original settings;
an unfinished sibling counterpart uses the same thread count as its existing
control. Untouched checkpoint pairs use eight threads sequentially. Every
retained cell records its actual fit threads, and paired reuse requires matching
counts. A combined matrix therefore records per-cell provenance rather than
claiming one thread count for all historical and new fits. LR search, seeds,
epochs, data and the primary reducer remain unchanged.


## Complete matched-attention recognition

Both checkpoints completed all eight Tower depths plus merged and projected
on the full 13,000-image recognition set. Each cell retains its own 11-rate
seed-0 search and three selected-rate retrainings. The two identical 100%
label-budget Tower fits were reused. The remaining 18 cells were fitted across
the CPU4 and CPU8 phases, with no interrupted fits or numeric payload edits.
The CPU8 continuation took 7,526.14 seconds and completed 17 new fits plus
one verified reuse from the preceding safe CPU4 boundary.

Actual thread counts are recorded per cell and match within every sibling
pair. Final Tower and Projected pairs use four threads. Merged and the other
seven Tower-depth pairs use eight. All cells share the verified GPU1 BF16
batch-one frozen-feature producer, image IDs and 9,100/1,950/1,950 split.

| Final Stage, matched attention | Fresh K3 accuracy | Pro accuracy | Pro minus K3, pp, 95% paired interval |
| --- | ---: | ---: | ---: |
| Tower | 83.93% | 84.12% | +0.19 [−0.53, +0.91] |
| Merged | 83.97% | 85.49% | +1.52 [+0.65, +2.41] |
| Projected | 84.39% | 84.91% | +0.51 [−0.17, +1.21] |

All ten sibling intervals and the Pro Projector-minus-Tower interval use the
actual saved three-seed predictions on the same 1,950 test IDs. The latter
is +0.79 percentage points with a 95% interval of [+0.10, +1.49]. All eleven
10,000-resample seed-0 bootstrap files replayed exactly under their recorded
bootstrap thread counts. These are per-cell intervals without multiplicity
adjustment. They support no causal attribution to a particular training phase.

[The content audit](cpu-matched-attention/matched-semantic-complete-audit.json)
records source hashes, full matrices, exact replay and matching thread counts.
[The comparison figure](matched-attention-comparison.png) covers every Tower
depth and final Stage. The default relative-depth semantic panel is regenerated
under ready-default-figures. Historical K3 results are preserved, with fresh
controls under sibling-control/recognition.

The inventory now contains 172 of 286 task cells and 67 of 102 comparisons.
Remaining work is 16 raw-attention recognition cells, 80 geometry cells,
12 ScanNet/SPair cells, four KITTI cells and two throughput cells, plus
35 paired comparisons. Four raw-attention cells completed normally on CPU8 at Tower L3 and L7
for both siblings. The remaining pairs transition to the explicitly reserved
GPU0, with no CPU training overlap and unchanged LR/seed/epoch protocol.
The combined matrices retain each cell's original fit device and protocol. Production official
TORCH_SDPA BF16 parity remains failed and is not reclassified by these scores.


The remaining-suite-preparation directory contains the prepared 43-job GPU
runner and its exact inventory-coverage audit. Its 98 primary cells and
27 paired dense comparisons are pending, with eight correspondence comparisons
to follow on CPU. No CUDA was initialized for this audit. A new explicit grant,
completed CPU training and the exclusive device lock are required for execution.
Actual correspondence selection IDs are rechecked before inference; SPair
exclusions must match each source-confirmed zero-keypoint ID. Prepared scripts
have compiled successfully, but their runtime validations await measurements.

The safe CPU-to-GPU boundary, four retained raw fits and 14 passing transition
checks are archived under cpu-to-gpu-transition. GPU0 is explicitly reserved
for the original remaining suite and held exclusively for the full CUDA process.
No follow-up model-card smoke is included in this reservation.


## Complete raw-attention recognition

Both siblings now have all ten raw-attention recognition cells on the full
13,000 images, completing the entire 80-cell recognition suite. Four CPU8
fits at Tower L3 and L7 were retained at the normal process boundary.
The other sixteen fits ran on GPU0 with eight threads. The sum of new-cell
run and validation times was 1,311.20 seconds. All sibling pairs use the same
fit device and thread count. All features retain the same verified GPU1
BF16 batch-one producer and 4x4 pooling.

Each cell retains its original protocol and saved three-seed test predictions.
The combined matrices explicitly identify the fit device per cell. The
11-rate seed-0 search, three selected-rate retrainings and 20 epochs are
unchanged. The independent
[complete raw-matrix audit](raw-attention-recognition/raw-recognition-complete-audit.json)
checks actual source hashes, labels, predictions, all ten slices and matched
sibling fit devices. Four retained CPU fits were not retrained.

| Final Stage, raw attention | Fresh K3 accuracy | Pro accuracy | Native head parameters |
| --- | ---: | ---: | ---: |
| Tower | 83.56% | 84.14% | 2,152,036 |
| Merged | 84.38% | 84.82% | 5,297,764 |
| Projected | 84.62% | 85.08% | 8,443,492 |

These raw-capacity point estimates use different head sizes between Stages.
The matched-attention section provides the required paired comparisons.
The raw phase completed at 188 of 286 task cells, with 67 of 102 comparisons;
the same locked GPU0 process continues the remaining dense suite.

Further model-card and Hugging Face repository work belongs to other actors.
The previously queued HF follow-up is cancelled, as recorded in
[ownership-boundary.json](ownership-boundary.json). This worker preserves the
published immutable conversion and parity evidence and continues only the
original benchmark, tests, results, figures, writeups and PR delivery.

The first complete Pro raw-depth matrix also passed all ten full-grid cells,
with 771 DIODE images and the authoritative 541/115/115 split. GPU0 continues
the remaining matrices and paired refits sequentially. No matched geometry
or cross-checkpoint interval is claimed from this first matrix alone.


## Complete raw-depth pair

Both checkpoints completed all ten raw-capacity DIODE depth cells, using
771 images and the authoritative 541/115/115 split. The full native grids
come from the same CPU4 BF16 batch-one extraction settings. Both sets of
heads ran on GPU0 with eight threads, six seed-0 learning rates, ten epochs
and three selected-rate retrainings. All actual source hashes, IDs, Stage
shapes, capacities, seed metrics and finite aggregates passed validation.
The independent content audit now covers 208 cells across 74 payload groups.

[The raw-depth comparison](raw-depth-comparison.png) shows all eight Tower
depths and final Stages. Raw head sizes are 1,706,752 parameters for Tower,
3,279,616 for merged and 4,852,480 for projected. These are descriptive
matrix points. Paired refits and matched-capacity geometry remain pending;
no confidence interval is inferred from separately fitted matrix means.

| Final Stage, raw depth | Fresh K3 δ1 | Pro δ1 |
| --- | ---: | ---: |
| Tower | 45.70% | 45.37% |
| Merged | 46.66% | 52.12% |
| Projected | 48.45% | 47.79% |

The current suite continues through matched depth, normals, transfer,
remaining correspondence, throughput and paired refits. The full experiment
inventory is not complete. PR publication additionally waits for execution
of the actual newly published inference.py. The current release lacks that
file; the bench-side candidate passed but is not a published-example pass.

Published inference.py is now verified at immutable 5311e0408a6107f78f8c70bb70f7aa205b94e724. The authorized parent one-file publication and clean CUDA BF16 batch1 pass are recorded in [the resolved functional gate](inference-smoke/resolution.json). All old file hashes and benchmark weights remain unchanged. Earlier missing-example reports are historical failures and no longer block PR publication. The remaining original benchmark suite is still active.

The ten-cell Pro matched-depth matrix is now complete and independently content-validated. Its actual full-layer-sweep source is registered for geometry Stage and Relative Depth figures. The geometry figure test retains all Stage and eight-depth value checks, uses the existing single-axis depth plot and public display labels, and now also checks depth coordinates. Six focused figure tests pass. Seven Projector interval panels remain pending real experiments; their coverage assertions are unchanged.


## Example Projector correction

The fresh published example at5311e040 passed the requested functional CUDA
BF16 smoke with all expected shapes and finite outputs. That proof is preserved.
A later code audit found tanh-approximate GELU in the example, while the benchmark
adapter and pinned official Projector use exact GELU. The bench-side example now
uses exact GELU. On one real-image merged cache it matches the adapter bit-exact
in CPU BF16; the old activation differs by maximum absolute0.0009765625 and
relative L2 0.0002586. Twelve focused regressions pass. This is a separate
example correction; no benchmark feature or fit changes.

[Correction evidence and owner patch](inference-smoke/projector-activation-correction/)
record the one-line change needed at HF examples/inference.py. The corrected
example has not been uploaded by this worker. Fresh published CUDA numerical
verification remains pending owner coordination and a free GPU reservation.
Cards, layout, manifests, licenses and weights remain owned by other actors.
