# Kimi K3.1 Pro checkpoint validation

This is an in-progress delivery for issue #24. The full benchmark suite is pending. Immutable release reload passed all
17 file hashes, 168 parent tensors and 30 bit-exact CPU BF16 math-reference cases. The immutable bundle
is published at revision 6ab95e2daf86e8100be2e712a7a71c0fba26dac4. Independent
math-reference conversion parity passes in CPU and CUDA BF16. Production
TORCH_SDPA BF16 parity failed for both checkpoints and remains a limitation.
No new capability measurements are reported here.

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
required before the delivery PR. Fresh K3 controls will be retained separately;
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

The remaining suite must reuse these caches. It does not yet contain any
new task-quality score. Raw and matched geometry probes, recognition,
correspondence, perturbation, transfer and paired intervals remain pending.

The source configuration and modeling files of the existing pinned K3 release
are byte-identical to the independently validated export; see
k3-existing-code-audit.json. The source parameter drift also has concatenated
per-block cosine and relative L2 metrics in weight-drift-aggregated.json,
rather than relying only on averages of individual tensor metrics.

The CPU extraction uses a separate namespace with identical settings for both
siblings. The full-suite plan now references these completed caches for geometry
probes and paired fits. No new task-quality score has yet been produced.

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
probes reuse the completed CPU DIODE caches. The remaining task-quality
probes, correspondence, paired intervals and final capability figures are
still required.
