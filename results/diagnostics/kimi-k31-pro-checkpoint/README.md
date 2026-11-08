# Kimi K3.1 Pro versus K3

K3.1 Pro and K3 share the published vision architecture, preprocessing and
Stage widths, but all 168 saved vision and Projector tensors differ. This
comparison measures frozen checkpoints at matched depth and Stage. It does
not identify which training phase caused a change or establish an RL-only effect.

The full inventory contains **286/286 primary cells and 102/102 paired comparisons**.
All primary payloads passed independent content checks, and every interval was
replayed exactly from its actual rows. Both original GPU phases are released.
[PR #29](https://github.com/immanuel-peter/vision-tower-bench/pull/29) contains
the implementation, measurements, audits and reproducibility records.

**Production official vLLM TORCH_SDPA BF16 parity failed for both checkpoints.**
Accepted conversion parity uses an independently executed mathematical
reference at the original zero tolerance. Official SDPA FP32 agrees, and
changing attention execution alone reproduces the BF16 disagreement.
Failed diagnostics and original gates remain preserved. Benchmark scores
do not establish production-backend agreement.

## Checkpoints and independent reference

| Component | Immutable source |
| --- | --- |
| Pro parent | `moonshotai/Kimi-K3.1-Pro@6d0a0202956796543364d666ab19c6ad1a494ee1` |
| K3 parent | `moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06` |
| Pro benchmark bundle | `immanuelpeter/MoonViT-K3.1-Pro@6ab95e2daf86e8100be2e712a7a71c0fba26dac4` |
| Fresh K3 control | `immanuelpeter/MoonViT-V2@bcaedfccd1f07d23ab20de8a0a168668c27cacb3` |
| Official vLLM | `1ecc9cc4ce6626352df39e4364579ec376c10761` |

The official Pro-supported loader constructs the Tower and Projector from
`kimi_k25_vit.py`. The harness executes that pinned source with each
checkpoint's own config and vision weights, without an LM or server.
[Reference provenance](reference-provenance.json) records code hashes and
narrow engine substitutions.

Official packed Triton rotary plus independent masked-SDPA math matches all
ten slices on three fixed 448 px images, bit-exact in BF16 for both checkpoints.
Final Tower includes encoder RMSNorm. Positional initialization, merge order
and Projector execution are covered. Still images use T=1, where the derived
temporal embedding is not added. [Controlled GPU evidence](gpu-parity2-summary.json)
separates exact math parity, passing official FP32 and failed production BF16.

Export identity covers exactly 165 Tower and 3 Projector BF16 tensors.
[Immutable reload](release-reload.json) verifies 17 file hashes, all 168 parent
tensors and 30 exact CPU math-reference BF16 cases. Pro retains its pinned
custom license and separate code license. The public K3 key, default loading
behavior and historical measurements are preserved; fresh controls live in
[sibling-control/](sibling-control/).

## Weight drift

All 168 tensors differ, including learned positions, final norm and every
Projector tensor. FP64 accumulated relative L2 uses the K3 norm as denominator.
The concatenated Projector changes by 10.35%, individual Tower blocks by
0.80% to 0.93%, spatial positions by 1.15%, the patch stem by 0.55%, and the
final norm by 0.37%. These measurements do not predict task quality.

[Per-tensor metrics](weight-drift.json), [concatenated groups](weight-drift-aggregated.json)
and the [drift figure](weight-drift-overview.png) retain the full evidence.

## Protocol and feature provenance

[Expected inventory](expected-inventory.json) was written before runs.
Recognition uses 11 seed-0 learning rates, then selected-rate three-seed
retraining for 20 epochs. Geometry uses 6 seed-0 rates, then selected-rate
three-seed retraining for 10 epochs. The primary reducer is unchanged.
The isolated standardized diagnostic follows archived patch SHA256
`f5a2bc1dcda0e6be9a2d6abea9c1bcc9cb5817282ff6406489b5022dc65e330a`.
Only training-token FP32 means and sample standard deviations enter its
scaling before the original PCA.

| Dataset | Full selection | Slices | Grid | Producer, both siblings |
| --- | ---: | ---: | --- | --- |
| ImageNet-100 | 13,000 | 10 | 4x4 pooled | GPU1, BF16, batch 1, 8 CPU threads |
| DIODE | 771 | 10 | Full native | 4 CPU threads, BF16, batch 1 |
| KITTI | 1,000 | Final Tower/projected | Full native | GPU1, BF16, batch 1, 8 CPU threads |
| Perturbation | 13 conditions x 2,000 | 3 per condition | 4x4 pooled | GPU1, BF16, batch 1, 8 CPU threads |

Every shard was reread for IDs, Stage/depth shape, raster order, finite BF16
values and content hashes. CPU/GPU entries occupy separate cache namespaces.
[CPU extraction](cpu-feature-extraction/pair-audit.json) covers 970 shards
per model and 2,562.77 s total. [GPU extraction](gpu-feature-extraction/pair-audit.json)
covers 80,000 image-condition entries, 1,660 shards, 48.18 GB and 2,554.31 s.
Extraction timings are distinct from the prescribed throughput measurement.

Every sibling pair shares actual fit device and thread settings. Completed
4 CPU threads/8 CPU threads recognition fits retain their provenance; sixteen remaining raw
fits ran on GPU0. Geometry reuses CPU-produced features with GPU-trained heads.
The [two-GPU ledger](two-gpu-scheduler/plan.json) assigns unique jobs before
launch, after the active K3 depth matrix finished normally. Each fresh paired
refit trains both arms on one physical GPU. Its point estimates come from
its own rows, never substituted matrix means.

## Recognition and geometry

Final matched-attention accuracy is 84.12% for the Pro Tower, 85.49% for merged
and 84.91% for projected. The paired sibling differences are +0.19pp
[-0.53,+0.91], +1.52pp [+0.65,+2.41] and +0.51pp [-0.17,+1.21]. The within-Pro
projected-minus-Tower interval is +0.79pp [+0.10,+1.49]. All ten slices of
both attention and mean, raw and matched recognition matrices are retained.

The [matched sibling figure](matched-sibling-relative-depth-intervals.png)
shows all eight Tower depths and final merged/projected comparisons for
recognition, depth and normals. Its 30 intervals use their own paired rows.
The [complete comparison report](paired-comparisons.md) gives every one of
the 102 required comparisons, with sample count, sign, units and source hash.
These are individual 95% intervals with 10,000 resamples and seed 0, without
multiplicity adjustment.

The geometry matrix points below describe the prescribed primary fits.
Normal error is in degrees and lower is better. Depth d1 is a percentage.

| Task | Stage | Fresh K3 | Pro |
| --- | --- | ---: | ---: |
| Matched depth | tower | 53.650 | 54.220 |
| Matched depth | merged | 61.510 | 58.260 |
| Matched depth | projected | 61.280 | 62.280 |
| Matched normal | tower | 32.217 | 31.566 |
| Matched normal | merged | 29.395 | 29.004 |
| Matched normal | projected | 29.253 | 28.804 |

All ten full-grid slices are retained in each raw and matched geometry
matrix. See the [raw depth](raw-depth-comparison.png),
[matched depth](matched-depth-comparison.png), [raw normals](raw-normal-comparison.png)
and [matched normals](matched-normal-comparison.png) figures. Head parameter
counts are printed separately for each task and Stage in those figures.
Matched width is 512 for both tasks; depth and normal output heads differ.

The paired matched-depth final Tower and projected intervals favor Pro;
the merged interval crosses zero. Matched-normal paired errors are lower
for Pro at all three final Stages. These statements concern the independent
paired refits. Geometry GPU bilinear backward is nondeterministic, and a
fresh validation search can select a different best rate. In particular,
the Pro merged-depth matrix selected 0.03 and scored 58.26% d1, while its
paired refit selected 0.01 and scored 61.28%. The sibling paired difference
is -0.014pp [-0.630,+0.594], rather than the matrix difference of -3.25pp.
[The actual-rate and point audit](geometry-matrix-versus-paired-points.json)
records both. Image-bootstrap intervals do not measure variation across
training runs and must not be attached to a different matrix point.

## Correspondence, perturbations and transfer

NAVI retains all 555 selected pairs, ScanNet all 1,500 and SPair all 3,600.
Each dataset has all three Stages, three sibling intervals and the within-Pro
projected-minus-Tower interval. NAVI uses paired CPU settings, ScanNet paired
physical GPU0 and SPair paired physical GPU1, with recorded BF16 batch-one
provenance. ScanNet uses the bench adaptation at evaluation side 64 with
1,000 sampled correspondences. It is not a Probe3D reproduction.

| Dataset / metric | Stage | Fresh K3 | Pro |
| --- | --- | ---: | ---: |
| navi / recall_2cm, % | tower | 33.651 | 34.551 |
| navi / recall_2cm, % | merged | 39.318 | 39.628 |
| navi / recall_2cm, % | projected | 39.739 | 40.150 |
| scannet / recall_10px, % | tower | 2.579 | 2.762 |
| scannet / recall_10px, % | merged | 4.712 | 4.919 |
| scannet / recall_10px, % | projected | 4.723 | 4.949 |
| spair / pck_0.1, % | tower | 27.649 | 28.605 |
| spair / pck_0.1, % | merged | 26.038 | 27.485 |
| spair / pck_0.1, % | projected | 26.457 | 27.801 |

[SPair source visibility](spair-visibility-audit.json) independently identifies
192 pairs with zero visible keypoints and 3,408 nonempty pairs. All 3,600
selection IDs remain in both payloads. Every nonempty metric and every
aggregate is finite. The paired finite intersection and exclusion union are
checked against the source, including every excluded ID and reason.
Class-macro SPair intervals appear alongside pair-mean intervals in the full
comparison report.

Both checkpoints have all 13 perturbation conditions at all three Stages,
78 primary cells and 52 paired comparisons, using the same 2,000 IDs and
1,400/300/300 split. [The perturbation figure](perturbation-paired-comparison.png)
retains clean accuracy alongside condition effects. KITTI transfer has both
final Tower and projected cells for each checkpoint and all three prescribed
comparisons. KITTI is a separate driving-scene transfer probe.

Prescribed Tower-only throughput is 45.989 images/s for Pro and 47.228 images/s
for fresh K3. Both measure 768 images at 448, BF16 batch 1, two warmup batches,
two loaders, physical GPU0 and 8 CPU threads. These point measurements do not
carry an interval or establish a meaningful speed ranking.

## Validation and reproduction

[The primary-content audit](independent-content-audits/completed-primary-independent-content-audit.json)
checks actual cell mappings, full sample and split IDs, seeds, exact search
grids, selected rates, finite metrics, dense grids and immutable provenance.
[The paired replay](independent-content-audits/all-paired-intervals-independent-replay.json)
recomputes all required intervals from their actual rows at the recorded
thread settings. The [inventory hash audit](full-experiment-inventory-hash-audit.json)
compares all expected cell IDs and source bytes. Counts alone are insufficient.

[The preservation audit](historical-k3-results-preserved.json) verifies all 36
named K3 result files against their base Git blobs and all 64 original shared
standardized rows and protocol fields. Fresh K3 controls do not replace them.

Run the paired report and figure only after those audits pass:

```bash
python -m scripts.summarize_kimi_checkpoint_pairs
python -m scripts.plot_kimi_checkpoint_pairs
python scripts/figures.py --out results/diagnostics/kimi-k31-pro-checkpoint/final-default-figures --format svg png pdf
VTB_SKIP_WEIGHTS=1 CUDA_VISIBLE_DEVICES="" pytest -q
```

The final CPU regression suite passed 176 tests and skipped
51 weight-dependent cases under `VTB_SKIP_WEIGHTS=1`.
[The test record](final-regression-tests.json) and [full output](final-regression-tests.txt)
retain the command, runtime and source hashes. Independent real-weight identity
and reference-forward evidence remain separate from the offline suite.

## Published inference and ownership

The original published functional gate passed from a clean immutable
`5311e0408a6107f78f8c70bb70f7aa205b94e724` snapshot, using the exact command
`python examples/inference.py /tmp/vtb-delivery/data/imagenet100/validation/validation_000000.jpg`.
A real 500x334 image became 448x448 after bicubic short-edge resizing and a
center crop. CUDA BF16 batch 1 on physical GPU1, logical cuda:0, exited zero
in 7.606 s with Tower `[1,1024,1024]`, merged `[1,256,4096]` and projected
`[1,256,7168]`. [The published proof](inference-smoke/published-example-pass/published-example-gate.json)
retains image and script hashes, preprocessing, immutable revision and GPU
release. The historical missing-example failure is resolved and retained.

A later numerical audit found approximate-tanh GELU in that example. The
bench-side example now uses exact GELU, matching the adapter and pinned
official Projector. On real cached merged tokens, the corrected BF16 output
is bit-exact; the old output differs by maximum absolute 0.0009765625 and
relative L2 0.0002586. CPU preprocessing also matches the adapter bit-exact
in FP32 and BF16. [Correction evidence](inference-smoke/projector-activation-correction/)
keeps the old and corrected numerical results separate.

The corrected published example passed from a new clean HF download at
immutable `d32b4a05dc0f52259334b0c808514b2bb44c3130`. Standalone inference
exited zero in 8.784626 s; a separate all-Stage
comparison exited zero in 8.600946 s. Tower, merged
and projected features are bit-exact with the bench adapter in CUDA BF16
batch 1 on physical GPU0, logical cuda:0, with 2 CPU threads.
[The corrected published proof](inference-smoke/corrected-publication/smoke/summary.json)
records both commands, the real image, actual shapes and immutable revision.
[The GPU release](inference-smoke/corrected-publication/smoke/gpu0-release.json)
verifies idle memory, no CUDA PIDs and lock reacquisition. A post-success CPU
wrapper descriptor-shutdown error is retained separately in
[the cleanup record](inference-smoke/corrected-publication/smoke/wrapper-cleanup.json);
both model checks passed before that cleanup. No numerical gate was loosened.
The original approximate-GELU functional proof remains historical evidence.

HF cards, layout, manifests, licenses and uploads belong to other actors.
This worker preserves the published bundles and provides the narrow example
fix and functional evidence without competing HF changes. The corrected
example changes no adapter, cached feature, benchmark method or result.

This is a matched vision-architecture checkpoint comparison. It does not
attribute changes to RL alone, identify which training stage changed each
weight, or infer full-model coding, finance, CAD or factuality capabilities.
[Historical delivery notes](history-through-b30b023.md) preserve earlier
progress and superseded blockers.
