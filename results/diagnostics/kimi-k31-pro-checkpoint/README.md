# Kimi K3.1 Pro versus K3

K3.1 Pro and K3 share the published vision architecture, preprocessing and
Stage widths, but all 168 saved vision and Projector tensors differ. This
comparison measures frozen checkpoints at matched depth and Stage. It does
not identify which training phase caused a change or establish an RL-only effect.

[PR #29](https://github.com/immanuel-peter/vision-tower-bench/pull/29) is active.
The verified inventory has **266/286 primary cells and 80/102 paired comparisons**.
Recognition, label budgets, standardized diagnostics, all 13 perturbation
conditions, NAVI, all 40 depth cells and Pro throughput are complete. Normals,
SPair, KITTI and dense paired refits are active
or queued. Final audits, figures and regression checks remain required.

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
ten slices on three fixed448 px images, bit-exact in BF16 for both checkpoints.
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

All168 tensors differ, including learned positions, final norm and every
Projector tensor. FP64 accumulated relative L2 uses the K3 norm as denominator.
The concatenated Projector changes by 10.35%, individual Tower blocks by
0.80% to0.93%, spatial positions by 1.15%, the patch stem by 0.55%, and the
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
| Perturbation | 13 conditions x2,000 | 3 per condition | 4x4 pooled | GPU1, BF16, batch 1, 8 CPU threads |

Every shard was reread for IDs, Stage/depth shape, raster order, finite BF16
values and content hashes. CPU/GPU entries occupy separate cache namespaces.
[CPU extraction](cpu-feature-extraction/pair-audit.json) covers 970 shards
per model and 2,562.77 s total. [GPU extraction](gpu-feature-extraction/pair-audit.json)
covers 80,000 image-condition entries,1,660 shards,48.18 GB and 2,554.31 s.
Extraction timings are distinct from the prescribed throughput measurement.

Every sibling pair shares actual fit device and thread settings. Completed
4 CPU threads/8 CPU threads recognition fits retain their provenance; sixteen remaining raw
fits ran on GPU0. Geometry reuses CPU-produced features with GPU-trained heads.
The [two-GPU ledger](two-gpu-scheduler/plan.json) assigns unique jobs before
launch, after the active K3 depth matrix finished normally. Each fresh paired
refit trains both arms on one physical GPU. Its point estimates come from
its own rows, never substituted matrix means.

## Completed recognition and depth

Both models have all 10 slices for mean/attention x raw/matched, totaling 80
recognition cells. All16 primary label-budget and 8 standardized diagnostic
cells are complete. Final matched-attention results are:

| Stage | Fresh K3 accuracy | Pro accuracy | Pro minus K3, pp, paired 95% interval |
| --- | ---: | ---: | ---: |
| Tower | 83.93% | 84.12% | +0.19 [-0.53,+0.91] |
| Merged | 83.97% | 85.49% | +1.52 [+0.65,+2.41] |
| Projected | 84.39% | 84.91% | +0.51 [-0.17,+1.21] |

Pro projected-minus-Tower is +0.79pp [+0.10,+1.49]. All11 semantic comparisons
use actual three-seed predictions on the same 1,950 test IDs and replay
exactly at recorded settings. These are individual intervals without
multiplicity adjustment. [Recognition audit](cpu-matched-attention/matched-semantic-complete-audit.json)
and [full-depth figure](matched-attention-comparison.png) retain the evidence.

Both raw and matched depth matrices have all 10 full-grid slices, with the
authoritative DIODE split 541/115/115. These are matrix points; dense paired
intervals are pending.

| Final Stage | K3 raw d1 | Pro raw d1 | K3 matched d1 | Pro matched d1 |
| --- | ---: | ---: | ---: | ---: |
| Tower | 45.70% | 45.37% | 53.65% | 54.22% |
| Merged | 46.66% | 52.12% | 61.51% | 58.26% |
| Projected | 48.45% | 47.79% | 61.28% | 62.28% |

Raw heads have 1,706,752 Tower parameters,3,279,616 merged and 4,852,480
projected. Matched heads have 1,444,608 parameters. Full eight-depth sweeps
are preserved in standard results and fresh controls. The [raw-depth figure](raw-depth-comparison.png)
shows descriptive points without attaching intervals from another run. The [matched-depth figure](matched-depth-comparison.png) uses the complete fresh sibling matrices.

## Correspondence and perturbations

NAVI is complete for 555 pairs, three Stages and four paired comparisons.
[Its figure](navi-stage-comparison.png) uses the same rows as the 10,000-resample,
seed-0 intervals. ScanNet retains 1,500 selected pairs; SPair retains 3,600.
Runners use exact roots and audit selected IDs before CUDA evaluation.

[SPair visibility audit](spair-visibility-audit.json) individually confirms
192 zero-visible-keypoint pairs and 3,408 nonempty pairs. All3,600 IDs remain.
Only source-confirmed empty pairs may have NaN per-pair PCK. Every nonempty
row and aggregate must be finite. Comparisons record actual finite
intersections and exclusion unions with IDs and reasons.

Both models have 13 perturbation conditions x three Stages, totaling 78 primary
cells and 52 paired comparisons. Actual 2,000 IDs,1,400/300/300 split, transform
hash and condition identities are checked. [The comparison figure](perturbation-comparison.png)
retains clean performance alongside condition effects.

## Published inference and ownership

The parent published only `examples/inference.py` at immutable
`5311e0408a6107f78f8c70bb70f7aa205b94e724`. A fresh snapshot ran the exact
published command on a real 500x334 image processed 448, CUDA BF16 batch 1,
physical GPU1/logical cuda:0, exit0 in 7.606 s. Actual outputs were Tower
`[1,1024,1024]`, merged `[1,256,4096]`, projected `[1,256,7168]`.
[Published functional proof](inference-smoke/published-example-pass/published-example-gate.json)
records command, image hash, preprocessing, revision, device and release.
The historical missing-file failure is explicitly resolved and retained.

A later audit found tanh-approximate GELU in that example, while the adapter
and pinned official Projector use exact GELU. The corrected bench example
matches the adapter bit-exact on one real-image merged cache in CPU BF16.
The old activation differs by maximum absolute0.0009765625 and relative
L2 0.0002586. Twelve regressions pass.
[Correction evidence and one-line owner patch](inference-smoke/projector-activation-correction/)
retain this separate issue. Corrected owner publication and fresh published
CUDA numerical proof are pending. No benchmark cache or fit changes.
The earlier functional shape/finite smoke remains a historical pass.

Cards, HF structure, manifests, licenses, layout and weights belong to other
actors. This worker makes no competing HF upload. The bench-side functional
fix and exact owner handoff are the retained exception.

## Audits and reproduction

[Actual inventory](completed-inventory.json) maps cells to source SHA256.
[Independent audits](independent-content-audits/) recheck actual IDs, splits,
grids, rates, seeds and metrics. After both GPUs release, the CPU continuation
will finish eight correspondence comparisons, replay all 102 required
intervals from actual rows and verify exact 286/102 inventory equality.
Final tests and all default figures remain completion requirements.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash -ic '
uv run python -m scripts.fetch_kimi_reference --out /tmp/kimi-validation
uv run python -m scripts.export_moonvit_v2 --checkpoint kimi_k31_pro --out /tmp/kimi-validation/bundle
uv run python -m scripts.export_moonvit_v2 --checkpoint moonvit_v2 --out /tmp/kimi-validation/k3-bundle
VTB_KIMI_VALIDATION_ROOT=/tmp/kimi-validation VTB_PARITY_DTYPE=bfloat16 uv run pytest -q tests/test_moonvit_v2_forward_parity.py
'
```

CUDA additionally requires an explicit physical GPU reservation and exclusive
full-process lock. [Earlier phase notes](history-through-b30b023.md) preserve
superseded progress counts and historical blockers.


## Newly completed ScanNet and throughput

Both ScanNet runs retain all 1,500 selected IDs with finite Stage metrics.
Both used physical GPU0, BF16 batch 1 and 8 CPU threads. Four intervals replay
exactly from these rows and are staged for the post-GPU inventory registration.
They are additional completed statistics, not included in the current 80
registered comparison count.

| Stage | K3 recall at 10px | Pro recall at 10px | Pro minus K3, pp, paired 95% interval |
| --- | ---: | ---: | ---: |
| Tower | 2.579% | 2.762% | +0.182 [+0.149,+0.216] |
| Merged | 4.712% | 4.919% | +0.207 [+0.154,+0.260] |
| Projected | 4.723% | 4.949% | +0.226 [+0.171,+0.281] |

The within-Pro projected-minus-Tower interval is +2.188pp [+2.066,+2.311].
This is the bench ScanNet adaptation at evaluation side 64 and 1,000 sampled
correspondences, not a Probe3D reproduction. All intervals are individual 95%
intervals with 10,000 resamples and seed 0; there is no multiplicity adjustment.

Prescribed Tower-only throughput is 45.989 images/s for Pro and 47.228 images/s
for fresh K3. Both measure 768 images at 448, BF16 batch 1, two warmup batches,
two loaders, physical GPU0 and 8 CPU threads. Runtime and exact immutable tensor
identity are retained with each source. These point measurements do not carry
an interval or establish a meaningful speed ranking.


## First complete normal matrix and preservation audit

The Pro raw-normal matrix is complete at all ten full-grid slices. All three
seed metrics, IDs, split hashes, six-rate search, Stage shapes and finite
aggregates passed the independent primary-content audit. Final Tower error
is 33.1201 degrees, merged 31.1327 and projected 30.1977. Fresh K3 raw normals and Pro matched normals are now complete. The K3
matched-normal control and remaining paired fits continue. No sibling
interval is inferred from matrix points alone.

[The preservation audit](historical-k3-results-preserved.json) verifies all 36
baseline result paths named for K3 are byte-identical to the requested main
base. All 64 original shared standardized label-budget rows and their protocol
fields are also exact; only four Pro rows were added. Fresh K3 controls stay
separate from historical measurements.


## Complete raw-normal controls and Pro matched normals

Both siblings have complete ten-cell raw-normal matrices at native grids.
The Pro matched-normal ten-cell matrix also passed. The independent primary
content audit now covers 266 cells across 83 payload groups. Both sibling raw
matrices used physical GPU1 and 8 CPU threads, reusing the same CPU4/BF16/batch1
DIODE feature producer. Final-Stage raw normal errors are descriptive points:

| Stage | Fresh K3 error, degrees | Pro error, degrees |
| --- | ---: | ---: |
| Tower | 33.3137 | 33.1201 |
| Merged | 31.4691 | 31.1327 |
| Projected | 30.4294 | 30.1977 |

[The full raw-normal comparison](raw-normal-comparison.png) preserves all
eight Tower depths and final Stages, with actual head parameter counts.
It attaches no interval from another fit. All 11 matched-depth comparisons
have also completed and replayed from their own 115 test-image rows. Normal,
transfer and remaining correspondence comparisons continue.

The exact fixed-image preprocessing check is bit-exact between the published
5311 processor and bench adapter in both FP32 and BF16. This CPU-only check
is [recorded separately](inference-smoke/projector-activation-correction/example-preprocess-equivalence.json)
from the pending corrected-published CUDA proof. A read-only Hub observation
at 04:08:53 found revision 81454a8cbd34425485f16204c4f96019c747fceb still carrying
the old tanh-GELU example hash 77516f1. No worker HF upload occurred.
