# MiniMax M3.1 Flash controlled comparison

The full 448-pixel suite and fresh compatible M3 controls are complete and
content-validated. The [per-Stage table](comparison-writeup.md) and
[JSON](per-stage-deltas.json) contain 65 comparisons, including both DIODE arms
and all perturbation conditions. A paper-ready paragraph is included in the
table writeup; this repository contains no manuscript source.

Each checkpoint has 138 primary score cells, four standardized diagnostic cells
and one throughput record. Searches use seed zero, followed by three-seed
retraining at the selected rate, under the existing primary protocols.

| Measurement | Cells per checkpoint | Selected data |
|---|---:|---|
| Recognition, attention/mean, raw/matched | 40 | 13,000 images, ten slices |
| DIODE depth/normals, raw/matched | 40 | 771 images, ten slices |
| Primary label budgets | 8 | Four fractions, two readouts, full recognition split |
| Standardized attention diagnostic | 4 | Four fractions, full recognition split |
| Correspondence | 9 | NAVI555, ScanNet1500, SPair3600 pairs, all three Stages |
| Perturbations | 39 | 2,000 images, thirteen conditions, all three Stages |
| KITTI depth | 2 | 1,000 images, final Tower and projected |
| Throughput | 1 | 768 images, batch eight, two loaders |

The two full inventory reports have no missing files:
[Flash](measurement-validation.json), [M3](measurement-validation-minimax_m3.json).
Their union covers 49 paired payloads: ten within each model and twenty-nine
sibling comparisons. Every interval uses point estimates from the same paired run.
[Paired source validation](paired-source-validation.json) checks correspondence
and perturbation arrays against their source measurements. Native clean-versus-
perturbed intervals and geometry/recognition intervals are also recomputed.
Exact paired test IDs are checked against the split manifest, and recognition
labels against the prepared label map.

The [measurement provenance](measurement-provenance.json) lists all 103 canonical
JSON payloads and preserves two original perturbation outputs before lossless
[identity-row materialization](identity-evidence.md). Historical M3 results stay
in their original directories. The [fresh control scope](m3-image-isolation/README.md)
and explicit figure-source overrides supply this comparison and the default
seventeen-Tower figures. All ten plots are retained as SVG, PDF and PNG under
[figures](figures/manifest.json).

## Source identity and forwards

Flash is `MiniMaxAI/MiniMax-M3.1-Flash` at
`54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e`. M3 is
`MiniMaxAI/MiniMax-M3` at `f0e1c1e04d40177e4673a22097036854f536e9c0`.
[validation.json](validation.json) records every source tensor name, hash,
shape and dtype: 515 Tower tensors and eight Projector tensors per checkpoint.
The export preserves source precision despite LM quantization.

The [image-isolation report](image-isolation.md) documents native attention
across unrelated images and the correction. Both checkpoints match an
independently loaded pinned-parent reference exactly in BF16 at 56 and 448
pixels, including batches one, two and eight, reversed order and changed
neighbors. All 200 batched slice comparisons in twenty cases are exact.
The [real-input check](real-input-cache-parity.json) adds 132 exact image-slice
comparisons from the first, middle and last samples of six primary caches.

The reference uses native HF classes, independent source loading, patch packing
and raster restoration. It is not a second modeling implementation or a full
language-model forward. Source and library hashes are retained. No image-size
field use occurs in the native modeling file. Runtime grids determine RoPE
positions; nominal sizes 2016 and 672 yield equal frequencies and exact forwards
on both diagnostic grids. This supports the training-only comparison at 448.

The immutable release is
[immanuelpeter/MiniMax-M3.1-Flash-Vision](https://huggingface.co/immanuelpeter/MiniMax-M3.1-Flash-Vision/commit/55510a4d372b8ae502a04cc570c0fd06a4361771).
[Independent release reload](hf-current-release-reload.json) verifies file hashes
and exact CPU BF16 Tower and standalone Projector forwards at 56 and 448.
The card documents the single-image constraint.

## Weight drift and input provenance

All 515 Tower and eight Projector tensors are compared. 509 Tower tensors and
all eight Projector tensors differ. [drift-summary.json](drift-summary.json)
groups arithmetic per-tensor statistics. [layer-drift.json](layer-drift.json)
and [weight-drift.png](weight-drift.png) pool float64 dot products and norms
within each of 32 layers. These concatenated statistics differ from arithmetic
per-tensor means; large-norm parameters can dominate. Layer drift is not
monotonic with depth.

[Primary cache manifests](cache-manifests.json) cover all six caches, and the
[Flash](perturbation-cache-audit-minimax_m31_flash.json) and
[M3](perturbation-cache-audit-minimax_m3.json) perturbation audits cover every
one of 312 shards. All thirteen conditions have exact 2,000-image selections,
finite BF16 tensors and file hashes at every Stage. The
[selection manifest](selection-manifest.json) records exact splits and label
budgets, bound to the actual semantic runner hashes.

The [prepared-data audit](prepared-data-audit.json) records parent preparation.
The complete 155,352-entry input hash list is archived as
[prepared-data-sha256.txt.gz](prepared-data-sha256.txt.gz). Its decompressed SHA256
is `255504f8a2ffed47bda8ca945ef0f2b1bbd142a0f4ba95927cfb319b025be8fc`.
All 3,600 SPair IDs are retained. Source visibility independently explains the
192 empty pairs; all 3,408 nonempty pairs have finite metrics at every Stage in
both models. Every excluded ID and reason is retained with the paired finite
intersection and exclusion union.

The [standardization method](standardization-method.md) preserves the recovered
patch, training-only sample statistics and RNG rules. It leaves the primary
Reducer unchanged. The [resume instructions](resume.md), helper hashes,
execution log, checklist and handoff retain the commands and validation scope.
