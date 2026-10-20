# MiMo 2.6 Tower L28 outlier channel, October 20 2026

The published MiMo 2.6 Pro and MiMo 2.6 Flash perturbation cells at Tower L28
are a readout failure, not a robustness measurement. Their clean accuracy is
0.036 (Pro) and 0.020 (Flash), which is chance for 100 classes. Their perturbed
accuracies and the Tower rows of
[perturbation-mimo26-pro-vs-flash.json](../../bootstrap/perturbation-mimo26-pro-vs-flash.json)
say nothing about robustness. No published file changed. Channel
standardization is reserved for a later protocol version that reruns every
model.

## Cause

Channel 572 of the final Tower layer carries a massive activation. On the
13,000-image recognition caches its RMS is 67,538 for Pro and 94,420 for Flash,
against median channel RMS of 25.0 and 27.7. `probe.Reducer` centers but does
not scale. Its first component is channel 572 (loading 0.999) and holds 99.9%
of the reduced variance on the perturbation training split, with a standard
deviation 101 times the second component's. The clean perturbation features
match the recognition cache bit for bit.

The attention readout therefore sees one direction. Key RMS reaches about 600,
against 0.13 at merged. Attention entropy over the 16 tokens falls to 0.58 to
0.76 nats out of 2.77, and training loss moves only from 4.6 to 4.4. No value
is NaN or infinite.

Escaping takes many optimizer steps. Recognition trains 9,100 images for 720
steps and reaches 0.78; the perturbation run trains 1,400 images for 120 steps.
Cutting recognition to 3 epochs (108 steps) drops validation accuracy to 0.016
to 0.021 for both Towers at 1e-4, 3e-4 and 1e-3. Giving the perturbation split
120 epochs (720 steps) recovers 0.55 at 3e-4. Zeroing the first component alone
gives 0.74 under the perturbation protocol, so the channel carries no class
signal.

## Channel standardization

Scaling every channel to unit variance on the training split before the PCA
fixes the cell. The patch is
`r2:vtb/backups/v3/2026-10-19/proposed/reducer-channel-standardization.patch`
and is not applied. It fails `test_reducer_keeps_a_linearly_separable_signal`,
whose signal is the only high-variance channel among independent noise.

- Perturbation, clean Tower L28: Pro rises from 0.036 to 0.718 and Flash from
  0.020 to 0.726. Merged and projected clean accuracy moves by 1.4 points or
  less. See `mimo26_{pro,flash}_perturbation_standardized.json`.
- Recognition Tower L28 is underestimated. The published cells read 0.7776
  (Pro) and 0.7550 (Flash); standardized, the same split reaches 0.829 and
  0.832 at 1e-4 (seed 0).
- Under the perturbation protocol, the same change moves clean accuracy for
  other models by at most 0.7 points: DINOv2 L24 -0.3, DINOv3 L24 +0.2,
  Nemotron 3.5 Super Tower, merged and projected +0.7, -0.2 and +0.7, and Bolt
  merged and projected +0.6 and -0.6.
- PCA whitening also restores Tower L28, but costs MiMo merged and projected 2
  to 5 points.

## Merged Pro-minus-Flash intervals

The committed merged intervals exclude zero at occlusion L2, occlusion L4, blur
L3 and blur L4, all with Flash ahead. They do not survive retraining the
readout. Rerunning `perturbation_run.py` with unchanged code moves three of the
four across zero; blur L3 survives, and scale L1 newly excludes zero. On that
rerun, Pro merged selected 3e-4 instead of 1e-3. With standardization, no
merged interval excludes zero
([perturbation_standardized_pro_vs_flash_merged.json](perturbation_standardized_pro_vs_flash_merged.json)).
The paired bootstrap resamples images only and leaves out readout-training
variance.

## The v2 Towers

For this check, `vtb.extract` ran on 1,000 ImageNet-100 validation images, the
first ten sorted ids of each class. It used 448 px and 4x4 pooling, and covered
every tapped Tower depth plus merged and projected. Kimi K2.6 and Kimi K3
were extracted one image per batch. A slice is flagged when the first
component's variance share after `fit_reducer` exceeds 0.5, or when its largest
channel RMS exceeds 100 times the median.

At the final Tower layer, the 1%-label cell repeats the published label-budget
protocol on 300 held-out images: 100 training images, 20 epochs and the
published learning rate. The full-label cell trains 700 images at the published
full-label rate, which is only 60 steps. It is a conditioning stress test, not a
replication. The extracts are not bit-identical to the published caches
(relative L2 0.85% for DINOv2 L24 and 1.2% to 1.6% for MiMo L28).

Only DINOv2 has no flagged slice. Each line gives the unpatched value first and
the standardized value second.

- SigLIP2: flagged at L10 to L24 (L10 ratio 173, share 0.93). The final L27 is
  clean (23, 0.09). 1% labels 0.614 vs 0.597; L10 full labels 0.171 vs 0.180.
- Muse Glimmer: flagged at L6 to L38, L50 and merged; final L50 ratio 102,
  share 0.69. 1% labels 0.373 vs 0.632; full labels 0.378 vs 0.740.
- GLM 5.3 Flash: flagged at L3 (ratio 120) and L18 (share 0.63); the final
  layer is clean. 1% labels 0.610 vs 0.592; L18 full labels 0.386 vs 0.391.
- Nemotron Omni: flagged at L32 (ratio 316, share 0.69) and merged. 1% labels
  0.514 vs 0.516; full labels 0.618 vs 0.668.
- Kimi K2.6: L3 to L24 have shares of 0.54 to 0.93 at ratio 10.3 or less, a
  dominant direction that is not channel-aligned. Also flagged at L27 (ratio
  216, share 0.59) and merged. 1% labels 0.283 vs 0.331; full labels 0.411 vs
  0.499; L7 full labels 0.069 vs 0.067.
- Qwen3.8: flagged at L27 (ratio 348, share 0.99), merged and projected. 1%
  labels 0.301 vs 0.426; full labels 0.287 vs 0.559.
- DeepSeek V4.1: flagged at L16 to L32 (ratio 146 to 487, share 0.90 to 0.996).
  1% labels 0.191 vs 0.318; full labels 0.228 vs 0.508.
- Gemma 4: flagged at L3 to L24 (ratio 203 to 1,302); the final L27 has ratio
  91 and share 0.14. 1% labels 0.272 vs 0.282; full labels 0.497 vs 0.459; L3
  full labels 0.046 vs 0.083.
- Kimi K3: flagged at L17 to L24 (share 0.57 to 0.76, ratio 39 or less); the
  final layer is clean. 1% labels 0.183 vs 0.183; L24 full labels 0.208 vs
  0.214.
- MiniMax M3: flagged only at projected (ratio 113, share 0.76). 1% labels
  0.018 vs 0.023. It sits near chance either way, so its low scores have
  another cause.

No v2 Tower collapses at its final layer the way MiMo does; MiMo reads 0.022
and 0.016 at 1% labels here. Muse Glimmer, Qwen3.8, DeepSeek V4.1 and Kimi K2.6
have final-layer outliers, and standardization lifts their 1%-label cells by 5
to 26 points. The full recognition split below confirms the low-label effect
and finds full-label cells within 2 points.

## Label budget on the full recognition split

The final Tower layer of each v2 Tower was re-extracted on all 13,000
ImageNet-100 recognition images: 448 px, 4x4 pooling, the released bundles, and
batch 1 for Kimi K2.6 and Kimi K3. The attention-matched label-budget protocol
then ran twice, once with unchanged code and once with the R2 patch. The
settings match `scripts/label_budget_matrix.sh`: fractions 0.01, 0.05, 0.20
and 1.0, 20 epochs, the full learning-rate search and 3 seeds. The unchanged
run reproduces the published cells with a median difference of 0.2 points. The
largest gap is Qwen3.8 at 1% labels: 0.212 against 0.245 published.

Test accuracy, unchanged then standardized:

| Tower | 1% | 5% | 20% | 100% |
|---|---|---|---|---|
| DINOv2 | 0.371 / 0.377 | 0.750 / 0.751 | 0.862 / 0.863 | 0.908 / 0.908 |
| SigLIP2 | 0.649 / 0.656 | 0.863 / 0.863 | 0.897 / 0.899 | 0.916 / 0.917 |
| Muse Glimmer | 0.432 / 0.703 | 0.842 / 0.878 | 0.898 / 0.907 | 0.922 / 0.918 |
| GLM 5.3 Flash | 0.673 / 0.683 | 0.893 / 0.895 | 0.908 / 0.907 | 0.918 / 0.919 |
| Nemotron Omni | 0.517 / 0.544 | 0.838 / 0.847 | 0.887 / 0.889 | 0.905 / 0.905 |
| Kimi K2.6 | 0.347 / 0.388 | 0.731 / 0.756 | 0.846 / 0.839 | 0.886 / 0.888 |
| Qwen3.8 | 0.212 / 0.463 | 0.596 / 0.791 | 0.858 / 0.853 | 0.879 / 0.878 |
| DeepSeek V4.1 | 0.241 / 0.369 | 0.668 / 0.737 | 0.825 / 0.844 | 0.872 / 0.877 |
| Gemma 4 | 0.325 / 0.329 | 0.691 / 0.692 | 0.828 / 0.833 | 0.876 / 0.876 |
| Kimi K3 | 0.218 / 0.228 | 0.567 / 0.576 | 0.767 / 0.779 | 0.837 / 0.852 |
| MiniMax M3 | 0.052 / 0.052 | 0.069 / 0.067 | 0.148 / 0.153 | 0.342 / 0.324 |

The outlier mainly distorts the low-label end of the curve. At 1% labels,
standardization adds 27 points to Muse Glimmer, 25 to Qwen3.8, 13 to DeepSeek
V4.1, 4 to Kimi K2.6 and 3 to Nemotron Omni. At 5% labels, Qwen3.8 rises from
0.596 to 0.791. By 20% the gaps are 2 points or less, and at full labels every
Tower stays within 2 points (Kimi K3 +1.5, MiniMax M3 -1.8). The published 1%
and 5% cells for those Towers understate what their features support, and they
change the low-label ranking: with standardization, Muse Glimmer at 1% moves
from fourth to first. MiniMax M3 stays near chance at low budgets either
way.

## Qwen3.8 depth

The full-label attention probe ran on Qwen3.8 Tower L20, L24 and L27, matched
and raw, under both arms ([qwen_depth.json](qwen_depth.json)). Test accuracy:

| Arm | L20 | L24 | L27 |
|---|---|---|---|
| Matched, unchanged | 0.8860 | 0.8858 | 0.8795 |
| Matched, standardized | 0.8800 | 0.8884 | 0.8778 |
| Raw, unchanged | 0.8824 | 0.8868 | 0.8745 |

L27 does not become the best layer: it is the lowest of the three in every arm.
The patch touches only the reducer, so the standardized raw arm is identical to
the unchanged one. With full labels, the L27 outlier (ratio 348, share 0.99)
does not cost Qwen3.8 its final-layer accuracy. The decline from L24 to L27
predates the patch and survives it.

## Files

- [channel_statistics.json](channel_statistics.json): per-slice channel and
  component statistics for MiMo and the eleven v2 Towers, the final-layer
  readouts, and the 1,000 image ids.
- [flagged_slice_sensitivity.json](flagged_slice_sensitivity.json): full-label
  readouts at each flagged Tower's most extreme slice.
- `mimo26_{pro,flash}_perturbation_standardized.json`: `perturbation_run.py`
  with the patch applied, on the 13-condition manifest.
- `perturbation_standardized_pro_vs_flash_{tower,merged,projected}.json`:
  `cross_run_bootstrap.py perturbation` intervals on those files.
- `label_budget_{unchanged,standardized}.json`: `vtb.probe_run` outputs, one
  `label-budget/` file per Tower and fraction under `runs`, keyed by the
  published file name.
- [qwen_depth.json](qwen_depth.json): Qwen3.8 matched and raw probes, one
  `recognition/` file per arm and readout under `arms`.
