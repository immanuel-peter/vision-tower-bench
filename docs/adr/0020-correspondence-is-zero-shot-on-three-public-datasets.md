# Correspondence is zero-shot on three public datasets

Probe3D evaluates correspondence without training a probe. Its three evaluations are:

- ScanNet geometric correspondence on the 1,500-pair SuperGlue test split. The method
  matches dense features with cosine nearest neighbours, ranks 1,000 matches with Lowe's
  ratio test, and reports correspondence recall at 5, 10, and 20 pixels. The paper's
  viewpoint analysis uses recall at 10 pixels in rotation bins `[0, 15)`, `[15, 30)`,
  `[30, 60)`, and `[60, 180)` degrees.
- NAVI geometric correspondence on paired wild-set views of the same object. Pairs have a
  relative rotation in `(0, 120]` degrees. The method uses the same nearest-neighbour and
  ratio-test procedure, then reports 3D recall at 1, 2, and 5 centimetres and 2D recall at
  5, 25, and 50 pixels. The paper's viewpoint analysis uses 2-centimetre recall in
  30-degree bins.
- SPair-71k semantic correspondence on the test split. Each annotated source keypoint is
  sampled from the source feature map and matched to the maximum-cosine location in the
  target feature map. The score is PCK at 10 percent of the target bounding-box scale,
  reported by annotated viewpoint difficulty and averaged over the 18 classes. Probe3D
  samples at most 200 pairs per class with seed 20.

The paper and released repository confirm that the first two are geometric correspondence
tests. SPair-71k is semantic correspondence and Probe3D explicitly warns that it is not, by
itself, a measure of 3D consistency. We report the three columns separately. NAVI is the
multiview 3D consistency result and carries the spatial headline. ScanNet is a second
geometric protocol adaptation. SPair-71k PCK measures semantic part matching across object
instances and does not decide the spatial claim.

All three released evaluation subsets have direct public download paths and require no
login. NAVI is a 31.10 GB public Google Cloud archive. SPair-71k is a 226.96 MB public
archive. The 1,500-pair ScanNet subset is a 1.10 GB public archive mirrored by LoFTR, with
its pair list and intrinsics also public. The full ScanNet dataset has separate terms, but
this run does not need it.

The bench runs all three evaluations at 448 square pixels with adapter-owned preprocessing.
For ScanNet, it first aligns the 1296 by 968 RGB frame to the 640 by 480 depth and
intrinsics frame, then applies the shared square crop. It scores full patch tokens from the
final `tower` Stage and every available `merged` and `projected` Stage. There is no
capacity-matched arm because no trainable readout exists. Cosine normalization removes
feature scale, and feature width is part of the representation being tested rather than
probe capacity.

Probe3D upsamples features onto a quarter-resolution target grid before geometric
matching. That would make the cost of an exact search grow sharply with the Projector's
width. This implementation instead bicubically resizes every full-token Stage map to a
shared 64 by 64 evaluation grid, then keeps Probe3D's 1,000 ratio-ranked matches. The
shared grid prevents a spatially merged Stage from receiving fewer matching candidates,
while retaining much more spatial detail than a pooled semantic cache. This is a declared
protocol adaptation, so the new numbers test Stage differences and are not presented as
reproductions of Probe3D's published rows.

Each result stores a score per image pair. Projector intervals compare `projected` with the
final `tower` Stage by paired bootstrap resampling over image pairs. Positive differences
favour `projected`. A disagreement with the depth and surface-normal result triggers the
run's stop condition before the headline claim changes.

Sources: El Banani et al., "Probing the 3D Awareness of Visual Foundation Models," CVPR
2024, sections 3.2 and A.3.3-A.3.4; and the authors' `mbanani/probe3d` repository.

## Result: NAVI supports the spatial claim; semantic matching splits

The continuation completed all six Towers and four Projectors. Full per-pair results are
in `results/correspondence/`.

| Tower | Stage | ScanNet recall@10px | NAVI recall@2cm | SPair macro PCK@0.1 |
| --- | --- | ---: | ---: | ---: |
| DINOv2 | `tower` | 0.08958 | 0.53891 | 0.55475 |
| SigLIP2 | `tower` | 0.05843 | 0.40133 | 0.39818 |
| MoonViT-V2 | `tower` | 0.02580 | 0.33342 | 0.27600 |
| MoonViT-V2 | `merged` | 0.04709 | 0.38818 | 0.25745 |
| MoonViT-V2 | `projected` | 0.04714 | 0.39244 | 0.26140 |
| Kimi K2.6 | `tower` | 0.05466 | 0.36458 | 0.17821 |
| Kimi K2.6 | `merged` | 0.01618 | 0.23138 | 0.06944 |
| Kimi K2.6 | `projected` | 0.06219 | 0.42111 | 0.29184 |
| Qwen3.8 | `tower` | 0.08337 | 0.37847 | 0.30806 |
| Qwen3.8 | `merged` | 0.00262 | 0.11297 | 0.02772 |
| Qwen3.8 | `projected` | 0.06396 | 0.39731 | 0.33091 |
| Muse Glimmer | `tower` | 0.02075 | 0.21392 | 0.09092 |
| Muse Glimmer | `merged` | 0.00898 | 0.18450 | 0.05641 |
| Muse Glimmer | `projected` | 0.05253 | 0.31970 | 0.22148 |

The paired Projector intervals are:

| Tower | column | `projected` minus `tower` | paired 95% interval |
| --- | --- | ---: | ---: |
| MoonViT-V2 | ScanNet recall@10px | +0.021341 | [+0.020162, +0.022543] |
| MoonViT-V2 | NAVI recall@2cm | +0.059020 | [+0.054193, +0.063709] |
| MoonViT-V2 | SPair PCK@0.1 | -0.014355 | [-0.021642, -0.007021] |
| Kimi K2.6 | ScanNet recall@10px | +0.007535 | [+0.006092, +0.008967] |
| Kimi K2.6 | NAVI recall@2cm | +0.056532 | [+0.052110, +0.060899] |
| Kimi K2.6 | SPair PCK@0.1 | +0.114964 | [+0.106328, +0.123475] |
| Qwen3.8 | ScanNet recall@10px | -0.019414 | [-0.020802, -0.018019] |
| Qwen3.8 | NAVI recall@2cm | +0.018836 | [+0.014590, +0.023038] |
| Qwen3.8 | SPair PCK@0.1 | +0.021649 | [+0.011830, +0.031645] |
| Muse Glimmer | ScanNet recall@10px | +0.031772 | [+0.029930, +0.033612] |
| Muse Glimmer | NAVI recall@2cm | +0.105779 | [+0.099457, +0.112103] |
| Muse Glimmer | SPair PCK@0.1 | +0.132510 | [+0.123934, +0.141108] |

NAVI carries the spatial result. All four Projectors improve recall at 2 centimetres, and
all four paired intervals exclude zero. Three retain that sign in every viewpoint bin.
Qwen3.8's highest-rotation bin crosses zero, but its overall interval remains positive.

The first run's ScanNet values were invalid. RGB used a different pixel frame from depth
and intrinsics, so the square-crop transform moved the principal point to the wrong place.
The old below-one-percent values remain in the completion runbook as an audit record and
are excluded from every claim. The corrected absolute scores are no longer floor-limited.
Three Projectors improve, while Qwen3.8 loses 0.01941 recall and its interval excludes zero.
These are bench-adaptation results, not Probe3D reproduction numbers.

SPair answers a different question. MoonViT-V2 loses 0.01436 PCK for semantic part
matching while gaining 0.05902 on NAVI geometric correspondence. Its semantic loss occurs
mostly from `tower` to `merged`, followed by a partial Projector recovery. The other three
Projectors improve SPair. It is accurate to say that one Connector loses semantic matching
ability while gaining geometric correspondence. That result does not refute the spatial
claim because SPair does not measure multiview 3D consistency.

The Kimi K2.6 raster-order audit found no bug. Its published merge code and the adapter both
emit 2 by 2 blocks in raster order, and the weight-backed test checks the regrouping bit for
bit. Qwen3.8 and Muse Glimmer show similar `merged` drops, so Kimi's anomaly is not unique.
Direct cosine scoring is sensitive to within-block phase after lossless concatenation;
trained probes can learn that layout, but this training-free scorer cannot.
