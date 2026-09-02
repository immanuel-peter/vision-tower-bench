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
itself, a measure of 3D consistency. We will report the three columns separately. ScanNet
recall at 10 pixels and NAVI recall at 2 centimetres carry the geometric conclusion;
SPair-71k PCK is supporting evidence about semantic matching.

All three released evaluation subsets have direct public download paths and require no
login. NAVI is a 31.10 GB public Google Cloud archive. SPair-71k is a 226.96 MB public
archive. The 1,500-pair ScanNet subset is a 1.10 GB public archive mirrored by LoFTR, with
its pair list and intrinsics also public. The full ScanNet dataset has separate terms, but
this run does not need it.

The bench will run all three evaluations at 448 square pixels with adapter-owned
preprocessing. It will score full patch tokens from the final `tower` Stage and every
available `merged` and `projected` Stage. There is no capacity-matched arm because no
trainable readout exists. Cosine normalization removes feature scale, and feature width is
part of the representation being tested rather than probe capacity.

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

## Result: geometric correspondence agrees, semantic correspondence does not

The run completed DINOv2, SigLIP2, MoonViT-V2, and Kimi K2.6 on all three
datasets before the stop condition fired. The other two Projectors were in flight and
their partial files were discarded. Full results are in `results/correspondence/`.

| Tower | Stage | ScanNet recall@10px | NAVI recall@2cm | SPair macro PCK@0.1 |
| --- | --- | ---: | ---: | ---: |
| DINOv2 | `tower` | 0.00748 | 0.53891 | 0.55475 |
| SigLIP2 | `tower` | 0.00449 | 0.40133 | 0.39818 |
| MoonViT-V2 | `tower` | 0.00396 | 0.33342 | 0.27600 |
| MoonViT-V2 | `merged` | 0.00506 | 0.38818 | 0.25745 |
| MoonViT-V2 | `projected` | 0.00510 | 0.39244 | 0.26140 |
| Kimi K2.6 | `tower` | 0.00483 | 0.36458 | 0.17821 |
| Kimi K2.6 | `merged` | 0.00333 | 0.23138 | 0.06944 |
| Kimi K2.6 | `projected` | 0.00535 | 0.42111 | 0.29184 |

The paired Projector intervals are:

| Tower | column | `projected` minus `tower` | paired 95% interval |
| --- | --- | ---: | ---: |
| MoonViT-V2 | ScanNet recall@10px | +0.001139 | [+0.000889, +0.001398] |
| MoonViT-V2 | NAVI recall@2cm | +0.059020 | [+0.054193, +0.063709] |
| MoonViT-V2 | SPair PCK@0.1 | **-0.014355** | **[-0.021642, -0.007021]** |
| Kimi K2.6 | ScanNet recall@10px | +0.000524 | [+0.000278, +0.000769] |
| Kimi K2.6 | NAVI recall@2cm | +0.056532 | [+0.052110, +0.060899] |
| Kimi K2.6 | SPair PCK@0.1 | +0.114964 | [+0.106328, +0.123475] |

Both completed Projectors improve on both geometric correspondence columns. NAVI is the
useful result: absolute recall is substantial, both intervals exclude zero, and every
viewpoint-bin interval is positive. ScanNet also resolves in the Projector's favour, but
absolute recall is below one percent for every Stage. The shared 64 by 64, square-crop
adaptation is floor-limited there and must not be presented as a Probe3D reproduction.

SPair contradicts a roster-wide preservation claim. Kimi K2.6 improves strongly, while
MoonViT-V2 loses 0.01436 PCK and the interval excludes zero. Most of MoonViT-V2's loss
appears at the `tower` to `merged` step: PCK falls from 0.27600 to 0.25745, then the
Projector recovers slightly to 0.26140. Because the direct scorer treats each merged token
as one spatial location, this separates the whole Connector outcome from the learned
Projector imperfectly. It does not make the negative `projected` versus `tower` comparison
go away.

This is exactly the disagreement named by the run's stop condition. The narrower statement
"the two completed Connectors preserve or improve geometric correspondence" is supported.
The broader statement "the Projector preserves correspondence" is not supported across
correspondence types or Projectors. The matrix was halted after 12 of 18 jobs, and
workstreams B through D were not run pending the user's decision about the headline claim.
