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
