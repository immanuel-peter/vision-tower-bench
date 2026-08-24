# Semantic pillar, first run

ImageNet-100 validation split, 13,000 images, 100 classes. Features cached at 448 square
and pooled to a 4x4 grid (ADR-0005), extracted on an A100 80GB. Each cell trains the
shared readout over an eight-point learning-rate grid, selects on validation, and reports
the mean over three seeds on a held-out test split.

Four arms per model: attention and mean readout, each with and without the PCA capacity
matching from ADR-0008. `matched` trains 1,627,748 parameters in every cell. `raw` trains
2,152,036 at a 1024-wide Stage and 8,443,492 at 7168.

Missing from these numbers: the geometry pillar, which carries the second half of the
headline claim, and the pooling validation ADR-0005 requires.

## What the run shows

The Projector does not lose semantics. Under capacity matching, MoonViT-V2 reads 0.8345
at `tower`, 0.8417 at `merged`, and 0.8412 at `projected`. Without matching the same three
are 0.8373, 0.8390, and 0.8362. The claim in PLAN.md pairs this with a spatial collapse,
and that half is still untested.

`merged` matching `tower` is not evidence about merging. For this Tower the merge is a
lossless regrouping of four Tower tokens into one, proved by `torch.equal` in
`tests/test_moonvit_v2.py`. The informative comparison is `projected` against `merged`.

Capacity matching passes the acceptance test ADR-0008 set for it. Across the attention
arms the largest gap between matched and raw is 0.0283 and the mean gap is 0.0113. Model
ordering is the same in both and both curves rise monotonically with Relative Depth, so
the reduction changes no conclusion.

The attention pool earns its place. It leads mean pooling by 8 to 12 points at shallow and
middle depths and the two converge at the last layer, 0.9026 against 0.8911 for DINOv2.
A single pooled vector would have lost that.

DINOv2 leads MoonViT-V2 at every comparable Relative Depth, 0.9026 against 0.8345 at the
last layer. DINOv2 trained on ImageNet, so read this as one task rather than a ranking.
