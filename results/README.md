# Semantic pillar, first run

This run uses the 13,000-image, 100-class ImageNet-100 validation split. An A100 80GB
extracted features at 448 square and pooled them to a 4x4 grid (ADR-0005). Each cell
searches eight learning rates, selects one on the validation subset, and reports mean
test accuracy over three seeds.

Each model has four arms. Attention and mean readouts both run with and without the PCA
capacity matching from ADR-0008. `matched` trains 1,627,748 parameters in every cell.
`raw` trains 2,152,036 at width 1024 and 8,443,492 at width 7168.

These results do not include geometry or the pooling validation required by ADR-0005.

## What the run shows

The Projector did not reduce semantic accuracy in this run. Under capacity matching,
MoonViT-V2 reads 0.8345 at `tower`, 0.8417 at `merged`, and 0.8412 at `projected`.
Without matching, the same three are 0.8373, 0.8390, and 0.8362. The claim in PLAN.md
pairs this with a spatial collapse, which is still untested.

`merged` matching `tower` does not measure information loss from merging. For this Tower,
the merge is a lossless regrouping of four Tower tokens into one. `torch.equal` verifies
the regrouping in `tests/test_moonvit_v2.py`. The informative comparison is `projected`
against `merged`.

Capacity matching passes ADR-0008's acceptance test. Across the attention arms, the
largest gap between matched and raw is 0.0283 and the mean gap is 0.0113. Model ordering
is the same in both, and both curves rise monotonically with Relative Depth. The reduction
changes no conclusion.

The attention pool leads mean pooling by 8 to 12 points at shallow and middle depths. The
two converge at the last DINOv2 layer, where attention reaches 0.9026 and mean pooling
reaches 0.8911.

DINOv2 leads MoonViT-V2 at every comparable Relative Depth, 0.9026 against 0.8345 at the
last layer. DINOv2 trained on ImageNet, so read this as one task rather than a ranking.
