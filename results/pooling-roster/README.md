# Pooling roster check, September 13 2026

Mean-readout agreement between the pooled (`4x4`) semantic grid and full
patch tokens, per Tower, on the first 1500 ImageNet-100 validation images
(1050 train). Eleven-rate grid `[1e-5 ... 1.0]`, 20 epochs, three seeds.
Deepest-layer cells:

| Tower | arm | pooled | full | difference |
|---|---|---|---|---|
| dinov2 | raw | 0.7763 | 0.7867 | +0.0104 |
| dinov2 | matched | 0.8119 | 0.8044 | -0.0075 |
| moonvit_v2 | raw | 0.6444 | 0.6370 | -0.0074 |
| moonvit_v2 | matched | 0.6904 | 0.7007 | +0.0103 |
| qwen3_5 | raw | 0.4548 | 0.4519 | -0.0029 |
| qwen3_5 | matched | 0.8133 | 0.7941 | -0.0192 |
| kimi_k26 | raw | 0.6844 | 0.6844 | +0.0000 |
| kimi_k26 | matched | 0.7970 | 0.7659 | -0.0311 |

Kimi rows use same-batch-size (`bs1`) extracts on both grids
(`kimi_k26_pool4_bs1_*`). The original `bs4` pooled Kimi cache reads
0.4711/0.6385 and is a batch-size artifact, not a pooling effect: Kimi
extraction is batch-size sensitive, so all Kimi extracts must pin `bs1` as
both Kimi adapter docstrings already require. SigLIP2 and Muse Glimmer
agreement was established earlier under `results/pooling/`.

The mean readout is validated roster-wide. The attention readout's
full-token control remains inconclusive (see
`results/pooling-diagnostic/README.md`); attention Relative Depth shapes
stay exploratory.
