# Label budgets change only the semantic training split

Hypothesis 3 names label efficiency alongside token count and latency. A label-budget run
must therefore change only the number of labelled training examples. Validation and test
images must remain identical or the cross-budget ranking is not comparable.

`vtb.probe_run --label-fraction` starts from the existing fixed-seed 70/15/15 split and
subsamples only `split.train`, at seed 0, in proportion to class frequency. It keeps an
exact rounded global count when that count can cover every represented class. When it
cannot, it keeps one example per class. On the 13,000-image ImageNet-100 validation export,
the full training split contains 9,100 images. A literal one-percent count would be 91 and
could not cover 100 classes, so the one-percent condition retains 100 images, an effective
1.10 percent. The output records requested fraction, actual training count, and full
training count.

The capacity-matching reducer is fit only on the budgeted training indices, as is the
readout. This follows the existing meaning of `split.train` and prevents unlabelled examples
from silently entering one part of the training path.

`scripts/label_budget_matrix.sh` restricts each cache to its deepest `tower` Stage, the
matched arm, and both attention and mean readouts. It runs uniform readout phases over
fractions 0.01, 0.05, 0.20, and 1.0. Six Towers therefore produce exactly 48 one-cell
outputs. Tests verify deterministic selection, complete class coverage at the smallest
budget, proportional allocation above that minimum, and unchanged validation and test
indices.

## Result

All 48 cells completed. The smallest condition retains 100 training images; the other
conditions retain 455, 1,820, and 9,100. Attention changes leaders from SigLIP2 at one and
five percent to Muse Glimmer at 20 and 100 percent. Mean keeps SigLIP2 first at every
budget, but its second and third rows change from DINOv2 then Muse Glimmer at one percent
to Muse Glimmer then DINOv2 thereafter. The ranking is therefore not identical across
budgets or readouts.

At one percent, attention accuracy ranks SigLIP2 0.6559, Muse Glimmer 0.4330, DINOv2
0.3689, Kimi K2.6 0.3448, Qwen3.8 0.2451, and MoonViT-V2 0.2191. At 100 percent it ranks
Muse Glimmer 0.9212, SigLIP2 0.9168, DINOv2 0.9079, Kimi K2.6 0.8862, Qwen3.8 0.8791,
and MoonViT-V2 0.8369. The mean-readout endpoints are 0.6824 to 0.2287 and 0.9128 to
0.8397, respectively, with SigLIP2 leading both.

Five Towers expose 1,024 patch tokens at 448 square; Qwen3.8 exposes 784 because its
configured patch size is 16 rather than 14. On an otherwise idle L40S, measured over 768
images including preprocessing, transfer, Tower forward, and final-Stage materialisation,
throughput is 64.918 images/s for Qwen3.8, 45.422 for SigLIP2, 44.205 for DINOv2, 35.127
for Kimi K2.6, 29.444 for MoonViT-V2, and 23.432 for Muse Glimmer.

The largest Tower does not dominate this comparison. Muse Glimmer loses to SigLIP2 under
both readouts at one and five percent and under mean at every budget. Its small full-label
attention lead over SigLIP2, 0.9212 against 0.9168, comes at about half the throughput with
the same token count. Qwen3.8 supplies the fewest tokens and the highest throughput but
does not approach the accuracy leaders. Hypothesis 3 survives as a multi-axis tradeoff,
not as evidence for one efficient winner.
