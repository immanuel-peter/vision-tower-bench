# Perturbation study uses transformed real photos, not rendered scenes

Apply object scale, occlusion, and blur transforms to a fixed set of real photos. Change
one factor at a time and record its exact value for every image.

Rendered scenes would add a synthetic-to-real gap without helping these three factors.
Duplicate-object counting did need rendered ground truth, but ADR-0003 already cut it.

The photos are a fixed, sorted 2,000-image subset of ImageNet-100 validation, not COCO.
The semantic readouts were trained against ImageNet-100 labels, so that choice permits a
direct clean-to-perturbed accuracy comparison. COCO labels do not name the readout's 100
classes. All transforms are programmatic and need no object annotations.

Train each capacity-matched attention readout once on the clean features. Freeze its PCA
reducer and classifier, then score the identity and transformed caches. Retraining for
each condition would measure adaptation rather than robustness. The shipped JSON Lines
dataset records every image's condition and exact transform parameters.

The continuation crossed $44 before this study could run, leaving less than $21 under its
$65 ceiling. ADR-0003 therefore reduces the study to occlusion, the factor with the most
direct information-removal interpretation, at area fractions 0, 0.10, 0.20, 0.35, and
0.50. The implementation retains scale and directional motion blur for a later expansion,
but neither unmeasured factor appears in the v1 result.

## Result

All 14 Stage cells completed without alerts. Every `merged`, `projected`, and multimodal
`tower` cell has a positive clean-minus-10-percent interval. DINOv2's Tower is the only
cell whose 10-percent interval crosses zero; its loss first resolves at 20 percent. There
is therefore no roster-wide Stage that degrades first.

Clean Stage ordering survives through 35 percent occlusion for MoonViT-V2 and Muse
Glimmer, through 20 percent for Qwen3.8, and only through 10 percent for Kimi K2.6. Severe
occlusion does not preserve the clean-image ranking. Kimi K2.6 is the clearest Connector
robustness result: `projected` loses 0.0356, 0.1544, 0.3689, and 0.5911 accuracy, while its
own `tower` loses 0.0444, 0.1689, 0.4056, and 0.6244. The Projector is descriptively more
robust at all four levels and becomes the highest-accuracy Kimi Stage at 20 percent.

MoonViT-V2's Projector is more robust than its Tower at three of four levels and becomes
the highest-accuracy Stage only at 50 percent. Qwen3.8 and Muse Glimmer go the other way:
their Projectors lose more accuracy than their own Towers at every level. The study puts
paired clean-to-condition intervals on each Stage, not a paired interval on the difference
between two Stages' degradation, so cross-Stage robustness gaps remain descriptive.
