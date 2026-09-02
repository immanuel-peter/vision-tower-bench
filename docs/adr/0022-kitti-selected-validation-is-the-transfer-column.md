# KITTI selected validation is the one driving transfer column

ADR-0001 permits one driving result as transfer evidence, not a driving benchmark. The
KITTI depth-completion selected validation archive is public, downloads without
credentials, and supplies 1,000 paired RGB images and metric depth targets. It is the
Transfer Probe dataset. No KITTI task, split, model adaptation, or driving claim is added
beyond this depth column.

`scripts/prep_kitti.py` pairs the selected RGB and accumulated ground-truth depth files,
decodes KITTI's unsigned 16-bit values at 1/256 metre, and preserves zero as invalid. The
prepared target archive uses the same image-keyed npz format as DIODE. Its manifest records
the observed range and valid-pixel coverage. This run observed a maximum of 85.766 metres
and 17.0848 percent mean coverage. `vtb.geometry_run` masks zero pixels in both its loss and
metrics and takes the depth head's bin ceiling from that manifest. A 10 metre default would
repeat the DIODE range error documented by ADR-0011.

The run uses every Tower at its final `tower` Stage and the `projected` Stage where one
exists. It keeps full patch tokens, the capacity-matched arm, the existing multiscale depth
head, validation-selected learning rate, and three seeds. Paired image-bootstrap intervals
compare each Projector with its own Tower. The report asks only whether the DIODE Stage
ordering transfers to these driving scenes.

## Result

All ten matrix cells and four paired intervals completed without alerts. DINOv2 leads the
final-Tower point estimates at 0.9799 `d1`, followed by Kimi K2.6 at 0.9652, SigLIP2 at
0.9642, Qwen3.8 at 0.9501, Muse Glimmer at 0.9433, and MoonViT-V2 at 0.9418. These are
descriptive Tower rows; no cross-model interval was run.

The DIODE Stage ordering does not transfer. `projected` minus `tower` is -0.004807 for Kimi
K2.6, paired 95 percent interval [-0.007738, -0.001601], and -0.003912 for Muse Glimmer,
interval [-0.007402, -0.000731]. Both significantly favour the Tower. Qwen3.8 reads
-0.002081, interval [-0.005270, 0.001242], and MoonViT-V2 reads +0.001354, interval
[-0.001105, 0.003828]. Both are unresolved. All four Projectors improved on DIODE; none
has a resolved advantage here.

This is evidence that the DIODE Projector ordering does not generalise to the selected
KITTI driving scenes. It is not evidence about a complete driving system, other KITTI
tasks or splits, or driving performance broadly. ADR-0001's project boundary remains in
force.
