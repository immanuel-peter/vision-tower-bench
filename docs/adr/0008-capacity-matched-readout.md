# The semantic readout matches capacity across Stages

The shared readout is an attention pool plus a linear classifier, and its trainable
parameter count scales with token width. Measured on the current roster, that is
2,152,036 parameters at a `tower` width of 1024, 5,297,764 at MoonViT-V2's `merged` width
of 4096, and 8,443,492 at its `projected` width of 7168. A `projected` cell would
use a head four times larger than a `tower` cell. Head capacity could then explain part
of the measured difference between the two Stages.

Fit PCA on each cell's training split, freeze it, and train the shared head on the reduced
features. This gives every cell 1,627,748 trainable parameters, within the protocol's one
to two million limit. A random projection would discard a larger share of a wide Stage
than a narrow one and bias `projected` downward. Fitting only on the training split keeps
the test data out of the reduction.

Every headline table also runs unmatched heads on raw token widths. Accept capacity
matching only if the model rankings and Relative Depth curves agree between the matched
and unmatched runs. Report any disagreement. ADR-0003 does not allow cutting this check.
