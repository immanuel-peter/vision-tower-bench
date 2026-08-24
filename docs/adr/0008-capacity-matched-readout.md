# The semantic readout matches capacity across Stages

The shared readout is an attention pool plus a linear classifier, and its trainable
parameter count scales with token width. Measured on the current roster, that is
2,152,036 parameters at a `tower` width of 1024, 5,297,764 at MoonViT-V2's `merged` width
of 4096, and 8,443,492 at its `projected` width of 7168. A `projected` cell would
therefore be read out by a head four times larger than the one reading a `tower` cell, and
part of any difference between them would be head capacity rather than feature content.
This bears directly on the headline claim, which compares exactly those two Stages.

Every cell therefore fits a PCA reduction to a common width on its own training split,
freezes it, and trains the same head on top: 1,627,748 parameters everywhere, inside the
one to two million the protocol allows. The reduction is fitted rather than drawn at
random. A random projection discards a larger share of a wide Stage than of a narrow one,
so it would depress `projected` relative to `tower` and manufacture the result the project
exists to test. Fitting on the training split only keeps the test split unseen.

Capacity matching is a shortcut, so it is validated the way pooling is in ADR-0005. Every
headline table also runs unmatched, on raw token widths, and the matched cache is accepted
only if the model ranking and the shape of the Relative Depth curves agree between the
two. A disagreement is a result to report. This validation joins the list in ADR-0003
that the cut order never touches.
