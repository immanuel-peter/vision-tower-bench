# Adapters own preprocessing, framing stays shared

DINOv2 takes a normalized `(3, H, W)` image. MoonViT-V2 takes a pre-patchified sequence
of 14 by 14 squares plus a grid descriptor, and normalizes on 0.5 rather than the
ImageNet constants. That is a different tensor contract, not a different constant, so the
single shared preprocessing pipeline that served the first adapter cannot serve the second.

Each adapter therefore owns the step from PIL image to model input, exposed as a
`preprocess()` transform and a `collate` function. Both are plain picklable objects that
hold no reference to the model, so DataLoader workers still run the transform in parallel.
Resize and center crop stay shared in `vtb.images.square_crop`, because framing decides
what content reaches the Tower, and a Tower shown different content is not comparable to
the others. Normalization and packing are below that line and belong to the model.

The move is tested rather than asserted. `tests/test_preprocess.py` checks that the DINOv2
transform is bit-identical to the pipeline it replaced, and that the MoonViT-V2 transform
matches the processor Moonshot AI published for that Tower.

The same review fixed an inconsistency at the deepest Relative Depth point. Transformers
exposes hidden states before the final norm, while the MoonViT-V2 encoder applies its
final norm before returning. Depth point `num_layers` now means the Tower's own output
representation, after the final norm, for every adapter. Earlier points remain the output
of block `i`. This affects the deepest slice only, and the cache was empty at the time.
