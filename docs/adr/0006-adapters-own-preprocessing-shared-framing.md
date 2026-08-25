# Adapters own preprocessing, framing stays shared

DINOv2 takes a normalized `(3, H, W)` image. MoonViT-V2 takes a pre-patchified sequence
of 14 by 14 squares plus a grid descriptor, and normalizes on 0.5 rather than the
ImageNet constants. The shared preprocessing pipeline built for DINOv2 cannot produce
both input formats.

Each adapter therefore owns the step from PIL image to model input, exposed as a
`preprocess()` transform and a `collate` function. Both are plain picklable objects that
hold no reference to the model, so DataLoader workers still run the transform in parallel.
Resize and center crop stay shared in `vtb.images.square_crop`, because framing decides
what content reaches the Tower. Normalization and packing belong to the adapter.

`tests/test_preprocess.py` checks that the DINOv2 transform is bit-identical to the
pipeline it replaced. It also compares the MoonViT-V2 transform with Moonshot AI's
published processor.

Transformers exposes hidden states before the final norm, while MoonViT-V2 applies its
final norm before returning. For every adapter, depth point `num_layers` now means the
Tower output after the final norm. Earlier points remain the output of block `i`. The
change affected only the deepest slice, and the cache was empty.
